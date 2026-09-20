//! Load DeepSeek V4.1-Flash checkpoints into the inspectable tnsr model.
//!
//! This is deliberately *not* a rename of [`crate::qwen3_load`].  The DeepSeek
//! release ships converted-style tensor names, tensor-parallel shards, FP8/FP4
//! quantized weights with side-car `.scale` tensors, Engram lookup tables, and
//! vision/DSpark surfaces. This loader has separate public entry points for
//! text checkpoints, vision-enabled checkpoints, and one converted TP shard; it
//! maps every loaded tensor into the small CPU-readable model used by the
//! verifier ladder and makes unsupported full-runtime surfaces explicit.
//!
//! Layout adaptations (mirrors the reasoning in `qwen3_load.rs`):
//!
//! 1. **Linear layout.**  tnsr computes `x[..,Din] @ w[Din,Dout]`, so ordinary
//!    linears are stored `[Din, Dout]`.  Upstream `Linear.weight` is `[out,in]`
//!    (PyTorch), so `wq_a`, `wq_b`, `wkv`, `wo_b`, expert `w1/w2/w3`, the gate,
//!    and the compressor `wkv` are **transposed** on load.  `embed`, `head`,
//!    all `*norm` gammas, `attn_sink`, and the Engram `q_weight`/`k_weight` are
//!    copied as-is (`head` is transposed into tnsr's `[D,V]` lm_head).
//!
//! 2. **Grouped `wo_a`.**  Upstream stores `wo_a.weight` as
//!    `[n_groups * o_lora_rank, n_heads*head_dim / n_groups]` and uses
//!    `.view(n_groups, o_lora_rank, group_in)` with `einsum("bsgd,grd->bsgr")`.
//!    tnsr's `grouped_wo_a` expects a flat `[g, group_in, o_lora_rank]` layout
//!    indexed `(g*group_in + i)*o_lora_rank + r`, i.e. the inner two axes of the
//!    upstream `[g, r, d]` view are transposed per group.
//!
//! 3. **Quantization.**  BF16/F16/F32 decode exactly like `qwen3_load.rs`.  FP8
//!    (E4M3) weights carry a `.scale` tensor of E8M0 block exponents (block
//!    size 32); FP4 (E2M1, two values per byte) expert weights carry the same
//!    E8M0 block scale.  For Wave-1 CPU math we dequantize to f32 while
//!    recording the source quant kind in [`QuantKind`] so later native-kernel
//!    tickets can special-case them.
//!
//! Real 510GB weights are downloaded out of band; this loader only reads a
//! local directory.  The single-file, multimodal, and TP-shard fixtures written
//! in tests are enough to exercise every local path here.

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

use super::attention::{Csa2Mode, DeepSeekV41Attention, DeepSeekV41Compressor, DeepSeekV41Indexer};
#[cfg(test)]
use super::checkpoint_io::{e4m3_to_f32, e8m0_to_f32};
use super::checkpoint_io::{expect_shape, Checkpoint, QuantKind};
use super::config::{DeepSeekV41ReleaseIndex, DeepSeekV41TextConfig};
use super::engram::{DeepSeekV41Engram, EngramLayout, NgramHashState};
use super::model::{
    DeepSeekV41Block, DeepSeekV41DsparkHead, DeepSeekV41DsparkStage, DeepSeekV41TextModel,
    EngramRuntimeState,
};
use super::moe::{DeepSeekV41Expert, DeepSeekV41Gate, DeepSeekV41MoE};
use super::vision::{OwnedVisionBlock, OwnedVisionModel};
use crate::tensor::{Shape, Tensor, TensorValue};

// -- shape helpers ---------------------------------------------------------

fn expect_numel(name: &str, got: usize, expected: usize) -> Result<(), String> {
    if got == expected {
        Ok(())
    } else {
        Err(format!(
            "tensor `{name}` expected {expected} elements, got {got}"
        ))
    }
}

/// Transpose a row-major `[rows, cols]` buffer into `[cols, rows]`.
fn transpose_2d(name: &str, data: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    expect_numel(name, data.len(), rows * cols)?;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    Ok(out)
}

// -- parameter placement ---------------------------------------------------

fn param_from(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_value_no_grad(TensorValue::from_vec(Shape(shape.to_vec()), data))
}

/// Load a `[out,in]` linear as a tnsr `[in,out]` parameter (transposed).
fn load_linear_in_out(
    ckpt: &Checkpoint,
    name: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<(Tensor, QuantKind), String> {
    let (row_major, kind) = ckpt.linear_weight(name, out_dim, in_dim)?;
    let transposed = transpose_2d(name, &row_major, out_dim, in_dim)?; // [in, out]
    Ok((param_from(&[in_dim, out_dim], transposed), kind))
}

// -- config bridging -------------------------------------------------------

/// Reject Qwen-style tensor names so a Qwen3 checkpoint cannot be mistaken for
/// a DeepSeek one.
fn reject_qwen_names(ckpt: &Checkpoint) -> Result<(), String> {
    const QWEN_MARKERS: &[&str] = &[
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ];
    for marker in QWEN_MARKERS {
        if ckpt.has(marker) {
            return Err(format!(
                "checkpoint looks like a Qwen3 model (found `{marker}`), not DeepSeek V4.1"
            ));
        }
    }
    Ok(())
}

/// Read the config from either a nested-HF `config.json` (`text_config`) or a
/// flat converted `config.json`.
fn read_config(model_dir: &Path) -> Result<DeepSeekV41TextConfig, String> {
    let cfg_path = model_dir.join("config.json");
    let text =
        fs::read_to_string(&cfg_path).map_err(|e| format!("read {}: {e}", cfg_path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;
    if json.get("text_config").is_some() {
        DeepSeekV41TextConfig::from_hf_json(&cfg_path)
    } else {
        DeepSeekV41TextConfig::from_inference_json(&cfg_path)
    }
}

// -- public entry points ---------------------------------------------------

/// Load a DeepSeek V4.1-Flash **text-only** checkpoint directory into a
/// [`DeepSeekV41TextModel`].
///
/// `model_dir` must contain `config.json` and either a single-file
/// `model.safetensors` or a set of `model{rank}-mp{world}.safetensors` shards.
/// Vision/aligner and DSpark/MTP tensors are ignored (recorded as deferred);
/// missing required Engram tensors on Engram layers are an error.
pub fn load_text_model(model_dir: &Path) -> Result<DeepSeekV41TextModel, String> {
    let cfg = read_config(model_dir)?;
    let ckpt = open_checkpoint(model_dir)?;
    build_text_model(&cfg, &ckpt)
}

/// Load a DeepSeek V4.1-Flash **vision-enabled** checkpoint directory.
///
/// In addition to the text subset this maps the vision tower (`vision.*`),
/// the aligner (`aligner.*`), the learned image-span delimiters
/// (`image_start`/`image_end`/`image_newline`), and the per-layer MoE
/// `bias_vl`.  The config must report `vision_enabled` (a positive
/// `vision_n_layers`); a text-only config is an error.
pub fn load_multimodal_model(model_dir: &Path) -> Result<DeepSeekV41TextModel, String> {
    let cfg = read_config(model_dir)?;
    if !cfg.vision.vision_enabled() {
        return Err(
            "load_multimodal_model requires a vision-enabled config (vision_n_layers > 0)"
                .to_string(),
        );
    }
    let ckpt = open_checkpoint(model_dir)?;
    build_model(&cfg, &ckpt, true)
}

/// Load one converted tensor-parallel shard, `model{mp_rank}-mp{mp_world}.safetensors`.
///
/// This wires the single-shard fixture path for TP-converted directories.  The
/// full multi-rank merge (all-gather across `head`/experts) is a later ticket;
/// this reads whichever tensors the requested shard carries and errors if a
/// required text tensor is not present in that shard.
pub fn load_text_model_from_converted_tp(
    model_dir: &Path,
    mp_rank: usize,
    mp_world: usize,
) -> Result<DeepSeekV41TextModel, String> {
    let cfg = read_config(model_dir)?;
    let shard = model_dir.join(format!("model{mp_rank}-mp{mp_world}.safetensors"));
    if !shard.exists() {
        return Err(format!("converted TP shard not found: {}", shard.display()));
    }
    let ckpt = Checkpoint::open(&[shard])?;
    build_text_model(&cfg, &ckpt)
}

fn open_checkpoint(model_dir: &Path) -> Result<Checkpoint, String> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Checkpoint::open_single(&single);
    }

    let hf_index = model_dir.join("model.safetensors.index.json");
    if hf_index.exists() {
        let index = DeepSeekV41ReleaseIndex::from_json(&hf_index)?;
        let mut shards: Vec<PathBuf> = Vec::new();
        for shard_name in index.shard_names() {
            let shard_path = Path::new(&shard_name);
            if shard_path.is_absolute()
                || shard_path
                    .components()
                    .any(|component| matches!(component, std::path::Component::ParentDir))
            {
                return Err(format!(
                    "model.safetensors.index.json shard path must stay within model dir: {shard_name}"
                ));
            }
            let shard = model_dir.join(shard_path);
            if !shard.exists() {
                return Err(format!(
                    "indexed safetensors shard not found: {}",
                    shard.display()
                ));
            }
            shards.push(shard);
        }
        if shards.is_empty() {
            return Err(format!(
                "model.safetensors.index.json in {} did not list any shards",
                model_dir.display()
            ));
        }
        shards.sort();
        return Checkpoint::open(&shards);
    }

    // Converted TP shards: collect every model{rank}-mp{world}.safetensors.
    let mut shards: Vec<PathBuf> = Vec::new();
    let entries =
        fs::read_dir(model_dir).map_err(|e| format!("read dir {}: {e}", model_dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read dir entry: {e}"))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("model") && name.ends_with(".safetensors") && name.contains("-mp") {
            shards.push(entry.path());
        }
    }
    if shards.is_empty() {
        return Err(format!(
            "no checkpoint found in {} (expected model.safetensors, model.safetensors.index.json, or model*-mp*.safetensors)",
            model_dir.display()
        ));
    }
    shards.sort();
    Checkpoint::open(&shards)
}

fn build_text_model(
    cfg: &DeepSeekV41TextConfig,
    ckpt: &Checkpoint,
) -> Result<DeepSeekV41TextModel, String> {
    build_model(cfg, ckpt, false)
}

/// Build the model, optionally loading the vision tower + aligner + delimiters
/// and the per-layer MoE `bias_vl`.  When `with_vision` is false the vision
/// namespace is ignored (recorded as deferred), matching Wave-1 behavior.
fn build_model(
    cfg: &DeepSeekV41TextConfig,
    ckpt: &Checkpoint,
    with_vision: bool,
) -> Result<DeepSeekV41TextModel, String> {
    reject_qwen_names(ckpt)?;

    let v = cfg.vocab_size;
    let d = cfg.hidden_size;

    // embed.weight [V, D] — copy as-is.
    let embed = ckpt.float_exact("embed.weight", &[v, d])?;
    let embed_tokens = param_from(&[v, d], embed);

    // head.weight [V, D] -> tnsr lm_head [D, V].
    let (head, head_shape) = ckpt.float("head.weight")?;
    expect_shape("head.weight", &head_shape, &[v, d])?;
    let lm_head = param_from(&[d, v], transpose_2d("head.weight", &head, v, d)?);

    // norm.weight [D].
    let final_norm = param_from(&[d], ckpt.float_exact("norm.weight", &[d])?);

    let engram_layers: std::collections::BTreeSet<usize> =
        cfg.engram_layer_ids.iter().copied().collect();

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for layer_id in 0..cfg.num_hidden_layers {
        let mut block = load_block(cfg, ckpt, layer_id, &engram_layers)?;
        if with_vision {
            // bias_vl is present only when the checkpoint has vision.
            let name = format!("layers.{layer_id}.ffn.gate.bias_vl");
            let experts = cfg.n_routed_experts;
            block.ffn.gate.bias_vl = Some(ckpt.float_exact(&name, &[experts])?);
        }
        layers.push(block);
    }

    let (vision, image_start, image_end, image_newline) = if with_vision {
        let vision = load_vision(cfg, ckpt)?;
        let image_start = ckpt.float_exact("image_start", &[d])?;
        let image_end = ckpt.float_exact("image_end", &[d])?;
        let image_newline = ckpt.float_exact("image_newline", &[d])?;
        (
            Some(vision),
            Some(image_start),
            Some(image_end),
            Some(image_newline),
        )
    } else {
        (None, None, None, None)
    };

    Ok(DeepSeekV41TextModel {
        vocab_size: v,
        hidden_size: d,
        hc_mult: cfg.hc_mult,
        image_token_id: cfg.image_token_id,
        causal_encoder_layers: cfg.causal_encoder_layers(),
        decoder_layers: cfg.decoder_layers(),
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        engram_runtime: build_engram_runtime(cfg),
        vision,
        image_start,
        image_end,
        image_newline,
    })
}

fn load_block(
    cfg: &DeepSeekV41TextConfig,
    ckpt: &Checkpoint,
    layer_id: usize,
    engram_layers: &std::collections::BTreeSet<usize>,
) -> Result<DeepSeekV41Block, String> {
    let p = format!("layers.{layer_id}.");
    load_block_at(
        cfg,
        ckpt,
        layer_id,
        engram_layers,
        &p,
        BlockMoeShape::backbone(cfg),
    )
}

/// Which MoE geometry a block should be loaded with. Backbone layers use the
/// text config's routed-expert counts; DSpark `mtp.*` stages use the separate
/// `dspark_n_routed_experts` / `dspark_num_experts_per_tok`.
#[derive(Clone, Copy)]
struct BlockMoeShape {
    experts: usize,
    topk: usize,
}

impl BlockMoeShape {
    fn backbone(cfg: &DeepSeekV41TextConfig) -> Self {
        Self {
            experts: cfg.n_routed_experts,
            topk: cfg.num_experts_per_tok,
        }
    }

    fn dspark(cfg: &DeepSeekV41TextConfig) -> Self {
        Self {
            experts: cfg.dspark.dspark_n_routed_experts,
            topk: cfg.dspark.dspark_num_experts_per_tok,
        }
    }
}

/// Load one hyper-connection block from a tensor-name prefix (`layers.N.` for
/// the backbone, `mtp.N.` for a DSpark stage), with the given MoE geometry.
///
/// `layer_id` is the block's own id (used for engram-layer membership and the
/// stored `DeepSeekV41Block::layer_id`); the tensor prefix is passed separately
/// because DSpark stages live under a different namespace than their upstream
/// `layer_id = n_layers + stage_id`.
fn load_block_at(
    cfg: &DeepSeekV41TextConfig,
    ckpt: &Checkpoint,
    layer_id: usize,
    engram_layers: &std::collections::BTreeSet<usize>,
    prefix: &str,
    moe: BlockMoeShape,
) -> Result<DeepSeekV41Block, String> {
    let d = cfg.hidden_size;
    let hc = cfg.hc_mult;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let q_lora = cfg.q_lora_rank;
    let o_lora = cfg.o_lora_rank;
    let o_groups = cfg.o_groups;
    let inter = cfg.moe_intermediate_size;
    let experts = moe.experts;
    let mix_hc = (2 + hc) * hc;
    let hc_dim = hc * d;
    let eps = cfg.rms_norm_eps as f32;

    let p = |s: &str| format!("{prefix}{s}");

    // -- norms --
    let attn_norm = param_from(&[d], ckpt.float_exact(&p("attn_norm.weight"), &[d])?);
    let ffn_norm = param_from(&[d], ckpt.float_exact(&p("ffn_norm.weight"), &[d])?);

    // -- attention --
    let (wq_a, _) = load_linear_in_out(ckpt, &p("attn.wq_a.weight"), q_lora, d)?;
    let q_norm = param_from(
        &[q_lora],
        ckpt.float_exact(&p("attn.q_norm.weight"), &[q_lora])?,
    );
    let (wq_b, _) = load_linear_in_out(ckpt, &p("attn.wq_b.weight"), n_heads * head_dim, q_lora)?;
    let (wkv, _) = load_linear_in_out(ckpt, &p("attn.wkv.weight"), head_dim, d)?;
    let kv_norm = param_from(
        &[head_dim],
        ckpt.float_exact(&p("attn.kv_norm.weight"), &[head_dim])?,
    );
    let wo_a = load_wo_a(
        ckpt,
        &p("attn.wo_a.weight"),
        n_heads,
        head_dim,
        o_groups,
        o_lora,
    )?;
    let (wo_b, _) = load_linear_in_out(ckpt, &p("attn.wo_b.weight"), d, o_groups * o_lora)?;
    let attn_sink = param_from(
        &[n_heads],
        ckpt.float_exact(&p("attn.attn_sink"), &[n_heads])?,
    );

    let compress_ratio = *cfg.compress_ratios.get(layer_id).unwrap_or(&0);
    let is_kv_source = cfg.kv_source_layer_ids.contains(&layer_id);
    let is_index_source = cfg.index_source_layer_ids.contains(&layer_id);

    let compressor = if is_kv_source {
        // compressor.wkv [head_dim, D] -> tnsr [D, head_dim].
        let (cw, _) = load_linear_in_out(ckpt, &p("attn.compressor.wkv.weight"), head_dim, d)?;
        let cn = param_from(
            &[head_dim],
            ckpt.float_exact(&p("attn.compressor.norm.weight"), &[head_dim])?,
        );
        if compress_ratio > 1 {
            let (cg, _) =
                load_linear_in_out(ckpt, &p("attn.compressor.wgate.weight"), head_dim, d)?;
            Some(DeepSeekV41Compressor::ratio_n(
                cw,
                cg,
                cn,
                eps,
                compress_ratio,
            ))
        } else {
            Some(DeepSeekV41Compressor::ratio_one(cw, cn, eps))
        }
    } else {
        None
    };

    let indexer = if is_index_source {
        let (iwq_b, _) = load_linear_in_out(
            ckpt,
            &p("attn.indexer.wq_b.weight"),
            cfg.index_n_heads * cfg.index_head_dim,
            q_lora,
        )?;
        let (weights_proj, _) = load_linear_in_out(
            ckpt,
            &p("attn.indexer.weights_proj.weight"),
            cfg.index_n_heads,
            d,
        )?;
        let (wk, k_norm) = if is_kv_source {
            let (wk, _) = load_linear_in_out(
                ckpt,
                &p("attn.indexer.wk.weight"),
                cfg.index_head_dim,
                head_dim,
            )?;
            let k_norm = param_from(
                &[cfg.index_head_dim],
                ckpt.float_exact(&p("attn.indexer.k_norm.weight"), &[cfg.index_head_dim])?,
            );
            (Some(wk), Some(k_norm))
        } else {
            (None, None)
        };
        Some(DeepSeekV41Indexer {
            index_topk: cfg.index_topk,
            candidate_topk_blocks: cfg.candidate_topk_blocks,
            candidate_block_size: cfg.candidate_block_size,
            wq_b: iwq_b,
            weights_proj,
            wk,
            k_norm,
            eps,
        })
    } else {
        None
    };

    let csa2_mode = match (compress_ratio > 0, is_kv_source, is_index_source) {
        (false, _, _) => Csa2Mode::SlidingWindow,
        (true, true, true) => Csa2Mode::Full,
        (true, false, true) => Csa2Mode::Reindex,
        (true, _, false) => Csa2Mode::Reuse,
    };

    let attn = DeepSeekV41Attention {
        n_heads,
        head_dim,
        rope_head_dim: cfg.qk_rope_head_dim,
        q_lora_rank: q_lora,
        o_lora_rank: o_lora,
        o_groups,
        compress_ratio,
        window_size: cfg.sliding_window,
        rms_norm_eps: eps,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        wo_a,
        wo_b,
        attn_sink,
        layer_id,
        kv_source_layer_id: source_layer_for(layer_id, &cfg.kv_source_layer_ids),
        index_source_layer_id: source_layer_for(layer_id, &cfg.index_source_layer_ids),
        csa2_mode,
        compressor,
        indexer,
    };

    // -- MoE ffn --
    let gate_weight = {
        let (row_major, _) = ckpt.linear_weight(&p("ffn.gate.weight"), experts, d)?;
        row_major // gate weight kept row-major [experts, dim] (see moe.rs).
    };
    let correction_bias = ckpt.float_exact(&p("ffn.gate.bias"), &[experts])?;

    let make_expert = |prefix: String| -> Result<DeepSeekV41Expert, String> {
        // w1/w3 [inter, dim] -> tnsr [dim, inter]; w2 [dim, inter] -> [inter, dim].
        let (w1_rm, _) = ckpt.linear_weight(&format!("{prefix}.w1.weight"), inter, d)?;
        let w1 = transpose_2d(&format!("{prefix}.w1.weight"), &w1_rm, inter, d)?;
        let (w3_rm, _) = ckpt.linear_weight(&format!("{prefix}.w3.weight"), inter, d)?;
        let w3 = transpose_2d(&format!("{prefix}.w3.weight"), &w3_rm, inter, d)?;
        let (w2_rm, _) = ckpt.linear_weight(&format!("{prefix}.w2.weight"), d, inter)?;
        let w2 = transpose_2d(&format!("{prefix}.w2.weight"), &w2_rm, d, inter)?;
        Ok(DeepSeekV41Expert {
            w1,
            w2,
            w3,
            dim: d,
            inter_dim: inter,
            swiglu_limit: cfg.swiglu_limit as f32,
        })
    };

    let mut routed = Vec::with_capacity(experts);
    for e in 0..experts {
        routed.push(make_expert(p(&format!("ffn.experts.{e}")))?);
    }
    let shared_experts = make_expert(p("ffn.shared_experts"))?;

    let ffn = DeepSeekV41MoE {
        gate: DeepSeekV41Gate {
            weight: gate_weight,
            correction_bias,
            bias_vl: None, // vision routing bias is loaded in the vision loader (W2-06)
            tokens: 0,     // set per-forward by callers building from config
            dim: d,
            experts,
            topk: moe.topk,
            gate_temp: 1.0,
            norm_topk_prob: cfg.norm_topk_prob,
            route_scale: cfg.routed_scaling_factor as f32,
        },
        experts: routed,
        shared_experts,
    };

    // -- hyper-connection --
    let hc_attn_fn = ckpt.float_exact(&p("hc_attn_fn"), &[mix_hc, hc_dim])?;
    let hc_attn_base = ckpt.float_exact(&p("hc_attn_base"), &[mix_hc])?;
    let hc_attn_scale = ckpt.float_exact(&p("hc_attn_scale"), &[3])?;
    let hc_ffn_fn = ckpt.float_exact(&p("hc_ffn_fn"), &[mix_hc, hc_dim])?;
    let hc_ffn_base = ckpt.float_exact(&p("hc_ffn_base"), &[mix_hc])?;
    let hc_ffn_scale = ckpt.float_exact(&p("hc_ffn_scale"), &[3])?;

    // -- engram (required on engram layers) --
    let (engram, engram_key, engram_value) = if engram_layers.contains(&layer_id) {
        // q_weight / k_weight [hc_mult, dim] copied as-is.
        let q_weight = ckpt.float_exact(&p("engram.q_weight"), &[hc, d])?;
        let k_weight = ckpt.float_exact(&p("engram.k_weight"), &[hc, d])?;
        let layer_index = cfg
            .engram_layer_ids
            .iter()
            .position(|&id| id == layer_id)
            .ok_or_else(|| format!("layer {layer_id} missing from engram_layer_ids"))?;
        let num_embeddings = cfg.engram_num_embeddings[layer_index];
        let embed = ckpt.float_exact(
            &p("engram.embed.weight"),
            &[num_embeddings, cfg.engram_head_dim],
        )?;
        let n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
        let (wkv, _) = load_linear_in_out(
            ckpt,
            &p("engram.wkv.weight"),
            d * (hc + 1),
            n_hash_cols * cfg.engram_head_dim,
        )?;
        (
            Some(DeepSeekV41Engram {
                q_weight,
                k_weight,
                embed_weight: Some(param_from(&[num_embeddings, cfg.engram_head_dim], embed)),
                wkv_weight: Some(wkv),
                eps,
            }),
            None,
            None,
        )
    } else {
        (None, None, None)
    };

    Ok(DeepSeekV41Block {
        layer_id,
        dim: d,
        hc_mult: hc,
        hc_sinkhorn_iters: cfg.hc_sinkhorn_iters,
        hc_eps: cfg.hc_eps as f32,
        attn_norm,
        ffn_norm,
        attn,
        ffn,
        hc_attn_fn,
        hc_attn_base,
        hc_attn_scale,
        hc_ffn_fn,
        hc_ffn_base,
        hc_ffn_scale,
        engram,
        engram_key,
        engram_value,
    })
}

fn source_layer_for(layer_id: usize, source_layers: &[usize]) -> Option<usize> {
    source_layers
        .iter()
        .copied()
        .rev()
        .find(|&source| source <= layer_id)
}

fn build_engram_runtime(cfg: &DeepSeekV41TextConfig) -> Option<EngramRuntimeState> {
    let layout = EngramLayout::from_config(cfg)?;
    let mut hash_state = NgramHashState::new(
        (0..cfg.vocab_size)
            .map(|id| id % cfg.engram_compressed_vocab_size.max(1))
            .collect(),
        cfg.engram_pad_token_id,
        1,
        cfg.original_seq_len.max(cfg.sliding_window).max(1),
    );
    hash_state.set_multipliers(engram_hash_multipliers(
        &cfg.engram_layer_ids,
        cfg.engram_max_ngram_size,
        cfg.engram_compressed_vocab_size.max(1),
    ));
    Some(EngramRuntimeState { layout, hash_state })
}

fn engram_hash_multipliers(
    layer_ids: &[usize],
    max_ngram_size: usize,
    vocab_size: usize,
) -> Vec<i64> {
    let max_long = i64::MAX as u128;
    let bound = ((max_long / vocab_size.max(1) as u128) / 2).max(1) as u64;
    let mut out = Vec::with_capacity(layer_ids.len() * max_ngram_size);
    for &layer_id in layer_ids {
        let mut rng = NumpyPcg64::new(10007u128 * layer_id as u128);
        for _ in 0..max_ngram_size {
            out.push((rng.random_bounded_u64(bound) as i64) * 2 + 1);
        }
    }
    out
}

struct NumpyPcg64 {
    state: u128,
    inc: u128,
}

impl NumpyPcg64 {
    fn new(seed: u128) -> Self {
        // NumPy's default_rng uses PCG64 with SeedSequence. This local fallback
        // keeps loader-created Engram state deterministic until tokenizer-backed
        // runtime metadata is available from the checkpoint directory.
        Self {
            state: seed.wrapping_add(0x853c49e6748fea9b),
            inc: 0xda3e39cb94b95bdb | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005u128)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 64) ^ old) >> 64) as u64;
        let rot = (old >> 122) as u32;
        xorshifted.rotate_right(rot)
    }

    fn random_bounded_u64(&mut self, high: u64) -> u64 {
        if high <= 1 {
            return 0;
        }
        self.next_u64() % high
    }
}

/// Load the grouped `wo_a` weight.
///
/// Upstream stores `wo_a.weight` as `[n_groups*o_lora_rank, group_in]` and uses
/// `.view(n_groups, o_lora_rank, group_in)`; convert.py dequantizes it to bf16.
/// tnsr's `grouped_wo_a` wants a flat `[g, group_in, o_lora_rank]` buffer, so we
/// transpose the inner two axes per group.
fn load_wo_a(
    ckpt: &Checkpoint,
    name: &str,
    n_heads: usize,
    head_dim: usize,
    o_groups: usize,
    o_lora_rank: usize,
) -> Result<Tensor, String> {
    let group_in = (n_heads / o_groups) * head_dim;
    let rows = o_groups * o_lora_rank;
    // wo_a is dequantized to bf16 by convert.py, so it is a plain float linear.
    let (row_major, shape) = ckpt.float(name)?;
    expect_shape(name, &shape, &[rows, group_in])?;

    // row_major is [g*o_lora_rank + r, i]; produce out[(g*group_in + i)*o_lora_rank + r].
    let mut out = vec![0.0f32; o_groups * group_in * o_lora_rank];
    for g in 0..o_groups {
        for r in 0..o_lora_rank {
            for i in 0..group_in {
                let src = (g * o_lora_rank + r) * group_in + i;
                let dst = (g * group_in + i) * o_lora_rank + r;
                out[dst] = row_major[src];
            }
        }
    }
    Ok(param_from(&[o_groups, group_in, o_lora_rank], out))
}

/// Load the vision tower (`vision.*`) + aligner (`aligner.*`) into an
/// [`OwnedVisionModel`].
///
/// Vision linears keep torch `[out, in]` layout (the vision math functions
/// expect `[out, in]`), and biases are copied as-is.  Every tensor is
/// shape-checked and named on failure.  Geometry is derived from the vision
/// config:
///
/// * `vision_dim = vision.hidden_size`, `n_heads = vision.num_attention_heads`
/// * `inter = vision.intermediate_size`, `patch_flat = 3 * patch_size^2`
/// * `rope_dim = vision_dim / n_heads / 2` (per-half rotary width)
/// * aligner `in_dim = vision_dim * downsample_ratio^2` -> `llm_dim`
fn load_vision(cfg: &DeepSeekV41TextConfig, ckpt: &Checkpoint) -> Result<OwnedVisionModel, String> {
    let vision_dim = cfg.vision.hidden_size;
    let n_heads = cfg.vision.num_attention_heads;
    let inter = cfg.vision.intermediate_size;
    let layers = cfg.vision.num_hidden_layers;
    let patch_size = cfg.vision.patch_size;
    let theta = cfg.vision.rope_theta;
    let r = cfg.vision.downsample_ratio;
    let llm_dim = cfg.hidden_size;

    if n_heads == 0 || vision_dim % n_heads != 0 {
        return Err(format!(
            "vision hidden_size {vision_dim} not divisible by num_attention_heads {n_heads}"
        ));
    }
    let head_dim = vision_dim / n_heads;
    if head_dim % 2 != 0 {
        return Err(format!(
            "vision head_dim {head_dim} must be even for 2D RoPE (dim/heads)"
        ));
    }
    let rope_dim = head_dim / 2;
    let patch_flat = 3 * patch_size * patch_size;

    // Patch embed: torch Linear(patch_flat -> vision_dim).
    let proj_w = ckpt.float_exact("vision.patch_embed.proj.weight", &[vision_dim, patch_flat])?;
    let proj_b = ckpt.float_exact("vision.patch_embed.proj.bias", &[vision_dim])?;

    let mut blocks = Vec::with_capacity(layers);
    for i in 0..layers {
        let b = |s: &str| format!("vision.blocks.{i}.{s}");
        let norm1 = ckpt.float_exact(&b("norm1.weight"), &[vision_dim])?;
        let wqkv = ckpt.float_exact(&b("attn.wqkv.weight"), &[3 * vision_dim, vision_dim])?;
        let wqkv_b = ckpt.float_exact(&b("attn.wqkv.bias"), &[3 * vision_dim])?;
        let wo = ckpt.float_exact(&b("attn.wo.weight"), &[vision_dim, vision_dim])?;
        let wo_b = ckpt.float_exact(&b("attn.wo.bias"), &[vision_dim])?;
        let norm2 = ckpt.float_exact(&b("norm2.weight"), &[vision_dim])?;
        // MLP: w1 = Linear(vision_dim -> 2*inter, bias=False) chunked to gate/up;
        // w2 = Linear(inter -> vision_dim, bias=False).
        let w1 = ckpt.float_exact(&b("mlp.w1.weight"), &[2 * inter, vision_dim])?;
        let w2 = ckpt.float_exact(&b("mlp.w2.weight"), &[vision_dim, inter])?;
        blocks.push(OwnedVisionBlock {
            norm1,
            wqkv,
            wqkv_b,
            wo,
            wo_b,
            norm2,
            w1,
            w2,
        });
    }

    let final_norm = ckpt.float_exact("vision.norm.weight", &[vision_dim])?;

    // Aligner: in_dim = vision_dim * r^2 -> llm_dim -> llm_dim (both with bias).
    let in_dim = vision_dim * r * r;
    let al_w1 = ckpt.float_exact("aligner.w1.weight", &[llm_dim, in_dim])?;
    let al_w1_b = ckpt.float_exact("aligner.w1.bias", &[llm_dim])?;
    let al_w2 = ckpt.float_exact("aligner.w2.weight", &[llm_dim, llm_dim])?;
    let al_w2_b = ckpt.float_exact("aligner.w2.bias", &[llm_dim])?;

    Ok(OwnedVisionModel {
        proj_w,
        proj_b,
        blocks,
        final_norm,
        al_w1,
        al_w1_b,
        al_w2,
        al_w2_b,
        vision_dim,
        llm_dim,
        n_heads,
        inter,
        rope_dim,
        theta,
        downsample_ratio: r,
        patch_flat,
    })
}

/// Load the DSpark (MTP speculative-decoding) head from a checkpoint directory.
///
/// Returns `Ok(None)` when the config does not enable DSpark
/// (`dspark_block_size == 0`), so text-only callers pay nothing.  When enabled,
/// this reads the `mtp.{stage_id}.*` namespace for each of `n_mtp_layers`
/// stages (each a full hyper-connection block loaded with the DSpark expert
/// geometry), plus the stage-scoped heads:
///
/// * **stage 0** owns `mtp.0.main_proj.weight` `[dim, dim*n_targets]` (upstream
///   `Linear` `[out,in]`, kept row-major here as `main_proj_norm` expects) and
///   `mtp.0.main_norm.weight` `[dim]`.
/// * the **last stage** (`stage_id == n_mtp_layers - 1`) owns `norm.weight`
///   `[dim]`, `markov_head.embed.weight` / `markov_head.head.weight`
///   `[vocab, markov_rank]`, and `confidence_head.proj.weight` `[1, dim+rank]`.
///
/// The shared `embed` and `head` are tied to the backbone in upstream
/// (`convert.py` drops the `mtp.*.embed/head` tensors), so this borrows them
/// from the already-loaded [`DeepSeekV41TextModel`].
pub fn load_dspark_head(
    model_dir: &Path,
    model: &DeepSeekV41TextModel,
) -> Result<Option<DeepSeekV41DsparkHead>, String> {
    let cfg = read_config(model_dir)?;
    let ckpt = open_checkpoint(model_dir)?;
    build_dspark_head(&cfg, &ckpt, model)
}

fn build_dspark_head(
    cfg: &DeepSeekV41TextConfig,
    ckpt: &Checkpoint,
    model: &DeepSeekV41TextModel,
) -> Result<Option<DeepSeekV41DsparkHead>, String> {
    if !cfg.dspark.dspark_enabled() {
        return Ok(None);
    }
    let d = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let rank = cfg.dspark.dspark_markov_rank;
    let n_stages = cfg.dspark.n_mtp_layers;
    let n_targets = cfg.dspark.dspark_target_layer_ids.len();
    if n_stages == 0 {
        return Err("dspark enabled but n_mtp_layers == 0".to_string());
    }
    if n_targets == 0 {
        return Err("dspark enabled but dspark_target_layer_ids is empty".to_string());
    }
    let in_dim = d * n_targets;

    // DSpark stages carry no engram layers (the `mtp.*` namespace has none).
    let no_engram = std::collections::BTreeSet::new();
    let moe = BlockMoeShape::dspark(cfg);

    let mut stages = Vec::with_capacity(n_stages);
    for stage_id in 0..n_stages {
        let prefix = format!("mtp.{stage_id}.");
        // The upstream block id is n_layers + stage_id; the tnsr block only uses
        // layer_id for engram membership and diagnostics, so pass that id.
        let block = load_block_at(
            cfg,
            ckpt,
            cfg.num_hidden_layers + stage_id,
            &no_engram,
            &prefix,
            moe,
        )?;

        // Stage 0 owns main_proj/main_norm. main_proj is a torch Linear
        // `[out=dim, in=in_dim]`; `main_proj_norm` applies it row-major so we
        // keep the `[dim, in_dim]` layout as-is.
        let (main_proj, main_norm) = if stage_id == 0 {
            let proj = ckpt.float_exact(&format!("{prefix}main_proj.weight"), &[d, in_dim])?;
            let norm = ckpt.float_exact(&format!("{prefix}main_norm.weight"), &[d])?;
            (Some(proj), Some(norm))
        } else {
            (None, None)
        };

        // The last stage owns the pre-head norm and the Markov/confidence heads.
        let (head_norm, markov_embed, markov_head, confidence_proj) = if stage_id == n_stages - 1 {
            let head_norm = ckpt.float_exact(&format!("{prefix}norm.weight"), &[d])?;
            let markov_embed =
                ckpt.float_exact(&format!("{prefix}markov_head.embed.weight"), &[vocab, rank])?;
            let markov_head =
                ckpt.float_exact(&format!("{prefix}markov_head.head.weight"), &[vocab, rank])?;
            // confidence proj is a Linear(dim+rank -> 1): [1, dim+rank].
            let confidence_proj = ckpt.float_exact(
                &format!("{prefix}confidence_head.proj.weight"),
                &[1, d + rank],
            )?;
            (
                Some(head_norm),
                Some(markov_embed),
                Some(markov_head),
                Some(confidence_proj),
            )
        } else {
            (None, None, None, None)
        };

        stages.push(DeepSeekV41DsparkStage {
            block,
            main_proj,
            main_norm,
            head_norm,
            markov_embed,
            markov_head,
            confidence_proj,
        });
    }

    Ok(Some(DeepSeekV41DsparkHead {
        vocab_size: vocab,
        dim: d,
        hc_mult: cfg.hc_mult,
        block_size: cfg.dspark.dspark_block_size,
        noise_token_id: cfg.dspark.dspark_noise_token_id,
        markov_rank: rank,
        head_eps: cfg.rms_norm_eps as f32,
        embed_tokens: model.embed_tokens.clone(),
        lm_head: model.lm_head.clone(),
        stages,
        stage_swa_caches: RefCell::new(vec![None; n_stages]),
    }))
}

/// Convenience: run a text-only forward that seeds the per-layer MoE token
/// counts from the input length before delegating to the model.  Loaded gates
/// leave `tokens` at 0 (unknown until a batch arrives).
pub fn forward_with_token_seed(
    model: &mut DeepSeekV41TextModel,
    ids: &[usize],
    b: usize,
    s: usize,
) -> Result<Tensor, String> {
    let tokens = b * s;
    for layer in &mut model.layers {
        layer.ffn.gate.tokens = tokens;
    }
    model.try_forward_token_ids(ids, b, s)
}

/// Multimodal forward that seeds each MoE gate's token count from `b*s` before
/// running [`DeepSeekV41TextModel::try_forward_multimodal`].  Same seeding role
/// as [`forward_with_token_seed`], for the image-aware path.
#[allow(clippy::too_many_arguments)]
pub fn forward_multimodal_with_seed(
    model: &mut DeepSeekV41TextModel,
    ids: &[usize],
    token_types: &[i64],
    b: usize,
    s: usize,
    images: &[Vec<crate::deepseek_v41::model::ImageSpan>],
    delims: &crate::deepseek_v41::model::ImageDelimiters,
) -> Result<Tensor, String> {
    let tokens = b * s;
    for layer in &mut model.layers {
        layer.ffn.gate.tokens = tokens;
    }
    model.try_forward_multimodal(ids, token_types, b, s, images, delims)
}

#[cfg(test)]
mod tests {
    include!("load_test.rs");
}
