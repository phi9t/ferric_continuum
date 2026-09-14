//! Load DeepSeek V4.1-Flash **text-only** checkpoints into a
//! [`DeepSeekV41TextModel`].
//!
//! This is deliberately *not* a rename of [`crate::qwen3_load`].  The DeepSeek
//! release ships converted-style tensor names, tensor-parallel shards, FP8/FP4
//! quantized weights with side-car `.scale` tensors, Engram lookup tables, and
//! vision/DSpark surfaces that Wave 1 does not execute.  This loader reads the
//! text subset needed to reproduce logits on CPU and makes every unsupported
//! surface explicit.
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
//! local directory.  The single-file / TP-shard fixtures written in tests are
//! enough to exercise every path here.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use half::bf16;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use super::attention::{DeepSeekV41Attention, DeepSeekV41Compressor, DeepSeekV41Indexer};
use super::config::DeepSeekV41TextConfig;
use super::engram::DeepSeekV41Engram;
use super::model::{
    DeepSeekV41Block, DeepSeekV41DsparkHead, DeepSeekV41DsparkStage, DeepSeekV41TextModel,
};
use super::moe::{DeepSeekV41Expert, DeepSeekV41Gate, DeepSeekV41MoE};
use super::vision::{OwnedVisionBlock, OwnedVisionModel};
use crate::tensor::{Shape, Tensor, TensorValue};

/// Source quantization of a loaded tensor.  Wave 1 always dequantizes to f32
/// for execution; this records what the checkpoint actually stored so later
/// native FP8/FP4 kernel tickets can reconstruct the packed form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// Plain float (bf16/f16/f32) decoded directly to f32.
    Float,
    /// FP8 E4M3 weight with an E8M0 block-scale (block size 32).
    Fp8E4m3,
    /// FP4 E2M1 packed weight (two values/byte) with an E8M0 block-scale.
    Fp4E2m1,
}

const FP_BLOCK_SIZE: usize = 32;

/// FP4 (E2M1) code -> value table, matching upstream `convert.py::FP4_TABLE`.
const FP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// -- error / shape helpers -------------------------------------------------

fn expect_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<(), String> {
    if got == expected {
        Ok(())
    } else {
        Err(format!(
            "tensor `{name}` expected {expected:?}, got {got:?}"
        ))
    }
}

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

// -- dtype decode ----------------------------------------------------------

/// Decode a float safetensors view (bf16/f16/f32) into f32.
fn view_float_to_f32(name: &str, view: &TensorView) -> Result<Vec<f32>, String> {
    let bytes = view.data();
    match view.dtype() {
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(format!("tensor `{name}`: bf16 byte length not even"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(format!("tensor `{name}`: f16 byte length not even"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(format!(
                    "tensor `{name}`: f32 byte length not multiple of 4"
                ));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        other => Err(format!(
            "tensor `{name}`: unsupported float dtype {other:?}"
        )),
    }
}

/// Decode an FP8 E4M3 byte to f32 (1 sign, 4 exponent bias 7, 3 mantissa;
/// `0x7F`/`0xFF` are NaN, matching `float8_e4m3fn`).
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        // subnormal: value = mant/8 * 2^(1-bias)
        sign * (mant as f32 / 8.0) * 2f32.powi(1 - 7)
    } else {
        // normal: value = (1 + mant/8) * 2^(exp-bias)
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Decode an E8M0 scale byte to a positive f32 (value = 2^(byte-127); `0xFF`
/// is NaN, matching `float8_e8m0fnu`).
fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return f32::NAN;
    }
    2f32.powi(byte as i32 - 127)
}

/// Raw bytes for a scale side-car tensor, kept in the source dtype.
fn scale_bytes(name: &str, view: &TensorView) -> Result<Vec<u8>, String> {
    // Scales are stored as E8M0 (`F8_E8M0`) or occasionally F32 in fixtures.
    match view.dtype() {
        // safetensors exposes E8M0 as an opaque 1-byte type; some builds label
        // it F8_E4M3 / U8.  We only need the raw byte, decoded as E8M0.
        Dtype::U8 => Ok(view.data().to_vec()),
        Dtype::F8_E4M3 => Ok(view.data().to_vec()),
        Dtype::F8_E5M2 => Ok(view.data().to_vec()),
        other => Err(format!(
            "tensor `{name}`: unsupported scale dtype {other:?} (expected 1-byte E8M0)"
        )),
    }
}

/// Dequantize an FP8 E4M3 `[out, in]` weight with an E8M0 `[ceil(out/32),
/// ceil(in/32)]` block-scale into a row-major f32 `[out, in]` buffer.
fn dequant_fp8(
    name: &str,
    weight: &TensorView,
    scale: &TensorView,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, String> {
    expect_shape(name, weight.shape(), &[out_dim, in_dim])?;
    let raw = weight.data();
    expect_numel(name, raw.len(), out_dim * in_dim)?;

    let sblk_out = out_dim.div_ceil(FP_BLOCK_SIZE);
    let sblk_in = in_dim.div_ceil(FP_BLOCK_SIZE);
    let scale_name = format!("{name}.scale");
    expect_shape(&scale_name, scale.shape(), &[sblk_out, sblk_in])?;
    let sbytes = scale_bytes(&scale_name, scale)?;
    expect_numel(&scale_name, sbytes.len(), sblk_out * sblk_in)?;

    let mut out = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            let w = e4m3_to_f32(raw[o * in_dim + i]);
            let s = e8m0_to_f32(sbytes[(o / FP_BLOCK_SIZE) * sblk_in + i / FP_BLOCK_SIZE]);
            out[o * in_dim + i] = w * s;
        }
    }
    Ok(out)
}

/// Dequantize an FP4 E2M1 packed `[out, in/2]` weight (two nibbles per byte,
/// low nibble first) with an E8M0 `[ceil(out/32), ceil(in/32)]` block-scale
/// into a row-major f32 `[out, in]` buffer.
fn dequant_fp4(
    name: &str,
    weight: &TensorView,
    scale: &TensorView,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, String> {
    if in_dim % 2 != 0 {
        return Err(format!(
            "tensor `{name}`: FP4 in_dim {in_dim} must be even (two values per byte)"
        ));
    }
    let packed_in = in_dim / 2;
    expect_shape(name, weight.shape(), &[out_dim, packed_in])?;
    let raw = weight.data();
    expect_numel(name, raw.len(), out_dim * packed_in)?;

    let sblk_out = out_dim.div_ceil(FP_BLOCK_SIZE);
    let sblk_in = in_dim.div_ceil(FP_BLOCK_SIZE);
    let scale_name = format!("{name}.scale");
    expect_shape(&scale_name, scale.shape(), &[sblk_out, sblk_in])?;
    let sbytes = scale_bytes(&scale_name, scale)?;
    expect_numel(&scale_name, sbytes.len(), sblk_out * sblk_in)?;

    let mut out = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for p in 0..packed_in {
            let byte = raw[o * packed_in + p];
            let low = FP4_TABLE[(byte & 0x0F) as usize];
            let high = FP4_TABLE[((byte >> 4) & 0x0F) as usize];
            // logical columns: 2p (low nibble), 2p+1 (high nibble).
            let i0 = 2 * p;
            let i1 = 2 * p + 1;
            let s0 = e8m0_to_f32(sbytes[(o / FP_BLOCK_SIZE) * sblk_in + i0 / FP_BLOCK_SIZE]);
            let s1 = e8m0_to_f32(sbytes[(o / FP_BLOCK_SIZE) * sblk_in + i1 / FP_BLOCK_SIZE]);
            out[o * in_dim + i0] = low * s0;
            out[o * in_dim + i1] = high * s1;
        }
    }
    Ok(out)
}

// -- checkpoint index ------------------------------------------------------

/// A flat view over one or more safetensors shards addressed by tensor name.
struct Checkpoint {
    shards: Vec<SafeTensors<'static>>,
    // name -> (shard idx)
    index: BTreeMap<String, usize>,
}

impl Checkpoint {
    /// Open a single-file `model.safetensors` checkpoint.
    fn open_single(path: &Path) -> Result<Checkpoint, String> {
        Checkpoint::open_files(&[path.to_path_buf()])
    }

    /// Open one or more shard files, building a name -> shard index.
    fn open_files(paths: &[PathBuf]) -> Result<Checkpoint, String> {
        let mut shards = Vec::new();
        let mut index = BTreeMap::new();
        for (shard_idx, path) in paths.iter().enumerate() {
            let raw = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
            // SafeTensors borrows from the buffer; leak the buffer into a
            // 'static slice so the parsed view stays valid for the life of the
            // Checkpoint (loaders are short-lived one-shot processes).
            let leaked: &'static [u8] = Box::leak(raw.into_boxed_slice());
            let st = SafeTensors::deserialize(leaked)
                .map_err(|e| format!("parse {}: {e}", path.display()))?;
            for name in st.names() {
                index.entry(name.to_string()).or_insert(shard_idx);
            }
            shards.push(st);
        }
        Ok(Checkpoint { shards, index })
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn view(&self, name: &str) -> Result<TensorView<'_>, String> {
        let shard = *self
            .index
            .get(name)
            .ok_or_else(|| format!("missing required tensor `{name}`"))?;
        self.shards[shard]
            .tensor(name)
            .map_err(|e| format!("tensor `{name}`: {e}"))
    }

    /// Decode a float tensor to f32 with its shape.
    fn float(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let view = self.view(name)?;
        let shape = view.shape().to_vec();
        Ok((view_float_to_f32(name, &view)?, shape))
    }

    fn float_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, String> {
        let (data, shape) = self.float(name)?;
        expect_shape(name, &shape, expected)?;
        Ok(data)
    }

    /// Decode a possibly-quantized `[out, in]` linear weight to f32 `[out, in]`
    /// row-major, returning the source quant kind.  Detects FP8/FP4 by the
    /// presence and dtype of a `.scale` side-car.
    fn linear_weight(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(Vec<f32>, QuantKind), String> {
        let view = self.view(name)?;
        let scale_name = format!("{name}.scale");
        match view.dtype() {
            Dtype::BF16 | Dtype::F16 | Dtype::F32 => {
                expect_shape(name, view.shape(), &[out_dim, in_dim])?;
                let data = view_float_to_f32(name, &view)?;
                Ok((data, QuantKind::Float))
            }
            Dtype::F8_E4M3 => {
                let scale = self.view(&scale_name)?;
                let data = dequant_fp8(name, &view, &scale, out_dim, in_dim)?;
                Ok((data, QuantKind::Fp8E4m3))
            }
            Dtype::I8 | Dtype::U8 => {
                // FP4 E2M1 packed: two logical columns per stored byte.
                let scale = self.view(&scale_name)?;
                let data = dequant_fp4(name, &view, &scale, out_dim, in_dim)?;
                Ok((data, QuantKind::Fp4E2m1))
            }
            other => Err(format!(
                "tensor `{name}`: unsupported linear weight dtype {other:?}"
            )),
        }
    }
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
    let ckpt = Checkpoint::open_files(&[shard])?;
    build_text_model(&cfg, &ckpt)
}

fn open_checkpoint(model_dir: &Path) -> Result<Checkpoint, String> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Checkpoint::open_single(&single);
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
            "no checkpoint found in {} (expected model.safetensors or model*-mp*.safetensors)",
            model_dir.display()
        ));
    }
    shards.sort();
    Checkpoint::open_files(&shards)
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
        embed_tokens,
        layers,
        final_norm,
        lm_head,
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
        // compressor.wkv [head_dim, D] (ratio-one) -> tnsr [D, head_dim].
        let (cw, _) = load_linear_in_out(ckpt, &p("attn.compressor.wkv.weight"), head_dim, d)?;
        let cn = param_from(
            &[head_dim],
            ckpt.float_exact(&p("attn.compressor.norm.weight"), &[head_dim])?,
        );
        Some(DeepSeekV41Compressor::ratio_one(cw, cn, eps))
    } else {
        None
    };

    let indexer = if is_index_source {
        Some(DeepSeekV41Indexer {
            index_topk: cfg.index_topk,
        })
    } else {
        None
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
        // The engram embed/wkv tables are required to be present; full runtime
        // hashing is a later ticket, so the block's key/value are seeded to
        // zeros here. Their absence is still an error.
        if !ckpt.has(&p("engram.wkv.weight")) {
            return Err(format!(
                "missing required Engram tensor `{}`",
                p("engram.wkv.weight")
            ));
        }
        if !ckpt.has(&p("engram.embed.weight")) {
            return Err(format!(
                "missing required Engram tensor `{}`",
                p("engram.embed.weight")
            ));
        }
        (
            Some(DeepSeekV41Engram {
                q_weight,
                k_weight,
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
