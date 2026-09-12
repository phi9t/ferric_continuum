use std::path::{Path, PathBuf};

use serde_json::Value;
use tnsr::{
    deepseek_v41::{
        attention::{
            AttentionLayerInput, DeepSeekV41Attention, DeepSeekV41Compressor, DeepSeekV41Indexer,
            SharedAttentionState,
        },
        engram::DeepSeekV41Engram,
        model::{DeepSeekV41Block, DeepSeekV41TextModel},
        moe::{DeepSeekV41Expert, DeepSeekV41Gate, DeepSeekV41MoE},
        residual::ResidualStream,
    },
    tensor::{Shape, Tensor, TensorValue},
};

fn runfile(path: &str) -> PathBuf {
    let cwd = std::env::current_dir().expect("current dir");
    let direct = cwd.join(path);
    if direct.exists() {
        return direct;
    }

    if let Ok(runfiles_dir) = std::env::var("RUNFILES_DIR") {
        let under_workspace = Path::new(&runfiles_dir).join("_main").join(path);
        if under_workspace.exists() {
            return under_workspace;
        }
        return Path::new(&runfiles_dir).join(path);
    }

    direct
}

fn fixture(name: &str) -> Value {
    let path = runfile(&format!(
        "ferric_continuum/tnsr/testdata/deepseek_v41/{name}"
    ));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("parse fixture {}: {err}", path.display()))
}

fn f32_array(value: &Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} should be array"))
        .iter()
        .map(|v| {
            v.as_f64()
                .unwrap_or_else(|| panic!("{key} item should be number")) as f32
        })
        .collect()
}

fn usize_array(value: &Value, key: &str) -> Vec<usize> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} should be array"))
        .iter()
        .map(|v| {
            v.as_u64()
                .unwrap_or_else(|| panic!("{key} item should be unsigned integer"))
                as usize
        })
        .collect()
}

fn bool_array(value: &Value, key: &str) -> Vec<bool> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} should be array"))
        .iter()
        .map(|v| {
            v.as_bool()
                .unwrap_or_else(|| panic!("{key} item should be bool"))
        })
        .collect()
}

fn usize_field(value: &Value, key: &str) -> usize {
    value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} should be unsigned integer")) as usize
}

fn f32_field(value: &Value, key: &str) -> f32 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} should be number")) as f32
}

fn string_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} should be string"))
}

fn assert_fixture_meta(fixture: &Value, upstream_function: &str) {
    assert_eq!(
        string_field(fixture, "upstream_function"),
        upstream_function
    );
    assert!(
        string_field(fixture, "source_file")
            == "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/model.py"
            || string_field(fixture, "source_file")
                == "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/engram.py",
        "unexpected source_file {:?}",
        string_field(fixture, "source_file")
    );
    assert!(
        string_field(fixture, "line_hint").contains(upstream_function),
        "line_hint should name {upstream_function}"
    );
    assert!(
        fixture["absolute_tolerance"].is_number(),
        "absolute_tolerance should be numeric"
    );
    let status = string_field(fixture, "upstream_call_status");
    assert!(
        status.starts_with("called-upstream:")
            || status.starts_with("fallback:")
            || status.starts_with("fixture-seam:"),
        "unexpected upstream_call_status {status:?}"
    );
}

/// Lightweight meta check for the prompt fixtures (Level 5). They come from
/// upstream `encoding.py`, not `model.py`, so they carry a different provenance
/// shape than the numeric op/layer fixtures.
fn assert_fixture_meta_prompt(fixture: &Value) {
    assert_eq!(
        string_field(fixture, "upstream_function"),
        "encode_messages"
    );
    assert_eq!(
        string_field(fixture, "source_file"),
        "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/encoding/encoding.py"
    );
    assert_eq!(
        string_field(fixture, "upstream_import_status"),
        "path-import-ok",
        "prompt fixtures must be rendered by upstream encoding.py"
    );
}

fn assert_close_slice(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "index {i}: got {g:.8}, want {w:.8}, diff {:.8}, tol {tol}",
            (g - w).abs()
        );
    }
}

fn tensor_from_fixture(value: &Value, data_key: &str, shape_key: &str) -> Tensor {
    let shape = usize_array(value, shape_key);
    let data = f32_array(value, data_key);
    Tensor::from_value_no_grad(TensorValue::from_vec(Shape(shape), data))
}

fn param(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_value(TensorValue::from_vec(Shape(shape.to_vec()), data), true)
}

fn tensor_param(value: &Value) -> Tensor {
    param(&usize_array(value, "shape"), f32_array(value, "data"))
}

fn attention_from_fixture(
    input: &Value,
    params: &Value,
    compressor: bool,
    indexer: bool,
) -> DeepSeekV41Attention {
    let shape = &input["attention_shape"];
    DeepSeekV41Attention {
        n_heads: usize_field(shape, "n_heads"),
        head_dim: usize_field(shape, "head_dim"),
        rope_head_dim: usize_field(shape, "rope_head_dim"),
        q_lora_rank: usize_field(shape, "q_lora_rank"),
        o_lora_rank: usize_field(shape, "o_lora_rank"),
        o_groups: usize_field(shape, "o_groups"),
        compress_ratio: usize_field(shape, "compress_ratio"),
        window_size: usize_field(shape, "window_size"),
        rms_norm_eps: f32_field(input, "rms_norm_eps"),
        wq_a: tensor_param(&params["attention"]["wq_a"]),
        q_norm: tensor_param(&params["attention"]["q_norm"]),
        wq_b: tensor_param(&params["attention"]["wq_b"]),
        wkv: tensor_param(&params["attention"]["wkv"]),
        kv_norm: tensor_param(&params["attention"]["kv_norm"]),
        wo_a: tensor_param(&params["attention"]["wo_a"]),
        wo_b: tensor_param(&params["attention"]["wo_b"]),
        attn_sink: tensor_param(&params["attention"]["attn_sink"]),
        compressor: compressor.then(|| {
            DeepSeekV41Compressor::ratio_one(
                tensor_param(&params["attention"]["compressor_wkv"]),
                tensor_param(&params["attention"]["compressor_norm"]),
                f32_field(input, "rms_norm_eps"),
            )
        }),
        indexer: indexer.then(|| DeepSeekV41Indexer {
            index_topk: usize_field(shape, "index_topk"),
        }),
    }
}

fn expert_from_fixture(input: &Value, value: &Value) -> DeepSeekV41Expert {
    DeepSeekV41Expert {
        w1: f32_array(&value["w1"], "data"),
        w2: f32_array(&value["w2"], "data"),
        w3: f32_array(&value["w3"], "data"),
        dim: usize_field(input, "dim"),
        inter_dim: usize_field(input, "inter_dim"),
        swiglu_limit: f32_field(input, "swiglu_limit"),
    }
}

fn moe_from_fixture(input: &Value, params: &Value) -> DeepSeekV41MoE {
    let experts = params["moe"]["experts"]
        .as_array()
        .expect("experts should be array")
        .iter()
        .map(|expert| expert_from_fixture(input, expert))
        .collect();
    DeepSeekV41MoE {
        gate: DeepSeekV41Gate {
            weight: f32_array(&params["moe"]["gate_weight"], "data"),
            correction_bias: f32_array(&params["moe"]["correction_bias"], "data"),
            tokens: usize_field(input, "tokens"),
            dim: usize_field(input, "dim"),
            experts: usize_field(input, "experts"),
            topk: usize_field(input, "topk"),
            gate_temp: f32_field(input, "gate_temp"),
            norm_topk_prob: input["norm_topk_prob"].as_bool().unwrap(),
            route_scale: f32_field(input, "route_scale"),
        },
        experts,
        shared_experts: expert_from_fixture(input, &params["moe"]["shared_expert"]),
    }
}

fn block_from_fixture(fixture: &Value) -> DeepSeekV41Block {
    let input = &fixture["input"];
    let params = &fixture["parameters"];
    let mode = string_field(input, "mode");
    let has_compressor = matches!(mode, "kv_source" | "index_source");
    let has_indexer = mode == "index_source";
    let has_engram = mode == "engram";
    DeepSeekV41Block {
        layer_id: usize_field(input, "layer_id"),
        dim: usize_field(input, "dim"),
        hc_mult: usize_field(input, "hc_mult"),
        hc_sinkhorn_iters: usize_field(input, "hc_sinkhorn_iters"),
        hc_eps: f32_field(input, "hc_eps"),
        attn_norm: tensor_param(&params["attn_norm"]),
        ffn_norm: tensor_param(&params["ffn_norm"]),
        attn: attention_from_fixture(input, params, has_compressor, has_indexer),
        ffn: moe_from_fixture(input, params),
        hc_attn_fn: f32_array(&params["hc_attn_fn"], "data"),
        hc_attn_base: f32_array(&params["hc_attn_base"], "data"),
        hc_attn_scale: f32_array(&params["hc_attn_scale"], "data"),
        hc_ffn_fn: f32_array(&params["hc_ffn_fn"], "data"),
        hc_ffn_base: f32_array(&params["hc_ffn_base"], "data"),
        hc_ffn_scale: f32_array(&params["hc_ffn_scale"], "data"),
        engram: has_engram.then(|| DeepSeekV41Engram {
            q_weight: f32_array(&params["engram"]["q_weight"], "data"),
            k_weight: f32_array(&params["engram"]["k_weight"], "data"),
            eps: f32_field(input, "hc_eps"),
        }),
        engram_key: has_engram.then(|| tensor_param(&params["engram"]["key"])),
        engram_value: has_engram.then(|| tensor_param(&params["engram"]["value"])),
    }
}

#[test]
fn attention_layer_checks_paths_sparse_indices_grouped_projection_and_shape() {
    let fixture = fixture("attention_layer_fixture.json");
    assert_fixture_meta(&fixture, "Attention.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let params = &fixture["parameters"];
    let shape = &input["shape"];

    let attention = DeepSeekV41Attention {
        n_heads: usize_field(shape, "n_heads"),
        head_dim: usize_field(shape, "head_dim"),
        rope_head_dim: usize_field(shape, "rope_head_dim"),
        q_lora_rank: usize_field(shape, "q_lora_rank"),
        o_lora_rank: usize_field(shape, "o_lora_rank"),
        o_groups: usize_field(shape, "o_groups"),
        compress_ratio: usize_field(shape, "compress_ratio"),
        window_size: usize_field(shape, "window_size"),
        rms_norm_eps: f32_field(input, "rms_norm_eps"),
        wq_a: param(
            &usize_array(&params["wq_a"], "shape"),
            f32_array(&params["wq_a"], "data"),
        ),
        q_norm: param(
            &usize_array(&params["q_norm"], "shape"),
            f32_array(&params["q_norm"], "data"),
        ),
        wq_b: param(
            &usize_array(&params["wq_b"], "shape"),
            f32_array(&params["wq_b"], "data"),
        ),
        wkv: param(
            &usize_array(&params["wkv"], "shape"),
            f32_array(&params["wkv"], "data"),
        ),
        kv_norm: param(
            &usize_array(&params["kv_norm"], "shape"),
            f32_array(&params["kv_norm"], "data"),
        ),
        wo_a: param(
            &usize_array(&params["wo_a"], "shape"),
            f32_array(&params["wo_a"], "data"),
        ),
        wo_b: param(
            &usize_array(&params["wo_b"], "shape"),
            f32_array(&params["wo_b"], "data"),
        ),
        attn_sink: param(
            &usize_array(&params["attn_sink"], "shape"),
            f32_array(&params["attn_sink"], "data"),
        ),
        compressor: Some(DeepSeekV41Compressor::ratio_one(
            param(
                &usize_array(&params["compressor_wkv"], "shape"),
                f32_array(&params["compressor_wkv"], "data"),
            ),
            param(
                &usize_array(&params["compressor_norm"], "shape"),
                f32_array(&params["compressor_norm"], "data"),
            ),
            f32_field(input, "rms_norm_eps"),
        )),
        indexer: Some(DeepSeekV41Indexer {
            index_topk: usize_field(shape, "index_topk"),
        }),
    };

    let x = tensor_from_fixture(input, "x", "x_shape");
    let mut state = SharedAttentionState::default();
    state.topk_indices = Some(usize_array(input, "sparse_indices"));

    let out = attention.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: usize_field(input, "start_pos"),
        shared: &mut state,
    });
    let value = out.inner.borrow().value.clone();
    assert_eq!(value.shape.0, usize_array(&fixture["expected"], "shape"));
    assert_close_slice(
        value.data.as_ref(),
        &f32_array(&fixture["expected"], "output"),
        tol,
    );

    let published = state
        .compressed_kv
        .as_ref()
        .expect("compressor should publish kv");
    assert_close_slice(
        published,
        &f32_array(&fixture["expected"], "compressed_kv"),
        tol,
    );
    assert!(state.consumed_sparse_indices);
}

#[test]
fn moe_layer_applies_selected_routed_experts_and_shared_expert() {
    let fixture = fixture("moe_layer_fixture.json");
    assert_fixture_meta(&fixture, "MoE.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let params = &fixture["parameters"];

    let gate = DeepSeekV41Gate {
        weight: f32_array(&params["gate_weight"], "data"),
        correction_bias: f32_array(&params["correction_bias"], "data"),
        tokens: usize_field(input, "tokens"),
        dim: usize_field(input, "dim"),
        experts: usize_field(input, "experts"),
        topk: usize_field(input, "topk"),
        gate_temp: f32_field(input, "gate_temp"),
        norm_topk_prob: input["norm_topk_prob"].as_bool().unwrap(),
        route_scale: f32_field(input, "route_scale"),
    };
    let experts = params["experts"]
        .as_array()
        .expect("experts should be array")
        .iter()
        .map(|expert| DeepSeekV41Expert {
            w1: f32_array(&expert["w1"], "data"),
            w2: f32_array(&expert["w2"], "data"),
            w3: f32_array(&expert["w3"], "data"),
            dim: usize_field(input, "dim"),
            inter_dim: usize_field(input, "inter_dim"),
            swiglu_limit: f32_field(input, "swiglu_limit"),
        })
        .collect();
    let shared_experts = DeepSeekV41Expert {
        w1: f32_array(&params["shared_expert"]["w1"], "data"),
        w2: f32_array(&params["shared_expert"]["w2"], "data"),
        w3: f32_array(&params["shared_expert"]["w3"], "data"),
        dim: usize_field(input, "dim"),
        inter_dim: usize_field(input, "inter_dim"),
        swiglu_limit: f32_field(input, "swiglu_limit"),
    };
    let moe = DeepSeekV41MoE {
        gate,
        experts,
        shared_experts,
    };
    let x = tensor_from_fixture(input, "x", "x_shape");
    let out = moe.forward_layer(&x);
    let value = out.inner.borrow().value.clone();
    assert_eq!(value.shape.0, usize_array(&fixture["expected"], "shape"));
    assert_eq!(
        moe.last_selected_experts(&x),
        usize_array(&fixture["expected"], "selected_experts")
    );
    assert_close_slice(
        value.data.as_ref(),
        &f32_array(&fixture["expected"], "output"),
        tol,
    );
}

#[test]
fn engram_layer_updates_text_tokens_and_leaves_masked_tokens_unchanged() {
    let fixture = fixture("engram_layer_fixture.json");
    assert_fixture_meta(&fixture, "Engram.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let params = &fixture["parameters"];

    let engram = DeepSeekV41Engram {
        q_weight: f32_array(&params["q_weight"], "data"),
        k_weight: f32_array(&params["k_weight"], "data"),
        eps: f32_field(input, "eps"),
    };

    let x = tensor_from_fixture(input, "x", "x_shape");
    let key = tensor_from_fixture(input, "key", "x_shape");
    let value = tensor_from_fixture(input, "value", "value_shape");
    let mask = bool_array(input, "token_mask");
    let out = engram.forward_layer(&x, &key, &value, Some(&mask));
    let value = out.inner.borrow().value.clone();
    assert_eq!(value.shape.0, usize_array(&fixture["expected"], "shape"));
    assert_close_slice(
        value.data.as_ref(),
        &f32_array(&fixture["expected"], "output"),
        tol,
    );
    assert_close_slice(
        &value.data.as_ref()
            [usize_field(input, "masked_token_start")..usize_field(input, "masked_token_end")],
        &f32_array(&fixture["expected"], "masked_token_original"),
        tol,
    );
}

fn run_block_fixture(name: &str) {
    let fixture = fixture(name);
    assert_fixture_meta(&fixture, "Block.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];

    let block = block_from_fixture(&fixture);
    let x = tensor_from_fixture(input, "x", "x_shape");
    let pre_mix = f32_array(input, "pre_mix");
    let mut shared = SharedAttentionState::default();
    if input.get("initial_sparse_indices").is_some() {
        shared.topk_indices = Some(usize_array(input, "initial_sparse_indices"));
    }
    let mask = input.get("token_mask").and_then(|v| {
        if v.is_null() {
            None
        } else {
            Some(bool_array(input, "token_mask"))
        }
    });
    let mut stream = ResidualStream::from_hc_tensor(&x, pre_mix);
    block.forward(
        &mut stream,
        usize_field(input, "start_pos"),
        &mut shared,
        mask.as_deref(),
    );
    let out = stream.collapse_hc();
    let next_pre_mix = stream.pre_mix().to_vec();
    let value = out.inner.borrow().value.clone();

    assert_eq!(value.shape.0, usize_array(&fixture["expected"], "shape"));
    assert_close_slice(
        value.data.as_ref(),
        &f32_array(&fixture["expected"], "output"),
        tol,
    );
    assert_close_slice(
        &next_pre_mix,
        &f32_array(&fixture["expected"], "next_pre_mix"),
        tol,
    );

    assert_eq!(
        shared.compressed_kv.is_some(),
        fixture["expected"]["published_compressed_kv"]
            .as_bool()
            .expect("published_compressed_kv should be bool")
    );
    assert_eq!(
        shared.topk_indices.is_some(),
        fixture["expected"]["has_topk_indices"]
            .as_bool()
            .expect("has_topk_indices should be bool")
    );
    assert_eq!(
        shared.consumed_sparse_indices,
        fixture["expected"]["consumed_sparse_indices"]
            .as_bool()
            .expect("consumed_sparse_indices should be bool")
    );
}

#[test]
fn block_forward_covers_sliding_window_only_mode() {
    run_block_fixture("block_sliding_window_fixture.json");
}

#[test]
fn block_forward_covers_kv_source_mode() {
    run_block_fixture("block_kv_source_fixture.json");
}

#[test]
fn block_forward_covers_index_source_mode() {
    run_block_fixture("block_index_source_fixture.json");
}

#[test]
fn block_forward_covers_reuse_mode() {
    run_block_fixture("block_reuse_fixture.json");
}

#[test]
fn block_forward_covers_engram_mode() {
    run_block_fixture("block_engram_layer_fixture.json");
}

#[test]
fn tiny_text_model_emits_bsv_logits_and_rejects_image_token_ids() {
    let fixture = fixture("tiny_model_fixture.json");
    assert_fixture_meta(&fixture, "Transformer.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let params = &fixture["parameters"];

    let model = DeepSeekV41TextModel {
        vocab_size: usize_field(input, "vocab_size"),
        hidden_size: usize_field(input, "hidden_size"),
        hc_mult: usize_field(input, "hc_mult"),
        image_token_id: usize_field(input, "image_token_id"),
        embed_tokens: tensor_param(&params["embed_tokens"]),
        layers: vec![block_from_fixture(&params["block_fixture"])],
        final_norm: tensor_param(&params["final_norm"]),
        lm_head: tensor_param(&params["lm_head"]),
    };

    let ids = usize_array(input, "ids");
    let out = model.forward_token_ids(
        &ids,
        usize_field(input, "batch"),
        usize_field(input, "seqlen"),
    );
    let value = out.inner.borrow().value.clone();
    assert_eq!(value.shape.0, usize_array(&fixture["expected"], "shape"));
    assert_close_slice(
        value.data.as_ref(),
        &f32_array(&fixture["expected"], "logits"),
        tol,
    );

    let image_ids = usize_array(input, "image_ids");
    let err = match model.try_forward_token_ids(
        &image_ids,
        usize_field(input, "batch"),
        usize_field(input, "seqlen"),
    ) {
        Ok(_) => panic!("text-only model must reject image-token inputs"),
        Err(err) => err,
    };
    assert!(err.contains("text-only"), "unexpected error: {err}");
}

/// Level-5 prompt-string parity: the prompt fixtures are rendered by upstream
/// `encoding.py`, so these assertions pin the V4.1 template surface the CLI
/// mirrors (special tokens, `</think>` suppression, numeric reasoning effort,
/// and DSML tool-call blocks) independent of any tokenizer asset.
#[test]
fn prompt_plain_fixture_renders_special_tokens() {
    let fixture = fixture("prompt_plain_fixture.json");
    assert_fixture_meta_prompt(&fixture);
    let prompt = string_field(&fixture["expected"], "prompt");
    assert!(
        prompt.starts_with('<'),
        "prompt should open with a special token"
    );
    assert!(
        prompt.contains("User"),
        "plain prompt should carry a user turn: {prompt}"
    );
    assert!(
        prompt.contains("Assistant"),
        "plain prompt should append an assistant generation header: {prompt}"
    );
    // No thinking effort prefix in chat mode.
    assert!(
        !prompt.contains("Reasoning Effort:"),
        "chat prompt must not render a reasoning-effort prefix: {prompt}"
    );
}

#[test]
fn prompt_chat_fixture_suppresses_thinking() {
    let fixture = fixture("prompt_chat_fixture.json");
    assert_fixture_meta_prompt(&fixture);
    let prompt = string_field(&fixture["expected"], "prompt");
    // Upstream chat re-rendering drops prior reasoning content but keeps the
    // `</think>` boundary token; the recorded flag documents that behavior.
    assert!(
        fixture["input"]["contains_thinking_end_token"]
            .as_bool()
            .expect("contains_thinking_end_token bool"),
        "chat fixture should record the </think> boundary"
    );
    assert!(
        !prompt.contains("respond politely"),
        "chat re-render must suppress prior reasoning content: {prompt}"
    );
}

#[test]
fn prompt_thinking_fixture_renders_numeric_reasoning_effort() {
    let fixture = fixture("prompt_thinking_fixture.json");
    assert_fixture_meta_prompt(&fixture);
    let prompt = string_field(&fixture["expected"], "prompt");
    let prefix = string_field(&fixture["input"], "reasoning_effort_prefix");
    assert!(
        fixture["input"]["contains_reasoning_effort_prefix"]
            .as_bool()
            .expect("contains_reasoning_effort_prefix bool"),
        "thinking fixture should record the reasoning-effort prefix"
    );
    assert!(
        prompt.contains(prefix),
        "thinking prompt should embed `{prefix}`: {prompt}"
    );
    assert!(
        prompt.contains("Reasoning Effort: 42"),
        "numeric reasoning effort should render: {prompt}"
    );
}

#[test]
fn prompt_dsml_fixture_renders_tool_block() {
    let fixture = fixture("prompt_dsml_fixture.json");
    assert_fixture_meta_prompt(&fixture);
    let prompt = string_field(&fixture["expected"], "prompt");
    let dsml_token = string_field(&fixture["input"], "dsml_token");
    assert!(
        fixture["input"]["contains_dsml_token"]
            .as_bool()
            .expect("contains_dsml_token bool"),
        "dsml fixture should record the DSML token"
    );
    assert!(
        prompt.contains(dsml_token),
        "DSML prompt should embed the `{dsml_token}` tool block: {prompt}"
    );
    assert!(
        prompt.contains("get_weather"),
        "DSML prompt should carry the tool name: {prompt}"
    );
}
