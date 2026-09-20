use std::cell::RefCell;
use std::path::{Path, PathBuf};

use serde_json::Value;

use tnsr::{
    deepseek_v41::{
        attention::{
            AttentionLayerInput, Csa2Mode, DeepSeekV41Attention, DeepSeekV41Compressor,
            DeepSeekV41Indexer, SharedAttentionState,
        },
        engram::DeepSeekV41Engram,
        model::{
            DeepSeekV41Block, DeepSeekV41DsparkHead, DeepSeekV41DsparkStage, DeepSeekV41TextModel,
        },
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

fn i64_array(value: &Value, key: &str) -> Vec<i64> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} should be array"))
        .iter()
        .map(|v| {
            v.as_i64()
                .unwrap_or_else(|| panic!("{key} item should be integer"))
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
        layer_id: usize_field(input, "layer_id"),
        kv_source_layer_id: None,
        index_source_layer_id: None,
        csa2_mode: Csa2Mode::SlidingWindow,
        compressor: compressor.then(|| {
            DeepSeekV41Compressor::ratio_one(
                tensor_param(&params["attention"]["compressor_wkv"]),
                tensor_param(&params["attention"]["compressor_norm"]),
                f32_field(input, "rms_norm_eps"),
            )
        }),
        indexer: indexer.then(|| DeepSeekV41Indexer {
            index_topk: usize_field(shape, "index_topk"),
            candidate_topk_blocks: 0,
            candidate_block_size: 0,
            wq_b: param(
                &[
                    usize_field(shape, "q_lora_rank"),
                    usize_field(shape, "n_heads") * usize_field(shape, "head_dim"),
                ],
                vec![
                    1.0;
                    usize_field(shape, "q_lora_rank")
                        * usize_field(shape, "n_heads")
                        * usize_field(shape, "head_dim")
                ],
            ),
            weights_proj: param(
                &[usize_field(input, "dim"), usize_field(shape, "n_heads")],
                vec![1.0; usize_field(input, "dim") * usize_field(shape, "n_heads")],
            ),
            wk: Some(param(
                &[
                    usize_field(shape, "head_dim"),
                    usize_field(shape, "head_dim"),
                ],
                vec![1.0, 0.0, 0.0, 1.0],
            )),
            k_norm: Some(param(
                &[usize_field(shape, "head_dim")],
                vec![1.0; usize_field(shape, "head_dim")],
            )),
            eps: f32_field(input, "rms_norm_eps"),
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
            bias_vl: None,
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
            embed_weight: None,
            wkv_weight: None,
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
        layer_id: 0,
        kv_source_layer_id: Some(0),
        index_source_layer_id: Some(0),
        csa2_mode: Csa2Mode::Full,
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
            candidate_topk_blocks: 0,
            candidate_block_size: 0,
            wq_b: param(&[3, 4], vec![1.0; 12]),
            weights_proj: param(&[4, 2], vec![1.0; 8]),
            wk: Some(param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0])),
            k_norm: Some(param(&[2], vec![1.0, 1.0])),
            eps: f32_field(input, "rms_norm_eps"),
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

fn attention_with_identity_output() -> DeepSeekV41Attention {
    DeepSeekV41Attention {
        n_heads: 1,
        head_dim: 2,
        rope_head_dim: 0,
        q_lora_rank: 2,
        o_lora_rank: 2,
        o_groups: 1,
        compress_ratio: 1,
        window_size: 4,
        rms_norm_eps: 1e-6,
        wq_a: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        q_norm: param(&[2], vec![1.0, 1.0]),
        wq_b: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        wkv: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        kv_norm: param(&[2], vec![1.0, 1.0]),
        wo_a: param(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        wo_b: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        attn_sink: param(&[1], vec![8.0]),
        layer_id: 0,
        kv_source_layer_id: None,
        index_source_layer_id: None,
        csa2_mode: Csa2Mode::SlidingWindow,
        compressor: None,
        indexer: None,
    }
}

#[test]
fn decoder_attention_uses_encoder_global_kv_handoff() {
    let attention = attention_with_identity_output();
    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 2]),
        vec![1.0, 1.0, 2.0, 2.0],
    ));
    let mut state = SharedAttentionState::default();
    state.decoder_encoder_hidden = Some(Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 2]),
        vec![10.0, 20.0, 30.0, 40.0],
    )));
    state.topk_indices = Some(vec![1, 0]);

    let out = attention.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: 0,
        shared: &mut state,
    });

    let value = out.inner.borrow().value.clone();
    assert_close_slice(
        value.data.as_ref(),
        &[0.003332341, 0.0045239152, 0.0033323406, 0.004523915],
        1e-5,
    );
    assert!(state.consumed_decoder_encoder_hidden);
    let cache = state
        .swa_cache
        .as_ref()
        .expect("decoder local KV should still seed SWA");
    assert_eq!(cache.batch, 1);
    assert_eq!(cache.window_size, 4);
    assert_eq!(cache.head_dim, 2);
    assert_close_slice(
        &cache.data,
        &[
            0.9999995, 0.9999995, 0.9999999, 0.9999999, 0.0, 0.0, 0.0, 0.0,
        ],
        1e-5,
    );
}

#[test]
fn attention_prefill_uses_window_size_for_local_swa_indices() {
    let mut attention = attention_with_identity_output();
    attention.window_size = 1;
    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 3, 2]),
        vec![1.0, 0.0, 0.0, 2.0, -3.0, 0.0],
    ));
    let mut state = SharedAttentionState::default();

    let out = attention.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: 0,
        shared: &mut state,
    });

    let value = out.inner.borrow().value.clone();
    assert_close_slice(
        value.data.as_ref(),
        &[1.4140968, 0.0, 0.0, 1.4140968, -1.4140968, 0.0],
        1e-5,
    );
}

#[test]
fn attention_decode_wraps_swa_cache_without_reusing_evicted_positions() {
    fn norm2(pair: [f32; 2]) -> [f32; 2] {
        let inv = ((pair[0] * pair[0] + pair[1] * pair[1]) / 2.0 + 1e-6)
            .sqrt()
            .recip();
        [pair[0] * inv, pair[1] * inv]
    }

    let attention = attention_with_identity_output();
    let mut state = SharedAttentionState::default();

    let prefill = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 2]),
        vec![1.0, 0.0, 0.0, 2.0],
    ));
    let _ = attention.forward_layer(AttentionLayerInput {
        x: &prefill,
        start_pos: 0,
        shared: &mut state,
    });

    for (step, pair) in [
        (2, [3.0, 0.0]),
        (3, [0.0, 4.0]),
        (4, [5.0, 0.0]),
        (5, [0.0, 6.0]),
    ] {
        let x =
            Tensor::from_value_no_grad(TensorValue::from_vec(Shape(vec![1, 1, 2]), pair.to_vec()));
        let _ = attention.forward_layer(AttentionLayerInput {
            x: &x,
            start_pos: step,
            shared: &mut state,
        });
    }

    let cache = state
        .swa_cache
        .as_ref()
        .expect("decode should keep SWA cache");
    assert_eq!(cache.batch, 1);
    assert_eq!(cache.window_size, 4);
    assert_eq!(cache.head_dim, 2);
    let mut expected = Vec::new();
    for pair in [[5.0, 0.0], [0.0, 6.0], [3.0, 0.0], [0.0, 4.0]] {
        expected.extend_from_slice(&norm2(pair));
    }
    assert_close_slice(&cache.data, &expected, 1e-5);
}

#[test]
fn csa2_not_v4_csa_hca_full_reindex_and_reuse_share_kv_index_k_and_topk() {
    fn indexer(wk: bool, weights: [f32; 2], candidates: bool) -> DeepSeekV41Indexer {
        DeepSeekV41Indexer {
            index_topk: 1,
            candidate_topk_blocks: if candidates { 1 } else { 0 },
            candidate_block_size: if candidates { 1 } else { 0 },
            wq_b: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            weights_proj: param(&[2, 1], weights.to_vec()),
            wk: wk.then(|| param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0])),
            k_norm: wk.then(|| param(&[2], vec![1.0, 1.0])),
            eps: 0.0,
        }
    }

    let base = attention_with_identity_output();
    let mut full = attention_with_identity_output();
    full.csa2_mode = Csa2Mode::Full;
    full.layer_id = 2;
    full.kv_source_layer_id = Some(2);
    full.index_source_layer_id = Some(2);
    full.compress_ratio = 1;
    full.window_size = 1;
    full.attn_sink = param(&[1], vec![-8.0]);
    full.compressor = Some(DeepSeekV41Compressor::ratio_one(
        param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        param(&[2], vec![1.0, 1.0]),
        0.0,
    ));
    full.indexer = Some(indexer(true, [1.0, 0.0], true));

    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 2]),
        vec![2.0, 0.0, 0.0, 4.0],
    ));
    let mut shared = SharedAttentionState::default();
    let _ = full.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: 0,
        shared: &mut shared,
    });
    assert_eq!(shared.csa2_full_updates, 1);
    assert_eq!(shared.csa2_reindex_updates, 0);
    assert_eq!(shared.csa2_reuse_reads, 0);
    assert!(shared.compressed_kv.is_some());
    assert!(shared.index_k.is_some());
    assert_eq!(shared.topk_indices.as_deref(), Some(&[0, 2, 1, 3][..]));
    assert_eq!(
        shared.candidate_mask.as_deref(),
        Some(&[true, false, false, true][..])
    );

    let compressed_after_full = shared.compressed_kv.clone();
    let mut reindex = base;
    reindex.csa2_mode = Csa2Mode::Reindex;
    reindex.layer_id = 3;
    reindex.kv_source_layer_id = Some(2);
    reindex.index_source_layer_id = Some(3);
    reindex.compress_ratio = 1;
    reindex.window_size = 1;
    reindex.indexer = Some(indexer(false, [1.0, 0.0], false));
    let _ = reindex.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: 0,
        shared: &mut shared,
    });
    assert_eq!(shared.csa2_full_updates, 1);
    assert_eq!(shared.csa2_reindex_updates, 1);
    assert_eq!(shared.csa2_reuse_reads, 0);
    assert_eq!(shared.compressed_kv, compressed_after_full);
    assert_eq!(shared.topk_indices.as_deref(), Some(&[0, 2, 1, 2][..]));

    let topk_after_reindex = shared.topk_indices.clone();
    let mut reuse = attention_with_identity_output();
    reuse.csa2_mode = Csa2Mode::Reuse;
    reuse.layer_id = 4;
    reuse.kv_source_layer_id = Some(2);
    reuse.index_source_layer_id = Some(3);
    reuse.compress_ratio = 1;
    let _ = reuse.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: 0,
        shared: &mut shared,
    });
    assert_eq!(shared.csa2_full_updates, 1);
    assert_eq!(shared.csa2_reindex_updates, 1);
    assert_eq!(shared.csa2_reuse_reads, 1);
    assert_eq!(shared.topk_indices, topk_after_reindex);
}

#[test]
fn csa2_full_source_layer_uses_ratio_n_prefill_compression() {
    let mut full = attention_with_identity_output();
    full.csa2_mode = Csa2Mode::Full;
    full.layer_id = 2;
    full.kv_source_layer_id = Some(2);
    full.index_source_layer_id = Some(2);
    full.compress_ratio = 2;
    full.window_size = 2;
    full.attn_sink = param(&[1], vec![-8.0]);
    full.compressor = Some(DeepSeekV41Compressor::ratio_n(
        param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        param(&[2, 2], vec![0.0, 0.0, 0.0, 0.0]),
        param(&[2], vec![1.0, 1.0]),
        0.0,
        2,
    ));
    full.indexer = Some(DeepSeekV41Indexer {
        index_topk: 1,
        candidate_topk_blocks: 0,
        candidate_block_size: 0,
        wq_b: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        weights_proj: param(&[2, 1], vec![1.0, 0.0]),
        wk: Some(param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0])),
        k_norm: Some(param(&[2], vec![1.0, 1.0])),
        eps: 0.0,
    });

    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 2]),
        vec![2.0, 0.0, 0.0, 4.0],
    ));
    let mut shared = SharedAttentionState::default();
    let _ = full.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: 0,
        shared: &mut shared,
    });

    let compressed = shared
        .compressed_kv
        .as_ref()
        .expect("CSA2 Full should publish compressed KV");
    assert_eq!(
        compressed.len(),
        2,
        "ratio-2 prefill compression should publish one KV row for two tokens"
    );
    assert_close_slice(compressed, &[0.6324555, 1.2649111], 1e-5);
    assert_eq!(
        shared.topk_indices.as_deref(),
        Some(&[0, usize::MAX, usize::MAX, 0, 1, 2][..]),
        "local window entries are followed by the visible compressed block"
    );
}

#[test]
fn compressor_ratio_n_prefill_drops_incomplete_group_per_batch() {
    let compressor = DeepSeekV41Compressor::ratio_n(
        param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        param(&[2, 2], vec![0.0, 0.0, 0.0, 0.0]),
        param(&[2], vec![1.0, 1.0]),
        0.0,
        2,
    );
    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![2, 3, 2]),
        vec![
            1.0, 2.0, 3.0, 4.0, 99.0, 99.0, //
            5.0, 6.0, 7.0, 8.0, 88.0, 88.0,
        ],
    ));

    let compressed = compressor
        .forward(&x, 0)
        .expect("two-token groups should compress during prefill");

    assert_eq!(compressed.shape().0, vec![2, 1, 2]);
    assert_close_slice(
        &compressed.inner.borrow().value.data,
        &[0.78446454, 1.1766968, 0.9203580, 1.0737510],
        1e-5,
    );
}

#[test]
fn csa2_reindex_and_reuse_hold_source_kv_across_decode_boundary() {
    fn indexer(wk: bool, weights: [f32; 2], candidates: bool) -> DeepSeekV41Indexer {
        DeepSeekV41Indexer {
            index_topk: 1,
            candidate_topk_blocks: if candidates { 1 } else { 0 },
            candidate_block_size: if candidates { 1 } else { 0 },
            wq_b: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            weights_proj: param(&[2, 1], weights.to_vec()),
            wk: wk.then(|| param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0])),
            k_norm: wk.then(|| param(&[2], vec![1.0, 1.0])),
            eps: 0.0,
        }
    }

    let mut full = attention_with_identity_output();
    full.csa2_mode = Csa2Mode::Full;
    full.layer_id = 2;
    full.kv_source_layer_id = Some(2);
    full.index_source_layer_id = Some(2);
    full.compress_ratio = 1;
    full.window_size = 2;
    full.compressor = Some(DeepSeekV41Compressor::ratio_one(
        param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        param(&[2], vec![1.0, 1.0]),
        0.0,
    ));
    full.indexer = Some(indexer(true, [1.0, 0.0], true));

    let prefill = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 2]),
        vec![2.0, 0.0, 0.0, 4.0],
    ));
    let mut shared = SharedAttentionState::default();
    let _ = full.forward_layer(AttentionLayerInput {
        x: &prefill,
        start_pos: 0,
        shared: &mut shared,
    });
    let source_full = shared
        .csa2_sources
        .get(&2)
        .expect("Full should publish source layer 2")
        .clone();
    let source_compressed = source_full
        .compressed_kv
        .as_ref()
        .expect("Full should publish compressed KV")
        .clone();
    let source_index_k = source_full
        .index_k
        .as_ref()
        .expect("Full should publish index K")
        .clone();

    let mut reindex = attention_with_identity_output();
    reindex.csa2_mode = Csa2Mode::Reindex;
    reindex.layer_id = 3;
    reindex.kv_source_layer_id = Some(2);
    reindex.index_source_layer_id = Some(3);
    reindex.compress_ratio = 1;
    reindex.window_size = 2;
    reindex.indexer = Some(indexer(false, [0.0, 1.0], false));
    let decode =
        Tensor::from_value_no_grad(TensorValue::from_vec(Shape(vec![1, 1, 2]), vec![5.0, 0.0]));
    let _ = reindex.forward_layer(AttentionLayerInput {
        x: &decode,
        start_pos: 2,
        shared: &mut shared,
    });
    assert_eq!(shared.csa2_full_updates, 1);
    assert_eq!(shared.csa2_reindex_updates, 1);
    assert_eq!(shared.csa2_reuse_reads, 0);
    assert_eq!(shared.compressed_kv.as_ref(), Some(&source_compressed));
    assert_eq!(shared.index_k.as_ref(), Some(&source_index_k));
    assert_eq!(
        shared.topk_indices.as_ref().map(Vec::len),
        Some(3),
        "decode row should combine two SWA slots and one recomputed compressed top-k"
    );

    let source_after_reindex = shared
        .csa2_sources
        .get(&3)
        .expect("Reindex should publish index source layer 3");
    assert_eq!(
        source_after_reindex.compressed_kv.as_ref(),
        Some(&source_compressed),
        "Reindex must not replace compressed KV from the KV source layer"
    );
    assert_eq!(source_after_reindex.index_k.as_ref(), Some(&source_index_k));
    assert_eq!(source_after_reindex.topk_indices, shared.topk_indices);
    let topk_after_reindex = shared.topk_indices.clone();
    let candidate_mask_after_reindex = shared.candidate_mask.clone();

    let mut reuse = attention_with_identity_output();
    reuse.csa2_mode = Csa2Mode::Reuse;
    reuse.layer_id = 4;
    reuse.kv_source_layer_id = Some(2);
    reuse.index_source_layer_id = Some(3);
    reuse.compress_ratio = 1;
    reuse.window_size = 2;
    let _ = reuse.forward_layer(AttentionLayerInput {
        x: &decode,
        start_pos: 3,
        shared: &mut shared,
    });
    assert_eq!(shared.csa2_full_updates, 1);
    assert_eq!(shared.csa2_reindex_updates, 1);
    assert_eq!(shared.csa2_reuse_reads, 1);
    assert_eq!(shared.compressed_kv.as_ref(), Some(&source_compressed));
    assert_eq!(shared.index_k.as_ref(), Some(&source_index_k));
    assert_eq!(shared.topk_indices, topk_after_reindex);
    assert_eq!(shared.candidate_mask, candidate_mask_after_reindex);
}

#[test]
#[should_panic(expected = "CSA2 Reuse mode requires source layer 99")]
fn csa2_reuse_requires_configured_index_source_layer() {
    let mut shared = SharedAttentionState::default();
    shared.topk_indices = Some(vec![0, 0]);

    let mut reuse = attention_with_identity_output();
    reuse.csa2_mode = Csa2Mode::Reuse;
    reuse.layer_id = 4;
    reuse.index_source_layer_id = Some(99);
    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 2]),
        vec![2.0, 0.0, 0.0, 4.0],
    ));

    let _ = reuse.forward_layer(AttentionLayerInput {
        x: &x,
        start_pos: 0,
        shared: &mut shared,
    });
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
        bias_vl: None,
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
        embed_weight: None,
        wkv_weight: None,
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

#[test]
#[should_panic(expected = "Engram embed_weight is required for hash lookup")]
fn engram_hash_forward_requires_loaded_table_weights() {
    let engram = DeepSeekV41Engram {
        q_weight: vec![1.0, 1.0],
        k_weight: vec![1.0, 1.0],
        embed_weight: None,
        wkv_weight: None,
        eps: 1e-6,
    };
    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 1, 1, 2]),
        vec![1.0, 2.0],
    ));
    let _ = engram.forward_hashes(&x, &[0], None);
}

#[test]
fn engram_hash_forward_uses_table_lookup_and_wkv_projection() {
    let engram = DeepSeekV41Engram {
        q_weight: vec![1.0, 1.0],
        k_weight: vec![1.0, 1.0],
        embed_weight: Some(param(&[3, 1], vec![10.0, 20.0, 30.0])),
        wkv_weight: Some(param(&[1, 4], vec![0.1, 0.2, 0.3, 0.4])),
        eps: 1e-6,
    };
    let x = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![1, 2, 1, 2]),
        vec![1.0, 1.0, 2.0, 2.0],
    ));

    let via_hash = engram.forward_hashes(&x, &[0, 2], None);
    let (key, value) = engram.lookup_key_value(&[0, 2], 1, 2);
    let direct = engram.forward_layer(&x, &key, &value, None);
    assert_close_slice(
        via_hash.inner.borrow().value.data.as_ref(),
        direct.inner.borrow().value.data.as_ref(),
        1e-6,
    );

    let skipped_key =
        Tensor::from_value_no_grad(TensorValue::from_vec(Shape(vec![1, 2, 1, 2]), vec![0.0; 4]));
    let skipped_value =
        Tensor::from_value_no_grad(TensorValue::from_vec(Shape(vec![1, 2, 2]), vec![0.0; 4]));
    let skipped = engram.forward_layer(&x, &skipped_key, &skipped_value, None);
    assert_ne!(
        via_hash.inner.borrow().value.data.as_ref(),
        skipped.inner.borrow().value.data.as_ref(),
        "hash runtime must not skip table lookup and wkv projection"
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
        causal_encoder_layers: 1,
        decoder_layers: 0,
        embed_tokens: tensor_param(&params["embed_tokens"]),
        layers: vec![block_from_fixture(&params["block_fixture"])],
        final_norm: tensor_param(&params["final_norm"]),
        lm_head: tensor_param(&params["lm_head"]),
        engram_runtime: None,
        vision: None,
        image_start: None,
        image_end: None,
        image_newline: None,
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

/// Level-6 tiny DSpark `forward_spec` parity: builds a single-stage DSpark head
/// from the fixture weights and checks the biased draft logits, greedy output
/// ids, and confidence against the composed upstream reference.
#[test]
fn tiny_dspark_head_forward_spec_matches_reference() {
    let fixture = fixture("dspark_tiny_model_fixture.json");
    assert_fixture_meta(&fixture, "Transformer.forward_spec");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let params = &fixture["parameters"];

    let stage = DeepSeekV41DsparkStage {
        block: {
            // The nested block fixture pins its gate token count to its own tiny
            // sequence; the DSpark draft stream is `batch * block_size` tokens,
            // so retarget the MoE gate's token count to the draft grid.
            let mut block = block_from_fixture(&params["block_fixture"]);
            block.ffn.gate.tokens = usize_field(input, "batch") * usize_field(input, "block_size");
            block
        },
        main_proj: Some(param_vec(&params["main_proj"])),
        main_norm: Some(param_vec(&params["main_norm"])),
        head_norm: Some(param_vec(&params["head_norm"])),
        markov_embed: Some(param_vec(&params["markov_embed"])),
        markov_head: Some(param_vec(&params["markov_head"])),
        confidence_proj: Some(param_vec(&params["confidence_proj"])),
    };
    let head = DeepSeekV41DsparkHead {
        vocab_size: usize_field(input, "vocab_size"),
        dim: usize_field(input, "dim"),
        hc_mult: usize_field(input, "hc_mult"),
        block_size: usize_field(input, "block_size"),
        noise_token_id: usize_field(input, "noise_token_id"),
        markov_rank: usize_field(input, "markov_rank"),
        head_eps: f32_field(input, "head_eps"),
        embed_tokens: tensor_param(&params["embed_tokens"]),
        lm_head: tensor_param(&params["lm_head"]),
        stages: vec![stage],
        stage_swa_caches: RefCell::new(vec![None]),
    };

    let input_ids = usize_array(input, "input_ids");
    let main_hidden = f32_array(input, "main_hidden");
    let out = head.forward_spec(&input_ids, &main_hidden);

    let expected = &fixture["expected"];
    assert_eq!(
        out.output_ids,
        usize_array(expected, "output_ids"),
        "greedy draft output ids must match"
    );
    assert_close(
        &out.logits,
        &f32_array(expected, "logits"),
        tol,
        "dspark biased logits",
    );
    assert_close(
        &out.confidence,
        &f32_array(expected, "confidence"),
        tol,
        "dspark confidence",
    );
}

fn dspark_head_from_fixture(fixture: &Value) -> DeepSeekV41DsparkHead {
    let input = &fixture["input"];
    let params = &fixture["parameters"];
    let stage = DeepSeekV41DsparkStage {
        block: {
            let mut block = block_from_fixture(&params["block_fixture"]);
            block.ffn.gate.tokens = usize_field(input, "batch") * usize_field(input, "block_size");
            block
        },
        main_proj: Some(param_vec(&params["main_proj"])),
        main_norm: Some(param_vec(&params["main_norm"])),
        head_norm: Some(param_vec(&params["head_norm"])),
        markov_embed: Some(param_vec(&params["markov_embed"])),
        markov_head: Some(param_vec(&params["markov_head"])),
        confidence_proj: Some(param_vec(&params["confidence_proj"])),
    };
    DeepSeekV41DsparkHead {
        vocab_size: usize_field(input, "vocab_size"),
        dim: usize_field(input, "dim"),
        hc_mult: usize_field(input, "hc_mult"),
        block_size: usize_field(input, "block_size"),
        noise_token_id: usize_field(input, "noise_token_id"),
        markov_rank: usize_field(input, "markov_rank"),
        head_eps: f32_field(input, "head_eps"),
        embed_tokens: tensor_param(&params["embed_tokens"]),
        lm_head: tensor_param(&params["lm_head"]),
        stages: vec![stage],
        stage_swa_caches: RefCell::new(vec![None]),
    }
}

#[test]
fn dspark_forward_spec_prefill_returns_none() {
    let fixture = fixture("dspark_tiny_model_fixture.json");
    let input = &fixture["input"];
    let head = dspark_head_from_fixture(&fixture);
    let input_ids = usize_array(input, "input_ids");
    let main_hidden = f32_array(input, "main_hidden");

    let out = head
        .try_forward_spec_at(&input_ids, &main_hidden, 0)
        .expect("prefill should seed and return no draft");

    assert!(
        out.is_none(),
        "DSpark forward_spec(start_pos=0) should return None"
    );
}

#[test]
fn dspark_decode_uses_main_hidden_and_start_pos() {
    let fixture = fixture("dspark_tiny_model_fixture.json");
    let input = &fixture["input"];
    let head = dspark_head_from_fixture(&fixture);
    let input_ids = usize_array(input, "input_ids");
    let main_hidden = f32_array(input, "main_hidden");
    let mut changed_main_hidden = main_hidden.clone();
    changed_main_hidden[0] += 1.0;

    let out = head
        .try_forward_spec_at(&input_ids, &main_hidden, 2)
        .expect("decode should succeed")
        .expect("decode should return draft output");
    let changed = head
        .try_forward_spec_at(&input_ids, &changed_main_hidden, 2)
        .expect("decode with changed main_hidden should succeed")
        .expect("decode should return draft output");

    assert_ne!(
        out.logits, changed.logits,
        "DSpark decode must use main_x derived from main_hidden"
    );

    let start_pos_zero = head
        .try_forward_spec_at(&input_ids, &main_hidden, 0)
        .expect("prefill should succeed");
    assert!(
        start_pos_zero.is_none(),
        "start_pos must not be hard-coded to decode behavior"
    );
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

fn param_vec(value: &Value) -> Vec<f32> {
    f32_array(value, "data")
}

/// The VL prompt fixture is produced by calling upstream `prepare_vl_inputs`
/// (with a stubbed tokenizer + `load_image`), so it pins the placeholder->span
/// expansion and `image_token_types` layout the multimodal CLI mirrors. The
/// assertions here reconstruct that layout independently and check it matches.
#[test]
fn vl_prompt_fixture_expands_image_span() {
    let fixture = fixture("prompt_vl_fixture.json");
    assert_eq!(
        string_field(&fixture, "source_file"),
        "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/image_processor.py"
    );
    let status = string_field(&fixture, "upstream_call_status");
    assert!(
        status.contains("called-upstream:prepare_vl_inputs"),
        "expected upstream prepare_vl_inputs execution, got {status:?}"
    );

    let input = &fixture["input"];
    let image_token_id = usize_field(input, "image_token_id");
    let expected = &fixture["expected"];
    let tokens = usize_array(expected, "tokens");
    let token_types = i64_array(expected, "token_types");
    assert_eq!(
        tokens.len(),
        token_types.len(),
        "tokens and token_types must align 1:1"
    );

    // Exactly one image span here; reconstruct its `[START, (IMAGE*w, NL)*h, END]`
    // token-type layout and check the fixture matches upstream.
    let images = expected["images"].as_array().expect("images array");
    assert_eq!(images.len(), 1, "vl fixture carries a single image");
    let img = &images[0];
    let start = usize_field(img, "start");
    let n_llm_h = usize_field(img, "n_llm_h");
    let n_llm_w = usize_field(img, "n_llm_w");
    let span_types = i64_array(img, "token_types");

    // IMAGE_START=0, IMAGE=1, IMAGE_NEW_LINE=2, IMAGE_END=3, TEXT=-1.
    let mut want_span: Vec<i64> = vec![0];
    for _ in 0..n_llm_h {
        want_span.extend(std::iter::repeat(1).take(n_llm_w));
        want_span.push(2);
    }
    want_span.push(3);
    assert_eq!(span_types, want_span, "reconstructed image span layout");
    assert_eq!(
        span_types.len(),
        n_llm_h * (n_llm_w + 1) + 2,
        "num_image_tokens = n_llm_h*(n_llm_w+1)+2"
    );

    // The span sits inside the full sequence at `start`, and every span position
    // carries the image token id (only token_types tells them apart).
    for (k, &ty) in span_types.iter().enumerate() {
        let pos = start + k;
        assert_eq!(token_types[pos], ty, "span type at {pos}");
        assert_eq!(
            tokens[pos], image_token_id,
            "image span position {pos} must carry image_token_id"
        );
    }
    // Text positions carry TEXT and are not the image token id.
    assert_eq!(token_types[0], -1, "leading text position is TEXT");
    assert_ne!(tokens[0], image_token_id, "leading token is text");
}

#[test]
fn vision_block_matches_upstream_vit() {
    use tnsr::deepseek_v41::vision::{
        apply_rotary_half_split, vision_block, vision_cos_sin, vision_patch_embed,
        VisionBlockWeights,
    };

    let fixture = fixture("vision_block_fixture.json");
    assert_eq!(
        string_field(&fixture, "source_file"),
        "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py"
    );
    let status = string_field(&fixture, "upstream_call_status");
    assert!(
        status.contains("called-upstream:ViT"),
        "expected upstream vision execution, got {status:?}"
    );

    let input = &fixture["input"];
    let dim = usize_field(input, "dim");
    let n_heads = usize_field(input, "n_heads");
    let head_dim = usize_field(input, "head_dim");
    let rope_dim = usize_field(input, "rope_dim");
    let inter = usize_field(input, "inter");
    let patch_flat = usize_field(input, "patch_flat");
    let theta = f32_field(input, "theta") as f64;
    let n_h = usize_field(input, "n_h");
    let n_w = usize_field(input, "n_w");
    let n_patch = usize_field(input, "n_patch");
    let patches = f32_array(input, "patches");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let params = &fixture["parameters"];
    let proj_w = param_vec(&params["proj_w"]);
    let proj_b = param_vec(&params["proj_b"]);
    let wqkv = param_vec(&params["wqkv"]);
    let wqkv_b = param_vec(&params["wqkv_b"]);
    let wo = param_vec(&params["wo"]);
    let wo_b = param_vec(&params["wo_b"]);
    let norm1 = param_vec(&params["norm1"]);
    let norm2 = param_vec(&params["norm2"]);
    let w1 = param_vec(&params["w1"]);
    let w2 = param_vec(&params["w2"]);

    // 2D RoPE tables.
    let (cos, sin) = vision_cos_sin(n_h, n_w, rope_dim, theta);
    assert_close(&cos, &f32_array(&fixture["expected"], "cos"), tol, "cos");
    assert_close(&sin, &f32_array(&fixture["expected"], "sin"), tol, "sin");

    // Patch embed.
    let embedded = vision_patch_embed(&patches, &proj_w, &proj_b, n_patch, patch_flat, dim);
    assert_close(
        &embedded,
        &f32_array(&fixture["expected"], "embedded"),
        tol,
        "embedded",
    );

    // Half-split RoPE applied to the raw embedded patches (upstream applies to
    // `embedded.view(n_patch, n_heads, head_dim)`).
    let rope_out = apply_rotary_half_split(&embedded, &cos, &sin, n_patch, n_heads, head_dim);
    assert_close(
        &rope_out,
        &f32_array(&fixture["expected"], "rope_out"),
        tol,
        "rope_out",
    );

    // Full block.
    let weights = VisionBlockWeights {
        norm1: &norm1,
        wqkv: &wqkv,
        wqkv_b: &wqkv_b,
        wo: &wo,
        wo_b: &wo_b,
        norm2: &norm2,
        w1: &w1,
        w2: &w2,
    };
    let block_out = vision_block(
        &embedded, &weights, &cos, &sin, n_patch, dim, n_heads, inter,
    );
    assert_close(
        &block_out,
        &f32_array(&fixture["expected"], "block_out"),
        tol,
        "block_out",
    );
}

#[test]
fn aligner_matches_upstream_downsample() {
    use tnsr::deepseek_v41::vision::{aligner_forward, AlignerWeights};

    let fixture = fixture("aligner_fixture.json");
    assert_eq!(
        string_field(&fixture, "source_file"),
        "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py"
    );
    let status = string_field(&fixture, "upstream_call_status");
    assert!(
        status.contains("called-upstream:Aligner"),
        "expected upstream aligner execution, got {status:?}"
    );

    let input = &fixture["input"];
    let vision_dim = usize_field(input, "vision_dim");
    let downsample_ratio = usize_field(input, "downsample_ratio");
    let dim = usize_field(input, "dim");
    let n_h = usize_field(input, "n_h");
    let n_w = usize_field(input, "n_w");
    let vit_rows = f32_array(input, "vit_rows");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let params = &fixture["parameters"];
    let w1 = param_vec(&params["w1"]);
    let w1_b = param_vec(&params["w1_b"]);
    let w2 = param_vec(&params["w2"]);
    let w2_b = param_vec(&params["w2_b"]);

    let weights = AlignerWeights {
        w1: &w1,
        w1_b: &w1_b,
        w2: &w2,
        w2_b: &w2_b,
    };
    let aligned = aligner_forward(
        &vit_rows,
        &weights,
        n_h,
        n_w,
        vision_dim,
        dim,
        downsample_ratio,
    );
    assert_close(
        &aligned,
        &f32_array(&fixture["expected"], "aligned"),
        tol,
        "aligned",
    );
}

#[test]
fn encode_image_matches_upstream_vit_plus_aligner() {
    use tnsr::deepseek_v41::vision::{
        encode_image, vit_forward, AlignerWeights, VisionBlockWeights, VitWeights,
    };

    let fixture = fixture("encode_image_fixture.json");
    assert_eq!(
        string_field(&fixture, "source_file"),
        "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py"
    );
    let status = string_field(&fixture, "upstream_call_status");
    assert!(
        status.contains("called-upstream:EncodeImage"),
        "expected upstream encode_image execution, got {status:?}"
    );

    let input = &fixture["input"];
    let dim = usize_field(input, "dim");
    let n_heads = usize_field(input, "n_heads");
    let rope_dim = usize_field(input, "rope_dim");
    let inter = usize_field(input, "inter");
    let patch_flat = usize_field(input, "patch_flat");
    let theta = f32_field(input, "theta") as f64;
    let downsample_ratio = usize_field(input, "downsample_ratio");
    let llm_dim = usize_field(input, "llm_dim");
    let n_h = usize_field(input, "n_h");
    let n_w = usize_field(input, "n_w");
    let patches = f32_array(input, "patches");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let params = &fixture["parameters"];
    let proj_w = param_vec(&params["proj_w"]);
    let proj_b = param_vec(&params["proj_b"]);
    let vit_norm = param_vec(&params["vit_norm"]);

    // Own the per-block Vec<f32> so the borrowed VisionBlockWeights outlive use.
    let block_data: Vec<[Vec<f32>; 8]> = params["blocks"]
        .as_array()
        .expect("blocks array")
        .iter()
        .map(|b| {
            [
                param_vec(&b["norm1"]),
                param_vec(&b["wqkv"]),
                param_vec(&b["wqkv_b"]),
                param_vec(&b["wo"]),
                param_vec(&b["wo_b"]),
                param_vec(&b["norm2"]),
                param_vec(&b["w1"]),
                param_vec(&b["w2"]),
            ]
        })
        .collect();
    let blocks: Vec<VisionBlockWeights> = block_data
        .iter()
        .map(|b| VisionBlockWeights {
            norm1: &b[0],
            wqkv: &b[1],
            wqkv_b: &b[2],
            wo: &b[3],
            wo_b: &b[4],
            norm2: &b[5],
            w1: &b[6],
            w2: &b[7],
        })
        .collect();

    let tower = VitWeights {
        proj_w: &proj_w,
        proj_b: &proj_b,
        blocks,
        final_norm: &vit_norm,
    };

    // ViT rows first, so a tower regression is localised.
    let vit_rows = vit_forward(
        &patches, &tower, n_h, n_w, patch_flat, dim, n_heads, inter, rope_dim, theta,
    );
    assert_close(
        &vit_rows,
        &f32_array(&fixture["expected"], "vit_rows"),
        tol,
        "vit_rows",
    );

    let al_w1 = param_vec(&params["al_w1"]);
    let al_w1_b = param_vec(&params["al_w1_b"]);
    let al_w2 = param_vec(&params["al_w2"]);
    let al_w2_b = param_vec(&params["al_w2_b"]);
    let aligner = AlignerWeights {
        w1: &al_w1,
        w1_b: &al_w1_b,
        w2: &al_w2,
        w2_b: &al_w2_b,
    };

    let aligned = encode_image(
        &patches,
        &tower,
        &aligner,
        n_h,
        n_w,
        patch_flat,
        dim,
        llm_dim,
        n_heads,
        inter,
        rope_dim,
        theta,
        downsample_ratio,
    );
    assert_close(
        &aligned,
        &f32_array(&fixture["expected"], "aligned"),
        tol,
        "aligned",
    );
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label} length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{label}[{i}]: got {g:.8}, want {w:.8}, diff {:.8}, tol {tol}",
            (g - w).abs()
        );
    }
}

#[test]
fn vl_gate_selects_vision_bias_on_image_tokens() {
    use tnsr::deepseek_v41::moe::DeepSeekV41Gate;

    let fixture = fixture("vl_gate_fixture.json");
    assert_fixture_meta(&fixture, "Gate.forward(image_mask)");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let shape = &input["shape"];
    let tokens = usize_field(shape, "tokens");
    let topk = usize_field(input, "topk");

    let gate = DeepSeekV41Gate {
        weight: f32_array(input, "gate_weight"),
        correction_bias: f32_array(input, "correction_bias"),
        bias_vl: Some(f32_array(input, "bias_vl")),
        tokens,
        dim: usize_field(shape, "dim"),
        experts: usize_field(shape, "experts"),
        topk,
        gate_temp: f32_field(input, "gate_temp"),
        norm_topk_prob: input["norm_topk_prob"].as_bool().unwrap(),
        route_scale: f32_field(input, "route_scale"),
    };
    let image_mask = bool_array(input, "image_mask");

    let (_, indices, weights) = gate.forward_masked(&f32_array(input, "x"), Some(&image_mask));
    let want_indices = usize_array(&fixture["expected"], "indices");
    assert_eq!(indices, want_indices, "vl-gate expert selection");
    assert_close(
        &weights,
        &f32_array(&fixture["expected"], "weights"),
        tol,
        "vl-gate weights",
    );

    // Sanity: without the mask, the image tokens select different experts (the
    // text bias, not bias_vl), proving the mask actually changed selection.
    let (_, plain_indices, _) = gate.forward_masked(&f32_array(input, "x"), None);
    assert_ne!(
        plain_indices, want_indices,
        "bias_vl should change image-token selection vs the text bias"
    );
}

#[test]
fn merge_image_embeddings_overwrites_span() {
    use tnsr::deepseek_v41::model::{merge_image_embeddings, ImageDelimiters, ImageSpan};

    let fixture = fixture("merge_image_embeddings_fixture.json");
    assert_fixture_meta(&fixture, "Transformer.merge_image_embeddings");
    let input = &fixture["input"];
    let b = usize_field(input, "b");
    let s = usize_field(input, "s");
    let dim = usize_field(input, "dim");
    let start = usize_field(input, "start");
    let token_types = i64_array(input, "token_types");
    let aligner_rows = f32_array(input, "aligner_rows");
    let image_start = f32_array(input, "image_start");
    let image_end = f32_array(input, "image_end");
    let image_newline = f32_array(input, "image_newline");

    let mut embed = f32_array(input, "embed");
    let span = ImageSpan {
        start,
        token_types: &token_types,
        aligner_rows: &aligner_rows,
    };
    let images = vec![vec![span]];
    let delims = ImageDelimiters {
        image_start: &image_start,
        image_end: &image_end,
        image_newline: &image_newline,
    };
    merge_image_embeddings(&mut embed, b, s, dim, &images, &delims).unwrap();
    assert_close(
        &embed,
        &f32_array(&fixture["expected"], "merged"),
        0.0,
        "merged",
    );
}
