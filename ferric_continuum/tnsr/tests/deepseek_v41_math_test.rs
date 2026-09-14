use std::path::{Path, PathBuf};

use serde_json::Value;
use tnsr::deepseek_v41::{
    config::DeepSeekV41TextConfig,
    dspark::{
        confidence_head_forward, decode_topk_indices, draft_input_ids, draft_loop, main_proj_norm,
        markov_head_forward, DraftLoopOutput,
    },
    engram::{
        compressed_token_map_from_vocab_entries, engram_update, ngram_hashes, EngramLayout,
        NgramHashState,
    },
    hyper::{hc_mixes, hc_post, hc_pre, HcShape},
    moe::{expert_swiglu, route_weights, select_experts, sqrtsoftplus_scores, GateShape},
    rope::{apply_rotary_adjacent_pairs, precompute_freqs, RopeShape, YarnRopeConfig},
    sparse::{
        compress_ratio_n, compress_ratio_one, select_candidate_blocks, window_topk_indices,
        CandidateLens, CandidateShape,
    },
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
        .map(|v| match v {
            Value::Number(_) => {
                v.as_f64()
                    .unwrap_or_else(|| panic!("{key} item should be number")) as f32
            }
            Value::String(s) if s == "-inf" => f32::NEG_INFINITY,
            Value::String(s) if s == "inf" => f32::INFINITY,
            other => panic!("{key} item should be number or infinity string, got {other}"),
        })
        .collect()
}

fn i32_array(value: &Value, key: &str) -> Vec<i32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} should be array"))
        .iter()
        .map(|v| {
            v.as_i64()
                .unwrap_or_else(|| panic!("{key} item should be integer")) as i32
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

fn bool_field(value: &Value, key: &str) -> bool {
    value[key]
        .as_bool()
        .unwrap_or_else(|| panic!("{key} should be bool"))
}

fn string_field(value: &Value, key: &str) -> String {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} should be string"))
        .to_string()
}

fn assert_fixture_meta(fixture: &Value, upstream_function: &str) {
    assert_eq!(fixture["upstream_function"], upstream_function);

    let source_file = fixture["source_file"]
        .as_str()
        .expect("source_file should be string");
    assert!(
        source_file == "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/model.py"
            || source_file
                == "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/engram.py",
        "unexpected source_file {source_file:?}"
    );

    let line_hint = fixture["line_hint"]
        .as_str()
        .expect("line_hint should be string");
    assert!(
        line_hint.contains(upstream_function),
        "line_hint {line_hint:?} should name {upstream_function}"
    );

    assert!(
        fixture.get("absolute_tolerance").is_some(),
        "absolute_tolerance should be present"
    );
    assert!(
        fixture["absolute_tolerance"].is_number(),
        "absolute_tolerance should be numeric"
    );

    let status = fixture
        .get("upstream_call_status")
        .or_else(|| fixture.get("upstream_import_status"))
        .expect("fixture should include upstream_call_status or upstream_import_status")
        .as_str()
        .expect("upstream status should be string");
    assert!(
        status.starts_with("called-upstream:")
            || status.starts_with("fallback:")
            || status.starts_with("fixture-seam:")
            || status.starts_with("path-import-skipped"),
        "unexpected upstream status {status:?}"
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

#[test]
fn image_grid_matches_upstream_plan_and_token_types() {
    use tnsr::deepseek_v41::config::DeepSeekV41VisionConfig;
    use tnsr::deepseek_v41::vision_grid::{image_token_types, num_image_tokens, plan_image_grid};

    let fixture = fixture("image_grid_fixture.json");
    assert_eq!(
        fixture["source_file"],
        "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/image_processor.py"
    );
    let status = fixture["upstream_call_status"]
        .as_str()
        .expect("upstream_call_status should be string");
    assert!(
        status.contains("called-upstream:plan_image_grid"),
        "expected upstream execution, got {status:?}"
    );

    let vc = &fixture["vision_config"];
    let cfg = DeepSeekV41VisionConfig {
        num_hidden_layers: 32,
        hidden_size: 1024,
        num_attention_heads: 16,
        intermediate_size: 2816,
        patch_size: usize_field(vc, "vision_patch_size"),
        rope_theta: 10000.0,
        downsample_ratio: usize_field(vc, "vision_downsample_ratio"),
        max_image_tokens: usize_field(vc, "vision_max_n_token"),
        min_pixels: usize_field(vc, "vision_min_pixels"),
        max_wh_ratio: vc["vision_max_wh_ratio"].as_f64(),
    };

    for case in fixture["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().unwrap_or("?");
        let width = usize_field(case, "width");
        let height = usize_field(case, "height");
        let expected = &case["expected"];

        let (n_llm_h, n_llm_w, best_h, best_w) = plan_image_grid(width, height, &cfg);
        assert_eq!(n_llm_h, usize_field(expected, "n_llm_h"), "{name} n_llm_h");
        assert_eq!(n_llm_w, usize_field(expected, "n_llm_w"), "{name} n_llm_w");
        assert_eq!(
            best_h,
            usize_field(expected, "best_height"),
            "{name} best_height"
        );
        assert_eq!(
            best_w,
            usize_field(expected, "best_width"),
            "{name} best_width"
        );
        assert_eq!(
            num_image_tokens(n_llm_h, n_llm_w),
            usize_field(expected, "num_image_tokens"),
            "{name} num_image_tokens"
        );
        assert_eq!(
            image_token_types(n_llm_h, n_llm_w),
            i64_array(expected, "token_types"),
            "{name} token_types"
        );
    }
}

#[test]
fn yarn_freqs_match_upstream_selected_positions() {
    let fixture = fixture("rope_yarn_fixture.json");
    assert_fixture_meta(&fixture, "precompute_freqs_cis");

    let cfg = &fixture["input"]["config"];
    let config = YarnRopeConfig {
        rope_head_dim: usize_field(cfg, "rope_head_dim"),
        max_seq_len: usize_field(cfg, "max_seq_len"),
        original_seq_len: usize_field(cfg, "original_seq_len"),
        rope_theta: f32_field(cfg, "rope_theta"),
        rope_factor: f32_field(cfg, "rope_factor"),
        beta_fast: f32_field(cfg, "beta_fast"),
        beta_slow: f32_field(cfg, "beta_slow"),
    };
    let freqs = precompute_freqs(&config, usize_field(&fixture["input"], "seqlen"));

    let expected_positions = fixture["expected"]["selected_freqs"]
        .as_array()
        .expect("selected freqs array");
    for sample in expected_positions {
        let pos = usize_field(sample, "position");
        let pair = usize_field(sample, "pair");
        let got = freqs[pos * (config.rope_head_dim / 2) + pair];
        let want = [f32_field(sample, "cos"), f32_field(sample, "sin")];
        assert_close_slice(&got, &want, f32_field(&fixture, "absolute_tolerance"));
    }
}

#[test]
fn adjacent_pair_rope_forward_and_inverse_match_upstream() {
    let fixture = fixture("rope_yarn_fixture.json");
    assert_fixture_meta(&fixture, "precompute_freqs_cis");

    let cfg = &fixture["input"]["config"];
    let config = YarnRopeConfig {
        rope_head_dim: usize_field(cfg, "rope_head_dim"),
        max_seq_len: usize_field(cfg, "max_seq_len"),
        original_seq_len: usize_field(cfg, "original_seq_len"),
        rope_theta: f32_field(cfg, "rope_theta"),
        rope_factor: f32_field(cfg, "rope_factor"),
        beta_fast: f32_field(cfg, "beta_fast"),
        beta_slow: f32_field(cfg, "beta_slow"),
    };
    let freqs = precompute_freqs(&config, usize_field(&fixture["input"], "seqlen"));
    let tol = f32_field(&fixture, "absolute_tolerance");

    let mut data3 = f32_array(&fixture["input"], "rope_3d_data");
    let original3 = data3.clone();
    let shape3 = &fixture["input"]["rope_3d_shape"];
    apply_rotary_adjacent_pairs(
        &mut data3,
        RopeShape::Bsd {
            batch: usize_field(shape3, "batch"),
            seqlen: usize_field(shape3, "seqlen"),
            dim: usize_field(shape3, "dim"),
        },
        &freqs,
        false,
    );
    assert_close_slice(
        &data3,
        &f32_array(&fixture["expected"], "rope_3d_forward"),
        tol,
    );
    apply_rotary_adjacent_pairs(
        &mut data3,
        RopeShape::Bsd {
            batch: usize_field(shape3, "batch"),
            seqlen: usize_field(shape3, "seqlen"),
            dim: usize_field(shape3, "dim"),
        },
        &freqs,
        true,
    );
    assert_close_slice(&data3, &original3, tol * 4.0);

    let mut data4 = f32_array(&fixture["input"], "rope_4d_data");
    let original4 = data4.clone();
    let shape4 = &fixture["input"]["rope_4d_shape"];
    apply_rotary_adjacent_pairs(
        &mut data4,
        RopeShape::Bshd {
            batch: usize_field(shape4, "batch"),
            seqlen: usize_field(shape4, "seqlen"),
            heads: usize_field(shape4, "heads"),
            dim: usize_field(shape4, "dim"),
        },
        &freqs,
        false,
    );
    assert_close_slice(
        &data4,
        &f32_array(&fixture["expected"], "rope_4d_forward"),
        tol,
    );
    apply_rotary_adjacent_pairs(
        &mut data4,
        RopeShape::Bshd {
            batch: usize_field(shape4, "batch"),
            seqlen: usize_field(shape4, "seqlen"),
            heads: usize_field(shape4, "heads"),
            dim: usize_field(shape4, "dim"),
        },
        &freqs,
        true,
    );
    assert_close_slice(&data4, &original4, tol * 4.0);
}

#[test]
fn window_topk_indices_match_prefill_and_decode_sentinel_rows() {
    let fixture = fixture("window_topk_fixture.json");
    assert_fixture_meta(&fixture, "get_window_topk_idxs");

    for case in fixture["cases"].as_array().expect("cases array") {
        let input = &case["input"];
        let got = window_topk_indices(
            usize_field(input, "window_size"),
            usize_field(input, "batch"),
            usize_field(input, "seqlen"),
            usize_field(input, "start_pos"),
        );
        assert_eq!(got, i32_array(case, "expected"));
    }
}

#[test]
fn candidate_block_selection_pins_newest_and_drops_unreachable_blocks() {
    let fixture = fixture("candidate_blocks_fixture.json");
    assert_fixture_meta(&fixture, "select_candidate_blocks");
    let input = &fixture["input"];
    let got = select_candidate_blocks(
        &f32_array(input, "logits"),
        CandidateShape {
            batch: usize_field(&input["shape"], "batch"),
            seqlen: usize_field(&input["shape"], "seqlen"),
            positions: usize_field(&input["shape"], "positions"),
        },
        CandidateLens::PerQuery(i32_array(input, "compress_lens")),
        usize_field(input, "topk_blocks"),
        usize_field(input, "block_size"),
    );

    assert_eq!(got, bool_array(&fixture["expected"], "mask"));
}

#[test]
fn dspark_topk_indices_match_decode_window_and_draft_positions() {
    let fixture = fixture("dspark_topk_fixture.json");
    assert_fixture_meta(&fixture, "get_dspark_topk_idxs");

    for case in fixture["cases"].as_array().expect("cases array") {
        let input = &case["input"];
        let got = decode_topk_indices(
            usize_field(input, "window_size"),
            usize_field(input, "batch"),
            usize_field(input, "block_size"),
            usize_field(input, "start_pos"),
        );
        assert_eq!(got, i32_array(case, "expected"));
    }
}

#[test]
fn dspark_markov_head_matches_upstream() {
    let fixture = fixture("dspark_markov_fixture.json");
    assert_fixture_meta(&fixture, "DSparkMarkovHead.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let (logits, markov_embed) = markov_head_forward(
        usize_field(input, "token_id"),
        &f32_array(input, "embed"),
        &f32_array(input, "head"),
        usize_field(input, "vocab_size"),
        usize_field(input, "rank"),
    );
    let expected = &fixture["expected"];
    assert_close_slice(&logits, &f32_array(expected, "logits"), tol);
    assert_close_slice(&markov_embed, &f32_array(expected, "markov_embed"), tol);
}

#[test]
fn dspark_confidence_head_matches_upstream() {
    let fixture = fixture("dspark_confidence_fixture.json");
    assert_fixture_meta(&fixture, "DSparkConfidenceHead.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let got = confidence_head_forward(
        &f32_array(input, "hidden"),
        &f32_array(input, "markov_embed"),
        &f32_array(input, "proj"),
    );
    let want = f32_field(&fixture["expected"], "confidence");
    assert!(
        (got - want).abs() <= tol,
        "confidence mismatch got {got} want {want}"
    );
}

#[test]
fn dspark_draft_input_ids_place_accepted_tokens_then_noise() {
    let fixture = fixture("dspark_draft_input_fixture.json");
    assert_fixture_meta(&fixture, "DSparkBlock.forward_embed");
    let input = &fixture["input"];
    let got = draft_input_ids(
        &usize_array(input, "input_ids"),
        usize_field(input, "batch"),
        usize_field(input, "block_size"),
        usize_field(input, "noise_token_id"),
    );
    assert_eq!(got, usize_array(&fixture["expected"], "draft_input_ids"));
}

#[test]
fn dspark_main_proj_norm_matches_upstream() {
    let fixture = fixture("dspark_main_proj_norm_fixture.json");
    assert_fixture_meta(&fixture, "DSparkBlock.forward_embed");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let got = main_proj_norm(
        &f32_array(input, "main_hidden"),
        &f32_array(input, "proj"),
        &f32_array(input, "norm_weight"),
        usize_field(input, "in_dim"),
        usize_field(input, "dim"),
        f32_field(input, "eps"),
    );
    assert_close_slice(&got, &f32_array(&fixture["expected"], "out"), tol);
}

#[test]
fn dspark_draft_loop_matches_upstream_forward_head() {
    let fixture = fixture("dspark_draft_loop_fixture.json");
    assert_fixture_meta(&fixture, "DSparkBlock.forward_head");
    let tol = f32_field(&fixture, "absolute_tolerance");
    let input = &fixture["input"];
    let DraftLoopOutput {
        output_ids,
        logits,
        confidence,
    } = draft_loop(
        usize_field(input, "input_id"),
        &f32_array(input, "base_logits"),
        &f32_array(input, "hidden"),
        &f32_array(input, "markov_embed"),
        &f32_array(input, "markov_head"),
        &f32_array(input, "confidence_proj"),
        usize_field(input, "block_size"),
        usize_field(input, "vocab_size"),
        usize_field(input, "rank"),
        usize_field(input, "dim"),
    );
    let expected = &fixture["expected"];
    assert_eq!(output_ids, usize_array(expected, "output_ids"));
    assert_close_slice(&logits, &f32_array(expected, "logits"), tol);
    assert_close_slice(&confidence, &f32_array(expected, "confidence"), tol);
}

#[test]
fn compressor_ratio_one_and_prefill_pooling_match_upstream() {
    let fixture = fixture("compressor_fixture.json");
    assert_fixture_meta(&fixture, "Compressor.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let ratio_one = &fixture["ratio_one"];
    let got_one = compress_ratio_one(
        &f32_array(&ratio_one["input"], "kv"),
        &f32_array(&ratio_one["input"], "norm_weight"),
        f32_field(&ratio_one["input"], "eps"),
    );
    assert_close_slice(&got_one, &f32_array(ratio_one, "expected"), tol);

    let ratio_n = &fixture["ratio_n_prefill"];
    let got_n = compress_ratio_n(
        &f32_array(&ratio_n["input"], "kv"),
        &f32_array(&ratio_n["input"], "score"),
        usize_field(&ratio_n["input"], "ratio"),
        &f32_array(&ratio_n["input"], "norm_weight"),
        f32_field(&ratio_n["input"], "eps"),
    );
    assert_close_slice(&got_n, &f32_array(ratio_n, "expected"), tol);
}

#[test]
#[should_panic(expected = "candidate logits must not contain NaN")]
fn candidate_block_selection_rejects_nan_logits() {
    select_candidate_blocks(
        &[0.0, f32::NAN],
        CandidateShape {
            batch: 1,
            seqlen: 1,
            positions: 2,
        },
        CandidateLens::Scalar(1),
        1,
        1,
    );
}

#[test]
#[should_panic(expected = "compress_lens must be non-negative")]
fn candidate_block_selection_rejects_negative_lens() {
    select_candidate_blocks(
        &[0.0, 1.0],
        CandidateShape {
            batch: 1,
            seqlen: 1,
            positions: 2,
        },
        CandidateLens::PerQuery(vec![-1]),
        1,
        1,
    );
}

#[test]
fn moe_sqrtsoftplus_routing_uses_bias_for_indices_but_not_weights() {
    let fixture = fixture("moe_gate_fixture.json");
    assert_fixture_meta(&fixture, "Gate.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let input = &fixture["input"];
    let shape = GateShape {
        tokens: usize_field(&input["shape"], "tokens"),
        dim: usize_field(&input["shape"], "dim"),
        experts: usize_field(&input["shape"], "experts"),
    };
    let got_scores = sqrtsoftplus_scores(
        &f32_array(input, "x"),
        &f32_array(input, "gate_weight"),
        f32_field(input, "gate_temp"),
        shape,
    );
    assert_close_slice(&got_scores, &f32_array(&fixture["expected"], "scores"), tol);

    let got_indices = select_experts(
        &got_scores,
        &f32_array(input, "correction_bias"),
        usize_field(input, "topk"),
    );
    assert_eq!(got_indices, usize_array(&fixture["expected"], "indices"));

    let expected_weights = f32_array(&fixture["expected"], "weights");
    let topk = usize_field(input, "topk");
    for token in 0..shape.tokens {
        let score_row = &got_scores[token * shape.experts..(token + 1) * shape.experts];
        let idx_row = &got_indices[token * topk..(token + 1) * topk];
        let got_weights = route_weights(
            score_row,
            idx_row,
            bool_field(input, "norm_topk_prob"),
            f32_field(input, "route_scale"),
        );
        assert_close_slice(
            &got_weights,
            &expected_weights[token * topk..(token + 1) * topk],
            tol,
        );
    }
}

#[test]
fn moe_expert_swiglu_matches_upstream_clamps() {
    let fixture = fixture("expert_swiglu_fixture.json");
    assert_fixture_meta(&fixture, "Expert.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let input = &fixture["input"];
    let got = expert_swiglu(
        &f32_array(input, "gate"),
        &f32_array(input, "up"),
        f32_field(input, "swiglu_limit"),
    );
    assert_close_slice(&got, &f32_array(&fixture["expected"], "output"), tol);
}

#[test]
fn hyper_connection_pre_and_post_match_layer_level_fixture() {
    let fixture = fixture("hyper_connection_fixture.json");
    assert_fixture_meta(&fixture, "Block.hc_mixes");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let input = &fixture["input"];
    let mixes = hc_mixes(
        &f32_array(input, "flat_hc"),
        &f32_array(input, "hc_fn"),
        &f32_array(input, "hc_scale"),
        &f32_array(input, "hc_base"),
        usize_field(input, "hc_mult"),
        usize_field(input, "dim"),
        usize_field(input, "sinkhorn_iters"),
        f32_field(input, "eps"),
    );
    let expected = &fixture["expected"];
    assert_close_slice(&mixes.pre, &f32_array(expected, "pre"), tol);
    assert_close_slice(&mixes.post, &f32_array(expected, "post"), tol);
    assert_close_slice(&mixes.comb, &f32_array(expected, "comb"), tol);
    let sums = &fixture["expected"]["comb_sums"];
    assert_close_slice(&mixes.comb_row_sums, &f32_array(sums, "row"), tol * 10.0);
    assert_close_slice(&mixes.comb_col_sums, &f32_array(sums, "col"), tol * 10.0);

    let layer = &fixture["layer"];
    let input = &layer["input"];
    let shape = HcShape {
        batch: usize_field(&input["shape"], "batch"),
        seqlen: usize_field(&input["shape"], "seqlen"),
        hc_mult: usize_field(&input["shape"], "hc_mult"),
        dim: usize_field(&input["shape"], "dim"),
    };
    let pre = hc_pre(
        &f32_array(input, "residual"),
        &f32_array(input, "pre_mix"),
        shape,
    );
    assert_close_slice(&pre, &f32_array(&layer["expected"], "pre_out"), tol);

    let post = hc_post(
        &f32_array(input, "sublayer"),
        &f32_array(input, "residual"),
        &f32_array(input, "post"),
        &f32_array(input, "comb"),
        shape,
    );
    assert_close_slice(&post, &f32_array(&layer["expected"], "post_out"), tol);
}

#[test]
fn engram_layout_primes_and_offsets_match_upstream() {
    let fixture = fixture("engram_layout_fixture.json");
    assert_fixture_meta(&fixture, "EngramLayout.from_args");

    let config_value = &fixture["input"]["config"];
    let config = DeepSeekV41TextConfig {
        vocab_size: usize_field(config_value, "vocab_size"),
        hidden_size: usize_field(config_value, "hidden_size"),
        moe_intermediate_size: usize_field(config_value, "moe_intermediate_size"),
        num_hidden_layers: usize_field(config_value, "num_hidden_layers"),
        num_attention_heads: usize_field(config_value, "num_attention_heads"),
        num_key_value_heads: usize_field(config_value, "num_key_value_heads"),
        head_dim: usize_field(config_value, "head_dim"),
        qk_rope_head_dim: usize_field(config_value, "qk_rope_head_dim"),
        q_lora_rank: usize_field(config_value, "q_lora_rank"),
        o_lora_rank: usize_field(config_value, "o_lora_rank"),
        o_groups: usize_field(config_value, "o_groups"),
        rms_norm_eps: f32_field(config_value, "rms_norm_eps") as f64,
        rope_theta: f32_field(config_value, "rope_theta") as f64,
        rope_factor: f32_field(config_value, "rope_factor") as f64,
        original_seq_len: usize_field(config_value, "original_seq_len"),
        beta_fast: f32_field(config_value, "beta_fast") as f64,
        beta_slow: f32_field(config_value, "beta_slow") as f64,
        sliding_window: usize_field(config_value, "sliding_window"),
        compress_ratios: usize_array(config_value, "compress_ratios"),
        compress_rope_theta: f32_field(config_value, "compress_rope_theta") as f64,
        kv_source_layer_ids: usize_array(config_value, "kv_source_layer_ids"),
        index_source_layer_ids: usize_array(config_value, "index_source_layer_ids"),
        index_n_heads: usize_field(config_value, "index_n_heads"),
        index_head_dim: usize_field(config_value, "index_head_dim"),
        index_topk: usize_field(config_value, "index_topk"),
        candidate_source_layer_id: usize_field(config_value, "candidate_source_layer_id"),
        candidate_topk_blocks: usize_field(config_value, "candidate_topk_blocks"),
        candidate_block_size: usize_field(config_value, "candidate_block_size"),
        hc_mult: usize_field(config_value, "hc_mult"),
        hc_sinkhorn_iters: usize_field(config_value, "hc_sinkhorn_iters"),
        hc_eps: f32_field(config_value, "hc_eps") as f64,
        n_routed_experts: usize_field(config_value, "n_routed_experts"),
        n_shared_experts: usize_field(config_value, "n_shared_experts"),
        num_experts_per_tok: usize_field(config_value, "num_experts_per_tok"),
        scoring_func: string_field(config_value, "scoring_func"),
        norm_topk_prob: bool_field(config_value, "norm_topk_prob"),
        routed_scaling_factor: f32_field(config_value, "routed_scaling_factor") as f64,
        swiglu_limit: f32_field(config_value, "swiglu_limit") as f64,
        engram_layer_ids: usize_array(config_value, "engram_layer_ids"),
        engram_num_embeddings: usize_array(config_value, "engram_num_embeddings"),
        engram_max_ngram_size: usize_field(config_value, "engram_max_ngram_size"),
        engram_vocab_size: usize_field(config_value, "engram_vocab_size"),
        engram_n_heads: usize_field(config_value, "engram_n_heads"),
        engram_head_dim: usize_field(config_value, "engram_head_dim"),
        engram_pad_token_id: usize_field(config_value, "engram_pad_token_id"),
        engram_compressed_vocab_size: usize_field(config_value, "engram_compressed_vocab_size"),
        image_token_id: usize_field(config_value, "image_token_id"),
        dtype: string_field(config_value, "dtype"),
        expert_dtype: string_field(config_value, "expert_dtype"),
        dspark: tnsr::deepseek_v41::config::DeepSeekV41DsparkConfig {
            n_mtp_layers: usize_field(&config_value["dspark"], "n_mtp_layers"),
            dspark_block_size: usize_field(&config_value["dspark"], "dspark_block_size"),
            dspark_noise_token_id: usize_field(&config_value["dspark"], "dspark_noise_token_id"),
            dspark_target_layer_ids: usize_array(
                &config_value["dspark"],
                "dspark_target_layer_ids",
            ),
            dspark_markov_rank: usize_field(&config_value["dspark"], "dspark_markov_rank"),
            dspark_n_routed_experts: usize_field(
                &config_value["dspark"],
                "dspark_n_routed_experts",
            ),
            dspark_num_experts_per_tok: usize_field(
                &config_value["dspark"],
                "dspark_num_experts_per_tok",
            ),
        },
        vision: tnsr::deepseek_v41::config::DeepSeekV41DeferredVisionConfig {
            num_hidden_layers: usize_field(&config_value["vision"], "num_hidden_layers"),
            hidden_size: usize_field(&config_value["vision"], "hidden_size"),
            num_attention_heads: usize_field(&config_value["vision"], "num_attention_heads"),
            intermediate_size: usize_field(&config_value["vision"], "intermediate_size"),
            patch_size: usize_field(&config_value["vision"], "patch_size"),
            rope_theta: f32_field(&config_value["vision"], "rope_theta") as f64,
            downsample_ratio: usize_field(&config_value["vision"], "downsample_ratio"),
            max_image_tokens: usize_field(&config_value["vision"], "max_image_tokens"),
            min_pixels: usize_field(&config_value["vision"], "min_pixels"),
            max_wh_ratio: None,
        },
    };

    let layout = EngramLayout::from_config(&config).expect("layout should be present");
    assert_eq!(
        layout.primes,
        usize_array(&fixture["expected"], "primes_flat")
    );
    assert_eq!(layout.offsets, usize_array(&fixture["expected"], "offsets"));
}

#[test]
fn engram_hashes_match_prefill_and_decode_updates() {
    let layout_fixture = fixture("engram_layout_fixture.json");
    assert_fixture_meta(&layout_fixture, "EngramLayout.from_args");
    let layout_input = &layout_fixture["input"]["layout"];
    let layout = EngramLayout {
        max_ngram_size: usize_field(layout_input, "max_ngram_size"),
        layer_ids: usize_array(layout_input, "layer_ids"),
        num_embeddings: usize_array(layout_input, "num_embeddings"),
        primes: usize_array(layout_input, "primes_flat"),
        offsets: usize_array(layout_input, "offsets"),
        n_heads: usize_field(layout_input, "n_heads"),
        head_dim: usize_field(layout_input, "head_dim"),
    };

    let fixture = fixture("engram_hash_fixture.json");
    assert_fixture_meta(&fixture, "NgramHashState.forward");

    let vocab_entries = fixture["input"]["vocab_entries"]
        .as_array()
        .expect("vocab_entries should be array")
        .iter()
        .map(|pair| {
            let token_id = usize_field(pair, "token_id");
            let text = string_field(pair, "text");
            (token_id, text)
        })
        .collect::<Vec<_>>();
    let token_map = compressed_token_map_from_vocab_entries(&vocab_entries);
    assert_eq!(
        token_map,
        usize_array(&fixture["input"], "expected_token_map"),
        "compressed token map must match upstream normalization semantics"
    );
    let mut state = NgramHashState::new(
        token_map,
        usize_field(&fixture["input"], "pad_token_id"),
        usize_field(&fixture["input"], "max_batch_size"),
        usize_field(&fixture["input"], "max_seq_len"),
    );
    state.set_multipliers(
        fixture["input"]["multipliers"]
            .as_array()
            .expect("multipliers should be array")
            .iter()
            .map(|v| {
                v.as_i64()
                    .unwrap_or_else(|| panic!("multiplier should be integer"))
                    as i64
            })
            .collect(),
    );

    let input = &fixture["input"];
    let prefill = ngram_hashes(
        &usize_array(input, "prefill_input_ids"),
        0,
        &layout,
        &mut state,
    );
    assert_eq!(prefill, usize_array(&fixture["expected"], "prefill_hashes"));

    let decode = ngram_hashes(
        &usize_array(input, "decode_input_ids"),
        usize_field(input, "decode_start_pos"),
        &layout,
        &mut state,
    );
    assert_eq!(decode, usize_array(&fixture["expected"], "decode_hashes"));
}

#[test]
fn engram_update_matches_signed_sqrt_gate_and_token_mask() {
    let fixture = fixture("engram_update_fixture.json");
    assert_fixture_meta(&fixture, "Engram.forward");
    let tol = f32_field(&fixture, "absolute_tolerance");

    let input = &fixture["input"];
    let token_mask = if input.get("token_mask").is_some() {
        Some(bool_array(input, "token_mask"))
    } else {
        None
    };
    let got = engram_update(
        &f32_array(input, "x"),
        &f32_array(input, "key"),
        &f32_array(input, "value"),
        &f32_array(input, "q_weight"),
        &f32_array(input, "k_weight"),
        f32_field(input, "eps"),
        token_mask.as_deref(),
    );
    assert_close_slice(&got, &f32_array(&fixture["expected"], "output"), tol);
}
