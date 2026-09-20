use std::cell::RefCell;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tnsr::{
    deepseek_v41::{
        attention::{Csa2Mode, DeepSeekV41Attention, DeepSeekV41Compressor, DeepSeekV41Indexer},
        engram::DeepSeekV41Engram,
        model::{DeepSeekV41Block, DeepSeekV41DsparkHead, DeepSeekV41DsparkStage},
        moe::{DeepSeekV41Expert, DeepSeekV41Gate, DeepSeekV41MoE},
    },
    tensor::{Shape, Tensor, TensorValue},
};

fn runfile(path: &str) -> PathBuf {
    let cwd = env::current_dir().expect("current dir");
    let direct = cwd.join(path);
    if direct.exists() {
        return direct;
    }

    if let Ok(runfiles_dir) = env::var("RUNFILES_DIR") {
        let under_workspace = Path::new(&runfiles_dir).join("_main").join(path);
        if under_workspace.exists() {
            return under_workspace;
        }
        return Path::new(&runfiles_dir).join(path);
    }

    direct
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

fn param(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_value(TensorValue::from_vec(Shape(shape.to_vec()), data), true)
}

fn tensor_param(value: &Value) -> Tensor {
    param(&usize_array(value, "shape"), f32_array(value, "data"))
}

fn param_vec(value: &Value) -> Vec<f32> {
    f32_array(value, "data")
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

fn main() {
    let mut args = env::args().skip(1);
    let fixture_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
        runfile("ferric_continuum/tnsr/testdata/deepseek_v41/dspark_tiny_model_fixture.json")
    });
    let text = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|err| panic!("read fixture {}: {err}", fixture_path.display()));
    let fixture: Value = serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("parse fixture {}: {err}", fixture_path.display()));
    let input = &fixture["input"];
    let head = dspark_head_from_fixture(&fixture);
    let input_ids = usize_array(input, "input_ids");
    let main_hidden = f32_array(input, "main_hidden");
    let out = head.forward_spec(&input_ids, &main_hidden);

    println!(
        "{}",
        json!({
            "token_ids": input_ids,
            "prompt": "",
            "vocab_size": usize_field(input, "vocab_size"),
            "model_type": "deepseek_v41_dspark",
            "logits": out.logits,
            "output_ids": out.output_ids,
            "confidence": out.confidence,
        })
    );
}
