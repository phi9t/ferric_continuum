use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use tnsr::deepseek_v41::config::{DeepSeekV41ReleaseIndex, DeepSeekV41TextConfig};

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

fn hf_config_path() -> PathBuf {
    runfile("ferric_continuum/tnsr/testdata/deepseek_v41/upstream/config.json")
}

fn inference_config_path() -> PathBuf {
    runfile("ferric_continuum/tnsr/testdata/deepseek_v41/upstream/inference/config.json")
}

fn index_path() -> PathBuf {
    runfile("ferric_continuum/tnsr/testdata/deepseek_v41/upstream/model.safetensors.index.json")
}

fn assert_release_text_config(config: &DeepSeekV41TextConfig) {
    assert_eq!(config.vocab_size, 129280);
    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.moe_intermediate_size, 2304);
    assert_eq!(config.num_hidden_layers, 40);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.num_key_value_heads, 1);
    assert_eq!(config.head_dim, 512);
    assert_eq!(config.qk_rope_head_dim, 64);
    assert_eq!(config.q_lora_rank, 1280);
    assert_eq!(config.o_lora_rank, 1024);
    assert_eq!(config.o_groups, 8);
    assert_eq!(config.rms_norm_eps, 1e-20);
    assert_eq!(config.rope_theta, 10000.0);
    assert_eq!(config.rope_factor, 16.0);
    assert_eq!(config.original_seq_len, 65536);
    assert_eq!(config.beta_fast, 32.0);
    assert_eq!(config.beta_slow, 1.0);
    assert_eq!(config.sliding_window, 128);
    assert_eq!(config.compress_ratios.len(), 43);
    assert_eq!(
        config.compress_ratios,
        vec![
            0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        ]
    );
    assert_eq!(config.compress_rope_theta, 160000.0);
    assert_eq!(config.kv_source_layer_ids, vec![2, 8, 14, 20]);
    assert_eq!(
        config.index_source_layer_ids,
        vec![2, 8, 14, 20, 24, 28, 32, 36]
    );
    assert_eq!(config.index_n_heads, 32);
    assert_eq!(config.index_head_dim, 128);
    assert_eq!(config.index_topk, 512);
    assert_eq!(config.candidate_source_layer_id, 20);
    assert_eq!(config.candidate_topk_blocks, 2048);
    assert_eq!(config.candidate_block_size, 8);
    assert_eq!(config.hc_mult, 4);
    assert_eq!(config.hc_sinkhorn_iters, 20);
    assert_eq!(config.hc_eps, 1e-6);
    assert_eq!(config.n_routed_experts, 384);
    assert_eq!(config.n_shared_experts, 1);
    assert_eq!(config.num_experts_per_tok, 6);
    assert_eq!(config.scoring_func, "sqrtsoftplus");
    assert!(config.norm_topk_prob);
    assert_eq!(config.routed_scaling_factor, 1.5);
    assert_eq!(config.swiglu_limit, 10.0);
    assert_eq!(config.engram_layer_ids, vec![1, 14]);
    assert_eq!(config.engram_num_embeddings, vec![384006168, 384016682]);
    assert_eq!(config.engram_max_ngram_size, 4);
    assert_eq!(config.engram_vocab_size, 16000000);
    assert_eq!(config.engram_n_heads, 8);
    assert_eq!(config.engram_head_dim, 256);
    assert_eq!(config.engram_pad_token_id, 2);
    assert_eq!(config.engram_compressed_vocab_size, 99092);

    assert_eq!(config.dspark.n_mtp_layers, 3);
    assert_eq!(config.dspark.dspark_block_size, 5);
    assert_eq!(config.dspark.dspark_noise_token_id, 128799);
    assert_eq!(config.dspark.dspark_target_layer_ids, vec![37, 38, 39]);
    assert_eq!(config.dspark.dspark_markov_rank, 256);
    assert_eq!(config.dspark.dspark_n_routed_experts, 128);
    assert_eq!(config.dspark.dspark_num_experts_per_tok, 3);

    assert_eq!(config.vision.num_hidden_layers, 32);
    assert_eq!(config.vision.hidden_size, 1024);
    assert_eq!(config.vision.num_attention_heads, 16);
    assert_eq!(config.vision.intermediate_size, 2816);
    assert_eq!(config.vision.patch_size, 14);
    assert_eq!(config.vision.rope_theta, 10000.0);
    assert_eq!(config.vision.downsample_ratio, 3);
    assert_eq!(config.vision.max_image_tokens, 1024);
    assert_eq!(config.vision.min_pixels, 295936);
    assert!(config.vision.max_wh_ratio.is_none());
    assert!(config.vision.vision_enabled());
    assert_eq!(config.image_token_id, 129264);
    assert_eq!(config.dtype, "fp8");
    assert_eq!(config.expert_dtype, "fp4");
}

#[test]
fn parses_nested_huggingface_config_release_fields() {
    let config = DeepSeekV41TextConfig::from_hf_json(&hf_config_path()).unwrap();
    assert_release_text_config(&config);
}

#[test]
fn parses_flat_inference_config_release_fields() {
    let config = DeepSeekV41TextConfig::from_inference_json(&inference_config_path()).unwrap();
    assert_release_text_config(&config);
}

#[test]
fn hf_and_inference_configs_agree_on_text_fields() {
    let hf = DeepSeekV41TextConfig::from_hf_json(&hf_config_path()).unwrap();
    let inference = DeepSeekV41TextConfig::from_inference_json(&inference_config_path()).unwrap();

    assert_eq!(hf, inference);
}

#[test]
fn parses_release_index_without_opening_weight_shards() {
    let index = DeepSeekV41ReleaseIndex::from_json(&index_path()).unwrap();

    assert_eq!(index.total_size, 510286023000);
    assert_eq!(index.tensor_count(), 96085);
    assert_eq!(index.shard_count(), 48);

    let expected_shards: BTreeSet<_> = (1..=48)
        .map(|i| format!("model-{i:05}-of-00048.safetensors"))
        .collect();
    assert_eq!(index.shard_names(), expected_shards);

    for name in [
        "embed.weight",
        "head.weight",
        "norm.weight",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_b.weight",
        "layers.0.attn.wkv.weight",
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.indexer.wq_b.weight",
        "layers.1.engram.wkv.weight",
    ] {
        assert!(index.contains_tensor(name), "missing tensor {name}");
    }

    assert!(index.contains_tensor_with_prefix("mtp.0."));
    assert!(index.contains_tensor_with_prefix("vision."));
    assert!(!index.contains_tensor("model.layers.0.self_attn.q_proj.weight"));
}

#[test]
fn malformed_config_field_returns_error() {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "tnsr-deepseek-v41-config-test-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad-config.json");
    std::fs::write(
        &path,
        r#"{
  "vocab_size": "129280"
}"#,
    )
    .unwrap();

    let err = match DeepSeekV41TextConfig::from_inference_json(&path) {
        Ok(_) => panic!("malformed config unexpectedly parsed"),
        Err(err) => err,
    };

    assert!(err.contains("vocab_size"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn semantically_invalid_config_is_rejected_by_validate() {
    // Start from the well-typed release inference config and corrupt a single
    // structural invariant (num_experts_per_tok > n_routed_experts). The file
    // parses field-by-field but must fail `validate` with a named field.
    let text = std::fs::read_to_string(inference_config_path()).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["n_activated_experts"] = serde_json::json!(100000);

    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "tnsr-deepseek-v41-validate-test-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("invalid-config.json");
    std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();

    let err = match DeepSeekV41TextConfig::from_inference_json(&path) {
        Ok(_) => panic!("semantically invalid config unexpectedly parsed"),
        Err(err) => err,
    };
    assert!(
        err.contains("num_experts_per_tok") && err.contains("n_routed_experts"),
        "unexpected error message: {err}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}
