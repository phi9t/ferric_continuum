//! Config and safetensors-index metadata for DeepSeek V4.1 Flash.
//!
//! This module is intentionally metadata-only.  It parses the release config
//! surfaces and checkpoint index without opening safetensors shards or binding
//! the values to future model structs.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41DeferredDsparkConfig {
    pub n_mtp_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: usize,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_num_experts_per_tok: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41VisionConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f64>,
}

impl DeepSeekV41VisionConfig {
    /// Vision tower is present iff it has at least one layer (upstream
    /// `ModelArgs.vision_enabled`).
    pub fn vision_enabled(&self) -> bool {
        self.num_hidden_layers > 0
    }
}

/// Deprecated alias retained for existing call sites; use
/// [`DeepSeekV41VisionConfig`].
pub type DeepSeekV41DeferredVisionConfig = DeepSeekV41VisionConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub original_seq_len: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f64,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer_id: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<usize>,
    pub engram_max_ngram_size: usize,
    pub engram_vocab_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub engram_pad_token_id: usize,
    pub engram_compressed_vocab_size: usize,
    pub image_token_id: usize,
    pub dtype: String,
    pub expert_dtype: String,
    pub dspark: DeepSeekV41DeferredDsparkConfig,
    pub vision: DeepSeekV41VisionConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41ReleaseIndex {
    pub total_size: u64,
    weight_map: BTreeMap<String, String>,
}

impl DeepSeekV41TextConfig {
    pub fn from_hf_json(path: &Path) -> Result<Self, String> {
        let json = read_json(path, "config.json")?;
        let root = as_object(&json, "config.json")?;
        let text = object_at(root, "text_config")?;
        let rope_scaling = object_at(text, "rope_scaling")?;
        let quant = object_at(root, "quantization_config")?;
        let vision = object_at(root, "vision_config")?;

        Ok(Self {
            vocab_size: usize_field(text, "vocab_size")?,
            hidden_size: usize_field(text, "hidden_size")?,
            moe_intermediate_size: usize_field(text, "moe_intermediate_size")?,
            num_hidden_layers: usize_field(text, "num_hidden_layers")?,
            num_attention_heads: usize_field(text, "num_attention_heads")?,
            num_key_value_heads: usize_field(text, "num_key_value_heads")?,
            head_dim: usize_field(text, "head_dim")?,
            qk_rope_head_dim: usize_field(text, "qk_rope_head_dim")?,
            q_lora_rank: usize_field(text, "q_lora_rank")?,
            o_lora_rank: usize_field(text, "o_lora_rank")?,
            o_groups: usize_field(text, "o_groups")?,
            rms_norm_eps: f64_field(text, "rms_norm_eps")?,
            rope_theta: f64_field(text, "rope_theta")?,
            rope_factor: f64_field(rope_scaling, "factor")?,
            original_seq_len: usize_field(rope_scaling, "original_max_position_embeddings")?,
            beta_fast: f64_field(rope_scaling, "beta_fast")?,
            beta_slow: f64_field(rope_scaling, "beta_slow")?,
            sliding_window: usize_field(text, "sliding_window")?,
            compress_ratios: usize_vec_field(text, "compress_ratios")?,
            compress_rope_theta: f64_field(text, "compress_rope_theta")?,
            kv_source_layer_ids: usize_vec_field(text, "kv_source_layer_ids")?,
            index_source_layer_ids: usize_vec_field(text, "index_source_layer_ids")?,
            index_n_heads: usize_field(text, "index_n_heads")?,
            index_head_dim: usize_field(text, "index_head_dim")?,
            index_topk: usize_field(text, "index_topk")?,
            candidate_source_layer_id: usize_field(text, "candidate_source_layer_id")?,
            candidate_topk_blocks: usize_field(text, "candidate_topk_blocks")?,
            candidate_block_size: usize_field(text, "candidate_block_size")?,
            hc_mult: usize_field(text, "hc_mult")?,
            hc_sinkhorn_iters: usize_field(text, "hc_sinkhorn_iters")?,
            hc_eps: f64_field(text, "hc_eps")?,
            n_routed_experts: usize_field(text, "n_routed_experts")?,
            n_shared_experts: usize_field(text, "n_shared_experts")?,
            num_experts_per_tok: usize_field(text, "num_experts_per_tok")?,
            scoring_func: string_field(text, "scoring_func")?,
            norm_topk_prob: bool_field(text, "norm_topk_prob")?,
            routed_scaling_factor: f64_field(text, "routed_scaling_factor")?,
            swiglu_limit: f64_field(text, "swiglu_limit")?,
            engram_layer_ids: usize_vec_field(text, "engram_layer_ids")?,
            engram_num_embeddings: usize_vec_field(text, "engram_num_embeddings")?,
            engram_max_ngram_size: usize_field(text, "engram_max_ngram_size")?,
            engram_vocab_size: usize_field(text, "engram_vocab_size")?,
            engram_n_heads: usize_field(text, "engram_n_heads")?,
            engram_head_dim: usize_field(text, "engram_head_dim")?,
            engram_pad_token_id: usize_field(text, "engram_pad_token_id")?,
            engram_compressed_vocab_size: usize_field(text, "engram_compressed_vocab_size")?,
            image_token_id: usize_field(root, "image_token_id")?,
            dtype: string_field(quant, "quant_method")?,
            expert_dtype: string_field(quant, "expert_dtype")?,
            dspark: DeepSeekV41DeferredDsparkConfig {
                n_mtp_layers: usize_field(text, "num_nextn_predict_layers")?,
                dspark_block_size: usize_field(text, "dspark_block_size")?,
                dspark_noise_token_id: usize_field(text, "dspark_noise_token_id")?,
                dspark_target_layer_ids: usize_vec_field(text, "dspark_target_layer_ids")?,
                dspark_markov_rank: usize_field(text, "dspark_markov_rank")?,
                dspark_n_routed_experts: usize_field(text, "dspark_n_routed_experts")?,
                dspark_num_experts_per_tok: usize_field(text, "dspark_num_experts_per_tok")?,
            },
            vision: DeepSeekV41VisionConfig {
                num_hidden_layers: usize_field(vision, "num_hidden_layers")?,
                hidden_size: usize_field(vision, "hidden_size")?,
                num_attention_heads: usize_field(vision, "num_attention_heads")?,
                intermediate_size: usize_field(vision, "intermediate_size")?,
                patch_size: usize_field(vision, "patch_size")?,
                rope_theta: f64_field(vision, "rope_theta")?,
                downsample_ratio: usize_field(vision, "downsample_ratio")?,
                max_image_tokens: usize_field(vision, "max_image_tokens")?,
                min_pixels: usize_field(vision, "min_pixels")?,
                max_wh_ratio: optional_f64_field(vision, "max_wh_ratio")?,
            },
        })
    }

    pub fn from_inference_json(path: &Path) -> Result<Self, String> {
        let json = read_json(path, "inference/config.json")?;
        let root = as_object(&json, "inference/config.json")?;

        Ok(Self {
            vocab_size: usize_field(root, "vocab_size")?,
            hidden_size: usize_field(root, "dim")?,
            moe_intermediate_size: usize_field(root, "moe_inter_dim")?,
            num_hidden_layers: usize_field(root, "n_layers")?,
            num_attention_heads: usize_field(root, "n_heads")?,
            num_key_value_heads: 1,
            head_dim: usize_field(root, "head_dim")?,
            qk_rope_head_dim: usize_field(root, "rope_head_dim")?,
            q_lora_rank: usize_field(root, "q_lora_rank")?,
            o_lora_rank: usize_field(root, "o_lora_rank")?,
            o_groups: usize_field(root, "o_groups")?,
            rms_norm_eps: f64_field(root, "norm_eps")?,
            rope_theta: f64_field(root, "rope_theta")?,
            rope_factor: f64_field(root, "rope_factor")?,
            original_seq_len: usize_field(root, "original_seq_len")?,
            beta_fast: f64_field(root, "beta_fast")?,
            beta_slow: f64_field(root, "beta_slow")?,
            sliding_window: usize_field(root, "window_size")?,
            compress_ratios: usize_vec_field(root, "compress_ratios")?,
            compress_rope_theta: f64_field(root, "compress_rope_theta")?,
            kv_source_layer_ids: usize_vec_field(root, "kv_source_layers")?,
            index_source_layer_ids: usize_vec_field(root, "index_source_layers")?,
            index_n_heads: usize_field(root, "index_n_heads")?,
            index_head_dim: usize_field(root, "index_head_dim")?,
            index_topk: usize_field(root, "index_topk")?,
            candidate_source_layer_id: usize_field(root, "candidate_source_layer")?,
            candidate_topk_blocks: usize_field(root, "candidate_topk_blocks")?,
            candidate_block_size: usize_field(root, "candidate_block_size")?,
            hc_mult: usize_field(root, "hc_mult")?,
            hc_sinkhorn_iters: usize_field(root, "hc_sinkhorn_iters")?,
            hc_eps: f64_field(root, "hc_eps")?,
            n_routed_experts: usize_field(root, "n_routed_experts")?,
            n_shared_experts: usize_field(root, "n_shared_experts")?,
            num_experts_per_tok: usize_field(root, "n_activated_experts")?,
            scoring_func: string_field(root, "score_func")?,
            norm_topk_prob: true,
            routed_scaling_factor: f64_field(root, "route_scale")?,
            swiglu_limit: f64_field(root, "swiglu_limit")?,
            engram_layer_ids: usize_vec_field(root, "engram_layer_ids")?,
            engram_num_embeddings: usize_vec_field(root, "engram_num_embeddings")?,
            engram_max_ngram_size: usize_field(root, "engram_max_ngram_size")?,
            engram_vocab_size: usize_field(root, "engram_vocab_size")?,
            engram_n_heads: usize_field(root, "engram_n_heads")?,
            engram_head_dim: usize_field(root, "engram_head_dim")?,
            engram_pad_token_id: usize_field(root, "engram_pad_id")?,
            engram_compressed_vocab_size: usize_field(root, "engram_compressed_vocab_size")?,
            image_token_id: usize_field(root, "image_token_id")?,
            dtype: string_field(root, "dtype")?,
            expert_dtype: string_field(root, "expert_dtype")?,
            dspark: DeepSeekV41DeferredDsparkConfig {
                n_mtp_layers: usize_field(root, "n_mtp_layers")?,
                dspark_block_size: usize_field(root, "dspark_block_size")?,
                dspark_noise_token_id: usize_field(root, "dspark_noise_token_id")?,
                dspark_target_layer_ids: usize_vec_field(root, "dspark_target_layer_ids")?,
                dspark_markov_rank: usize_field(root, "dspark_markov_rank")?,
                dspark_n_routed_experts: usize_field(root, "dspark_n_routed_experts")?,
                dspark_num_experts_per_tok: usize_field(root, "dspark_n_activated_experts")?,
            },
            vision: DeepSeekV41VisionConfig {
                num_hidden_layers: usize_field(root, "vision_n_layers")?,
                hidden_size: usize_field(root, "vision_dim")?,
                num_attention_heads: usize_field(root, "vision_n_heads")?,
                intermediate_size: usize_field(root, "vision_inter_dim")?,
                patch_size: usize_field(root, "vision_patch_size")?,
                rope_theta: f64_field(root, "vision_rope_theta")?,
                downsample_ratio: usize_field(root, "vision_downsample_ratio")?,
                max_image_tokens: usize_field(root, "vision_max_n_token")?,
                min_pixels: usize_field(root, "vision_min_pixels")?,
                max_wh_ratio: optional_f64_field(root, "vision_max_wh_ratio")?,
            },
        })
    }
}

impl DeepSeekV41ReleaseIndex {
    pub fn from_json(path: &Path) -> Result<Self, String> {
        let json = read_json(path, "model.safetensors.index.json")?;
        let root = as_object(&json, "model.safetensors.index.json")?;
        let metadata = object_at(root, "metadata")?;
        let total_size = metadata
            .get("total_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                "model.safetensors.index.json missing u64 field `metadata.total_size`".to_string()
            })?;
        let raw_weight_map = object_at(root, "weight_map")?;
        let mut weight_map = BTreeMap::new();
        for (name, shard) in raw_weight_map {
            let shard = shard.as_str().ok_or_else(|| {
                format!("model.safetensors.index.json field `weight_map.{name}` is not a string")
            })?;
            weight_map.insert(name.clone(), shard.to_string());
        }

        Ok(Self {
            total_size,
            weight_map,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.shard_names().len()
    }

    pub fn shard_names(&self) -> BTreeSet<String> {
        self.weight_map.values().cloned().collect()
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn contains_tensor_with_prefix(&self, prefix: &str) -> bool {
        self.weight_map.keys().any(|name| name.starts_with(prefix))
    }
}

fn read_json(path: &Path, label: &str) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {label}: {e}"))
}

fn as_object<'a>(
    json: &'a Value,
    label: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    json.as_object()
        .ok_or_else(|| format!("{label} root is not a JSON object"))
}

fn object_at<'a>(
    json: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    json.get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("config missing object field `{key}`"))
}

fn usize_field(json: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    json.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("config missing usize field `{key}`"))
}

fn f64_field(json: &serde_json::Map<String, Value>, key: &str) -> Result<f64, String> {
    json.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("config missing float field `{key}`"))
}

fn optional_f64_field(
    json: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<f64>, String> {
    match json.get(key) {
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("config field `{key}` is not null or float")),
        None => Err(format!("config missing optional float field `{key}`")),
    }
}

fn bool_field(json: &serde_json::Map<String, Value>, key: &str) -> Result<bool, String> {
    json.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("config missing bool field `{key}`"))
}

fn string_field(json: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    json.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("config missing string field `{key}`"))
}

fn usize_vec_field(json: &serde_json::Map<String, Value>, key: &str) -> Result<Vec<usize>, String> {
    let values = json
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("config missing usize array field `{key}`"))?;
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            value
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("config field `{key}` has non-usize element at index {idx}"))
        })
        .collect()
}
