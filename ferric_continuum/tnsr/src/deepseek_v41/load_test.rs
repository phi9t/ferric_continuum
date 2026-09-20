// Tests for the DeepSeek V4.1 text-only checkpoint loader.
//
// Included from `load.rs` via `include!` so they share the module's private
// decode helpers.  Tiny safetensors fixtures are written inside each test
// following the `qwen3_load.rs` pattern.

use super::*;
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// A minimal in-test safetensors tensor of an arbitrary dtype.
struct FixtureTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl FixtureTensor {
    fn f32(shape: &[usize], data: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            bytes,
        }
    }

    fn f32_zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product::<usize>();
        Self::f32(shape, &vec![0.0f32; n])
    }

    fn bytes(dtype: Dtype, shape: &[usize], bytes: Vec<u8>) -> Self {
        Self {
            dtype,
            shape: shape.to_vec(),
            bytes,
        }
    }
}

impl View for FixtureTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

fn unique_tmp_dir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("tnsr-dsv41-load-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_safetensors(dir: &Path, file: &str, tensors: Vec<(String, FixtureTensor)>) {
    let bytes = safetensors::tensor::serialize(tensors, &None).unwrap();
    std::fs::write(dir.join(file), bytes).unwrap();
}

/// A tiny converted-style config the loader can parse via `from_inference_json`.
/// dim=4, 1 layer (sliding-window only), hc_mult=2, 2 experts, topk=1.
const TINY_CONFIG_JSON: &str = r#"{
  "vocab_size": 5,
  "dim": 4,
  "moe_inter_dim": 3,
  "n_layers": 1,
  "n_heads": 2,
  "head_dim": 2,
  "rope_head_dim": 1,
  "q_lora_rank": 3,
  "o_lora_rank": 2,
  "o_groups": 2,
  "norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "rope_factor": 1.0,
  "original_seq_len": 0,
  "beta_fast": 32.0,
  "beta_slow": 1.0,
  "window_size": 4,
  "compress_ratios": [0],
  "compress_rope_theta": 40000.0,
  "kv_source_layers": [],
  "index_source_layers": [],
  "index_n_heads": 2,
  "index_head_dim": 2,
  "index_topk": 1,
  "candidate_source_layer": 0,
  "candidate_topk_blocks": 1,
  "candidate_block_size": 1,
  "hc_mult": 2,
  "hc_sinkhorn_iters": 2,
  "hc_eps": 1e-6,
  "n_routed_experts": 2,
  "n_shared_experts": 1,
  "n_activated_experts": 1,
  "score_func": "sqrtsoftplus",
  "route_scale": 1.0,
  "swiglu_limit": 0.0,
  "engram_layer_ids": [],
  "engram_num_embeddings": [],
  "engram_max_ngram_size": 3,
  "engram_vocab_size": 5,
  "engram_n_heads": 2,
  "engram_head_dim": 2,
  "engram_pad_id": 2,
  "engram_compressed_vocab_size": 5,
  "image_token_id": 4,
  "dtype": "bf16",
  "expert_dtype": "bf16",
  "n_mtp_layers": 0,
  "dspark_block_size": 0,
  "dspark_noise_token_id": 0,
  "dspark_target_layer_ids": [],
  "dspark_markov_rank": 0,
  "dspark_n_routed_experts": 0,
  "dspark_n_activated_experts": 0,
  "vision_n_layers": 0,
  "vision_dim": 0,
  "vision_n_heads": 0,
  "vision_inter_dim": 0,
  "vision_patch_size": 0,
  "vision_rope_theta": 0.0,
  "vision_downsample_ratio": 0,
  "vision_max_n_token": 0,
  "vision_min_pixels": 0,
  "vision_max_wh_ratio": null
}"#;

/// The full set of required f32 tensors for the tiny 1-layer text model.
/// dim=4, n_heads=2, head_dim=2, q_lora=3, o_lora=2, o_groups=2, inter=3,
/// experts=2, hc_mult=2 => mix_hc=(2+2)*2=8, hc_dim=8.
fn tiny_tensor_map() -> Vec<(String, FixtureTensor)> {
    let mut t: Vec<(String, FixtureTensor)> = Vec::new();
    // globals
    t.push(("embed.weight".into(), FixtureTensor::f32_zeros(&[5, 4])));
    t.push(("head.weight".into(), FixtureTensor::f32_zeros(&[5, 4])));
    t.push(("norm.weight".into(), FixtureTensor::f32_zeros(&[4])));
    // layer 0
    t.push((
        "layers.0.attn_norm.weight".into(),
        FixtureTensor::f32_zeros(&[4]),
    ));
    t.push((
        "layers.0.ffn_norm.weight".into(),
        FixtureTensor::f32_zeros(&[4]),
    ));
    // attention: upstream [out,in]
    t.push((
        "layers.0.attn.wq_a.weight".into(),
        FixtureTensor::f32_zeros(&[3, 4]),
    )); // [q_lora, dim]
    t.push((
        "layers.0.attn.q_norm.weight".into(),
        FixtureTensor::f32_zeros(&[3]),
    ));
    t.push((
        "layers.0.attn.wq_b.weight".into(),
        FixtureTensor::f32_zeros(&[4, 3]),
    )); // [n_heads*head_dim, q_lora]
    t.push((
        "layers.0.attn.wkv.weight".into(),
        FixtureTensor::f32_zeros(&[2, 4]),
    )); // [head_dim, dim]
    t.push((
        "layers.0.attn.kv_norm.weight".into(),
        FixtureTensor::f32_zeros(&[2]),
    ));
    // wo_a [n_groups*o_lora, group_in] = [2*2, 2] = [4,2]
    t.push((
        "layers.0.attn.wo_a.weight".into(),
        FixtureTensor::f32_zeros(&[4, 2]),
    ));
    t.push((
        "layers.0.attn.wo_b.weight".into(),
        FixtureTensor::f32_zeros(&[4, 4]),
    )); // [dim, o_groups*o_lora]
    t.push((
        "layers.0.attn.attn_sink".into(),
        FixtureTensor::f32_zeros(&[2]),
    ));
    // moe
    t.push((
        "layers.0.ffn.gate.weight".into(),
        FixtureTensor::f32_zeros(&[2, 4]),
    )); // [experts, dim]
    t.push((
        "layers.0.ffn.gate.bias".into(),
        FixtureTensor::f32_zeros(&[2]),
    ));
    for e in 0..2 {
        t.push((
            format!("layers.0.ffn.experts.{e}.w1.weight"),
            FixtureTensor::f32_zeros(&[3, 4]),
        )); // [inter, dim]
        t.push((
            format!("layers.0.ffn.experts.{e}.w2.weight"),
            FixtureTensor::f32_zeros(&[4, 3]),
        )); // [dim, inter]
        t.push((
            format!("layers.0.ffn.experts.{e}.w3.weight"),
            FixtureTensor::f32_zeros(&[3, 4]),
        ));
    }
    t.push((
        "layers.0.ffn.shared_experts.w1.weight".into(),
        FixtureTensor::f32_zeros(&[3, 4]),
    ));
    t.push((
        "layers.0.ffn.shared_experts.w2.weight".into(),
        FixtureTensor::f32_zeros(&[4, 3]),
    ));
    t.push((
        "layers.0.ffn.shared_experts.w3.weight".into(),
        FixtureTensor::f32_zeros(&[3, 4]),
    ));
    // hc: mix_hc=8, hc_dim=8
    t.push((
        "layers.0.hc_attn_fn".into(),
        FixtureTensor::f32_zeros(&[8, 8]),
    ));
    t.push((
        "layers.0.hc_attn_base".into(),
        FixtureTensor::f32_zeros(&[8]),
    ));
    t.push((
        "layers.0.hc_attn_scale".into(),
        FixtureTensor::f32_zeros(&[3]),
    ));
    t.push((
        "layers.0.hc_ffn_fn".into(),
        FixtureTensor::f32_zeros(&[8, 8]),
    ));
    t.push((
        "layers.0.hc_ffn_base".into(),
        FixtureTensor::f32_zeros(&[8]),
    ));
    t.push((
        "layers.0.hc_ffn_scale".into(),
        FixtureTensor::f32_zeros(&[3]),
    ));
    t
}

fn replace(tensors: &mut [(String, FixtureTensor)], name: &str, t: FixtureTensor) {
    let slot = tensors
        .iter_mut()
        .find(|(k, _)| k == name)
        .unwrap_or_else(|| panic!("no fixture tensor {name}"));
    slot.1 = t;
}

fn remove(tensors: &mut Vec<(String, FixtureTensor)>, name: &str) {
    tensors.retain(|(k, _)| k != name);
}

#[test]
fn e4m3_decode_matches_known_values() {
    // 0x00 -> +0; exp=7 (0b0111) mant=0 -> 1.0; sign bit -> -1.0; exp=8 mant=0 -> 2.0.
    assert_eq!(e4m3_to_f32(0x00), 0.0);
    assert_eq!(e4m3_to_f32(0b0_0111_000), 1.0);
    assert_eq!(e4m3_to_f32(0b1_0111_000), -1.0);
    assert_eq!(e4m3_to_f32(0b0_1000_000), 2.0);
    // exp=7 mant=4 (0.5) -> 1.5
    assert_eq!(e4m3_to_f32(0b0_0111_100), 1.5);
}

#[test]
fn e8m0_decode_matches_known_values() {
    assert_eq!(e8m0_to_f32(127), 1.0);
    assert_eq!(e8m0_to_f32(128), 2.0);
    assert_eq!(e8m0_to_f32(126), 0.5);
}

#[test]
fn fp8_dequant_matches_known_f32() {
    // 2x2 fp8 weight, block size 32 => one scale block [1,1].
    // bytes: 1.0, 2.0, -1.0, 1.5 ; scale exponent 128 => *2 => 2,4,-2,3.
    let w = FixtureTensor::bytes(
        Dtype::F8_E4M3,
        &[2, 2],
        vec![0b0_0111_000, 0b0_1000_000, 0b1_0111_000, 0b0_0111_100],
    );
    let s = FixtureTensor::bytes(Dtype::U8, &[1, 1], vec![128]);
    let dir = unique_tmp_dir("fp8");
    write_safetensors(
        &dir,
        "m.safetensors",
        vec![("w".into(), w), ("w.scale".into(), s)],
    );
    let ckpt = Checkpoint::open_single(&dir.join("m.safetensors")).unwrap();
    let (out, kind) = ckpt.linear_weight("w", 2, 2).unwrap();
    assert_eq!(kind, QuantKind::Fp8E4m3);
    assert_eq!(out, vec![2.0, 4.0, -2.0, 3.0]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn fp4_dequant_matches_known_f32() {
    // out=2, in=4 => packed [2, 2]. byte low nibble first.
    // row0: byte0 = (high=1 -> 0.5)<<4 | (low=2 -> 1.0) ; byte1 = (high=3 -> 1.5)<<4 | (low=4 -> 2.0)
    // logical row0 = [1.0, 0.5, 2.0, 1.5]
    // row1: byte0 = (high=9 -> -0.5)<<4 | (low=8 -> 0.0) ; byte1 = (high=0xA -> -1.0)<<4 | (low=0 -> 0.0)
    // logical row1 = [0.0, -0.5, 0.0, -1.0]
    let b = |low: u8, high: u8| (high << 4) | (low & 0x0F);
    let w = FixtureTensor::bytes(
        Dtype::I8,
        &[2, 2],
        vec![b(2, 1), b(4, 3), b(8, 9), b(0, 0x0A)],
    );
    // scale [ceil(2/32), ceil(4/32)] = [1,1], exponent 127 => *1.
    let s = FixtureTensor::bytes(Dtype::U8, &[1, 1], vec![127]);
    let dir = unique_tmp_dir("fp4");
    write_safetensors(
        &dir,
        "m.safetensors",
        vec![("w".into(), w), ("w.scale".into(), s)],
    );
    let ckpt = Checkpoint::open_single(&dir.join("m.safetensors")).unwrap();
    let (out, kind) = ckpt.linear_weight("w", 2, 4).unwrap();
    assert_eq!(kind, QuantKind::Fp4E2m1);
    assert_eq!(out, vec![1.0, 0.5, 2.0, 1.5, 0.0, -0.5, 0.0, -1.0]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tiny_converted_checkpoint_loads() {
    let dir = unique_tmp_dir("tiny-ok");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    write_safetensors(&dir, "model.safetensors", tiny_tensor_map());
    let mut model = load_text_model(&dir).expect("tiny checkpoint should load");
    assert_eq!(model.vocab_size, 5);
    assert_eq!(model.hidden_size, 4);
    assert_eq!(model.hc_mult, 2);
    assert_eq!(model.layers.len(), 1);
    assert_eq!(model.encoder_decoder_split(), (1, 0));
    // A forward over in-vocab ids should produce [b, s, vocab] logits.  The
    // seed helper wires the per-batch token count into each MoE gate first.
    let out = forward_with_token_seed(&mut model, &[0, 1], 1, 2).expect("forward");
    assert_eq!(out.inner.borrow().value.shape.0, vec![1, 2, 5]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn wrong_shape_returns_err_naming_tensor() {
    let dir = unique_tmp_dir("bad-shape");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    let mut tensors = tiny_tensor_map();
    // wq_a expected [3,4]; give [3,5].
    replace(
        &mut tensors,
        "layers.0.attn.wq_a.weight",
        FixtureTensor::f32_zeros(&[3, 5]),
    );
    write_safetensors(&dir, "model.safetensors", tensors);
    let err = match load_text_model(&dir) {
        Ok(_) => panic!("expected load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("layers.0.attn.wq_a.weight"), "err: {err}");
    assert!(
        err.contains("[3, 4]") && err.contains("[3, 5]"),
        "err: {err}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_required_tensor_returns_err() {
    let dir = unique_tmp_dir("missing");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    let mut tensors = tiny_tensor_map();
    remove(&mut tensors, "layers.0.attn.kv_norm.weight");
    write_safetensors(&dir, "model.safetensors", tensors);
    let err = match load_text_model(&dir) {
        Ok(_) => panic!("expected load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("layers.0.attn.kv_norm.weight"), "err: {err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn qwen_style_names_are_rejected() {
    let dir = unique_tmp_dir("qwen");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    let mut tensors = tiny_tensor_map();
    // Inject a Qwen marker.
    tensors.push((
        "model.embed_tokens.weight".into(),
        FixtureTensor::f32_zeros(&[5, 4]),
    ));
    write_safetensors(&dir, "model.safetensors", tensors);
    let err = match load_text_model(&dir) {
        Ok(_) => panic!("expected load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("Qwen3"), "err: {err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn vision_and_mtp_tensors_ignored_in_text_mode() {
    let dir = unique_tmp_dir("vision-mtp");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    let mut tensors = tiny_tensor_map();
    // Deferred surfaces present in the file must not break a text-only load.
    tensors.push((
        "vision.blocks.0.attn.qkv.weight".into(),
        FixtureTensor::f32_zeros(&[2, 2]),
    ));
    tensors.push((
        "aligner.w1.weight".into(),
        FixtureTensor::f32_zeros(&[2, 2]),
    ));
    tensors.push((
        "mtp.0.attn.wq_a.weight".into(),
        FixtureTensor::f32_zeros(&[3, 4]),
    ));
    write_safetensors(&dir, "model.safetensors", tensors);
    let model = load_text_model(&dir).expect("text-only load should ignore vision/mtp");
    assert_eq!(model.layers.len(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_engram_tensor_on_engram_layer_is_error() {
    // Build a 2-layer config where layer 1 is an engram layer, then omit the
    // required engram tensor so the loader must error.
    let cfg = TINY_CONFIG_JSON
        .replace("\"n_layers\": 1", "\"n_layers\": 2")
        .replace("\"compress_ratios\": [0]", "\"compress_ratios\": [0, 0]")
        .replace("\"engram_layer_ids\": []", "\"engram_layer_ids\": [1]")
        .replace(
            "\"engram_num_embeddings\": []",
            "\"engram_num_embeddings\": [4]",
        );
    let dir = unique_tmp_dir("engram-missing");
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    let mut tensors = tiny_tensor_map();
    // Duplicate layer-0 tensors as layer-1 (rename), so layer 1 is otherwise complete.
    let layer1: Vec<(String, FixtureTensor)> = tiny_tensor_map()
        .into_iter()
        .filter(|(k, _)| k.starts_with("layers.0."))
        .map(|(k, v)| (k.replacen("layers.0.", "layers.1.", 1), v))
        .collect();
    tensors.extend(layer1);
    // Add engram q_weight/k_weight but NOT engram.wkv.weight -> must error.
    tensors.push((
        "layers.1.engram.q_weight".into(),
        FixtureTensor::f32_zeros(&[2, 4]),
    ));
    tensors.push((
        "layers.1.engram.k_weight".into(),
        FixtureTensor::f32_zeros(&[2, 4]),
    ));
    write_safetensors(&dir, "model.safetensors", tensors);
    let err = match load_text_model(&dir) {
        Ok(_) => panic!("expected load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("layers.1.engram.embed.weight"), "err: {err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn engram_tensors_load_into_runtime_representation() {
    let cfg = TINY_CONFIG_JSON
        .replace("\"n_layers\": 1", "\"n_layers\": 2")
        .replace("\"compress_ratios\": [0]", "\"compress_ratios\": [0, 0]")
        .replace("\"engram_layer_ids\": []", "\"engram_layer_ids\": [1]")
        .replace(
            "\"engram_num_embeddings\": []",
            "\"engram_num_embeddings\": [4]",
        );
    let dir = unique_tmp_dir("engram-loaded");
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    let mut tensors = tiny_tensor_map();
    let layer1: Vec<(String, FixtureTensor)> = tiny_tensor_map()
        .into_iter()
        .filter(|(k, _)| k.starts_with("layers.0."))
        .map(|(k, v)| (k.replacen("layers.0.", "layers.1.", 1), v))
        .collect();
    tensors.extend(layer1);
    tensors.push((
        "layers.1.engram.q_weight".into(),
        FixtureTensor::f32_zeros(&[2, 4]),
    ));
    tensors.push((
        "layers.1.engram.k_weight".into(),
        FixtureTensor::f32_zeros(&[2, 4]),
    ));
    tensors.push((
        "layers.1.engram.embed.weight".into(),
        FixtureTensor::f32_zeros(&[4, 2]),
    ));
    // n_hash_cols=(max_ngram_size - 1) * n_heads = 4, head_dim=2,
    // out=dim*(hc+1)=12, in=8, stored as torch [out, in].
    tensors.push((
        "layers.1.engram.wkv.weight".into(),
        FixtureTensor::f32_zeros(&[12, 8]),
    ));
    write_safetensors(&dir, "model.safetensors", tensors);

    let model = load_text_model(&dir).expect("engram-enabled checkpoint should load");
    let engram = model.layers[1].engram.as_ref().expect("engram loaded");
    assert_eq!(
        engram
            .embed_weight
            .as_ref()
            .unwrap()
            .inner
            .borrow()
            .value
            .shape
            .0,
        vec![4, 2]
    );
    assert_eq!(
        engram
            .wkv_weight
            .as_ref()
            .unwrap()
            .inner
            .borrow()
            .value
            .shape
            .0,
        vec![8, 12]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn converted_tp_shard_loads() {
    let dir = unique_tmp_dir("tp");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    write_safetensors(&dir, "model0-mp1.safetensors", tiny_tensor_map());
    let model = load_text_model_from_converted_tp(&dir, 0, 1).expect("tp shard should load");
    assert_eq!(model.layers.len(), 1);
    // load_text_model auto-discovers the mp shard when no single-file exists.
    let model2 = load_text_model(&dir).expect("auto-discovery of mp shard");
    assert_eq!(model2.layers.len(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hf_indexed_checkpoint_loads_public_shard_names() {
    let dir = unique_tmp_dir("hf-index");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    let mut shard0 = Vec::new();
    let mut shard1 = Vec::new();
    let mut weight_map_entries = Vec::new();
    for (index, (name, tensor)) in tiny_tensor_map().into_iter().enumerate() {
        let shard_name = if index % 2 == 0 {
            shard0.push((name.clone(), tensor));
            "model-00001-of-00002.safetensors"
        } else {
            shard1.push((name.clone(), tensor));
            "model-00002-of-00002.safetensors"
        };
        weight_map_entries.push(format!("    \"{name}\": \"{shard_name}\""));
    }
    write_safetensors(&dir, "model-00001-of-00002.safetensors", shard0);
    write_safetensors(&dir, "model-00002-of-00002.safetensors", shard1);
    let weight_map = weight_map_entries.join(",\n");
    let index_json = format!(
        r#"{{
  "metadata": {{"total_size": 1}},
  "weight_map": {{
{weight_map}
  }}
}}"#
    );
    std::fs::write(dir.join("model.safetensors.index.json"), index_json).unwrap();

    let model = load_text_model(&dir).expect("hf-indexed checkpoint should load");
    assert_eq!(model.layers.len(), 1);
    assert_eq!(model.vocab_size, 5);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hf_indexed_checkpoint_rejects_unsafe_or_missing_shards() {
    let dir = unique_tmp_dir("hf-index-bad");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
  "metadata": {"total_size": 1},
  "weight_map": {
    "embed.weight": "../outside.safetensors"
  }
}"#,
    )
    .unwrap();
    let err = match load_text_model(&dir) {
        Ok(_) => panic!("expected unsafe shard path to fail"),
        Err(e) => e,
    };
    assert!(
        err.contains("shard path must stay within model dir"),
        "err: {err}"
    );

    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
  "metadata": {"total_size": 1},
  "weight_map": {
    "embed.weight": "model-00001-of-00048.safetensors"
  }
}"#,
    )
    .unwrap();
    let err = match load_text_model(&dir) {
        Ok(_) => panic!("expected missing indexed shard to fail"),
        Err(e) => e,
    };
    assert!(
        err.contains("indexed safetensors shard not found"),
        "err: {err}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn wo_a_grouped_layout_transposes_inner_axes() {
    // group_in=(2/2)*2=2, o_lora=2, o_groups=2 -> upstream [4,2], tnsr [2,2,2].
    // upstream row-major [g*o_lora+r, i]:
    //   g0: r0=[a00,a01], r1=[a10,a11]
    //   g1: r0=[b00,b01], r1=[b10,b11]
    // tnsr dst[(g*group_in+i)*o_lora+r] expects [g,i,r].
    let data = vec![
        1.0, 2.0, // g0 r0 (i0,i1)
        3.0, 4.0, // g0 r1
        5.0, 6.0, // g1 r0
        7.0, 8.0, // g1 r1
    ];
    let w = FixtureTensor::f32(&[4, 2], &data);
    let dir = unique_tmp_dir("wo-a");
    write_safetensors(&dir, "m.safetensors", vec![("wo_a".into(), w)]);
    let ckpt = Checkpoint::open_single(&dir.join("m.safetensors")).unwrap();
    let t = load_wo_a(&ckpt, "wo_a", 2, 2, 2, 2).unwrap();
    let v = t.inner.borrow().value.clone();
    assert_eq!(v.shape.0, vec![2, 2, 2]);
    // g0: i0 r0=1(a00), i0 r1=3(a10), i1 r0=2(a01), i1 r1=4(a11)
    // g1: i0 r0=5, i0 r1=7, i1 r0=6, i1 r1=8
    assert_eq!(
        v.data.as_ref(),
        &vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

// -- W2-06: vision checkpoint loader ---------------------------------------

/// A vision-enabled variant of `TINY_CONFIG_JSON`. Reuses the text geometry
/// (dim=4) and adds a tiny 1-layer ViT: vision_dim=8, n_heads=2 (head_dim=4,
/// rope_dim=2, which `vision_cos_sin` requires to be even), inter=3,
/// patch_size=1 (patch_flat=3), downsample_ratio=1, llm_dim=dim=4.
fn tiny_vision_config_json() -> String {
    TINY_CONFIG_JSON
        .replace("\"vision_n_layers\": 0", "\"vision_n_layers\": 1")
        .replace("\"vision_dim\": 0", "\"vision_dim\": 8")
        .replace("\"vision_n_heads\": 0", "\"vision_n_heads\": 2")
        .replace("\"vision_inter_dim\": 0", "\"vision_inter_dim\": 3")
        .replace("\"vision_patch_size\": 0", "\"vision_patch_size\": 1")
        .replace(
            "\"vision_rope_theta\": 0.0",
            "\"vision_rope_theta\": 10000.0",
        )
        .replace(
            "\"vision_downsample_ratio\": 0",
            "\"vision_downsample_ratio\": 1",
        )
        .replace("\"vision_max_n_token\": 0", "\"vision_max_n_token\": 16")
        .replace("\"vision_min_pixels\": 0", "\"vision_min_pixels\": 1")
}

/// Append the vision tower + aligner + image delimiter + per-layer `bias_vl`
/// tensors to a base tiny tensor map. vision_dim=8, n_heads=2, inter=3,
/// patch_flat=3, downsample_ratio=1 => aligner in_dim = 8*1*1 = 8, llm_dim=4.
fn add_vision_tensors(t: &mut Vec<(String, FixtureTensor)>) {
    let vd = 8;
    let inter = 3;
    let patch_flat = 3;
    let llm_dim = 4;
    let in_dim = vd * 1 * 1;
    // patch embed: proj.weight [vision_dim, patch_flat], proj.bias [vision_dim]
    t.push((
        "vision.patch_embed.proj.weight".into(),
        FixtureTensor::f32_zeros(&[vd, patch_flat]),
    ));
    t.push((
        "vision.patch_embed.proj.bias".into(),
        FixtureTensor::f32_zeros(&[vd]),
    ));
    // one block
    t.push((
        "vision.blocks.0.norm1.weight".into(),
        FixtureTensor::f32_zeros(&[vd]),
    ));
    t.push((
        "vision.blocks.0.attn.wqkv.weight".into(),
        FixtureTensor::f32_zeros(&[3 * vd, vd]),
    ));
    t.push((
        "vision.blocks.0.attn.wqkv.bias".into(),
        FixtureTensor::f32_zeros(&[3 * vd]),
    ));
    t.push((
        "vision.blocks.0.attn.wo.weight".into(),
        FixtureTensor::f32_zeros(&[vd, vd]),
    ));
    t.push((
        "vision.blocks.0.attn.wo.bias".into(),
        FixtureTensor::f32_zeros(&[vd]),
    ));
    t.push((
        "vision.blocks.0.norm2.weight".into(),
        FixtureTensor::f32_zeros(&[vd]),
    ));
    t.push((
        "vision.blocks.0.mlp.w1.weight".into(),
        FixtureTensor::f32_zeros(&[2 * inter, vd]),
    ));
    t.push((
        "vision.blocks.0.mlp.w2.weight".into(),
        FixtureTensor::f32_zeros(&[vd, inter]),
    ));
    t.push(("vision.norm.weight".into(), FixtureTensor::f32_zeros(&[vd])));
    // aligner
    t.push((
        "aligner.w1.weight".into(),
        FixtureTensor::f32_zeros(&[llm_dim, in_dim]),
    ));
    t.push((
        "aligner.w1.bias".into(),
        FixtureTensor::f32_zeros(&[llm_dim]),
    ));
    t.push((
        "aligner.w2.weight".into(),
        FixtureTensor::f32_zeros(&[llm_dim, llm_dim]),
    ));
    t.push((
        "aligner.w2.bias".into(),
        FixtureTensor::f32_zeros(&[llm_dim]),
    ));
    // image delimiters [dim]
    t.push(("image_start".into(), FixtureTensor::f32_zeros(&[4])));
    t.push(("image_end".into(), FixtureTensor::f32_zeros(&[4])));
    t.push(("image_newline".into(), FixtureTensor::f32_zeros(&[4])));
    // per-layer MoE vision routing bias [experts]
    t.push((
        "layers.0.ffn.gate.bias_vl".into(),
        FixtureTensor::f32_zeros(&[2]),
    ));
}

#[test]
fn tiny_multimodal_checkpoint_loads() {
    let dir = unique_tmp_dir("mm-ok");
    std::fs::write(dir.join("config.json"), tiny_vision_config_json()).unwrap();
    let mut tensors = tiny_tensor_map();
    add_vision_tensors(&mut tensors);
    write_safetensors(&dir, "model.safetensors", tensors);
    let model = load_multimodal_model(&dir).expect("multimodal checkpoint should load");
    assert_eq!(model.layers.len(), 1);
    let vision = model.vision.as_ref().expect("vision tower loaded");
    assert_eq!(vision.vision_dim, 8);
    assert_eq!(vision.n_heads, 2);
    assert_eq!(vision.rope_dim, 2);
    assert_eq!(vision.inter, 3);
    assert_eq!(vision.patch_flat, 3);
    assert_eq!(vision.downsample_ratio, 1);
    assert_eq!(vision.llm_dim, 4);
    assert_eq!(vision.blocks.len(), 1);
    assert!(model.image_start.is_some());
    assert!(model.image_end.is_some());
    assert!(model.image_newline.is_some());
    // Per-layer vision routing bias is populated.
    assert!(model.layers[0].ffn.gate.bias_vl.is_some());
    // The owned vision model can encode a 1x1 patch grid.
    let out = vision.encode_image(&[0.0, 0.0, 0.0], 1, 1);
    assert_eq!(out.len(), 4); // one aligner cell -> llm_dim
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn multimodal_wrong_vision_shape_returns_err() {
    let dir = unique_tmp_dir("mm-bad-shape");
    std::fs::write(dir.join("config.json"), tiny_vision_config_json()).unwrap();
    let mut tensors = tiny_tensor_map();
    add_vision_tensors(&mut tensors);
    // wqkv expected [3*8, 8] = [24, 8]; give [24, 9].
    replace(
        &mut tensors,
        "vision.blocks.0.attn.wqkv.weight",
        FixtureTensor::f32_zeros(&[24, 9]),
    );
    write_safetensors(&dir, "model.safetensors", tensors);
    let err = match load_multimodal_model(&dir) {
        Ok(_) => panic!("expected load to fail"),
        Err(e) => e,
    };
    assert!(
        err.contains("vision.blocks.0.attn.wqkv.weight"),
        "err: {err}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn multimodal_missing_image_start_returns_err() {
    let dir = unique_tmp_dir("mm-missing-delim");
    std::fs::write(dir.join("config.json"), tiny_vision_config_json()).unwrap();
    let mut tensors = tiny_tensor_map();
    add_vision_tensors(&mut tensors);
    remove(&mut tensors, "image_start");
    write_safetensors(&dir, "model.safetensors", tensors);
    let err = match load_multimodal_model(&dir) {
        Ok(_) => panic!("expected load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("image_start"), "err: {err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn multimodal_requires_vision_enabled_config() {
    // A text-only config passed to load_multimodal_model must error before any
    // tensor read.
    let dir = unique_tmp_dir("mm-text-cfg");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    write_safetensors(&dir, "model.safetensors", tiny_tensor_map());
    let err = match load_multimodal_model(&dir) {
        Ok(_) => panic!("expected load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("vision-enabled"), "err: {err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn text_only_load_ignores_vision_tensors_present() {
    // Even when a vision-enabled tensor set is present, a text-only load must
    // succeed and leave the vision surfaces unpopulated.
    let dir = unique_tmp_dir("text-ignores-vision");
    std::fs::write(dir.join("config.json"), tiny_vision_config_json()).unwrap();
    let mut tensors = tiny_tensor_map();
    add_vision_tensors(&mut tensors);
    write_safetensors(&dir, "model.safetensors", tensors);
    let model = load_text_model(&dir).expect("text-only load ignores vision");
    assert!(model.vision.is_none());
    assert!(model.image_start.is_none());
    assert!(model.layers[0].ffn.gate.bias_vl.is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

// -- W3-04: DSpark (mtp.*) checkpoint loader --------------------------------

/// A DSpark-enabled variant of `TINY_CONFIG_JSON`: 2 MTP stages, block_size=3,
/// markov_rank=3, target layers [0] (n_targets=1 => main_proj in_dim=dim=4),
/// and a separate 2-expert DSpark MoE geometry.
fn tiny_dspark_config_json() -> String {
    TINY_CONFIG_JSON
        .replace("\"n_mtp_layers\": 0", "\"n_mtp_layers\": 2")
        .replace("\"dspark_block_size\": 0", "\"dspark_block_size\": 3")
        .replace(
            "\"dspark_noise_token_id\": 0",
            "\"dspark_noise_token_id\": 4",
        )
        .replace(
            "\"dspark_target_layer_ids\": []",
            "\"dspark_target_layer_ids\": [0]",
        )
        .replace("\"dspark_markov_rank\": 0", "\"dspark_markov_rank\": 3")
        .replace(
            "\"dspark_n_routed_experts\": 0",
            "\"dspark_n_routed_experts\": 2",
        )
        .replace(
            "\"dspark_n_activated_experts\": 0",
            "\"dspark_n_activated_experts\": 1",
        )
}

/// Append the `mtp.{stage_id}.*` tensors for one DSpark stage to a base tensor
/// map. Reuses the layer-0 block geometry (dim=4, n_heads=2, head_dim=2,
/// q_lora=3, o_lora=2, o_groups=2, inter=3, hc_mult=2 => mix_hc=8, hc_dim=8)
/// with a 2-expert DSpark MoE, and adds `main_proj`/`main_norm` on stage 0 and
/// the pre-head norm + Markov/confidence heads on the last stage.
fn add_dspark_stage_tensors(
    t: &mut Vec<(String, FixtureTensor)>,
    stage_id: usize,
    is_last: bool,
    dim: usize,
    vocab: usize,
    rank: usize,
    n_targets: usize,
) {
    let p = |s: &str| format!("mtp.{stage_id}.{s}");
    // norms
    t.push((p("attn_norm.weight"), FixtureTensor::f32_zeros(&[dim])));
    t.push((p("ffn_norm.weight"), FixtureTensor::f32_zeros(&[dim])));
    // attention (upstream [out,in])
    t.push((p("attn.wq_a.weight"), FixtureTensor::f32_zeros(&[3, dim])));
    t.push((p("attn.q_norm.weight"), FixtureTensor::f32_zeros(&[3])));
    t.push((p("attn.wq_b.weight"), FixtureTensor::f32_zeros(&[4, 3])));
    t.push((p("attn.wkv.weight"), FixtureTensor::f32_zeros(&[2, dim])));
    t.push((p("attn.kv_norm.weight"), FixtureTensor::f32_zeros(&[2])));
    t.push((p("attn.wo_a.weight"), FixtureTensor::f32_zeros(&[4, 2])));
    t.push((p("attn.wo_b.weight"), FixtureTensor::f32_zeros(&[dim, 4])));
    t.push((p("attn.attn_sink"), FixtureTensor::f32_zeros(&[2])));
    // MoE: 2 DSpark experts
    t.push((p("ffn.gate.weight"), FixtureTensor::f32_zeros(&[2, dim])));
    t.push((p("ffn.gate.bias"), FixtureTensor::f32_zeros(&[2])));
    for e in 0..2 {
        t.push((
            p(&format!("ffn.experts.{e}.w1.weight")),
            FixtureTensor::f32_zeros(&[3, dim]),
        ));
        t.push((
            p(&format!("ffn.experts.{e}.w2.weight")),
            FixtureTensor::f32_zeros(&[dim, 3]),
        ));
        t.push((
            p(&format!("ffn.experts.{e}.w3.weight")),
            FixtureTensor::f32_zeros(&[3, dim]),
        ));
    }
    t.push((
        p("ffn.shared_experts.w1.weight"),
        FixtureTensor::f32_zeros(&[3, dim]),
    ));
    t.push((
        p("ffn.shared_experts.w2.weight"),
        FixtureTensor::f32_zeros(&[dim, 3]),
    ));
    t.push((
        p("ffn.shared_experts.w3.weight"),
        FixtureTensor::f32_zeros(&[3, dim]),
    ));
    // hc
    t.push((p("hc_attn_fn"), FixtureTensor::f32_zeros(&[8, 8])));
    t.push((p("hc_attn_base"), FixtureTensor::f32_zeros(&[8])));
    t.push((p("hc_attn_scale"), FixtureTensor::f32_zeros(&[3])));
    t.push((p("hc_ffn_fn"), FixtureTensor::f32_zeros(&[8, 8])));
    t.push((p("hc_ffn_base"), FixtureTensor::f32_zeros(&[8])));
    t.push((p("hc_ffn_scale"), FixtureTensor::f32_zeros(&[3])));
    // stage 0 owns main_proj [dim, dim*n_targets] / main_norm [dim]
    if stage_id == 0 {
        t.push((
            p("main_proj.weight"),
            FixtureTensor::f32_zeros(&[dim, dim * n_targets]),
        ));
        t.push((p("main_norm.weight"), FixtureTensor::f32_zeros(&[dim])));
    }
    // last stage owns norm + markov/confidence heads
    if is_last {
        t.push((p("norm.weight"), FixtureTensor::f32_zeros(&[dim])));
        t.push((
            p("markov_head.embed.weight"),
            FixtureTensor::f32_zeros(&[vocab, rank]),
        ));
        t.push((
            p("markov_head.head.weight"),
            FixtureTensor::f32_zeros(&[vocab, rank]),
        ));
        t.push((
            p("confidence_head.proj.weight"),
            FixtureTensor::f32_zeros(&[1, dim + rank]),
        ));
    }
}

/// Build a tiny DSpark checkpoint (backbone + 2 mtp stages) in a temp dir and
/// return the loaded text model so the DSpark loader can borrow embed/head.
fn write_tiny_dspark_checkpoint(name: &str) -> PathBuf {
    let dir = unique_tmp_dir(name);
    std::fs::write(dir.join("config.json"), tiny_dspark_config_json()).unwrap();
    let mut tensors = tiny_tensor_map();
    add_dspark_stage_tensors(&mut tensors, 0, false, 4, 5, 3, 1);
    add_dspark_stage_tensors(&mut tensors, 1, true, 4, 5, 3, 1);
    write_safetensors(&dir, "model.safetensors", tensors);
    dir
}

#[test]
fn dspark_head_loads_from_mtp_namespace() {
    let dir = write_tiny_dspark_checkpoint("dspark-ok");
    let model = load_text_model(&dir).expect("text backbone loads");
    let head = load_dspark_head(&dir, &model)
        .expect("dspark head loads")
        .expect("dspark enabled => Some head");
    assert_eq!(head.stages.len(), 2);
    assert_eq!(head.block_size, 3);
    assert_eq!(head.markov_rank, 3);
    assert_eq!(head.noise_token_id, 4);
    // Stage 0 owns main_proj/main_norm; stages after it do not.
    assert!(head.stages[0].main_proj.is_some());
    assert!(head.stages[0].main_norm.is_some());
    assert!(head.stages[1].main_proj.is_none());
    // The last stage owns the heads; stage 0 does not.
    assert!(head.stages[1].markov_head.is_some());
    assert!(head.stages[1].confidence_proj.is_some());
    assert!(head.stages[0].markov_head.is_none());
    // DSpark MoE geometry (2 experts, topk 1) is used for the stage blocks.
    assert_eq!(head.stages[0].block.ffn.experts.len(), 2);
    assert_eq!(head.stages[0].block.ffn.gate.topk, 1);
    // Shared embed/head are borrowed from the backbone.
    assert_eq!(head.embed_tokens.inner.borrow().value.shape.0, vec![5, 4]);
    assert_eq!(head.lm_head.inner.borrow().value.shape.0, vec![4, 5]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dspark_head_is_none_when_disabled() {
    // The plain tiny (text-only) config leaves dspark_block_size == 0.
    let dir = unique_tmp_dir("dspark-off");
    std::fs::write(dir.join("config.json"), TINY_CONFIG_JSON).unwrap();
    write_safetensors(&dir, "model.safetensors", tiny_tensor_map());
    let model = load_text_model(&dir).expect("text backbone loads");
    let head = load_dspark_head(&dir, &model).expect("no error when dspark disabled");
    assert!(head.is_none(), "dspark disabled => None");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dspark_missing_markov_head_returns_err_naming_tensor() {
    let dir = unique_tmp_dir("dspark-missing-markov");
    std::fs::write(dir.join("config.json"), tiny_dspark_config_json()).unwrap();
    let mut tensors = tiny_tensor_map();
    add_dspark_stage_tensors(&mut tensors, 0, false, 4, 5, 3, 1);
    add_dspark_stage_tensors(&mut tensors, 1, true, 4, 5, 3, 1);
    remove(&mut tensors, "mtp.1.markov_head.head.weight");
    write_safetensors(&dir, "model.safetensors", tensors);
    let model = load_text_model(&dir).expect("text backbone loads");
    let err = match load_dspark_head(&dir, &model) {
        Ok(_) => panic!("expected dspark load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("mtp.1.markov_head.head.weight"), "err: {err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dspark_wrong_main_proj_shape_returns_err() {
    let dir = unique_tmp_dir("dspark-bad-mainproj");
    std::fs::write(dir.join("config.json"), tiny_dspark_config_json()).unwrap();
    let mut tensors = tiny_tensor_map();
    add_dspark_stage_tensors(&mut tensors, 0, false, 4, 5, 3, 1);
    add_dspark_stage_tensors(&mut tensors, 1, true, 4, 5, 3, 1);
    // main_proj expected [dim, dim*n_targets] = [4, 4]; give [4, 8].
    replace(
        &mut tensors,
        "mtp.0.main_proj.weight",
        FixtureTensor::f32_zeros(&[4, 8]),
    );
    write_safetensors(&dir, "model.safetensors", tensors);
    let model = load_text_model(&dir).expect("text backbone loads");
    let err = match load_dspark_head(&dir, &model) {
        Ok(_) => panic!("expected dspark load to fail"),
        Err(e) => e,
    };
    assert!(err.contains("mtp.0.main_proj.weight"), "err: {err}");
    assert!(
        err.contains("[4, 4]") && err.contains("[4, 8]"),
        "err: {err}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}
