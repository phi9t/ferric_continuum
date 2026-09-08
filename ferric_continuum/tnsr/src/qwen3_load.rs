//! Load real Hugging Face Qwen3 checkpoints into a [`crate::qwen3::Qwen3Model`].
//!
//! tnsr's Qwen3 math is a faithful from-scratch port, but its tensor layout and
//! two conventions differ from Hugging Face `transformers`.  This module adapts
//! a downloaded HF checkpoint (`config.json` + `model.safetensors`, bf16) into
//! tnsr's f32 parameters so the same forward pass produces the same logits as
//! PyTorch.  Three adaptations matter:
//!
//! 1. **Linear layout.**  tnsr computes `x[..,Din] @ w[Din,Dout]`
//!    ([`ops::linear`](crate::ops::linear)), so every weight is stored
//!    `[Din, Dout]`.  HF stores `nn.Linear` weight as `[out, in]`.  Every
//!    projection is therefore **transposed** on load.  `embed_tokens` and all
//!    `*norm` gammas are copied as-is.
//!
//! 2. **RoPE pairing.**  tnsr's [`ops::rope`](crate::ops::rope) rotates the
//!    *interleaved* pairs `(x[2i], x[2i+1])` with angle `θ_i`.  HF Qwen3 uses
//!    the *half-split* `rotate_half`: it pairs `x[i]` with `x[i+Dh/2]`, both at
//!    angle `θ_i`.  Both schedules use the same `θ_i = pos·base^(-2i/Dh)`.  We
//!    reconcile this **entirely in the loader** — no change to `rope.rs`.
//!
//!    For a Q/K vector produced by tnsr we want the interleaved op to compute
//!    exactly the HF result.  That holds iff tnsr head-dim slot `2i` carries the
//!    value HF put in slot `i`, and slot `2i+1` carries HF slot `i+Dh/2`.  Since
//!    Q/K are `x @ w`, permuting the **output columns** of `q_proj`/`k_proj`
//!    (grouped per head) with `perm[2i]=i, perm[2i+1]=i+Dh/2` achieves it. The
//!    per-head `q_norm`/`k_norm` gammas act on the same head-dim axis *before*
//!    RoPE, so they get the same permutation.
//!    `v_proj` is never rotated (no permute); `o_proj` consumes the
//!    already-un-roped attention output (no permute).
//!
//! 3. **RMSNorm eps.**  HF Qwen3 uses `1e-6`; tnsr's shared `norm::EPS` is set
//!    to `1e-6` to match.
//!
//! The checkpoint file must be downloaded out-of-band (the Bazel sandbox has no
//! network); this loader only reads a local directory path.

use std::fs;
use std::path::Path;

use half::bf16;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use crate::qwen3::{Qwen3Config, Qwen3Model};
use crate::tensor::{Shape, Tensor, TensorValue};

/// Parse an HF `config.json` into a [`Qwen3Config`].
impl Qwen3Config {
    /// Build a config from a Hugging Face `config.json` at `path`.
    pub fn from_hf_json(path: &Path) -> Result<Qwen3Config, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parse config.json: {e}"))?;

        let u = |k: &str| -> Result<usize, String> {
            json.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json missing usize field `{k}`"))
        };
        let f = |k: &str| -> Result<f32, String> {
            json.get(k)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .ok_or_else(|| format!("config.json missing float field `{k}`"))
        };
        let b = |k: &str, default: bool| -> bool {
            json.get(k).and_then(|v| v.as_bool()).unwrap_or(default)
        };

        Ok(Qwen3Config {
            vocab_size: u("vocab_size")?,
            num_hidden_layers: u("num_hidden_layers")?,
            hidden_size: u("hidden_size")?,
            intermediate_size: u("intermediate_size")?,
            num_attention_heads: u("num_attention_heads")?,
            num_key_value_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            rope_theta: f("rope_theta")?,
            attention_bias: b("attention_bias", false),
            tie_word_embeddings: b("tie_word_embeddings", false),
        })
    }
}

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

/// Permute the head-dim columns of a `[Din, n_heads*head_dim]` matrix so the
/// interleaved RoPE op reproduces HF's half-split rotation (see module docs).
///
/// For each head block of `head_dim` output columns, output column `2i` reads
/// source column `i` and output column `2i+1` reads source column `i+head_dim/2`.
fn interleave_headdim(
    name: &str,
    data: &[f32],
    din: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    let cols = n_heads * head_dim;
    expect_numel(name, data.len(), din * cols)?;
    if head_dim % 2 != 0 {
        return Err(format!(
            "tensor `{name}` requires an even head_dim for RoPE interleave, got {head_dim}"
        ));
    }
    let half = head_dim / 2;
    let mut out = vec![0.0f32; din * cols];
    for row in 0..din {
        let base = row * cols;
        for h in 0..n_heads {
            let hb = h * head_dim;
            for i in 0..half {
                out[base + hb + 2 * i] = data[base + hb + i];
                out[base + hb + 2 * i + 1] = data[base + hb + i + half];
            }
        }
    }
    Ok(out)
}

/// Permute a per-head `[head_dim]` gamma with the same interleave map applied to
/// the Q/K projections, so q_norm/k_norm operate on the reordered axis.
fn interleave_gamma(name: &str, data: &[f32], head_dim: usize) -> Result<Vec<f32>, String> {
    expect_numel(name, data.len(), head_dim)?;
    if head_dim % 2 != 0 {
        return Err(format!(
            "tensor `{name}` requires an even head_dim for RoPE interleave, got {head_dim}"
        ));
    }
    let half = head_dim / 2;
    let mut out = vec![0.0f32; head_dim];
    for i in 0..half {
        out[2 * i] = data[i];
        out[2 * i + 1] = data[i + half];
    }
    Ok(out)
}

/// Decode a safetensors `TensorView` into an f32 vector (bf16 or f32 source).
fn view_to_f32(view: &TensorView) -> Result<Vec<f32>, String> {
    let bytes = view.data();
    match view.dtype() {
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err("bf16 tensor byte length not even".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err("f32 tensor byte length not multiple of 4".into());
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err("f16 tensor byte length not even".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        other => Err(format!("unsupported safetensors dtype {other:?}")),
    }
}

/// Overwrite `dst`'s underlying value with `data` reshaped to `shape`.
fn set_param(dst: &Tensor, name: &str, shape: &[usize], data: Vec<f32>) -> Result<(), String> {
    let want = dst.inner.borrow().value.shape.0.clone();
    if want != shape {
        return Err(format!(
            "destination `{name}` expected adapted shape {want:?}, got {shape:?}"
        ));
    }
    expect_numel(name, data.len(), shape.iter().product::<usize>())?;
    dst.inner.borrow_mut().value = TensorValue::from_vec(Shape(shape.to_vec()), data);
    Ok(())
}

/// Load a Hugging Face Qwen3 checkpoint directory into a [`Qwen3Model`].
///
/// `model_dir` must contain `config.json` and `model.safetensors`.  The returned
/// model has all parameters overwritten with adapted checkpoint weights; grad
/// flags are left as constructed (callers run inference under
/// [`grad_mode::NoGradGuard`](crate::grad_mode::NoGradGuard)).
pub fn load_qwen3(model_dir: &Path) -> Result<Qwen3Model, String> {
    let cfg = Qwen3Config::from_hf_json(&model_dir.join("config.json"))?;
    let model = Qwen3Model::new(cfg.clone());

    let st_path = model_dir.join("model.safetensors");
    let raw = fs::read(&st_path).map_err(|e| format!("read {}: {e}", st_path.display()))?;
    let st = SafeTensors::deserialize(&raw).map_err(|e| format!("parse safetensors: {e}"))?;

    let d = cfg.hidden_size;
    let f = cfg.intermediate_size;
    let hq = cfg.num_attention_heads;
    let hk = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let v = cfg.vocab_size;

    let get_shaped = |name: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
        let view = st
            .tensor(name)
            .map_err(|e| format!("tensor `{name}`: {e}"))?;
        let shape = view.shape().to_vec();
        Ok((view_to_f32(&view)?, shape))
    };
    let get_exact = |name: &str, expected: &[usize]| -> Result<Vec<f32>, String> {
        let (data, shape) = get_shaped(name)?;
        expect_shape(name, &shape, expected)?;
        Ok(data)
    };

    // embed_tokens [V, D] — copy as-is.
    set_param(
        &model.embed_tokens,
        "model.embed_tokens.weight",
        &[v, d],
        get_exact("model.embed_tokens.weight", &[v, d])?,
    )?;

    // lm_head [D, V] — transpose of HF lm_head.weight [V, D] (tied fallback to
    // transpose of embed_tokens).
    let lm_head = if st.tensor("lm_head.weight").is_ok() {
        let (w, sh) = get_shaped("lm_head.weight")?; // [V, D]
        expect_shape("lm_head.weight", &sh, &[v, d])?;
        transpose_2d("lm_head.weight", &w, v, d)?
    } else {
        let emb = get_exact("model.embed_tokens.weight", &[v, d])?; // [V, D]
        transpose_2d("model.embed_tokens.weight", &emb, v, d)?
    };
    set_param(&model.lm_head, "lm_head.weight", &[d, v], lm_head)?;

    // final norm [D].
    set_param(
        &model.final_norm,
        "model.norm.weight",
        &[d],
        get_exact("model.norm.weight", &[d])?,
    )?;

    for (li, layer) in model.layers.iter().enumerate() {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        // RMSNorm gammas [D] — copy as-is.
        let input_norm_name = p("input_layernorm.weight");
        set_param(
            &layer.input_layernorm,
            &input_norm_name,
            &[d],
            get_exact(&input_norm_name, &[d])?,
        )?;
        let post_attn_norm_name = p("post_attention_layernorm.weight");
        set_param(
            &layer.post_attention_layernorm,
            &post_attn_norm_name,
            &[d],
            get_exact(&post_attn_norm_name, &[d])?,
        )?;

        // q_proj: HF [Hq*Dh, D] -> transpose [D, Hq*Dh] -> interleave head cols.
        let q_proj_name = p("self_attn.q_proj.weight");
        let (wq, sq) = get_shaped(&q_proj_name)?;
        expect_shape(&q_proj_name, &sq, &[hq * dh, d])?;
        let wq_t = transpose_2d(&q_proj_name, &wq, hq * dh, d)?; // [D, Hq*Dh]
        let wq_i = interleave_headdim(&q_proj_name, &wq_t, d, hq, dh)?;
        set_param(&layer.self_attn.wq, &q_proj_name, &[d, hq * dh], wq_i)?;

        // k_proj: HF [Hk*Dh, D] -> transpose [D, Hk*Dh] -> interleave head cols.
        let k_proj_name = p("self_attn.k_proj.weight");
        let (wk, sk) = get_shaped(&k_proj_name)?;
        expect_shape(&k_proj_name, &sk, &[hk * dh, d])?;
        let wk_t = transpose_2d(&k_proj_name, &wk, hk * dh, d)?; // [D, Hk*Dh]
        let wk_i = interleave_headdim(&k_proj_name, &wk_t, d, hk, dh)?;
        set_param(&layer.self_attn.wk, &k_proj_name, &[d, hk * dh], wk_i)?;

        // v_proj: HF [Hk*Dh, D] -> transpose [D, Hk*Dh]; NO permute (V unroped).
        let v_proj_name = p("self_attn.v_proj.weight");
        let (wv, sv) = get_shaped(&v_proj_name)?;
        expect_shape(&v_proj_name, &sv, &[hk * dh, d])?;
        let wv_t = transpose_2d(&v_proj_name, &wv, hk * dh, d)?;
        set_param(&layer.self_attn.wv, &v_proj_name, &[d, hk * dh], wv_t)?;

        // o_proj: HF [D, Hq*Dh] -> transpose [Hq*Dh, D]; NO permute.
        let o_proj_name = p("self_attn.o_proj.weight");
        let (wo, so) = get_shaped(&o_proj_name)?;
        expect_shape(&o_proj_name, &so, &[d, hq * dh])?;
        let wo_t = transpose_2d(&o_proj_name, &wo, d, hq * dh)?;
        set_param(&layer.self_attn.wo, &o_proj_name, &[hq * dh, d], wo_t)?;

        // q_norm / k_norm [Dh] — interleave to match the permuted Q/K axis.
        let q_norm_name = p("self_attn.q_norm.weight");
        let qn = get_exact(&q_norm_name, &[dh])?;
        set_param(
            &layer.self_attn.q_norm,
            &q_norm_name,
            &[dh],
            interleave_gamma(&q_norm_name, &qn, dh)?,
        )?;
        let k_norm_name = p("self_attn.k_norm.weight");
        let kn = get_exact(&k_norm_name, &[dh])?;
        set_param(
            &layer.self_attn.k_norm,
            &k_norm_name,
            &[dh],
            interleave_gamma(&k_norm_name, &kn, dh)?,
        )?;

        // gate_proj / up_proj: HF [F, D] -> transpose [D, F].
        let gate_proj_name = p("mlp.gate_proj.weight");
        let (gate, sg) = get_shaped(&gate_proj_name)?;
        expect_shape(&gate_proj_name, &sg, &[f, d])?;
        set_param(
            &layer.mlp.gate_proj,
            &gate_proj_name,
            &[d, f],
            transpose_2d(&gate_proj_name, &gate, f, d)?,
        )?;
        let up_proj_name = p("mlp.up_proj.weight");
        let (up, su) = get_shaped(&up_proj_name)?;
        expect_shape(&up_proj_name, &su, &[f, d])?;
        set_param(
            &layer.mlp.up_proj,
            &up_proj_name,
            &[d, f],
            transpose_2d(&up_proj_name, &up, f, d)?,
        )?;

        // down_proj: HF [D, F] -> transpose [F, D].
        let down_proj_name = p("mlp.down_proj.weight");
        let (down, sd) = get_shaped(&down_proj_name)?;
        expect_shape(&down_proj_name, &sd, &[d, f])?;
        set_param(
            &layer.mlp.down_proj,
            &down_proj_name,
            &[f, d],
            transpose_2d(&down_proj_name, &down, d, f)?,
        )?;
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, View};
    use std::borrow::Cow;
    use std::path::PathBuf;

    struct F32FixtureTensor {
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl F32FixtureTensor {
        fn zeros(shape: &[usize]) -> Self {
            let elems = shape.iter().product::<usize>();
            Self {
                shape: shape.to_vec(),
                bytes: vec![0; elems * 4],
            }
        }
    }

    impl View for F32FixtureTensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
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

    fn unique_tmp_model_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("tnsr-qwen3-load-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tiny_config(dir: &std::path::Path) {
        std::fs::write(
            dir.join("config.json"),
            r#"{
  "vocab_size": 11,
  "num_hidden_layers": 1,
  "hidden_size": 16,
  "intermediate_size": 32,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 4,
  "rope_theta": 10000.0,
  "attention_bias": false,
  "tie_word_embeddings": true
}"#,
        )
        .unwrap();
    }

    fn tiny_tensor_map() -> Vec<(String, F32FixtureTensor)> {
        vec![
            (
                "model.embed_tokens.weight".into(),
                F32FixtureTensor::zeros(&[11, 16]),
            ),
            ("model.norm.weight".into(), F32FixtureTensor::zeros(&[16])),
            ("lm_head.weight".into(), F32FixtureTensor::zeros(&[11, 16])),
            (
                "model.layers.0.input_layernorm.weight".into(),
                F32FixtureTensor::zeros(&[16]),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                F32FixtureTensor::zeros(&[16]),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                F32FixtureTensor::zeros(&[16, 16]),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                F32FixtureTensor::zeros(&[8, 16]),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".into(),
                F32FixtureTensor::zeros(&[8, 16]),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".into(),
                F32FixtureTensor::zeros(&[16, 16]),
            ),
            (
                "model.layers.0.self_attn.q_norm.weight".into(),
                F32FixtureTensor::zeros(&[4]),
            ),
            (
                "model.layers.0.self_attn.k_norm.weight".into(),
                F32FixtureTensor::zeros(&[4]),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                F32FixtureTensor::zeros(&[32, 16]),
            ),
            (
                "model.layers.0.mlp.up_proj.weight".into(),
                F32FixtureTensor::zeros(&[32, 16]),
            ),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                F32FixtureTensor::zeros(&[16, 32]),
            ),
        ]
    }

    fn replace_fixture_shape(
        tensors: &mut [(String, F32FixtureTensor)],
        name: &str,
        shape: &[usize],
    ) {
        let (_, tensor) = tensors
            .iter_mut()
            .find(|(candidate, _)| candidate == name)
            .unwrap();
        *tensor = F32FixtureTensor::zeros(shape);
    }

    fn write_safetensors(dir: &std::path::Path, tensors: Vec<(String, F32FixtureTensor)>) {
        let bytes = safetensors::tensor::serialize(tensors, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    #[test]
    fn transpose_roundtrip() {
        // [2,3] row-major -> [3,2].
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose_2d("test", &a, 2, 3).unwrap();
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // transposing back recovers the original.
        assert_eq!(transpose_2d("test", &t, 3, 2).unwrap(), a);
    }

    #[test]
    fn interleave_headdim_maps_pairs() {
        // 1 row, 1 head, head_dim=4: half=2. Source columns [a,b,c,d].
        // out[0]=src[0]=a, out[1]=src[2]=c, out[2]=src[1]=b, out[3]=src[3]=d.
        let src = vec![10.0, 11.0, 12.0, 13.0];
        let out = interleave_headdim("test", &src, 1, 1, 4).unwrap();
        assert_eq!(out, vec![10.0, 12.0, 11.0, 13.0]);
    }

    #[test]
    fn interleave_headdim_two_heads() {
        // 1 row, 2 heads, head_dim=2: half=1. head0 [a,b], head1 [c,d].
        // Per head: out[0]=src[0], out[1]=src[1] (half=1 is identity here).
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let out = interleave_headdim("test", &src, 1, 2, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn interleave_gamma_matches() {
        let g = vec![1.0, 2.0, 3.0, 4.0]; // head_dim=4, half=2
        assert_eq!(
            interleave_gamma("test", &g, 4).unwrap(),
            vec![1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn load_qwen3_returns_error_for_mismatched_checkpoint_tensor_shape() {
        let dir = unique_tmp_model_dir("bad-q-proj-shape");
        write_tiny_config(&dir);
        let mut tensors = tiny_tensor_map();
        // Expected shape is [16, 16]. A malformed checkpoint should return Err
        // from load_qwen3 rather than panic past the Result seam.
        replace_fixture_shape(
            &mut tensors,
            "model.layers.0.self_attn.q_proj.weight",
            &[15, 16],
        );
        write_safetensors(&dir, tensors);

        let err = match load_qwen3(&dir) {
            Ok(_) => panic!("malformed q_proj shape unexpectedly loaded"),
            Err(err) => err,
        };

        assert!(err.contains("model.layers.0.self_attn.q_proj.weight"));
        assert!(err.contains("expected [16, 16]"));
        assert!(err.contains("got [15, 16]"));

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn load_qwen3_returns_error_for_same_numel_wrong_checkpoint_shape() {
        let dir = unique_tmp_model_dir("bad-embed-rank");
        write_tiny_config(&dir);
        let mut tensors = tiny_tensor_map();
        replace_fixture_shape(&mut tensors, "model.embed_tokens.weight", &[176]);
        write_safetensors(&dir, tensors);

        let err = match load_qwen3(&dir) {
            Ok(_) => panic!("same-numel embed_tokens shape unexpectedly loaded"),
            Err(err) => err,
        };

        assert!(err.contains("model.embed_tokens.weight"));
        assert!(err.contains("expected [11, 16]"));
        assert!(err.contains("got [176]"));

        std::fs::remove_dir_all(dir).unwrap();
    }
}
