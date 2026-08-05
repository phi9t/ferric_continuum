//! Load real Hugging Face Qwen3 checkpoints into a [`qwen3::Qwen3Model`].
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
//!    projection is therefore **transposed** on load
//!    ([`transpose_2d`]).  `embed_tokens` and all `*norm` gammas are copied
//!    as-is.
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
//!    (grouped per head) with `perm[2i]=i, perm[2i+1]=i+Dh/2` achieves it
//!    ([`interleave_headdim`]).  The per-head `q_norm`/`k_norm` gammas act on the
//!    same head-dim axis *before* RoPE, so they get the same permutation.
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
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
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

/// Transpose a row-major `[rows, cols]` buffer into `[cols, rows]`.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols, "transpose_2d: size mismatch");
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Permute the head-dim columns of a `[Din, n_heads*head_dim]` matrix so the
/// interleaved RoPE op reproduces HF's half-split rotation (see module docs).
///
/// For each head block of `head_dim` output columns, output column `2i` reads
/// source column `i` and output column `2i+1` reads source column `i+head_dim/2`.
fn interleave_headdim(data: &[f32], din: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let cols = n_heads * head_dim;
    assert_eq!(data.len(), din * cols, "interleave_headdim: size mismatch");
    assert!(head_dim % 2 == 0, "interleave_headdim: head_dim must be even");
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
    out
}

/// Permute a per-head `[head_dim]` gamma with the same interleave map applied to
/// the Q/K projections, so q_norm/k_norm operate on the reordered axis.
fn interleave_gamma(data: &[f32], head_dim: usize) -> Vec<f32> {
    assert_eq!(data.len(), head_dim, "interleave_gamma: size mismatch");
    assert!(head_dim % 2 == 0, "interleave_gamma: head_dim must be even");
    let half = head_dim / 2;
    let mut out = vec![0.0f32; head_dim];
    for i in 0..half {
        out[2 * i] = data[i];
        out[2 * i + 1] = data[i + half];
    }
    out
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

/// Overwrite `dst`'s underlying value with `data` reshaped to `shape`, asserting
/// the shape matches the destination parameter exactly.
fn set_param(dst: &Tensor, shape: &[usize], data: Vec<f32>) {
    let want = dst.inner.borrow().value.shape.0.clone();
    assert_eq!(
        want, shape,
        "load_qwen3: destination shape {want:?} != adapted shape {shape:?}"
    );
    assert_eq!(shape.iter().product::<usize>(), data.len(), "numel mismatch");
    dst.inner.borrow_mut().value = TensorValue::from_vec(Shape(shape.to_vec()), data);
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
    let st = SafeTensors::deserialize(&raw)
        .map_err(|e| format!("parse safetensors: {e}"))?;

    let d = cfg.hidden_size;
    let f = cfg.intermediate_size;
    let hq = cfg.num_attention_heads;
    let hk = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let v = cfg.vocab_size;

    let get = |name: &str| -> Result<Vec<f32>, String> {
        let view = st
            .tensor(name)
            .map_err(|e| format!("tensor `{name}`: {e}"))?;
        view_to_f32(&view)
    };
    // Fetch and return (data, shape) so we can transpose with real dims.
    let get_shaped = |name: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
        let view = st
            .tensor(name)
            .map_err(|e| format!("tensor `{name}`: {e}"))?;
        let shape = view.shape().to_vec();
        Ok((view_to_f32(&view)?, shape))
    };

    // embed_tokens [V, D] — copy as-is.
    set_param(&model.embed_tokens, &[v, d], get("model.embed_tokens.weight")?);

    // lm_head [D, V] — transpose of HF lm_head.weight [V, D] (tied fallback to
    // transpose of embed_tokens).
    let lm_head = if st.tensor("lm_head.weight").is_ok() {
        let (w, sh) = get_shaped("lm_head.weight")?; // [V, D]
        assert_eq!(sh, vec![v, d], "lm_head.weight shape");
        transpose_2d(&w, v, d)
    } else {
        let emb = get("model.embed_tokens.weight")?; // [V, D]
        transpose_2d(&emb, v, d)
    };
    set_param(&model.lm_head, &[d, v], lm_head);

    // final norm [D].
    set_param(&model.final_norm, &[d], get("model.norm.weight")?);

    for (li, layer) in model.layers.iter().enumerate() {
        let p = |s: &str| format!("model.layers.{li}.{s}");

        // RMSNorm gammas [D] — copy as-is.
        set_param(&layer.input_layernorm, &[d], get(&p("input_layernorm.weight"))?);
        set_param(
            &layer.post_attention_layernorm,
            &[d],
            get(&p("post_attention_layernorm.weight"))?,
        );

        // q_proj: HF [Hq*Dh, D] -> transpose [D, Hq*Dh] -> interleave head cols.
        let (wq, sq) = get_shaped(&p("self_attn.q_proj.weight"))?;
        assert_eq!(sq, vec![hq * dh, d], "q_proj shape");
        let wq_t = transpose_2d(&wq, hq * dh, d); // [D, Hq*Dh]
        let wq_i = interleave_headdim(&wq_t, d, hq, dh);
        set_param(&layer.self_attn.wq, &[d, hq * dh], wq_i);

        // k_proj: HF [Hk*Dh, D] -> transpose [D, Hk*Dh] -> interleave head cols.
        let (wk, sk) = get_shaped(&p("self_attn.k_proj.weight"))?;
        assert_eq!(sk, vec![hk * dh, d], "k_proj shape");
        let wk_t = transpose_2d(&wk, hk * dh, d); // [D, Hk*Dh]
        let wk_i = interleave_headdim(&wk_t, d, hk, dh);
        set_param(&layer.self_attn.wk, &[d, hk * dh], wk_i);

        // v_proj: HF [Hk*Dh, D] -> transpose [D, Hk*Dh]; NO permute (V unroped).
        let (wv, sv) = get_shaped(&p("self_attn.v_proj.weight"))?;
        assert_eq!(sv, vec![hk * dh, d], "v_proj shape");
        let wv_t = transpose_2d(&wv, hk * dh, d);
        set_param(&layer.self_attn.wv, &[d, hk * dh], wv_t);

        // o_proj: HF [D, Hq*Dh] -> transpose [Hq*Dh, D]; NO permute.
        let (wo, so) = get_shaped(&p("self_attn.o_proj.weight"))?;
        assert_eq!(so, vec![d, hq * dh], "o_proj shape");
        let wo_t = transpose_2d(&wo, d, hq * dh);
        set_param(&layer.self_attn.wo, &[hq * dh, d], wo_t);

        // q_norm / k_norm [Dh] — interleave to match the permuted Q/K axis.
        let qn = get(&p("self_attn.q_norm.weight"))?;
        set_param(&layer.self_attn.q_norm, &[dh], interleave_gamma(&qn, dh));
        let kn = get(&p("self_attn.k_norm.weight"))?;
        set_param(&layer.self_attn.k_norm, &[dh], interleave_gamma(&kn, dh));

        // gate_proj / up_proj: HF [F, D] -> transpose [D, F].
        let (gate, sg) = get_shaped(&p("mlp.gate_proj.weight"))?;
        assert_eq!(sg, vec![f, d], "gate_proj shape");
        set_param(&layer.mlp.gate_proj, &[d, f], transpose_2d(&gate, f, d));
        let (up, su) = get_shaped(&p("mlp.up_proj.weight"))?;
        assert_eq!(su, vec![f, d], "up_proj shape");
        set_param(&layer.mlp.up_proj, &[d, f], transpose_2d(&up, f, d));

        // down_proj: HF [D, F] -> transpose [F, D].
        let (down, sd) = get_shaped(&p("mlp.down_proj.weight"))?;
        assert_eq!(sd, vec![d, f], "down_proj shape");
        set_param(&layer.mlp.down_proj, &[f, d], transpose_2d(&down, d, f));
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_roundtrip() {
        // [2,3] row-major -> [3,2].
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose_2d(&a, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // transposing back recovers the original.
        assert_eq!(transpose_2d(&t, 3, 2), a);
    }

    #[test]
    fn interleave_headdim_maps_pairs() {
        // 1 row, 1 head, head_dim=4: half=2. Source columns [a,b,c,d].
        // out[0]=src[0]=a, out[1]=src[2]=c, out[2]=src[1]=b, out[3]=src[3]=d.
        let src = vec![10.0, 11.0, 12.0, 13.0];
        let out = interleave_headdim(&src, 1, 1, 4);
        assert_eq!(out, vec![10.0, 12.0, 11.0, 13.0]);
    }

    #[test]
    fn interleave_headdim_two_heads() {
        // 1 row, 2 heads, head_dim=2: half=1. head0 [a,b], head1 [c,d].
        // Per head: out[0]=src[0], out[1]=src[1] (half=1 is identity here).
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let out = interleave_headdim(&src, 1, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn interleave_gamma_matches() {
        let g = vec![1.0, 2.0, 3.0, 4.0]; // head_dim=4, half=2
        assert_eq!(interleave_gamma(&g, 4), vec![1.0, 3.0, 2.0, 4.0]);
    }
}
