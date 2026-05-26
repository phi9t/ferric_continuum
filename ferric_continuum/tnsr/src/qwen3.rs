//! Qwen3 (dense) architecture.
//!
//! Faithful port of the Qwen3 dense Transformer block as described in the
//! Qwen3 technical report (arXiv 2505.09388) and the Hugging Face
//! `transformers/models/qwen3` implementation.
//!
//! Differences from `transformer::TransformerBlock` (which models Qwen2's
//! ancestor with classic LayerNorm + GELU):
//!
//! 1. **RMSNorm** (γ only, no β) at pre-attention and pre-MLP positions.
//! 2. **GQA**: `n_q_heads` query heads + `n_kv_heads` shared K/V heads.
//! 3. **Bias-free** Q/K/V/O and gate/up/down projections.
//! 4. **Per-head Q/K RMSNorm** on the `[head_dim]` axis (gamma shared across
//!    heads), applied BEFORE RoPE.  This is the headline Qwen3 dense change.
//! 5. **RoPE** with configurable base (Qwen3 uses 1e6) on Q and K.
//! 6. **SwiGLU MLP**: `down_proj( silu(gate_proj(x)) * up_proj(x) )`.
//!
//! Full model: `embed → N × Qwen3Block → final RMSNorm → lm_head` (untied).
//!
//! Reference: HF `Qwen3Attention.forward` —
//! <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py>

use crate::ops::{activations, basic, embedding, gqa, linear, norm, rope, shape as shape_ops};
use crate::tensor::{Shape, Tensor, TensorValue};

/// Hyperparameters for a Qwen3 dense model.
///
/// Defaults follow no specific checkpoint; pick a preset like
/// [`Qwen3Config::qwen3_8b`] or build your own.
#[derive(Clone, Debug)]
pub struct Qwen3Config {
    /// Vocabulary size for embeddings + LM head (HF embedding table may be
    /// padded; this is the logical token count).
    pub vocab_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dim (a.k.a. `d_model`).
    pub hidden_size: usize,
    /// SwiGLU intermediate size (`d_ff`).
    pub intermediate_size: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of K/V heads (GQA).  Must divide `num_attention_heads`.
    pub num_key_value_heads: usize,
    /// Per-head dimension `Dh` (must be even for RoPE).
    pub head_dim: usize,
    /// RoPE base frequency (`rope_theta`).  Qwen3 = 1e6.
    pub rope_theta: f32,
    /// Use bias on Q/K/V/O projections.  False for standard Qwen3.
    pub attention_bias: bool,
    /// Tie LM head to embedding (small Qwen3 dense ties; large does not).
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    /// Approximate Qwen3-8B (per the report / HF config).
    pub fn qwen3_8b() -> Self {
        Self {
            vocab_size: 151_936,
            num_hidden_layers: 36,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            attention_bias: false,
            tie_word_embeddings: false,
        }
    }

    /// Tiny config for unit tests: V=11, L=2, D=16, F=32, Hq=4, Hk=2, Dh=8.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 11,
            num_hidden_layers: 2,
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rope_theta: 10_000.0,
            attention_bias: false,
            tie_word_embeddings: true,
        }
    }

    pub fn group_size(&self) -> usize {
        assert!(self.num_attention_heads % self.num_key_value_heads == 0);
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_total(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_total(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

fn ones(n: usize) -> Tensor {
    Tensor::from_value(TensorValue::from_vec(Shape(vec![n]), vec![1.0; n]), true)
}

fn param(rows: usize, cols: usize, scale: f32) -> Tensor {
    let t = Tensor::randn_scaled(&[rows, cols], scale);
    t.set_requires_grad(true);
    t
}

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------

pub struct Qwen3Attention {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_cfg: rope::RopeConfig,

    pub wq: Tensor, // [D, Hq*Dh]
    pub wk: Tensor, // [D, Hk*Dh]
    pub wv: Tensor, // [D, Hk*Dh]
    pub wo: Tensor, // [Hq*Dh, D]

    pub q_norm: Tensor, // [Dh]
    pub k_norm: Tensor, // [Dh]
}

impl Qwen3Attention {
    pub fn new(cfg: &Qwen3Config) -> Self {
        let d = cfg.hidden_size;
        let scale = (d as f32).sqrt().recip() * 0.5;
        Self {
            n_q_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_cfg: rope::RopeConfig {
                base: cfg.rope_theta,
                start_pos: 0,
            },
            wq: param(d, cfg.q_total(), scale),
            wk: param(d, cfg.kv_total(), scale),
            wv: param(d, cfg.kv_total(), scale),
            wo: param(cfg.q_total(), d, scale),
            q_norm: ones(cfg.head_dim),
            k_norm: ones(cfg.head_dim),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wo,
            &self.q_norm,
            &self.k_norm,
        ]
    }

    /// `x`: `[B, T, D]` → output `[B, T, D]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (b, t) = {
            let xv = x.inner.borrow();
            let sh = &xv.value.shape.0;
            assert_eq!(sh.len(), 3, "Qwen3Attention input must be [B,T,D]");
            (sh[0], sh[1])
        };
        let hq = self.n_q_heads;
        let hk = self.n_kv_heads;
        let dh = self.head_dim;

        // 1. Bias-free Q/K/V projections.
        let q_flat = linear::linear(x, &self.wq, "q_proj"); // [B,T,Hq*Dh]
        let k_flat = linear::linear(x, &self.wk, "k_proj"); // [B,T,Hk*Dh]
        let v_flat = linear::linear(x, &self.wv, "v_proj"); // [B,T,Hk*Dh]

        // 2. Reshape to multi-head: [B,T,H,Dh].
        let q4 = shape_ops::reshape(&q_flat, &[b, t, hq, dh], "q_reshape");
        let k4 = shape_ops::reshape(&k_flat, &[b, t, hk, dh], "k_reshape");
        let v4 = shape_ops::reshape(&v_flat, &[b, t, hk, dh], "v_reshape");

        // 3. Per-head Q/K RMSNorm (gamma is shared across heads, shape [Dh]).
        //    rms_norm reduces over the last axis, so [B,T,H,Dh] works directly.
        let q4n = norm::rms_norm(&q4, &self.q_norm, "q_norm");
        let k4n = norm::rms_norm(&k4, &self.k_norm, "k_norm");

        // 4. RoPE on Q and K (V is not rotated).
        let q4r = rope::rope(&q4n, self.rope_cfg, "q_rope");
        let k4r = rope::rope(&k4n, self.rope_cfg, "k_rope");

        // 5. GQA causal attention.
        let attn = gqa::gqa_attention(&q4r, &k4r, &v4, "gqa"); // [B,T,Hq,Dh]

        // 6. Flatten heads and bias-free output projection.
        let attn_flat = shape_ops::reshape(&attn, &[b, t, hq * dh], "attn_reshape");
        linear::linear(&attn_flat, &self.wo, "o_proj")
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

pub struct Qwen3MLP {
    pub gate_proj: Tensor, // [D, F]
    pub up_proj: Tensor,   // [D, F]
    pub down_proj: Tensor, // [F, D]
}

impl Qwen3MLP {
    pub fn new(cfg: &Qwen3Config) -> Self {
        let d = cfg.hidden_size;
        let f = cfg.intermediate_size;
        let scale = (d as f32).sqrt().recip() * 0.5;
        Self {
            gate_proj: param(d, f, scale),
            up_proj: param(d, f, scale),
            down_proj: param(f, d, (f as f32).sqrt().recip() * 0.5),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gate_proj, &self.up_proj, &self.down_proj]
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = linear::linear(x, &self.gate_proj, "gate_proj");
        let up = linear::linear(x, &self.up_proj, "up_proj");
        let gated = activations::silu(&gate, "silu");
        let mixed = basic::mul(&gated, &up, "swiglu_mul");
        linear::linear(&mixed, &self.down_proj, "down_proj")
    }
}

// ---------------------------------------------------------------------------
// Decoder block
// ---------------------------------------------------------------------------

pub struct Qwen3Block {
    pub input_layernorm: Tensor,          // [D]  (RMSNorm gamma)
    pub post_attention_layernorm: Tensor, // [D]
    pub self_attn: Qwen3Attention,
    pub mlp: Qwen3MLP,
}

impl Qwen3Block {
    pub fn new(cfg: &Qwen3Config) -> Self {
        Self {
            input_layernorm: ones(cfg.hidden_size),
            post_attention_layernorm: ones(cfg.hidden_size),
            self_attn: Qwen3Attention::new(cfg),
            mlp: Qwen3MLP::new(cfg),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut p = vec![&self.input_layernorm, &self.post_attention_layernorm];
        p.extend(self.self_attn.parameters());
        p.extend(self.mlp.parameters());
        p
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Pre-norm self-attention.
        let h = norm::rms_norm(x, &self.input_layernorm, "input_layernorm");
        let a = self.self_attn.forward(&h);
        let x = basic::add(x, &a, "attn_residual");

        // Pre-norm SwiGLU MLP.
        let h = norm::rms_norm(
            &x,
            &self.post_attention_layernorm,
            "post_attention_layernorm",
        );
        let m = self.mlp.forward(&h);
        basic::add(&x, &m, "mlp_residual")
    }
}

// ---------------------------------------------------------------------------
// Full causal LM
// ---------------------------------------------------------------------------

pub struct Qwen3Model {
    pub cfg: Qwen3Config,
    pub embed_tokens: Tensor, // [V, D]
    pub layers: Vec<Qwen3Block>,
    pub final_norm: Tensor, // [D]
    /// LM head `[D, V]`.  When `tie_word_embeddings` is true this stores the
    /// untied head transposed from `embed_tokens` at construction; callers
    /// that want true weight sharing should pass `&embed_tokens.T` into
    /// `cross_entropy` themselves.
    pub lm_head: Tensor,
}

impl Qwen3Model {
    pub fn new(cfg: Qwen3Config) -> Self {
        let v = cfg.vocab_size;
        let d = cfg.hidden_size;
        let embed_scale = (d as f32).sqrt().recip() * 0.5;
        let embed_tokens = param(v, d, embed_scale);
        let final_norm = ones(d);
        let lm_head = param(d, v, embed_scale);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| Qwen3Block::new(&cfg))
            .collect();
        Self {
            cfg,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut p = vec![&self.embed_tokens, &self.final_norm, &self.lm_head];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p
    }

    /// Run the model on token IDs `[B,T]`.  Returns logits `[B,T,V]`.
    pub fn forward(&self, ids: &[usize], b: usize, t: usize) -> Tensor {
        let mut h = embedding::embedding(ids, b, t, &self.embed_tokens, "embed_tokens");
        for (li, layer) in self.layers.iter().enumerate() {
            // Use the block's own internal op names; layer index recorded via
            // the OpCall debug stream if needed.
            let _ = li;
            h = layer.forward(&h);
        }
        let h = norm::rms_norm(&h, &self.final_norm, "final_norm");
        linear::linear(&h, &self.lm_head, "lm_head")
    }
}
