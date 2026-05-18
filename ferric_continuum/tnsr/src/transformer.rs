use crate::ops::{activations, attention, basic, linear, norm};
use crate::tensor::Tensor;

pub struct TransformerConfig {
    pub batch: usize,
    pub seq: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
}

impl TransformerConfig {
    pub fn tiny_4_7_29() -> Self {
        Self {
            batch: 4,
            seq: 7,
            d_model: 29,
            d_ff: 116,
            n_heads: 1,
        }
    }
}

pub struct TransformerBlock {
    pub cfg: TransformerConfig,

    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,

    pub w1: Tensor,
    pub w2: Tensor,

    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
}

impl TransformerBlock {
    pub fn new(cfg: TransformerConfig) -> Self {
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let scale = (d as f32).sqrt().recip() * 0.5;

        let param = |rows: usize, cols: usize| -> Tensor {
            let t = Tensor::randn_scaled(&[rows, cols], scale);
            t.set_requires_grad(true);
            t
        };

        let ones = |n: usize| -> Tensor {
            Tensor::from_value(
                crate::tensor::TensorValue::from_vec(crate::tensor::Shape(vec![n]), vec![1.0; n]),
                true,
            )
        };

        let zeros = |n: usize| -> Tensor {
            Tensor::from_value(
                crate::tensor::TensorValue::zeros(crate::tensor::Shape(vec![n])),
                true,
            )
        };

        Self {
            cfg,
            wq: param(d, d),
            wk: param(d, d),
            wv: param(d, d),
            wo: param(d, d),
            w1: param(d, ff),
            w2: param(ff, d),
            ln1_gamma: ones(d),
            ln1_beta: zeros(d),
            ln2_gamma: ones(d),
            ln2_beta: zeros(d),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wo,
            &self.w1,
            &self.w2,
            &self.ln1_gamma,
            &self.ln1_beta,
            &self.ln2_gamma,
            &self.ln2_beta,
        ]
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Pre-norm attention
        let x1 = norm::layer_norm(x, &self.ln1_gamma, &self.ln1_beta, "ln1");

        let q = linear::linear(&x1, &self.wq, "q_proj");
        let k = linear::linear(&x1, &self.wk, "k_proj");
        let v = linear::linear(&x1, &self.wv, "v_proj");

        let scale = (self.cfg.d_model as f32).sqrt().recip();
        let scores = attention::attention_scores(&q, &k, scale, "attn_scores");
        let masked = attention::causal_mask(&scores, "attn_mask");
        let p = attention::softmax_last_dim(&masked, "attn_softmax");
        let attn = attention::attention_mix(&p, &v, "attn_mix");

        let attn_out = linear::linear(&attn, &self.wo, "o_proj");
        let x2 = basic::add(x, &attn_out, "attn_residual");

        // Pre-norm MLP
        let x3 = norm::layer_norm(&x2, &self.ln2_gamma, &self.ln2_beta, "ln2");

        let h = linear::linear(&x3, &self.w1, "mlp_up");
        let g = activations::gelu(&h, "mlp_gelu");
        let mlp_out = linear::linear(&g, &self.w2, "mlp_down");

        basic::add(&x2, &mlp_out, "mlp_residual")
    }
}
