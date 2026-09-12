//! DeepSeek V4.1 text-only block and tiny model skeleton.
//!
//! This module wires the already verified DeepSeek V4.1 layer seams in the
//! upstream text order. Real checkpoint loading, quantized kernels, vision, and
//! DSpark speculative decoding are later tickets.

use crate::ops::{embedding, linear, norm};
use crate::tensor::Tensor;

use super::attention::{AttentionLayerInput, DeepSeekV41Attention, SharedAttentionState};
use super::engram::DeepSeekV41Engram;
use super::moe::DeepSeekV41MoE;
use super::residual::{HcStepWeights, ResidualStream};

pub struct DeepSeekV41Block {
    pub layer_id: usize,
    pub dim: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub attn: DeepSeekV41Attention,
    pub ffn: DeepSeekV41MoE,

    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,

    pub engram: Option<DeepSeekV41Engram>,
    pub engram_key: Option<Tensor>,
    pub engram_value: Option<Tensor>,
}

impl DeepSeekV41Block {
    pub fn forward(
        &self,
        stream: &mut ResidualStream,
        start_pos: usize,
        shared: &mut SharedAttentionState,
        token_mask: Option<&[bool]>,
    ) {
        if let Some(engram) = &self.engram {
            let updated = engram.forward_layer(
                &stream.collapse_hc(),
                self.engram_key
                    .as_ref()
                    .expect("engram_key is required when engram is present"),
                self.engram_value
                    .as_ref()
                    .expect("engram_value is required when engram is present"),
                token_mask,
            );
            stream.replace_buffer(&updated);
        }

        let shape = stream.shape();
        assert_eq!(shape.dim, self.dim, "block dim mismatch");
        assert_eq!(shape.hc_mult, self.hc_mult, "block hc_mult mismatch");

        stream.step(
            HcStepWeights {
                hc_fn: &self.hc_attn_fn,
                hc_base: &self.hc_attn_base,
                hc_scale: &self.hc_attn_scale,
                sinkhorn_iters: self.hc_sinkhorn_iters,
                eps: self.hc_eps,
            },
            |attn_in| {
                let attn_in = norm::rms_norm(attn_in, &self.attn_norm, "deepseek.block.attn_norm");
                self.attn.forward_layer(AttentionLayerInput {
                    x: &attn_in,
                    start_pos,
                    shared,
                })
            },
        );

        stream.step(
            HcStepWeights {
                hc_fn: &self.hc_ffn_fn,
                hc_base: &self.hc_ffn_base,
                hc_scale: &self.hc_ffn_scale,
                sinkhorn_iters: self.hc_sinkhorn_iters,
                eps: self.hc_eps,
            },
            |ffn_in| {
                let ffn_in = norm::rms_norm(ffn_in, &self.ffn_norm, "deepseek.block.ffn_norm");
                self.ffn.forward_layer(&ffn_in)
            },
        );
    }
}

pub struct DeepSeekV41TextModel {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub hc_mult: usize,
    pub image_token_id: usize,
    pub embed_tokens: Tensor,
    pub layers: Vec<DeepSeekV41Block>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
}

impl DeepSeekV41TextModel {
    pub fn try_forward_token_ids(
        &self,
        ids: &[usize],
        b: usize,
        s: usize,
    ) -> Result<Tensor, String> {
        if ids.len() != b * s {
            return Err(format!(
                "ids length {} does not match b*s {}",
                ids.len(),
                b * s
            ));
        }
        if ids.iter().any(|&id| id == self.image_token_id) {
            return Err("DeepSeekV41TextModel is text-only and rejects image token inputs".into());
        }
        if ids.iter().any(|&id| id >= self.vocab_size) {
            return Err(format!("token id must be < vocab_size {}", self.vocab_size));
        }

        let h = embedding::embedding(ids, b, s, &self.embed_tokens, "deepseek.embed_tokens");
        let mut stream = ResidualStream::from_embedding(&h, self.hc_mult);
        let mut shared = SharedAttentionState::default();
        for layer in &self.layers {
            layer.forward(&mut stream, 0, &mut shared, None);
        }
        let collapsed = stream.collapse();
        let collapsed = norm::rms_norm(&collapsed, &self.final_norm, "deepseek.final_norm");
        Ok(linear::linear(
            &collapsed,
            &self.lm_head,
            "deepseek.lm_head",
        ))
    }

    pub fn forward_token_ids(&self, ids: &[usize], b: usize, s: usize) -> Tensor {
        self.try_forward_token_ids(ids, b, s)
            .expect("DeepSeekV41TextModel forward_token_ids failed")
    }
}
