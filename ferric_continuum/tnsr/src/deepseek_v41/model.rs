//! DeepSeek V4.1 text-only block and tiny model skeleton.
//!
//! This module wires the already verified DeepSeek V4.1 layer seams in the
//! upstream text order. Real checkpoint loading, quantized kernels, vision, and
//! DSpark speculative decoding are later tickets.

use crate::ops::{embedding, linear, norm};
use crate::tensor::{Shape, Tensor, TensorValue};

use super::attention::{AttentionLayerInput, DeepSeekV41Attention, SharedAttentionState};
use super::engram::DeepSeekV41Engram;
use super::moe::DeepSeekV41MoE;
use super::residual::{HcStepWeights, ResidualStream};
use super::vision::OwnedVisionModel;
use super::vision_grid::{IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START};

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
        self.forward_with_masks(stream, start_pos, shared, token_mask, None)
    }

    /// Block forward with distinct engram and image masks. `engram_mask`
    /// suppresses the engram update on the masked-out tokens (upstream passes
    /// `~image_mask`); `image_mask` selects the VL routing bias in the MoE gate.
    pub fn forward_with_masks(
        &self,
        stream: &mut ResidualStream,
        start_pos: usize,
        shared: &mut SharedAttentionState,
        engram_mask: Option<&[bool]>,
        image_mask: Option<&[bool]>,
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
                engram_mask,
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
                self.ffn.forward_layer_masked(&ffn_in, image_mask)
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

    /// Vision tower + aligner, present only for a vision-enabled checkpoint.
    pub vision: Option<OwnedVisionModel>,
    /// Learned image-span delimiter embeddings (`[hidden_size]` each), present
    /// only for a vision-enabled checkpoint.
    pub image_start: Option<Vec<f32>>,
    pub image_end: Option<Vec<f32>>,
    pub image_newline: Option<Vec<f32>>,
}

/// A single image's contribution to a multimodal forward: its token span layout
/// (`token_types` over `n_llm_h*(n_llm_w+1)+2` positions) and the aligner rows
/// (`[n_image_slots, hidden_size]`) that fill the `IMAGE` slots. `start` is the
/// position of `IMAGE_START` within the sequence.
pub struct ImageSpan<'a> {
    pub start: usize,
    pub token_types: &'a [i64],
    pub aligner_rows: &'a [f32],
}

/// Learned image-span delimiter embeddings (`[hidden_size]` each). Mirrors
/// `Transformer.image_start/image_end/image_newline`.
pub struct ImageDelimiters<'a> {
    pub image_start: &'a [f32],
    pub image_end: &'a [f32],
    pub image_newline: &'a [f32],
}

/// Overwrite each image's token span in the `[B, S, D]` embedding buffer with
/// its delimiter embeddings and aligner rows. Mirrors
/// `Transformer.merge_image_embeddings`: the `IMAGE` slots take aligner rows in
/// reading order; the sentinels take the learned delimiters.
pub fn merge_image_embeddings(
    embed: &mut [f32],
    b: usize,
    s: usize,
    dim: usize,
    images: &[Vec<ImageSpan>],
    delims: &ImageDelimiters,
) {
    assert_eq!(embed.len(), b * s * dim, "embed buffer shape mismatch");
    assert_eq!(images.len(), b, "one image list per batch row");
    for (bi, sample) in images.iter().enumerate() {
        for img in sample {
            let mut aligner_off = 0;
            for (k, &ty) in img.token_types.iter().enumerate() {
                let pos = img.start + k;
                assert!(pos < s, "image span overflows sequence length");
                let dst = &mut embed[(bi * s + pos) * dim..(bi * s + pos + 1) * dim];
                match ty {
                    IMAGE_START => dst.copy_from_slice(delims.image_start),
                    IMAGE_END => dst.copy_from_slice(delims.image_end),
                    IMAGE_NEW_LINE => dst.copy_from_slice(delims.image_newline),
                    IMAGE => {
                        let src = &img.aligner_rows[aligner_off * dim..(aligner_off + 1) * dim];
                        dst.copy_from_slice(src);
                        aligner_off += 1;
                    }
                    _ => panic!("unexpected image token type {ty} inside an image span"),
                }
            }
        }
    }
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

    /// Multimodal forward. `ids` are the real token ids (`image_token_id` fills
    /// every `IMAGE`/delimiter slot); `token_types` is `[B*S]` with `TEXT=-1`
    /// on text positions and the image tags inside image spans. `images` supplies
    /// the aligner rows per span. Mirrors `Transformer.forward(images, token_types)`.
    ///
    /// The image mask (`token_types >= 0`) selects the VL routing bias in each
    /// MoE gate; its negation suppresses the engram update on image tokens.
    pub fn try_forward_multimodal(
        &self,
        ids: &[usize],
        token_types: &[i64],
        b: usize,
        s: usize,
        images: &[Vec<ImageSpan>],
        delims: &ImageDelimiters,
    ) -> Result<Tensor, String> {
        if ids.len() != b * s {
            return Err(format!(
                "ids length {} does not match b*s {}",
                ids.len(),
                b * s
            ));
        }
        if token_types.len() != b * s {
            return Err(format!(
                "token_types length {} does not match b*s {}",
                token_types.len(),
                b * s
            ));
        }
        if ids.iter().any(|&id| id >= self.vocab_size) {
            return Err(format!("token id must be < vocab_size {}", self.vocab_size));
        }

        let dim = self.hidden_size;
        // Embed, then overwrite image spans BEFORE the HC expansion (upstream
        // merges into `h` then unsqueezes to hc_mult copies).
        let h = embedding::embedding(ids, b, s, &self.embed_tokens, "deepseek.embed_tokens");
        let mut buffer = h.inner.borrow().value.data.as_ref().clone();
        merge_image_embeddings(&mut buffer, b, s, dim, images, delims);
        let merged =
            Tensor::from_value_no_grad(TensorValue::from_vec(Shape(vec![b, s, dim]), buffer));

        // image_mask: TEXT (-1) is text; anything >= 0 is inside an image span.
        let image_mask: Vec<bool> = token_types.iter().map(|&t| t >= 0).collect();
        let engram_mask: Vec<bool> = image_mask.iter().map(|&m| !m).collect();

        let mut stream = ResidualStream::from_embedding(&merged, self.hc_mult);
        let mut shared = SharedAttentionState::default();
        for layer in &self.layers {
            layer.forward_with_masks(
                &mut stream,
                0,
                &mut shared,
                Some(&engram_mask),
                Some(&image_mask),
            );
        }
        let collapsed = stream.collapse();
        let collapsed = norm::rms_norm(&collapsed, &self.final_norm, "deepseek.final_norm");
        Ok(linear::linear(
            &collapsed,
            &self.lm_head,
            "deepseek.lm_head",
        ))
    }
}
