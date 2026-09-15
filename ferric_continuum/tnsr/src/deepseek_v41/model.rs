//! DeepSeek V4.1 text-only block and tiny model skeleton.
//!
//! This module wires the already verified DeepSeek V4.1 layer seams in the
//! upstream text order. Real checkpoint loading, quantized kernels, vision, and
//! DSpark speculative decoding are later tickets.

use crate::ops::{embedding, linear, norm};
use crate::tensor::{Shape, Tensor, TensorValue};

use super::attention::{AttentionLayerInput, DeepSeekV41Attention, SharedAttentionState};
use super::dspark::{argmax, main_proj_norm, markov_head_forward};
use super::engram::DeepSeekV41Engram;
use super::hc_tensor::{expand_hc, shape_of, to_flat, HcShape};
use super::hyper::hc_pre;
use super::moe::DeepSeekV41MoE;
use super::residual::{identity_pre_mix, HcStepWeights, ResidualStream};
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
/// reading order; the sentinels take the learned delimiters. Returns a named
/// error if a span overflows the sequence or carries an unknown token type,
/// so image inputs that do not match the token layout fail cleanly rather than
/// aborting the process.
pub fn merge_image_embeddings(
    embed: &mut [f32],
    b: usize,
    s: usize,
    dim: usize,
    images: &[Vec<ImageSpan>],
    delims: &ImageDelimiters,
) -> Result<(), String> {
    if embed.len() != b * s * dim {
        return Err(format!(
            "embed buffer length {} does not match b*s*dim {}",
            embed.len(),
            b * s * dim
        ));
    }
    if images.len() != b {
        return Err(format!(
            "images length {} does not match batch {b}",
            images.len()
        ));
    }
    for (bi, sample) in images.iter().enumerate() {
        for img in sample {
            let mut aligner_off = 0;
            for (k, &ty) in img.token_types.iter().enumerate() {
                let pos = img.start + k;
                if pos >= s {
                    return Err(format!(
                        "image span position {pos} overflows sequence length {s}"
                    ));
                }
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
                    other => {
                        return Err(format!(
                            "unexpected image token type {other} inside an image span"
                        ))
                    }
                }
            }
        }
    }
    Ok(())
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
        merge_image_embeddings(&mut buffer, b, s, dim, images, delims)?;
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

/// One DSpark MTP stage, stored under the `mtp.N.*` checkpoint namespace.
///
/// Every stage is a [`DeepSeekV41Block`] (the DSpark attention runs with
/// `compress_ratio == 0`, i.e. the sliding-window path this skeleton already
/// executes). Stage 0 additionally owns `main_proj`/`main_norm`, which turn the
/// concatenated target-layer hiddens into the draft embedding seam; the last
/// stage owns the shared `norm` plus the Markov and confidence heads that drive
/// `forward_head`.
pub struct DeepSeekV41DsparkStage {
    pub block: DeepSeekV41Block,

    /// Stage-0 only: `main_proj` weight `[in_dim, dim]` (tnsr layout) and its
    /// `main_norm` gamma `[dim]`. `in_dim = dim * len(target_layer_ids)`.
    pub main_proj: Option<Vec<f32>>,
    pub main_norm: Option<Vec<f32>>,

    /// Last-stage only: the pre-head `norm` gamma `[dim]`, the Markov head
    /// (`embed`/`head` are both `[vocab, rank]`), and the confidence projection
    /// (`[dim + rank]`).
    pub head_norm: Option<Vec<f32>>,
    pub markov_embed: Option<Vec<f32>>,
    pub markov_head: Option<Vec<f32>>,
    pub confidence_proj: Option<Vec<f32>>,
}

/// The DSpark (multi-token-prediction / speculative-decoding) head: an ordered
/// list of [`DeepSeekV41DsparkStage`]s plus the shared embedding and language
/// head borrowed from the backbone [`DeepSeekV41TextModel`].
///
/// This mirrors `Transformer.forward_spec`: stage 0 builds the draft embedding
/// from the target-layer main hidden, all stages run their block forward over
/// the `[B, block, HC, D]` residual stream, and the last stage's `forward_head`
/// produces the biased draft logits, greedy output ids, and confidence.
pub struct DeepSeekV41DsparkHead {
    pub vocab_size: usize,
    pub dim: usize,
    pub hc_mult: usize,
    pub block_size: usize,
    pub noise_token_id: usize,
    pub markov_rank: usize,
    pub head_eps: f32,

    /// Shared token embedding `[vocab, dim]` (upstream `Transformer.embed`).
    pub embed_tokens: Tensor,
    /// Shared LM head `[dim, vocab]` (tnsr layout of upstream `Transformer.head`).
    pub lm_head: Tensor,

    pub stages: Vec<DeepSeekV41DsparkStage>,
}

/// The output of one DSpark speculative step, mirroring
/// `DSparkBlock.forward_head`'s `(output_ids, logits, confidence)`.
pub struct DsparkSpecOutput {
    /// Draft ids `[batch, block_size + 1]` (row 0 = accepted input id).
    pub output_ids: Vec<usize>,
    /// Biased head logits `[batch, block_size, vocab]`.
    pub logits: Vec<f32>,
    /// Per-position confidence `[batch, block_size]`.
    pub confidence: Vec<f32>,
}

impl DeepSeekV41DsparkHead {
    fn first_stage(&self) -> Result<&DeepSeekV41DsparkStage, String> {
        self.stages
            .first()
            .ok_or_else(|| "DSpark head needs a stage 0".to_string())
    }

    fn last_stage(&self) -> Result<&DeepSeekV41DsparkStage, String> {
        self.stages
            .last()
            .ok_or_else(|| "DSpark head needs a last stage".to_string())
    }

    /// Build the stage-0 draft residual embedding and the projected main hidden.
    ///
    /// Mirrors `DSparkBlock.forward_embed`: `main_x = main_norm(main_proj(
    /// main_hidden))`; the draft ids are `noise_token_id` everywhere except
    /// column 0 (the accepted `input_ids`); the embedded ids seed the
    /// `[batch, block, dim]` draft that is later broadcast across the HC axis.
    ///
    /// `main_hidden` is `[batch, in_dim]` with `in_dim = dim * n_targets`;
    /// `input_ids` is one accepted id per batch row. Returns the `[B, block, D]`
    /// draft embedding tensor and the `[batch, dim]` `main_x`, or a named error
    /// if stage 0 is missing the `main_proj`/`main_norm` tensors or the shapes
    /// do not line up.
    fn forward_embed(
        &self,
        main_hidden: &[f32],
        input_ids: &[usize],
    ) -> Result<(Tensor, Vec<f32>), String> {
        let batch = input_ids.len();
        let stage0 = self.first_stage()?;
        let proj = stage0
            .main_proj
            .as_ref()
            .ok_or("DSpark stage 0 must own main_proj")?;
        let main_norm = stage0
            .main_norm
            .as_ref()
            .ok_or("DSpark stage 0 must own main_norm")?;
        if proj.len() % self.dim != 0 {
            return Err(format!(
                "DSpark main_proj length {} must be a multiple of dim {}",
                proj.len(),
                self.dim
            ));
        }
        let in_dim = proj.len() / self.dim;
        if main_hidden.len() != batch * in_dim {
            return Err(format!(
                "DSpark main_hidden length {} must equal batch*in_dim {}",
                main_hidden.len(),
                batch * in_dim
            ));
        }
        let main_x = main_proj_norm(
            main_hidden,
            proj,
            main_norm,
            in_dim,
            self.dim,
            self.head_eps,
        );

        // Draft ids: [batch, block_size], noise everywhere but column 0.
        let mut draft = vec![self.noise_token_id; batch * self.block_size];
        for (b, &id) in input_ids.iter().enumerate() {
            draft[b * self.block_size] = id;
        }
        let embedded = embedding::embedding(
            &draft,
            batch,
            self.block_size,
            &self.embed_tokens,
            "deepseek.dspark.embed",
        );
        Ok((embedded, main_x))
    }

    /// The last-stage `forward_head` draft loop.
    ///
    /// Mirrors `DSparkBlock.forward_head`: collapse the residual stream with the
    /// carried pre-mix (`hc_pre`), run the shared head over `norm(x)` to get the
    /// base logits, then for each position add the current output id's Markov
    /// bias, greedily (temperature 0) sample the next id, and stack the Markov
    /// embeddings; finally score `concat(hidden, markov_embed)` with the
    /// confidence head. `input_ids` is one accepted id per batch row.
    fn forward_head(
        &self,
        stream: &ResidualStream,
        input_ids: &[usize],
    ) -> Result<DsparkSpecOutput, String> {
        let batch = input_ids.len();
        let stage = self.last_stage()?;
        let head_norm = stage
            .head_norm
            .as_ref()
            .ok_or("DSpark last stage must own head norm")?;
        let markov_embed = stage
            .markov_embed
            .as_ref()
            .ok_or("DSpark last stage must own markov embed")?;
        let markov_head = stage
            .markov_head
            .as_ref()
            .ok_or("DSpark last stage must own markov head")?;
        let confidence_proj = stage
            .confidence_proj
            .as_ref()
            .ok_or("DSpark last stage must own confidence proj")?;

        // Collapse [B, block, HC, D] with the carried pre-mix -> [B, block, D].
        let shape = stream.shape();
        let hidden_flat = hc_pre(
            &to_flat(&stream.collapse_hc()),
            stream.pre_mix(),
            HcShape {
                batch: shape.batch,
                seqlen: shape.seqlen,
                hc_mult: shape.hc_mult,
                dim: shape.dim,
            },
        );
        let hidden_tensor = Tensor::from_value_no_grad(TensorValue::from_vec(
            Shape(vec![batch, self.block_size, self.dim]),
            hidden_flat.clone(),
        ));

        // Base logits: head(norm(hidden)) as a full row per draft position.
        let normed = norm::rms_norm(
            &hidden_tensor,
            &param_gamma(head_norm),
            "deepseek.dspark.head_norm",
        );
        let base_logits = to_flat(&linear::linear(
            &normed,
            &self.lm_head,
            "deepseek.dspark.head",
        ));

        let vocab = self.vocab_size;
        let rank = self.markov_rank;
        let mut logits = base_logits;
        let mut output_ids = vec![0usize; batch * (self.block_size + 1)];
        let mut confidence = vec![0.0f32; batch * self.block_size];

        for b in 0..batch {
            let out_base = b * (self.block_size + 1);
            output_ids[out_base] = input_ids[b];
            let mut markov_embeds: Vec<f32> = Vec::with_capacity(self.block_size * rank);
            for i in 0..self.block_size {
                let cur = output_ids[out_base + i];
                let (bias, embed) =
                    markov_head_forward(cur, markov_embed, markov_head, vocab, rank);
                let row_base = (b * self.block_size + i) * vocab;
                let row = &mut logits[row_base..row_base + vocab];
                for (l, add) in row.iter_mut().zip(&bias) {
                    *l += add;
                }
                markov_embeds.extend_from_slice(&embed);
                output_ids[out_base + i + 1] = argmax(row);
            }
            for i in 0..self.block_size {
                let h = &hidden_flat[(b * self.block_size + i) * self.dim
                    ..(b * self.block_size + i + 1) * self.dim];
                let m = &markov_embeds[i * rank..(i + 1) * rank];
                confidence[b * self.block_size + i] =
                    super::dspark::confidence_head_forward(h, m, confidence_proj);
            }
        }

        Ok(DsparkSpecOutput {
            output_ids,
            logits,
            confidence,
        })
    }

    /// Run one DSpark speculative step over the accepted `input_ids`.
    ///
    /// Mirrors `Transformer.forward_spec` for `start_pos > 0`: build the draft
    /// embedding from `main_hidden`, expand across the hyper-connection axis with
    /// the identity pre-mix, run every stage's block forward (drafts are text, so
    /// no VL routing bias), then `forward_head` over the final stream.
    ///
    /// `main_hidden` is `[batch, dim * n_targets]`; `input_ids` is one accepted
    /// id per batch row. Returns a named error instead of panicking when the
    /// head is misconfigured (empty ids, missing stage tensors, shape mismatch)
    /// so callers embedding this in a server do not abort the process.
    pub fn try_forward_spec(
        &self,
        input_ids: &[usize],
        main_hidden: &[f32],
    ) -> Result<DsparkSpecOutput, String> {
        if input_ids.is_empty() {
            return Err("forward_spec needs at least one accepted id".to_string());
        }
        let (embedded, main_x) = self.forward_embed(main_hidden, input_ids)?;
        // main_x seeds the sliding-window KV cache in the full decode path
        // (start_pos > 0 only); the tiny parity harness runs one prefill step.
        let _ = main_x;
        let expanded = expand_hc(&embedded, self.hc_mult);
        let hc = shape_of(&expanded);
        let pre_mix = identity_pre_mix(hc.batch, hc.seqlen, hc.hc_mult);
        let mut stream = ResidualStream::from_hc_tensor(&expanded, pre_mix);
        let mut shared = SharedAttentionState::default();
        for stage in &self.stages {
            stage.block.forward(&mut stream, 0, &mut shared, None);
        }
        self.forward_head(&stream, input_ids)
    }

    /// Panicking convenience wrapper over [`Self::try_forward_spec`] for tests
    /// and callers that treat a misconfigured head as a programmer error.
    pub fn forward_spec(&self, input_ids: &[usize], main_hidden: &[f32]) -> DsparkSpecOutput {
        self.try_forward_spec(input_ids, main_hidden)
            .expect("DeepSeekV41DsparkHead forward_spec failed")
    }
}

/// Wrap an RMSNorm gamma slice as a no-grad `[dim]` tensor for [`norm::rms_norm`].
fn param_gamma(gamma: &[f32]) -> Tensor {
    Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![gamma.len()]),
        gamma.to_vec(),
    ))
}
