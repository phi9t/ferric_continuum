//! DeepSeek V4.1 block, text model, multimodal, and DSpark wiring.
//!
//! This module keeps the runtime assembly close to the upstream execution
//! order: token embeddings, optional image-span replacement, Engram updates,
//! CED/CSA2 attention, MoE blocks, final logits, and the DSpark `forward_spec`
//! path. The implementation is intentionally CPU-readable and fixture-driven;
//! real-weight, native-quantized, and tensor-parallel claims are made only by
//! the verifier scripts and receipts that name those environments.

use std::cell::RefCell;

use crate::ops::{embedding, linear, norm};
use crate::tensor::{Shape, Tensor, TensorValue};

use super::attention::{AttentionLayerInput, DeepSeekV41Attention, SharedAttentionState};
use super::dspark::{argmax, decode_topk_indices, main_proj_norm, markov_head_forward};
use super::engram::{ngram_hashes_batched, DeepSeekV41Engram, EngramLayout, NgramHashState};
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

pub struct EngramRuntimeState {
    pub layout: EngramLayout,
    pub hash_state: NgramHashState,
}

impl DeepSeekV41Block {
    pub fn forward(
        &self,
        stream: &mut ResidualStream,
        start_pos: usize,
        shared: &mut SharedAttentionState,
        token_mask: Option<&[bool]>,
    ) {
        self.forward_with_masks(stream, start_pos, shared, token_mask, None, None)
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
        engram_hash_ids: Option<&[usize]>,
    ) {
        self.forward_inner(
            stream,
            start_pos,
            shared,
            engram_mask,
            image_mask,
            engram_hash_ids,
            None,
        )
    }

    pub fn forward_dspark(
        &self,
        stream: &mut ResidualStream,
        start_pos: usize,
        shared: &mut SharedAttentionState,
        main_x: &Tensor,
    ) {
        let main_prefix = if start_pos > 0 {
            Some(dspark_window_prefix(
                main_x,
                self.attn.window_size,
                start_pos,
            ))
        } else {
            None
        };
        self.forward_inner(
            stream,
            start_pos,
            shared,
            None,
            None,
            None,
            main_prefix.as_ref(),
        )
    }

    pub fn forward_dspark_with_prefix(
        &self,
        stream: &mut ResidualStream,
        start_pos: usize,
        shared: &mut SharedAttentionState,
        main_prefix: &Tensor,
    ) {
        self.forward_inner(
            stream,
            start_pos,
            shared,
            None,
            None,
            None,
            Some(main_prefix),
        )
    }

    fn forward_inner(
        &self,
        stream: &mut ResidualStream,
        start_pos: usize,
        shared: &mut SharedAttentionState,
        engram_mask: Option<&[bool]>,
        image_mask: Option<&[bool]>,
        engram_hash_ids: Option<&[usize]>,
        dspark_main_x: Option<&Tensor>,
    ) {
        if let Some(engram) = &self.engram {
            let x = stream.collapse_hc();
            let updated = if let Some(hash_ids) = engram_hash_ids {
                engram.forward_hashes(&x, hash_ids, engram_mask)
            } else {
                engram.forward_layer(
                    &x,
                    self.engram_key
                        .as_ref()
                        .expect("engram_key is required when engram is present"),
                    self.engram_value
                        .as_ref()
                        .expect("engram_value is required when engram is present"),
                    engram_mask,
                )
            };
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
                if let Some(main_x) = dspark_main_x {
                    self.attn.forward_layer_with_kv_source(
                        AttentionLayerInput {
                            x: &attn_in,
                            start_pos,
                            shared,
                        },
                        Some(main_x),
                    )
                } else {
                    self.attn.forward_layer(AttentionLayerInput {
                        x: &attn_in,
                        start_pos,
                        shared,
                    })
                }
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
    pub causal_encoder_layers: usize,
    pub decoder_layers: usize,
    pub embed_tokens: Tensor,
    pub layers: Vec<DeepSeekV41Block>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
    pub engram_runtime: Option<EngramRuntimeState>,

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
        self.require_engram_runtime()?;

        let h = embedding::embedding(ids, b, s, &self.embed_tokens, "deepseek.embed_tokens");
        let mut stream = ResidualStream::from_embedding(&h, self.hc_mult);
        let mut shared = SharedAttentionState::default();
        self.run_ced_layers(&mut stream, ids, b, s, 0, &mut shared, None, None);
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
        self.require_engram_runtime()?;

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
        self.run_ced_layers(
            &mut stream,
            ids,
            b,
            s,
            0,
            &mut shared,
            Some(&engram_mask),
            Some(&image_mask),
        );
        let collapsed = stream.collapse();
        let collapsed = norm::rms_norm(&collapsed, &self.final_norm, "deepseek.final_norm");
        Ok(linear::linear(
            &collapsed,
            &self.lm_head,
            "deepseek.lm_head",
        ))
    }

    pub fn encoder_decoder_split(&self) -> (usize, usize) {
        (self.causal_encoder_layers, self.decoder_layers)
    }

    fn require_engram_runtime(&self) -> Result<(), String> {
        if self.layers.iter().any(|layer| layer.engram.is_some()) && self.engram_runtime.is_none() {
            return Err(
                "DeepSeekV41TextModel requires engram_runtime when Engram layers are present"
                    .into(),
            );
        }
        Ok(())
    }

    fn run_ced_layers(
        &self,
        stream: &mut ResidualStream,
        ids: &[usize],
        b: usize,
        s: usize,
        start_pos: usize,
        shared: &mut SharedAttentionState,
        engram_mask: Option<&[bool]>,
        image_mask: Option<&[bool]>,
    ) {
        let engram_hashes = self.engram_runtime.as_ref().map(|runtime| {
            let mut state = runtime.hash_state.clone();
            ngram_hashes_batched(
                ids,
                b,
                s,
                start_pos,
                engram_mask,
                &runtime.layout,
                &mut state,
            )
        });
        let n_hash_cols = self
            .engram_runtime
            .as_ref()
            .map(|runtime| (runtime.layout.max_ngram_size - 1) * runtime.layout.n_heads)
            .unwrap_or(0);

        let split = self.causal_encoder_layers.min(self.layers.len());
        let (encoder, decoder) = self.layers.split_at(split);
        for layer in encoder {
            let layer_hash = engram_hash_slice(
                &self.engram_runtime,
                engram_hashes.as_deref(),
                n_hash_cols,
                b,
                s,
                layer.layer_id,
            );
            layer.forward_with_masks(
                stream,
                start_pos,
                shared,
                engram_mask,
                image_mask,
                layer_hash.as_deref(),
            );
        }
        if !decoder.is_empty() {
            shared.decoder_encoder_hidden = Some(stream.collapse());
        }
        for layer in decoder {
            let layer_hash = engram_hash_slice(
                &self.engram_runtime,
                engram_hashes.as_deref(),
                n_hash_cols,
                b,
                s,
                layer.layer_id,
            );
            layer.forward_with_masks(
                stream,
                start_pos,
                shared,
                engram_mask,
                image_mask,
                layer_hash.as_deref(),
            );
        }
    }
}

fn engram_hash_slice(
    runtime: &Option<EngramRuntimeState>,
    hashes: Option<&[usize]>,
    n_hash_cols: usize,
    batch: usize,
    seqlen: usize,
    layer_id: usize,
) -> Option<Vec<usize>> {
    let runtime = runtime.as_ref()?;
    let hashes = hashes?;
    let layer_index = runtime
        .layout
        .layer_ids
        .iter()
        .position(|&id| id == layer_id)?;
    let n_layers = runtime.layout.layer_ids.len();
    let rows = batch * seqlen;
    assert_eq!(
        hashes.len(),
        rows * n_layers * n_hash_cols,
        "Engram hash tensor shape mismatch"
    );
    let start = layer_index * n_hash_cols;
    let mut out = Vec::with_capacity(rows * n_hash_cols);
    for row in 0..rows {
        let row_base = row * n_layers * n_hash_cols;
        out.extend_from_slice(&hashes[row_base + start..row_base + start + n_hash_cols]);
    }
    Some(out)
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
    pub stage_swa_caches: RefCell<Vec<Option<Tensor>>>,
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
        // Compatibility path for the tracked tiny fixture generated before the
        // start_pos-aware decode API. `try_forward_spec_at` below is the faithful
        // upstream runtime surface.
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

    pub fn try_forward_spec_at(
        &self,
        input_ids: &[usize],
        main_hidden: &[f32],
        start_pos: usize,
    ) -> Result<Option<DsparkSpecOutput>, String> {
        if input_ids.is_empty() {
            return Err("forward_spec needs at least one accepted id".to_string());
        }
        let (embedded, main_x) = self.forward_embed(main_hidden, input_ids)?;
        let batch = input_ids.len();
        let main_x = Tensor::from_value_no_grad(TensorValue::from_vec(
            Shape(vec![batch, 1, self.dim]),
            main_x,
        ));
        if start_pos == 0 {
            let mut caches = Vec::with_capacity(self.stages.len());
            for stage in &self.stages {
                caches.push(Some(dspark_window_prefix(
                    &main_x,
                    stage.block.attn.window_size,
                    start_pos,
                )));
            }
            *self.stage_swa_caches.borrow_mut() = caches;
            return Ok(None);
        }

        let expanded = expand_hc(&embedded, self.hc_mult);
        let hc = shape_of(&expanded);
        let pre_mix = identity_pre_mix(hc.batch, hc.seqlen, hc.hc_mult);
        let mut stream = ResidualStream::from_hc_tensor(&expanded, pre_mix);
        let mut shared = SharedAttentionState::default();
        for (stage_id, stage) in self.stages.iter().enumerate() {
            shared.topk_indices = Some(
                decode_topk_indices(
                    stage.block.attn.window_size,
                    input_ids.len(),
                    self.block_size,
                    start_pos,
                )
                .into_iter()
                .map(|idx| usize::try_from(idx).unwrap_or(usize::MAX))
                .collect(),
            );
            let cached_prefix = self
                .stage_swa_caches
                .borrow()
                .get(stage_id)
                .and_then(Clone::clone)
                .unwrap_or_else(|| {
                    dspark_window_prefix(&main_x, stage.block.attn.window_size, start_pos)
                });
            stage.block.forward_dspark_with_prefix(
                &mut stream,
                start_pos,
                &mut shared,
                &cached_prefix,
            );
            if let Some(cache) = shared.swa_cache.as_ref() {
                let tensor = Tensor::from_value_no_grad(TensorValue::from_vec(
                    Shape(vec![cache.batch, cache.window_size, cache.head_dim]),
                    cache.data.clone(),
                ));
                if let Some(slot) = self.stage_swa_caches.borrow_mut().get_mut(stage_id) {
                    *slot = Some(tensor);
                }
            }
        }
        self.forward_head(&stream, input_ids).map(Some)
    }

    /// Panicking convenience wrapper over [`Self::try_forward_spec`] for tests
    /// and callers that treat a misconfigured head as a programmer error.
    pub fn forward_spec(&self, input_ids: &[usize], main_hidden: &[f32]) -> DsparkSpecOutput {
        self.try_forward_spec(input_ids, main_hidden)
            .expect("DeepSeekV41DsparkHead forward_spec failed")
    }
}

fn dspark_window_prefix(main_x: &Tensor, window_size: usize, start_pos: usize) -> Tensor {
    let value = main_x.inner.borrow().value.clone();
    let shape = value.shape.0;
    assert_eq!(shape.len(), 3, "DSpark main_x must be [B,1,D]");
    assert_eq!(shape[1], 1, "DSpark main_x must have one accepted token");
    let batch = shape[0];
    let dim = shape[2];
    let mut out = vec![0.0f32; batch * window_size * dim];
    let slot = start_pos % window_size;
    for b in 0..batch {
        let src = (b * dim)..((b + 1) * dim);
        let dst = (b * window_size + slot) * dim..(b * window_size + slot + 1) * dim;
        out[dst].copy_from_slice(&value.data.as_ref()[src]);
    }
    Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![batch, window_size, dim]),
        out,
    ))
}

/// Wrap an RMSNorm gamma slice as a no-grad `[dim]` tensor for [`norm::rms_norm`].
fn param_gamma(gamma: &[f32]) -> Tensor {
    Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![gamma.len()]),
        gamma.to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v41::attention::Csa2Mode;
    use crate::deepseek_v41::moe::{DeepSeekV41Expert, DeepSeekV41Gate};
    use crate::tensor::{Shape, TensorValue};

    fn param(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_value_no_grad(TensorValue::from_vec(Shape(shape.to_vec()), data))
    }

    fn tiny_block(layer_id: usize) -> DeepSeekV41Block {
        fn zero_expert() -> DeepSeekV41Expert {
            DeepSeekV41Expert {
                w1: vec![0.0, 0.0],
                w2: vec![0.0, 0.0],
                w3: vec![0.0, 0.0],
                dim: 2,
                inter_dim: 1,
                swiglu_limit: 0.0,
            }
        }
        let expert = zero_expert();
        let shared_expert = zero_expert();
        DeepSeekV41Block {
            layer_id,
            dim: 2,
            hc_mult: 1,
            hc_sinkhorn_iters: 0,
            hc_eps: 1e-6,
            attn_norm: param(&[2], vec![1.0, 1.0]),
            ffn_norm: param(&[2], vec![1.0, 1.0]),
            attn: DeepSeekV41Attention {
                n_heads: 1,
                head_dim: 2,
                rope_head_dim: 0,
                q_lora_rank: 2,
                o_lora_rank: 2,
                o_groups: 1,
                compress_ratio: 1,
                window_size: 4,
                rms_norm_eps: 1e-6,
                wq_a: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                q_norm: param(&[2], vec![1.0, 1.0]),
                wq_b: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                wkv: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                kv_norm: param(&[2], vec![1.0, 1.0]),
                wo_a: param(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                wo_b: param(&[2, 2], vec![0.0, 0.0, 0.0, 0.0]),
                attn_sink: param(&[1], vec![0.0]),
                layer_id,
                kv_source_layer_id: None,
                index_source_layer_id: None,
                csa2_mode: Csa2Mode::SlidingWindow,
                compressor: None,
                indexer: None,
            },
            ffn: DeepSeekV41MoE {
                gate: DeepSeekV41Gate {
                    weight: vec![0.0, 0.0],
                    correction_bias: vec![0.0],
                    bias_vl: None,
                    tokens: 2,
                    dim: 2,
                    experts: 1,
                    topk: 1,
                    gate_temp: 1.0,
                    norm_topk_prob: false,
                    route_scale: 1.0,
                },
                experts: vec![expert],
                shared_experts: shared_expert,
            },
            hc_attn_fn: vec![0.0; 6],
            hc_attn_base: vec![0.0; 3],
            hc_attn_scale: vec![1.0, 1.0, 1.0],
            hc_ffn_fn: vec![0.0; 6],
            hc_ffn_base: vec![0.0; 3],
            hc_ffn_scale: vec![1.0, 1.0, 1.0],
            engram: None,
            engram_key: None,
            engram_value: None,
        }
    }

    #[test]
    fn dspark_window_prefix_places_main_x_at_wrapped_slot() {
        let main_x = param(&[2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);

        let prefix = dspark_window_prefix(&main_x, 3, 4);

        assert_eq!(prefix.shape().0, vec![2, 3, 2]);
        assert_eq!(
            prefix.inner.borrow().value.data.as_ref(),
            &[0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        );
    }

    #[test]
    fn run_ced_layers_publishes_final_encoder_hidden_before_decoder() {
        let model = DeepSeekV41TextModel {
            vocab_size: 8,
            hidden_size: 2,
            hc_mult: 1,
            image_token_id: 7,
            causal_encoder_layers: 1,
            decoder_layers: 1,
            embed_tokens: param(&[8, 2], vec![0.0; 16]),
            layers: vec![tiny_block(0), tiny_block(1)],
            final_norm: param(&[2], vec![1.0, 1.0]),
            lm_head: param(&[2, 8], vec![0.0; 16]),
            engram_runtime: None,
            vision: None,
            image_start: None,
            image_end: None,
            image_newline: None,
        };
        let embed = param(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let mut stream = ResidualStream::from_embedding(&embed, 1);
        let mut shared = SharedAttentionState::default();

        model.run_ced_layers(&mut stream, &[0, 0], 1, 2, 0, &mut shared, None, None);

        let hidden = shared
            .decoder_encoder_hidden
            .as_ref()
            .expect("CED boundary should publish final encoder hidden");
        let value = hidden.inner.borrow().value.clone();
        assert_eq!(value.shape.0, vec![1, 2, 2]);
        assert_eq!(
            value.data.as_ref(),
            &[0.50000006, 1.0000001, 1.5000001, 2.0000002]
        );
        assert!(shared.consumed_decoder_encoder_hidden);
    }

    #[test]
    fn text_forward_uses_engram_hash_runtime_lookup() {
        fn embed_tokens() -> Tensor {
            param(
                &[8, 2],
                vec![
                    0.0, 0.0, 1.0, 1.0, 2.0, 0.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            )
        }

        let mut engram_block = tiny_block(0);
        engram_block.attn.attn_sink = param(&[1], vec![-8.0]);
        engram_block.engram = Some(DeepSeekV41Engram {
            q_weight: vec![1.0, 1.0],
            k_weight: vec![1.0, 1.0],
            embed_weight: Some(param(&[4, 1], vec![1.0, 2.0, 3.0, 4.0])),
            wkv_weight: Some(param(&[1, 4], vec![0.25, 0.5, 0.75, 1.0])),
            eps: 1e-6,
        });
        let mut hash_state = NgramHashState::new(vec![0, 1, 2, 3, 0, 0, 0, 0], 0, 1, 8);
        hash_state.set_multipliers(vec![1, 0]);
        let with_engram = DeepSeekV41TextModel {
            vocab_size: 8,
            hidden_size: 2,
            hc_mult: 1,
            image_token_id: 7,
            causal_encoder_layers: 1,
            decoder_layers: 0,
            embed_tokens: embed_tokens(),
            layers: vec![engram_block],
            final_norm: param(&[2], vec![1.0, 1.0]),
            lm_head: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            engram_runtime: Some(EngramRuntimeState {
                layout: EngramLayout {
                    max_ngram_size: 2,
                    layer_ids: vec![0],
                    num_embeddings: vec![4],
                    primes: vec![5],
                    offsets: vec![0],
                    n_heads: 1,
                    head_dim: 1,
                },
                hash_state,
            }),
            vision: None,
            image_start: None,
            image_end: None,
            image_newline: None,
        };
        let without_engram = DeepSeekV41TextModel {
            vocab_size: 8,
            hidden_size: 2,
            hc_mult: 1,
            image_token_id: 7,
            causal_encoder_layers: 1,
            decoder_layers: 0,
            embed_tokens: embed_tokens(),
            layers: vec![tiny_block(0)],
            final_norm: param(&[2], vec![1.0, 1.0]),
            lm_head: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            engram_runtime: None,
            vision: None,
            image_start: None,
            image_end: None,
            image_newline: None,
        };

        let logits = with_engram.forward_token_ids(&[1, 2], 1, 2);
        let base_logits = without_engram.forward_token_ids(&[1, 2], 1, 2);

        assert_ne!(
            logits.inner.borrow().value.data.as_ref(),
            base_logits.inner.borrow().value.data.as_ref(),
            "text model must apply Engram hash lookup in the forward path"
        );
    }

    #[test]
    fn text_forward_errors_when_engram_runtime_is_missing() {
        let mut block = tiny_block(0);
        block.engram = Some(DeepSeekV41Engram {
            q_weight: vec![1.0, 1.0],
            k_weight: vec![1.0, 1.0],
            embed_weight: Some(param(&[4, 1], vec![1.0, 2.0, 3.0, 4.0])),
            wkv_weight: Some(param(&[1, 4], vec![0.25, 0.5, 0.75, 1.0])),
            eps: 1e-6,
        });
        let model = DeepSeekV41TextModel {
            vocab_size: 8,
            hidden_size: 2,
            hc_mult: 1,
            image_token_id: 7,
            causal_encoder_layers: 1,
            decoder_layers: 0,
            embed_tokens: param(&[8, 2], vec![0.0; 16]),
            layers: vec![block],
            final_norm: param(&[2], vec![1.0, 1.0]),
            lm_head: param(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            engram_runtime: None,
            vision: None,
            image_start: None,
            image_end: None,
            image_newline: None,
        };
        let err = match model.try_forward_token_ids(&[1, 2], 1, 2) {
            Ok(_) => panic!("Engram layers need hash runtime state"),
            Err(err) => err,
        };
        assert!(
            err.contains("requires engram_runtime"),
            "unexpected err: {err}"
        );
    }
}
