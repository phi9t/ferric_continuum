//! DeepSeek V4.1 DSpark (MTP speculative-decoding) op helpers.
//!
//! These functions model the parameter-free math of the upstream DSpark heads
//! and stage plumbing so the verifier ladder can pin numeric parity before the
//! stage model and `forward_spec` are wired. They implement the Markov head, the
//! fp32 confidence head, the noise-token draft-input construction, and the
//! `main_proj`/`main_norm` seam that turns target-layer hiddens into the stage-0
//! embedding. Checkpoint loading and stage execution are later tickets.

use super::sparse::dspark_topk_indices;

/// DSpark Markov head over one token id. Mirrors `DSparkMarkovHead.forward`:
/// `embed` is `[vocab, rank]` (row per token id), `head` is `[vocab, rank]`
/// (`ParallelHead` weight, applied as `logits[v] = sum_r embed[r] * head[v, r]`).
/// Returns `(logits[vocab], markov_embed[rank])`.
pub fn markov_head_forward(
    token_id: usize,
    embed: &[f32],
    head: &[f32],
    vocab_size: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(rank > 0, "markov rank must be positive");
    assert_eq!(
        embed.len(),
        vocab_size * rank,
        "embed shape must be [vocab, rank]"
    );
    assert_eq!(
        head.len(),
        vocab_size * rank,
        "head shape must be [vocab, rank]"
    );
    assert!(token_id < vocab_size, "token id out of vocab range");

    let markov_embed = embed[token_id * rank..(token_id + 1) * rank].to_vec();
    let mut logits = vec![0.0f32; vocab_size];
    for (v, logit) in logits.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let w = &head[v * rank..(v + 1) * rank];
        for r in 0..rank {
            acc += markov_embed[r] * w[r];
        }
        *logit = acc;
    }
    (logits, markov_embed)
}

/// DSpark fp32 confidence head. Mirrors `DSparkConfidenceHead.forward`: a linear
/// projection `[dim + rank] -> 1` over `concat(hidden, markov_embed)`, returned
/// as a scalar. `proj` is `[1, dim + rank]` stored row-major.
pub fn confidence_head_forward(hidden: &[f32], markov_embed: &[f32], proj: &[f32]) -> f32 {
    let input_dim = hidden.len() + markov_embed.len();
    assert_eq!(
        proj.len(),
        input_dim,
        "confidence proj shape must be [1, dim + rank]"
    );
    let mut acc = 0.0f32;
    for (i, &w) in proj.iter().enumerate() {
        let x = if i < hidden.len() {
            hidden[i]
        } else {
            markov_embed[i - hidden.len()]
        };
        acc += x * w;
    }
    acc
}

/// Build the DSpark stage-0 draft input ids. Mirrors `forward_embed`: a
/// `[batch, block_size]` grid filled with `noise_token_id`, with column 0 set to
/// the accepted `input_ids` (one per batch row). Returned row-major.
pub fn draft_input_ids(
    input_ids: &[usize],
    batch: usize,
    block_size: usize,
    noise_token_id: usize,
) -> Vec<usize> {
    assert_eq!(input_ids.len(), batch, "one input id per batch row");
    assert!(block_size > 0, "block_size must be positive");
    let mut out = vec![noise_token_id; batch * block_size];
    for (b, &id) in input_ids.iter().enumerate() {
        out[b * block_size] = id;
    }
    out
}

/// DSpark `main_proj` + `main_norm` seam. Mirrors `forward_embed`'s
/// `main_norm(main_proj(main_hidden))`: a linear `[in_dim] -> [dim]` per row
/// followed by an RMSNorm over `dim`. `main_hidden` is `[rows, in_dim]` where
/// `in_dim = dim * len(target_layer_ids)`; `proj` is `[dim, in_dim]` row-major
/// (upstream `Linear` weight, applied as `y[o] = sum_i x[i] * proj[o, i]`).
pub fn main_proj_norm(
    main_hidden: &[f32],
    proj: &[f32],
    norm_weight: &[f32],
    in_dim: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    assert!(in_dim > 0 && dim > 0, "dims must be positive");
    assert_eq!(
        proj.len(),
        dim * in_dim,
        "main_proj shape must be [dim, in_dim]"
    );
    assert_eq!(norm_weight.len(), dim, "main_norm shape must be [dim]");
    assert_eq!(
        main_hidden.len() % in_dim,
        0,
        "main_hidden length must be rows * in_dim"
    );
    let rows = main_hidden.len() / in_dim;
    let mut out = Vec::with_capacity(rows * dim);
    for row in 0..rows {
        let x = &main_hidden[row * in_dim..(row + 1) * in_dim];
        let mut projected = vec![0.0f32; dim];
        for (o, proj_out) in projected.iter_mut().enumerate() {
            let w = &proj[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += x[i] * w[i];
            }
            *proj_out = acc;
        }
        let mean_square = projected.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for (v, w) in projected.iter().zip(norm_weight) {
            out.push(v * scale * w);
        }
    }
    out
}

/// Argmax over a logits row, mirroring upstream `sample(logits, temperature=0)`
/// (the DSpark draft loop forces greedy acceptance in the parity harness).
pub fn argmax(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "argmax over empty logits");
    let mut best = 0usize;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// One DSpark `forward_head` draft loop over `block_size` positions. Mirrors the
/// loop in `DSparkBlock.forward_head`: starting from `input_id`, each position
/// adds the Markov bias for the current output id to the base head logits,
/// argmax-samples the next id, and stacks the Markov embeddings; the confidence
/// head then scores `concat(hidden[i], markov_embed[i])` per position.
///
/// - `base_logits`: `[block_size, vocab]` head logits (row per draft position).
/// - `hidden`: `[block_size, dim]` post-`hc_pre` hidden states.
/// - Returns `(output_ids[block_size + 1], biased_logits[block_size*vocab],
///   confidence[block_size])`.
pub struct DraftLoopOutput {
    pub output_ids: Vec<usize>,
    pub logits: Vec<f32>,
    pub confidence: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn draft_loop(
    input_id: usize,
    base_logits: &[f32],
    hidden: &[f32],
    markov_embed: &[f32],
    markov_head_weight: &[f32],
    confidence_proj: &[f32],
    block_size: usize,
    vocab_size: usize,
    rank: usize,
    dim: usize,
) -> DraftLoopOutput {
    assert_eq!(
        base_logits.len(),
        block_size * vocab_size,
        "base_logits shape"
    );
    assert_eq!(hidden.len(), block_size * dim, "hidden shape");

    let mut logits = base_logits.to_vec();
    let mut output_ids = vec![0usize; block_size + 1];
    output_ids[0] = input_id;
    let mut markov_embeds: Vec<f32> = Vec::with_capacity(block_size * rank);

    for i in 0..block_size {
        let (bias, embed) = markov_head_forward(
            output_ids[i],
            markov_embed,
            markov_head_weight,
            vocab_size,
            rank,
        );
        let row = &mut logits[i * vocab_size..(i + 1) * vocab_size];
        for (l, b) in row.iter_mut().zip(&bias) {
            *l += b;
        }
        markov_embeds.extend_from_slice(&embed);
        output_ids[i + 1] = argmax(row);
    }

    let mut confidence = Vec::with_capacity(block_size);
    for i in 0..block_size {
        let h = &hidden[i * dim..(i + 1) * dim];
        let m = &markov_embeds[i * rank..(i + 1) * rank];
        confidence.push(confidence_head_forward(h, m, confidence_proj));
    }

    DraftLoopOutput {
        output_ids,
        logits,
        confidence,
    }
}

/// Re-export so DSpark call sites and tests can reach the decode index math from
/// the DSpark module namespace without importing `sparse` directly.
pub fn decode_topk_indices(
    window_size: usize,
    batch: usize,
    block_size: usize,
    start_pos: usize,
) -> Vec<i32> {
    dspark_topk_indices(window_size, batch, block_size, start_pos)
}
