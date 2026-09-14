//! DeepSeek V4.1 sparse-attention fixture helpers.
//!
//! These functions model upstream index and compression math only. They do not
//! implement attention, cache mutation, quantization, or model block execution.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateShape {
    pub batch: usize,
    pub seqlen: usize,
    pub positions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateLens {
    Scalar(usize),
    PerQuery(Vec<i32>),
}

pub fn window_topk_indices(
    window_size: usize,
    batch: usize,
    seqlen: usize,
    start_pos: usize,
) -> Vec<i32> {
    assert!(window_size > 0, "window_size must be positive");
    let rows = if start_pos == 0 { seqlen } else { 1 };
    let width = if start_pos == 0 {
        seqlen.min(window_size)
    } else {
        window_size
    };
    let mut one_batch = Vec::with_capacity(rows * width);

    if start_pos == 0 {
        for end in 0..seqlen {
            let row_start = (end + 1).saturating_sub(window_size);
            for slot in 0..width {
                let idx = row_start + slot;
                one_batch.push(if idx > end { -1 } else { idx as i32 });
            }
        }
    } else {
        let oldest = start_pos % window_size + 1;
        for idx in oldest..window_size {
            one_batch.push(if idx > start_pos { -1 } else { idx as i32 });
        }
        for idx in 0..oldest {
            one_batch.push(if idx > start_pos { -1 } else { idx as i32 });
        }
    }

    let mut out = Vec::with_capacity(batch * one_batch.len());
    for _ in 0..batch {
        out.extend_from_slice(&one_batch);
    }
    out
}

/// DSpark decode-step candidate indices, mirroring upstream
/// `get_dspark_topk_idxs`. The row concatenates the window ring positions
/// `0..min(window_size, start_pos+1)` with the freshly written draft positions
/// `window_size + (0..block_size)`, then expands over `[batch, block_size, -1]`.
/// Upstream asserts `start_pos > 0` because prefill only seeds the window cache.
pub fn dspark_topk_indices(
    window_size: usize,
    batch: usize,
    block_size: usize,
    start_pos: usize,
) -> Vec<i32> {
    assert!(window_size > 0, "window_size must be positive");
    assert!(block_size > 0, "block_size must be positive");
    assert!(start_pos > 0, "DSpark decode requires start_pos > 0");

    let window_rows = window_size.min(start_pos + 1);
    let mut row = Vec::with_capacity(window_rows + block_size);
    for idx in 0..window_rows {
        row.push(idx as i32);
    }
    for idx in 0..block_size {
        row.push((window_size + idx) as i32);
    }

    let mut out = Vec::with_capacity(batch * block_size * row.len());
    for _ in 0..batch {
        for _ in 0..block_size {
            out.extend_from_slice(&row);
        }
    }
    out
}

pub fn select_candidate_blocks(
    logits: &[f32],
    shape: CandidateShape,
    compress_lens: CandidateLens,
    topk_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    assert!(block_size > 0, "block_size must be positive");
    assert_eq!(
        logits.len(),
        shape.batch * shape.seqlen * shape.positions,
        "candidate logits length does not match shape"
    );
    assert!(
        logits.iter().all(|v| !v.is_nan()),
        "candidate logits must not contain NaN"
    );
    let num_blocks = shape.positions.div_ceil(block_size);
    let keep_count = topk_blocks.min(num_blocks);
    let mut out = vec![false; logits.len()];

    for b in 0..shape.batch {
        for s in 0..shape.seqlen {
            let query = b * shape.seqlen + s;
            let lens = match &compress_lens {
                CandidateLens::Scalar(lens) => *lens as i32,
                CandidateLens::PerQuery(lens) => {
                    assert_eq!(
                        lens.len(),
                        shape.batch * shape.seqlen,
                        "compress_lens length must match batch*seqlen"
                    );
                    lens[query]
                }
            };
            assert!(lens >= 0, "compress_lens must be non-negative");
            let last = if lens > 0 {
                Some(((lens - 1) as usize) / block_size)
            } else {
                None
            };
            let row_base = query * shape.positions;
            let mut scores = Vec::with_capacity(num_blocks);
            for block in 0..num_blocks {
                let mut best = f32::NEG_INFINITY;
                for offset in 0..block_size {
                    let pos = block * block_size + offset;
                    if pos < shape.positions {
                        best = best.max(logits[row_base + pos]);
                    }
                }
                if Some(block) == last {
                    best = f32::INFINITY;
                }
                scores.push((block, best));
            }

            scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            for &(block, score) in scores.iter().take(keep_count) {
                if score > f32::NEG_INFINITY {
                    for offset in 0..block_size {
                        let pos = block * block_size + offset;
                        if pos < shape.positions {
                            out[row_base + pos] = true;
                        }
                    }
                }
            }
        }
    }

    out
}

pub fn compress_ratio_one(kv: &[f32], norm_weight: &[f32], eps: f32) -> Vec<f32> {
    assert!(!norm_weight.is_empty(), "norm_weight must not be empty");
    assert_eq!(
        kv.len() % norm_weight.len(),
        0,
        "kv length must be a multiple of norm dim"
    );
    rms_norm_rows(kv, norm_weight, eps)
}

pub fn compress_ratio_n(
    kv: &[f32],
    score: &[f32],
    ratio: usize,
    norm_weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert!(ratio > 1, "ratio_n path requires ratio > 1");
    assert_eq!(kv.len(), score.len(), "kv and score shapes must match");
    let dim = norm_weight.len();
    assert!(dim > 0, "norm_weight must not be empty");
    assert_eq!(kv.len() % dim, 0, "kv length must be rows * dim");
    let rows = kv.len() / dim;
    let complete_groups = rows / ratio;
    let mut pooled = Vec::with_capacity(complete_groups * dim);

    for group in 0..complete_groups {
        for d in 0..dim {
            let mut max_score = f32::NEG_INFINITY;
            for r in 0..ratio {
                max_score = max_score.max(score[(group * ratio + r) * dim + d]);
            }

            let mut denom = 0.0f32;
            let mut numer = 0.0f32;
            for r in 0..ratio {
                let idx = (group * ratio + r) * dim + d;
                let weight = (score[idx] - max_score).exp();
                denom += weight;
                numer += kv[idx] * weight;
            }
            pooled.push(numer / denom);
        }
    }

    rms_norm_rows(&pooled, norm_weight, eps)
}

fn rms_norm_rows(rows: &[f32], norm_weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = norm_weight.len();
    let mut out = Vec::with_capacity(rows.len());
    for row in rows.chunks_exact(dim) {
        let mean_square = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for (v, w) in row.iter().zip(norm_weight) {
            out.push(v * scale * w);
        }
    }
    out
}
