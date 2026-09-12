//! DeepSeek V4.1 Hyper-Connection math helpers.
//!
//! This module implements the core verifier-seam operations around
//! `Block.hc_mixes`, `hc_pre`, and `hc_post` from upstream `model.py`.
//! It is intentionally pure math over `Vec<f32>`.
//!
//! `HcShape` (the residual-stream layout descriptor) now lives in
//! [`super::hc_tensor`] and is re-exported here for existing importers.

pub use super::hc_tensor::HcShape;

#[derive(Debug, Clone, PartialEq)]
pub struct HcMixes {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    pub comb_row_sums: Vec<f32>,
    pub comb_col_sums: Vec<f32>,
}

pub fn hc_mixes(
    flat_hc: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    hc_mult: usize,
    dim: usize,
    iters: usize,
    eps: f32,
) -> HcMixes {
    assert_eq!(hc_scale.len(), 3, "hc_scale must be length 3");
    assert!(hc_mult > 0, "hc_mult must be positive");
    assert!(dim > 0, "dim must be positive");
    assert!(eps > 0.0, "eps must be positive");

    let tokens = flat_hc
        .len()
        .checked_div(hc_mult * dim)
        .expect("flat_hc shape");
    assert_eq!(
        flat_hc.len(),
        tokens * hc_mult * dim,
        "flat_hc must be [tokens, hc_mult, dim]"
    );

    let mix_hc = (2 + hc_mult) * hc_mult;
    assert_eq!(
        hc_fn.len(),
        mix_hc * (hc_mult * dim),
        "hc_fn shape mismatch"
    );
    assert_eq!(hc_base.len(), mix_hc, "hc_base shape mismatch");

    // Upstream:
    // x = x.flatten(2).float()
    // rsqrt = rsqrt(mean(x^2) + norm_eps)
    // mixes = linear(x, hc_fn) * rsqrt
    // pre/post/comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    let mut mixes = vec![0.0f32; tokens * mix_hc];
    for t in 0..tokens {
        let x_base = t * hc_mult * dim;
        let mut mean_sq = 0.0f32;
        for v in &flat_hc[x_base..x_base + hc_mult * dim] {
            mean_sq += v * v;
        }
        mean_sq /= (hc_mult * dim) as f32;
        let rsqrt = 1.0 / (mean_sq + eps).sqrt();

        for out_idx in 0..mix_hc {
            let w_base = out_idx * hc_mult * dim;
            let mut acc = 0.0f32;
            for i in 0..hc_mult * dim {
                acc += flat_hc[x_base + i] * hc_fn[w_base + i];
            }
            mixes[t * mix_hc + out_idx] = acc * rsqrt;
        }
    }

    split_sinkhorn(&mixes, hc_scale, hc_base, hc_mult, iters, eps)
}

pub fn hc_pre(x: &[f32], pre_mix: &[f32], shape: HcShape) -> Vec<f32> {
    assert_eq!(
        x.len(),
        shape.batch * shape.seqlen * shape.hc_mult * shape.dim,
        "x length does not match shape"
    );
    assert_eq!(
        pre_mix.len(),
        shape.batch * shape.seqlen * shape.hc_mult,
        "pre_mix length does not match shape"
    );
    let mut out = vec![0.0f32; shape.batch * shape.seqlen * shape.dim];
    for b in 0..shape.batch {
        for s in 0..shape.seqlen {
            let token = b * shape.seqlen + s;
            let x_base = token * shape.hc_mult * shape.dim;
            let mix_base = token * shape.hc_mult;
            let out_base = token * shape.dim;
            for h in 0..shape.hc_mult {
                let coeff = pre_mix[mix_base + h];
                for d in 0..shape.dim {
                    out[out_base + d] += coeff * x[x_base + h * shape.dim + d];
                }
            }
        }
    }
    out
}

pub fn hc_post(
    sublayer: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    shape: HcShape,
) -> Vec<f32> {
    assert_eq!(
        sublayer.len(),
        shape.batch * shape.seqlen * shape.dim,
        "sublayer length does not match shape"
    );
    assert_eq!(
        residual.len(),
        shape.batch * shape.seqlen * shape.hc_mult * shape.dim,
        "residual length does not match shape"
    );
    assert_eq!(
        post.len(),
        shape.batch * shape.seqlen * shape.hc_mult,
        "post length does not match shape"
    );
    assert_eq!(
        comb.len(),
        shape.batch * shape.seqlen * shape.hc_mult * shape.hc_mult,
        "comb length does not match shape"
    );

    // Upstream:
    // y = post[...,hc] * x + sum_{src} comb[dst,src] * residual[src]
    let mut out = vec![0.0f32; shape.batch * shape.seqlen * shape.hc_mult * shape.dim];
    for b in 0..shape.batch {
        for s in 0..shape.seqlen {
            let token = b * shape.seqlen + s;
            let sub_base = token * shape.dim;
            let res_base = token * shape.hc_mult * shape.dim;
            let post_base = token * shape.hc_mult;
            let comb_base = token * shape.hc_mult * shape.hc_mult;
            for dst in 0..shape.hc_mult {
                let coeff_post = post[post_base + dst];
                let out_base = res_base + dst * shape.dim;
                for d in 0..shape.dim {
                    out[out_base + d] = coeff_post * sublayer[sub_base + d];
                }
                for src in 0..shape.hc_mult {
                    let coeff = comb[comb_base + dst * shape.hc_mult + src];
                    if coeff == 0.0 {
                        continue;
                    }
                    let src_base = res_base + src * shape.dim;
                    for d in 0..shape.dim {
                        out[out_base + d] += coeff * residual[src_base + d];
                    }
                }
            }
        }
    }
    out
}

fn split_sinkhorn(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    hc_mult: usize,
    iters: usize,
    eps: f32,
) -> HcMixes {
    let tokens = mixes.len() / ((2 + hc_mult) * hc_mult);
    let mix_hc = (2 + hc_mult) * hc_mult;
    assert_eq!(mixes.len(), tokens * mix_hc, "mixes shape mismatch");

    let mut pre = vec![0.0f32; tokens * hc_mult];
    let mut post = vec![0.0f32; tokens * hc_mult];
    let mut comb = vec![0.0f32; tokens * hc_mult * hc_mult];

    // Matches upstream `kernel.hc_split_sinkhorn` semantics (see
    // `ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/kernel.py`):
    // - pre = sigmoid(mixes[:hc] * hc_scale[0] + hc_base[:hc]) + eps
    // - post = 2 * sigmoid(mixes[hc:2hc] * hc_scale[1] + hc_base[hc:2hc])
    // - comb logits = mixes[2hc:] * hc_scale[2] + hc_base[2hc:]
    // - comb = softmax(-1) + eps
    // - comb column-normalize once
    // - then (iters-1) rounds of row-normalize then column-normalize.
    let scale_pre = scale[0];
    let scale_post = scale[1];
    let scale_comb = scale[2];

    for t in 0..tokens {
        let base_row = t * mix_hc;
        let pre_row = t * hc_mult;
        let post_row = t * hc_mult;
        let comb_row = t * hc_mult * hc_mult;

        for h in 0..hc_mult {
            pre[pre_row + h] = sigmoid(mixes[base_row + h] * scale_pre + base[h]) + eps;
            post[post_row + h] =
                2.0 * sigmoid(mixes[base_row + hc_mult + h] * scale_post + base[hc_mult + h]);
        }

        let comb_base = base_row + 2 * hc_mult;
        let comb_bias_base = 2 * hc_mult;
        for dst in 0..hc_mult {
            for src in 0..hc_mult {
                let flat = dst * hc_mult + src;
                comb[comb_row + flat] =
                    mixes[comb_base + flat] * scale_comb + base[comb_bias_base + flat];
            }
        }

        // Normalize comb to be approximately doubly stochastic via Sinkhorn.
        sinkhorn_in_place(
            &mut comb[comb_row..comb_row + hc_mult * hc_mult],
            hc_mult,
            iters,
            eps,
        );
    }

    let (comb_row_sums, comb_col_sums) = comb_sums(&comb, tokens, hc_mult);
    HcMixes {
        pre,
        post,
        comb,
        comb_row_sums,
        comb_col_sums,
    }
}

fn sinkhorn_in_place(matrix: &mut [f32], n: usize, iters: usize, eps: f32) {
    assert_eq!(matrix.len(), n * n, "matrix must be n*n");
    // comb = softmax(-1) + eps
    for r in 0..n {
        let row_base = r * n;
        let mut max = f32::NEG_INFINITY;
        for c in 0..n {
            max = max.max(matrix[row_base + c]);
        }
        let mut sum_exp = 0.0f32;
        for c in 0..n {
            let v = (matrix[row_base + c] - max).exp();
            matrix[row_base + c] = v;
            sum_exp += v;
        }
        for c in 0..n {
            matrix[row_base + c] = matrix[row_base + c] / sum_exp + eps;
        }
    }

    // comb = comb / (comb.sum(-2) + eps)
    column_normalize_in_place(matrix, n, eps);

    for _ in 0..iters.saturating_sub(1) {
        // comb = comb / (comb.sum(-1) + eps)
        row_normalize_in_place(matrix, n, eps);
        // comb = comb / (comb.sum(-2) + eps)
        column_normalize_in_place(matrix, n, eps);
    }
}

fn comb_sums(comb: &[f32], tokens: usize, hc_mult: usize) -> (Vec<f32>, Vec<f32>) {
    let mut row_sums = vec![0.0f32; tokens * hc_mult];
    let mut col_sums = vec![0.0f32; tokens * hc_mult];
    for t in 0..tokens {
        let base = t * hc_mult * hc_mult;
        for r in 0..hc_mult {
            let mut sum = 0.0f32;
            for c in 0..hc_mult {
                sum += comb[base + r * hc_mult + c];
            }
            row_sums[t * hc_mult + r] = sum;
        }
        for c in 0..hc_mult {
            let mut sum = 0.0f32;
            for r in 0..hc_mult {
                sum += comb[base + r * hc_mult + c];
            }
            col_sums[t * hc_mult + c] = sum;
        }
    }
    (row_sums, col_sums)
}

fn row_normalize_in_place(matrix: &mut [f32], n: usize, eps: f32) {
    for r in 0..n {
        let row_base = r * n;
        let mut sum = 0.0f32;
        for c in 0..n {
            sum += matrix[row_base + c];
        }
        let denom = sum + eps;
        for c in 0..n {
            matrix[row_base + c] /= denom;
        }
    }
}

fn column_normalize_in_place(matrix: &mut [f32], n: usize, eps: f32) {
    let mut sums = vec![0.0f32; n];
    for c in 0..n {
        let mut sum = 0.0f32;
        for r in 0..n {
            sum += matrix[r * n + c];
        }
        sums[c] = sum;
    }
    for c in 0..n {
        let denom = sums[c] + eps;
        for r in 0..n {
            matrix[r * n + c] /= denom;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
