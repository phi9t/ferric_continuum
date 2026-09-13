//! DeepSeek V4.1 vision tower (ViT) math helpers.
//!
//! Pure functions mirroring `vision.py`. The ViT is full bidirectional
//! attention over one image's patches with 2D RoPE. The 2D RoPE here is
//! *half-split* (`chunk(2)`) and deliberately does not reuse the text
//! adjacent-pair RoPE. Vision RMSNorm uses eps `1e-6`, distinct from the text
//! `1e-20`.

/// Vision RMSNorm epsilon (upstream `RMSNorm` default).
pub const VISION_RMS_EPS: f32 = 1e-6;

/// 2D RoPE tables for an `n_h x n_w` patch grid.
///
/// Mirrors `get_vision_cos_sin`: `inv_freq` over `arange(0, dim, 2) / dim`,
/// per-patch `(hpos, wpos)` frequencies stacked then flattened. Returns
/// `(cos, sin)` each of length `n_h * n_w * dim` laid out `[patch, dim]`, where
/// `dim == rope_dim` (the per-half rotary width `vision_dim / n_heads / 2`).
pub fn vision_cos_sin(n_h: usize, n_w: usize, dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    assert!(dim % 2 == 0, "rope dim must be even");
    let half = dim / 2;
    // inv_freq[i] = 1 / theta^((2i)/dim) for i in 0..half
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / dim as f64))
        .collect();

    let n_patch = n_h * n_w;
    let mut cos = vec![0.0f32; n_patch * dim];
    let mut sin = vec![0.0f32; n_patch * dim];
    for h in 0..n_h {
        for w in 0..n_w {
            let patch = h * n_w + w;
            // freqs = [hpos * inv_freq, wpos * inv_freq] flattened -> length dim.
            for i in 0..half {
                let fh = (h as f64) * inv_freq[i];
                cos[patch * dim + i] = fh.cos() as f32;
                sin[patch * dim + i] = fh.sin() as f32;
            }
            for i in 0..half {
                let fw = (w as f64) * inv_freq[i];
                cos[patch * dim + half + i] = fw.cos() as f32;
                sin[patch * dim + half + i] = fw.sin() as f32;
            }
        }
    }
    (cos, sin)
}

/// Apply the half-split 2D RoPE to `x` laid out `[n, n_heads, head_dim]`.
///
/// `cos`/`sin` are `[n, rope_dim]` where `rope_dim == head_dim / 2`; the same
/// table is broadcast across heads. Mirrors `apply_rotary`:
/// `x1, x2 = x.chunk(2)`; `cat([x1*cos - x2*sin, x2*cos + x1*sin])`.
pub fn apply_rotary_half_split(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    n: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert!(head_dim % 2 == 0, "head_dim must be even");
    let rope_dim = head_dim / 2;
    assert_eq!(x.len(), n * n_heads * head_dim, "x length mismatch");
    assert_eq!(cos.len(), n * rope_dim, "cos length mismatch");
    assert_eq!(sin.len(), n * rope_dim, "sin length mismatch");

    let mut out = vec![0.0f32; x.len()];
    for p in 0..n {
        for head in 0..n_heads {
            let base = (p * n_heads + head) * head_dim;
            let rbase = p * rope_dim;
            for i in 0..rope_dim {
                let x1 = x[base + i];
                let x2 = x[base + rope_dim + i];
                let c = cos[rbase + i];
                let s = sin[rbase + i];
                out[base + i] = x1 * c - x2 * s;
                out[base + rope_dim + i] = x2 * c + x1 * s;
            }
        }
    }
    out
}

/// Vision RMSNorm over the last dim, mirroring `vision.RMSNorm.forward`.
/// `x` is `[rows, dim]`, `weight` is `[dim]`.
pub fn vision_rms_norm(x: &[f32], weight: &[f32], eps: f32, rows: usize, dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * dim, "x length mismatch");
    assert_eq!(weight.len(), dim, "weight length mismatch");
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let base = r * dim;
        let mut sum_sq = 0.0f32;
        for d in 0..dim {
            sum_sq += x[base + d] * x[base + d];
        }
        let inv = (sum_sq / dim as f32 + eps).sqrt().recip();
        for d in 0..dim {
            out[base + d] = weight[d] * (x[base + d] * inv);
        }
    }
    out
}

/// Row-wise linear `y = x @ w^T + b` where `w` is torch layout `[out, in]`.
/// `x` is `[rows, in]`, optional `b` is `[out]`; returns `[rows, out]`.
pub fn vision_linear(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_dim, "x length mismatch");
    assert_eq!(w.len(), out_dim * in_dim, "w length mismatch");
    if let Some(b) = b {
        assert_eq!(b.len(), out_dim, "bias length mismatch");
    }
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = b.map_or(0.0, |b| b[o]);
            let wb = o * in_dim;
            let xb = r * in_dim;
            for i in 0..in_dim {
                acc += x[xb + i] * w[wb + i];
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Patch embedding: flatten each patch and project. `patches` is
/// `[n_patch, 3*patch^2]`, `proj_w` is `[dim, 3*patch^2]`, `proj_b` is `[dim]`.
pub fn vision_patch_embed(
    patches: &[f32],
    proj_w: &[f32],
    proj_b: &[f32],
    n_patch: usize,
    patch_flat: usize,
    dim: usize,
) -> Vec<f32> {
    vision_linear(patches, proj_w, Some(proj_b), n_patch, patch_flat, dim)
}

/// Full bidirectional self-attention over all patches with 2D RoPE on q,k.
///
/// `x` is `[n, dim]`. `wqkv` is `[3*dim, dim]`, `wqkv_b` is `[3*dim]`, `wo` is
/// `[dim, dim]`, `wo_b` is `[dim]`. Mirrors `vision.Attention.forward`: split
/// qkv, RoPE q/k, scaled dot-product attention (softmax over all patches, no
/// mask), project out.
#[allow(clippy::too_many_arguments)]
pub fn vision_attention(
    x: &[f32],
    wqkv: &[f32],
    wqkv_b: &[f32],
    wo: &[f32],
    wo_b: &[f32],
    cos: &[f32],
    sin: &[f32],
    n: usize,
    dim: usize,
    n_heads: usize,
) -> Vec<f32> {
    assert_eq!(dim % n_heads, 0, "dim must be divisible by n_heads");
    let head_dim = dim / n_heads;
    let qkv = vision_linear(x, wqkv, Some(wqkv_b), n, dim, 3 * dim);

    // Split into q, k, v each [n, n_heads, head_dim].
    let mut q = vec![0.0f32; n * dim];
    let mut k = vec![0.0f32; n * dim];
    let mut v = vec![0.0f32; n * dim];
    for p in 0..n {
        let src = p * 3 * dim;
        q[p * dim..(p + 1) * dim].copy_from_slice(&qkv[src..src + dim]);
        k[p * dim..(p + 1) * dim].copy_from_slice(&qkv[src + dim..src + 2 * dim]);
        v[p * dim..(p + 1) * dim].copy_from_slice(&qkv[src + 2 * dim..src + 3 * dim]);
    }
    let q = apply_rotary_half_split(&q, cos, sin, n, n_heads, head_dim);
    let k = apply_rotary_half_split(&k, cos, sin, n, n_heads, head_dim);

    let scale = (head_dim as f32).powf(-0.5);
    // Output [n, n_heads, head_dim] in the same [n, dim] layout.
    let mut attn_out = vec![0.0f32; n * dim];
    for head in 0..n_heads {
        for i in 0..n {
            // scores over all j, softmax-stable.
            let mut scores = vec![0.0f32; n];
            let qb = i * dim + head * head_dim;
            for j in 0..n {
                let kb = j * dim + head * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qb + d] * k[kb + d];
                }
                scores[j] = dot * scale;
            }
            let mut max = f32::NEG_INFINITY;
            for &s in &scores {
                if s > max {
                    max = s;
                }
            }
            let mut denom = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                denom += *s;
            }
            let ob = i * dim + head * head_dim;
            for j in 0..n {
                let wgt = scores[j] / denom;
                let vb = j * dim + head * head_dim;
                for d in 0..head_dim {
                    attn_out[ob + d] += wgt * v[vb + d];
                }
            }
        }
    }
    vision_linear(&attn_out, wo, Some(wo_b), n, dim, dim)
}

/// Chunked SwiGLU MLP with no bias. `x` is `[n, dim]`, `w1` is
/// `[2*inter, dim]`, `w2` is `[dim, inter]`. Mirrors `vision.MLP.forward`:
/// `gate, up = w1(x).chunk(2)`; `w2(silu(gate) * up)`.
pub fn vision_mlp(
    x: &[f32],
    w1: &[f32],
    w2: &[f32],
    n: usize,
    dim: usize,
    inter: usize,
) -> Vec<f32> {
    let hidden = vision_linear(x, w1, None, n, dim, 2 * inter);
    let mut act = vec![0.0f32; n * inter];
    for r in 0..n {
        let hb = r * 2 * inter;
        for i in 0..inter {
            let gate = hidden[hb + i];
            let up = hidden[hb + inter + i];
            act[r * inter + i] = silu(gate) * up;
        }
    }
    vision_linear(&act, w2, None, n, inter, dim)
}

/// Weights for a single vision block.
pub struct VisionBlockWeights<'a> {
    pub norm1: &'a [f32],
    pub wqkv: &'a [f32],
    pub wqkv_b: &'a [f32],
    pub wo: &'a [f32],
    pub wo_b: &'a [f32],
    pub norm2: &'a [f32],
    pub w1: &'a [f32],
    pub w2: &'a [f32],
}

/// Pre-norm residual block: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`.
#[allow(clippy::too_many_arguments)]
pub fn vision_block(
    x: &[f32],
    weights: &VisionBlockWeights,
    cos: &[f32],
    sin: &[f32],
    n: usize,
    dim: usize,
    n_heads: usize,
    inter: usize,
) -> Vec<f32> {
    let normed = vision_rms_norm(x, weights.norm1, VISION_RMS_EPS, n, dim);
    let attn = vision_attention(
        &normed,
        weights.wqkv,
        weights.wqkv_b,
        weights.wo,
        weights.wo_b,
        cos,
        sin,
        n,
        dim,
        n_heads,
    );
    let mut h = vec![0.0f32; n * dim];
    for i in 0..n * dim {
        h[i] = x[i] + attn[i];
    }
    let normed2 = vision_rms_norm(&h, weights.norm2, VISION_RMS_EPS, n, dim);
    let mlp = vision_mlp(&normed2, weights.w1, weights.w2, n, dim, inter);
    for i in 0..n * dim {
        h[i] += mlp[i];
    }
    h
}

// ---------------------------------------------------------------------------
// Aligner (ViT downsample -> LLM dim)
// ---------------------------------------------------------------------------

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun approximation, max error ~1.5e-7 (matches ops::gelu).
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
}

/// Unfold the ViT patch grid into aligner input columns, mirroring the upstream
/// `Aligner.forward` reshape/permute/pad/`F.unfold` chain.
///
/// `vit_rows` is `[n_h*n_w, vision_dim]` in reading order. Returns
/// `(cols, n_cell_h, n_cell_w)` where `cols` is
/// `[n_cell_h*n_cell_w, vision_dim*r*r]` in `F.unfold` element order
/// (channel-major, then row-major over the `r*r` window). Spatial axes are
/// zero-padded to a multiple of `r`.
pub fn aligner_unfold(
    vit_rows: &[f32],
    n_h: usize,
    n_w: usize,
    vision_dim: usize,
    r: usize,
) -> (Vec<f32>, usize, usize) {
    assert_eq!(vit_rows.len(), n_h * n_w * vision_dim, "vit_rows length");
    assert!(r > 0, "downsample ratio must be positive");
    let pad_h = (r - n_h % r) % r;
    let pad_w = (r - n_w % r) % r;
    let hp = n_h + pad_h;
    let wp = n_w + pad_w;
    let n_cell_h = hp / r;
    let n_cell_w = wp / r;
    let in_dim = vision_dim * r * r;

    let mut cols = vec![0.0f32; n_cell_h * n_cell_w * in_dim];
    for ch in 0..n_cell_h {
        for cw in 0..n_cell_w {
            let cell = ch * n_cell_w + cw;
            let mut off = 0;
            for c in 0..vision_dim {
                for kh in 0..r {
                    for kw in 0..r {
                        let hh = ch * r + kh;
                        let ww = cw * r + kw;
                        // Zero outside the original (unpadded) grid.
                        let value = if hh < n_h && ww < n_w {
                            vit_rows[(hh * n_w + ww) * vision_dim + c]
                        } else {
                            0.0
                        };
                        cols[cell * in_dim + off] = value;
                        off += 1;
                    }
                }
            }
        }
    }
    (cols, n_cell_h, n_cell_w)
}

/// Aligner weights (torch `[out, in]` layout, with bias).
pub struct AlignerWeights<'a> {
    pub w1: &'a [f32],
    pub w1_b: &'a [f32],
    pub w2: &'a [f32],
    pub w2_b: &'a [f32],
}

/// Downsample the ViT rows and project them into the LLM `dim`.
/// Mirrors `Aligner.forward`: unfold, then `w2(gelu(w1(x)))`. Returns
/// `[n_cell_h*n_cell_w, dim]`.
pub fn aligner_forward(
    vit_rows: &[f32],
    weights: &AlignerWeights,
    n_h: usize,
    n_w: usize,
    vision_dim: usize,
    dim: usize,
    r: usize,
) -> Vec<f32> {
    let in_dim = vision_dim * r * r;
    let (cols, n_cell_h, n_cell_w) = aligner_unfold(vit_rows, n_h, n_w, vision_dim, r);
    let rows = n_cell_h * n_cell_w;
    let hidden = vision_linear(&cols, weights.w1, Some(weights.w1_b), rows, in_dim, dim);
    let mut activated = vec![0.0f32; rows * dim];
    for (i, &v) in hidden.iter().enumerate() {
        activated[i] = gelu_scalar(v);
    }
    vision_linear(&activated, weights.w2, Some(weights.w2_b), rows, dim, dim)
}

/// ViT tower weights: patch embed, one `VisionBlockWeights` per layer, final norm.
pub struct VitWeights<'a> {
    pub proj_w: &'a [f32],
    pub proj_b: &'a [f32],
    pub blocks: Vec<VisionBlockWeights<'a>>,
    pub final_norm: &'a [f32],
}

/// Run the full ViT tower: patch embed -> N blocks -> final RMSNorm.
/// Mirrors `ViT.forward`. Returns `[n_patch, dim]`.
#[allow(clippy::too_many_arguments)]
pub fn vit_forward(
    patches: &[f32],
    weights: &VitWeights,
    n_h: usize,
    n_w: usize,
    patch_flat: usize,
    dim: usize,
    n_heads: usize,
    inter: usize,
    rope_dim: usize,
    theta: f64,
) -> Vec<f32> {
    let n_patch = n_h * n_w;
    let (cos, sin) = vision_cos_sin(n_h, n_w, rope_dim, theta);
    let mut x = vision_patch_embed(
        patches,
        weights.proj_w,
        weights.proj_b,
        n_patch,
        patch_flat,
        dim,
    );
    for block in &weights.blocks {
        x = vision_block(&x, block, &cos, &sin, n_patch, dim, n_heads, inter);
    }
    vision_rms_norm(&x, weights.final_norm, VISION_RMS_EPS, n_patch, dim)
}

/// Compose the ViT tower and aligner: patches -> ViT rows -> downsampled LLM
/// rows. Mirrors `Transformer.encode_image` (ViT then Aligner). Returns
/// `[n_cell_h*n_cell_w, dim]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_image(
    patches: &[f32],
    tower: &VitWeights,
    aligner: &AlignerWeights,
    n_h: usize,
    n_w: usize,
    patch_flat: usize,
    vision_dim: usize,
    dim: usize,
    n_heads: usize,
    inter: usize,
    rope_dim: usize,
    theta: f64,
    downsample_ratio: usize,
) -> Vec<f32> {
    let vit_rows = vit_forward(
        patches, tower, n_h, n_w, patch_flat, vision_dim, n_heads, inter, rope_dim, theta,
    );
    aligner_forward(
        &vit_rows,
        aligner,
        n_h,
        n_w,
        vision_dim,
        dim,
        downsample_ratio,
    )
}

// ---------------------------------------------------------------------------
// Owned vision weights (loaded from a checkpoint)
// ---------------------------------------------------------------------------

/// One vision block's weights, owned (torch `[out, in]` layout, biases as-is).
pub struct OwnedVisionBlock {
    pub norm1: Vec<f32>,
    pub wqkv: Vec<f32>,
    pub wqkv_b: Vec<f32>,
    pub wo: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub norm2: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
}

impl OwnedVisionBlock {
    fn borrow(&self) -> VisionBlockWeights<'_> {
        VisionBlockWeights {
            norm1: &self.norm1,
            wqkv: &self.wqkv,
            wqkv_b: &self.wqkv_b,
            wo: &self.wo,
            wo_b: &self.wo_b,
            norm2: &self.norm2,
            w1: &self.w1,
            w2: &self.w2,
        }
    }
}

/// A complete, owned vision tower + aligner ready to `encode_image`. Mirrors the
/// upstream `Transformer.vision`/`Transformer.aligner` pair, plus the geometry
/// (`vision_dim`, `n_heads`, `inter`, `rope_dim`, `theta`, `downsample_ratio`,
/// `patch_flat`, `llm_dim`) needed to run one image.
pub struct OwnedVisionModel {
    pub proj_w: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub blocks: Vec<OwnedVisionBlock>,
    pub final_norm: Vec<f32>,
    pub al_w1: Vec<f32>,
    pub al_w1_b: Vec<f32>,
    pub al_w2: Vec<f32>,
    pub al_w2_b: Vec<f32>,
    pub vision_dim: usize,
    pub llm_dim: usize,
    pub n_heads: usize,
    pub inter: usize,
    pub rope_dim: usize,
    pub theta: f64,
    pub downsample_ratio: usize,
    pub patch_flat: usize,
}

impl OwnedVisionModel {
    /// Run one image: `patches` is `[n_h*n_w, patch_flat]`. Returns the
    /// downsampled aligner rows `[n_cell_h*n_cell_w, llm_dim]`.
    pub fn encode_image(&self, patches: &[f32], n_h: usize, n_w: usize) -> Vec<f32> {
        let blocks: Vec<VisionBlockWeights> =
            self.blocks.iter().map(OwnedVisionBlock::borrow).collect();
        let tower = VitWeights {
            proj_w: &self.proj_w,
            proj_b: &self.proj_b,
            blocks,
            final_norm: &self.final_norm,
        };
        let aligner = AlignerWeights {
            w1: &self.al_w1,
            w1_b: &self.al_w1_b,
            w2: &self.al_w2,
            w2_b: &self.al_w2_b,
        };
        encode_image(
            patches,
            &tower,
            &aligner,
            n_h,
            n_w,
            self.patch_flat,
            self.vision_dim,
            self.llm_dim,
            self.n_heads,
            self.inter,
            self.rope_dim,
            self.theta,
            self.downsample_ratio,
        )
    }
}
