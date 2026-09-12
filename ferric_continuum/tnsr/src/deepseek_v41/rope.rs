//! DeepSeek V4.1 interleaved YaRN RoPE helpers.
//!
//! These are pure math utilities for verifier fixtures. They intentionally do
//! not depend on the autograd `Tensor` type or mutate any cache state.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnRopeConfig {
    pub rope_head_dim: usize,
    pub max_seq_len: usize,
    pub original_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeShape {
    Bsd {
        batch: usize,
        seqlen: usize,
        dim: usize,
    },
    Bshd {
        batch: usize,
        seqlen: usize,
        heads: usize,
        dim: usize,
    },
}

impl RopeShape {
    fn parts(self) -> (usize, usize, usize, usize) {
        match self {
            RopeShape::Bsd { batch, seqlen, dim } => (batch, seqlen, 1, dim),
            RopeShape::Bshd {
                batch,
                seqlen,
                heads,
                dim,
            } => (batch, seqlen, heads, dim),
        }
    }
}

pub fn precompute_freqs(config: &YarnRopeConfig, seqlen: usize) -> Vec<[f32; 2]> {
    assert!(
        config.rope_head_dim % 2 == 0,
        "DeepSeek V4.1 RoPE head dim must be even"
    );
    assert!(
        seqlen <= config.max_seq_len,
        "requested seqlen exceeds configured max_seq_len"
    );

    let dim = config.rope_head_dim;
    let half = dim / 2;
    let mut base_freqs = Vec::with_capacity(half);
    for pair in 0..half {
        let exponent = (2 * pair) as f32 / dim as f32;
        base_freqs.push(1.0 / config.rope_theta.powf(exponent));
    }

    if config.original_seq_len > 0 {
        let corrected_dim = |rotations: f32| -> f32 {
            dim as f32
                * (config.original_seq_len as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * config.rope_theta.ln())
        };
        let low = corrected_dim(config.beta_fast).floor().max(0.0);
        let high = corrected_dim(config.beta_slow).ceil().min((dim - 1) as f32);
        let denom = (high - low).max(1e-3);
        for (pair, freq) in base_freqs.iter_mut().enumerate() {
            let ramp = ((pair as f32 - low) / denom).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / config.rope_factor * (1.0 - smooth) + *freq * smooth;
        }
    }

    let mut out = Vec::with_capacity(seqlen * half);
    for pos in 0..seqlen {
        for freq in &base_freqs {
            let theta = pos as f32 * *freq;
            out.push([theta.cos(), theta.sin()]);
        }
    }
    out
}

pub fn apply_rotary_adjacent_pairs(
    data: &mut [f32],
    shape: RopeShape,
    freqs: &[[f32; 2]],
    inverse: bool,
) {
    let (batch, seqlen, heads, dim) = shape.parts();
    assert!(dim % 2 == 0, "DeepSeek V4.1 RoPE dim must be even");
    assert_eq!(
        data.len(),
        batch * seqlen * heads * dim,
        "RoPE data length does not match shape"
    );
    assert!(
        freqs.len() >= seqlen * (dim / 2),
        "RoPE frequency table is too short for shape"
    );

    let half = dim / 2;
    for b in 0..batch {
        for s in 0..seqlen {
            for h in 0..heads {
                let base = ((b * seqlen + s) * heads + h) * dim;
                let freq_base = s * half;
                for pair in 0..half {
                    let [cos, sin] = freqs[freq_base + pair];
                    let sin = if inverse { -sin } else { sin };
                    let x0 = data[base + 2 * pair];
                    let x1 = data[base + 2 * pair + 1];
                    data[base + 2 * pair] = x0 * cos - x1 * sin;
                    data[base + 2 * pair + 1] = x0 * sin + x1 * cos;
                }
            }
        }
    }
}
