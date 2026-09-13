//! Image grid planning for the DeepSeek V4.1 vision tower.
//!
//! An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and, after the
//! aligner downsample, a `n_llm_h x n_llm_w` token grid the LLM sees as
//!
//! ```text
//! [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
//! ```
//!
//! These are pure functions of the original image size and the vision config.
//! They mirror `image_processor.py` exactly; integer division and `ceil`
//! ordering match upstream so the grids are bit-stable.

use super::config::DeepSeekV41VisionConfig;

/// Token-type tags. Text positions are `TEXT`; the four image tags distinguish
/// the image-span positions that all carry `image_token_id` in `input_ids`.
pub const TEXT: i64 = -1;
pub const IMAGE_START: i64 = 0;
pub const IMAGE: i64 = 1;
pub const IMAGE_NEW_LINE: i64 = 2;
pub const IMAGE_END: i64 = 3;

/// Number of LLM tokens an `n_llm_h x n_llm_w` aligner grid occupies:
/// one IMAGE_NEW_LINE per row plus the IMAGE_START/IMAGE_END sentinels.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// Token grid the aligner produces from a patch grid of this pixel size.
pub fn llm_grid(
    best_height: usize,
    best_width: usize,
    patch_size: usize,
    downsample_ratio: usize,
) -> (usize, usize) {
    (
        ceil_div(best_height / patch_size, downsample_ratio),
        ceil_div(best_width / patch_size, downsample_ratio),
    )
}

/// Largest aspect-preserving pixel size whose token grid still fits in
/// `max_n_token`. Includes the tall (single-column) and wide (single-row)
/// collapse branches.
pub fn solve_resize_ratio(
    height: usize,
    width: usize,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize) {
    let h = height as f64;
    let w = width as f64;
    let r = h / w;
    let max_w_float = (((max_n_token as f64) - 2.0) / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = patch_size * downsample_ratio;
    if max_w_float < 1.0 {
        // very tall: collapse to a single column
        return ((max_n_token - 2) / 2 * cell, cell);
    }
    if max_h_float < 1.0 {
        // very wide: collapse to a single row
        return (cell, (max_n_token - 3) * cell);
    }
    let beta = f64::min(
        max_w_float.floor() * (cell as f64) / w,
        max_h_float.floor() * (cell as f64) / h,
    );
    let best_h = ((h * beta / (patch_size as f64)).floor() as usize) * patch_size;
    let best_w = ((w * beta / (patch_size as f64)).floor() as usize) * patch_size;
    (best_h, best_w)
}

/// Shrink the pixel size until the image costs at most `max_n_token` LLM tokens.
/// Returns `(n_llm_h, n_llm_w, best_height, best_width)`.
pub fn safe_resize(
    height: usize,
    width: usize,
    best_height: usize,
    best_width: usize,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize, usize, usize) {
    let (mut n_llm_h, mut n_llm_w) =
        llm_grid(best_height, best_width, patch_size, downsample_ratio);
    let (mut best_height, mut best_width) = (best_height, best_width);
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token {
        let (bh, bw) = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token);
        best_height = bh;
        best_width = bw;
        let (h, w) = llm_grid(best_height, best_width, patch_size, downsample_ratio);
        n_llm_h = h;
        n_llm_w = w;
        debug_assert!(num_image_tokens(n_llm_h, n_llm_w) <= max_n_token);
    }
    (n_llm_h, n_llm_w, best_height, best_width)
}

/// Resize plan for an image of the given original size; a pure function of its
/// arguments. Returns `(n_llm_h, n_llm_w, best_height, best_width)`.
pub fn plan_image_grid(
    width: usize,
    height: usize,
    cfg: &DeepSeekV41VisionConfig,
) -> (usize, usize, usize, usize) {
    let p = cfg.patch_size;
    let mut width = width;
    if let Some(ratio) = cfg.max_wh_ratio {
        // `width > height * ratio` in upstream; the clamp keeps float-then-int
        // truncation identical (`width = height * ratio`).
        if (width as f64) > (height as f64) * ratio {
            width = ((height as f64) * ratio) as usize;
        }
    }
    let mut height = height;
    let area = width * height;
    if area > 0 && area < cfg.min_pixels {
        let ratio = ((cfg.min_pixels as f64) / (area as f64)).sqrt();
        width = ((width as f64) * ratio) as usize;
        height = ((height as f64) * ratio) as usize;
    }
    let best_width = ceil_div(width, p) * p;
    let best_height = ceil_div(height, p) * p;
    safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        cfg.downsample_ratio,
        cfg.max_image_tokens,
    )
}

/// Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row.
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i64> {
    let mut types = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    types.push(IMAGE_START);
    for _ in 0..n_llm_h {
        for _ in 0..n_llm_w {
            types.push(IMAGE);
        }
        types.push(IMAGE_NEW_LINE);
    }
    types.push(IMAGE_END);
    types
}

/// Integer `ceil(a / b)` matching Python `math.ceil(a / b)` for non-negative
/// integers.
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
