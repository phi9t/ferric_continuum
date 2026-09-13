//! DeepSeek V4.1-Flash inference CLI (text-only by default; opt-in multimodal).
//!
//! Loads a converted-style checkpoint directory (`config.json`, plus either a
//! single-file `model.safetensors` or `model{rank}-mp{world}.safetensors`
//! shards) via [`tnsr::deepseek_v41::load`], feeds an explicit token-id sequence
//! through the forward, and optionally writes a full-precision last-position
//! logits row for numeric parity against upstream `model.py`.
//!
//! This CLI keeps the Qwen3 `--token-ids` isolation pattern: DeepSeek V4.1 ships
//! no Jinja chat template (upstream prompt formatting lives in
//! `encoding/encoding.py`), and image bytes are never decoded in Rust.  Feeding
//! upstream's own ids into tnsr separates a tokenizer/preprocessor mismatch from
//! a model-math mismatch.
//!
//! ```text
//! # text-only (default)
//! deepseek_v41_infer --model-dir <path> --token-ids 1,2,3 \
//!                    --max-new-tokens 0 --dump-logits /tmp/logits.json
//!
//! # multimodal: ids/types come from upstream `prepare_vl_inputs`, aligner
//! # patches from a JSON file (no PIL/image decode in Rust)
//! deepseek_v41_infer --model-dir <path> --multimodal \
//!                    --token-ids 3,4,4,...,2 --token-types -1,0,1,...,-1 \
//!                    --image-patches spans.json --dump-logits /tmp/logits.json
//! ```
//!
//! DSpark/speculative decode (Wave 3) is out of scope; those flags are rejected.

use std::path::PathBuf;
use std::process::exit;

use tnsr::deepseek_v41::load::{
    forward_multimodal_with_seed, forward_with_token_seed, load_multimodal_model, load_text_model,
};
use tnsr::deepseek_v41::model::{DeepSeekV41TextModel, ImageDelimiters, ImageSpan};
use tnsr::tensor::tensor_value_stats;
use tracing::{info, warn, Level};

/// Index of the maximum element in `row` (first on ties).
fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Top-`k` (id, logit) pairs of `row`, highest first — a numeric sanity anchor.
fn top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    idx.into_iter().take(k).map(|i| (i, row[i])).collect()
}

/// Prompt-rendering mode for fixture compatibility.  This wave feeds token ids
/// directly, so the mode is recorded for logging/metadata but does not itself
/// perform tokenization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThinkingMode {
    Chat,
    Thinking,
}

struct Args {
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    dump_logits: Option<PathBuf>,
    token_ids: Option<Vec<usize>>,
    token_types: Option<Vec<i64>>,
    image_patches: Option<PathBuf>,
    text_only: bool,
    multimodal: bool,
    thinking_mode: ThinkingMode,
    reasoning_effort: u32,
}

/// Flags that belong to the unimplemented DSpark/speculative wave; passing any
/// of them is an error so a caller cannot silently believe speculative decode is
/// happening.  Image handling is supported via `--multimodal` (see below), so
/// image flags are intentionally not rejected here.
const REJECTED_FLAGS: &[&str] = &[
    "--dspark",
    "--speculative",
    "--spec-decode",
    "--draft-model",
    "--mtp",
];

/// One image's aligner-slot geometry and raw ViT patches, read from the
/// `--image-patches` JSON file.  Mirrors the upstream `ImageInput` fields the
/// Rust forward needs; image bytes are never decoded here.
#[derive(serde::Deserialize)]
struct ImagePatchRecord {
    /// Position of `IMAGE_START` within the token sequence.
    start: usize,
    n_vit_h: usize,
    n_vit_w: usize,
    n_llm_h: usize,
    n_llm_w: usize,
    /// ViT patch grid, `[n_vit_h*n_vit_w, patch_flat]` row-major.
    patches: Vec<f32>,
}

fn parse_args() -> Args {
    let mut model_dir: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut max_new_tokens = 0usize;
    let mut dump_logits: Option<PathBuf> = None;
    let mut token_ids: Option<Vec<usize>> = None;
    let mut token_types: Option<Vec<i64>> = None;
    let mut image_patches: Option<PathBuf> = None;
    let mut text_only = true;
    let mut multimodal = false;
    let mut thinking_mode = ThinkingMode::Chat;
    let mut reasoning_effort = 50u32;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        // Reject deferred-wave flags before anything else.
        if REJECTED_FLAGS.contains(&a.as_str()) {
            eprintln!(
                "deepseek_v41_infer: flag `{a}` targets a deferred wave \
                 (DSpark/speculative) and is not supported"
            );
            exit(2);
        }
        match a.as_str() {
            "--model-dir" => model_dir = it.next().map(PathBuf::from),
            "--prompt" => prompt = it.next(),
            "--max-new-tokens" => {
                max_new_tokens = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(max_new_tokens)
            }
            "--dump-logits" => dump_logits = it.next().map(PathBuf::from),
            "--text-only" => {
                text_only = true;
                multimodal = false;
            }
            "--multimodal" => {
                multimodal = true;
                text_only = false;
            }
            "--image-patches" => image_patches = it.next().map(PathBuf::from),
            "--thinking-mode" => {
                thinking_mode = match it.next().as_deref() {
                    Some("chat") => ThinkingMode::Chat,
                    Some("thinking") => ThinkingMode::Thinking,
                    other => {
                        eprintln!(
                            "deepseek_v41_infer: --thinking-mode expects chat|thinking, got {other:?}"
                        );
                        exit(2);
                    }
                };
            }
            "--reasoning-effort" => {
                reasoning_effort = it
                    .next()
                    .and_then(|s| s.parse::<u32>().ok())
                    .filter(|&n| (1..=100).contains(&n))
                    .unwrap_or_else(|| {
                        eprintln!("deepseek_v41_infer: --reasoning-effort expects 1..=100");
                        exit(2);
                    });
            }
            "--token-ids" => {
                token_ids = it.next().map(|s| {
                    s.split(',')
                        .map(str::trim)
                        .filter(|p| !p.is_empty())
                        .map(|p| {
                            p.parse::<usize>().unwrap_or_else(|_| {
                                eprintln!("deepseek_v41_infer: bad token id `{p}` in --token-ids");
                                exit(2);
                            })
                        })
                        .collect()
                });
            }
            "--token-types" => {
                token_types = it.next().map(|s| {
                    s.split(',')
                        .map(str::trim)
                        .filter(|p| !p.is_empty())
                        .map(|p| {
                            p.parse::<i64>().unwrap_or_else(|_| {
                                eprintln!(
                                    "deepseek_v41_infer: bad token type `{p}` in --token-types"
                                );
                                exit(2);
                            })
                        })
                        .collect()
                });
            }
            other => {
                eprintln!("deepseek_v41_infer: unknown arg `{other}`");
                exit(2);
            }
        }
    }

    let model_dir = model_dir.unwrap_or_else(|| {
        eprintln!("deepseek_v41_infer: --model-dir <path> is required");
        exit(2);
    });
    Args {
        model_dir,
        prompt: prompt.unwrap_or_default(),
        max_new_tokens,
        dump_logits,
        token_ids,
        token_types,
        image_patches,
        text_only,
        multimodal,
        thinking_mode,
        reasoning_effort,
    }
}

/// Serialize a logits row to JSON matching `tools/deepseek_v41_reference.py`'s
/// schema.  Written by hand to keep this binary's dep set minimal and to stream
/// the large `[f32; V]` array as text.  When `token_types` is `Some`, the row is
/// tagged `deepseek_v41_multimodal` and echoes the types; otherwise it is the
/// text-only schema.
fn write_logits_json(
    path: &std::path::Path,
    token_ids: &[usize],
    token_types: Option<&[i64]>,
    prompt: &str,
    vocab_size: usize,
    logits: &[f32],
) -> std::io::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    write!(w, "{{\"token_ids\":[")?;
    for (i, id) in token_ids.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "{id}")?;
    }
    write!(w, "]")?;
    let model_type = if token_types.is_some() {
        "deepseek_v41_multimodal"
    } else {
        "deepseek_v41_text"
    };
    if let Some(types) = token_types {
        write!(w, ",\"token_types\":[")?;
        for (i, t) in types.iter().enumerate() {
            if i > 0 {
                write!(w, ",")?;
            }
            write!(w, "{t}")?;
        }
        write!(w, "]")?;
    }
    write!(w, ",\"prompt\":\"")?;
    for ch in prompt.chars() {
        match ch {
            '"' => write!(w, "\\\"")?,
            '\\' => write!(w, "\\\\")?,
            '\n' => write!(w, "\\n")?,
            '\r' => write!(w, "\\r")?,
            '\t' => write!(w, "\\t")?,
            c if (c as u32) < 0x20 => write!(w, "\\u{:04x}", c as u32)?,
            c => write!(w, "{c}")?,
        }
    }
    write!(
        w,
        "\",\"vocab_size\":{vocab_size},\"model_type\":\"{model_type}\",\"logits\":["
    )?;
    for (i, x) in logits.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "{x}")?;
    }
    write!(w, "]}}")?;
    w.flush()
}

/// Load `--image-patches` JSON into aligner rows per image, running each image
/// through the vision tower.  Returns the owned aligner-row buffers plus the
/// span metadata the multimodal forward needs.  Errors on any geometry mismatch.
fn build_image_rows(
    model: &DeepSeekV41TextModel,
    path: &std::path::Path,
) -> Result<Vec<(usize, Vec<i64>, Vec<f32>)>, String> {
    use tnsr::deepseek_v41::vision_grid::image_token_types;

    let vision = model
        .vision
        .as_ref()
        .ok_or_else(|| "checkpoint has no vision tower; --multimodal not supported".to_string())?;
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let records: Vec<ImagePatchRecord> =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;

    let mut out = Vec::with_capacity(records.len());
    for (i, rec) in records.into_iter().enumerate() {
        let n_patch = rec.n_vit_h * rec.n_vit_w;
        let expect = n_patch * vision.patch_flat;
        if rec.patches.len() != expect {
            return Err(format!(
                "image {i}: patches len {} != n_vit_h*n_vit_w*patch_flat {expect}",
                rec.patches.len()
            ));
        }
        // Aligner rows in reading order; the span's IMAGE slots consume them.
        let aligner_rows = vision.encode_image(&rec.patches, rec.n_vit_h, rec.n_vit_w);
        let types = image_token_types(rec.n_llm_h, rec.n_llm_w);
        out.push((rec.start, types, aligner_rows));
    }
    Ok(out)
}

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let args = parse_args();

    info!(
        model_dir = %args.model_dir.display(),
        text_only = args.text_only,
        multimodal = args.multimodal,
        thinking_mode = ?args.thinking_mode,
        reasoning_effort = args.reasoning_effort,
        "loading DeepSeek V4.1 checkpoint"
    );
    let mut model = if args.multimodal {
        match load_multimodal_model(&args.model_dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("deepseek_v41_infer: multimodal load failed: {e}");
                exit(1);
            }
        }
    } else {
        match load_text_model(&args.model_dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("deepseek_v41_infer: load failed: {e}");
                exit(1);
            }
        }
    };
    let v = model.vocab_size;
    info!(
        vocab = model.vocab_size,
        layers = model.layers.len(),
        hidden = model.hidden_size,
        hc_mult = model.hc_mult,
        has_vision = model.vision.is_some(),
        "config"
    );

    // Weight-load sanity: adapted embeddings should be finite and non-degenerate.
    {
        let emb = tensor_value_stats(&model.embed_tokens.inner.borrow().value);
        info!(
            embed_std = emb.std,
            embed_mean = emb.mean,
            embed_nan = emb.nan_count,
            embed_inf = emb.inf_count,
            "loaded weight stats"
        );
        assert_eq!(emb.nan_count, 0, "embed_tokens has NaNs");
        assert_eq!(emb.inf_count, 0, "embed_tokens has Infs");
        assert!(emb.std > 0.0, "embed_tokens is degenerate");
    }

    // Both modes require explicit ids: DeepSeek V4.1 has no bundled BPE table in
    // this repo, and `--token-ids` is the isolation surface parity relies on.
    let mut ids = match &args.token_ids {
        Some(explicit) => explicit.clone(),
        None => {
            eprintln!(
                "deepseek_v41_infer: requires --token-ids \
                 (no tokenizer is bundled); got --prompt only"
            );
            exit(2);
        }
    };
    if ids.is_empty() {
        eprintln!("deepseek_v41_infer: --token-ids produced zero tokens");
        exit(1);
    }
    if let Some(&bad) = ids.iter().find(|&&id| id >= v) {
        eprintln!("deepseek_v41_infer: token id {bad} out of range (vocab_size={v})");
        exit(1);
    }
    // Text-only mode rejects the image placeholder id outright; multimodal mode
    // expects it inside image spans.
    if !args.multimodal {
        if let Some(&img) = ids.iter().find(|&&id| id == model.image_token_id) {
            eprintln!("deepseek_v41_infer: image token id {img} is not allowed in text-only mode");
            exit(1);
        }
    }
    let prompt_len = ids.len();
    info!(prompt = %args.prompt, token_ids = ?ids, "input token ids");

    // In multimodal mode, assemble token_types + per-image aligner rows once; the
    // dump/decode paths below reuse them.
    let mm = if args.multimodal {
        let token_types = args.token_types.clone().unwrap_or_else(|| {
            eprintln!("deepseek_v41_infer: --multimodal requires --token-types");
            exit(2);
        });
        if token_types.len() != ids.len() {
            eprintln!(
                "deepseek_v41_infer: --token-types len {} != --token-ids len {}",
                token_types.len(),
                ids.len()
            );
            exit(1);
        }
        let patches_path = args.image_patches.clone().unwrap_or_else(|| {
            eprintln!("deepseek_v41_infer: --multimodal requires --image-patches <json>");
            exit(2);
        });
        let images = match build_image_rows(&model, &patches_path) {
            Ok(rows) => rows,
            Err(e) => {
                eprintln!("deepseek_v41_infer: image-patches failed: {e}");
                exit(1);
            }
        };
        Some((token_types, images))
    } else {
        None
    };

    // Run one prefill forward for the current `ids`, returning the logits tensor.
    // Threads the multimodal spans through when present.  Delimiter embeddings are
    // cloned out first so the `&mut model` forward call doesn't alias them.
    let delims_owned = mm.as_ref().map(|_| {
        (
            model.image_start.clone().expect("image_start weight"),
            model.image_end.clone().expect("image_end weight"),
            model.image_newline.clone().expect("image_newline weight"),
        )
    });
    let run_forward = |model: &mut DeepSeekV41TextModel, ids: &[usize]| {
        let t = ids.len();
        if let Some((token_types, images)) = &mm {
            let spans: Vec<ImageSpan> = images
                .iter()
                .map(|(start, types, rows)| ImageSpan {
                    start: *start,
                    token_types: types,
                    aligner_rows: rows,
                })
                .collect();
            let (start_emb, end_emb, nl_emb) = delims_owned.as_ref().expect("delims cloned");
            let delims = ImageDelimiters {
                image_start: start_emb,
                image_end: end_emb,
                image_newline: nl_emb,
            };
            forward_multimodal_with_seed(model, ids, token_types, 1, t, &[spans], &delims)
        } else {
            forward_with_token_seed(model, ids, 1, t)
        }
    };

    // Opt-in full-precision logits dump: a single prefill forward whose
    // last-position row is written as JSON for numeric parity comparison.
    if let Some(dump_path) = &args.dump_logits {
        let t = ids.len();
        let logits = match run_forward(&mut model, &ids) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("deepseek_v41_infer: forward failed: {e}");
                exit(1);
            }
        };
        let lv = logits.inner.borrow();
        let data = lv.value.data.as_ref();
        let last_row = &data[(t - 1) * v..t * v];
        let token_types = mm.as_ref().map(|(tt, _)| tt.as_slice());
        match write_logits_json(dump_path, &ids, token_types, &args.prompt, v, last_row) {
            Ok(()) => {
                info!(path = %dump_path.display(), vocab = v, "dumped last-position logits")
            }
            Err(e) => {
                eprintln!("deepseek_v41_infer: dump-logits write failed: {e}");
                exit(1);
            }
        }
    }

    // Greedy decode.  DeepSeek V4.1 EOS ids are model-specific; without a
    // tokenizer we stop only at max_new_tokens.  Decode is text-only: appending a
    // generated id past the image spans keeps `token_types` valid, but multimodal
    // decode beyond the seeded prefix is out of scope, so guard it.
    if args.max_new_tokens > 0 && args.multimodal {
        eprintln!("deepseek_v41_infer: multimodal decode past prefill is out of scope; use --max-new-tokens 0 with --dump-logits");
        exit(2);
    }
    for step in 0..args.max_new_tokens {
        let t = ids.len();
        let logits = match forward_with_token_seed(&mut model, &ids, 1, t) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("deepseek_v41_infer: forward failed: {e}");
                exit(1);
            }
        };
        let next = {
            let lv = logits.inner.borrow();
            let data = lv.value.data.as_ref();
            let last_row = &data[(t - 1) * v..t * v];
            if step == 0 {
                let tops = top_k(last_row, 5);
                info!(top5 = ?tops, "first-step top-5 (id, logit)");
            }
            argmax(last_row)
        };
        ids.push(next);
    }

    if args.max_new_tokens > 0 && ids.len() == prompt_len {
        warn!("no tokens generated");
    }
    info!(generated_ids = ?ids, "final token-id sequence");
    println!("token_ids: {ids:?}");
}
