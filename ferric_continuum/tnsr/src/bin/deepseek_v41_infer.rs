//! Text-only DeepSeek V4.1-Flash inference CLI.
//!
//! Loads a converted-style checkpoint directory (`config.json`, plus either a
//! single-file `model.safetensors` or `model{rank}-mp{world}.safetensors`
//! shards) via [`tnsr::deepseek_v41::load`], feeds an explicit token-id sequence
//! through the text-only forward, and optionally writes a full-precision
//! last-position logits row for numeric parity against upstream `model.py`.
//!
//! This wave is deliberately text-only: DeepSeek V4.1 ships no Jinja chat
//! template (upstream prompt formatting lives in `encoding/encoding.py`), so the
//! CLI keeps the Qwen3 `--token-ids` isolation pattern and does not require any
//! tokenizer files.  Feeding upstream's own ids into tnsr separates a tokenizer
//! mismatch from a model-math mismatch.
//!
//! ```text
//! deepseek_v41_infer --model-dir <path> --token-ids 1,2,3 \
//!                    --max-new-tokens 0 --dump-logits /tmp/logits.json
//! ```
//!
//! Multimodal (Wave 2) and DSpark/speculative decode (Wave 3) are out of scope;
//! image placeholders and DSpark/speculative flags are rejected.

use std::path::PathBuf;
use std::process::exit;

use tnsr::deepseek_v41::load::{forward_with_token_seed, load_text_model};
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
    text_only: bool,
    thinking_mode: ThinkingMode,
    reasoning_effort: u32,
}

/// Flags that belong to unimplemented waves; passing any of them is an error so
/// a caller cannot silently believe multimodal/DSpark decode is happening.
const REJECTED_FLAGS: &[&str] = &[
    "--image",
    "--image-file",
    "--images",
    "--vision",
    "--dspark",
    "--speculative",
    "--spec-decode",
    "--draft-model",
    "--mtp",
];

fn parse_args() -> Args {
    let mut model_dir: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut max_new_tokens = 0usize;
    let mut dump_logits: Option<PathBuf> = None;
    let mut token_ids: Option<Vec<usize>> = None;
    let mut text_only = true;
    let mut thinking_mode = ThinkingMode::Chat;
    let mut reasoning_effort = 50u32;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        // Reject deferred-wave flags before anything else.
        if REJECTED_FLAGS.contains(&a.as_str()) {
            eprintln!(
                "deepseek_v41_infer: flag `{a}` targets a deferred wave \
                 (multimodal/DSpark) and is not supported in text-only mode"
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
            "--text-only" => text_only = true,
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
        text_only,
        thinking_mode,
        reasoning_effort,
    }
}

/// Serialize a logits row to JSON matching `tools/deepseek_v41_reference.py`'s
/// schema.  Written by hand to keep this binary's dep set minimal and to stream
/// the large `[f32; V]` array as text.
fn write_logits_json(
    path: &std::path::Path,
    token_ids: &[usize],
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
    write!(w, "],\"prompt\":\"")?;
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
        "\",\"vocab_size\":{vocab_size},\"model_type\":\"deepseek_v41_text\",\"logits\":["
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

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let args = parse_args();

    info!(
        model_dir = %args.model_dir.display(),
        text_only = args.text_only,
        thinking_mode = ?args.thinking_mode,
        reasoning_effort = args.reasoning_effort,
        "loading DeepSeek V4.1 text-only checkpoint"
    );
    let mut model = match load_text_model(&args.model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("deepseek_v41_infer: load failed: {e}");
            exit(1);
        }
    };
    let v = model.vocab_size;
    info!(
        vocab = model.vocab_size,
        layers = model.layers.len(),
        hidden = model.hidden_size,
        hc_mult = model.hc_mult,
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

    // This wave requires explicit ids: DeepSeek V4.1 has no bundled BPE table in
    // this repo, and `--token-ids` is the isolation surface parity relies on.
    let mut ids = match &args.token_ids {
        Some(explicit) => explicit.clone(),
        None => {
            eprintln!(
                "deepseek_v41_infer: text-only wave requires --token-ids \
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
    // Text-only mode rejects the image placeholder id outright.
    if args.text_only {
        if let Some(&img) = ids.iter().find(|&&id| id == model.image_token_id) {
            eprintln!("deepseek_v41_infer: image token id {img} is not allowed in text-only mode");
            exit(1);
        }
    }
    let prompt_len = ids.len();
    info!(prompt = %args.prompt, token_ids = ?ids, "input token ids");

    // Opt-in full-precision logits dump: a single prefill forward whose
    // last-position row is written as JSON for numeric parity comparison.
    if let Some(dump_path) = &args.dump_logits {
        let t = ids.len();
        let logits = match forward_with_token_seed(&mut model, &ids, 1, t) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("deepseek_v41_infer: forward failed: {e}");
                exit(1);
            }
        };
        let lv = logits.inner.borrow();
        let data = lv.value.data.as_ref();
        let last_row = &data[(t - 1) * v..t * v];
        match write_logits_json(dump_path, &ids, &args.prompt, v, last_row) {
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
    // tokenizer we stop only at max_new_tokens.
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
