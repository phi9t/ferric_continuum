//! End-to-end Qwen3 inference on **real** Hugging Face weights.
//!
//! Loads a downloaded `Qwen/Qwen3-0.6B` checkpoint directory (`config.json`,
//! `model.safetensors`, `vocab.json`, `merges.txt`) via [`tnsr::qwen3_load`],
//! builds a byte-level BPE tokenizer ([`tnsr::bpe`]), tokenizes a prompt, runs
//! greedy autoregressive decode under a [`NoGradGuard`], and prints the decoded
//! continuation.  Coherent, on-topic output is the practical proof that tnsr's
//! from-scratch math (plus the loader's transpose + RoPE-interleave bridge)
//! matches PyTorch.
//!
//! ```text
//! qwen3_infer --model-dir <path> --prompt "The capital of France is" \
//!             --max-new-tokens 20
//! ```
//!
//! For numeric compatibility checks against upstream HuggingFace, an opt-in
//! full-precision logits dump is available.  It runs a single **prefill**
//! forward and writes the last-position full-vocabulary logits row as JSON,
//! matching the schema emitted by `tools/hf_reference.py`:
//!
//! ```text
//! qwen3_infer --model-dir <path> --prompt "The capital of France is" \
//!             --dump-logits /tmp/tnsr_logits.json
//! ```
//!
//! `--token-ids "a,b,c"` bypasses the (intentionally approximate) BPE and feeds
//! an explicit token-id sequence, so a tokenizer mismatch can be isolated from a
//! model-math mismatch by feeding HF's own ids into tnsr.

use std::path::PathBuf;
use std::process::exit;

use tnsr::{
    bpe::Tokenizer,
    grad_mode::NoGradGuard,
    qwen3_load::load_qwen3,
    tensor::tensor_value_stats,
};
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

struct Args {
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    dump_logits: Option<PathBuf>,
    token_ids: Option<Vec<usize>>,
}

fn parse_args() -> Args {
    let mut model_dir: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut max_new_tokens = 20usize;
    let mut dump_logits: Option<PathBuf> = None;
    let mut token_ids: Option<Vec<usize>> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
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
            "--token-ids" => {
                token_ids = it.next().map(|s| {
                    s.split(',')
                        .map(str::trim)
                        .filter(|p| !p.is_empty())
                        .map(|p| {
                            p.parse::<usize>().unwrap_or_else(|_| {
                                eprintln!("qwen3_infer: bad token id `{p}` in --token-ids");
                                exit(2);
                            })
                        })
                        .collect()
                });
            }
            other => {
                eprintln!("qwen3_infer: unknown arg `{other}`");
                exit(2);
            }
        }
    }

    let model_dir = model_dir.unwrap_or_else(|| {
        eprintln!("qwen3_infer: --model-dir <path> is required");
        exit(2);
    });
    Args {
        model_dir,
        prompt: prompt.unwrap_or_else(|| "The capital of France is".to_string()),
        max_new_tokens,
        dump_logits,
        token_ids,
    }
}

/// Serialize a logits row to JSON matching `tools/hf_reference.py`'s schema.
///
/// Written by hand (rather than pulling `serde_json` into this binary's dep
/// set) to keep the target's deps unchanged and to stream the large `[f32; V]`
/// array as text without an intermediate `Vec<Value>` over ~150k floats.
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
    write!(w, "{{\"token_ids\":")?;
    write!(w, "[")?;
    for (i, id) in token_ids.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "{id}")?;
    }
    write!(w, "],\"prompt\":\"")?;
    // Minimal JSON string escaping for the prompt.
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
    write!(w, "\",\"vocab_size\":{vocab_size},\"logits\":[")?;
    for (i, x) in logits.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        // Full f32 precision; `{}` on f32 round-trips.
        write!(w, "{x}")?;
    }
    write!(w, "]}}")?;
    w.flush()
}

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let args = parse_args();

    info!(model_dir = %args.model_dir.display(), "loading Qwen3 checkpoint");
    let model = match load_qwen3(&args.model_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("qwen3_infer: load failed: {e}");
            exit(1);
        }
    };
    let cfg = model.cfg.clone();
    let v = cfg.vocab_size;
    info!(
        vocab = cfg.vocab_size,
        layers = cfg.num_hidden_layers,
        hidden = cfg.hidden_size,
        heads = cfg.num_attention_heads,
        kv_heads = cfg.num_key_value_heads,
        head_dim = cfg.head_dim,
        rope_theta = cfg.rope_theta,
        "config"
    );

    // Weight-load sanity: adapted stats should be finite and non-degenerate.
    {
        let emb = tensor_value_stats(&model.embed_tokens.inner.borrow().value);
        let fnorm = tensor_value_stats(&model.final_norm.inner.borrow().value);
        info!(
            embed_std = emb.std,
            embed_mean = emb.mean,
            embed_nan = emb.nan_count,
            embed_inf = emb.inf_count,
            final_norm_mean = fnorm.mean,
            "loaded weight stats"
        );
        assert_eq!(emb.nan_count, 0, "embed_tokens has NaNs");
        assert_eq!(emb.inf_count, 0, "embed_tokens has Infs");
        assert!(emb.std > 0.0, "embed_tokens is degenerate");
    }

    let tok = match Tokenizer::from_files(&args.model_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("qwen3_infer: tokenizer load failed: {e}");
            exit(1);
        }
    };

    // `--token-ids` bypasses BPE (used to feed HF's own ids into tnsr and thus
    // isolate a tokenizer mismatch from a model-math mismatch); otherwise
    // encode the prompt with tnsr's approximate byte-level BPE.
    let mut ids = match &args.token_ids {
        Some(explicit) => explicit.clone(),
        None => tok.encode(&args.prompt),
    };
    if ids.is_empty() {
        eprintln!("qwen3_infer: prompt encoded to zero tokens");
        exit(1);
    }
    // Explicit ids can be out of range; catch it here with a clear message
    // rather than panicking deep inside the embedding gather.
    if let Some(&bad) = ids.iter().find(|&&id| id >= v) {
        eprintln!("qwen3_infer: token id {bad} out of range (vocab_size={v})");
        exit(1);
    }
    let prompt_len = ids.len();
    info!(prompt = %args.prompt, token_ids = ?ids, "encoded prompt");

    // Opt-in full-precision logits dump: a single prefill forward whose
    // last-position row is written as JSON for numeric comparison against a
    // real HuggingFace `Qwen3ForCausalLM` forward.  Done before greedy decode
    // so the printed continuation is unaffected.
    if let Some(dump_path) = &args.dump_logits {
        let _guard = NoGradGuard::new();
        let t = ids.len();
        let logits = model.forward(&ids, 1, t);
        let lv = logits.inner.borrow();
        let data = lv.value.data.as_ref();
        let last_row = &data[(t - 1) * v..t * v];
        match write_logits_json(dump_path, &ids, &args.prompt, v, last_row) {
            Ok(()) => info!(path = %dump_path.display(), vocab = v, "dumped last-position logits"),
            Err(e) => {
                eprintln!("qwen3_infer: dump-logits write failed: {e}");
                exit(1);
            }
        }
    }

    // EOS ids for Qwen3 (im_end / endoftext).
    const EOS: [usize; 2] = [151645, 151643];

    let _guard = NoGradGuard::new();

    for step in 0..args.max_new_tokens {
        let t = ids.len();
        let logits = model.forward(&ids, 1, t);
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
        if EOS.contains(&next) {
            info!(step, next_id = next, "hit EOS, stopping");
            break;
        }
    }

    let continuation = tok.decode(&ids[prompt_len..]);
    let full = tok.decode(&ids);
    if continuation.trim().is_empty() {
        warn!("continuation decoded to empty/whitespace");
    }
    info!(generated_ids = ?ids, "final token-id sequence");
    println!("\n=== PROMPT ===\n{}", args.prompt);
    println!("\n=== CONTINUATION ===\n{}", continuation);
    println!("\n=== FULL ===\n{}", full);
}
