//! Qwen3 text-generation demo — greedy autoregressive decode.
//!
//! Runs the (untrained, random-weight) `qwen3::Qwen3Model` in inference mode
//! under a `grad_mode::NoGradGuard` so no autograd graph is built. Starting from
//! a short seed id sequence, each step runs `model.forward(ids, 1, t)`, reads the
//! last-position logits row from `logits[B,T,V]`, takes the argmax as the next
//! token id, appends it, and repeats for a fixed number of steps. The output is
//! not meaningful text (random weights) — it demonstrates the inference path.

use tnsr::{
    grad_mode::NoGradGuard,
    qwen3::{Qwen3Config, Qwen3Model},
};
use tracing::{info, Level};

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

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    info!("tnsr: Qwen3 greedy autoregressive generation demo (untrained weights)");

    let cfg = Qwen3Config::tiny();
    let model = Qwen3Model::new(cfg.clone());
    let v = cfg.vocab_size;

    // Seed sequence and number of tokens to generate.
    let mut ids: Vec<usize> = vec![1, 2, 3];
    let num_new_tokens = 12usize;

    info!(seed = ?ids, "seed ids");

    // Inference: disable grad so forward builds no autograd graph.
    let _guard = NoGradGuard::new();

    for step in 0..num_new_tokens {
        let t = ids.len();
        let logits = model.forward(&ids, 1, t);

        // Read the last-position logits row from [B=1, T, V].
        let next = {
            let lv = logits.inner.borrow();
            let data = lv.value.data.as_ref();
            let last_row = &data[(t - 1) * v..t * v];
            argmax(last_row)
        };

        ids.push(next);
        info!(step, next_id = next, seq_len = ids.len(), "generated token");
    }

    info!(generated = ?ids, "final token-id sequence");
}
