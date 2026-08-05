//! Qwen3 forward/train demo — end-to-end forward + loss + backward.
//!
//! Follows the structure/logging idiom of `src/main.rs`: build a tiny
//! `qwen3::Qwen3Model`, create the autograd `Engine` (which installs the global
//! recorder), run a forward to `logits[B,T,V]`, compute `cross_entropy` against
//! random targets, run `backward`, then print the op table and gradient stats
//! for a couple of parameters. Weights are random/untrained — this exercises the
//! training path end to end, not a converged model.

use tnsr::{
    autograd::Engine,
    ops::loss,
    qwen3::{Qwen3Config, Qwen3Model},
};
use tracing::{info, Level};

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    info!("tnsr: Qwen3 forward + loss + backward demo");

    let cfg = Qwen3Config::tiny();
    let model = Qwen3Model::new(cfg.clone());

    // Engine::new() must be called BEFORE forward to capture op/save events.
    let mut engine = Engine::new();

    let b = 1usize;
    let t = 8usize;
    let ids: Vec<usize> = (0..t).map(|i| i % cfg.vocab_size).collect();

    let logits = model.forward(&ids, b, t);

    // Report logits shape [B,T,V] and the first row of values.
    {
        let lv = logits.inner.borrow();
        info!(shape = %lv.value.shape, "logits");
        let v = cfg.vocab_size;
        let first_row: Vec<String> = lv.value.data[..v.min(lv.value.data.len())]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect();
        info!("logits[0,0,:] = [{}]", first_row.join(", "));
    }

    // Random next-token targets and the fused softmax cross-entropy loss.
    let targets: Vec<usize> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab_size).collect();
    let l = loss::cross_entropy(&logits, &targets, "cross_entropy");
    {
        let lv = l.inner.borrow();
        info!(loss = %format!("{:.6}", lv.value.data[0]), "cross_entropy");
    }

    engine.backward(&l);

    // Full op table for the traced forward DAG.
    info!("op table:");
    engine.print_op_table();

    // Gradient stats for a couple of representative parameters.
    for (label, param) in [
        ("embed_tokens", &model.embed_tokens),
        ("lm_head", &model.lm_head),
        ("layer0.self_attn.wq", &model.layers[0].self_attn.wq),
        ("layer0.mlp.down_proj", &model.layers[0].mlp.down_proj),
    ] {
        if let Some(stats) = param.grad_stats() {
            info!(
                param = label,
                min = %format!("{:.6}", stats.min),
                max = %format!("{:.6}", stats.max),
                mean = %format!("{:.6}", stats.mean),
                std = %format!("{:.6}", stats.std),
                numel = stats.numel,
                "grad stats"
            );
        } else {
            info!(param = label, "no grad recorded");
        }
    }
}
