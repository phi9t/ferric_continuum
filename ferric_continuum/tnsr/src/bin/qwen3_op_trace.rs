//! Qwen3 op-closure tracer — the tnsr analog of `docs/trace_qwen3.py`.
//!
//! `trace_qwen3.py` runs a tiny HuggingFace `Qwen3ForCausalLM` under a
//! `TorchDispatchMode` and records the transitive closure of ATen ops the whole
//! `nn.Module` stack decomposes into, with per-leaf-module attribution. This
//! binary does the Rust equivalent: it runs `qwen3::Qwen3Model` end-to-end
//! (forward + cross-entropy + backward) under `tnsr`'s native op-recording seam
//! (`autograd::Engine` + `debug::DebugRecorder`) and reports:
//!
//! - the **forward closure**: distinct `OpKind`s + per-kind call counts, taken
//!   from `Engine::topo` (the topologically-sorted forward DAG),
//! - the **forward+backward closure**: the forward set plus the `OpKind`s
//!   applied during the reverse walk (`DebugRecorder::backward_apply_kinds`),
//! - **per-component attribution**: each forward `OpCallRecord` bucketed by the
//!   transformer component parsed from its op name (the analog of PyTorch's
//!   per-`nn.Module` breakdown).
//!
//! Unlike PyTorch there is no view/metadata noise here: `tnsr`'s `OpKind`s are
//! already the "minimal compute kernel set" the doc converges on, so the Rust
//! closure is small and every op is arithmetic.

use std::collections::{BTreeMap, HashMap};

use tnsr::{
    autograd::{Engine, OpKind},
    ops::loss,
    qwen3::{Qwen3Config, Qwen3Model},
};
use tracing::{info, Level};

/// Map a forward op's `name` (e.g. `q_proj`, `input_layernorm`, `swiglu_mul`)
/// to a coarse transformer component, mirroring PyTorch's leaf-module buckets.
fn component_of(name: &str) -> &'static str {
    match name {
        "embed_tokens" => "Embedding",
        "lm_head" => "LmHead",
        "final_norm" => "FinalNorm",
        "input_layernorm" | "post_attention_layernorm" => "RMSNorm (block)",
        "q_norm" | "k_norm" => "RMSNorm (per-head q/k)",
        "q_proj" | "k_proj" | "v_proj" | "o_proj" => "Attention projections",
        "q_reshape" | "k_reshape" | "v_reshape" | "attn_reshape" => "Attention reshape",
        "q_rope" | "k_rope" => "RoPE",
        "gqa" => "GQA attention core",
        "attn_residual" | "mlp_residual" => "Residual add",
        "gate_proj" | "up_proj" | "down_proj" => "MLP projections",
        "silu" => "MLP activation (SiLU)",
        "swiglu_mul" => "MLP combine (SwiGLU)",
        "cross_entropy" | "loss" => "Loss",
        _ => "Other",
    }
}

/// Print a "kind : count" table sorted by descending count, then name.
fn print_kind_counts(counts: &HashMap<OpKind, usize>) {
    let mut rows: Vec<(OpKind, usize)> = counts.iter().map(|(k, c)| (*k, *c)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| format!("{:?}", a.0).cmp(&format!("{:?}", b.0))));
    for (kind, count) in rows {
        info!("  {:>5}  {:?}", count, kind);
    }
}

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    info!("tnsr: Qwen3 op-closure tracer (Rust analog of trace_qwen3.py)");

    // Tiny Qwen3: V=11, L=2, D=16, F=32, Hq=4, Hk=2, Dh=4.
    let cfg = Qwen3Config::tiny();
    info!(
        vocab_size = cfg.vocab_size,
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        num_attention_heads = cfg.num_attention_heads,
        num_key_value_heads = cfg.num_key_value_heads,
        head_dim = cfg.head_dim,
        "tiny Qwen3 config"
    );
    let model = Qwen3Model::new(cfg.clone());

    // Engine::new() installs the global recorder; do it BEFORE forward so every
    // op registers its OpCall + saved tensors into the recorder.
    let mut engine = Engine::new();

    // A short token sequence, one batch row.
    let b = 1usize;
    let t = 8usize;
    let ids: Vec<usize> = (0..t).map(|i| i % cfg.vocab_size).collect();

    let logits = model.forward(&ids, b, t);

    // Next-token style targets (arbitrary — we only trace ops, not train).
    let targets: Vec<usize> = (0..b * t).map(|i| (i + 1) % cfg.vocab_size).collect();
    let loss = loss::cross_entropy(&logits, &targets, "cross_entropy");

    engine.backward(&loss);

    // ── Forward closure: count OpKind over the topo-sorted forward DAG ───────
    let mut fwd_counts: HashMap<OpKind, usize> = HashMap::new();
    for call in &engine.topo {
        *fwd_counts.entry(call.kind).or_default() += 1;
    }
    let fwd_total: usize = fwd_counts.values().sum();

    info!("");
    info!(
        "== Forward closure: {} distinct ops, {} total dispatches ==",
        fwd_counts.len(),
        fwd_total
    );
    print_kind_counts(&fwd_counts);

    // ── Backward closure: forward set ∪ backward-applied kinds ───────────────
    let bwd_kinds = engine.debug.backward_apply_kinds();
    let mut bwd_counts: HashMap<OpKind, usize> = HashMap::new();
    for kind in &bwd_kinds {
        *bwd_counts.entry(*kind).or_default() += 1;
    }

    // Union of forward + backward distinct kinds.
    let mut union_counts = fwd_counts.clone();
    for (kind, count) in &bwd_counts {
        *union_counts.entry(*kind).or_default() += *count;
    }
    let union_total: usize = union_counts.values().sum();

    info!("");
    info!(
        "== Forward+backward closure: {} distinct ops, {} total dispatches ==",
        union_counts.len(),
        union_total
    );
    print_kind_counts(&union_counts);

    info!("");
    info!(
        "  (backward pass alone applied {} ops across {} distinct kinds)",
        bwd_kinds.len(),
        bwd_counts.len()
    );

    // ── Per-component attribution (analog of PyTorch's per-nn.Module table) ──
    let records = engine.debug.op_call_records();
    let mut by_component: BTreeMap<&'static str, HashMap<OpKind, usize>> = BTreeMap::new();
    for rec in &records {
        let comp = component_of(&rec.name);
        *by_component.entry(comp).or_default().entry(rec.kind).or_default() += 1;
    }

    info!("");
    info!("== Ops by transformer component ==");
    for (comp, kinds) in &by_component {
        let total: usize = kinds.values().sum();
        info!("  {} ({} ops):", comp, total);
        let mut rows: Vec<(OpKind, usize)> = kinds.iter().map(|(k, c)| (*k, *c)).collect();
        rows.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)))
        });
        for (kind, count) in rows {
            info!("      {:>4}  {:?}", count, kind);
        }
    }

    info!("");
    info!("Traced Qwen3 op closure — this is the minimal compute kernel set from");
    info!("docs/qwen3_pytorch_op_closure.md, now observed natively on the Rust side.");
}
