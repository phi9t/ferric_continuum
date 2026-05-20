use std::rc::Rc;

use tnsr::{
    autograd::Engine,
    checkpoint::{checkpoint, TransformerSelectivePolicy, WholeBlockCheckpoint},
    ops::basic,
    scaling::report::{format_report, scale_report},
    tensor::Tensor,
    transformer::{TransformerBlock, TransformerConfig},
};
use tracing::{info, Level};

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    info!("tnsr: transformer autograd + checkpointing demo");

    let cfg = TransformerConfig::tiny_4_7_29();
    let block = Rc::new(TransformerBlock::new(cfg));

    // -----------------------------------------------------------------------
    // 1. Baseline: no checkpointing
    // Engine::new() must be called BEFORE forward to capture save events.
    // -----------------------------------------------------------------------
    info!("[1] No checkpoint");
    {
        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let y = block.forward(&x);
        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);

        engine.print_op_table();

        if let Some(stats) = x.grad_stats() {
            info!(
                min = %format!("{:.4}", stats.min),
                max = %format!("{:.4}", stats.max),
                mean = %format!("{:.4}", stats.mean),
                std = %format!("{:.4}", stats.std),
                "x.grad stats"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. Whole-block checkpoint
    // -----------------------------------------------------------------------
    info!("[2] Whole-block checkpoint");
    {
        let cfg2 = TransformerConfig::tiny_4_7_29();
        let block2 = Rc::new(TransformerBlock::new(cfg2));

        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let policy = Rc::new(WholeBlockCheckpoint);

        let y = checkpoint("block0", policy, std::slice::from_ref(&x), {
            let block2 = block2.clone();
            move |xs| block2.forward(&xs[0])
        });

        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);

        engine.print_checkpoint_report();
        engine.print_saved_tensor_table();

        if let Some(stats) = x.grad_stats() {
            info!(
                min = %format!("{:.4}", stats.min),
                max = %format!("{:.4}", stats.max),
                mean = %format!("{:.4}", stats.mean),
                std = %format!("{:.4}", stats.std),
                "x.grad stats"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Selective checkpoint
    // -----------------------------------------------------------------------
    info!("[3] Selective checkpoint");
    {
        let cfg3 = TransformerConfig::tiny_4_7_29();
        let block3 = Rc::new(TransformerBlock::new(cfg3));

        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let policy = Rc::new(TransformerSelectivePolicy {
            save_softmax_under_bytes: 4096,
            recompute_activation_over_bytes: 8192,
        });

        let y = checkpoint("block0_selective", policy, std::slice::from_ref(&x), {
            let block3 = block3.clone();
            move |xs| block3.forward(&xs[0])
        });

        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);

        engine.print_checkpoint_report();
        engine.print_saved_tensor_table();

        if let Some(stats) = x.grad_stats() {
            info!(
                min = %format!("{:.4}", stats.min),
                max = %format!("{:.4}", stats.max),
                mean = %format!("{:.4}", stats.mean),
                std = %format!("{:.4}", stats.std),
                "x.grad stats"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. DOT graph
    // -----------------------------------------------------------------------
    info!("[4] DOT graph output");
    {
        let cfg4 = TransformerConfig::tiny_4_7_29();
        let block4 = Rc::new(TransformerBlock::new(cfg4));
        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let y = block4.forward(&x);
        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);
        engine.write_dot("/tmp/block.dot");
        info!("Written /tmp/block.dot");
    }

    // -----------------------------------------------------------------------
    // 5. Scaling report
    // -----------------------------------------------------------------------
    info!("[5] Scaling report");
    {
        let cfg5 = TransformerConfig::tiny_4_7_29();
        let report = scale_report(&cfg5);
        let table = format_report(&report);
        for line in table.lines() {
            info!("{}", line);
        }
    }
}
