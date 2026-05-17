use std::rc::Rc;

use tinytensor::{
    autograd::Engine,
    checkpoint::{checkpoint, TransformerSelectivePolicy, WholeBlockCheckpoint},
    ops::basic,
    tensor::Tensor,
    transformer::{TransformerBlock, TransformerConfig},
};

fn main() {
    println!("=== tinytensor: transformer autograd + checkpointing demo ===\n");

    let cfg = TransformerConfig::tiny_4_7_29();
    let block = Rc::new(TransformerBlock::new(cfg));

    // -----------------------------------------------------------------------
    // 1. Baseline: no checkpointing
    // Engine::new() must be called BEFORE forward to capture save events.
    // -----------------------------------------------------------------------
    println!("--- [1] No checkpoint ---");
    {
        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let y = block.forward(&x);
        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);

        engine.print_op_table();
        println!();

        if let Some(stats) = x.grad_stats() {
            println!("x.grad: min={:.4} max={:.4} mean={:.4} std={:.4}", stats.min, stats.max, stats.mean, stats.std);
        }
    }

    // -----------------------------------------------------------------------
    // 2. Whole-block checkpoint
    // -----------------------------------------------------------------------
    println!("\n--- [2] Whole-block checkpoint ---");
    {
        let cfg2 = TransformerConfig::tiny_4_7_29();
        let block2 = Rc::new(TransformerBlock::new(cfg2));

        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let policy = Rc::new(WholeBlockCheckpoint);

        let y = checkpoint("block0", policy, &[x.clone()], {
            let block2 = block2.clone();
            move |xs| block2.forward(&xs[0])
        });

        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);

        engine.print_checkpoint_report();
        engine.print_saved_tensor_table();
        println!();

        if let Some(stats) = x.grad_stats() {
            println!("x.grad: min={:.4} max={:.4} mean={:.4} std={:.4}", stats.min, stats.max, stats.mean, stats.std);
        }
    }

    // -----------------------------------------------------------------------
    // 3. Selective checkpoint
    // -----------------------------------------------------------------------
    println!("\n--- [3] Selective checkpoint ---");
    {
        let cfg3 = TransformerConfig::tiny_4_7_29();
        let block3 = Rc::new(TransformerBlock::new(cfg3));

        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let policy = Rc::new(TransformerSelectivePolicy {
            save_softmax_under_bytes: 4096,
            recompute_activation_over_bytes: 8192,
        });

        let y = checkpoint("block0_selective", policy, &[x.clone()], {
            let block3 = block3.clone();
            move |xs| block3.forward(&xs[0])
        });

        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);

        engine.print_checkpoint_report();
        engine.print_saved_tensor_table();
        println!();

        if let Some(stats) = x.grad_stats() {
            println!("x.grad: min={:.4} max={:.4} mean={:.4} std={:.4}", stats.min, stats.max, stats.mean, stats.std);
        }
    }

    // -----------------------------------------------------------------------
    // 4. DOT graph
    // -----------------------------------------------------------------------
    println!("\n--- [4] DOT graph output ---");
    {
        let cfg4 = TransformerConfig::tiny_4_7_29();
        let block4 = Rc::new(TransformerBlock::new(cfg4));
        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let y = block4.forward(&x);
        let loss = basic::sum(&y, "loss");
        engine.backward(&loss);
        engine.write_dot("/tmp/block.dot");
        println!("Written /tmp/block.dot");
    }
}
