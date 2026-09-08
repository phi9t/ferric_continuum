use std::rc::Rc;

use tnsr::{
    autograd::Engine,
    checkpoint::{checkpoint, TransformerSelectivePolicy, WholeBlockCheckpoint},
    ops::basic,
    playground::{DemoOptions, DemoOutput},
    scaling::distributed::{
        fsdp::ZeroStage,
        mesh::DeviceMesh,
        report::{distributed_report, format_distributed_report},
    },
    scaling::memory::{
        training_memory_report, ActivationCheckpointing, Precision, TrainingMemoryConfig,
        ZeroStage as MemoryZeroStage,
    },
    scaling::report::{format_report, scale_report},
    scaling::roofline::a100_bf16,
    tensor::Tensor,
    transformer::{TransformerBlock, TransformerConfig},
};
use tracing::{info, Level};

fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    let options = match DemoOptions::parse_from(std::env::args()) {
        Ok(options) => options,
        Err(message) if message.starts_with("Usage:") => {
            println!("{message}");
            return;
        }
        Err(message) => {
            eprintln!("{message}\n\n{}", DemoOptions::usage());
            std::process::exit(2);
        }
    };

    if let Err(err) = run_demo(options) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run_demo(options: DemoOptions) -> Result<(), String> {
    info!("tnsr: transformer autograd + checkpointing demo");

    let cfg = TransformerConfig::tiny_4_7_29();
    let block = Rc::new(TransformerBlock::new(cfg));

    // -----------------------------------------------------------------------
    // 1. Baseline: no checkpointing
    // Recording is scoped explicitly to the forward execution.
    // -----------------------------------------------------------------------
    info!("[1] No checkpoint");
    {
        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let loss = engine.with_recording(|| {
            let y = block.forward(&x);
            basic::sum(&y, "loss")
        });
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

        let loss = engine.with_recording(|| {
            let y = checkpoint("block0", policy, std::slice::from_ref(&x), {
                let block2 = block2.clone();
                move |xs| block2.forward(&xs[0])
            });
            basic::sum(&y, "loss")
        });
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

        let loss = engine.with_recording(|| {
            let y = checkpoint("block0_selective", policy, std::slice::from_ref(&x), {
                let block3 = block3.clone();
                move |xs| block3.forward(&xs[0])
            });
            basic::sum(&y, "loss")
        });
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
    // 4. Playground artifacts
    // -----------------------------------------------------------------------
    info!("[4] Playground artifact output");
    {
        let cfg4 = TransformerConfig::tiny_4_7_29();
        let block4 = Rc::new(TransformerBlock::new(cfg4));
        let mut engine = Engine::new();
        let x = Tensor::randn(&[4, 7, 29]).requires_grad();
        let loss = engine.with_recording(|| {
            let y = block4.forward(&x);
            basic::sum(&y, "loss")
        });
        engine.backward(&loss);
        for output in &options.outputs {
            match output {
                DemoOutput::Dot(path) => {
                    ensure_parent_dir(path)?;
                    std::fs::write(path, engine.dot_string())
                        .map_err(|e| format!("write DOT {}: {}", path.display(), e))?;
                    info!("Written {}", path.display());
                }
                DemoOutput::TraceJson(path) => {
                    ensure_parent_dir(path)?;
                    std::fs::write(path, engine.debug.trace_json_pretty())
                        .map_err(|e| format!("write trace JSON {}: {}", path.display(), e))?;
                    info!("Written {}", path.display());
                }
            }
        }
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

    // -----------------------------------------------------------------------
    // 6. Training memory report
    // -----------------------------------------------------------------------
    info!("[6] Training memory report");
    {
        let cfg6 = TransformerConfig::tiny_4_7_29();
        let report = training_memory_report(
            &cfg6,
            TrainingMemoryConfig {
                num_layers: 4,
                precision: Precision::Bf16,
                activation_checkpointing: ActivationCheckpointing::Selective,
                data_parallel_shards: 4,
                zero_stage: MemoryZeroStage::Stage3,
            },
        );
        info!(
            parameter_bytes = report.parameter_bytes,
            gradient_bytes = report.gradient_bytes,
            optimizer_state_bytes = report.optimizer_state_bytes,
            activation_bytes = report.activation_bytes,
            kv_cache_bytes = report.kv_cache_bytes,
            total_training_bytes = report.total_training_bytes(),
            "bf16 selective ZeRO-3 memory estimate"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Distributed parallelism report
    // 2×2 mesh: 2-way data parallel × 2-way tensor parallel, ZeRO-3 (FSDP),
    // scored against an A100 BF16 roofline. Symbolic estimate only — no devices.
    // -----------------------------------------------------------------------
    info!("[7] Distributed parallelism report");
    {
        let cfg6 = TransformerConfig::tiny_4_7_29();
        let mesh = DeviceMesh::new_2d(2, "dp", 2, "tp");
        let num_layers = 4;
        let report = distributed_report(&cfg6, num_layers, &mesh, ZeroStage::Stage3, &a100_bf16());
        let table = format_distributed_report(&report);
        for line in table.lines() {
            info!("{}", line);
        }
    }

    Ok(())
}

fn ensure_parent_dir(path: &std::path::Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create output dir {}: {}", parent.display(), e))?;
        }
    }
    Ok(())
}
