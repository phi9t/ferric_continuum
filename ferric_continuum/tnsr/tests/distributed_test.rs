//! Golden values + invariants for `scaling::distributed`.
//!
//! Style mirrors `tests/scaling_test.rs`: exact golden numbers plus algebraic
//! invariants that must hold regardless of the concrete config.

use tnsr::scaling::distributed::collectives::{
    collective_cost, sim_all_gather, sim_all_reduce_sum, sim_broadcast, sim_reduce_scatter_sum,
    sim_ring_all_reduce_sum, Collective,
};
use tnsr::scaling::distributed::data_parallel::data_parallel_cost;
use tnsr::scaling::distributed::fsdp::{fsdp_memory, ZeroStage};
use tnsr::scaling::distributed::mesh::DeviceMesh;
use tnsr::scaling::distributed::pipeline::pipeline_schedule;
use tnsr::scaling::distributed::report::{distributed_report, format_distributed_report};
use tnsr::scaling::distributed::tensor_parallel::{
    sim_column_then_row, tensor_parallel_cost, MlpShapes,
};
use tnsr::scaling::model_stats::model_stats;
use tnsr::scaling::roofline::a100_bf16;
use tnsr::transformer::TransformerConfig;

fn tiny() -> TransformerConfig {
    TransformerConfig::tiny_4_7_29()
}

// ---------------------------------------------------------------------------
// Collectives — cost model
// ---------------------------------------------------------------------------

#[test]
fn test_collective_cost_allreduce_golden() {
    // 2·(D−1)/D·bytes = 2·3/4·1000 = 1500
    let c = collective_cost(Collective::AllReduce, 4, 1000);
    assert_eq!(c.comm_bytes_per_device, 1500);
}

#[test]
fn test_collective_cost_gather_scatter_broadcast_golden() {
    // (D−1)/D·bytes = 3/4·1000 = 750
    assert_eq!(
        collective_cost(Collective::AllGather, 4, 1000).comm_bytes_per_device,
        750
    );
    assert_eq!(
        collective_cost(Collective::ReduceScatter, 4, 1000).comm_bytes_per_device,
        750
    );
    assert_eq!(
        collective_cost(Collective::Broadcast, 4, 1000).comm_bytes_per_device,
        750
    );
}

#[test]
fn test_collective_cost_single_device_is_zero() {
    // Nothing to communicate with one device.
    assert_eq!(
        collective_cost(Collective::AllReduce, 1, 1000).comm_bytes_per_device,
        0
    );
}

// ---------------------------------------------------------------------------
// Collectives — simulations
// ---------------------------------------------------------------------------

#[test]
fn test_sim_all_reduce_sum_golden() {
    let shards = vec![
        vec![1.0, 2.0, 3.0],
        vec![10.0, 20.0, 30.0],
        vec![100.0, 200.0, 300.0],
    ];
    let out = sim_all_reduce_sum(&shards);
    // Every device gets the same summed vector.
    for dev in &out {
        assert_eq!(dev, &vec![111.0, 222.0, 333.0]);
    }
    assert_eq!(out.len(), 3);
}

#[test]
fn test_sim_ring_all_reduce_matches_naive() {
    // Bit-exact on integer-valued f32 inputs.
    let shards = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![10.0, 20.0, 30.0, 40.0],
        vec![100.0, 200.0, 300.0, 400.0],
        vec![1000.0, 2000.0, 3000.0, 4000.0],
    ];
    let naive = sim_all_reduce_sum(&shards);
    let ring = sim_ring_all_reduce_sum(&shards);
    assert_eq!(naive, ring);
}

#[test]
fn test_reduce_scatter_then_all_gather_equals_all_reduce() {
    let shards = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![10.0, 20.0, 30.0, 40.0],
        vec![100.0, 200.0, 300.0, 400.0],
        vec![1000.0, 2000.0, 3000.0, 4000.0],
    ];
    let scattered = sim_reduce_scatter_sum(&shards);
    let gathered = sim_all_gather(&scattered);
    let all_reduce = sim_all_reduce_sum(&shards);
    assert_eq!(gathered, all_reduce);
}

#[test]
fn test_sim_broadcast_copies_root() {
    let shards = vec![vec![1.0, 2.0], vec![9.0, 9.0], vec![0.0, 0.0]];
    let out = sim_broadcast(&shards, 0);
    for dev in &out {
        assert_eq!(dev, &vec![1.0, 2.0]);
    }
}

// ---------------------------------------------------------------------------
// Device mesh
// ---------------------------------------------------------------------------

#[test]
fn test_mesh_total_devices_and_coords() {
    let mesh = DeviceMesh::new_2d(2, "dp", 2, "tp");
    assert_eq!(mesh.total_devices(), 4);
    assert_eq!(mesh.coords(3), vec![1, 1]);
    assert_eq!(mesh.coords(0), vec![0, 0]);
    assert_eq!(mesh.coords(1), vec![0, 1]);
    assert_eq!(mesh.coords(2), vec![1, 0]);
    assert_eq!(mesh.axis_size("dp"), Some(2));
    assert_eq!(mesh.axis_size("tp"), Some(2));
    assert_eq!(mesh.axis_size("pp"), None);
}

// ---------------------------------------------------------------------------
// Data parallel (DDP)
// ---------------------------------------------------------------------------

#[test]
fn test_ddp_per_device_memory_independent_of_dp() {
    let stats = model_stats(&tiny());
    let dp1 = data_parallel_cost(&stats, 1, 2);
    let dp8 = data_parallel_cost(&stats, 8, 2);
    // DDP invariant: full replica everywhere, memory unchanged by dp.
    assert_eq!(dp1.param_bytes_per_device, dp8.param_bytes_per_device);
    assert_eq!(dp1.grad_bytes_per_device, dp8.grad_bytes_per_device);
    assert_eq!(
        dp1.optimizer_state_bytes_per_device,
        dp8.optimizer_state_bytes_per_device
    );
}

#[test]
fn test_ddp_grad_allreduce_bytes_golden() {
    let stats = model_stats(&tiny());
    // params_total = 10208 → grad_bytes = 10208·4 = 40832
    let grad_bytes = stats.params_total as u64 * 4;
    let dp = 8u64;
    let expected = 2 * (dp - 1) * grad_bytes / dp; // 2·(D−1)/D·grad_bytes
    let cost = data_parallel_cost(&stats, dp as usize, 2);
    assert_eq!(cost.grad_allreduce_bytes_per_device, expected);
}

// ---------------------------------------------------------------------------
// FSDP / ZeRO
// ---------------------------------------------------------------------------

#[test]
fn test_fsdp_stage3_shards_params() {
    let stats = model_stats(&tiny());
    let full = stats.params_total as u64 * 4;
    let d = 8;
    let m = fsdp_memory(&stats, d, ZeroStage::Stage3, 2);
    assert_eq!(m.param_bytes, full / d as u64);
    assert_eq!(m.grad_bytes, full / d as u64);
}

#[test]
fn test_fsdp_stage_memory_ordering() {
    let stats = model_stats(&tiny());
    let d = 8;
    let s1 = fsdp_memory(&stats, d, ZeroStage::Stage1, 2);
    let s2 = fsdp_memory(&stats, d, ZeroStage::Stage2, 2);
    let s3 = fsdp_memory(&stats, d, ZeroStage::Stage3, 2);
    // Stage3 ≤ Stage2 ≤ Stage1 per-device memory.
    assert!(s3.total_bytes() <= s2.total_bytes());
    assert!(s2.total_bytes() <= s1.total_bytes());
    // Stage1 leaves gradients unsharded (full replica).
    let full = stats.params_total as u64 * 4;
    assert_eq!(s1.grad_bytes, full);
}

// ---------------------------------------------------------------------------
// Tensor parallel
// ---------------------------------------------------------------------------

#[test]
fn test_sim_column_then_row_equals_unsharded() {
    // Megatron MLP Z = (X·W1)·W2 across tp=3, sharding the hidden dim H.
    // M=3, K=4, H=6, N=5.
    let m = 3;
    let k = 4;
    let h = 6;
    let n = 5;
    let tp = 3;
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let w1: Vec<f32> = (0..k * h).map(|i| (i as f32) * 0.25 + 0.1).collect();
    let w2: Vec<f32> = (0..h * n).map(|i| (i as f32) * 0.125 - 0.3).collect();

    // Reference: unsharded Y = X·W1 (M×H), then Z = Y·W2 (M×N).
    let mut y = vec![0.0f32; m * h];
    for row in 0..m {
        for col in 0..h {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[row * k + kk] * w1[kk * h + col];
            }
            y[row * h + col] = acc;
        }
    }
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for hh in 0..h {
                acc += y[row * h + hh] * w2[hh * n + col];
            }
            expected[row * n + col] = acc;
        }
    }

    let got = sim_column_then_row(&x, &w1, &w2, MlpShapes { m, k, h, n }, tp);
    // Float accumulation order differs (per-shard partials), so allow a tiny eps.
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-4, "got {g} expected {e}");
    }
}

#[test]
fn test_tensor_parallel_column_no_allreduce_row_has_allreduce() {
    let cost = tensor_parallel_cost(&tiny(), 4);
    // Column-parallel up-projection needs no all-reduce.
    assert_eq!(cost.column.allreduce_bytes, 0);
    // Row-parallel down-projection carries the all-reduce.
    assert!(cost.row.allreduce_bytes > 0);
    assert_eq!(cost.allreduce_bytes(), cost.row.allreduce_bytes);
}

// ---------------------------------------------------------------------------
// Pipeline parallel
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_bubble_fraction_golden() {
    // (P−1)/(M+P−1) = 3/11 for P=4, M=8.
    let s = pipeline_schedule(&tiny(), 4, 8);
    assert!((s.bubble_fraction - 3.0 / 11.0).abs() < 1e-12);
}

#[test]
fn test_pipeline_bubble_shrinks_with_more_microbatches() {
    let few = pipeline_schedule(&tiny(), 4, 2);
    let many = pipeline_schedule(&tiny(), 4, 16);
    assert!(many.bubble_fraction < few.bubble_fraction);
}

#[test]
fn test_pipeline_handoff_uses_largest_uneven_microbatch() {
    let cfg = TransformerConfig {
        batch: 5,
        seq: 7,
        d_model: 29,
        d_ff: 116,
        n_heads: 1,
    };

    let s = pipeline_schedule(&cfg, 3, 2);

    assert_eq!(s.num_microbatches, 2);
    // ceil(5 / 2) * seq * d_model * sizeof(f32)
    assert_eq!(s.activation_handoff_bytes, 3 * 7 * 29 * 4);
}

// ---------------------------------------------------------------------------
// Aggregate report
// ---------------------------------------------------------------------------

#[test]
fn test_distributed_report_wires_mesh_axes() {
    let mesh = DeviceMesh::new_2d(2, "dp", 2, "tp");
    let r = distributed_report(&tiny(), 4, &mesh, ZeroStage::Stage3, &a100_bf16());
    assert_eq!(r.dp, 2);
    assert_eq!(r.tp, 2);
    assert_eq!(r.num_layers, 4);
}

#[test]
fn test_format_distributed_report_headers() {
    let mesh = DeviceMesh::new_2d(2, "dp", 2, "tp");
    let r = distributed_report(&tiny(), 4, &mesh, ZeroStage::Stage3, &a100_bf16());
    let table = format_distributed_report(&r);
    assert!(table.contains("tnsr Distributed Training Report"));
    assert!(table.contains("Mesh: dp=2 tp=2"));
    assert!(table.contains("DDP grad all-reduce"));
    assert!(table.contains("Bottleneck:"));
}

#[test]
fn test_distributed_report_summarizes_hybrid_parallelism() {
    let mesh = DeviceMesh {
        shape: vec![2, 2, 3, 4],
        axis_names: vec!["dp".into(), "tp".into(), "pp".into(), "sp".into()],
    };

    let r = distributed_report(&tiny(), 4, &mesh, ZeroStage::Stage3, &a100_bf16());

    assert_eq!(r.dp, 2);
    assert_eq!(r.tp, 2);
    assert_eq!(r.pp, 3);
    assert_eq!(r.sequence_parallel, 4);
    assert_eq!(r.num_pipeline_microbatches, 3);
    assert_eq!(r.pipeline_microbatch_size, 2);
    assert_eq!(r.activation_shard_factor, 8);
    assert_eq!(r.pipeline.num_stages, 3);
    assert_eq!(r.pipeline.num_microbatches, 3);
    assert_eq!(
        r.pipeline_step_handoff_bytes,
        r.pipeline.activation_handoff_bytes * r.num_pipeline_microbatches as u64 * 2
    );
    assert_eq!(
        r.total_comm_bytes_per_device,
        r.data_parallel.grad_allreduce_bytes_per_device
            + r.fsdp.extra_comm_bytes_per_device
            + r.tensor_parallel.allreduce_bytes() * r.num_layers as u64
            + r.pipeline_step_handoff_bytes
    );

    let names: Vec<&str> = r
        .symbolic_collectives
        .iter()
        .map(|collective| collective.name)
        .collect();
    assert_eq!(
        names,
        vec![
            "DDP gradient all-reduce",
            "FSDP parameter all-gather",
            "FSDP gradient reduce-scatter",
            "TP row-parallel all-reduce",
            "PP activation send/recv",
            "context-parallel K/V exchange",
        ]
    );
    assert_eq!(r.symbolic_collectives[0].participants, 2);
    assert_eq!(r.symbolic_collectives[3].participants, 2);
    assert_eq!(r.symbolic_collectives[4].participants, 3);
    assert_eq!(r.symbolic_collectives[5].participants, 4);

    let table = format_distributed_report(&r);
    assert!(table.contains("Mesh: dp=2 tp=2 pp=3 sp=4"));
    assert!(table.contains("Pipeline microbatches:"));
    assert!(table.contains("Pipeline microbatch size:"));
    assert!(table.contains("Activation shard factor:"));
    assert!(table.contains("Pipeline handoff"));
    assert!(table.contains("context-parallel K/V exchange"));
}
