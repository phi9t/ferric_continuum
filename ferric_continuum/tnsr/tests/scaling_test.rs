use tnsr::scaling::inference::{kv_cache_bytes, peak_activation_bytes};
use tnsr::scaling::model_stats::model_stats;
use tnsr::scaling::op_cost::{dense_matmul_fwd_flops, total_fwd_flops, total_train_flops};
use tnsr::scaling::parallelism::{
    parallel_cost, pipeline_bubble, MoeConfig, ParallelConfig, PipelineSchedule, ZeroStage,
};
use tnsr::scaling::report::scale_report;
use tnsr::scaling::roofline::{a100_bf16, roofline, Bottleneck};
use tnsr::scaling::sharding::{shard_matmul, ShardingCase};
use tnsr::transformer::TransformerConfig;

fn tiny() -> TransformerConfig {
    TransformerConfig::tiny_4_7_29()
}

// ---------------------------------------------------------------------------
// Parameter counts
// ---------------------------------------------------------------------------

#[test]
fn test_model_stats_tiny_params_total() {
    // B=4, T=7, D=29, F=116
    // QKVO: 4 * 29 * 29 = 3364
    // MLP:  2 * 29 * 116 = 6728
    // norm: 4 * 29 = 116
    // total = 3364 + 6728 + 116 = 10208
    let s = model_stats(&tiny());
    assert_eq!(s.params_total, 10208);
}

#[test]
fn test_model_stats_tiny_params_matmul() {
    // matmul params = QKVO + MLP = 3364 + 6728 = 10092
    let s = model_stats(&tiny());
    assert_eq!(s.params_matmul, 10092);
    assert_eq!(s.params_qkvo, 3364);
    assert_eq!(s.params_mlp, 6728);
    assert_eq!(s.params_norm, 116);
}

#[test]
fn test_model_stats_tiny_tokens_per_batch() {
    let s = model_stats(&tiny());
    assert_eq!(s.tokens_per_batch, 4 * 7); // B * T = 28
}

// ---------------------------------------------------------------------------
// FLOPs
// ---------------------------------------------------------------------------

#[test]
fn test_dense_matmul_fwd_flops_tiny() {
    // Q,K,V,O: 4 * 2*B*T*D^2 = 4 * 2*4*7*841 = 188384
    // attn_scores: 2*B*T^2*D = 2*4*49*29 = 11368
    // attn_mix:    2*B*T^2*D = 11368
    // mlp_up:   2*B*T*D*F = 2*4*7*29*116 = 188384
    // mlp_down: 2*B*T*F*D = 188384
    // total = 4*47096 + 22736 + 376768 = 187384 + 22736 + 376768
    // = 188384 + 22736 + 376768 = 587888
    let got = dense_matmul_fwd_flops(&tiny());
    assert_eq!(got, 587_888);
}

#[test]
fn test_attention_train_flops_tiny() {
    // attn_scores fwd: 2*B*T^2*D = 11368
    // attn_mix fwd:    2*B*T^2*D = 11368
    // total attn fwd = 22736; training (×3) = 68208
    use tnsr::scaling::op_cost::all_op_costs;
    let costs = all_op_costs(&tiny());
    let attn_fwd: u64 = costs
        .iter()
        .filter(|c| matches!(c.name, "attn_scores" | "attn_mix"))
        .map(|c| c.fwd_flops)
        .sum();
    let attn_bwd: u64 = costs
        .iter()
        .filter(|c| matches!(c.name, "attn_scores" | "attn_mix"))
        .map(|c| c.bwd_flops)
        .sum();
    assert_eq!(attn_fwd + attn_bwd, 68_208);
}

#[test]
fn test_dense_matmul_rule_tiny() {
    // 6 * params_matmul * tokens = 6 * 10092 * 28 = 1,695,456
    let s = model_stats(&tiny());
    let rule_flops = 6u64 * s.params_matmul as u64 * s.tokens_per_batch as u64;
    assert_eq!(rule_flops, 1_695_456);
}

#[test]
fn test_total_fwd_flops_positive() {
    // Just verify total is larger than the dense matmul portion
    let total = total_fwd_flops(&tiny());
    let dense = dense_matmul_fwd_flops(&tiny());
    assert!(total > dense, "total={total} dense={dense}");
}

// ---------------------------------------------------------------------------
// Roofline
// ---------------------------------------------------------------------------

#[test]
fn test_roofline_compute_bound() {
    let hw = a100_bf16();
    // Very high arithmetic intensity → compute bound
    let est = roofline(1_000_000_000_000, 1_000, &hw);
    assert_eq!(est.bottleneck, Bottleneck::Compute);
}

#[test]
fn test_roofline_memory_bound() {
    let hw = a100_bf16();
    // Very low arithmetic intensity → memory bound
    let est = roofline(1_000, 1_000_000_000, &hw);
    assert_eq!(est.bottleneck, Bottleneck::Memory);
}

// ---------------------------------------------------------------------------
// Sharding
// ---------------------------------------------------------------------------

#[test]
fn test_shard_matmul_inner_reduction_allreduce() {
    // Inner-reduction sharding requires an all-reduce of M*N f32 elements.
    let cost = shard_matmul(128, 256, 512, 4, ShardingCase::InnerReduction);
    assert_eq!(cost.allreduce_bytes, 128 * 256 * 4);
    assert_eq!(cost.local_fwd_flops, 2 * 128 * 512 * 256 / 4);
}

#[test]
fn test_shard_matmul_rowwise_no_allreduce() {
    let cost = shard_matmul(128, 256, 512, 4, ShardingCase::XRowwise);
    assert_eq!(cost.allreduce_bytes, 0);
}

// ---------------------------------------------------------------------------
// KV cache and inference
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_bytes_f32() {
    // 2 * 4 bytes * 1 layer * B=1 * T=8 * D=16 = 1024
    assert_eq!(kv_cache_bytes(1, 1, 8, 16, 4), 1024);
}

#[test]
fn test_peak_activation_bytes() {
    // 1 layer, B=4, T=7, D=29, F=116, f32 (4 bytes)
    // attn = 2*B*T*D = 2*4*7*29 = 1624 elements
    // mlp  = 2*B*T*F = 2*4*7*116 = 6496 elements
    // total = (1624 + 6496) * 4 bytes * 1 layer = 32480
    assert_eq!(peak_activation_bytes(1, 4, 7, 29, 116, 4), 32_480);
}

// ---------------------------------------------------------------------------
// total_train_flops consistency
// ---------------------------------------------------------------------------

#[test]
fn test_total_train_flops_matches_report() {
    // total_train_flops must equal report.total_fwd + report.total_bwd
    let cfg = tiny();
    let train = total_train_flops(&cfg);
    let r = scale_report(&cfg);
    assert_eq!(train, r.total_fwd_flops + r.total_bwd_flops);
}

// ---------------------------------------------------------------------------
// 5D parallelism cost model
// ---------------------------------------------------------------------------
//
// Reference model for these tests: D=16, F=64, B=8, T=8, L=4.
//   P_attn = 4*D*D     = 1024
//   P_mlp  = 2*D*F     = 2048
//   P_norm = 4*D       = 64
//   P_block            = 3136
// B=8 so num_microbatches in {1,2,4,8} all divide the local batch.

const PL: usize = 4; // num_layers

fn par_cfg() -> TransformerConfig {
    TransformerConfig {
        batch: 8,
        seq: 8,
        d_model: 16,
        d_ff: 64,
        n_heads: 1,
    }
}

/// dp/tp/pp/cp/ep all = 1, ZeRO None, AFAB, Adam(8), dense — tweak per test.
fn base_pc() -> ParallelConfig {
    ParallelConfig {
        dp: 1,
        tp: 1,
        pp: 1,
        cp: 1,
        ep: 1,
        zero: ZeroStage::None,
        num_microbatches: 1,
        schedule: PipelineSchedule::Afab,
        optimizer_bytes_per_param: 8,
        moe: None,
    }
}

#[test]
fn test_zero_stage0_no_sharding() {
    // dp=2, no ZeRO: mp_params = 4*3136 = 12544.
    let pc = ParallelConfig { dp: 2, ..base_pc() };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.param_bytes, 12544 * 4); // 50176
    assert_eq!(c.grad_bytes, 50176);
    assert_eq!(c.optimizer_bytes, 12544 * 8); // 100352
}

#[test]
fn test_zero_stage1_shards_optimizer() {
    // dp=2, ZeRO-1: only optimizer state is sharded.
    let pc = ParallelConfig {
        dp: 2,
        zero: ZeroStage::OptimizerState,
        ..base_pc()
    };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.param_bytes, 50176); // unchanged
    assert_eq!(c.grad_bytes, 50176); // unchanged
    assert_eq!(c.optimizer_bytes, 50176); // 100352 / 2
}

#[test]
fn test_zero_stage3_fsdp_shards_all() {
    // dp=4, ZeRO-3/FSDP: params + grads + optimizer all sharded.
    let pc = ParallelConfig {
        dp: 4,
        zero: ZeroStage::Full,
        ..base_pc()
    };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.param_bytes, 12544); // 50176 / 4
    assert_eq!(c.grad_bytes, 12544);
    assert_eq!(c.optimizer_bytes, 25088); // 100352 / 4
}

#[test]
fn test_tp_shards_params_and_flops() {
    // tp=2: mp_params = 4*(1024/2 + 2048/2 + 64) = 4*1600 = 6400.
    let dense = parallel_cost(&par_cfg(), PL, &base_pc());
    let pc = ParallelConfig { tp: 2, ..base_pc() };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.param_bytes, 6400 * 4); // 25600
    assert_eq!(c.flops_per_device, dense.flops_per_device / 2);
}

#[test]
fn test_pp_shards_layers() {
    // pp=2, L=4: layers_per_stage = 2, mp_params = 2*3136 = 6272.
    let pc = ParallelConfig {
        pp: 2,
        num_microbatches: 4,
        ..base_pc()
    };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.layers_per_stage, 2);
    assert_eq!(c.param_bytes, 6272 * 4); // 25088
    assert!(c.pp_comm_bytes > 0);
    assert_eq!(c.tp_comm_bytes, 0);
}

#[test]
fn test_cp_shards_activation_and_flops() {
    let dense = parallel_cost(&par_cfg(), PL, &base_pc());
    let pc = ParallelConfig { cp: 2, ..base_pc() };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.activation_bytes, dense.activation_bytes / 2);
    assert_eq!(c.flops_per_device, dense.flops_per_device / 2);
    assert!(c.cp_comm_bytes > 0);
}

#[test]
fn test_afab_vs_1f1b_activation_memory() {
    // pp=2, m=4: AFAB stores m=4 microbatches, 1F1B stores min(pp,m)=2.
    // act_per_layer_micro = (2*8*8*16 + 2*8*8*64)*4 / 4 = 10240; S = 2.
    let afab = parallel_cost(
        &par_cfg(),
        PL,
        &ParallelConfig {
            pp: 2,
            num_microbatches: 4,
            schedule: PipelineSchedule::Afab,
            ..base_pc()
        },
    );
    let onef1b = parallel_cost(
        &par_cfg(),
        PL,
        &ParallelConfig {
            pp: 2,
            num_microbatches: 4,
            schedule: PipelineSchedule::OneF1B,
            ..base_pc()
        },
    );
    assert_eq!(afab.activation_bytes, 81920); // 10240 * 2 * 4
    assert_eq!(onef1b.activation_bytes, 40960); // 10240 * 2 * 2
    assert_eq!(afab.activation_bytes, 2 * onef1b.activation_bytes);
    assert_eq!(afab.microbatch_size, 2); // B/m = 8/4
}

#[test]
fn test_pipeline_bubble_formula() {
    assert_eq!(pipeline_bubble(4, 8), 0.375); // (4-1)/8
    assert_eq!(pipeline_bubble(1, 8), 0.0);
    assert_eq!(pipeline_bubble(4, 0), 0.0);

    let c = parallel_cost(
        &par_cfg(),
        PL,
        &ParallelConfig {
            pp: 4,
            num_microbatches: 8,
            ..base_pc()
        },
    );
    assert_eq!(c.bubble_over_ideal, 0.375);
    assert!((c.pipeline_efficiency - 8.0 / 11.0).abs() < 1e-9); // m/(m+pp-1)
    assert!((c.pipeline_idle_fraction - 3.0 / 11.0).abs() < 1e-9);
}

#[test]
fn test_dp_comm_exact_allreduce() {
    // dp=2 gradient all-reduce of mp_params=12544 f32:
    //   2 * (2-1)/2 * 12544 * 4 = 50176
    let pc = ParallelConfig { dp: 2, ..base_pc() };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.dp_comm_bytes, 50176);
}

#[test]
fn test_pp_comm_independent_of_microbatches() {
    // pp=2, tp=cp=1: 2 * (B*T*D) * 4 = 2 * 1024 * 4 = 8192, regardless of m.
    let m1 = parallel_cost(
        &par_cfg(),
        PL,
        &ParallelConfig {
            pp: 2,
            num_microbatches: 1,
            ..base_pc()
        },
    );
    let m4 = parallel_cost(
        &par_cfg(),
        PL,
        &ParallelConfig {
            pp: 2,
            num_microbatches: 4,
            ..base_pc()
        },
    );
    assert_eq!(m1.pp_comm_bytes, 8192);
    assert_eq!(m4.pp_comm_bytes, 8192);
}

#[test]
fn test_tp_comm_exact() {
    // tp=2, L=4, pp=1 -> S=4, cp=1: S * 2 * all_reduce(B*T*D=1024, 2)
    //   all_reduce = 2 * (1*1024*4/2) = 4096; total = 4 * 2 * 4096 = 32768
    let pc = ParallelConfig { tp: 2, ..base_pc() };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.tp_comm_bytes, 32768);
}

#[test]
fn test_comm_zero_when_degree_one() {
    let c = parallel_cost(&par_cfg(), PL, &base_pc());
    assert_eq!(c.tp_comm_bytes, 0);
    assert_eq!(c.dp_comm_bytes, 0);
    assert_eq!(c.pp_comm_bytes, 0);
    assert_eq!(c.cp_comm_bytes, 0);
    assert_eq!(c.ep_comm_bytes, 0);
    assert_eq!(c.bubble_over_ideal, 0.0);
    assert_eq!(c.pipeline_efficiency, 1.0);
}

#[test]
fn test_moe_expert_parallel() {
    // MoE: 4 experts, ep=2, top_k=1. experts_per_rank = 2.
    // mp_params = 4*(1024 + 2*2048 + 64) = 4*5184 = 20736 (> dense 12544).
    let pc = ParallelConfig {
        ep: 2,
        moe: Some(MoeConfig {
            num_experts: 4,
            top_k: 1,
        }),
        ..base_pc()
    };
    let c = parallel_cost(&par_cfg(), PL, &pc);
    assert_eq!(c.param_bytes, 20736 * 4);
    assert!(c.ep_comm_bytes > 0);
}

#[test]
fn test_world_size() {
    let pc = ParallelConfig {
        dp: 2,
        tp: 2,
        pp: 2,
        ..base_pc()
    };
    assert_eq!(pc.world_size(), 8);
}

#[test]
fn test_validate_rejects_bad_configs() {
    let cfg = par_cfg();
    // zero degree
    assert!(ParallelConfig { tp: 0, ..base_pc() }
        .validate(&cfg, PL)
        .is_err());
    // m=0
    assert!(ParallelConfig {
        num_microbatches: 0,
        ..base_pc()
    }
    .validate(&cfg, PL)
    .is_err());
    // batch not divisible by m (8 % 3 != 0)
    assert!(ParallelConfig {
        num_microbatches: 3,
        ..base_pc()
    }
    .validate(&cfg, PL)
    .is_err());
    // pp > num_layers
    assert!(ParallelConfig { pp: 5, ..base_pc() }
        .validate(&cfg, PL)
        .is_err());
    // ep > 1 without MoE
    assert!(ParallelConfig { ep: 2, ..base_pc() }
        .validate(&cfg, PL)
        .is_err());
    // num_experts not divisible by ep
    assert!(ParallelConfig {
        ep: 2,
        moe: Some(MoeConfig {
            num_experts: 3,
            top_k: 1
        }),
        ..base_pc()
    }
    .validate(&cfg, PL)
    .is_err());
    // top_k > num_experts
    assert!(ParallelConfig {
        ep: 2,
        moe: Some(MoeConfig {
            num_experts: 4,
            top_k: 5
        }),
        ..base_pc()
    }
    .validate(&cfg, PL)
    .is_err());
    // a valid config passes
    assert!(base_pc().validate(&cfg, PL).is_ok());
}
