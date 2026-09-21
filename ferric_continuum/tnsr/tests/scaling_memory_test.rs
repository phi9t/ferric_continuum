use tnsr::scaling::memory::{
    activation_memory_bytes, kv_cache_memory_bytes, training_memory_report,
    zero_partitioned_memory, ActivationCheckpointing, Precision, TrainingMemoryConfig, ZeroStage,
};
use tnsr::scaling::model_stats::model_stats;
use tnsr::transformer::TransformerConfig;

fn tiny() -> TransformerConfig {
    TransformerConfig::tiny_4_7_29()
}

#[test]
fn test_training_memory_report_tiny_f32_no_checkpoint_goldens() {
    let cfg = tiny();
    let report = training_memory_report(&cfg, TrainingMemoryConfig::default());

    assert_eq!(report.stats.params_total, 10_208);
    assert_eq!(report.parameter_bytes, 40_832);
    assert_eq!(report.gradient_bytes, 40_832);
    assert_eq!(report.optimizer_state_bytes, 81_664);
    assert_eq!(report.activation_bytes, 59_248);
    assert_eq!(report.kv_cache_bytes, 6_496);
    assert_eq!(report.total_training_bytes(), 222_576);
}

#[test]
fn test_activation_checkpointing_modes_tiny_goldens() {
    let cfg = tiny();

    assert_eq!(
        activation_memory_bytes(&cfg, 1, Precision::F32, ActivationCheckpointing::None),
        59_248
    );
    assert_eq!(
        activation_memory_bytes(&cfg, 1, Precision::F32, ActivationCheckpointing::Selective),
        4_816
    );
    assert_eq!(
        activation_memory_bytes(&cfg, 1, Precision::F32, ActivationCheckpointing::Full),
        0
    );
}

#[test]
fn test_selective_activation_formula_keeps_large_attention_sites() {
    let cfg = TransformerConfig {
        batch: 1,
        seq: 128,
        d_model: 16,
        d_ff: 64,
        n_heads: 1,
    };

    // Runtime TransformerSelectivePolicy always prefers saving AttentionScores
    // [B,T,T] and AttentionMix [B,T,D], while the Softmax [B,T,T] site is too
    // large for the small-save threshold here.
    assert_eq!(
        activation_memory_bytes(&cfg, 1, Precision::F32, ActivationCheckpointing::Selective),
        (128 * 128 + 128 * 16) * 4
    );
}

#[test]
fn test_bf16_zero3_partitions_params_grads_optimizer_and_master_weights() {
    let cfg = tiny();
    let report = training_memory_report(
        &cfg,
        TrainingMemoryConfig {
            num_layers: 1,
            precision: Precision::Bf16,
            activation_checkpointing: ActivationCheckpointing::Selective,
            data_parallel_shards: 4,
            zero_stage: ZeroStage::Stage3,
        },
    );

    assert_eq!(report.parameter_bytes, 5_104);
    assert_eq!(report.gradient_bytes, 5_104);
    assert_eq!(report.optimizer_state_bytes, 20_416);
    assert_eq!(report.master_parameter_bytes, 10_208);
    assert_eq!(report.activation_bytes, 2_408);
    assert_eq!(report.kv_cache_bytes, 3_248);
    assert_eq!(report.total_training_bytes(), 43_240);
}

#[test]
fn test_kv_cache_uses_model_dimension_for_existing_transformer_config() {
    let cfg = tiny();

    assert_eq!(kv_cache_memory_bytes(&cfg, 2, Precision::Bf16), 6_496);
}

#[test]
fn test_zero_partition_summary_matches_existing_fsdp_ladder() {
    let stats = model_stats(&tiny());
    let summary = zero_partitioned_memory(&stats, 8);
    let full = stats.params_total as u64 * 4;

    assert_eq!(summary.stage1.param_bytes, full);
    assert_eq!(summary.stage1.grad_bytes, full);
    assert_eq!(summary.stage1.optimizer_state_bytes, full * 2 / 8);
    assert_eq!(summary.stage2.param_bytes, full);
    assert_eq!(summary.stage2.grad_bytes, full / 8);
    assert_eq!(summary.stage3.param_bytes, full / 8);
    assert_eq!(summary.stage3.grad_bytes, full / 8);
    assert!(summary.stage3.total_bytes() <= summary.stage2.total_bytes());
    assert!(summary.stage2.total_bytes() <= summary.stage1.total_bytes());
}

#[test]
fn test_stage1_and_stage2_training_report_partition_goldens() {
    let cfg = tiny();
    let stage1 = training_memory_report(
        &cfg,
        TrainingMemoryConfig {
            data_parallel_shards: 8,
            zero_stage: ZeroStage::Stage1,
            ..TrainingMemoryConfig::default()
        },
    );
    let stage2 = training_memory_report(
        &cfg,
        TrainingMemoryConfig {
            data_parallel_shards: 8,
            zero_stage: ZeroStage::Stage2,
            ..TrainingMemoryConfig::default()
        },
    );

    assert_eq!(stage1.parameter_bytes, 40_832);
    assert_eq!(stage1.gradient_bytes, 40_832);
    assert_eq!(stage1.optimizer_state_bytes, 10_208);
    assert_eq!(stage2.parameter_bytes, 40_832);
    assert_eq!(stage2.gradient_bytes, 5_104);
    assert_eq!(stage2.optimizer_state_bytes, 10_208);
}

#[test]
fn test_zero_partition_summary_uses_same_ceil_rounding_as_report() {
    let cfg = TransformerConfig {
        batch: 1,
        seq: 1,
        d_model: 5,
        d_ff: 13,
        n_heads: 1,
    };
    let stats = model_stats(&cfg);
    let report = training_memory_report(
        &cfg,
        TrainingMemoryConfig {
            data_parallel_shards: 3,
            zero_stage: ZeroStage::Stage3,
            ..TrainingMemoryConfig::default()
        },
    );
    let summary = zero_partitioned_memory(&stats, 3);

    assert_eq!(stats.params_total, 250);
    assert_eq!(report.parameter_bytes, 334);
    assert_eq!(report.gradient_bytes, 334);
    assert_eq!(report.optimizer_state_bytes, 667);
    assert_eq!(summary.stage3.param_bytes, report.parameter_bytes);
    assert_eq!(summary.stage3.grad_bytes, report.gradient_bytes);
    assert_eq!(
        summary.stage3.optimizer_state_bytes,
        report.optimizer_state_bytes
    );
}
