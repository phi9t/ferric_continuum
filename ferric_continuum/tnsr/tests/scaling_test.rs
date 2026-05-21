use tnsr::scaling::inference::{kv_cache_bytes, peak_activation_bytes};
use tnsr::scaling::model_stats::model_stats;
use tnsr::scaling::op_cost::{dense_matmul_fwd_flops, total_fwd_flops, total_train_flops};
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
