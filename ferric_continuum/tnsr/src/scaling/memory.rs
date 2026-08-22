//! Training and inference memory estimates for transformer blocks.
//!
//! This module complements [`super::model_stats`], [`super::report`], and
//! [`super::distributed::fsdp`]: it owns byte accounting, while the distributed
//! module remains the source of truth for ZeRO/FSDP stage semantics and
//! communication costs.
//!
//! Formulas are symbolic and allocation-free. They intentionally model the
//! `TransformerConfig` block used by this crate rather than a generic LLM
//! schema:
//!
//! - parameters: `params_total * parameter_bytes`
//! - gradients: `params_total * gradient_bytes`
//! - Adam optimizer state: `params_total * 2 * optimizer_state_bytes`
//! - bf16 mixed precision master weights: `params_total * 4`
//! - KV cache: `2 * layers * B * T * D * elem_bytes`
//! - activations without checkpointing:
//!   `layers * (10*B*T*D + 2*B*T*F + B*T*T) * elem_bytes`
//! - selective activations mirror `TransformerSelectivePolicy` at the formula
//!   level: save the runtime inputs for attention scores and attention mix,
//!   save small softmax outputs, and recompute GELU, LayerNorm, and large
//!   Linear sites.

use crate::transformer::TransformerConfig;

use super::distributed::fsdp::FsdpMemory;
use super::model_stats::{model_stats, ModelStats};

/// Default `TransformerSelectivePolicy::save_softmax_under_bytes` used by this
/// formula-level estimate.
const SELECTIVE_SAVE_BELOW_BYTES: u64 = 4096;

/// Numeric precision used for model tensors in a memory estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Native f32 training: params, grads, and optimizer state are f32.
    F32,
    /// bf16 mixed precision: params and grads are bf16, Adam state and master
    /// weights remain f32.
    Bf16,
}

impl Precision {
    /// Bytes used by a stored parameter element.
    pub fn parameter_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }

    /// Bytes used by a gradient element.
    pub fn gradient_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }

    /// Bytes used by one optimizer-state element. Adam keeps two such buffers.
    pub fn optimizer_state_bytes(self) -> u64 {
        match self {
            Self::F32 | Self::Bf16 => 4,
        }
    }

    /// Bytes used by a fp32 master-weight element for mixed-precision training.
    pub fn master_parameter_bytes(self) -> u64 {
        match self {
            Self::F32 => 0,
            Self::Bf16 => 4,
        }
    }
}

/// Activation memory policy for formula estimates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationCheckpointing {
    /// Store all listed forward activation sites.
    None,
    /// Mirror the runtime `TransformerSelectivePolicy` shape decisions: save
    /// the inputs used by AttentionScores and AttentionMix, save only small
    /// Softmax outputs, and recompute GELU, LayerNorm, and large Linear sites.
    Selective,
    /// Store no internal activation sites; recompute the block from boundaries.
    Full,
}

/// ZeRO stage for pure byte partitioning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroStage {
    /// DDP-style full replicas of params, gradients, and optimizer state.
    Stage0,
    /// Shard optimizer state only.
    Stage1,
    /// Shard optimizer state and gradients.
    Stage2,
    /// Shard optimizer state, gradients, and parameters; equivalent to FSDP.
    Stage3,
}

/// Inputs for [`training_memory_report`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingMemoryConfig {
    /// Number of identical transformer blocks in the model.
    pub num_layers: usize,
    /// Numeric precision used for parameter, gradient, and optimizer estimates.
    pub precision: Precision,
    /// Activation checkpointing/rematerialization policy.
    pub activation_checkpointing: ActivationCheckpointing,
    /// Data-parallel partition count for ZeRO stages. Values below 1 are
    /// treated as 1 because there is no fractional device.
    pub data_parallel_shards: usize,
    /// ZeRO partitioning stage applied to params, grads, optimizer state, and
    /// mixed-precision master weights.
    pub zero_stage: ZeroStage,
}

impl Default for TrainingMemoryConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            precision: Precision::F32,
            activation_checkpointing: ActivationCheckpointing::None,
            data_parallel_shards: 1,
            zero_stage: ZeroStage::Stage0,
        }
    }
}

/// Per-device memory report for one model shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingMemoryReport {
    /// Per-block model statistics used by every formula.
    pub stats: ModelStats,
    /// Number of blocks represented by this report.
    pub num_layers: usize,
    /// Numeric precision used for byte multipliers.
    pub precision: Precision,
    /// ZeRO partitioning stage.
    pub zero_stage: ZeroStage,
    /// Effective data-parallel shard count.
    pub data_parallel_shards: usize,
    /// Parameter bytes per device.
    pub parameter_bytes: u64,
    /// Gradient bytes per device.
    pub gradient_bytes: u64,
    /// Optimizer-state bytes per device.
    pub optimizer_state_bytes: u64,
    /// Mixed-precision fp32 master-weight bytes per device.
    pub master_parameter_bytes: u64,
    /// Forward activation bytes per device under the checkpointing policy.
    pub activation_bytes: u64,
    /// KV-cache bytes for inference with the same layer/shape/precision.
    pub kv_cache_bytes: u64,
}

impl TrainingMemoryReport {
    /// Training footprint excluding the inference-only KV cache.
    pub fn total_training_bytes(&self) -> u64 {
        self.parameter_bytes
            + self.gradient_bytes
            + self.optimizer_state_bytes
            + self.master_parameter_bytes
            + self.activation_bytes
    }

    /// Training footprint plus the KV-cache estimate.
    pub fn total_with_kv_cache_bytes(&self) -> u64 {
        self.total_training_bytes() + self.kv_cache_bytes
    }
}

/// Summary of the existing ZeRO/FSDP ladder for a single precision-agnostic
/// f32 model replica.
///
/// Unlike `distributed::fsdp::fsdp_memory`, this public memory-report view uses
/// ceil division for sharded byte counts so every logical byte is assigned to a
/// device even when the tensor size is not divisible by the shard count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroPartitionSummary {
    pub stage1: FsdpMemory,
    pub stage2: FsdpMemory,
    pub stage3: FsdpMemory,
}

/// Estimate per-device training memory for `cfg`.
pub fn training_memory_report(
    cfg: &TransformerConfig,
    memory_cfg: TrainingMemoryConfig,
) -> TrainingMemoryReport {
    let stats = model_stats(cfg);
    let num_layers = memory_cfg.num_layers.max(1);
    let shards = memory_cfg.data_parallel_shards.max(1) as u64;
    let params = stats.params_total as u64 * num_layers as u64;

    let raw_parameter_bytes = params * memory_cfg.precision.parameter_bytes();
    let raw_gradient_bytes = params * memory_cfg.precision.gradient_bytes();
    let raw_optimizer_state_bytes = params * 2 * memory_cfg.precision.optimizer_state_bytes();
    let raw_master_parameter_bytes = params * memory_cfg.precision.master_parameter_bytes();

    let parameter_bytes = shard_for_stage(
        raw_parameter_bytes,
        shards,
        matches!(memory_cfg.zero_stage, ZeroStage::Stage3),
    );
    let gradient_bytes = shard_for_stage(
        raw_gradient_bytes,
        shards,
        matches!(memory_cfg.zero_stage, ZeroStage::Stage2 | ZeroStage::Stage3),
    );
    let optimizer_state_bytes = shard_for_stage(
        raw_optimizer_state_bytes,
        shards,
        matches!(
            memory_cfg.zero_stage,
            ZeroStage::Stage1 | ZeroStage::Stage2 | ZeroStage::Stage3
        ),
    );
    let master_parameter_bytes = shard_for_stage(
        raw_master_parameter_bytes,
        shards,
        matches!(memory_cfg.zero_stage, ZeroStage::Stage3),
    );

    TrainingMemoryReport {
        stats,
        num_layers,
        precision: memory_cfg.precision,
        zero_stage: memory_cfg.zero_stage,
        data_parallel_shards: shards as usize,
        parameter_bytes,
        gradient_bytes,
        optimizer_state_bytes,
        master_parameter_bytes,
        activation_bytes: activation_memory_bytes(
            cfg,
            num_layers,
            memory_cfg.precision,
            memory_cfg.activation_checkpointing,
        ),
        kv_cache_bytes: kv_cache_memory_bytes(cfg, num_layers, memory_cfg.precision),
    }
}

/// Estimate forward activation bytes for `num_layers` transformer blocks.
///
/// The no-checkpoint formula tracks repeated block save sites by shape:
///
/// `10*B*T*D` for token-dimension activations, `2*B*T*F` for MLP hidden
/// activations, and `B*T*T` for attention probabilities/scores. Selective mode
/// keeps the runtime inputs saved by `TransformerSelectivePolicy`:
/// `AttentionScores` saves Q and K (`2*B*T*D`), `AttentionMix` saves attention
/// probabilities and V (`B*T*T + B*T*D`), and `Softmax` saves its output
/// (`B*T*T`) only when the site is below the small-save threshold. Full
/// checkpointing keeps no internal sites.
pub fn activation_memory_bytes(
    cfg: &TransformerConfig,
    num_layers: usize,
    precision: Precision,
    checkpointing: ActivationCheckpointing,
) -> u64 {
    let sites = ActivationSites::new(cfg);
    let elem_bytes = precision.parameter_bytes();
    let layers = num_layers.max(1) as u64;
    let elems = match checkpointing {
        ActivationCheckpointing::None => sites.full_elements(),
        ActivationCheckpointing::Selective => sites.selective_elements(elem_bytes),
        ActivationCheckpointing::Full => 0,
    };
    elems * elem_bytes * layers
}

/// Estimate KV-cache bytes for autoregressive inference.
///
/// Formula: `2 * layers * B * T * D * elem_bytes`, where `2` accounts for one
/// key tensor and one value tensor per layer. The local `TransformerConfig`
/// exposes only full model dimension `D`, so this intentionally mirrors
/// [`super::inference::kv_cache_bytes`] rather than inventing head-group inputs.
pub fn kv_cache_memory_bytes(
    cfg: &TransformerConfig,
    num_layers: usize,
    precision: Precision,
) -> u64 {
    super::inference::kv_cache_bytes(
        num_layers.max(1),
        cfg.batch,
        cfg.seq,
        cfg.d_model,
        precision.parameter_bytes() as usize,
    )
}

/// Return Stage1/2/3 memory summaries with the same rounding contract as
/// [`training_memory_report`].
///
/// This is a convenience view for memory reports. It reuses the existing
/// distributed FSDP return type, but intentionally omits extra communication
/// costs because ceil-rounded logical byte ownership is a memory-accounting
/// contract, not the communication model owned by [`super::distributed::fsdp`].
pub fn zero_partitioned_memory(
    stats: &ModelStats,
    data_parallel_shards: usize,
) -> ZeroPartitionSummary {
    let shards = data_parallel_shards.max(1) as u64;
    let full_params = stats.params_total as u64 * super::F32_BYTES;
    let full_grads = full_params;
    let full_optimizer = full_params * 2;

    ZeroPartitionSummary {
        stage1: fsdp_stage_memory(
            super::distributed::fsdp::ZeroStage::Stage1,
            shards,
            full_params,
            full_grads,
            shard_bytes(full_optimizer, shards),
        ),
        stage2: fsdp_stage_memory(
            super::distributed::fsdp::ZeroStage::Stage2,
            shards,
            full_params,
            shard_bytes(full_grads, shards),
            shard_bytes(full_optimizer, shards),
        ),
        stage3: fsdp_stage_memory(
            super::distributed::fsdp::ZeroStage::Stage3,
            shards,
            shard_bytes(full_params, shards),
            shard_bytes(full_grads, shards),
            shard_bytes(full_optimizer, shards),
        ),
    }
}

fn shard_for_stage(bytes: u64, shards: u64, sharded: bool) -> u64 {
    if sharded {
        shard_bytes(bytes, shards)
    } else {
        bytes
    }
}

fn shard_bytes(bytes: u64, shards: u64) -> u64 {
    bytes.div_ceil(shards)
}

fn fsdp_stage_memory(
    stage: super::distributed::fsdp::ZeroStage,
    num_shards: u64,
    param_bytes: u64,
    grad_bytes: u64,
    optimizer_state_bytes: u64,
) -> FsdpMemory {
    FsdpMemory {
        stage,
        num_shards: num_shards as usize,
        param_bytes,
        grad_bytes,
        optimizer_state_bytes,
        extra_comm_bytes_per_device: 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActivationSites {
    token_dim_elements: u64,
    token_hidden_elements: u64,
    attention_square_elements: u64,
}

impl ActivationSites {
    fn new(cfg: &TransformerConfig) -> Self {
        let b = cfg.batch as u64;
        let t = cfg.seq as u64;
        Self {
            token_dim_elements: b * t * cfg.d_model as u64,
            token_hidden_elements: b * t * cfg.d_ff as u64,
            attention_square_elements: b * t * t,
        }
    }

    fn full_elements(self) -> u64 {
        10 * self.token_dim_elements
            + 2 * self.token_hidden_elements
            + self.attention_square_elements
    }

    fn selective_elements(self, elem_bytes: u64) -> u64 {
        let attention_scores_inputs = 2 * self.token_dim_elements;
        let attention_mix_inputs = self.attention_square_elements + self.token_dim_elements;
        let softmax = kept_small_site_elements(self.attention_square_elements, elem_bytes);
        attention_scores_inputs + attention_mix_inputs + softmax
    }
}

fn kept_small_site_elements(elements: u64, elem_bytes: u64) -> u64 {
    if elements * elem_bytes <= SELECTIVE_SAVE_BELOW_BYTES {
        elements
    } else {
        0
    }
}
