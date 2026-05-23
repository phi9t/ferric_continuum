//! Symbolic 5D-parallelism cost model (Ultra-Scale Playbook).
//!
//! Reference: HuggingFace "The Ultra-Scale Playbook",
//! <https://huggingface.co/spaces/nanotron/ultrascale-playbook>.
//! Complements the JAX scaling book Ch.5 ("training",
//! <https://jax-ml.github.io/scaling-book/training/>).
//!
//! tnsr is CPU-only and single-threaded — NO real collective communication
//! happens here.  This module is a **local algebra** over the parallelism
//! degrees: it computes per-device memory, per-device FLOPs, and the
//! communication *volume* (bytes) a real cluster would move, plus the pipeline
//! bubble.  It does not execute collectives, simulate latency/overlap, choose
//! placement, or implement a distributed runtime.
//!
//! # Assumptions / limitations
//! - f32 tensors only; the only width knob is `optimizer_bytes_per_param`
//!   (Adam = 8 = fp32 m + fp32 v; SGD-momentum = 4; SGD = 0).
//! - `cfg.batch` is the **local batch per data-parallel rank per optimizer
//!   step**; `num_microbatches` splits that local batch (so `batch % m == 0`).
//! - Communication is reported as per-device bytes per optimizer step.
//! - Memory fields are *persistent* estimates; ZeRO-3/FSDP transient
//!   parameter-gather working sets are NOT included.
//! - MoE routing is assumed balanced (no capacity-factor / drop modeling).
//! - Per-device FLOPs are the **max-stage** estimate (the slowest pipeline
//!   stage bottlenecks throughput), using `S = ceil(num_layers / pp)` layers.
//! - Sharding degrees must evenly divide what they shard (`tp | d_model`,
//!   `tp | d_ff`, `cp | seq`, `ep | num_experts`, `dp | model-parallel param
//!   count`); `validate` rejects configs that would otherwise truncate or
//!   floor a reported figure to zero.  MoE per-token routing is the one
//!   exception — it is assumed balanced and may round.
//!
//! # Notation
//! `B`=local step batch, `T`=seq, `D`=d_model, `F`=d_ff, `L`=num_layers,
//! `m`=num_microbatches, `S = ceil(L/pp)` layers per pipeline stage.
//! Parallel degrees: `dp` data, `tp` tensor, `pp` pipeline, `cp` context,
//! `ep` expert.  World size = `dp·tp·pp·cp·ep`.

use crate::transformer::TransformerConfig;

const F32: u128 = super::F32_BYTES as u128;

/// ZeRO / FSDP sharding stage applied across the data-parallel axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroStage {
    /// Stage 0: params, grads, optimizer state fully replicated on every DP
    /// rank.  Gradient sync is a plain all-reduce.
    None,
    /// Stage 1: optimizer state sharded across `dp`; grads/params replicated.
    OptimizerState,
    /// Stage 2: optimizer state + gradients sharded across `dp`.
    OptimizerGradients,
    /// Stage 3 / FSDP: optimizer state + gradients + parameters all sharded
    /// across `dp` (params gathered just-in-time per layer in real systems).
    Full,
}

impl ZeroStage {
    fn shards_optimizer(self) -> bool {
        matches!(
            self,
            ZeroStage::OptimizerState | ZeroStage::OptimizerGradients | ZeroStage::Full
        )
    }
    fn shards_gradients(self) -> bool {
        matches!(self, ZeroStage::OptimizerGradients | ZeroStage::Full)
    }
    fn shards_parameters(self) -> bool {
        matches!(self, ZeroStage::Full)
    }
}

/// Pipeline scheduling strategy (Ultra-Scale Playbook, pipeline parallelism).
///
/// Both schedules share the same bubble `(pp-1)/m`; 1F1B is a *memory*
/// optimization that bounds the number of in-flight microbatches to ~`pp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// All-Forward-All-Backward: all `m` microbatches forward, then all
    /// backward.  Stores activations for all `m` in-flight microbatches.
    Afab,
    /// 1F1B (one-forward-one-backward): interleaves to bound in-flight
    /// activations to `min(pp, m)` microbatches.
    OneF1B,
}

/// Mixture-of-Experts configuration for expert parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    /// Total number of experts in each MoE layer.
    pub num_experts: usize,
    /// Number of experts each token is routed to (1 = Switch-style, 2 = many
    /// modern MoEs).
    pub top_k: usize,
}

/// The five parallelism degrees plus ZeRO stage and pipeline settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelConfig {
    pub dp: usize,
    pub tp: usize,
    pub pp: usize,
    pub cp: usize,
    pub ep: usize,
    pub zero: ZeroStage,
    /// Number of pipeline micro-batches (`m` in the bubble `(pp-1)/m`).
    pub num_microbatches: usize,
    pub schedule: PipelineSchedule,
    /// Bytes of optimizer state per parameter (Adam = 8, SGD-momentum = 4).
    pub optimizer_bytes_per_param: u64,
    /// `None` = dense model (requires `ep == 1`); `Some` = MoE layers.
    pub moe: Option<MoeConfig>,
}

impl ParallelConfig {
    /// Pure data-parallel baseline: `dp` replicas, no other axis, no ZeRO.
    pub fn data_parallel(dp: usize) -> Self {
        Self {
            dp,
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

    /// Total devices = `dp·tp·pp·cp·ep`.
    pub fn world_size(&self) -> usize {
        self.dp * self.tp * self.pp * self.cp * self.ep
    }

    /// Model-parallel parameter count held on one device (one pipeline stage,
    /// sharded by `tp` and — for MoE — by `ep`), before any ZeRO/dp sharding.
    ///
    /// Shared by [`Self::validate`] and [`parallel_cost`] so the divisibility
    /// guard and the reported memory can never diverge.  Assumes `tp`/`ep`
    /// already divide their dimensions (enforced by `validate`).
    fn model_parallel_params(&self, cfg: &TransformerConfig, num_layers: usize) -> u128 {
        let d = cfg.d_model as u128;
        let f = cfg.d_ff as u128;
        let tp = self.tp as u128;
        let s = num_layers.div_ceil(self.pp) as u128;
        let p_attn = 4 * d * d;
        let p_mlp = 2 * d * f;
        let p_norm = 4 * d;
        let mlp_local = match self.moe {
            Some(m) => (m.num_experts / self.ep) as u128 * p_mlp / tp,
            None => p_mlp / tp,
        };
        s * (p_attn / tp + mlp_local + p_norm)
    }

    /// Validate the configuration against a model.  Returns `Err` with a
    /// human-readable message for any inconsistency.
    pub fn validate(&self, cfg: &TransformerConfig, num_layers: usize) -> Result<(), String> {
        for (name, deg) in [
            ("dp", self.dp),
            ("tp", self.tp),
            ("pp", self.pp),
            ("cp", self.cp),
            ("ep", self.ep),
        ] {
            if deg == 0 {
                return Err(format!("{name} degree must be >= 1"));
            }
        }
        if self.num_microbatches == 0 {
            return Err("num_microbatches must be >= 1".into());
        }
        if num_layers == 0 {
            return Err("num_layers must be >= 1".into());
        }
        if self.pp > num_layers {
            return Err(format!(
                "pp ({}) must be <= num_layers ({num_layers})",
                self.pp
            ));
        }
        if cfg.batch % self.num_microbatches != 0 {
            return Err(format!(
                "batch ({}) must be divisible by num_microbatches ({})",
                cfg.batch, self.num_microbatches
            ));
        }
        if cfg.d_model % self.tp != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by tp ({})",
                cfg.d_model, self.tp
            ));
        }
        if cfg.d_ff % self.tp != 0 {
            return Err(format!(
                "d_ff ({}) must be divisible by tp ({})",
                cfg.d_ff, self.tp
            ));
        }
        if cfg.seq % self.cp != 0 {
            return Err(format!(
                "seq ({}) must be divisible by cp ({})",
                cfg.seq, self.cp
            ));
        }
        match self.moe {
            None => {
                if self.ep != 1 {
                    return Err("ep must be 1 when moe is None (dense model)".into());
                }
            }
            Some(m) => {
                if m.num_experts % self.ep != 0 {
                    return Err(format!(
                        "num_experts ({}) must be divisible by ep ({})",
                        m.num_experts, self.ep
                    ));
                }
                if m.top_k == 0 || m.top_k > m.num_experts {
                    return Err(format!(
                        "top_k ({}) must be in 1..=num_experts ({})",
                        m.top_k, m.num_experts
                    ));
                }
            }
        }
        // ZeRO sharding and the dp gradient all-reduce divide the model-parallel
        // parameter count by `dp`; require an even split so reported bytes are
        // exact and never floor to zero for an oversized `dp`.
        if self.dp > 1 {
            let mp = self.model_parallel_params(cfg, num_layers);
            if mp % self.dp as u128 != 0 {
                return Err(format!(
                    "model-parallel param count ({mp}) must be divisible by dp ({}) \
                     for exact ZeRO / all-reduce sharding",
                    self.dp
                ));
            }
        }
        Ok(())
    }
}

/// Per-device cost of training a model under a [`ParallelConfig`].
///
/// Byte fields are per device; comm fields are bytes that axis moves per
/// optimizer step; FLOPs are the max-stage per-device estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelCost {
    pub world_size: usize,
    /// `ceil(num_layers / pp)` — layers resident on this pipeline stage.
    pub layers_per_stage: usize,
    /// `batch / num_microbatches`.
    pub microbatch_size: usize,

    // ---- per-device memory (bytes) ----
    pub param_bytes: u64,
    pub grad_bytes: u64,
    pub optimizer_bytes: u64,
    pub activation_bytes: u64,
    /// param + grad + optimizer.
    pub persistent_mem_bytes: u64,
    /// persistent + activation.
    pub peak_mem_bytes: u64,

    // ---- per-device compute (max-stage) ----
    pub flops_per_device: u128,

    // ---- communication volume (bytes / optimizer step) ----
    pub tp_comm_bytes: u64,
    pub dp_comm_bytes: u64,
    pub pp_comm_bytes: u64,
    pub cp_comm_bytes: u64,
    pub ep_comm_bytes: u64,

    // ---- pipeline ----
    /// `(pp-1)/m` — bubble (idle) time over ideal compute time.
    pub bubble_over_ideal: f64,
    /// `m/(m+pp-1)` — fraction of wall-clock spent doing useful work.
    pub pipeline_efficiency: f64,
    /// `(pp-1)/(m+pp-1)` — fraction of wall-clock spent idle in the bubble.
    pub pipeline_idle_fraction: f64,
}

/// Per-device, per-step bytes moved by an all-gather / reduce-scatter /
/// balanced all-to-all of `elems` f32 elements over `n` ranks.
fn one_way(elems: u128, n: usize) -> u128 {
    if n <= 1 {
        0
    } else {
        (n as u128 - 1) * elems * F32 / n as u128
    }
}

/// Per-device, per-step bytes moved by a ring all-reduce of `elems` f32
/// elements over `n` ranks (= reduce-scatter + all-gather).
fn all_reduce(elems: u128, n: usize) -> u128 {
    2 * one_way(elems, n)
}

/// Bubble fraction over ideal compute time: `(pp-1)/m` (0 if `pp<=1`/`m==0`).
pub fn pipeline_bubble(pp: usize, num_microbatches: usize) -> f64 {
    if pp <= 1 || num_microbatches == 0 {
        0.0
    } else {
        (pp - 1) as f64 / num_microbatches as f64
    }
}

/// Compute per-device training costs for a transformer of `num_layers` blocks
/// described by `cfg` under parallel configuration `pc`.
///
/// Panics if `pc.validate(cfg, num_layers)` fails.
pub fn parallel_cost(
    cfg: &TransformerConfig,
    num_layers: usize,
    pc: &ParallelConfig,
) -> ParallelCost {
    pc.validate(cfg, num_layers)
        .unwrap_or_else(|e| panic!("invalid ParallelConfig: {e}"));

    let b = cfg.batch as u128;
    let t = cfg.seq as u128;
    let d = cfg.d_model as u128;
    let f = cfg.d_ff as u128;

    let tp = pc.tp as u128;
    let cp = pc.cp as u128;
    let ep = pc.ep as u128;
    let m = pc.num_microbatches as u128;

    let layers_per_stage = num_layers.div_ceil(pc.pp);
    let s = layers_per_stage as u128;

    let p_attn = 4 * d * d;
    let p_mlp = 2 * d * f;

    let top_k = match pc.moe {
        Some(moe) => moe.top_k as u128,
        None => 1,
    };

    // ---- parameters (model-parallel local count) ----
    let mp_params = pc.model_parallel_params(cfg, num_layers);

    let param_shard = if pc.zero.shards_parameters() {
        pc.dp as u128
    } else {
        1
    };
    let grad_shard = if pc.zero.shards_gradients() {
        pc.dp as u128
    } else {
        1
    };
    let opt_shard = if pc.zero.shards_optimizer() {
        pc.dp as u128
    } else {
        1
    };

    let param_bytes = mp_params / param_shard * F32;
    let grad_bytes = mp_params / grad_shard * F32;
    let optimizer_bytes = mp_params / opt_shard * pc.optimizer_bytes_per_param as u128;
    let persistent = param_bytes + grad_bytes + optimizer_bytes;

    // ---- activations ----
    let act_micro_bytes = match pc.moe {
        Some(_) => {
            let attn = 2 * b * t * d / (tp * cp);
            let mlp = 2 * b * t * f * top_k / (tp * cp * ep);
            (attn + mlp) * F32 / m
        }
        None => (2 * b * t * d + 2 * b * t * f) * F32 / (tp * cp * m),
    };
    let inflight = match pc.schedule {
        PipelineSchedule::Afab => m,
        PipelineSchedule::OneF1B => (pc.pp as u128).min(m),
    };
    let activation_bytes = act_micro_bytes * s * inflight;

    // ---- FLOPs (max-stage; dp does not divide work) ----
    let mlp_flops = match pc.moe {
        Some(_) => 6 * p_mlp * b * t * top_k / ep,
        None => 6 * p_mlp * b * t,
    };
    let per_layer_flops = 6 * p_attn * b * t + mlp_flops + 12 * b * t * t * d;
    let flops_per_device = s * per_layer_flops / (tp * cp);

    // ---- communication volume ----
    // TP: 2 forward all-reduces per layer (attention + MLP); backward
    // symmetric traffic omitted as a teaching simplification.
    let tp_comm = s * 2 * all_reduce(b * t * d / cp, pc.tp);
    // DP: gradient all-reduce; ZeRO reduce-scatter + all-gather ≈ same volume.
    let dp_comm = all_reduce(mp_params, pc.dp);
    // PP: per stage boundary per step (fwd activation + bwd gradient).
    let pp_comm = if pc.pp > 1 {
        2 * (b * t * d / (tp * cp)) * F32
    } else {
        0
    };
    // CP: ring all-gather of K and V across the sequence shards.
    let cp_comm = s * one_way(2 * b * t * d / tp, pc.cp);
    // EP: dispatch + combine all-to-all of routed tokens.
    let ep_comm = s * 2 * one_way(b * t * d * top_k / (tp * cp), pc.ep);

    let bubble_over_ideal = pipeline_bubble(pc.pp, pc.num_microbatches);
    let (pipeline_efficiency, pipeline_idle_fraction) = if pc.pp > 1 && pc.num_microbatches > 0 {
        let denom = (pc.num_microbatches + pc.pp - 1) as f64;
        (
            pc.num_microbatches as f64 / denom,
            (pc.pp - 1) as f64 / denom,
        )
    } else {
        (1.0, 0.0)
    };

    ParallelCost {
        world_size: pc.world_size(),
        layers_per_stage,
        microbatch_size: cfg.batch / pc.num_microbatches,
        param_bytes: param_bytes as u64,
        grad_bytes: grad_bytes as u64,
        optimizer_bytes: optimizer_bytes as u64,
        activation_bytes: activation_bytes as u64,
        persistent_mem_bytes: persistent as u64,
        peak_mem_bytes: (persistent + activation_bytes) as u64,
        flops_per_device,
        tp_comm_bytes: tp_comm as u64,
        dp_comm_bytes: dp_comm as u64,
        pp_comm_bytes: pp_comm as u64,
        cp_comm_bytes: cp_comm as u64,
        ep_comm_bytes: ep_comm as u64,
        bubble_over_ideal,
        pipeline_efficiency,
        pipeline_idle_fraction,
    }
}

/// Render a [`ParallelCost`] as a multi-line table (no external deps).
pub fn format_parallel_cost(
    cfg: &TransformerConfig,
    num_layers: usize,
    pc: &ParallelConfig,
    cost: &ParallelCost,
) -> String {
    let mut out = String::new();
    out.push_str("=== tnsr 5D Parallelism Cost ===\n");
    out.push_str(&format!(
        "Model:  B={} T={} D={} F={} L={}\n",
        cfg.batch, cfg.seq, cfg.d_model, cfg.d_ff, num_layers
    ));
    let moe = match pc.moe {
        Some(m) => format!("MoE(experts={}, top_k={})", m.num_experts, m.top_k),
        None => "dense".to_string(),
    };
    out.push_str(&format!(
        "Parallel: dp={} tp={} pp={} cp={} ep={}  world={}  {moe}\n",
        pc.dp, pc.tp, pc.pp, pc.cp, pc.ep, cost.world_size
    ));
    out.push_str(&format!(
        "ZeRO={:?}  schedule={:?}  microbatches={} (size={})  layers/stage={}\n",
        pc.zero, pc.schedule, pc.num_microbatches, cost.microbatch_size, cost.layers_per_stage
    ));
    out.push('\n');

    out.push_str("Memory per device (bytes):\n");
    out.push_str(&format!("  params:      {:>14}\n", cost.param_bytes));
    out.push_str(&format!("  grads:       {:>14}\n", cost.grad_bytes));
    out.push_str(&format!("  optimizer:   {:>14}\n", cost.optimizer_bytes));
    out.push_str(&format!("  activations: {:>14}\n", cost.activation_bytes));
    out.push_str(&format!(
        "  persistent:  {:>14}\n",
        cost.persistent_mem_bytes
    ));
    out.push_str(&format!("  peak:        {:>14}\n", cost.peak_mem_bytes));
    out.push('\n');

    out.push_str(&format!(
        "FLOPs/device (max-stage): {}\n",
        cost.flops_per_device
    ));
    out.push('\n');

    out.push_str("Comm per device (bytes/step):\n");
    out.push_str(&format!("  tp: {:>14}\n", cost.tp_comm_bytes));
    out.push_str(&format!("  dp: {:>14}\n", cost.dp_comm_bytes));
    out.push_str(&format!("  pp: {:>14}\n", cost.pp_comm_bytes));
    out.push_str(&format!("  cp: {:>14}\n", cost.cp_comm_bytes));
    out.push_str(&format!("  ep: {:>14}\n", cost.ep_comm_bytes));
    out.push('\n');

    out.push_str(&format!(
        "Pipeline: bubble/ideal={:.4}  efficiency={:.4}  idle={:.4}\n",
        cost.bubble_over_ideal, cost.pipeline_efficiency, cost.pipeline_idle_fraction
    ));
    out
}
