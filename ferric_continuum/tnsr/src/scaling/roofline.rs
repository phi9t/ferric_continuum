//! Symbolic roofline model.
//!
//! Book reference: Ch.1 "Roofline Analysis",
//! https://jax-ml.github.io/scaling-book/roofline/
//!
//! Given a workload's FLOPs and bytes, and hardware peak performance, the
//! roofline model predicts the theoretical time bound:
//!
//! ```text
//! arithmetic_intensity = FLOPs / bytes
//! compute_bound  = FLOPs / peak_flops_per_sec
//! memory_bound   = bytes / memory_bandwidth
//! actual_time    >= max(compute_bound, memory_bound)
//! ```
//!
//! If arithmetic_intensity > ridge_point, the op is compute-bound; otherwise
//! it is memory-bandwidth-bound.  Most transformer ops are memory-bound at
//! small batch sizes.
//!
//! **Note**: tnsr is a CPU-only reference implementation — these numbers are
//! symbolic estimates intended to illustrate the concept, not measured results.

/// Theoretical peak hardware numbers for a reference accelerator.
#[derive(Debug, Clone)]
pub struct HardwareSpec {
    /// Theoretical peak FLOP/s (e.g. 312e12 for A100 BF16 tensor-core).
    pub peak_flops_per_sec: f64,
    /// Memory bandwidth in bytes/sec (e.g. 2e12 for A100 HBM).
    pub memory_bandwidth: f64,
}

impl HardwareSpec {
    /// Ridge point: minimum arithmetic intensity (FLOPs/byte) to be compute-bound.
    pub fn ridge_point(&self) -> f64 {
        self.peak_flops_per_sec / self.memory_bandwidth
    }
}

/// Whether the operation is bottlenecked by compute or memory bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
}

/// Roofline estimate for a single operation.
#[derive(Debug, Clone)]
pub struct RooflineEstimate {
    /// FLOPs per byte of data moved.
    pub arithmetic_intensity: f64,
    /// Theoretical time if compute-bound (seconds).
    pub compute_bound_secs: f64,
    /// Theoretical time if memory-bound (seconds).
    pub memory_bound_secs: f64,
    /// Which bound is tighter.
    pub bottleneck: Bottleneck,
}

/// Compute a roofline estimate for an operation with `flops` multiply-adds
/// and `bytes` of data movement on hardware `hw`.
pub fn roofline(flops: u64, bytes: u64, hw: &HardwareSpec) -> RooflineEstimate {
    let f = flops as f64;
    let b = bytes as f64;
    let intensity = if b > 0.0 { f / b } else { f64::INFINITY };
    let compute_secs = f / hw.peak_flops_per_sec;
    let memory_secs = b / hw.memory_bandwidth;
    let bottleneck = if intensity >= hw.ridge_point() {
        Bottleneck::Compute
    } else {
        Bottleneck::Memory
    };
    RooflineEstimate {
        arithmetic_intensity: intensity,
        compute_bound_secs: compute_secs,
        memory_bound_secs: memory_secs,
        bottleneck,
    }
}

/// Hardcoded reference spec for an NVIDIA A100 80 GB with BF16 tensor cores.
/// Source: NVIDIA A100 datasheet.
pub fn a100_bf16() -> HardwareSpec {
    HardwareSpec {
        peak_flops_per_sec: 312e12, // 312 TFLOP/s BF16
        memory_bandwidth: 2.0e12,   // 2 TB/s HBM2e
    }
}

/// Hardcoded reference spec for a Google TPU v4.
/// Source: Google Cloud TPU v4 documentation.
pub fn tpu_v4() -> HardwareSpec {
    HardwareSpec {
        peak_flops_per_sec: 275e12, // 275 TFLOP/s BF16
        memory_bandwidth: 1.2e12,   // 1.2 TB/s HBM
    }
}
