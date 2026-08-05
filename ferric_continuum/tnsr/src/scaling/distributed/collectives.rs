//! Collective communication primitives — cost model + single-process sims.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/> (the "collective ops"
//! sidebar) and the sharding chapter's all-reduce accounting.
//!
//! PyTorch counterpart: `torch.distributed` collectives —
//! [`all_reduce`], [`all_gather`], [`reduce_scatter`], [`broadcast`].
//!
//! [`all_reduce`]: https://pytorch.org/docs/stable/distributed.html
//! [`all_gather`]: https://pytorch.org/docs/stable/distributed.html
//! [`reduce_scatter`]: https://pytorch.org/docs/stable/distributed.html
//! [`broadcast`]: https://pytorch.org/docs/stable/distributed.html
//!
//! # Two views of a collective
//!
//! Everything in this file describes a collective in two complementary ways:
//!
//! 1. **Cost** — [`collective_cost`] returns the *bytes moved per device* under
//!    the standard **ring** algorithm. `tnsr` is CPU-only and single-threaded,
//!    so no bytes actually cross a wire; the number is a symbolic estimate you
//!    can plug into [`super::super::roofline`].
//! 2. **Behaviour** — the `sim_*` functions execute the collective *exactly*
//!    over a `&[Vec<f32>]` where each inner `Vec` is one logical device's
//!    shard. The loop over `D` shards runs in one process, so the result is
//!    provably correct without any devices, threads, or network.
//!
//! # The ring model (why the constants look the way they do)
//!
//! A ring collective sends data around `D` devices in `D−1` steps, each step
//! moving `bytes/D` of payload. Summing the steps:
//!
//! ```text
//! all-gather / reduce-scatter / broadcast :  (D−1)/D · bytes   per device
//! all-reduce  = reduce-scatter + all-gather:  2·(D−1)/D · bytes per device
//! ```
//!
//! An **all-reduce** is exactly a **reduce-scatter** followed by an
//! **all-gather** (each device ends up owning the fully-summed vector), which
//! is why its cost is the sum of the other two — and why
//! [`sim_ring_all_reduce_sum`] is implemented as that composition and must
//! agree bit-for-bit with the naive [`sim_all_reduce_sum`].
//!
//! `all-to-all` (used by expert / sequence parallelism) is deliberately **not**
//! modelled here; it needs a permutation across shards that the single-matmul
//! teaching scope does not cover.

use super::super::F32_BYTES;

/// One of the four collectives modelled by this module.
///
/// The discriminants mirror the `torch.distributed` op names so the teaching
/// mapping is one-to-one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collective {
    /// Sum (or other reduction) a vector across all devices; every device ends
    /// up with the full reduced result. `torch.distributed.all_reduce`.
    AllReduce,
    /// Concatenate every device's shard so all devices hold the full vector.
    /// `torch.distributed.all_gather`.
    AllGather,
    /// Sum shard-by-shard, leaving each device with one reduced slice.
    /// `torch.distributed.reduce_scatter`.
    ReduceScatter,
    /// Copy one device's buffer to every device. `torch.distributed.broadcast`.
    Broadcast,
}

/// Communication volume for one collective under the ring model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveCost {
    /// Which collective this describes.
    pub collective: Collective,
    /// Number of participating devices (`D`).
    pub num_devices: usize,
    /// Logical payload size in bytes (the full vector, not the per-shard slice).
    pub payload_bytes: u64,
    /// Bytes that traverse one device's links, `≈ k·(D−1)/D · payload` where
    /// `k = 2` for all-reduce and `k = 1` otherwise.
    pub comm_bytes_per_device: u64,
}

/// Bytes moved per device for `c` over `num_devices` devices with a
/// `payload_bytes`-sized logical vector, using the ring algorithm.
///
/// With a single device there is nothing to communicate, so the result is `0`
/// (the `(D−1)/D` factor already gives this, but we short-circuit for clarity).
pub fn collective_cost(c: Collective, num_devices: usize, payload_bytes: u64) -> CollectiveCost {
    let d = num_devices as u64;
    // (D−1)/D · payload, computed as payload·(D−1)/D to avoid truncating early.
    let per_leg = if d <= 1 {
        0
    } else {
        payload_bytes * (d - 1) / d
    };
    let comm_bytes_per_device = match c {
        // all-reduce = reduce-scatter + all-gather → two legs.
        Collective::AllReduce => 2 * per_leg,
        Collective::AllGather | Collective::ReduceScatter | Collective::Broadcast => per_leg,
    };
    CollectiveCost {
        collective: c,
        num_devices,
        payload_bytes,
        comm_bytes_per_device,
    }
}

/// Convenience: byte count of a single f32 shard vector of `len` elements.
pub fn f32_vec_bytes(len: usize) -> u64 {
    len as u64 * F32_BYTES
}

// ---------------------------------------------------------------------------
// Functional simulations over &[Vec<f32>]
//
// Each `shards[i]` is device i's local buffer. The functions loop over all
// shards in one process — this is the "run it to prove it" half of the module.
// ---------------------------------------------------------------------------

/// Element-wise sum of every device's vector, delivered to **all** devices.
///
/// This is the naive/reference implementation: sum first, then hand every
/// device a clone of the total. All input vectors must have equal length.
///
/// Returns one summed vector per device (all identical).
pub fn sim_all_reduce_sum(shards: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(!shards.is_empty(), "all-reduce needs at least one shard");
    let len = shards[0].len();
    assert!(
        shards.iter().all(|s| s.len() == len),
        "all shards must share a length"
    );

    // ── Step 1: reduce — accumulate every shard into one total ───────────────
    let mut total = vec![0.0f32; len];
    for shard in shards {
        for (t, &v) in total.iter_mut().zip(shard.iter()) {
            *t += v;
        }
    }

    // ── Step 2: broadcast — every device receives the same total ─────────────
    vec![total; shards.len()]
}

/// Ring all-reduce: **reduce-scatter followed by all-gather**.
///
/// Bit-for-bit identical to [`sim_all_reduce_sum`] on inputs whose partial sums
/// are exactly representable (e.g. integers stored as f32), because the ring
/// algorithm computes the same reduction — just distributed across legs. This
/// composition is the whole point: the cost `2·(D−1)/D·bytes` falls out of
/// running one leg of each sub-collective.
pub fn sim_ring_all_reduce_sum(shards: &[Vec<f32>]) -> Vec<Vec<f32>> {
    // ── Step 1: reduce-scatter → each device owns one summed slice ───────────
    let scattered = sim_reduce_scatter_sum(shards);

    // ── Step 2: all-gather → concatenate the slices back to the full vector ──
    sim_all_gather(&scattered)
}

/// Concatenate every device's shard, delivering the full vector to all devices.
///
/// Unlike the reductions, shards may have *different* lengths (ragged
/// partitions are legal for gather). The gathered vector is the shards laid
/// end-to-end in device order.
pub fn sim_all_gather(shards: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(!shards.is_empty(), "all-gather needs at least one shard");
    let mut full = Vec::new();
    for shard in shards {
        full.extend_from_slice(shard);
    }
    vec![full; shards.len()]
}

/// Sum shards element-wise, then hand device `i` the `i`-th equal slice.
///
/// The summed vector's length must be divisible by the device count so the
/// scatter is even (the transformer sharding cases always partition evenly).
/// Returns one slice per device; concatenating them (an all-gather) reproduces
/// the full all-reduced vector — the invariant `reduce_scatter ∘ all_gather ==
/// all_reduce` that the tests check.
pub fn sim_reduce_scatter_sum(shards: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(
        !shards.is_empty(),
        "reduce-scatter needs at least one shard"
    );
    let len = shards[0].len();
    let d = shards.len();
    assert!(
        shards.iter().all(|s| s.len() == len),
        "all shards must share a length"
    );
    assert!(
        len % d == 0,
        "reduce-scatter needs len ({len}) divisible by device count ({d})"
    );

    // ── Step 1: reduce — full element-wise sum ───────────────────────────────
    let mut total = vec![0.0f32; len];
    for shard in shards {
        for (t, &v) in total.iter_mut().zip(shard.iter()) {
            *t += v;
        }
    }

    // ── Step 2: scatter — slice the total into D equal, contiguous pieces ────
    let slice = len / d;
    (0..d)
        .map(|i| total[i * slice..(i + 1) * slice].to_vec())
        .collect()
}

/// Copy device `root`'s buffer to every device.
///
/// Mirrors `torch.distributed.broadcast(tensor, src=root)`: the other shards'
/// current contents are discarded and replaced by `root`'s.
pub fn sim_broadcast(shards: &[Vec<f32>], root: usize) -> Vec<Vec<f32>> {
    assert!(!shards.is_empty(), "broadcast needs at least one shard");
    assert!(
        root < shards.len(),
        "broadcast root {root} out of range for {} devices",
        shards.len()
    );
    let source = shards[root].clone();
    vec![source; shards.len()]
}
