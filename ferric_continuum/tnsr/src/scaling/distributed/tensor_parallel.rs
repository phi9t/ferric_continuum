//! Tensor parallelism (Megatron-style) — split individual weight matrices.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/> (the "model / tensor
//! parallelism" section) and Ch.3 "Sharded Matrices".
//!
//! PyTorch counterpart: [`torch.distributed.tensor.parallel`]
//! (`ColwiseParallel` / `RowwiseParallel`), i.e. Megatron-LM tensor parallelism.
//!
//! [`torch.distributed.tensor.parallel`]: https://pytorch.org/docs/stable/distributed.tensor.parallel.html
//!
//! # The column-then-row pattern
//!
//! Megatron shards an MLP (or QKV→O) block as a **column-parallel** matmul
//! feeding a **row-parallel** one so that exactly one all-reduce is needed per
//! block:
//!
//! ```text
//! column-parallel  Y = X·W  : shard W by columns (N-dim)  → no comm, Y is col-sharded
//! row-parallel     Z = Y·V  : shard V by rows (K-dim)     → partial sums, all-reduce Z
//! ```
//!
//! The cost half of this file delegates to [`super::super::sharding::shard_matmul`]:
//! the column matmul is [`ShardingCase::WColwise`] (zero collective bytes) and
//! the row matmul is [`ShardingCase::InnerReduction`] (an all-reduce). The
//! behaviour half — [`sim_column_then_row`] — actually runs a single
//! column-parallel matmul across `tp` shards and reassembles the output,
//! proving it equals the unsharded `Y = X·W` element-for-element.

use crate::transformer::TransformerConfig;

use super::super::sharding::{shard_matmul, ShardingCase, ShardingCost};
use super::collectives::sim_all_reduce_sum;

/// Per-device cost of one tensor-parallel MLP block over `tp` devices.
///
/// The two matmuls use `tiny_4_7_29`-scale shapes derived from `cfg`:
///   * up-projection `[B·T, D] · [D, F]` sharded column-wise (no all-reduce),
///   * down-projection `[B·T, F] · [F, D]` sharded row-wise (one all-reduce).
#[derive(Debug, Clone)]
pub struct TensorParallelCost {
    /// Tensor-parallel degree (`tp`).
    pub tp: usize,
    /// Column-parallel up-projection cost (all-reduce bytes are 0).
    pub column: ShardingCost,
    /// Row-parallel down-projection cost (carries the all-reduce).
    pub row: ShardingCost,
}

impl TensorParallelCost {
    /// Total all-reduce bytes per device across the block (row matmul only).
    pub fn allreduce_bytes(&self) -> u64 {
        self.column.allreduce_bytes + self.row.allreduce_bytes
    }
}

/// Cost of tensor-parallel MLP for `cfg` across `tp` devices.
pub fn tensor_parallel_cost(cfg: &TransformerConfig, tp: usize) -> TensorParallelCost {
    let m = cfg.batch * cfg.seq; // tokens
    let d = cfg.d_model;
    let f = cfg.d_ff;

    // Up-projection Y = X·W1 : (M×D)·(D×F). Column-shard W1 → no communication.
    let column = shard_matmul(m, f, d, tp, ShardingCase::WColwise);
    // Down-projection Z = Y·W2 : (M×F)·(F×D). Row-shard W2 along K=F → all-reduce.
    let row = shard_matmul(m, d, f, tp, ShardingCase::InnerReduction);

    TensorParallelCost { tp, column, row }
}

/// Row-major shapes for the two-matmul MLP `Z = (X·W1)·W2` that
/// [`sim_column_then_row`] simulates: `X` is `M×K`, `W1` is `K×H`, `W2` is
/// `H×N`. The hidden dim `H` is the axis tensor parallelism shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlpShapes {
    /// Rows of the input (tokens, `M`).
    pub m: usize,
    /// Input feature dim (`K`).
    pub k: usize,
    /// Hidden dim contracted by both matmuls (`H`) — the sharded axis.
    pub h: usize,
    /// Output feature dim (`N`).
    pub n: usize,
}

/// Run the Megatron **column-then-row** MLP `Z = (X·W1)·W2` across `tp` logical
/// devices and prove the reassembled result equals the unsharded computation.
///
/// Shapes come from `s` (see [`MlpShapes`]). The hidden dim `H` is the one both
/// matmuls contract against, so it is what gets sharded:
///
/// * **Column-parallel** `Y = X·W1`: device `i` owns columns
///   `[i·H/tp .. (i+1)·H/tp)` of `W1`, producing a column-slice of `Y` with
///   **no communication**.
/// * **Row-parallel** `Z = Y·W2`: device `i` owns the matching rows of `W2`, so
///   it multiplies its `Y`-slice by its `W2`-rows into a **partial** `M×N` sum.
///   An [`sim_all_reduce_sum`] over the `tp` partials yields the full `Z` — the
///   single collective the pattern is designed around.
///
/// Because the column split of `W1` lines up with the row split of `W2`, summing
/// the partial products reconstructs `(X·W1)·W2` exactly. Returns the flat
/// row-major `M×N` `Z`, which the caller compares against the unsharded result.
pub fn sim_column_then_row(x: &[f32], w1: &[f32], w2: &[f32], s: MlpShapes, tp: usize) -> Vec<f32> {
    let MlpShapes { m, k, h, n } = s;
    assert_eq!(x.len(), m * k, "x must be M×K");
    assert_eq!(w1.len(), k * h, "w1 must be K×H");
    assert_eq!(w2.len(), h * n, "w2 must be H×N");
    assert!(
        tp >= 1 && h % tp == 0,
        "hidden dim H ({h}) must divide evenly by tp ({tp})"
    );

    let hidden_per_shard = h / tp;

    // Each device produces a full-shape M×N partial sum; we all-reduce them.
    let mut partials: Vec<Vec<f32>> = Vec::with_capacity(tp);

    for shard in 0..tp {
        let hid_lo = shard * hidden_per_shard;
        let mut partial = vec![0.0f32; m * n];

        // ── Step 1: column-parallel Y-slice = X · W1[:, hid_lo..hid_hi] ──────
        // ── Step 2: row-parallel — multiply that slice by W2[hid_lo.., :] ───
        // Fused: for each hidden index this device owns, scatter its rank-1
        // contribution X[:,·]·W1[·,hid] ⊗ W2[hid,:] into the partial M×N sum.
        for row in 0..m {
            for local_hid in 0..hidden_per_shard {
                let hid = hid_lo + local_hid;
                // y = (X·W1)[row, hid] for this device's hidden column.
                let mut y = 0.0f32;
                for kk in 0..k {
                    y += x[row * k + kk] * w1[kk * h + hid];
                }
                // Accumulate y · W2[hid, :] across the output row.
                for col in 0..n {
                    partial[row * n + col] += y * w2[hid * n + col];
                }
            }
        }

        partials.push(partial);
    }

    // ── Step 3: all-reduce the per-device partial sums into the full Z ───────
    // (Every device would end up with this; we return the single copy.)
    sim_all_reduce_sum(&partials).into_iter().next().unwrap()
}
