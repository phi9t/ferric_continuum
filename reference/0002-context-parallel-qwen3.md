# Qwen3 context-parallel reference

[Lesson 0002](../lessons/0002-context-parallel-qwen3.md) ·
[Typed transformer math](../lessons/0003-typed-transformer-math.md) ·
[Repository map](0001-tnsr-repository-map.md)

## Implementation status

The correctness-first CPU implementation exposes these final interfaces:

```rust
pub struct ContextParallelGqaShape {
    pub b: usize,
    pub cp: usize,
    pub local_t: usize,
    pub global_t: usize,
    pub hq: usize,
    pub hk: usize,
    pub dh: usize,
}

pub struct ContextParallelGqaSaved {
    pub shape: ContextParallelGqaShape,
    // Saved probability blocks are private implementation state.
}

pub struct ContextParallelGqaGrads {
    pub dq: Vec<TensorValue>,
    pub dk: Vec<TensorValue>,
    pub dv: Vec<TensorValue>,
}

pub fn raw_context_parallel_gqa_forward(
    q_shards: &[TensorValue],
    k_shards: &[TensorValue],
    v_shards: &[TensorValue],
) -> (Vec<TensorValue>, ContextParallelGqaSaved);

pub fn raw_context_parallel_gqa_backward(
    dout_shards: &[TensorValue],
    q_shards: &[TensorValue],
    k_shards: &[TensorValue],
    v_shards: &[TensorValue],
    saved: &ContextParallelGqaSaved,
) -> ContextParallelGqaGrads;

pub fn context_parallel_gqa_attention(
    q_shards: &[Tensor],
    k_shards: &[Tensor],
    v_shards: &[Tensor],
    name: &str,
) -> Vec<Tensor>;

pub fn context_parallel_gqa_attention_typed(
    q_shards: &[ShardQueryHeads],
    k_shards: &[ShardKvHeads],
    v_shards: &[ShardKvHeads],
    name: &str,
) -> Vec<ShardQueryHeads>;

impl Qwen3Attention {
    pub fn forward_context_parallel(&self, x_shards: &[Tensor]) -> Vec<Tensor>;

    pub fn forward_typed(&self, x: &FullHiddenStates) -> FullHiddenStates;

    pub fn forward_context_parallel_typed(
        &self,
        x_shards: &[ShardHiddenStates],
    ) -> Vec<ShardHiddenStates>;
}
```

It accepts equal, contiguous, nonempty logical-rank shards; materializes global
batch-major K/V; saves one `[B,Hq,S,T]` probability block per rank; accumulates
global `dK/dV`; and scatters those gradients back to their sequence owners. It
is a single-process f32 simulation, not a CUDA/NCCL runtime.

The typed entry points are also implemented:

```rust
Qwen3Attention::forward_typed(
    &FullHiddenStates,
) -> FullHiddenStates

Qwen3Attention::forward_context_parallel_typed(
    &[ShardHiddenStates],
) -> Vec<ShardHiddenStates>
```

Inputs and gradient edges use the fixed order `Q[0..P], K[0..P], V[0..P]`.
Every shard is nonempty and shaped `[B,S,H,Dh]`; shard count and local extent
must agree, and `Hq` must be divisible by `Hkv`.

The typed methods make full-sequence versus shard ownership, query heads versus
KV heads, head split/merge transformations, and dynamic extent meaning visible
in Rust types. Axis attachment and erasure add no additional tensor copy or
autograd node; the mathematical operations still create their ordinary graph
nodes. Independently executed legacy and typed derivations have distinct raw
tensors, while their canonical graph observations agree.

Each typed Qwen3 call creates one ephemeral validated state from the current
public scalars and parameter handles. It is not cached. Cloning handles rather
than values is coherent under the current single-threaded, callback-free
execution model. Equal-contiguous sequence ownership is concentrated in a
crate-internal layout module, while ordinary and context-parallel attention
retain separate explicit equations.

## Definitions

| Symbol | Meaning | Qwen3-8B value |
|---|---|---:|
| `B` | batch size | workload-dependent |
| `T` | global sequence length | workload-dependent |
| `P` | context-parallel degree | design choice |
| `D` | model width | 4096 |
| `Hq` | query heads | 32 |
| `Hkv` | key/value heads | 8 |
| `Dh` | head dimension | 128 |
| `S` | local sequence length, `T/P` | workload-dependent |

## Target full-block contract

Only `Qwen3Attention` has a context-parallel executable path in the current
milestone. The following is the target contract for later full-model sharding:

```text
input X_r  [B,S,D]
  ├─ token-local RMSNorm
  ├─ token-local Q/K/V projections
  │    Q_r [B,S,Hq,Dh]
  │    K_r [B,S,Hkv,Dh]
  │    V_r [B,S,Hkv,Dh]
  ├─ token-local Q/K RMSNorm + global-position RoPE
  ├─ context-parallel GQA(Q_r, exchanged K/V)
  │    O_r [B,S,Hq,Dh]
  ├─ token-local output projection + residual
  └─ token-local RMSNorm + SwiGLU + residual
output X'_r [B,S,D]
```

Only GQA communicates across the CP group during forward.

## Correctness-first KV all-gather

```text
for rank r:
    positions_r = global positions owned by r
    Q_r = project_q(X_r)
    K_r = project_k(X_r)
    V_r = project_v(X_r)
    Q_r = rope(normalize_q(Q_r), positions_r)
    K_r = rope(normalize_k(K_r), positions_r)
    K = all_gather(K_r, cp_group)
    V = all_gather(V_r, cp_group)
    O_r = causal_gqa(Q_r, K, V,
                     query_positions=positions_r,
                     key_positions=all_positions)
```

Forward KV traffic per device under a ring all-gather:

```text
global_kv_bytes = 2 × B × T × Hkv × Dh × element_bytes
comm/device      = (P - 1) / P × global_kv_bytes
```

Local non-attention activation memory ideally falls by approximately `P`; naive all-gather temporarily restores global K/V memory during attention.

## Ring/online-softmax merge

For one query row, maintain running maximum `m`, denominator `l`, and unnormalized output `o`.

Given score block `S_j` and value block `V_j`:

```text
m_j   = rowmax(S_j)
m_new = max(m, m_j)
alpha = exp(m - m_new)
P_j   = exp(S_j - m_new)       # masked entries contribute zero
l_new = alpha × l + rowsum(P_j)
o_new = alpha × o + P_j @ V_j
```

After all KV blocks:

```text
O = o / l
```

Initialize `m = -∞`, `l = 0`, and `o = 0`. Apply GQA's mapping `kv_head = query_head / (Hq/Hkv)` within each block.

## Causal block rules

For a query block with global interval `[q0,q1)` and KV block `[k0,k1)`:

| Relation | Action |
|---|---|
| `k0 >= q1` | Skip: every key is in the future |
| `k1 <= q0` | Compute without elementwise causal masking |
| Intervals overlap | Compute with `key_position <= query_position` mask |

Contiguous shards are correct but imbalanced: later query shards attend to more keys. A production design should add a load-balanced token assignment, while retaining explicit global positions.

## Backward ownership

| Gradient | Current single-process implementation | Future distributed execution |
|---|---|---|
| `dQ_r` | Written directly to query shard `r` | Remains with query owner `r` |
| `dK_j`, `dV_j` | All query contributions accumulate in global buffers, then scatter to shard `j` | Reduce-scatter or ring-route contributions back to owner `j` |
| Weight gradients | Every local branch references the same Tensor handle, so `Engine` accumulation sums them | Reduce across every group that physically replicates the weight |

## Prefill versus decode

```text
PREFILL
Q_r has S queries
KV blocks circulate or gather
O_r remains context-sharded

DECODE
Q has one new query token
rank r owns a shard of historical KV cache
rank r computes local (m_r, l_r, o_r)
all ranks merge states into the mathematically equivalent global softmax output
```

Decode requires a real per-layer GQA KV cache. The generic inference module has
a small `KvCache`, but `qwen3_infer` does not yet integrate one and recomputes
the complete prefix.

## Qwen3-specific byte formulas

Use `kv_width = Hkv × Dh`, not `D`:

```text
kv_cache_bytes = 2 × layers × B × T × Hkv × Dh × element_bytes
```

For Qwen3-8B, `Hkv × Dh = 8 × 128 = 1024`, while `D = 4096`. The current generic `scaling::inference::kv_cache_bytes(..., d, ...)` formula therefore overestimates this GQA cache by 4× when passed `D`.

Example, f32, `B=1`, `T=32768`, 36 layers:

```text
Qwen3-aware KV cache = 2 × 36 × 32768 × 1024 × 4
                     = 9 GiB
```

## Implementation roadmap

| Stage | Status | Location / change |
|---|---|---|
| Checked Shape arithmetic | Implemented | Checked element/byte products with deterministic overflow failures |
| Explicit all-gather forward | Implemented | `ops/context_parallel_gqa.rs`: local queries against materialized global K/V |
| Explicit backward | Implemented | Local `dQ`, global accumulation and owner-scatter for `dK/dV` |
| Multi-output autograd | Implemented | One operation node for `Q[0..P], K[0..P], V[0..P] → O[0..P]` |
| Qwen3 attention integration | Implemented | `Qwen3Attention::forward_context_parallel`; rank-offset RoPE and local projections |
| Typed tensor interface | Implemented | Zero-copy `TypedTensor<A>` plus distinct typed ordinary/CP Qwen3 methods |
| Named extent evidence | Implemented | `AxisExtent<A>`, checked merged widths, and canonical layout accessors |
| Equal-contiguous layout | Implemented | Checked Query blocks plus batch-major `TensorValue` gather/scatter |
| Scoped recording | Implemented | Explicit `Engine::with_recording`; backward owns unpack/recompute events |
| Checkpoint unwind safety | Implemented | Exact frame/registry cleanup and clean retry after recomputation panic |
| Graph observation | Implemented | Canonical ID-independent operations, roots, shapes, and producer lookup |
| Independent verifiers | Implemented | Scalar ordinary-GQA oracle, causal/batch/graph checks, compile-fail contracts, finite differences, and Qwen3 gradient parity |
| Coherent Qwen call state | Implemented | Fresh validated scalar/handle snapshot for every typed call; no cache |
| Optional static extents | Specified, not implemented | Wave B refinement over semantic axes; prototype required first |
| Runtime named-axis seam | Specified, not implemented | Wave C out-of-band, versioned descriptor; registry prototype required first |
| Long-offset RoPE precision | Future | Preserve current f32 numerics here; any higher-precision change needs its own compatibility decision |
| GQA-aware cost model | Future | Add CP traffic/activation estimates and correct GQA KV-cache width |
| Ring simulator | Future | Block rotation and online-softmax state merge |
| Load-balanced positions | Future | Ragged/non-contiguous partitions with explicit position IDs |
| Full-model training | Future | Sequence-sharded blocks, logits/loss, and gradient-group semantics |
| Inference | Future | Per-layer KV cache and separate prefill/decode APIs |
| Real execution | Future | Persistent device memory, rank ownership, NCCL, CUDA kernels, and streams |

## Minimum invariant tests

1. `P=1` equals existing unsharded GQA.
2. Concatenated `P=2` and `P=4` outputs match unsharded causal GQA.
3. A query just after a shard transition attends to earlier-rank keys.
4. No query attends to a future-rank key.
5. Global-position RoPE matches the unsharded result.
6. GQA mapping remains correct for `Hq=4,Hkv=2` and `Hq=32,Hkv=8`.
7. Empty, unequal, or otherwise ragged shard sets are rejected explicitly.
8. Backward gradients match the unsharded reference for Q, K, V, and weights.
9. Typed and legacy full/CP derivations agree in values, gradients, and
   canonical graph topology.

Ring online-softmax agreement becomes an additional invariant when the future
ring simulator is implemented.

## Primary sources

- [Megatron Core Context Parallel Package](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
