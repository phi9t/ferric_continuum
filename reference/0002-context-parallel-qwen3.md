# Qwen3 context-parallel reference

[Lesson 0002](../lessons/0002-context-parallel-qwen3.md) · [Repository map](0001-tnsr-repository-map.md)

## Implementation status

The correctness-first CPU implementation is now available:

```rust
use tnsr::ops::context_parallel_gqa::{
    context_parallel_gqa_attention,
    raw_context_parallel_gqa_backward,
    raw_context_parallel_gqa_forward,
    ContextParallelGqaGrads,
    ContextParallelGqaSaved,
    ContextParallelGqaShape,
};

let output_shards = attention.forward_context_parallel(&input_shards);
```

It accepts equal, contiguous, nonempty logical-rank shards; materializes global
batch-major K/V; saves one `[B,Hq,S,T]` probability block per rank; accumulates
global `dK/dV`; and scatters those gradients back to their sequence owners. It
is a single-process f32 simulation, not a CUDA/NCCL runtime.

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

## Sharded block contract

```text
input X_r  [B,S,D]
  ├─ token-local RMSNorm
  ├─ token-local Q/K/V projections
  │    Q_r [B,S,Hq,Dh]
  │    K_r [B,S,Hkv,Dh]
  │    V_r [B,S,Hkv,Dh]
  ├─ global-position Q/K norm + RoPE
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
    Q_r, K_r, V_r = project_and_rope(X_r, positions_r)
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

| Gradient | Ownership |
|---|---|
| `dQ_r` | Remains with query owner `r` |
| `dK_j`, `dV_j` | Receive contributions from every query block that attended to KV block `j`; reduce-scatter/ring-route them back to owner `j` |
| Weight gradients | Weights are CP-replicated, so aggregate across CP ranks, often through the data-parallel gradient group |

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
all ranks merge states into the exact global softmax output
```

Decode requires a real KV cache. The repository currently only estimates cache size and recomputes the entire prefix.

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

## Proposed `tnsr` change map

| Stage | Status | Location / change |
|---|---|---|
| Explicit all-gather forward | Implemented | `ops/context_parallel_gqa.rs`: local queries against materialized global K/V |
| Explicit backward | Implemented | Local `dQ`, global accumulation and owner-scatter for `dK/dV` |
| Multi-output autograd | Implemented | One operation node for `Q[0..P], K[0..P], V[0..P] → O[0..P]` |
| Qwen3 attention integration | Implemented | `Qwen3Attention::forward_context_parallel`; rank-offset RoPE and local projections |
| GQA-aware cost model | Future | Add CP traffic/activation estimates and correct GQA KV-cache width |
| Ring simulator | Future | Block rotation and online-softmax state merge |
| Load-balanced positions | Future | Ragged/non-contiguous partitions with explicit position IDs |
| Full-model training | Future | Sequence-sharded blocks, logits/loss, and gradient-group semantics |
| Inference | Future | Per-layer KV cache and separate prefill/decode APIs |
| Real execution | Future | Persistent device memory, rank ownership, NCCL, CUDA kernels, and streams |

## Minimum invariant tests

1. `P=1` equals existing unsharded GQA.
2. Concatenated `P=2` and `P=4` outputs match unsharded causal GQA.
3. A query just after a shard boundary attends to earlier-rank keys.
4. No query attends to a future-rank key.
5. Global-position RoPE matches the unsharded result.
6. GQA mapping remains correct for `Hq=4,Hkv=2` and `Hq=32,Hkv=8`.
7. Ragged `T % P != 0` partitions are either supported or rejected explicitly.
8. Ring online softmax agrees with all-gather attention within a documented tolerance.
9. Backward gradients match the unsharded reference for Q, K, V, and weights.

## Primary sources

- [Megatron Core Context Parallel Package](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
