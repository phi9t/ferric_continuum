# Lesson 0002: Context-parallel Qwen3

**Time:** 12 minutes
**Tangible win:** Design a context-parallel Qwen3 block and state exactly what must communicate, what remains local, and why decode needs a different plan.

[Previous lesson](0001-three-layers-of-tnsr.md) · [Reference sheet](../reference/0002-context-parallel-qwen3.md)

## Start with the axis

Qwen3 receives hidden states shaped:

```text
X: [B, T, D]
```

Tensor parallelism shards a feature or head dimension. Context parallelism shards `T`, the token dimension:

```text
rank r owns X_r: [B, T/P, D]
```

where `P` is the context-parallel degree.

This is stronger than the narrower use of “sequence parallelism” in some tensor-parallel systems. Context parallelism keeps **all activations** sequence-sharded throughout the block. NVIDIA's [Megatron Core context-parallel guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) uses this definition.

## Predict the exception

Which Qwen3 sublayer cannot operate independently on each token shard?

<details>
<summary>Reveal the answer</summary>

Attention. Norms, projections, RoPE, residuals, and the MLP are token-local. A local query still needs keys and values belonging to other ranks.

</details>

## Qwen3 shapes under context parallelism

For the repository's Qwen3-8B configuration:

```text
D   = 4096
Hq  = 32 query heads
Hkv = 8 key/value heads
Dh  = 128 dimensions per head
```

On each rank:

```text
X_r: [B, T/P, 4096]
Q_r: [B, T/P, 32, 128]
K_r: [B, T/P,  8, 128]
V_r: [B, T/P,  8, 128]
```

The weights are replicated across the context-parallel group unless context parallelism is composed with tensor parallelism.

## The correctness-first design

Start with KV all-gather, not ring attention. It is inefficient enough to expose the costs and simple enough to prove correct.

For each Qwen3 block and rank:

1. Run RMSNorm and the Q/K/V projections on local `X_r`.
2. Apply Q/K RMSNorm locally.
3. Apply RoPE using **global token positions**.
4. All-gather `K_r` and `V_r` into global `K` and `V`.
5. Compute attention only for local queries `Q_r` against global `K,V`.
6. Apply the causal mask using global query and key positions.
7. Keep output `O_r: [B,T/P,Hq,Dh]` sequence-sharded.
8. Run the output projection, residual, RMSNorm, and SwiGLU MLP locally.

The block therefore begins and ends with `[B,T/P,D]`. Only attention needs context-group communication.

### Causal masking example

Let `T=16` and `P=4`. Rank 2 owns query positions `8..11`.

- Query 8 may see keys `0..8`.
- Query 11 may see keys `0..11`.
- Keys `12..15`, even though gathered, must remain masked.

The current `gqa_attention` compares local indices with `si <= ti`. That becomes wrong after sharding. It needs explicit global query and key positions.

### RoPE example

`rope.rs` already provides `RopeConfig.start_pos`, but `Qwen3Attention` fixes it to zero. For contiguous sharding, rank `r` needs:

```text
start_pos = r × (T/P)
```

Otherwise every rank rotates its first local token as position zero. A later load-balanced, non-contiguous partition needs explicit position IDs rather than one `start_pos`.

## Why the current GQA operation must change

`ops/gqa.rs` currently assumes:

```text
Q: [B,T,Hq,Dh]
K: [B,T,Hkv,Dh]
V: [B,T,Hkv,Dh]
```

and materializes:

```text
P_attention: [B,Hq,T,T]
```

Context parallelism requires different query and KV lengths:

```text
Q_local: [B,T/P,Hq,Dh]
K_seen:  [B,T,Hkv,Dh]
V_seen:  [B,T,Hkv,Dh]
```

So this is not merely a wrapper around `gqa_attention`. The operation contract, causal indexing, saved backward state, and attention implementation all need redesign.

## Communication for Qwen3-8B

One rank's local K+V block contains:

```text
2 × B × (T/P) × Hkv × Dh × element_bytes
```

For `B=1`, `T=32768`, `P=4`, `Hkv=8`, `Dh=128`, and the current f32 storage:

```text
local K+V block                  = 64 MiB
three remote blocks per rank    = 192 MiB per layer forward
36 Qwen3 layers                 = 6.75 GiB per full forward
```

This excludes backward, protocol overhead, weight-gradient reduction, and any tensor parallelism.

GQA matters: Qwen3 has 8 KV heads rather than 32. KV traffic is therefore 4× smaller than full multi-head attention with one KV head per query head.

## From all-gather to ring attention

All-gather proves the decomposition, but it temporarily gives every rank global K and V. A bounded-memory design circulates one KV block around a ring, as in [Ring Attention](https://arxiv.org/abs/2310.01889).

Each rank keeps its local Q block fixed. At each ring step it:

1. receives one K/V block;
2. computes one block of scores;
3. masks by global positions;
4. merges that block into an online-softmax state;
5. sends the K/V block onward.

The running state per query row is:

```text
m: maximum score seen so far
l: sum of exp(score - m)
o: weighted value numerator
```

When the next score block arrives, rescale the old state to the new maximum before adding the new block. This is the numerical step that makes blockwise attention exactly equal to unsharded softmax without materializing `[T/P,T]` probabilities.

## Training and inference diverge

| Phase | Context-parallel behavior |
|---|---|
| Training forward | Local queries consume exchanged K/V blocks |
| Training backward | `dQ` stays local; partial `dK,dV` contributions return to their owning ranks, commonly by reduce-scatter/ring exchange |
| Prefill | Same structure as training forward, without backward state |
| Decode | There is only one new query token; shard the existing KV cache and merge partial online-softmax states across ranks |

The current `qwen3_infer` has no KV cache. It reruns the complete prefix for every generated token. Therefore the proper inference order is:

1. add a per-layer GQA KV cache;
2. separate prefill from single-token decode;
3. make prefill context-parallel;
4. shard the decode KV cache by past-token range;
5. reduce/merge the per-rank attention statistics and weighted outputs each layer.

Simply context-sharding the current generation loop would teach the forward decomposition, but it would not resemble an efficient serving design.

## The `tnsr` research ladder

Implement in this order:

1. **Cost model:** Add Qwen3-aware CP memory and communication formulas. Use `Hkv × Dh`, not full `D`, for KV bytes.
2. **All-gather simulation:** Split Q/K/V vectors by context in one process, gather K/V, compute local causal outputs, concatenate, and compare with unsharded GQA.
3. **Ring simulation:** Replace global K/V with rotating blocks and online-softmax merging; prove agreement with the same reference.
4. **Model integration:** Pass global positions through RoPE and GQA while keeping every non-attention operation sequence-local.
5. **Inference split:** Add KV-cache-backed prefill and decode paths.
6. **Real runtime:** Only then add persistent device tensors, a rank/process abstraction, NCCL communication, CUDA attention kernels, and stream overlap.

The current host-buffer CUDA bridge is not a viable final substrate for step 6: every operation returns to host memory and there is no device or communicator ownership model.

## Retrieval practice

Answer before revealing.

<details>
<summary>1. Why do Qwen3's MLP and norms need no context-group collective?</summary>

They operate independently on each token. Sharding the token axis changes the number of rows, not the mathematical dependency of those operations.

</details>

<details>
<summary>2. Why is `start_pos = 0` incorrect on rank 2?</summary>

RoPE encodes absolute token position. Rank 2's local token zero represents a later global position, so reusing zero changes every attention dot product involving that shard.

</details>

<details>
<summary>3. Why is context-parallel decode not “split the one new token four ways”?</summary>

The new query has no useful sequence dimension to split. The long object is the past KV cache, so ranks own different cache ranges and combine their partial attention results.

</details>

## Primary source

Read NVIDIA's [Context Parallel Package](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html). While reading, map its “all modules except attention work as usual” statement onto `Qwen3Block::forward` line by line.

## Before the next lesson

From memory, describe the local shapes of Q, K, and V for `P=4`, then explain the two correctness bugs caused by keeping `gqa_attention` and RoPE unchanged.

Ask follow-up questions whenever a derivation or design choice is unclear.
