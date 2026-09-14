# Lesson 0003: Make transformer mathematics visible in Rust types

**Time:** 30 minutes

**Tangible win:** Read ordinary and context-parallel Qwen3 attention from top
to bottom while the compiler prevents several axis-category mistakes.

[Previous lesson](0002-context-parallel-qwen3.md) ·
[Context-parallel reference](../reference/0002-context-parallel-qwen3.md)

## The problem is meaning, not storage

The original `Tensor` abstraction is useful. It owns a runtime shape, f32
values, autograd metadata, an optional producer, and distributed layout
metadata. None of that should be replaced.

Its shape alone cannot distinguish these two values:

```text
queries: [B, T, Hq,  Dh]
keys:    [B, T, Hkv, Dh]
```

Both are rank four. A reader has to remember what every index means, and a
function that accepts raw `Tensor` values can accidentally receive them in the
wrong semantic role.

Wave A therefore adds a zero-copy typed handle:

```rust
pub struct TypedTensor<A: Axes> {
    tensor: Tensor,
    // A exists only in the Rust type system.
}
```

`TypedTensor` owns the same cheap `Rc`-backed `Tensor` handle. Attaching axes,
cloning the typed handle, or erasing it back to `Tensor` does not copy data and
does not create an autograd operation.

## Read a type as a tensor equation

The canonical transformer axes are:

```text
Batch
Sequence<Full>       complete sequence
Sequence<Shard>      one contiguous context-parallel shard
Hidden
QueryHead
KvHead
HeadDim
```

`Merged<A, B>` says that two logical axes currently occupy one physical
dimension. The principal aliases are:

```rust
type HiddenStates<S> =
    TypedTensor<Axes3<Batch, Sequence<S>, Hidden>>;

type ProjectedQueries<S> =
    TypedTensor<Axes3<Batch, Sequence<S>, Merged<QueryHead, HeadDim>>>;

type ProjectedKv<S> =
    TypedTensor<Axes3<Batch, Sequence<S>, Merged<KvHead, HeadDim>>>;

type QueryHeads<S> =
    TypedTensor<Axes4<Batch, Sequence<S>, QueryHead, HeadDim>>;

type KvHeads<S> =
    TypedTensor<Axes4<Batch, Sequence<S>, KvHead, HeadDim>>;
```

For example, this Rust type:

```rust
FullQueryHeads
```

means this mathematical object:

```text
Q[B, T, Hq, Dh]
```

and this type:

```rust
ShardKvHeads
```

means:

```text
K_r or V_r [B, S, Hkv, Dh]
```

where `S` is the local contiguous sequence extent on logical rank `r`.

## What the compiler proves—and what it does not

Wave A proves semantic axis order, operation categories, and the meaning of
extents passed across typed operation seams. These mistakes do not compile:

```text
- pass HiddenStates to RoPE
- pass QueryHeads as GQA keys or values
- pass full-sequence heads to context-parallel GQA
- pass a KV projection weight to query projection
- use an `AxisExtent<KvHead>` to split projected queries
- assign [Batch,Hidden,Sequence] to [Batch,Sequence,Hidden]
```

Runtime extents remain dynamic. Code must still check facts such as:

```text
Hq % Hkv == 0
last(Q) == last(K) == last(V) == Dh
K.shape == V.shape
wq.shape == [D, Hq*Dh]
all context shards have the same [B,S,D]
```

This division is deliberate: checkpoint-loaded and variable-length models need
dynamic extents, while Rust types can still preserve what each extent means.

One subtle attachment rule matters. `TypedTensor::from_tensor` checks only rank.
The caller supplies the semantic assertion through the chosen Rust type:

```rust
let x = FullHiddenStates::from_tensor(raw_x)?;
```

This verifies rank three; it cannot inspect raw bytes and infer that dimension
two is truly a model-hidden axis.

The legacy `Tensor` handle still exposes interior mutation for compatibility.
A typed handle therefore revalidates its rank whenever an operation borrows it
or reads named extents. This catches rank-changing mutation through an untyped
alias before stale axis evidence reaches transformer math. A same-rank change
cannot reveal semantic meaning from storage; the original caller assertion
continues to define that meaning.

## Extents are computational evidence

Axis names are useful only if implementations keep using them. `AxisExtent<A>`
therefore carries a dynamic `usize` together with its semantic axis:

```rust
let query_heads = AxisExtent::<QueryHead>::new(32);
let kv_heads = AxisExtent::<KvHead>::new(8);
let head_dim = AxisExtent::<HeadDim>::new(128);

let query_width = query_heads.checked_merge(head_dim).unwrap();
let kv_width = kv_heads.checked_merge(head_dim).unwrap();
```

Both widths contain `usize` values, but their Rust types differ:

```text
query_width: AxisExtent<Merged<QueryHead, HeadDim>>
kv_width:    AxisExtent<Merged<KvHead, HeadDim>>
```

Canonical typed tensors expose matching named accessors:

```rust
let batch = hidden_states.batch_extent();
let sequence = hidden_states.sequence_extent();
let hidden = hidden_states.hidden_extent();

let queries = split_query_heads(
    &projected_queries,
    query_heads,
    head_dim,
    "q_reshape",
);
```

This keeps the computation readable after the public function signature. The
compiler rejects `kv_heads` in that call because query-head and KV-head extents
are different evidence. Existing callers may still pass `usize` values to the
typed split functions for compatibility; the Qwen3 typed derivations use the
evidence-bearing form.

All extent products used by `Shape` are checked. `checked_numel` and
`checked_bytes_f32` return `None` on overflow, while the existing `numel` and
`bytes_f32` conveniences panic with deterministic messages. A zero extent
short-circuits the product to zero, independent of factor order.

## Operations state their axis transformations

The typed functions live beside the original mathematical operations. Their
signatures are a compact shape calculus:

```rust
project_queries:
    HiddenStates<S> × QueryProjectionWeight -> ProjectedQueries<S>

split_query_heads:
    ProjectedQueries<S> -> QueryHeads<S>

normalize_queries:
    QueryHeads<S> × HeadScale -> QueryHeads<S>

rotate_queries:
    QueryHeads<S> -> QueryHeads<S>

gqa_attention_typed:
    FullQueryHeads × FullKvHeads × FullKvHeads -> FullQueryHeads

context_parallel_gqa_attention_typed:
    [ShardQueryHeads] × [ShardKvHeads] × [ShardKvHeads]
    -> [ShardQueryHeads]

merge_query_heads:
    QueryHeads<S> -> ProjectedQueries<S>

project_attention_output:
    ProjectedQueries<S> × OutputProjectionWeight -> HiddenStates<S>
```

These wrappers invoke the existing explicit f32 forward/backward operation.
They do not introduce an attention strategy enum, a generic operation-mode
dispatcher, extra graph nodes, or another tensor storage abstraction.

## Ordinary Qwen3 attention reads like the derivation

`Qwen3Attention::forward_typed` is intentionally a distinct, linear program:

```rust
pub fn forward_typed(&self, x: &FullHiddenStates) -> FullHiddenStates {
    let state = self.validated_attention_state();

    let projected_queries = project_queries(x, &state.wq, "q_proj");
    let projected_keys    = project_keys(x, &state.wk, "k_proj");
    let projected_values  = project_values(x, &state.wv, "v_proj");

    let queries = split_query_heads(
        &projected_queries,
        state.geometry.query_heads,
        state.geometry.head_dim,
        "q_reshape",
    );
    let keys = split_kv_heads(
        &projected_keys,
        state.geometry.kv_heads,
        state.geometry.head_dim,
        "k_reshape",
    );
    let values = split_kv_heads(
        &projected_values,
        state.geometry.kv_heads,
        state.geometry.head_dim,
        "v_reshape",
    );

    let queries = normalize_queries(&queries, &state.q_norm, "q_norm");
    let keys    = normalize_keys(&keys, &state.k_norm, "k_norm");

    let queries = rotate_queries(&queries, state.rope_cfg, "q_rope");
    let keys    = rotate_keys(&keys, state.rope_cfg, "k_rope");

    let attended = gqa_attention_typed(&queries, &keys, &values, "gqa");
    let attended = merge_query_heads(&attended, "attn_reshape");
    project_attention_output(&attended, &state.wo, "o_proj")
}
```

The implementation contains comments, but it does not need a runtime flag to
explain which attention algorithm is executing. The sequence of typed values
is the explanation.

The legacy `forward(&Tensor)` stays independent. Some duplication is valuable
here: it gives us two readable derivations and lets compatibility tests compare
their values, gradients, and graph observations without one delegating to the
other.

## Context-parallel Qwen3 is a second explicit program

Context-parallel attention is not a Boolean branch inside ordinary attention.
Its ownership and communication are different, so it receives a separate
method:

```rust
pub fn forward_context_parallel_typed(
    &self,
    x_shards: &[ShardHiddenStates],
) -> Vec<ShardHiddenStates>
```

The dataflow is:

```text
for each logical rank r:
    X_r[B,S,D]
      -> local Q/K/V projections
      -> explicit Q-head and KV-head axes
      -> local Q/K RMSNorm
      -> Q/K RoPE at start_pos + r*S

one cross-rank mathematical operation:
    CP-GQA({Q_r}, {K_r}, {V_r}) -> {O_r}

for each logical rank r:
    O_r[B,S,Hq,Dh]
      -> merge Hq*Dh
      -> local output projection
      -> Y_r[B,S,D]
```

Slice position defines logical rank. The current milestone supports equal,
contiguous, nonempty shards only. A crate-internal
`EqualContiguousAttentionLayout` centralizes the checked `P*S` rule and returns
a `QueryBlock` for each rank. Qwen3 obtains the RoPE offset from that block
rather than repeating rank arithmetic:

```rust
let query_block = layout
    .query_block(rank)
    .expect("validated context-parallel rank");
let start_pos = state
    .rope_cfg
    .start_pos
    .checked_add(query_block.start())
    .expect("RoPE start position overflow");
```

The same Query block maps local query positions to global query positions in
context-parallel GQA. This deep module owns only equal-contiguous sequence
ownership. It does not own attention equations.

That split balances depth and locality. The layout module provides leverage by
hiding repeated batch-major ownership indexing, while each attention
implementation keeps its equations together. A shallow strategy flag shared
by ordinary and context-parallel attention would save lines but scatter the
mathematical explanation across runtime branches.

RoPE position-span and table-size arithmetic are checked before allocation.
Its established `f32` angle equation remains unchanged so the additive typed
path preserves the legacy operation's valid-input numerics.

### What crosses shard ownership

The implementation in `ops/context_parallel_gqa.rs` is a correctness-first,
single-process model. Its gather/scatter interface accepts whole
`TensorValue`s, infers `B`, `H`, and `Dh`, and preserves batch-major layout:

```text
Q_r stays local.
K_r and V_r are gathered into batch-major global K and V buffers.
Each rank computes only its local query rows against global causal K/V.
```

For local query `i` on rank `r`:

```text
global_query = r*S + i
visible keys = 0 ..= global_query
```

Let `groups = Hq/Hkv` and `kv_head(h) = h/groups`. For each batch row,
query head, and local query position, forward computes:

```text
score[j] = dot(Q_r[b,i,h,:], K[b,j,kv_head(h),:]) / sqrt(Dh)
           for 0 <= j <= global_query

row_max  = max_j score[j]
weight[j] = exp(score[j] - row_max)
prob[j]   = weight[j] / sum_k weight[k]

O_r[b,i,h,:] = sum_j prob[j] * V[b,j,kv_head(h),:]
```

Future positions are absent from the softmax row rather than normalized and
then zeroed. Subtracting `row_max` is the stable-softmax step.

It saves one probability block per rank:

```text
Prob_r[B, Hq, S, T]
```

Backward is equally explicit:

```text
dP[s] = dot(dO, V[s])
dV[s] += prob[s] * dO

dS[s] = prob[s] * (dP[s] - sum_j prob[j]*dP[j])

dQ += dS[s] * K[s] / sqrt(Dh)
dK[s] += dS[s] * Q / sqrt(Dh)
```

`dQ_r` is written directly to its query-owner shard. Cross-rank `dK` and `dV`
contributions first accumulate into batch-major global buffers and are then
scattered to the owning equal sequence shards.

The tensor adapter creates one `finish_op_multi` node. Its backward recipe
snapshots forward-time Q/K/V `TensorValue`s and the saved probability blocks;
its gradient targets still point to the original input handles. Inputs and
gradient edges are ordered as every Q shard, then every K shard, then every V
shard. If an output shard is unused downstream, the engine supplies a zero
incoming gradient for that output while the shared recipe runs once.

This is deliberately not yet ring attention, NCCL, or CUDA execution.

## Public mutable parameters need a validation seam

`Qwen3Attention` exposes its parameter tensors publicly. A caller can replace
`wk` after construction, so the typed method cannot merely label it
`KvProjectionWeight` and hope its shape is still correct.

Before returning any privately tagged handle, the validator establishes:

```text
Hq > 0
Hkv > 0
Hq % Hkv == 0
Dh > 0 and Dh is even
Hq*Dh and Hkv*Dh do not overflow
wq == [D, Hq*Dh]
wk == [D, Hkv*Dh]
wv == [D, Hkv*Dh]
wo == [Hq*Dh, D]
q_norm == [Dh]
k_norm == [Dh]
```

Validation is two-stage. Raw rank checks establish that each handle can be
given its private axis type; named accessors on those local typed views then
check every semantic extent above. The completed state is returned only after
both stages succeed.

The resulting `ValidatedAttentionState` is ephemeral: each typed call copies
the scalar geometry and RoPE configuration and clones the current cheap tensor
handles. It does not copy parameter values and is never cached in
`Qwen3Attention`. This handle snapshot is coherent under tnsr's present
single-threaded, callback-free execution model; concurrent mutation would need
a different ownership design.

After this state is constructed, both typed derivations read only from it. A
valid between-call test changes every head extent, the RoPE configuration, and
all attention parameter handles. It compares the typed result with an
independent legacy derivation over the replacement state, checks fresh Q/K
reshape extents, and proves gradients route only to replacement weights. This
is the structural reason a public mutation cannot leave a stale typed view
behind.

## Recording belongs to an execution

Autograd graph construction does not require debug recording. `Engine::new()`
is deliberately inert; a caller scopes forward observation explicitly:

```rust
let mut engine = Engine::new();
let loss = engine.with_recording(|| {
    let output = attention.forward_typed(&hidden);
    sum(output.as_tensor(), "loss")
});
engine.backward(&loss);
```

Scopes may nest, including across different engines. RAII guards restore the
outer recorder after normal return or panic. `backward` installs its engine's
recorder for the whole reverse pass, so saved-tensor unpack and checkpoint
recomputation events belong to the engine performing that backward call.

Checkpoint state is unwind-safe too. A failed original forward removes its
exact stack and registry entries. A failed recomputation clears its partial
cache, resets the save cursor and `is_recomputing`, and allows a retry to replay
the complete checkpoint body.

Code that needs only gradients may omit `with_recording`; the producer DAG and
backward computation remain valid. The scope controls observability, not math.

## Observe topology without reaching into tensors

`GraphObservation` is the verifier seam for differentiable topology:

```rust
use tracing::info;

let graph = GraphObservation::from_outputs(&[&loss]);
for operation in graph.operations() {
    info!(kind = ?operation.kind, name = %operation.name, "observed operation");
}
```

The observation contains operation kinds and names, ordered input/output
edges, shapes, and caller-selected roots. Each edge is one atomic `GraphTensor`
containing both its normalized identity and captured shape, so those facts
cannot drift into mismatched parallel arrays. `from_outputs` visits roots in
caller order and performs producer-before-consumer DFS in each operation's
declared input order. Raw process-global tensor and operation IDs are normalized
away, so separately executed typed and legacy derivations can be compared
directly. Multi-output checks can ask `producer_of` whether several live tensors
share the same observed operation, then resolve the returned index with
`operation`. `producer_of` returns `Ok(None)` for an observed leaf,
`Ok(Some(index))` for a produced tensor, and an error for a tensor outside the
observation.

Three smaller `Tensor` observations serve same-execution checks:

```text
value()                cheap immutable TensorValue snapshot
shares_storage_with() exact handle/storage identity
producer_id()          raw producer identity within one execution
```

Use `GraphObservation` instead when comparing separate executions, because raw
tensor and producer IDs intentionally differ.

`DebugRecorder::graph_observation()` uses the same record format for an
explicitly recorded prefix, including a prefix that ends in validation
failure. Its operation order is recorder insertion order and its roots are
terminal recorded outputs, so it is not necessarily equal to a reachable-DAG
observation whose independent siblings executed in another order. Trace JSON
version 1 remains unchanged.

## The verifier stack

No one comparison is enough. The current architecture proof uses several
independent layers:

| Risk | Verifier |
|---|---|
| Shape product wraparound | Zero-factor, ordinary-value, and deterministic overflow tests |
| Axis-category misuse | `rust_doc_test` legal/`compile_fail` pairs |
| Wrong rank declaration | `AxisError` runtime test |
| Accidental tensor copy or graph node | Public handle identity and `GraphObservation` tests |
| Wrong ordinary GQA equation | Independent scalar causal-GQA oracle |
| Causal leak | Separate future-K and future-V perturbations |
| Batch-major indexing error | Batch-isolation and `B=2` tests |
| Wrong backward | Finite differences for Q, K, and V |
| Wrong CP gather/scatter | `P=1,2,4`, reconstruction, and gradient parity |
| Broken multi-output autograd | Shared producer and unused-output tests |
| Wrong RoPE offset | Full-vs-sharded Qwen3 output parity |
| Missed shared-weight accumulation | Full-vs-sharded parameter-gradient parity |
| Typed derivation drift | Typed-vs-legacy values, gradients, and canonical graph parity |
| Recorder ownership leak | Nested, panic-restoration, and cross-engine ownership tests |
| Stale checkpoint replay state | Failed-forward release and panic-once/retry-success tests |
| Stale Qwen call state | Valid between-call scalar/handle replacement test |

The chain is intentionally independent. The scalar oracle calls neither
production GQA nor the ownership layout, so it checks ordinary forward math.
Test-local split/join loops—also independent of
`EqualContiguousAttentionLayout`—compare CP results with ordinary GQA. Finite
differences check the backward equations. Finally, typed/legacy value,
gradient, and graph parity check that the zero-copy adapters preserve those
derivations. Reusing production partition arithmetic in the verifier would let
the same indexing bug make both sides agree.

Run the architecture proof suite with:

```sh
bazel test --lockfile_mode=off \
  //ferric_continuum/tnsr:tnsr_tests \
  //ferric_continuum/tnsr:lib_tests \
  //ferric_continuum/tnsr:debug_trace_tests \
  //ferric_continuum/tnsr:typed_tensor_compile_tests \
  //ferric_continuum/tnsr:typed_tensor_tests \
  //ferric_continuum/tnsr:gqa_verifier_tests \
  //ferric_continuum/tnsr:qwen3_tests \
  //ferric_continuum/tnsr:context_parallel_gqa_tests
```

## Wave B specification: optional static extents

Wave B is specified but not implemented.

Its purpose is to refine selected fixed-shape teaching programs and kernel
experiments. It must not replace dynamic checkpoint loading or variable-length
inference.

The leading design is:

```rust
StaticTensor<Axes, Extents>
```

where promotion from `TypedTensor<Axes>` performs one complete runtime extent
check and then preserves the same `TensorId`, storage, and autograd graph.

Candidate extent leaves look like:

```rust
Dim<const N: usize>
```

Stable Rust cannot generally prove expressions such as `Hq * Dh` in arbitrary
generic const contexts. Wave B must therefore use one of these honest options:

1. supply flattened products as explicit constants and check them at promotion;
2. generate concrete aliases for one model configuration; or
3. keep products dynamic even when leaf extents are static.

Before implementation, prototype all three interface shapes:

```text
A. StaticTensor<Axes, Extents> layered over Wave A      (recommended)
B. generated model-specific aliases
C. a separate fixed-shape tensor family
```

Selection criteria are compiler diagnostics, signature readability, required
const expressions, compile time, and preservation of the original tensor
handle. Ragged shards and arbitrary dynamic batches remain outside Wave B.

## Wave C specification: named runtime interchange seams

Wave C is specified but not implemented.

Static Rust types disappear at checkpoint, FFI, serialization, Python, and
distributed-planning seams. Those places may use an out-of-band object:

```rust
struct NamedTensorDescriptor {
    schema_version: u32,
    axes: Vec<NamedAxis>,
}

struct NamedAxis {
    stable_id: String,
    extent: usize,
}
```

Rules:

```text
- stable_id, not display_name, is serialized
- IDs are ordered and versioned
- duplicate and unknown IDs are rejected
- descriptor extents must equal Tensor::shape
- if TensorLayoutMeta exists, extents must also agree with global_shape
- disagreement is an error; no metadata source silently repairs another
- promotion validates once and returns TypedTensor<A> over the same handle
- erasure derives names from A instead of storing duplicate mutable metadata
```

`TensorLayoutMeta` remains the authority for physical distribution. The Wave C
descriptor describes semantic axis order. These are complementary facts, not
two competing tensor implementations.

Before implementation, Wave C still needs registry/versioning rules, precise
error types, serialization examples, and a prototype showing promotion to both
Wave A and optional Wave B values.

## Reading path in the repository

Read these files in order:

1. `ferric_continuum/tnsr/src/tensor.rs` — checked Shape arithmetic and public
   value/identity observation.
2. `ferric_continuum/tnsr/src/typed.rs` — semantic axes, extents, and carrier.
3. `ferric_continuum/tnsr/src/ops/gqa.rs` — explicit ordinary GQA math.
4. `ferric_continuum/tnsr/src/attention_layout.rs` — equal-contiguous
   ownership, Query blocks, and batch-major gather/scatter.
5. `ferric_continuum/tnsr/src/ops/context_parallel_gqa.rs` — explicit CP
   gather, causal forward, backward accumulation, and scatter.
6. `ferric_continuum/tnsr/src/qwen3.rs` — independent legacy and typed Qwen3
   attention derivations.
7. `ferric_continuum/tnsr/src/autograd.rs` — Engine and canonical graph
   observation.
8. `ferric_continuum/tnsr/src/debug.rs` — scoped recorder stack and recorded
   graph projection.
9. `ferric_continuum/tnsr/src/checkpoint.rs` — unwind-safe checkpoint and
   recomputation state.
10. `ferric_continuum/tnsr/tests/gqa_verifier_test.rs` — independent oracle.
11. `ferric_continuum/tnsr/tests/typed_tensor_test.rs` — identity, integration,
   gradients, and graph proofs.

The design rule to carry forward is simple: one concrete mathematical
algorithm should look like one straightforward program. New execution
algorithms may duplicate a little setup, but should not hide their equations
behind a forest of runtime strategy branches.
