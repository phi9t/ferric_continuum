# tnsr × JAX Scaling Book — Chapter Map

`tnsr` is a readable, CPU-only, single-chip Rust implementation of the per-device
transformer math taught in **Chapter 4** of the JAX scaling book
("How To Scale Your Model", <https://jax-ml.github.io/scaling-book/>), plus the
gradient checkpointing / rematerialisation strategies from Chapter 5.

The book then takes that same per-device math and scales it across thousands of
accelerator chips — sharding, tensor/pipeline parallelism, hardware rooflines,
inference serving.  Those topics are mostly out of scope for a single-threaded
f32 library; the `src/scaling/` module provides *executable symbolic estimates*
(FLOPs, bytes, roofline, sharding cost) without requiring distributed hardware.

---

## Notation Bridge

The book uses multi-head GQA notation.  `tnsr` flattens everything into `d_model`
and uses `n_heads = 1` in the `tiny_4_7_29` reference config.

| Book symbol | Meaning | tnsr equivalent |
|-------------|---------|-----------------|
| N | number of Q heads | `cfg.n_heads` (= 1 in tiny) |
| K | number of KV heads (GQA) | same as N (no GQA) |
| H | head dimension | `d_model / n_heads` (= D when N = 1) |
| D | d_model = N × H | `cfg.d_model` |
| F | FFN hidden dimension | `cfg.d_ff` |
| B | batch size | `cfg.batch` |
| T | sequence length | `cfg.seq` |

In all tnsr formulas below, N = 1 and H = D, so "N·H" collapses to "D".

---

## Full Chapter Map

| Ch | Book section (slug) | tnsr module(s) | Status |
|----|---------------------|----------------|--------|
| 1  | Roofline Analysis (`roofline`) | `src/scaling/roofline.rs` — symbolic compute-vs-memory estimate | Executable-estimable |
| 2  | How to Think About TPUs (`tpus`) | — (CPU-only, single-threaded) | Absent |
| 3  | Sharded Matrices (`sharding`) | `src/scaling/sharding.rs` — 4 sharding cases as local algebra | Executable-estimable |
| 4  | All the Transformer Math (`transformers`) | `src/transformer.rs`, all of `src/ops/`, `src/autograd.rs`, `src/scaling/` | **Implemented** |
| 5  | Parallelize a Transformer for Training (`training`) | remat → `src/checkpoint.rs`; DP/FSDP/TP/PP → `src/scaling/distributed/` (executable estimates + single-process sims) | Executable-estimable |
| 6  | Training LLaMA 3 on TPUs (`applied-training`) | generic block only; no LLaMA-specific heads or TPU specifics | Absent |
| 7  | Transformer Inference (`inference`) | attention math is shared; no KV cache; `src/scaling/inference.rs` — KV bytes estimate | Conceptual |
| 8  | Serving LLaMA 3 (`applied-inference`) | — | Absent |
| 9  | Profile TPU Code (`profiling`) | `src/debug.rs` — op table / saved-tensor table / DOT graph | Conceptual |
| 10 | Programming TPUs in JAX (`jax-stuff`) | `checkpoint::checkpoint()` is the manual Rust analog of `jax.remat` | Conceptual |
| 11 | Conclusions (`conclusion`) | — | n/a |
| 12 | How to Think About GPUs (`gpus`) | Optional CUDA forward for matmul + softmax via `cuda_ffi` + `//ferric_continuum/cuda_kernels` (`--config=cuda`); gym under `cuda_gym/` | Partial (fwd GPU, bwd CPU) |

---

## Chapter 4 Deep-Dive: Component → Formula → tnsr Function

### Notation

FLOPs are multiply-accumulate pairs (one MAC = 2 FLOPs per the book convention).
"Fwd FLOPs" = forward pass only.  "Bwd FLOPs" ≈ 2 × Fwd for matmuls (each of
`∂L/∂X` and `∂L/∂W` costs one full matmul).

### Component Table

| Component | Params | Fwd FLOPs | Bwd FLOPs | tnsr function |
|-----------|--------|-----------|-----------|---------------|
| Embedding lookup | V × D | O(B·T) reads — no matmul | O(B·T) scatter | `ops/embedding.rs::embedding` |
| Unembedding + CE loss | V × D | 2·B·T·D·V | 4·B·T·D·V | `ops/loss.rs::cross_entropy` |
| Q projection | D² | 2·B·T·D² | 4·B·T·D² | `ops/linear.rs::linear` (wq) |
| K projection | D² | 2·B·T·D² | 4·B·T·D² | `ops/linear.rs::linear` (wk) |
| V projection | D² | 2·B·T·D² | 4·B·T·D² | `ops/linear.rs::linear` (wv) |
| O projection | D² | 2·B·T·D² | 4·B·T·D² | `ops/linear.rs::linear` (wo) |
| Attention scores | 0 | 2·B·T²·D | 4·B·T²·D | `ops/attention.rs::attention_scores` |
| Causal mask | 0 | O(B·T²) writes | 0 (identity grad) | `ops/attention.rs::causal_mask` |
| Softmax | 0 | O(B·T²) | O(B·T²) | `ops/attention.rs::softmax_last_dim` |
| Attention mix | 0 | 2·B·T²·D | 4·B·T²·D | `ops/attention.rs::attention_mix` |
| MLP up (w1) | D × F | 2·B·T·D·F | 4·B·T·D·F | `ops/linear.rs::linear` (w1) |
| GELU | 0 | O(B·T·F) | O(B·T·F) | `ops/activations.rs::gelu` |
| MLP down (w2) | F × D | 2·B·T·F·D | 4·B·T·F·D | `ops/linear.rs::linear` (w2) |
| LayerNorm × 2 | 2 × 2D | O(B·T·D) each | O(B·T·D) each | `ops/norm.rs::layer_norm` |
| Residual add × 2 | 0 | O(B·T·D) each | O(B·T·D) each | `ops/basic.rs::add` |

### Embedding vs Unembedding (important distinction)

`ops/embedding.rs::embedding` performs a **lookup**: token IDs `[B,T]` index
into a weight matrix `[V,D]` to produce `[B,T,D]` — this is a gather operation
with O(B·T·D) memory reads and **zero matmul FLOPs**.

The book's `6·B·T·D·V` matmul formula applies to the reverse direction:
projecting hidden states to vocabulary logits (`hidden [B,T,D] @ W_out [D,V]`).
In tnsr, this matmul is not a separate op; `ops/loss.rs::cross_entropy` receives
already-computed logits (the caller performs the matmul via `ops/linear.rs`).

### MLP: 2-matrix GELU vs Book's Gated 3-matrix FFN

| | Book reference (SwiGLU) | tnsr `TransformerBlock` |
|-|-------------------------|-------------------------|
| Architecture | W_In1, W_In2 (gate), W_Out | w1, w2 |
| Params | 3 · D · F | 2 · D · F |
| Fwd FLOPs | 18 · B·T·D·F | 12 · B·T·D·F |
| Activation | SiLU gate | GELU |
| Attn-dominant when | T > 8D | T > 6D |

`ops/activations.rs::swiglu` implements the gated variant (input `[..., 2F]`,
output `[..., F]`) but it is **not wired into `TransformerBlock`**.

### Causal Mask Cost in tnsr

`ops/attention.rs::causal_mask` iterates over every element of the `[B,T,T]`
score tensor and overwrites the upper triangle with `-∞`.  Cost: O(B·T²) writes.
A fused kernel can skip the upper triangle, halving the cost; tnsr pays the full
price.  The backward pass is an identity (masked positions produce `exp(-∞) = 0`
through softmax, so no gradient flows — `IdentityBackward` passes the incoming
gradient through unchanged).

### Backward Pass and the 6·N·T Rule

For a matmul Y = X·W with shapes (M×K) and (K×N):

```text
Fwd:  Y = X·W         → 2·M·K·N  FLOPs
Bwd:  ∂L/∂X = (∂L/∂Y)·Wᵀ  → 2·M·K·N  FLOPs  (raw_linear_dx)
      ∂L/∂W = Xᵀ·(∂L/∂Y)  → 2·M·K·N  FLOPs  (raw_linear_dw)
Total: 6·M·K·N  FLOPs per weight update
```

Summing over all weight matrices in one block:
`6 × params_matmul × tokens_per_batch = 6 × (4D² + 2DF) × BT`

For `tiny_4_7_29` (D=29, F=116, B=4, T=7):
- `params_matmul = 4×841 + 2×3364 = 10092`
- `6 × 10092 × 28 = 1,695,456 FLOPs` (linear layers only)
- Attention scores + mix add `6 × B × T² × D = 68,208 FLOPs`

These values are verified in `tests/scaling_test.rs`.

---

## Chapter 4 / 5 Deep-Dive: Gradient Checkpointing

### `jax.remat` → `checkpoint::checkpoint()`

```text
// JAX:
@functools.partial(jax.remat, policy=...)
def block_forward(x): ...

// tnsr:
let y = checkpoint("block0", policy, &[x], |xs| block.forward(&xs[0]));
```

The `checkpoint()` function in `src/checkpoint.rs` replays the forward function
during backward to recompute saved tensors, exactly like `jax.remat`.

### Two Remat Strategies

| Book strategy | Book cost | tnsr policy | tnsr implementation |
|---------------|-----------|-------------|---------------------|
| Block Remat | ~8N FLOPs/token (vs 6N) | `WholeBlockCheckpoint` | Saves `BoundaryInput` (block input x); recomputes all activations via `do_recompute()` |
| Big Matmuls Only | Save large matmul outputs, recompute cheap ops | `TransformerSelectivePolicy` | **Inspired by** the book strategy; saves softmax output (if small) + Q/K activations; recomputes GELU and LayerNorm |

**Important:** `TransformerSelectivePolicy` is an *analog*, not an exact match.
The book saves **outputs** of large matmuls (to avoid recomputing them); tnsr
saves **inputs** (to recompute the op from the input).  The set of saved tensors
also differs (tnsr makes decisions based on `SaveRole` and byte thresholds).

### `SavedTensor` Storage Variants

| Variant | When used | Memory cost |
|---------|-----------|-------------|
| `Materialized` | Default for activations | Full f32 copy |
| `Borrowed` | Parameters (`SaveRole::Parameter`) | Weak ref, zero copy |
| `Recompute` | When policy returns `MustRecompute` | Handle only (~32 bytes) |

`src/debug.rs::print_saved_tensor_table()` shows per-tensor bytes for a given run,
making the memory savings from remat directly observable.

---

## Chapter 5 Deep-Dive: Parallelism

`tnsr` is CPU-only and single-threaded, so it teaches distributed training the
only faithful way one process can: **executable symbolic cost estimates** (the
`scaling/` pattern) **plus runnable single-process simulations over `Vec<f32>`**
that make each mechanism provably correct. No real devices, threads, or network
— the simulation loops over `D` logical shards in one process. All of this lives
in `src/scaling/distributed/`.

### Collectives (ring model) → `distributed::collectives`

For `D` devices and a `bytes`-sized logical vector, per-device volume is:

```text
all-gather / reduce-scatter / broadcast :  (D−1)/D · bytes
all-reduce  = reduce-scatter + all-gather:  2·(D−1)/D · bytes
```

| Function | What it does |
|----------|--------------|
| `collective_cost(c, D, bytes)` | ring-model bytes per device for a `Collective` |
| `sim_all_reduce_sum` | naive: sum then broadcast the total to all devices |
| `sim_ring_all_reduce_sum` | reduce-scatter ∘ all-gather; bit-exact match to naive |
| `sim_all_gather` | concatenate shards to all devices |
| `sim_reduce_scatter_sum` | sum then scatter one equal slice per device |
| `sim_broadcast` | copy one device's buffer to all |

`all-to-all` is deferred (needs a cross-shard permutation beyond single-matmul
scope). PyTorch counterpart: `torch.distributed.{all_reduce,all_gather,reduce_scatter,broadcast}`.

### Device mesh → `distributed::mesh`

`DeviceMesh { shape, axis_names }` (row-major ranks) mirrors
`torch.distributed.device_mesh.DeviceMesh`: `new_1d`, `new_2d`,
`total_devices`, `axis_size(name)`, `coords(rank)`.

### Data parallel (DDP) → `distributed::data_parallel`

Every device holds a full replica; only the gradient all-reduce scales with
`dp`, and **per-device memory is independent of `dp`** (the DDP invariant).

```text
grad_allreduce_bytes/device = 2·(dp−1)/dp · grad_bytes
```

PyTorch counterpart: `torch.nn.parallel.DistributedDataParallel`.

### FSDP / ZeRO → `distributed::fsdp`

`ZeroStage {Stage1, Stage2, Stage3}`; **Stage3 == FSDP**. Each stage shards
strictly more of the per-device memory, so `Stage1 ≥ Stage2 ≥ Stage3`:

| Stage | Optimizer state | Gradients | Parameters |
|-------|-----------------|-----------|------------|
| 1     | `1/D`           | full      | full       |
| 2     | `1/D`           | `1/D`     | full       |
| 3     | `1/D`           | `1/D`     | `1/D`      |

Stage 2/3 reduce-scatter gradients; Stage 3 also all-gathers parameters — extra
comm vs DDP. PyTorch counterpart: `torch.distributed.fsdp.FullyShardedDataParallel`.

### Tensor parallel (Megatron) → `distributed::tensor_parallel`

Column-parallel matmul (`WColwise`, no all-reduce) feeding a row-parallel one
(`InnerReduction`, one all-reduce), delegating to `sharding::shard_matmul`.
`sim_column_then_row(...)` runs the full `Z = (X·W1)·W2` MLP across `tp` shards —
column-sharding the hidden dim of `W1`, row-sharding `W2`, then all-reducing the
per-device partials — and proves the result equals the unsharded computation.
PyTorch counterpart: `torch.distributed.tensor.parallel`
(`ColwiseParallel`/`RowwiseParallel`).

### Pipeline parallel → `distributed::pipeline`

GPipe bubble fraction and per-boundary activation handoff:

```text
bubble_fraction = (P − 1) / (M + P − 1)
```

for `P` stages and `M` microbatches; the bubble shrinks as `M` grows. PyTorch
counterpart: `torch.distributed.pipelining`.

### Aggregate report → `distributed::report`

`distributed_report(cfg, num_layers, &mesh, fsdp_stage, &hw)` reads the mesh's
`"dp"`/`"tp"` axes, sums per-device collective bytes across DDP + FSDP-extra +
TP, and reuses `roofline` to flag comm-vs-compute bound.
`format_distributed_report(...)` renders an ASCII table like `report::format_report`.

Golden values and invariants are verified in `tests/distributed_test.rs`.

---

## What tnsr Does NOT Cover

These topics are out of scope for a CPU-only, single-threaded, f32 library.
See the corresponding book chapters for the full treatment.

| Topic | Book chapter | Why absent from tnsr |
|-------|-------------|----------------------|
| Tensor / pipeline / FSDP parallelism | Ch.5 | Modelled in `src/scaling/distributed/` as executable estimates + single-process simulations over `Vec<f32>` — no real devices, threads, or network |
| Roofline hardware measurements | Ch.1 | tnsr provides symbolic estimates only |
| TPU / GPU programming model | Ch.2, 12 | Hardware-specific |
| LLaMA-3 specifics (GQA, RoPE, etc.) | Ch.6, 8 | Only a generic single-head block is implemented |
| KV cache (autoregressive inference) | Ch.7 | Training-only; `scaling/inference.rs` has byte estimates only |
| Serving infra | Ch.8 | Out of scope |
| Quantization, sparsity | — | Not implemented |

---

## Executable Scaling Math (`src/scaling/`)

The `scaling` module makes the above formulas runnable against a real
`TransformerConfig`:

```text
use tnsr::scaling::report::{scale_report, format_report};
use tnsr::transformer::TransformerConfig;

let cfg = TransformerConfig::tiny_4_7_29();
println!("{}", format_report(&scale_report(&cfg)));
```

| Sub-module | What it computes |
|------------|-----------------|
| `model_stats` | Total/QKVO/MLP/norm param counts; `params_matmul` for the 6·N·T rule |
| `op_cost` | Per-op fwd/bwd FLOPs + activation bytes; `dense_matmul_fwd_flops` |
| `report` | Aggregate `ScaleReport` + ASCII table formatter |
| `roofline` | Compute-vs-memory bottleneck for given FLOPs + bytes + `HardwareSpec` |
| `sharding` | 4 sharding cases as local algebra: per-device FLOPs + all-reduce bytes |
| `inference` | KV cache bytes; peak activation memory estimate |
| `distributed::collectives` | ring-model collective cost + `Vec<f32>` sims (all-reduce, all-gather, reduce-scatter, broadcast) |
| `distributed::mesh` | named multi-axis `DeviceMesh` (dp/tp axes, ranks → coords) |
| `distributed::data_parallel` | DDP grad all-reduce bytes; per-device memory invariant |
| `distributed::fsdp` | ZeRO 1/2/3 (Stage3 == FSDP) sharded memory + extra comm |
| `distributed::tensor_parallel` | Megatron column-then-row cost + matmul-equivalence sim |
| `distributed::pipeline` | GPipe bubble fraction + activation-handoff bytes |
| `distributed::report` | aggregate 2-D-mesh report + ASCII table + roofline verdict |

Golden values for `tiny_4_7_29` are verified in `tests/scaling_test.rs`.
