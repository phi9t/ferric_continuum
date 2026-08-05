# Transitive closure of PyTorch ops in a Qwen3 transformer

**Question:** what is the full set of low-level PyTorch (ATen) ops that a
transformer decomposes into, using HuggingFace's **Qwen3** as the example?

**Method:** empirical, not source-reading. A tiny `Qwen3ForCausalLM`
(`hidden_size=32`, 2 layers, 4 query heads / 2 KV heads → GQA, `head_dim=8`,
seq len 8) is run under a `torch.utils._python_dispatch.TorchDispatchMode` that
records every ATen op the whole `nn.Module` stack dispatches to. Dispatch mode
sits *below* the Python module code, so it observes the true transitive closure
after all decompositions. Ops are also attributed to the leaf `nn.Module` that
emitted them via forward hooks. Reproduce with `trace_qwen3.py` (see bottom).

Environment: `torch 2.13.0+cpu`, `transformers 5.14.1`.

Counts below are per this tiny config; the *set* of ops is what matters and is
stable across sizes (more layers/heads just scales the counts).

---

## 1. Closure size at a glance

| Configuration | distinct ATen ops | total dispatches |
|---|---|---|
| forward, `attn_implementation="eager"` | 29 | 273 |
| forward, `attn_implementation="sdpa"` | 25 | 202 |
| forward + backward, `eager` | 39 | 716 |
| forward + backward, `sdpa` | 35 | 599 |

`sdpa` has fewer ops because the whole scaled-dot-product-attention math
collapses into one fused kernel (`aten::_scaled_dot_product_flash_attention_for_cpu`)
instead of decomposing into `bmm`/`softmax`/`mul`.

---

## 2. Forward closure — eager attention (29 ops)

The full transitive closure of a Qwen3 forward pass with the reference (`eager`)
attention, sorted by call count:

```
   33  aten::view                 24  aten::unsqueeze         9  aten::rsqrt        1  aten::cos
   32  aten::mul                  22  aten::add               8  aten::slice        1  aten::embedding
   24  aten::_unsafe_view         16  aten::expand            6  aten::clone        1  aten::le
   15  aten::mm                   15  aten::t                 5  aten::arange       1  aten::scalar_tensor
   11  aten::transpose             9  aten::cat               5  aten::bmm          1  aten::sin
    9  aten::mean                  9  aten::pow               5  aten::lift_fresh   1  aten::where
    4  aten::neg                   2  aten::_softmax          2  aten::silu         1  aten::_to_copy
    1  aten::alias
```

### Categorized

| Category | ATen ops |
|---|---|
| **Matmul / linear algebra** | `mm` (all `nn.Linear`: Q/K/V/O proj, gate/up/down, lm_head), `bmm` (attention `QKᵀ`, `PV`, RoPE freq outer product) |
| **Elementwise** | `mul`, `add`, `pow`, `rsqrt`, `neg`, `silu`, `cos`, `sin`, `div`* |
| **Normalization** | `mean`, `pow`, `rsqrt`, `mul` (compose Qwen3 RMSNorm) |
| **Softmax** | `_softmax` |
| **Shape / view (no compute)** | `view`, `_unsafe_view`, `transpose`, `unsqueeze`, `squeeze`*, `expand`, `slice`, `cat`, `clone`, `t`, `alias`, `lift_fresh` |
| **Indexing / gather** | `embedding` |
| **Construction / misc** | `arange`, `scalar_tensor`, `_to_copy`, `le`, `where` (causal-mask / dtype plumbing) |

\* `div`, `squeeze` appear in the backward closure (§4).

---

## 3. Ops mapped to transformer components (leaf-module attribution)

Attributed to the leaf `nn.Module` that emitted them:

| Transformer piece | `nn.Module` | ATen ops it decomposes into |
|---|---|---|
| **Token embedding** | `nn.Embedding` | `embedding` |
| **All projections** (Q, K, V, O, gate, up, down, lm_head) | `nn.Linear` | `t` (weight transpose), `mm` (matmul), `view` / `_unsafe_view` (2D↔3D reshape around the matmul) |
| **RMSNorm** (input & post-attn norms, and Qwen3's per-head q/k norms) | `Qwen3RMSNorm` | `pow` (x²), `mean` (over hidden), `add` (+eps), `rsqrt`, `mul` ×2 (normalize, then scale by weight) |
| **Rotary position embedding** | `Qwen3RotaryEmbedding` | `arange`→`_to_copy`, `bmm`/`expand`/`transpose`/`view` (inv_freq ⊗ positions), `cat`, `cos`, `sin` (build the cos/sin tables) |
| **RoPE application** (`apply_rotary_pos_emb`, functional) | — (module-free) | `mul`, `add`, `neg`, `cat`, `slice`, `unsqueeze` (rotate-half then `x*cos + rotate_half(x)*sin`) |
| **Attention core — eager** | `Qwen3Attention` (functional math) | `unsqueeze`/`expand`/`reshape` (GQA KV-head repeat), `bmm`/`matmul` (`QKᵀ`), `mul` (× `1/√d`), `add` (+ mask), `_softmax`, `bmm` (`P·V`), `transpose` |
| **Attention core — sdpa** | same | one fused `_scaled_dot_product_flash_attention_for_cpu` instead of the `bmm`/`softmax`/`mul` chain |
| **MLP activation** (SwiGLU gate) | `SiLUActivation` | `silu` |
| **MLP combine** | `Qwen3MLP` (functional) | `mul` (`silu(gate) * up`) — plus the three `nn.Linear` above |
| **Residual adds** | — (functional `+`) | `add` |
| **Causal mask build** | — (functional) | `arange`, `le`, `where`, `scalar_tensor`, `slice`, `mul` |

**GQA note:** Qwen3 uses grouped-query attention (fewer KV heads than query
heads). The KV-head replication (`repeat_kv`) shows up as
`unsqueeze` + `expand` + `reshape`/`_unsafe_view` rather than a real copy — a
view-only broadcast before the attention matmuls.

---

## 4. Backward pass adds these ops (training closure)

Forward+backward (`eager`) reaches **39** distinct ops. The forward set plus:

| New op (backward) | Produced by |
|---|---|
| `sum` | reduction/broadcast grads (bias-like, RMSNorm, matmul reductions) |
| `t`, `mm` (many more) | `mm` backward → two matmuls per linear (∂X, ∂W) |
| `detach`, `alias` | autograd graph plumbing |
| `_softmax_backward_data` | softmax backward |
| `silu_backward` | SiLU backward |
| `embedding_dense_backward` | embedding table gradient (scatter-add) |
| `slice_backward` | grad of `slice` (RoPE half-splits, masks) |
| `div`, `squeeze`, `neg` | RMSNorm / RoPE backward |
| `zeros`, `ones_like` | grad initialization (loss seed = `ones_like`) |

With `sdpa`, the single attention backward op is
`aten::_scaled_dot_product_flash_attention_for_cpu_backward` (replacing the
`bmm`/`_softmax_backward_data` chain).

---

## 5. Minimal op kernel set to reimplement Qwen3

If you wanted to implement Qwen3 inference from scratch, the *compute* primitives
(dropping pure view/metadata ops, which are just stride math) are:

- **`matmul`** (`mm`/`bmm`) — dominates FLOPs: all projections + attention scores/values.
- **`softmax`** — attention weights (or a fused SDPA kernel).
- **elementwise:** `mul`, `add`, `pow`, `rsqrt` (→ RMSNorm), `silu`, `neg`, `cos`, `sin` (→ RoPE).
- **reductions:** `mean` (RMSNorm), `sum` (backward only).
- **gather:** `embedding`.
- **masking:** `where` / compare (`le`) for the causal mask.

Everything else in the closure (`view`, `transpose`, `expand`, `unsqueeze`,
`cat`, `slice`, `t`, `clone`, `_unsafe_view`, `lift_fresh`, `_to_copy`, `alias`)
is layout/metadata, not arithmetic.

This is exactly the set `ferric_continuum`'s `tnsr` and `cuda_kernels` target:
`gemm` (matmul), `softmax`, RMSNorm/RoPE/GQA (elementwise + reductions), and the
single-head attention primitive.

---

## Reproduce

```bash
python -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python trace_qwen3.py eager              # forward, decomposed attention
python trace_qwen3.py sdpa               # forward, fused attention
python trace_qwen3.py eager --backward   # full training closure
```

`trace_qwen3.py` builds a tiny Qwen3 and records ops via `TorchDispatchMode`
with per-leaf-module attribution (script in this repo's trace tooling).
