# Ferric Continuum

[![CI](https://github.com/phi9t/ferric_continuum/actions/workflows/ci.yml/badge.svg?branch=ultron/mainline)](https://github.com/phi9t/ferric_continuum/actions/workflows/ci.yml)

*Forging performance through parallelism and precision — in C++ and Rust.*

Ferric Continuum is a multi-language systems playground built with Bazel. It focuses on side-by-side C++/Rust examples, clear teaching artifacts, and a Python/C++ optimizer prototype.

The repo name nods to a line in *Use of Weapons* about Minds blurring the boundary between tactics and strategy:

> “The Minds did not assume such distinctions; to them, there was no cut-off between the two. Tactics cohered into strategy, strategy disintegrated into tactics, in the sliding scale of their dialectical moral algebra.”

In that spirit, this codebase blends C++, Rust, and Python so tensor work can flow across the stack without hard cutoffs between systems-level kernels, safe concurrency, and high-level orchestration.

---

## What Lives Here

- **Collocated C++ and Rust examples** in `ferric_continuum/hello` and `ferric_continuum/foundation`.
- **Foundation modules** covering value semantics, move semantics, parameter passing, smart pointers/RAII, and constructor rules.
- **`tnsr` Rust tensor/transformer library** in `ferric_continuum/tnsr`: reverse-mode
  autograd, transformer blocks, and a HuggingFace-verified inference path for
  Qwen3 and DeepSeek-V4.1-Flash (text, multimodal, and DSpark speculative waves).
- **Muon optimizer prototype** in `ferric_continuum/optimizers/muon` using a C++ backend exposed to Python via pybind11.
- **CUDA gym** in `ferric_continuum/cuda_gym` (lessons + challenges) and shared kernels in `ferric_continuum/cuda_kernels`, with an opt-in GPU forward path for `tnsr`.
- **Bazel-first workflows** for builds, tests, and demos.

---

## Quick Start

```bash
# Build everything
bazel build //...

# Run all tests
bazel test //...
```

---

## Examples

### Hello World

```bash
bazel run //ferric_continuum/hello:hello_cc
bazel run //ferric_continuum/hello:hello_rs
```

### Foundation Demos

```bash
bazel run //ferric_continuum/foundation:value_semantics_demo_cc
bazel run //ferric_continuum/foundation:value_semantics_demo_rs

bazel run //ferric_continuum/foundation:move_semantics_demo_cc
bazel run //ferric_continuum/foundation:move_semantics_demo_rs

bazel run //ferric_continuum/foundation:parameter_passing_demo_cc
bazel run //ferric_continuum/foundation:parameter_passing_demo_rs

bazel run //ferric_continuum/foundation:smart_pointers_demo_cc
bazel run //ferric_continuum/foundation:smart_pointers_demo_rs

bazel run //ferric_continuum/foundation:constructor_rules_demo_cc
```

### Muon Optimizer (Python + C++)

```bash
# Run the demo
bazel run //ferric_continuum/optimizers/muon:muon_demo

# Run the Python test
bazel test //ferric_continuum/optimizers/muon:muon_py_test
```

### CUDA Gym (opt-in, GPU)

CUDA is disabled by default so CPU-only builds and CI stay hermetic. Enable it
with `--config=cuda` on a machine with a CUDA toolkit and GPU.

**Prerequisites:** a locally-installed CUDA toolkit (auto-detected via
`CUDA_PATH` or `/usr/local/cuda`) and a CUDA-capable GPU to *run* tests/demos.
The default arch list covers Ampere→Blackwell including **sm_100 (B200)**; note
CUDA 13.x dropped Volta (`compute_70`). Trim to your GPU for faster builds, e.g.
`--config=cuda --cuda_archs=compute_100:sm_100`.

```bash
# Shared production kernels / lessons (wildcards work under --config=cuda)
bazel test --config=cuda //ferric_continuum/cuda_kernels/...
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/...

# Lesson 01 demo
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/01_hello_gpu:hello_gpu_demo

# Challenge self-check (green). Student :grade fails until stubs are filled.
bazel test --config=cuda //ferric_continuum/cuda_gym/challenges/vector_add:grade_reference

# tnsr GPU forward (matmul + softmax)
bazel test --config=cuda //ferric_continuum/tnsr:cuda_forward_tests
```

See `ferric_continuum/cuda_gym/README.md` for the full lesson order, challenge
workflow, and architecture flags (`--cuda_archs=...`).

---

## `tnsr`: Rust transformer inference

`ferric_continuum/tnsr` is the Bazel-first Rust tensor/transformer library. On
top of readable reverse-mode autograd and transformer-block mechanics, it ships
a CPU-first (f32) inference path for **Qwen3** and **DeepSeek-V4.1-Flash**. The
DeepSeek work is organized into waves — text (Wave 1), multimodal vision
(Wave 2), DSpark speculative decoding (Wave 3), and cost/perf accounting
(Wave 4) — each verified for numeric parity against the upstream reference.

```bash
# CPU behavior suite
bazel test //ferric_continuum/tnsr/...

# DeepSeek-V4.1-Flash inference CLI (text-only by default). `--token-ids` is the
# tokenizer-free isolation surface parity relies on; `--dump-logits` writes the
# last-position logits row as JSON.
bazel run //ferric_continuum/tnsr:deepseek_v41_infer -- \
  --model-dir <checkpoint-dir> --token-ids 1,2,3 \
  --max-new-tokens 0 --dump-logits /tmp/logits.json

# Qwen3 inference CLI (same isolation pattern)
bazel run //ferric_continuum/tnsr:qwen3_infer -- --help
```

### Parity verifiers

Each DeepSeek-V4.1-Flash wave has a self-contained verifier that builds the
source, runs the Bazel suites, and checks tiny end-to-end logits parity against
an upstream-executed reference. Real 510 GB weights are not shipped, so
real-checkpoint parity reports an explicit **SKIP** (never presented as PASS)
unless `DEEPSEEK_V41_MODEL_DIR`/`MODEL_DIR` points at local weights.

```bash
ferric_continuum/tnsr/tools/verify_deepseek_v41_text_compat.sh       # Wave 1
ferric_continuum/tnsr/tools/verify_deepseek_v41_multimodal_compat.sh # Wave 2
ferric_continuum/tnsr/tools/verify_deepseek_v41_dspark_compat.sh     # Wave 3
ferric_continuum/tnsr/tools/verify_deepseek_v41_perf_compat.sh       # Wave 4
```

See `ferric_continuum/tnsr/README.md` for the full module tour and
`ferric_continuum/tnsr/docs/deepseek_v41/spec.org` for the wave specification.

---

## Repository Layout

```
ferric_continuum/
├── hello/                 # C++/Rust hello world example
├── foundation/            # Core C++/Rust concepts with demos and tests
├── cuda_gym/              # CUDA lessons + graded challenges
├── cuda_kernels/          # Shared production GEMM / softmax / attention kernels
├── tnsr/                  # Rust tensor/transformer lib + Qwen3/DeepSeek inference
└── optimizers/muon/       # Muon optimizer (pybind11 + numpy)
```

---

## Documentation

- `CONSTITUTION.md` - Engineering principles (read before changing code)
- `docs/agents/agentic-engineering.md` - Agentic engineering workflow
- `AGENTS.md` - Agent roadmap and design notes (planned system)
- `CXX_ENGINEERING.md` - C++ engineering fundamentals (short guide)
- `ENGINEERING.md` - Coding standards and tooling guidance
- `ferric_continuum/tnsr/README.md` - `tnsr` module tour (autograd → inference)
- `ferric_continuum/tnsr/docs/deepseek_v41/spec.org` - DeepSeek-V4.1-Flash wave spec
- `ferric_continuum/hello/README.md` - Hello world walkthrough
- `ferric_continuum/foundation/README.md` - Foundation module deep dive

---

## Repository hygiene (desensitization)

Tracked files must not leak personally- or machine-identifiable data (user
home absolute paths, real usernames, MAC addresses, routable IPv4 literals, or
personal emails). This is enforced by a dependency-free checker:

```
# Scan tracked files, including the checker self-test (exit 1 on findings)
scripts/desensitize.sh

# Auto-fix the safe substitutions (home abspaths -> ${HOME}, pinned bazel
# launcher path -> plain `bazel`), then re-run the scan
python3 tools/desensitize_check.py --fix

# Install local hooks:
# - commit-msg scans commit text
# - pre-commit scans staged content
# - pre-push scans tracked files and outgoing commits
scripts/install-git-hooks.sh
```

The same wrapper runs as the blocking `Desensitize` job on pull requests and
pushes, and PR commits are scanned with `scripts/check-commit-scrub.sh`.
Allowlisted intended-public tokens live in
`tools/desensitize_allowlist.txt`.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

**Ferric Continuum** — *Forging performance through parallelism and precision.*
