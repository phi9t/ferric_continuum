# AGENTS.md

**Project:** Ferric Continuum

**Tagline:** *Forging performance through parallelism and precision — in C++ and Rust.*

---

## Overview

Ferric Continuum is a Bazel-built monorepo designed to grow into an agent-driven HPC system, but the current repository focuses on foundational C++/Rust examples and a Python/C++ optimizer prototype. This document describes the **planned** agent architecture and how it should integrate once implemented.

---

## Current Scope

- C++/Rust learning modules in `ferric_continuum/hello` and `ferric_continuum/foundation`.
- A Muon optimizer prototype in `ferric_continuum/optimizers/muon` (pybind11 + numpy).
- A **CUDA gym** in `ferric_continuum/cuda_gym` (8 teaching lessons + graded challenges) plus shared production kernels in `ferric_continuum/cuda_kernels`, which also back `tnsr`'s opt-in GPU forward path. All CUDA targets are opt-in via `--config=cuda` and need a GPU to run; CPU-only `bazel test //...` skips them.
- No agent implementations are present yet (`/agents` does not exist today).

---

## Agent Roadmap (Planned)

| Agent | Purpose | Core Language | Key Interfaces | Status |
|--------|----------|----------------|----------------|--------|
| **BuildMind** | Configure and compile C++/Rust targets with Bazel. | Python | Bazel CLI | Planned |
| **PerfSmith** | Run benchmarks and track performance regressions. | Rust | Criterion.rs / Google Benchmark | Planned |
| **AsyncHermes** | Distributed task orchestration and message passing. | Rust | Tokio, gRPC / Tonic | Planned |
| **SpackSentinel** | Toolchain and dependency management for HPC stacks. | Python | Spack API | Planned |
| **ContinuumSupervisor** | Meta-controller for coordinating experiments and reports. | Python | CLI / REST, YAML / JSON | Planned |

---

## Planned Integration Flow

1. **SpackSentinel** provisions toolchains and libraries.
2. **BuildMind** compiles and links artifacts.
3. **PerfSmith** profiles and benchmarks them.
4. **AsyncHermes** deploys and manages distributed workloads.
5. **ContinuumSupervisor** aggregates results and reports.

---

## Implementation Conventions (When Agents Land)

- Each agent should expose a CLI entry point and a structured logging interface.
- Bazel targets should follow the pattern `//agents:<agent_name>`.
- Agents should be hermetic and runnable in a Bazel sandbox.
- C++ logging uses Abseil; Rust logging uses `tracing`.

---

## Agent skills

### Issue tracker

Issues and specs are tracked as local org-mode files under `.scratch/`. See `docs/agents/issue-tracker.org`.

### Triage labels

The repo uses the default five Matt Pocock skill triage roles as file-backed status values. See `docs/agents/triage-labels.md`.

### Domain docs

Ferric Continuum is treated as a multi-context repo with a root context map and context-local docs. See `docs/agents/domain.md`.

---

## Suggested Learning Path (Current Repo)

1. Run the C++/Rust hello world examples.
2. Work through foundation demos and tests.
3. Explore the Muon optimizer and its Python/C++ bridge.

See `README.md` for runnable commands and `ENGINEERING.md` for coding standards.

<!-- ultron-agentic-workflow:start -->
## Agentic engineering workflow

**Mandatory:** Read and follow `CONSTITUTION.md` before acting. Before planning,
building, fixing, or changing code, read and follow
`docs/agents/agentic-engineering.md`. Direct user instructions and more specific
repository guidance take precedence.
<!-- ultron-agentic-workflow:end -->
