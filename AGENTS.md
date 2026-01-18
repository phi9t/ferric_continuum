# AGENTS.md

**Project:** Ferric Continuum

**Tagline:** *Forging performance through parallelism and precision — in C++ and Rust.*

---

## Overview

Ferric Continuum is designed to grow into an agent-driven HPC system, but the current repository focuses on foundational C++/Rust examples and a Python/C++ optimizer prototype. This document describes the **planned** agent architecture and how it should integrate once implemented.

---

## Current Scope

- C++/Rust learning modules in `ferric_continuum/hello` and `ferric_continuum/foundation`.
- A Muon optimizer prototype in `ferric_continuum/optimizers/muon` (pybind11 + numpy).
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

## Suggested Learning Path (Current Repo)

1. Run the C++/Rust hello world examples.
2. Work through foundation demos and tests.
3. Explore the Muon optimizer and its Python/C++ bridge.

See `README.md` for runnable commands and `ENGINEERING.md` for coding standards.
