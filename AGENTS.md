# AGENTS.md

**Project:** Ferric Continuum

**Tagline:** *Forging performance through parallelism and precision — in C++ and Rust.*

---

## Overview

In *Ferric Continuum*, **agents** are autonomous modules that extend and maintain the system's self-organizing intelligence.

They assist with:

- Building and testing C++ and Rust components via **Bazel**
- Managing dependencies and toolchains via **Spack**
- Running distributed workloads and profiling performance
- Coordinating benchmarking and continuous experimentation

Each agent operates as a self-contained subsystem, capable of acting independently or collaborating with others—
much like the specialized Minds of the *Culture* universe.

---

## Agent Taxonomy

| Agent | Purpose | Core Language | Key Interfaces |
|--------|----------|----------------|----------------|
| **BuildMind** | Configures and compiles all C++/Rust targets using Bazel toolchains; interfaces with Spack-provisioned compilers and libraries. | Python | Bazel CLI, Spack API |
| **PerfSmith** | Runs benchmarks, gathers profiling data, and tracks performance regressions. | Rust | Google Benchmark, Criterion.rs |
| **AsyncHermes** | Manages distributed task orchestration and message passing. | Rust | Tokio, gRPC / Tonic |
| **SpackSentinel** | Monitors Spack environments and synchronizes toolchain packages for Bazel builds (CUDA, BLAS, LAPACK, MPI). | Python | Spack API, Bazel toolchain rules |
| **ContinuumSupervisor** | Meta-controller that coordinates all agents, orchestrates experiments, and generates reports. | Python | CLI / REST, YAML / JSON config |

---

## Integration Flow

**SpackSentinel** provisions toolchains and libraries

**BuildMind** compiles and links artifacts

**PerfSmith** profiles and benchmarks them

**AsyncHermes** deploys and manages distributed workloads

**ContinuumSupervisor** oversees the entire pipeline and aggregates metrics

---

## Development Philosophy

Each agent:

- Operates under **hermetic, reproducible builds**
- Is **observable** (metrics, logs, telemetry)
- Exposes a **CLI and API interface**
- Runs in **containerized Bazel environments** or **Kubernetes pods**

Agents communicate through async channels, local sockets, or via the Bazel workspace graph to ensure deterministic reproducibility.

**Coding Standards:** All C++ and Rust code follows the engineering standards documented in [ENGINEERING.md](ENGINEERING.md), which includes:
- Best practices from Google C++ Style Guide and C++ Core Guidelines
- **Namespace usage**: Never use `using namespace` (see ENGINEERING.md § Project-Specific Standards)
- **Logging standards**: Use Abseil logging (C++) and tracing (Rust) for structured logging (see ENGINEERING.md § Logging Standards)
- **Testing and benchmarking**: Google Test/Benchmark (C++) and built-in testing/Criterion (Rust)

Refer to [ENGINEERING.md](ENGINEERING.md) for detailed guidelines, examples, and tooling configuration.

---

## Example Workflow

```bash
# 1. Provision the HPC environment
spack env activate ferric-env
spack install cuda openblas lapack

# 2. Launch BuildMind
bazel run //agents:buildmind -- --targets=//cpp:matrix_mul //rust:parallel_sort

# 3. Run performance suite
bazel run //agents:perfsmith -- --compare=baseline.json

# 4. Deploy distributed training simulation
bazel run //agents:asynchromes -- --nodes=4 --role=trainer

# 5. Aggregate results and publish report
bazel run //agents:continuum_supervisor -- --summary
```

---

## Curriculum Overview

**Week 1 — Systems Setup & Toolchain Deep Dive**  
- Install and configure Bazel and Spack environments  
- Explore compiler toolchains for C++ and Rust  
- Mini-project: Build a simple "Hello World" with Bazel and Spack  

**Week 2 — Advanced Bazel Build Rules (C++ & Rust)**  
- Write custom Bazel build and test rules for both C++ and Rust targets  
- Understand Bazel's sandboxing and hermetic build features  
- Mini-project: Create a multi-language build with C++/Rust cross-dependencies  

**Week 3 — Spack Package Management & Integration (C++ & Rust)**  
- Manage HPC libraries with Spack (CUDA, BLAS, LAPACK)  
- Integrate Spack-managed dependencies into Bazel builds for both C++ and Rust  
- Mini-project: Build numerical libraries in C++ and Rust linked against Spack packages  

**Week 4 — Systems Programming Fundamentals (C++ & Rust)**  
- Compare C++17/20 and Rust memory management models  
- Explore ownership, borrowing, and RAII patterns in both languages  
- Mini-project: Implement a memory-efficient data structure in both C++ and Rust  

**Week 5 — Concurrency and Parallelism (C++ & Rust)**  
- Compare C++ threads/OpenMP with Rust's std::thread and async/await  
- Implement parallel algorithms using both language ecosystems  
- Mini-project: Build a parallel matrix multiplication kernel in C++ and Rust  

**Week 6 — Benchmarking and Profiling (C++ & Rust)**  
- Use Google Benchmark (C++) and Criterion.rs (Rust) for performance measurement  
- Profile both C++ and Rust applications with perf, valgrind, and flamegraph tools  
- Mini-project: Benchmark and optimize a sorting algorithm in both languages  

**Week 7 — Distributed Workload Orchestration (C++ & Rust)**  
- Compare gRPC implementations in C++ and Rust (Tonic)  
- Design async message passing systems in both languages  
- Mini-project: Build a distributed task scheduler with C++ workers and Rust orchestrator  

**Week 8 — Continuous Integration and Experimentation (C++ & Rust)**  
- Set up CI pipelines using Bazel and containerized builds for both languages  
- Automate benchmarking and regression detection across C++ and Rust components  
- Mini-project: Implement a CI workflow tracking performance of multi-language builds  

**Week 9 — Observability and Telemetry (C++ & Rust)**  
- Integrate logging, metrics, and tracing into C++ and Rust agents  
- Use Prometheus and Grafana for monitoring mixed-language systems  
- Mini-project: Build an observability dashboard aggregating metrics from both languages  

**Week 10 — Meta-Controller and System Synthesis (C++ & Rust)**  
- Coordinate multiple agents with CLI and REST interfaces across languages  
- Aggregate experimental data from C++ and Rust components  
- Mini-project: Develop a meta-controller orchestrating a full C++/Rust pipeline  

---

## Learning Outcomes

By the end of this 10-week program, participants will have:

- Mastered Bazel and Spack for reproducible, hermetic builds across C++ and Rust  
- Developed high-performance applications in both C++ and Rust, comparing their approaches to parallelism and concurrency  
- Implemented distributed systems and asynchronous orchestration using multi-language architectures  
- Utilized benchmarking, profiling, and observability tools for both C++ and Rust codebases  
- Built integrated CI pipelines and meta-controllers managing complex mixed-language HPC workflows  
- Gained practical experience through hands-on projects that interleave C++ and Rust, reflecting real-world polyglot HPC challenges  
- Developed fluency in navigating and integrating C++ and Rust within the same codebase and build system
