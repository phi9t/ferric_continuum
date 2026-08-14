# Mission: Use `tnsr` to Research Foundation-Model Parallelism

## Why
Use the small, readable `tnsr` stack as a research vehicle for understanding and experimenting with foundation-model parallelism across both training and inference. The aim is to connect model math, CUDA execution, memory movement, and distributed communication rather than treating parallelism as a framework black box.

## Success looks like
- Trace a Qwen3 forward or backward operation from the Rust model API to its CPU or CUDA implementation.
- Explain which `tnsr` paths execute real work and which only estimate or simulate parallel execution.
- Measure compute, memory, and communication costs and use them to choose among data, sharded-data, tensor, and pipeline parallelism.
- Extend `tnsr` with a small, validated parallel training or inference experiment whose limitations are explicit.

## Constraints
- Begin with repository-grounded experiments that remain small enough to inspect end to end.
- Treat correctness and measured data movement as prerequisites for performance conclusions.
- Lessons and references must be readable as Markdown on a remote machine, without a browser.
- Current CUDA, Rust, C++, Bazel, and distributed-systems proficiency has not yet been established; lesson depth should adapt from demonstrated work.

## Out of scope
- Turning `tnsr` immediately into a production-scale distributed framework.
- Unrelated `hello`, `foundation`, and Muon optimizer modules unless they unblock the mission.
