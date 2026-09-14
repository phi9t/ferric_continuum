Status: complete

Files changed:
- ferric_continuum/tnsr/BUILD.bazel
- ferric_continuum/tnsr/src/dtensor/mod.rs
- ferric_continuum/tnsr/src/dtensor/mesh.rs
- ferric_continuum/tnsr/tests/mesh_sim_test.rs

Summary:
- Added the 5D mesh semantics API for TorchTitan-style dimensions.
- Re-exported MeshAxis, MeshError, ParallelDims5D, and RankCoord5D from tnsr::dtensor.
- Registered the new mesh module in the Bazel tnsr source list.
- Added mesh simulation tests for dense product helpers, world-size validation, row-major rank coordinates, and out-of-range rank errors.

Tests run with results:
- RED: bazel test //ferric_continuum/tnsr:mesh_sim_tests failed to build before implementation with unresolved imports for MeshAxis, MeshError, ParallelDims5D, and RankCoord5D.
- GREEN: bazel test //ferric_continuum/tnsr:mesh_sim_tests passed with 1 test target passing after implementation.

Commit hash:
- 31f1207 Add dtensor 5D mesh semantics
- This report was stamped after the implementation commit because a commit cannot contain its own final hash.

Self-review notes:
- Scope is limited to the dtensor BUILD entry, module exports, new mesh module, mesh simulation tests, and this report file.
- The coord mapping uses tp as the fastest-varying axis, then cp, dp_shard, dp_replicate, and pp.
- ParallelDims5D::new asserts all axes are at least 1, matching the exact task brief.
- MeshError includes ZeroAxis for the declared interface and Display formatting, although new() panics on zero axes as specified.

Concerns:
- The worktree already had an untracked .scratch/dtensor-mesh-simulation/ directory before this task; it was left untouched.
