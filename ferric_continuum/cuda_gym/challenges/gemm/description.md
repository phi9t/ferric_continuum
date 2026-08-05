# Challenge: gemm

Compute dense GEMM `C = A * B` in row-major FP32:
- A is `(m x k)`, B is `(k x n)`, C is `(m x n)`.

You may implement a naive kernel; linking the shared production kernel is also fine for the reference.

## Contract

Case fields: `m`, `n`, `k`, `seed`, `rtol`, `atol`. Output shape: `[m, n]`.
