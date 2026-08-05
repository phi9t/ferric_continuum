# Challenge: reduce

Sum a float32 vector on the GPU: `out = sum(a[i])`. Return a length-1 vector with the scalar result.

## Contract

- Case fields: `n`, `seed`, `rtol`, `atol`
- Output shape: `[1]`

Use any strategy that is correct (tree reduction in shared memory recommended; atomics ok for v1).
