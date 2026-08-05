# Challenge: vector_add

Implement element-wise `out[i] = a[i] + b[i]` for equal-length FP32 vectors on the GPU.

## Contract

- Case JSON fields: `n` (element count), `seed`, `rtol`, `atol`
- Binary argv[1]: `{"case": {...}, "seed": N}` (see `harness/challenge_io.hh`)
- Output: JSON with `status`, `shape` (`[n]`), `data` (length `n`), `elapsed_ms`

## Student entry

Fill in the body of `student.cu` (the kernel launch + copies). Do not change the JSON I/O contract.

```bash
# Reference self-check (should pass):
bazel test --config=cuda //ferric_continuum/cuda_gym/challenges/vector_add:grade_reference
# Student grade (fails until student.cu is filled):
bazel test --config=cuda //ferric_continuum/cuda_gym/challenges/vector_add:grade
```
