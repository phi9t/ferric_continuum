//! CUDA-vs-CPU agreement tests for tnsr (GPU-tagged; requires `--config=cuda`).
//!
//!     bazel test --config=cuda //ferric_continuum/tnsr:cuda_forward_tests
//!
//! Compares GPU FFI output to the dedicated CPU helpers — no process-env
//! mutation, so tests stay safe under parallel libtest.

use tnsr::ops::{attention, linear};
use tnsr::tensor::{Shape, TensorValue};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn gemm_cuda_matches_cpu() {
    assert!(
        tnsr::cuda_ffi::use_cuda(),
        "cuda_forward_tests require --config=cuda (crate feature `cuda`)"
    );
    let m = 7usize;
    let k = 5usize;
    let n = 9usize;
    let mut a = Vec::with_capacity(m * k);
    let mut b = Vec::with_capacity(k * n);
    for i in 0..(m * k) {
        a.push((i as f32) * 0.01 - 0.2);
    }
    for i in 0..(k * n) {
        b.push((i as f32) * 0.02 - 0.3);
    }

    let gpu = tnsr::cuda_ffi::gemm_f32(m, n, k, &a, &b).expect("cuda gemm");
    let x = TensorValue::from_vec(Shape(vec![m, k]), a);
    let w = TensorValue::from_vec(Shape(vec![k, n]), b);
    let cpu = linear::raw_linear_forward_cpu(&x, &w);

    let err = max_abs_diff(&gpu, cpu.data.as_ref());
    assert!(err < 1e-4, "gemm mismatch max abs err {err}");
}

#[test]
fn softmax_cuda_matches_cpu() {
    assert!(
        tnsr::cuda_ffi::use_cuda(),
        "cuda_forward_tests require --config=cuda (crate feature `cuda`)"
    );
    let rows = 4usize;
    let cols = 16usize;
    let mut xdata = Vec::with_capacity(rows * cols);
    for i in 0..(rows * cols) {
        xdata.push((i as f32) * 0.05 - 1.0);
    }

    let gpu = tnsr::cuda_ffi::softmax_f32(rows, cols, &xdata).expect("cuda softmax");
    let x = TensorValue::from_vec(Shape(vec![rows, cols]), xdata);
    let cpu = attention::raw_softmax_cpu(&x);

    let err = max_abs_diff(&gpu, cpu.data.as_ref());
    assert!(err < 1e-5, "softmax mismatch max abs err {err}");
}
