//! Optional CUDA forward path for gemm / softmax via `cuda_kernels` C ABI.
//!
//! Enabled only when the crate is built with `--features cuda` (Bazel:
//! `--config=cuda` selects this feature and links `//ferric_continuum/cuda_kernels`).
//! Autograd backward always stays on the CPU path after forward returns host data.

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::c_int;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FerricCudaStatus {
        Ok = 0,
        ErrInvalidArg = 1,
        ErrDevice = 2,
    }

    extern "C" {
        pub fn ferric_cuda_gemm_f32(
            m: c_int,
            n: c_int,
            k: c_int,
            a_host: *const f32,
            b_host: *const f32,
            c_host: *mut f32,
        ) -> FerricCudaStatus;

        pub fn ferric_cuda_softmax_f32(
            rows: c_int,
            cols: c_int,
            x_host: *const f32,
            out_host: *mut f32,
        ) -> FerricCudaStatus;
    }
}

/// Returns true when this build linked CUDA kernels and the caller has not forced CPU.
pub fn use_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Optional force-CPU override for debugging without rebuilding.
        match std::env::var("FERRIC_TNSR_DEVICE") {
            Ok(v) if v.eq_ignore_ascii_case("cpu") => false,
            _ => true,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Row-major C = A(m×k) · B(k×n). Returns None if CUDA is off or a call fails.
#[cfg(feature = "cuda")]
pub fn gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    assert_eq!(a.len(), m.saturating_mul(k));
    assert_eq!(b.len(), k.saturating_mul(n));
    if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
        return None;
    }
    let mut c = vec![0.0f32; m.saturating_mul(n)];
    // Safety: pointers are valid for the host-buffer contract of the C ABI.
    let status = unsafe {
        ffi::ferric_cuda_gemm_f32(
            m as i32,
            n as i32,
            k as i32,
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
        )
    };
    if status == ffi::FerricCudaStatus::Ok {
        Some(c)
    } else {
        None
    }
}

#[cfg(not(feature = "cuda"))]
pub fn gemm_f32(_m: usize, _n: usize, _k: usize, _a: &[f32], _b: &[f32]) -> Option<Vec<f32>> {
    None
}

/// Row-wise softmax over last dim; `x` has shape (rows×cols).
#[cfg(feature = "cuda")]
pub fn softmax_f32(rows: usize, cols: usize, x: &[f32]) -> Option<Vec<f32>> {
    assert_eq!(x.len(), rows.saturating_mul(cols));
    if rows > i32::MAX as usize || cols > i32::MAX as usize {
        return None;
    }
    let mut out = vec![0.0f32; x.len()];
    let status = unsafe {
        ffi::ferric_cuda_softmax_f32(
            rows as i32,
            cols as i32,
            x.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if status == ffi::FerricCudaStatus::Ok {
        Some(out)
    } else {
        None
    }
}

#[cfg(not(feature = "cuda"))]
pub fn softmax_f32(_rows: usize, _cols: usize, _x: &[f32]) -> Option<Vec<f32>> {
    None
}
