//! Checkpoint I/O for DeepSeek/Qwen-style safetensors directories.
//!
//! This is the deep seam under the DeepSeek V4.1 loader: it owns the shard
//! byte buffers, parses the safetensors headers once into an owned index, and
//! exposes a small total interface — [`Checkpoint::open`], [`Checkpoint::has`],
//! [`Checkpoint::float`], [`Checkpoint::float_exact`], and
//! [`Checkpoint::linear_weight`] — that decodes bf16/f16/f32 and block-scaled
//! FP8-E4M3 / FP4-E2M1 weights to f32.
//!
//! Buffers are **owned** (`Vec<u8>` per shard) and tensor bytes are sliced from
//! them on demand, so nothing is leaked for `'static`; a `Checkpoint` frees all
//! shard memory when dropped.  This matters for the real 510GB release and any
//! long-lived server that opens more than one checkpoint.
//!
//! Layout/quant conventions match `convert.py`:
//!
//! * ordinary linears are stored `[out, in]` (PyTorch); callers transpose.
//! * FP8 weights carry a `.scale` side-car of E8M0 block exponents (block 32).
//! * FP4 weights are E2M1 packed (two values/byte) with the same E8M0 scale.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use half::bf16;
use safetensors::tensor::{Dtype, SafeTensors};

/// Source quantization of a loaded tensor.  Wave 1 always dequantizes to f32
/// for execution; this records what the checkpoint actually stored so later
/// native FP8/FP4 kernel tickets can reconstruct the packed form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// Plain float (bf16/f16/f32) decoded directly to f32.
    Float,
    /// FP8 E4M3 weight with an E8M0 block-scale (block size 32).
    Fp8E4m3,
    /// FP4 E2M1 packed weight (two values/byte) with an E8M0 block-scale.
    Fp4E2m1,
}

/// FP8/FP4 block-scale block size (matches `convert.py`).
pub const FP_BLOCK_SIZE: usize = 32;

/// FP4 (E2M1) code -> value table, matching upstream `convert.py::FP4_TABLE`.
const FP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// -- error / shape helpers -------------------------------------------------

pub(crate) fn expect_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<(), String> {
    if got == expected {
        Ok(())
    } else {
        Err(format!(
            "tensor `{name}` expected {expected:?}, got {got:?}"
        ))
    }
}

fn expect_numel(name: &str, got: usize, expected: usize) -> Result<(), String> {
    if got == expected {
        Ok(())
    } else {
        Err(format!(
            "tensor `{name}` expected {expected} elements, got {got}"
        ))
    }
}

// -- dtype decode ----------------------------------------------------------

/// Decode a float byte buffer (bf16/f16/f32) into f32.
fn decode_float(name: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(format!("tensor `{name}`: bf16 byte length not even"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(format!("tensor `{name}`: f16 byte length not even"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(format!(
                    "tensor `{name}`: f32 byte length not multiple of 4"
                ));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        other => Err(format!(
            "tensor `{name}`: unsupported float dtype {other:?}"
        )),
    }
}

/// Decode an FP8 E4M3 byte to f32 (1 sign, 4 exponent bias 7, 3 mantissa;
/// `0x7F`/`0xFF` are NaN, matching `float8_e4m3fn`).
pub(crate) fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        // subnormal: value = mant/8 * 2^(1-bias)
        sign * (mant as f32 / 8.0) * 2f32.powi(1 - 7)
    } else {
        // normal: value = (1 + mant/8) * 2^(exp-bias)
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Decode an E8M0 scale byte to a positive f32 (value = 2^(byte-127); `0xFF`
/// is NaN, matching `float8_e8m0fnu`).
pub(crate) fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return f32::NAN;
    }
    2f32.powi(byte as i32 - 127)
}

/// Guard a decoded FP8/FP4 weight against non-finite values.  A NaN/Inf in a
/// weight or scale byte would silently poison every downstream matmul, so we
/// reject it at load time with a named error rather than propagate it.
fn reject_nonfinite(name: &str, data: &[f32]) -> Result<(), String> {
    if let Some(pos) = data.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "tensor `{name}`: non-finite value ({}) at element {pos} after dequant",
            data[pos]
        ));
    }
    Ok(())
}

/// Raw bytes for a scale side-car tensor, kept in the source dtype (E8M0 is an
/// opaque 1-byte type exposed as U8/F8_E4M3/F8_E5M2 by different writers).
fn scale_bytes(name: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<u8>, String> {
    match dtype {
        Dtype::U8 | Dtype::F8_E4M3 | Dtype::F8_E5M2 => Ok(bytes.to_vec()),
        other => Err(format!(
            "tensor `{name}`: unsupported scale dtype {other:?} (expected 1-byte E8M0)"
        )),
    }
}

// -- checkpoint index ------------------------------------------------------

/// Metadata for one tensor: which shard holds it, its dtype, shape, and the
/// byte range within that shard's owned data buffer.
struct Entry {
    shard: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A flat view over one or more owned safetensors shard buffers, addressed by
/// tensor name.  Owns its buffers (no `Box::leak`); drops all memory on drop.
pub struct Checkpoint {
    shards: Vec<Vec<u8>>,
    index: BTreeMap<String, Entry>,
}

impl Checkpoint {
    /// Open a single-file `model.safetensors` checkpoint.
    pub fn open_single(path: &Path) -> Result<Checkpoint, String> {
        Checkpoint::open(&[path.to_path_buf()])
    }

    /// Open one or more shard files, building an owned name -> entry index.
    ///
    /// Each file is read into an owned buffer whose header is parsed once; the
    /// per-tensor byte offsets are read from the header (relative to the data
    /// section) and stored so [`Checkpoint::bytes`] can slice the owned buffer
    /// without leaking it for `'static`.
    pub fn open(paths: &[PathBuf]) -> Result<Checkpoint, String> {
        let mut shards: Vec<Vec<u8>> = Vec::with_capacity(paths.len());
        let mut index = BTreeMap::new();
        for (shard_idx, path) in paths.iter().enumerate() {
            let raw = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
            // Parse the header to learn each tensor's dtype/shape and its byte
            // range within the data section; keep the owned buffer alive.
            let (header_len, meta) = SafeTensors::read_metadata(&raw)
                .map_err(|e| format!("parse {}: {e}", path.display()))?;
            let data_base = header_len + 8;
            for (name, info) in meta.tensors() {
                let (off0, off1) = info.data_offsets;
                let start = data_base + off0;
                let end = data_base + off1;
                if end > raw.len() {
                    return Err(format!(
                        "tensor `{name}` offsets [{off0}, {off1}] exceed shard {}",
                        path.display()
                    ));
                }
                index.entry(name).or_insert(Entry {
                    shard: shard_idx,
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    start,
                    end,
                });
            }
            shards.push(raw);
        }
        Ok(Checkpoint { shards, index })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn entry(&self, name: &str) -> Result<&Entry, String> {
        self.index
            .get(name)
            .ok_or_else(|| format!("missing required tensor `{name}`"))
    }

    /// Raw byte slice for `name` within its owning shard buffer.
    fn bytes(&self, name: &str) -> Result<(&Entry, &[u8]), String> {
        let e = self.entry(name)?;
        Ok((e, &self.shards[e.shard][e.start..e.end]))
    }

    /// Decode a float tensor to f32 with its shape.
    pub fn float(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let (e, bytes) = self.bytes(name)?;
        Ok((decode_float(name, e.dtype, bytes)?, e.shape.clone()))
    }

    pub fn float_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, String> {
        let (data, shape) = self.float(name)?;
        expect_shape(name, &shape, expected)?;
        Ok(data)
    }

    /// Decode a possibly-quantized `[out, in]` linear weight to f32 `[out, in]`
    /// row-major, returning the source quant kind.  Detects FP8/FP4 by the
    /// dtype of the weight and reads the `.scale` side-car when quantized.
    pub fn linear_weight(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(Vec<f32>, QuantKind), String> {
        let (e, bytes) = self.bytes(name)?;
        match e.dtype {
            Dtype::BF16 | Dtype::F16 | Dtype::F32 => {
                expect_shape(name, &e.shape, &[out_dim, in_dim])?;
                Ok((decode_float(name, e.dtype, bytes)?, QuantKind::Float))
            }
            Dtype::F8_E4M3 => {
                let data = self.dequant_fp8(name, e, bytes, out_dim, in_dim)?;
                Ok((data, QuantKind::Fp8E4m3))
            }
            Dtype::I8 | Dtype::U8 => {
                let data = self.dequant_fp4(name, e, bytes, out_dim, in_dim)?;
                Ok((data, QuantKind::Fp4E2m1))
            }
            other => Err(format!(
                "tensor `{name}`: unsupported linear weight dtype {other:?}"
            )),
        }
    }

    /// Dequantize an FP8 E4M3 `[out, in]` weight with an E8M0 block-scale.
    fn dequant_fp8(
        &self,
        name: &str,
        e: &Entry,
        raw: &[u8],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>, String> {
        expect_shape(name, &e.shape, &[out_dim, in_dim])?;
        expect_numel(name, raw.len(), out_dim * in_dim)?;
        let sblk_out = out_dim.div_ceil(FP_BLOCK_SIZE);
        let sblk_in = in_dim.div_ceil(FP_BLOCK_SIZE);
        let scale_name = format!("{name}.scale");
        let (se, sraw) = self.bytes(&scale_name)?;
        expect_shape(&scale_name, &se.shape, &[sblk_out, sblk_in])?;
        let sbytes = scale_bytes(&scale_name, se.dtype, sraw)?;
        expect_numel(&scale_name, sbytes.len(), sblk_out * sblk_in)?;

        let mut out = vec![0.0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                let w = e4m3_to_f32(raw[o * in_dim + i]);
                let s = e8m0_to_f32(sbytes[(o / FP_BLOCK_SIZE) * sblk_in + i / FP_BLOCK_SIZE]);
                out[o * in_dim + i] = w * s;
            }
        }
        reject_nonfinite(name, &out)?;
        Ok(out)
    }

    /// Dequantize an FP4 E2M1 packed `[out, in/2]` weight (two nibbles per byte,
    /// low nibble first) with an E8M0 block-scale into row-major f32 `[out, in]`.
    fn dequant_fp4(
        &self,
        name: &str,
        e: &Entry,
        raw: &[u8],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>, String> {
        if in_dim % 2 != 0 {
            return Err(format!(
                "tensor `{name}`: FP4 in_dim {in_dim} must be even (two values per byte)"
            ));
        }
        let packed_in = in_dim / 2;
        expect_shape(name, &e.shape, &[out_dim, packed_in])?;
        expect_numel(name, raw.len(), out_dim * packed_in)?;
        let sblk_out = out_dim.div_ceil(FP_BLOCK_SIZE);
        let sblk_in = in_dim.div_ceil(FP_BLOCK_SIZE);
        let scale_name = format!("{name}.scale");
        let (se, sraw) = self.bytes(&scale_name)?;
        expect_shape(&scale_name, &se.shape, &[sblk_out, sblk_in])?;
        let sbytes = scale_bytes(&scale_name, se.dtype, sraw)?;
        expect_numel(&scale_name, sbytes.len(), sblk_out * sblk_in)?;

        let mut out = vec![0.0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for p in 0..packed_in {
                let byte = raw[o * packed_in + p];
                let low = FP4_TABLE[(byte & 0x0F) as usize];
                let high = FP4_TABLE[((byte >> 4) & 0x0F) as usize];
                let i0 = 2 * p;
                let i1 = 2 * p + 1;
                let s0 = e8m0_to_f32(sbytes[(o / FP_BLOCK_SIZE) * sblk_in + i0 / FP_BLOCK_SIZE]);
                let s1 = e8m0_to_f32(sbytes[(o / FP_BLOCK_SIZE) * sblk_in + i1 / FP_BLOCK_SIZE]);
                out[o * in_dim + i0] = low * s0;
                out[o * in_dim + i1] = high * s1;
            }
        }
        reject_nonfinite(name, &out)?;
        Ok(out)
    }
}
