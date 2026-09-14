//! CUDA-vs-CPU forward-agreement test for the DeepSeek V4.1 text model
//! (GPU-tagged; requires `--config=cuda`).
//!
//!     bazel test --config=cuda //ferric_continuum/tnsr:deepseek_v41_cuda_forward_tests
//!
//! Wave 4 seam: the tnsr matmul/softmax ops dispatch to the CUDA FFI when the
//! crate is built with feature `cuda` (unless `FERRIC_TNSR_DEVICE=cpu` forces
//! the host path).  This test builds one tiny DeepSeek V4.1 block + LM head,
//! runs the *same* `forward_token_ids` twice — once forcing CPU, once on the
//! default (GPU) path — and asserts the two logits rows agree to a tight
//! tolerance.  It proves the DeepSeek forward composes correctly on top of the
//! GPU gemm/softmax kernels, not just the standalone kernels in
//! `cuda_forward_test.rs`.
//!
//! Native FP8/FP4 matmul kernels and tensor-parallel serving parity are
//! deliberately out of Wave-4 scope (deferred; see the Wave 4 tickets).  This
//! test uses the f32 CPU-dequantized weights the loader already produces.

use tnsr::deepseek_v41::attention::DeepSeekV41Attention;
use tnsr::deepseek_v41::model::{DeepSeekV41Block, DeepSeekV41TextModel};
use tnsr::deepseek_v41::moe::{DeepSeekV41Expert, DeepSeekV41Gate, DeepSeekV41MoE};
use tnsr::tensor::{Shape, Tensor, TensorValue};

// Tiny geometry: dim=4, n_heads=2, head_dim=2, q_lora=3, o_lora=2, o_groups=2,
// inter=3, experts=2, topk=1, hc_mult=2, vocab=5.
const DIM: usize = 4;
const N_HEADS: usize = 2;
const HEAD_DIM: usize = 2;
const Q_LORA: usize = 3;
const O_LORA: usize = 2;
const O_GROUPS: usize = 2;
const INTER: usize = 3;
const EXPERTS: usize = 2;
const HC_MULT: usize = 2;
const VOCAB: usize = 5;

fn ramp(n: usize, modulus: usize, sub: f32, div: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % modulus) as f32 - sub) / div).collect()
}

fn param(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_value(TensorValue::from_vec(Shape(shape.to_vec()), data), true)
}

fn tiny_expert(seed: usize) -> DeepSeekV41Expert {
    DeepSeekV41Expert {
        w1: ramp(DIM * INTER, 5, 2.0, 7.0 + seed as f32),
        w2: ramp(INTER * DIM, 7, 3.0, 6.0 + seed as f32),
        w3: ramp(DIM * INTER, 11, 5.0, 8.0 + seed as f32),
        dim: DIM,
        inter_dim: INTER,
        swiglu_limit: 10.0,
    }
}

fn tiny_block() -> DeepSeekV41Block {
    let group_in = (N_HEADS / O_GROUPS) * HEAD_DIM;
    let mix_hc = (2 + HC_MULT) * HC_MULT;
    DeepSeekV41Block {
        layer_id: 0,
        dim: DIM,
        hc_mult: HC_MULT,
        hc_sinkhorn_iters: 3,
        hc_eps: 1e-6,
        attn_norm: param(&[DIM], vec![1.0, 0.875, 1.25, 0.75]),
        ffn_norm: param(&[DIM], vec![0.8, 1.1, 0.9, 1.2]),
        attn: DeepSeekV41Attention {
            n_heads: N_HEADS,
            head_dim: HEAD_DIM,
            rope_head_dim: 1,
            q_lora_rank: Q_LORA,
            o_lora_rank: O_LORA,
            o_groups: O_GROUPS,
            compress_ratio: 0,
            window_size: 4,
            rms_norm_eps: 1e-6,
            wq_a: param(&[DIM, Q_LORA], ramp(DIM * Q_LORA, 5, 2.0, 7.0)),
            q_norm: param(&[Q_LORA], vec![1.0, 0.75, 1.25]),
            wq_b: param(
                &[Q_LORA, N_HEADS * HEAD_DIM],
                ramp(Q_LORA * N_HEADS * HEAD_DIM, 11, 5.0, 6.0),
            ),
            wkv: param(&[DIM, HEAD_DIM], ramp(DIM * HEAD_DIM, 13, 6.0, 8.0)),
            kv_norm: param(&[HEAD_DIM], vec![1.0, 1.5]),
            wo_a: param(
                &[O_GROUPS, group_in, O_LORA],
                ramp(O_GROUPS * group_in * O_LORA, 9, 4.0, 6.0),
            ),
            wo_b: param(
                &[O_GROUPS * O_LORA, DIM],
                ramp(O_GROUPS * O_LORA * DIM, 7, 3.0, 5.0),
            ),
            attn_sink: param(&[N_HEADS], vec![0.25, -0.1]),
            compressor: None,
            indexer: None,
        },
        ffn: DeepSeekV41MoE {
            gate: DeepSeekV41Gate {
                weight: ramp(EXPERTS * DIM, 7, 3.0, 5.0),
                correction_bias: vec![0.0; EXPERTS],
                bias_vl: None,
                tokens: 0,
                dim: DIM,
                experts: EXPERTS,
                topk: 1,
                gate_temp: 1.2,
                norm_topk_prob: true,
                route_scale: 1.5,
            },
            experts: (0..EXPERTS).map(tiny_expert).collect(),
            shared_experts: tiny_expert(EXPERTS),
        },
        hc_attn_fn: ramp(mix_hc * HC_MULT * DIM, 9, 4.0, 6.0),
        hc_attn_base: ramp(mix_hc, 5, 2.0, 4.0),
        hc_attn_scale: vec![0.5, 0.25, 0.75],
        hc_ffn_fn: ramp(mix_hc * HC_MULT * DIM, 11, 5.0, 7.0),
        hc_ffn_base: ramp(mix_hc, 7, 3.0, 5.0),
        hc_ffn_scale: vec![0.75, 0.5, 0.25],
        engram: None,
        engram_key: None,
        engram_value: None,
    }
}

fn tiny_model() -> DeepSeekV41TextModel {
    DeepSeekV41TextModel {
        vocab_size: VOCAB,
        hidden_size: DIM,
        hc_mult: HC_MULT,
        image_token_id: 4,
        embed_tokens: param(&[VOCAB, DIM], ramp(VOCAB * DIM, 11, 5.0, 7.0)),
        layers: vec![tiny_block()],
        final_norm: param(&[DIM], vec![1.0, 0.875, 1.125, 0.75]),
        lm_head: param(&[DIM, VOCAB], ramp(DIM * VOCAB, 13, 6.0, 8.0)),
        vision: None,
        image_start: None,
        image_end: None,
        image_newline: None,
    }
}

fn forward_logits(force_cpu: bool) -> Vec<f32> {
    if force_cpu {
        std::env::set_var("FERRIC_TNSR_DEVICE", "cpu");
    } else {
        std::env::remove_var("FERRIC_TNSR_DEVICE");
    }
    let mut model = tiny_model();
    // Seed the MoE gate token count for the [1,2] workload.
    for layer in &mut model.layers {
        layer.ffn.gate.tokens = 2;
    }
    let out = model.forward_token_ids(&[1, 3], 1, 2);
    let logits = out.inner.borrow().value.data.as_ref().to_vec();
    std::env::remove_var("FERRIC_TNSR_DEVICE");
    logits
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn deepseek_v41_forward_cuda_matches_cpu() {
    assert!(
        tnsr::cuda_ffi::use_cuda(),
        "deepseek_v41_cuda_forward_tests require --config=cuda (crate feature `cuda`)"
    );
    // NOTE: this test mutates FERRIC_TNSR_DEVICE, so it must run single-threaded
    // relative to other device-sensitive tests; keep it isolated in its own
    // crate (this file) so the default per-crate libtest run is safe.
    let cpu = forward_logits(true);
    let gpu = forward_logits(false);
    assert_eq!(cpu.len(), VOCAB * 2, "expected [1,2,vocab] logits");
    let err = max_abs_diff(&cpu, &gpu);
    assert!(
        err < 1e-4,
        "DeepSeek V4.1 forward CUDA/CPU mismatch: max abs err {err}\ncpu={cpu:?}\ngpu={gpu:?}"
    );
}
