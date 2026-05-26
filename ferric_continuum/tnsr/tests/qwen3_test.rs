//! Gradient checks and structural tests for the Qwen3 architecture
//! (`src/qwen3.rs`) and its supporting ops (`reshape`, `rope`, `gqa`).

use tnsr::{
    autograd::Engine,
    ops::{basic, gqa, rope, shape},
    qwen3::{Qwen3Attention, Qwen3Block, Qwen3Config, Qwen3MLP, Qwen3Model},
    tensor::{Tensor, TensorValue},
};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Finite-difference grad check on the first `n.min(12)` entries of each
/// `requires_grad` input.
fn grad_check<F>(f: F, inputs: &[Tensor], eps: f32, tol: f32)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let mut engine = Engine::new();
    let loss = f(inputs);
    engine.backward(&loss);

    for (idx, inp) in inputs.iter().enumerate() {
        let analytic = match inp.grad() {
            Some(g) => g,
            None => continue,
        };
        let n = inp.inner.borrow().value.shape.numel();
        for i in 0..n.min(12) {
            let base = inp.inner.borrow().value.data.as_ref().clone();
            let shape = inp.inner.borrow().value.shape.clone();
            let mut dp = base.clone();
            dp[i] += eps;
            let mut dm = base.clone();
            dm[i] -= eps;
            let tp = Tensor::from_value_no_grad(TensorValue::from_vec(shape.clone(), dp));
            let tm = Tensor::from_value_no_grad(TensorValue::from_vec(shape, dm));
            let mut ip = inputs.to_vec();
            let mut im = inputs.to_vec();
            ip[idx] = tp;
            im[idx] = tm;
            let fp = f(&ip).inner.borrow().value.data[0];
            let fm = f(&im).inner.borrow().value.data[0];
            let fd = (fp - fm) / (2.0 * eps);
            let an = analytic.data[i];
            assert!(
                approx_eq(fd, an, tol),
                "grad mismatch input[{}][{}]: fd={:.5} an={:.5} diff={:.5}",
                idx,
                i,
                fd,
                an,
                (fd - an).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// reshape
// ---------------------------------------------------------------------------

#[test]
fn reshape_forward_preserves_buffer() {
    let x = Tensor::randn(&[2, 3, 4]);
    let y = shape::reshape(&x, &[2, 6, 2], "r");
    let xv = x.inner.borrow().value.data.as_ref().clone();
    let yv = y.inner.borrow().value.data.as_ref().clone();
    assert_eq!(xv, yv);
    assert_eq!(y.shape().0, vec![2, 6, 2]);
}

#[test]
fn reshape_backward() {
    let x = Tensor::randn(&[2, 3, 4]).requires_grad();
    grad_check(
        |inp| basic::sum(&shape::reshape(&inp[0], &[2, 12], "r"), "loss"),
        &[x],
        1e-3,
        2e-3,
    );
}

// ---------------------------------------------------------------------------
// rope
// ---------------------------------------------------------------------------

#[test]
fn rope_preserves_norm_per_pair() {
    // RoPE is a per-pair rotation, so |y[..,2i]|^2 + |y[..,2i+1]|^2 must match.
    let x = Tensor::randn(&[1, 4, 2, 6]);
    let y = rope::rope(&x, rope::RopeConfig::default(), "r");
    let xv = x.inner.borrow().value.data.as_ref().clone();
    let yv = y.inner.borrow().value.data.as_ref().clone();
    for i in 0..xv.len() / 2 {
        let nx = xv[2 * i] * xv[2 * i] + xv[2 * i + 1] * xv[2 * i + 1];
        let ny = yv[2 * i] * yv[2 * i] + yv[2 * i + 1] * yv[2 * i + 1];
        assert!(approx_eq(nx, ny, 1e-5), "pair {}: {} vs {}", i, nx, ny);
    }
}

#[test]
fn rope_position_zero_is_identity() {
    let x = Tensor::randn(&[1, 1, 2, 4]);
    let y = rope::rope(&x, rope::RopeConfig::default(), "r");
    for (a, b) in x
        .inner
        .borrow()
        .value
        .data
        .iter()
        .zip(y.inner.borrow().value.data.iter())
    {
        assert!(approx_eq(*a, *b, 1e-6));
    }
}

#[test]
fn rope_backward() {
    let x = Tensor::randn(&[1, 3, 2, 4]).requires_grad();
    grad_check(
        |inp| {
            let y = rope::rope(
                &inp[0],
                rope::RopeConfig {
                    base: 1000.0,
                    start_pos: 0,
                },
                "r",
            );
            basic::sum(&y, "loss")
        },
        &[x],
        1e-3,
        2e-3,
    );
}

// ---------------------------------------------------------------------------
// gqa
// ---------------------------------------------------------------------------

#[test]
fn gqa_forward_shape() {
    let b = 2;
    let t = 5;
    let hq = 4;
    let hk = 2;
    let dh = 4;
    let q = Tensor::randn(&[b, t, hq, dh]);
    let k = Tensor::randn(&[b, t, hk, dh]);
    let v = Tensor::randn(&[b, t, hk, dh]);
    let o = gqa::gqa_attention(&q, &k, &v, "gqa");
    assert_eq!(o.shape().0, vec![b, t, hq, dh]);
}

#[test]
fn gqa_grad_check_all_inputs() {
    let q = Tensor::randn(&[1, 3, 2, 4]).requires_grad();
    let k = Tensor::randn(&[1, 3, 1, 4]).requires_grad();
    let v = Tensor::randn(&[1, 3, 1, 4]).requires_grad();
    grad_check(
        |inp| {
            basic::sum(
                &gqa::gqa_attention(&inp[0], &inp[1], &inp[2], "gqa"),
                "loss",
            )
        },
        &[q, k, v],
        1e-3,
        3e-3,
    );
}

#[test]
fn gqa_is_causal() {
    // Future tokens (s > t) must not influence earlier outputs.  Change V at
    // position 2 and verify output position 0 is unchanged.
    let b = 1;
    let t = 3;
    let hq = 2;
    let hk = 1;
    let dh = 4;
    let q = Tensor::randn(&[b, t, hq, dh]);
    let k = Tensor::randn(&[b, t, hk, dh]);
    let v = Tensor::randn(&[b, t, hk, dh]);
    let o1 = gqa::gqa_attention(&q, &k, &v, "g1");

    let mut v_data = v.inner.borrow().value.data.as_ref().clone();
    let last_start = 2 * hk * dh;
    for i in 0..hk * dh {
        v_data[last_start + i] += 1.0;
    }
    let v2 = Tensor::from_value_no_grad(TensorValue::from_vec(v.shape(), v_data));
    let o2 = gqa::gqa_attention(&q, &k, &v2, "g2");

    let o1d = o1.inner.borrow().value.data.as_ref().clone();
    let o2d = o2.inner.borrow().value.data.as_ref().clone();
    // Position 0 row (first hq*dh elements) must match exactly.
    for i in 0..hq * dh {
        assert!(
            approx_eq(o1d[i], o2d[i], 1e-6),
            "causal leak at index {}: {} vs {}",
            i,
            o1d[i],
            o2d[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Qwen3 blocks
// ---------------------------------------------------------------------------

#[test]
fn qwen3_attention_forward_shape() {
    let cfg = Qwen3Config::tiny();
    let attn = Qwen3Attention::new(&cfg);
    let x = Tensor::randn(&[2, 5, cfg.hidden_size]);
    let y = attn.forward(&x);
    assert_eq!(y.shape().0, vec![2, 5, cfg.hidden_size]);
}

#[test]
fn qwen3_attention_param_count() {
    let cfg = Qwen3Config::qwen3_8b();
    let attn = Qwen3Attention::new(&cfg);
    let d = cfg.hidden_size;
    let q = cfg.q_total();
    let kv = cfg.kv_total();
    // wq:D*Hq*Dh + wk+wv:2*D*Hk*Dh + wo:Hq*Dh*D + q_norm+k_norm:2*Dh
    let expected = d * q + 2 * d * kv + q * d + 2 * cfg.head_dim;
    let actual: usize = attn
        .parameters()
        .iter()
        .map(|p| p.inner.borrow().value.shape.numel())
        .sum();
    assert_eq!(actual, expected);
}

#[test]
fn qwen3_mlp_forward_shape() {
    let cfg = Qwen3Config::tiny();
    let mlp = Qwen3MLP::new(&cfg);
    let x = Tensor::randn(&[2, 5, cfg.hidden_size]);
    let y = mlp.forward(&x);
    assert_eq!(y.shape().0, vec![2, 5, cfg.hidden_size]);
}

#[test]
fn qwen3_mlp_no_bias_params() {
    let cfg = Qwen3Config::tiny();
    let mlp = Qwen3MLP::new(&cfg);
    let d = cfg.hidden_size;
    let f = cfg.intermediate_size;
    let expected = d * f * 2 + f * d; // gate + up + down, no biases
    let actual: usize = mlp
        .parameters()
        .iter()
        .map(|p| p.inner.borrow().value.shape.numel())
        .sum();
    assert_eq!(actual, expected);
}

#[test]
fn qwen3_block_forward_shape() {
    let cfg = Qwen3Config::tiny();
    let block = Qwen3Block::new(&cfg);
    let x = Tensor::randn(&[1, 4, cfg.hidden_size]);
    let y = block.forward(&x);
    assert_eq!(y.shape().0, vec![1, 4, cfg.hidden_size]);
}

#[test]
fn qwen3_block_backward_runs() {
    // Smoke test: full Qwen3 dense block backward must complete and produce
    // finite gradients for every parameter.
    let cfg = Qwen3Config::tiny();
    let block = Qwen3Block::new(&cfg);
    let x = Tensor::randn(&[1, 3, cfg.hidden_size]).requires_grad();

    let mut engine = Engine::new();
    let y = block.forward(&x);
    let loss = basic::sum(&y, "loss");
    engine.backward(&loss);

    for (i, p) in block.parameters().iter().enumerate() {
        let g = p
            .grad()
            .unwrap_or_else(|| panic!("param {} has no grad", i));
        for &v in g.data.iter() {
            assert!(v.is_finite(), "non-finite grad in param {}: {}", i, v);
        }
    }
    let xg = x.grad().expect("x has no grad");
    for &v in xg.data.iter() {
        assert!(v.is_finite());
    }
}

#[test]
fn qwen3_model_forward_logits_shape() {
    let cfg = Qwen3Config::tiny();
    let v = cfg.vocab_size;
    let model = Qwen3Model::new(cfg);
    let ids = vec![1usize, 2, 3, 4];
    let logits = model.forward(&ids, 1, 4);
    assert_eq!(logits.shape().0, vec![1, 4, v]);
}

#[test]
fn qwen3_attention_dropping_bias_matches_report() {
    // Standard Qwen3 has attention_bias=false; confirm Qwen3Attention parameter
    // list contains no bias tensors (only the 4 projection matrices + 2 norms).
    let cfg = Qwen3Config::qwen3_8b();
    assert!(!cfg.attention_bias);
    let attn = Qwen3Attention::new(&cfg);
    let n_params = attn.parameters().len();
    assert_eq!(
        n_params, 6,
        "Qwen3 attention should expose exactly 4 projections + 2 norms"
    );
}
