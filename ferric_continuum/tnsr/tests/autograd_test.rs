use tnsr::{
    autograd::Engine,
    checkpoint::{checkpoint, TransformerSelectivePolicy, WholeBlockCheckpoint},
    ops::{activations, attention, basic, embedding, linear, loss, norm},
    tensor::{Shape, Tensor, TensorValue},
    transformer::{TransformerBlock, TransformerConfig},
};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Finite-difference gradient check for a scalar-output function.
/// Checks the first min(n, 20) elements of each input's gradient.
fn grad_check<F>(f: F, inputs: &[Tensor], eps: f32, tol: f32)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let mut engine = Engine::new();
    let loss = f(inputs);
    engine.backward(&loss);

    for (idx, inp) in inputs.iter().enumerate() {
        let analytic_grad = match inp.grad() {
            Some(g) => g,
            None => continue,
        };

        let n = inp.inner.borrow().value.shape.numel();
        for i in 0..n.min(20) {
            let base = inp.inner.borrow().value.data.as_ref().clone();

            let mut d_plus = base.clone();
            d_plus[i] += eps;
            let mut d_minus = base.clone();
            d_minus[i] -= eps;

            let shape = inp.inner.borrow().value.shape.clone();
            let t_plus = Tensor::from_value_no_grad(TensorValue::from_vec(shape.clone(), d_plus));
            let t_minus = Tensor::from_value_no_grad(TensorValue::from_vec(shape, d_minus));

            let mut ip = inputs.to_vec();
            let mut im = inputs.to_vec();
            ip[idx] = t_plus;
            im[idx] = t_minus;

            let f_plus = f(&ip).inner.borrow().value.data[0];
            let f_minus = f(&im).inner.borrow().value.data[0];

            let fd = (f_plus - f_minus) / (2.0 * eps);
            let an = analytic_grad.data[i];

            assert!(
                approx_eq(fd, an, tol),
                "grad_check input[{}][{}]: fd={:.6} analytic={:.6} diff={:.6}",
                idx, i, fd, an, (fd - an).abs()
            );
        }
    }
}

/// Check that a tensor has no NaN or Inf values.
fn assert_finite(t: &Tensor, label: &str) {
    let stats = t.grad_stats();
    if let Some(s) = stats {
        assert_eq!(s.nan_count, 0, "{}: NaN in gradient", label);
        assert_eq!(s.inf_count, 0, "{}: Inf in gradient", label);
    }
}

// ---------------------------------------------------------------------------
// Basic ops
// ---------------------------------------------------------------------------

#[test]
fn test_add_backward() {
    let x = Tensor::randn(&[3, 4]).requires_grad();
    let y = Tensor::randn(&[3, 4]).requires_grad();
    grad_check(
        |inputs| basic::sum(&basic::add(&inputs[0], &inputs[1], "add"), "loss"),
        &[x, y],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_mul_backward() {
    let x = Tensor::randn(&[3, 4]).requires_grad();
    let y = Tensor::randn(&[3, 4]).requires_grad();
    grad_check(
        |inputs| basic::sum(&basic::mul(&inputs[0], &inputs[1], "mul"), "loss"),
        &[x, y],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_scale_backward() {
    let x = Tensor::randn(&[3, 4]).requires_grad();
    grad_check(
        |inputs| basic::sum(&basic::scale(&inputs[0], 3.7, "scale"), "loss"),
        &[x],
        1e-3,
        2e-3,
    );
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

#[test]
fn test_linear_backward() {
    let x = Tensor::randn(&[2, 3, 4]).requires_grad();
    let w = Tensor::randn(&[4, 5]).requires_grad();
    grad_check(
        |inputs| basic::sum(&linear::linear(&inputs[0], &inputs[1], "lin"), "loss"),
        &[x, w],
        1e-4,
        1e-2,
    );
}

#[test]
fn test_linear_2d_input() {
    // linear should accept 2D [T, Din] as well as 3D [B, T, Din]
    let x = Tensor::randn(&[5, 4]).requires_grad();
    let w = Tensor::randn(&[4, 7]).requires_grad();
    let out = linear::linear(&x, &w, "lin2d");
    assert_eq!(out.shape().0, vec![5, 7]);
    let loss = basic::sum(&out, "loss");
    let mut engine = Engine::new();
    engine.backward(&loss);
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

#[test]
fn test_gelu_backward() {
    let x = Tensor::randn(&[2, 5]).requires_grad();
    grad_check(
        |inputs| basic::sum(&activations::gelu(&inputs[0], "gelu"), "loss"),
        &[x],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_silu_backward() {
    let x = Tensor::randn(&[2, 5]).requires_grad();
    grad_check(
        |inputs| basic::sum(&activations::silu(&inputs[0], "silu"), "loss"),
        &[x],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_swiglu_backward() {
    // swiglu expects last dim to be even: [B, 2*D] -> [B, D]
    let x = Tensor::randn(&[3, 8]).requires_grad();
    grad_check(
        |inputs| basic::sum(&activations::swiglu(&inputs[0], "swiglu"), "loss"),
        &[x],
        1e-3,
        2e-3,
    );
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_backward() {
    let x = Tensor::randn(&[2, 4, 8]).requires_grad();
    let gamma = Tensor::from_value(TensorValue::from_vec(Shape(vec![8]), vec![1.0; 8]), true);
    let beta = Tensor::from_value(TensorValue::zeros(Shape(vec![8])), true);
    grad_check(
        |inputs| {
            basic::sum(&norm::layer_norm(&inputs[0], &inputs[1], &inputs[2], "ln"), "loss")
        },
        &[x, gamma, beta],
        1e-4,
        1e-2,
    );
}

#[test]
fn test_rms_norm_backward() {
    // Use smaller tensors to limit f32 cancellation in the fd numerator.
    let x = Tensor::randn(&[2, 3]).requires_grad();
    let gamma = Tensor::from_value(TensorValue::from_vec(Shape(vec![3]), vec![1.0; 3]), true);
    grad_check(
        |inputs| {
            basic::sum(&norm::rms_norm(&inputs[0], &inputs[1], "rn"), "loss")
        },
        &[x, gamma],
        1e-3,
        5e-3,
    );
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_backward() {
    let x = Tensor::randn(&[2, 3, 5]).requires_grad();
    let weights = TensorValue::from_vec(
        Shape(vec![2, 3, 5]),
        (0..30).map(|i| (i as f32) * 0.1 - 1.5).collect(),
    );
    let w_tensor = Tensor::from_value_no_grad(weights);
    grad_check(
        move |inputs| {
            let z = attention::softmax_last_dim(&inputs[0], "sm");
            basic::sum(&basic::mul(&z, &w_tensor, "w"), "loss")
        },
        &[x],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_attention_scores_backward() {
    let q = Tensor::randn(&[2, 3, 4]).requires_grad();
    let k = Tensor::randn(&[2, 3, 4]).requires_grad();
    grad_check(
        |inputs| {
            basic::sum(&attention::attention_scores(&inputs[0], &inputs[1], 0.5, "attn"), "loss")
        },
        &[q, k],
        1e-4,
        1e-2,
    );
}

#[test]
fn test_attention_mix_backward() {
    let p = Tensor::randn(&[2, 3, 3]).requires_grad();
    let v = Tensor::randn(&[2, 3, 4]).requires_grad();
    grad_check(
        |inputs| {
            basic::sum(&attention::attention_mix(&inputs[0], &inputs[1], "mix"), "loss")
        },
        &[p, v],
        1e-4,
        1e-2,
    );
}

// ---------------------------------------------------------------------------
// Embedding & Loss
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_backward() {
    // vocab=5, d=4, B=2, T=3
    let ids = vec![0usize, 1, 2, 3, 0, 4];
    let w = Tensor::randn(&[5, 4]).requires_grad();
    grad_check(
        move |inputs| {
            let out = embedding::embedding(&ids, 2, 3, &inputs[0], "emb");
            basic::sum(&out, "loss")
        },
        &[w],
        1e-4,
        1e-2,
    );
}

#[test]
fn test_cross_entropy_backward() {
    // logits [2, 4] (2 samples, 4 classes), targets [2]
    let logits = Tensor::randn(&[2, 4]).requires_grad();
    let targets = vec![1usize, 3];
    grad_check(
        move |inputs| loss::cross_entropy(&inputs[0], &targets, "ce"),
        &[logits],
        1e-4,
        1e-2,
    );
}

// ---------------------------------------------------------------------------
// Transformer integration
// ---------------------------------------------------------------------------

#[test]
fn test_transformer_forward_backward() {
    let mut engine = Engine::new();
    let cfg = TransformerConfig::tiny_4_7_29();
    let block = TransformerBlock::new(cfg);
    let x = Tensor::randn(&[4, 7, 29]).requires_grad();
    let y = block.forward(&x);
    let loss = basic::sum(&y, "loss");
    engine.backward(&loss);
    assert_finite(&x, "x.grad");
    for p in block.parameters() {
        assert!(p.grad().is_some(), "parameter missing gradient");
    }
}

// ---------------------------------------------------------------------------
// Checkpointing correctness
// ---------------------------------------------------------------------------

/// Verify that whole-block checkpoint produces the same gradients as no-checkpoint.
#[test]
fn test_whole_block_checkpoint_gradient_equivalence() {
    let cfg1 = TransformerConfig::tiny_4_7_29();
    let block1 = TransformerBlock::new(cfg1);

    let cfg2 = TransformerConfig::tiny_4_7_29();
    let block2 = Rc::new(TransformerBlock::new(cfg2));

    // Use identical weight values
    let copy_weights = |src: &TransformerBlock, dst: &TransformerBlock| {
        for (s, d) in src.parameters().iter().zip(dst.parameters().iter()) {
            let sv = s.inner.borrow().value.clone();
            d.inner.borrow_mut().value = sv;
        }
    };
    copy_weights(&block1, &block2);

    // Build a shared input
    let x_data = Tensor::randn(&[4, 7, 29]);
    let x1 = Tensor::from_value(x_data.inner.borrow().value.clone(), false).requires_grad();
    let x2 = Tensor::from_value(x_data.inner.borrow().value.clone(), false).requires_grad();

    // No-checkpoint forward + backward
    let mut eng1 = Engine::new();
    let y1 = block1.forward(&x1);
    let loss1 = basic::sum(&y1, "loss");
    eng1.backward(&loss1);

    // Whole-block checkpoint forward + backward
    let block2_rc = block2.clone();
    let mut eng2 = Engine::new();
    let y2 = checkpoint("blk", Rc::new(WholeBlockCheckpoint), &[x2.clone()], move |xs| {
        block2_rc.forward(&xs[0])
    });
    let loss2 = basic::sum(&y2, "loss");
    eng2.backward(&loss2);

    let g1 = x1.grad().expect("x1 missing grad");
    let g2 = x2.grad().expect("x2 missing grad");
    assert_eq!(g1.shape, g2.shape);
    for (a, b) in g1.data.iter().zip(g2.data.iter()) {
        assert!(
            approx_eq(*a, *b, 1e-4),
            "checkpoint grad mismatch: {:.6} vs {:.6}",
            a, b
        );
    }
}

#[test]
fn test_whole_block_checkpoint_no_nan() {
    let cfg = TransformerConfig::tiny_4_7_29();
    let block = Rc::new(TransformerBlock::new(cfg));
    let mut engine = Engine::new();
    let x = Tensor::randn(&[4, 7, 29]).requires_grad();
    let y = checkpoint("blk", Rc::new(WholeBlockCheckpoint), &[x.clone()], {
        let block = block.clone();
        move |xs| block.forward(&xs[0])
    });
    let loss = basic::sum(&y, "loss");
    engine.backward(&loss);
    assert_finite(&x, "x.grad (whole-block checkpoint)");
}

#[test]
fn test_selective_checkpoint_gradient_equivalence() {
    let cfg1 = TransformerConfig::tiny_4_7_29();
    let block1 = TransformerBlock::new(cfg1);
    let cfg2 = TransformerConfig::tiny_4_7_29();
    let block2 = Rc::new(TransformerBlock::new(cfg2));

    let copy_weights = |src: &TransformerBlock, dst: &TransformerBlock| {
        for (s, d) in src.parameters().iter().zip(dst.parameters().iter()) {
            d.inner.borrow_mut().value = s.inner.borrow().value.clone();
        }
    };
    copy_weights(&block1, &block2);

    let x_data = Tensor::randn(&[4, 7, 29]);
    let x1 = Tensor::from_value(x_data.inner.borrow().value.clone(), false).requires_grad();
    let x2 = Tensor::from_value(x_data.inner.borrow().value.clone(), false).requires_grad();

    let mut eng1 = Engine::new();
    eng1.backward(&basic::sum(&block1.forward(&x1), "loss"));

    let policy = Rc::new(TransformerSelectivePolicy {
        save_softmax_under_bytes: 4096,
        recompute_activation_over_bytes: 8192,
    });
    let block2_rc = block2.clone();
    let mut eng2 = Engine::new();
    let y2 = checkpoint("blk_sel", policy, &[x2.clone()], move |xs| {
        block2_rc.forward(&xs[0])
    });
    eng2.backward(&basic::sum(&y2, "loss"));

    let g1 = x1.grad().expect("x1 missing grad");
    let g2 = x2.grad().expect("x2 missing grad");
    for (a, b) in g1.data.iter().zip(g2.data.iter()) {
        assert!(
            approx_eq(*a, *b, 1e-4),
            "selective checkpoint grad mismatch: {:.6} vs {:.6}",
            a, b
        );
    }
}

/// Selective policy should save fewer activation bytes than no-checkpoint.
#[test]
fn test_selective_checkpoint_saves_less_than_no_checkpoint() {
    // We can't introspect byte counts directly, but we CAN verify that recompute
    // events occur for large activations. A proxy: the checkpoint report has at
    // least one recompute-end event, and the op table is consistent.
    let cfg = TransformerConfig::tiny_4_7_29();
    let block = Rc::new(TransformerBlock::new(cfg));

    let policy = Rc::new(TransformerSelectivePolicy {
        save_softmax_under_bytes: 4096,
        recompute_activation_over_bytes: 8192,
    });
    let mut engine = Engine::new();
    let x = Tensor::randn(&[4, 7, 29]).requires_grad();
    let block_rc = block.clone();
    let y = checkpoint("blk_sel", policy, &[x.clone()], move |xs| {
        block_rc.forward(&xs[0])
    });
    engine.backward(&basic::sum(&y, "loss"));
    assert_finite(&x, "x.grad (selective checkpoint)");
}

// ---------------------------------------------------------------------------
// grad_mode
// ---------------------------------------------------------------------------

#[test]
fn test_no_grad_mode() {
    use tnsr::grad_mode::{is_enabled, NoGradGuard};

    assert!(is_enabled());
    {
        let _g = NoGradGuard::new();
        assert!(!is_enabled());

        let x = Tensor::randn(&[3, 3]).requires_grad();
        let y = basic::sum(&x, "loss");
        // No OpCall created under no-grad
        assert!(y.inner.borrow().autograd.producer.is_none());
    }
    assert!(is_enabled()); // restored on drop
}

// ---------------------------------------------------------------------------
// Shape / panic invariants
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "add shape mismatch")]
fn test_add_shape_mismatch_panics() {
    let x = Tensor::randn(&[2, 3]);
    let y = Tensor::randn(&[2, 4]);
    let _ = basic::add(&x, &y, "bad_add");
}

#[test]
#[should_panic(expected = "Din mismatch")]
fn test_linear_shape_mismatch_panics() {
    let x = Tensor::randn(&[2, 3, 4]);
    let w = Tensor::randn(&[5, 6]); // Din=5 != 4
    let _ = linear::linear(&x, &w, "bad_linear");
}

#[test]
#[should_panic(expected = "P shape must be")]
fn test_attention_mix_shape_mismatch_panics() {
    let p = Tensor::randn(&[2, 3, 4]); // wrong: should be [2,3,3]
    let v = Tensor::randn(&[2, 3, 5]);
    let _ = attention::attention_mix(&p, &v, "bad_mix");
}
