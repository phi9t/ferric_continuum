use tinytensor::{
    autograd::Engine,
    ops::{activations, attention, basic, linear, norm},
    tensor::{Shape, Tensor, TensorValue},
};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// Finite-difference gradient check for a scalar-output function
fn grad_check<F>(f: F, inputs: &[Tensor], eps: f32, tol: f32)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let loss = f(inputs);
    let mut engine = Engine::new();
    engine.backward(&loss);

    for (idx, inp) in inputs.iter().enumerate() {
        let analytic_grad = match inp.grad() {
            Some(g) => g,
            None => continue,
        };

        let n = inp.inner.borrow().value.shape.numel();
        for i in 0..n.min(20) {
            // +eps
            let mut data_plus = inp.inner.borrow().value.data.as_ref().clone();
            data_plus[i] += eps;
            let plus_val = TensorValue::from_vec(inp.inner.borrow().value.shape.clone(), data_plus);

            // -eps
            let mut data_minus = inp.inner.borrow().value.data.as_ref().clone();
            data_minus[i] -= eps;
            let minus_val = TensorValue::from_vec(inp.inner.borrow().value.shape.clone(), data_minus);

            let mut inputs_plus = inputs.to_vec();
            inputs_plus[idx] = Tensor::from_value_no_grad(plus_val);
            let mut inputs_minus = inputs.to_vec();
            inputs_minus[idx] = Tensor::from_value_no_grad(minus_val);

            let f_plus = f(&inputs_plus).inner.borrow().value.data[0];
            let f_minus = f(&inputs_minus).inner.borrow().value.data[0];

            let fd_grad = (f_plus - f_minus) / (2.0 * eps);
            let an_grad = analytic_grad.data[i];

            assert!(
                approx_eq(fd_grad, an_grad, tol),
                "grad_check input[{}][{}]: fd={:.6} analytic={:.6} diff={:.6}",
                idx,
                i,
                fd_grad,
                an_grad,
                (fd_grad - an_grad).abs()
            );
        }
    }
}

#[test]
fn test_add_backward() {
    let x = Tensor::randn(&[3, 4]).requires_grad();
    let y = Tensor::randn(&[3, 4]).requires_grad();
    grad_check(
        |inputs| {
            let z = basic::add(&inputs[0], &inputs[1], "add");
            basic::sum(&z, "sum")
        },
        &[x, y],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_linear_backward() {
    let x = Tensor::randn(&[2, 3, 4]).requires_grad();
    let w = Tensor::randn(&[4, 5]).requires_grad();
    grad_check(
        |inputs| {
            let z = linear::linear(&inputs[0], &inputs[1], "lin");
            basic::sum(&z, "sum")
        },
        &[x, w],
        1e-4,
        1e-2,
    );
}

#[test]
fn test_gelu_backward() {
    let x = Tensor::randn(&[2, 5]).requires_grad();
    grad_check(
        |inputs| {
            let z = activations::gelu(&inputs[0], "gelu");
            basic::sum(&z, "sum")
        },
        &[x],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_softmax_backward() {
    // Use a weighted sum so the objective is non-trivial (sum of softmax is constant=1 per row).
    let x = Tensor::randn(&[2, 3, 5]).requires_grad();
    // Fixed weights to project softmax output to a scalar
    let weights = TensorValue::from_vec(
        Shape(vec![2, 3, 5]),
        (0..30).map(|i| (i as f32) * 0.1 - 1.5).collect(),
    );
    let w_tensor = Tensor::from_value_no_grad(weights);
    grad_check(
        move |inputs| {
            let z = attention::softmax_last_dim(&inputs[0], "sm");
            let weighted = basic::mul(&z, &w_tensor, "weight");
            basic::sum(&weighted, "sum")
        },
        &[x],
        1e-3,
        2e-3,
    );
}

#[test]
fn test_layer_norm_backward() {
    let x = Tensor::randn(&[2, 4, 8]).requires_grad();
    let gamma = Tensor::from_value(TensorValue::from_vec(Shape(vec![8]), vec![1.0; 8]), true);
    let beta = Tensor::from_value(TensorValue::zeros(Shape(vec![8])), true);
    grad_check(
        |inputs| {
            let z = norm::layer_norm(&inputs[0], &inputs[1], &inputs[2], "ln");
            basic::sum(&z, "sum")
        },
        &[x, gamma, beta],
        1e-4,
        1e-2,
    );
}

#[test]
fn test_attention_scores_backward() {
    let q = Tensor::randn(&[2, 3, 4]).requires_grad();
    let k = Tensor::randn(&[2, 3, 4]).requires_grad();
    grad_check(
        |inputs| {
            let z = attention::attention_scores(&inputs[0], &inputs[1], 0.5, "attn");
            basic::sum(&z, "sum")
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
            let z = attention::attention_mix(&inputs[0], &inputs[1], "mix");
            basic::sum(&z, "sum")
        },
        &[p, v],
        1e-4,
        1e-2,
    );
}

#[test]
fn test_transformer_forward_backward() {
    use tinytensor::transformer::{TransformerBlock, TransformerConfig};

    let cfg = TransformerConfig::tiny_4_7_29();
    let block = TransformerBlock::new(cfg);
    let x = Tensor::randn(&[4, 7, 29]).requires_grad();

    let y = block.forward(&x);
    let loss = basic::sum(&y, "loss");

    let mut engine = Engine::new();
    engine.backward(&loss);

    let stats = x.grad_stats().expect("x should have gradient");
    assert!(stats.nan_count == 0, "NaN in gradient");
    assert!(stats.inf_count == 0, "Inf in gradient");
}

#[test]
fn test_whole_block_checkpoint() {
    use std::rc::Rc;
    use tinytensor::checkpoint::{checkpoint, WholeBlockCheckpoint};
    use tinytensor::transformer::{TransformerBlock, TransformerConfig};

    let cfg = TransformerConfig::tiny_4_7_29();
    let block = Rc::new(TransformerBlock::new(cfg));
    let x = Tensor::randn(&[4, 7, 29]).requires_grad();

    let y = checkpoint("blk", Rc::new(WholeBlockCheckpoint), &[x.clone()], {
        let block = block.clone();
        move |xs| block.forward(&xs[0])
    });

    let loss = basic::sum(&y, "loss");
    let mut engine = Engine::new();
    engine.backward(&loss);

    let stats = x.grad_stats().expect("x should have gradient");
    assert!(stats.nan_count == 0, "NaN in gradient");
    assert!(stats.inf_count == 0, "Inf in gradient");
}
