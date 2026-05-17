use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::saved::SavedTensor;
use crate::tensor::{Tensor, TensorValue};

// ---------------------------------------------------------------------------
// Raw numeric helpers
// ---------------------------------------------------------------------------

pub fn raw_add_values(a: &TensorValue, b: &TensorValue) -> TensorValue {
    assert_eq!(
        a.shape, b.shape,
        "add shape mismatch: {:?} vs {:?}",
        a.shape, b.shape
    );
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
    TensorValue::from_vec(a.shape.clone(), data)
}

pub fn raw_mul_values(a: &TensorValue, b: &TensorValue) -> TensorValue {
    assert_eq!(a.shape, b.shape, "mul shape mismatch");
    let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect();
    TensorValue::from_vec(a.shape.clone(), data)
}

pub fn raw_scale_value(a: &TensorValue, s: f32) -> TensorValue {
    let data: Vec<f32> = a.data.iter().map(|x| x * s).collect();
    TensorValue::from_vec(a.shape.clone(), data)
}

pub fn raw_sum_value(a: &TensorValue) -> TensorValue {
    let s: f32 = a.data.iter().sum();
    TensorValue::from_vec(crate::tensor::Shape(vec![1]), vec![s])
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

struct AddBackward {
    x_target: GradTarget,
    y_target: GradTarget,
}

impl BackwardRecipe for AddBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = grad_outputs[0].clone();
        vec![
            GradEdge { target: self.x_target.clone(), grad: dy.clone() },
            GradEdge { target: self.y_target.clone(), grad: dy },
        ]
    }
}

pub fn add(x: &Tensor, y: &Tensor, name: &str) -> Tensor {
    let out_value = raw_add_values(&x.inner.borrow().value, &y.inner.borrow().value);
    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[x, y]) {
            Some(Box::new(AddBackward {
                x_target: x.grad_target(),
                y_target: y.grad_target(),
            }))
        } else {
            None
        };
    crate::ops::finish_op(OpKind::Add, name, &[x, y], out_value, recipe, vec![])
}

// ---------------------------------------------------------------------------
// mul (elementwise)
// ---------------------------------------------------------------------------

struct MulBackward {
    x: SavedTensor,
    y: SavedTensor,
    x_target: GradTarget,
    y_target: GradTarget,
}

impl BackwardRecipe for MulBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dout = &grad_outputs[0];
        let x = ctx.unpack(&self.x);
        let y = ctx.unpack(&self.y);
        vec![
            GradEdge { target: self.x_target.clone(), grad: raw_mul_values(dout, &y) },
            GradEdge { target: self.y_target.clone(), grad: raw_mul_values(dout, &x) },
        ]
    }
}

pub fn mul(x: &Tensor, y: &Tensor, name: &str) -> Tensor {
    let out_value = raw_mul_values(&x.inner.borrow().value, &y.inner.borrow().value);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x, y])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        use crate::saved::{SaveRole, SaveSite};
        let sx = SaveSite::new(OpKind::Mul, format!("{name}.lhs"), SaveRole::Activation, x);
        let sy = SaveSite::new(OpKind::Mul, format!("{name}.rhs"), SaveRole::Activation, y);
        let saved_x = SavedTensor::save(x, sx.clone());
        let saved_y = SavedTensor::save(y, sy.clone());
        let recipe: Box<dyn BackwardRecipe> = Box::new(MulBackward {
            x: saved_x,
            y: saved_y,
            x_target: x.grad_target(),
            y_target: y.grad_target(),
        });
        (Some(recipe), vec![sx, sy])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(OpKind::Mul, name, &[x, y], out_value, recipe, debug_saved)
}

// ---------------------------------------------------------------------------
// scale (multiply by scalar)
// ---------------------------------------------------------------------------

struct ScaleBackward {
    scale: f32,
    x_target: GradTarget,
}

impl BackwardRecipe for ScaleBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dx = raw_scale_value(&grad_outputs[0], self.scale);
        vec![GradEdge { target: self.x_target.clone(), grad: dx }]
    }
}

pub fn scale(x: &Tensor, s: f32, name: &str) -> Tensor {
    let out_value = raw_scale_value(&x.inner.borrow().value, s);
    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[x]) {
            Some(Box::new(ScaleBackward { scale: s, x_target: x.grad_target() }))
        } else {
            None
        };
    crate::ops::finish_op(OpKind::Scale, name, &[x], out_value, recipe, vec![])
}

// ---------------------------------------------------------------------------
// sum (all elements -> scalar)
// ---------------------------------------------------------------------------

struct SumBackward {
    x_target: GradTarget,
}

impl BackwardRecipe for SumBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let g = grad_outputs[0].data[0];
        let shape = self.x_target.shape.clone();
        let n = shape.numel();
        let dx = TensorValue::from_vec(shape, vec![g; n]);
        vec![GradEdge { target: self.x_target.clone(), grad: dx }]
    }
}

pub fn sum(x: &Tensor, name: &str) -> Tensor {
    let out_value = raw_sum_value(&x.inner.borrow().value);
    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[x]) {
            Some(Box::new(SumBackward { x_target: x.grad_target() }))
        } else {
            None
        };
    crate::ops::finish_op(OpKind::Sum, name, &[x], out_value, recipe, vec![])
}
