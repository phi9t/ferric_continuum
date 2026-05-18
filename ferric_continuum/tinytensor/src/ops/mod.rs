pub mod activations;
pub mod attention;
pub mod basic;
pub mod embedding;
pub mod linear;
pub mod loss;
pub mod norm;

use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::autograd::{BackwardRecipe, OpCall, OpKind};
use crate::saved::SaveSite;
use crate::tensor::{Tensor, TensorValue};

static OP_CALL_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn fresh_op_call_id() -> crate::autograd::OpCallId {
    crate::autograd::OpCallId(OP_CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

pub fn finish_op(
    kind: OpKind,
    name: impl Into<String>,
    inputs: &[&Tensor],
    output_value: TensorValue,
    backward: Option<Box<dyn BackwardRecipe>>,
    debug_saved: Vec<SaveSite>,
) -> Tensor {
    let requires_grad = inputs
        .iter()
        .any(|t| t.inner.borrow().autograd.requires_grad);

    let out = Tensor::from_value(output_value, requires_grad);

    if requires_grad && crate::grad_mode::is_enabled() {
        let call = Rc::new(OpCall {
            id: fresh_op_call_id(),
            kind,
            name: name.into(),
            inputs: inputs.iter().map(|t| t.grad_target()).collect(),
            outputs: vec![out.inner.borrow().id],
            output_shapes: vec![out.inner.borrow().value.shape.clone()],
            backward: backward.expect("requires_grad op needs backward recipe"),
            debug_saved,
        });

        crate::debug::record_op_call_global(&call);

        out.inner.borrow_mut().autograd.producer = Some(call);
        out.inner.borrow_mut().autograd.is_leaf = false;
    }

    out
}
