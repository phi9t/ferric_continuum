pub mod activations;
pub mod attention;
pub mod basic;
pub mod embedding;
pub mod gqa;
pub mod linear;
pub mod loss;
pub mod norm;
pub mod rope;
pub mod shape;

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

/// Multi-output variant of [`finish_op`]: register a single [`OpCall`] that
/// produced several output tensors from the given inputs.
///
/// This is the multi-output analogue used by ops like `split3` (one tensor in,
/// several out). The key difference from `finish_op` is that **every** output
/// tensor's `producer` points at the *same* `OpCall`. During the backward pass
/// the engine gathers one incoming gradient per output into `grad_outputs[..]`,
/// and the op's single `BackwardRecipe` folds them all into the input grads.
///
/// The topo-sort visits the shared node once (deduped by `OpCallId`), so
/// wiring several outputs to one call is correct and not double-counted.
pub fn finish_op_multi(
    kind: OpKind,
    name: impl Into<String>,
    inputs: &[&Tensor],
    output_values: Vec<TensorValue>,
    backward: Option<Box<dyn BackwardRecipe>>,
    debug_saved: Vec<SaveSite>,
) -> Vec<Tensor> {
    let requires_grad = inputs
        .iter()
        .any(|t| t.inner.borrow().autograd.requires_grad);

    let outs: Vec<Tensor> = output_values
        .into_iter()
        .map(|v| Tensor::from_value(v, requires_grad))
        .collect();

    if requires_grad && crate::grad_mode::is_enabled() {
        let call = Rc::new(OpCall {
            id: fresh_op_call_id(),
            kind,
            name: name.into(),
            inputs: inputs.iter().map(|t| t.grad_target()).collect(),
            outputs: outs.iter().map(|o| o.inner.borrow().id).collect(),
            output_shapes: outs
                .iter()
                .map(|o| o.inner.borrow().value.shape.clone())
                .collect(),
            backward: backward.expect("requires_grad op needs backward recipe"),
            debug_saved,
        });

        crate::debug::record_op_call_global(&call);

        // Point every output at the shared call, and mark them non-leaf.
        for o in &outs {
            let mut inner = o.inner.borrow_mut();
            inner.autograd.producer = Some(call.clone());
            inner.autograd.is_leaf = false;
        }
    }

    outs
}
