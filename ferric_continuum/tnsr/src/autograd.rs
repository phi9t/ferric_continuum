use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use crate::debug::DebugRecorder;
use crate::saved::SavedTensor;
use crate::tensor::{Shape, Tensor, TensorId, TensorInner, TensorValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Add,
    Mul,
    Scale,
    Sum,
    Embedding,
    Linear,
    LayerNorm,
    RmsNorm,
    Gelu,
    Silu,
    SwiGlu,
    AttentionScores,
    Softmax,
    AttentionMix,
    Dropout,
    CrossEntropy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpCallId(pub usize);

pub type OpCallRef = Rc<OpCall>;

pub struct OpCall {
    pub id: OpCallId,
    pub kind: OpKind,
    pub name: String,
    pub inputs: Vec<GradTarget>,
    pub outputs: Vec<TensorId>,
    pub output_shapes: Vec<Shape>,
    pub backward: Box<dyn BackwardRecipe>,
    pub debug_saved: Vec<crate::saved::SaveSite>,
}

#[derive(Clone)]
pub struct GradTarget {
    pub id: TensorId,
    pub shape: Shape,
    pub requires_grad: bool,
    pub producer: Option<OpCallRef>,
    pub leaf: Option<Weak<RefCell<TensorInner>>>,
}

pub struct GradEdge {
    pub target: GradTarget,
    pub grad: TensorValue,
}

pub trait BackwardRecipe {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge>;
}

pub struct BackwardCtx {
    pub debug: DebugRecorder,
}

impl BackwardCtx {
    pub fn unpack(&mut self, saved: &SavedTensor) -> TensorValue {
        saved.unpack(&self.debug)
    }
}

pub struct Engine {
    pub grads: HashMap<TensorId, TensorValue>,
    pub debug: DebugRecorder,
    pub topo: Vec<OpCallRef>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let debug = DebugRecorder::new();
        crate::debug::set_global_recorder(debug.clone());
        Self {
            grads: HashMap::new(),
            debug,
            topo: vec![],
        }
    }

    pub fn backward(&mut self, loss: &Tensor) {
        {
            let loss_inner = loss.inner.borrow();
            assert_eq!(
                loss_inner.value.shape.numel(),
                1,
                "backward requires scalar loss"
            );
            let loss_id = loss_inner.id;
            self.grads.insert(
                loss_id,
                TensorValue::from_vec(loss_inner.value.shape.clone(), vec![1.0]),
            );
        }

        let topo = self.collect_topo(loss);
        self.topo = topo.clone();

        for call in topo.into_iter().rev() {
            let grad_outputs: Vec<TensorValue> = call
                .outputs
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    self.grads
                        .remove(id)
                        .unwrap_or_else(|| TensorValue::zeros(call.output_shapes[i].clone()))
                })
                .collect();

            self.debug.record_backward_call(&call);

            let mut ctx = BackwardCtx {
                debug: self.debug.clone(),
            };

            let edges = call.backward.backward(&grad_outputs, &mut ctx);

            for edge in edges {
                if !edge.target.requires_grad {
                    continue;
                }
                self.accumulate(edge.target, edge.grad);
            }
        }
    }

    fn accumulate(&mut self, target: GradTarget, grad: TensorValue) {
        self.debug.record_grad_accum(&target, &grad);

        if let Some(existing) = self.grads.get_mut(&target.id) {
            let summed = crate::ops::basic::raw_add_values(existing, &grad);
            *existing = summed;
        } else {
            self.grads.insert(target.id, grad.clone());
        }

        if let Some(leaf_weak) = &target.leaf {
            if let Some(leaf) = leaf_weak.upgrade() {
                let mut leaf = leaf.borrow_mut();
                let current = leaf.autograd.grad.take();
                leaf.autograd.grad = Some(match current {
                    None => grad,
                    Some(old) => crate::ops::basic::raw_add_values(&old, &grad),
                });
            }
        }
    }

    fn collect_topo(&self, loss: &Tensor) -> Vec<OpCallRef> {
        let mut out = vec![];
        let mut seen = HashSet::new();
        let producer = loss.inner.borrow().autograd.producer.clone();
        if let Some(call) = producer {
            Self::dfs_call(&call, &mut seen, &mut out);
        }
        out
    }

    fn dfs_call(call: &OpCallRef, seen: &mut HashSet<OpCallId>, out: &mut Vec<OpCallRef>) {
        if !seen.insert(call.id) {
            return;
        }
        for input in &call.inputs {
            if let Some(parent) = &input.producer {
                Self::dfs_call(parent, seen, out);
            }
        }
        out.push(call.clone());
    }

    pub fn print_op_table(&self) {
        self.debug.print_op_table(&self.topo);
    }

    pub fn print_saved_tensor_table(&self) {
        self.debug.print_saved_tensor_table();
    }

    pub fn print_checkpoint_report(&self) {
        self.debug.print_checkpoint_report();
    }

    pub fn write_dot(&self, path: &str) {
        self.debug.write_dot(&self.topo, path);
    }
}
