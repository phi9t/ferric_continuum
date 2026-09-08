//! Autograd engine: DAG construction and reverse-mode backward pass.
//!
//! Book reference: Ch.4 "Backward Pass FLOPs",
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! # What this module does
//!
//! This is a *define-by-run* (a.k.a. eager / dynamic) reverse-mode autodiff
//! engine, in the same family as PyTorch's autograd. There are two phases:
//!
//! 1. **Forward pass — build the graph.** As you call ops (`add`, `mul`,
//!    `linear`, ...) each one records an [`OpCall`] node describing how to
//!    turn output gradients back into input gradients. The graph is stored
//!    *implicitly*: every output tensor keeps a `producer` back-pointer to the
//!    `OpCall` that created it (see `TensorInner::autograd.producer` in
//!    `tensor.rs`). Following `producer` links backwards reconstructs the DAG.
//!
//! 2. **Backward pass — walk the graph in reverse.** [`Engine::backward`]
//!    starts from a scalar loss, topologically sorts the DAG, then visits nodes
//!    from output to input. Each node's [`BackwardRecipe`] applies the local
//!    chain rule, and gradients are *accumulated* (summed) at each tensor that
//!    feeds more than one consumer.
//!
//! # The chain rule, concretely
//!
//! For an op `y = f(x)`, backward receives `∂L/∂y` (the gradient flowing in
//! from downstream) and must return `∂L/∂x = (∂y/∂x)ᵀ · ∂L/∂y`. That local
//! `∂y/∂x` is exactly what each `impl BackwardRecipe` encodes. Summing over all
//! paths that reach a tensor is the multivariate chain rule, handled by the
//! engine's private gradient accumulator.
//!
//! # Mental model of the types
//!
//! - [`OpCall`]      — one node in the DAG (one forward op that ran).
//! - [`GradTarget`]  — a reference to an *input* of an op: where to send its
//!   gradient, and how to recurse further back.
//! - [`GradEdge`]    — a computed gradient paired with the target it belongs to.
//! - [`BackwardRecipe`] — the per-op rule turning output grads into input grads.
//! - [`Engine`]      — owns the accumulation map and drives the reverse walk.
//!
//! # Cost intuition (why the book's 6·N·T rule holds)
//!
//! The backward pass costs ≈ 2× the forward pass for matmuls (computing both
//! ∂L/∂X and ∂L/∂W each cost one matmul), so total training FLOPs ≈ 3× forward,
//! giving the book's `6·N·T` rule (2 fwd + 4 bwd).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use crate::debug::DebugRecorder;
use crate::saved::SavedTensor;
use crate::tensor::{Shape, Tensor, TensorId, TensorInner, TensorValue};

/// Discriminant for every differentiable operation in the graph.
/// Used by `DebugRecorder`, `SaveSite`, and `ScaleReport` for FLOPs accounting.
/// See `src/scaling/op_cost.rs` for per-kind FLOPs formulas.
///
/// Book reference: Ch.4 "All the Transformer Math You Need to Know",
/// https://jax-ml.github.io/scaling-book/transformers/
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
    CausalMask,
    Softmax,
    AttentionMix,
    Dropout,
    CrossEntropy,
    Reshape,
    Split,
    Rope,
    GqaAttention,
    ContextParallelGqaAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpCallId(pub usize);

pub type OpCallRef = Rc<OpCall>;

/// Tensor identity normalized within one [`GraphObservation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphTensorId(pub usize);

/// Index of an operation in producer-before-consumer observation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphOpIndex(pub usize);

/// One normalized tensor edge together with its observed runtime shape.
///
/// Keeping identity and shape in one value prevents graph consumers from
/// accidentally pairing entries from separate parallel arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTensor {
    /// Tensor identity normalized within this observation.
    pub id: GraphTensorId,
    /// Runtime shape captured when the graph was observed.
    pub shape: Shape,
}

/// Stable, recipe-free view of one differentiable operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphOp {
    /// Mathematical operation category.
    pub kind: OpKind,
    /// Caller-supplied operation name.
    pub name: String,
    /// Declared input edges in equation order.
    pub inputs: Vec<GraphTensor>,
    /// Produced output edges in output-index order.
    pub outputs: Vec<GraphTensor>,
}

/// One caller-selected or recorder-inferred graph output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRoot {
    /// Normalized identity of the selected output tensor.
    pub tensor: GraphTensorId,
    /// Runtime shape captured when the root was observed.
    pub shape: Shape,
}

/// A live tensor does not belong to this observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphObservationError {
    /// The queried live tensor was not reachable from the observation roots.
    UnobservedTensor,
}

impl std::fmt::Display for GraphObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnobservedTensor => write!(f, "tensor is not part of this graph observation"),
        }
    }
}

impl std::error::Error for GraphObservationError {}

/// Canonical, recipe-free observation of a differentiable tensor graph.
///
/// Tensor identities are normalized per observation, allowing independently
/// built but structurally equal graphs to compare equal.
#[derive(Clone)]
pub struct GraphObservation {
    operations: Vec<GraphOp>,
    roots: Vec<GraphRoot>,
    raw_tensors: HashMap<TensorId, GraphTensorId>,
    raw_producers: HashMap<TensorId, GraphOpIndex>,
}

impl std::fmt::Debug for GraphObservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphObservation")
            .field("operations", &self.operations)
            .field("roots", &self.roots)
            .finish()
    }
}

impl PartialEq for GraphObservation {
    fn eq(&self, other: &Self) -> bool {
        self.operations == other.operations && self.roots == other.roots
    }
}

impl Eq for GraphObservation {}

struct RawGraphOp {
    kind: OpKind,
    name: String,
    inputs: Vec<RawGraphTensor>,
    outputs: Vec<RawGraphTensor>,
}

struct RawGraphTensor {
    id: TensorId,
    shape: Shape,
}

struct RawGraphRoot {
    tensor: TensorId,
    shape: Shape,
}

impl GraphObservation {
    /// Observe the differentiable DAG reachable from `outputs`.
    ///
    /// Roots are visited in caller order. Each operation's input producers are
    /// visited in declared input order with post-order DFS, yielding a stable
    /// producer-before-consumer sequence.
    pub fn from_outputs(outputs: &[&Tensor]) -> Self {
        let mut calls = Vec::new();
        let mut seen = HashSet::new();
        for output in outputs {
            if let Some(producer) = output.inner.borrow().autograd.producer.clone() {
                Self::observe_call(&producer, &mut seen, &mut calls);
            }
        }

        let raw_operations = calls
            .into_iter()
            .map(|call| {
                assert_eq!(
                    call.outputs.len(),
                    call.output_shapes.len(),
                    "graph observation: operation output identities and shapes must have equal counts"
                );
                RawGraphOp {
                    kind: call.kind,
                    name: call.name.clone(),
                    inputs: call
                    .inputs
                    .iter()
                    .map(|input| RawGraphTensor {
                        id: input.id,
                        shape: input.shape.clone(),
                    })
                    .collect(),
                    outputs: call
                        .outputs
                        .iter()
                        .copied()
                        .zip(call.output_shapes.iter().cloned())
                        .map(|(id, shape)| RawGraphTensor { id, shape })
                        .collect(),
                }
            })
            .collect();
        let raw_roots = outputs
            .iter()
            .map(|output| RawGraphRoot {
                tensor: output.id(),
                shape: output.shape(),
            })
            .collect();
        Self::normalize(raw_operations, raw_roots)
    }

    /// Return operations in stable producer-before-consumer order.
    pub fn operations(&self) -> &[GraphOp] {
        &self.operations
    }

    /// Resolve an observation-local operation index.
    pub fn operation(&self, index: GraphOpIndex) -> Option<&GraphOp> {
        self.operations.get(index.0)
    }

    /// Return selected or inferred graph roots in caller or recorder order.
    pub fn roots(&self) -> &[GraphRoot] {
        &self.roots
    }

    /// Whether this observation contains no differentiable operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Locate a live tensor's producer within this observation.
    pub fn producer_of(
        &self,
        tensor: &Tensor,
    ) -> Result<Option<GraphOpIndex>, GraphObservationError> {
        let raw_id = tensor.id();
        if !self.raw_tensors.contains_key(&raw_id) {
            return Err(GraphObservationError::UnobservedTensor);
        }
        Ok(self.raw_producers.get(&raw_id).copied())
    }

    pub(crate) fn from_recorded_ops(records: &[crate::debug::OpCallRecord]) -> Self {
        let consumed: HashSet<TensorId> = records
            .iter()
            .flat_map(|record| record.input_tensor_ids.iter().copied())
            .collect();
        let raw_roots = records
            .iter()
            .flat_map(|record| {
                record
                    .output_tensor_ids
                    .iter()
                    .copied()
                    .zip(record.output_shapes.iter().cloned())
            })
            .filter(|(tensor, _)| !consumed.contains(tensor))
            .map(|(tensor, shape)| RawGraphRoot { tensor, shape })
            .collect();
        let raw_operations = records
            .iter()
            .map(|record| {
                assert_eq!(
                    record.input_tensor_ids.len(),
                    record.input_shapes.len(),
                    "graph observation: recorded input identities and shapes must have equal counts"
                );
                assert_eq!(
                    record.output_tensor_ids.len(),
                    record.output_shapes.len(),
                    "graph observation: recorded output identities and shapes must have equal counts"
                );
                RawGraphOp {
                    kind: record.kind,
                    name: record.name.clone(),
                    inputs: record
                        .input_tensor_ids
                        .iter()
                        .copied()
                        .zip(record.input_shapes.iter().cloned())
                        .map(|(id, shape)| RawGraphTensor { id, shape })
                        .collect(),
                    outputs: record
                        .output_tensor_ids
                        .iter()
                        .copied()
                        .zip(record.output_shapes.iter().cloned())
                        .map(|(id, shape)| RawGraphTensor { id, shape })
                        .collect(),
                }
            })
            .collect();
        Self::normalize(raw_operations, raw_roots)
    }

    fn observe_call(
        call: &OpCallRef,
        seen: &mut HashSet<OpCallId>,
        operations: &mut Vec<OpCallRef>,
    ) {
        if !seen.insert(call.id) {
            return;
        }
        for input in &call.inputs {
            if let Some(producer) = &input.producer {
                Self::observe_call(producer, seen, operations);
            }
        }
        operations.push(call.clone());
    }

    fn normalize(raw_operations: Vec<RawGraphOp>, raw_roots: Vec<RawGraphRoot>) -> Self {
        fn tensor_id(raw: TensorId, ids: &mut HashMap<TensorId, GraphTensorId>) -> GraphTensorId {
            if let Some(id) = ids.get(&raw) {
                return *id;
            }
            let id = GraphTensorId(ids.len());
            ids.insert(raw, id);
            id
        }

        let mut raw_tensors = HashMap::new();
        let mut raw_producers = HashMap::new();
        let mut operations = Vec::with_capacity(raw_operations.len());
        for (operation_index, raw) in raw_operations.into_iter().enumerate() {
            let inputs = raw
                .inputs
                .into_iter()
                .map(|input| GraphTensor {
                    id: tensor_id(input.id, &mut raw_tensors),
                    shape: input.shape,
                })
                .collect();
            for output in &raw.outputs {
                raw_producers.insert(output.id, GraphOpIndex(operation_index));
            }
            let outputs = raw
                .outputs
                .into_iter()
                .map(|output| GraphTensor {
                    id: tensor_id(output.id, &mut raw_tensors),
                    shape: output.shape,
                })
                .collect();
            operations.push(GraphOp {
                kind: raw.kind,
                name: raw.name,
                inputs,
                outputs,
            });
        }
        let roots = raw_roots
            .into_iter()
            .map(|root| GraphRoot {
                tensor: tensor_id(root.tensor, &mut raw_tensors),
                shape: root.shape,
            })
            .collect();
        Self {
            operations,
            roots,
            raw_tensors,
            raw_producers,
        }
    }
}

/// One node in the autograd DAG: a single forward op that ran while grad mode
/// was enabled. `OpCall`s are heap-allocated behind an `Rc` ([`OpCallRef`]) so
/// that both the produced tensor (via its `producer` link) and the topo-sort
/// list can share ownership cheaply.
///
/// Reading order for learners: `inputs` tells you *where gradients flow to*,
/// `backward` tells you *how to compute them*, and `outputs`/`output_shapes`
/// tell you *what gradients flow in* during the reverse pass.
pub struct OpCall {
    /// Unique id, used to dedupe nodes during topo-sort (a tensor reused by
    /// several consumers is still produced by exactly one `OpCall`).
    pub id: OpCallId,
    /// Op discriminant, used for FLOPs accounting and debug tables.
    pub kind: OpKind,
    /// Human-readable label (e.g. `"mlp.fc1"`) for graphs and op tables.
    pub name: String,
    /// One entry per input tensor: the address to route this op's input
    /// gradients to, plus the `producer` link needed to recurse further back.
    pub inputs: Vec<GradTarget>,
    /// Ids of the tensors this op produced (usually one).
    pub outputs: Vec<TensorId>,
    /// Shapes of the outputs, so backward can synthesize a zero grad when an
    /// output turned out to be unused downstream.
    pub output_shapes: Vec<Shape>,
    /// The local chain-rule rule: `∂L/∂outputs` in, `∂L/∂inputs` out.
    pub backward: Box<dyn BackwardRecipe>,
    /// Book-keeping of activations this op saved for its backward, used by the
    /// saved-tensor / checkpoint memory reports.
    pub debug_saved: Vec<crate::saved::SaveSite>,
}

/// A reference to one *input* of an op — the answer to "when I have this op's
/// gradient for this input, where do I put it and how do I keep going back?".
///
/// It is a *snapshot* taken at forward time (see `Tensor::grad_target`), so the
/// backward pass does not need to re-borrow the live tensor.
#[derive(Clone)]
pub struct GradTarget {
    /// Identity of the input tensor; the key used in the accumulation map.
    pub id: TensorId,
    /// Shape of the input, so grads can be zero-initialized when needed.
    pub shape: Shape,
    /// If false, this edge is pruned — no gradient is computed or stored.
    pub requires_grad: bool,
    /// The op that produced this input, or `None` if it is a leaf (a parameter
    /// or model input). Following this recursively reconstructs the DAG.
    pub producer: Option<OpCallRef>,
    /// For leaves only: a weak handle back to the tensor's storage, so the final
    /// gradient can be deposited into `leaf.autograd.grad` for the caller to read.
    /// Weak to avoid a reference cycle keeping the tensor alive forever.
    pub leaf: Option<Weak<RefCell<TensorInner>>>,
}

/// A computed input gradient (`grad`) paired with the input it belongs to
/// (`target`). Backward recipes return a `Vec<GradEdge>`, one per input.
pub struct GradEdge {
    pub target: GradTarget,
    pub grad: TensorValue,
}

/// The per-op chain rule. Given the gradients flowing into each *output*
/// (`grad_outputs`), return the gradients flowing to each *input*.
///
/// Implementors typically `ctx.unpack` any activations they saved at forward
/// time (e.g. `mul` needs the original operands to compute `dout·y`, `dout·x`).
/// Each op lives in `src/ops/*` next to its forward function.
pub trait BackwardRecipe {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge>;
}

/// Scratch context threaded through a single backward pass. Currently it just
/// carries the [`DebugRecorder`] so recipes can unpack saved tensors (and
/// record recompute events for gradient checkpointing).
pub struct BackwardCtx {
    pub debug: DebugRecorder,
}

impl BackwardCtx {
    /// Materialize a saved activation — either returning the stashed value or
    /// recomputing it, transparently to the recipe.
    pub fn unpack(&mut self, saved: &SavedTensor) -> TensorValue {
        saved.unpack(&self.debug)
    }
}

/// Drives the reverse pass and owns the gradient accumulator.
pub struct Engine {
    /// Accumulated gradient per tensor id. During the reverse walk a tensor's
    /// entry is *summed* across every consumer that sends it a gradient, then
    /// `remove`d and consumed when that tensor's producer is visited.
    pub grads: HashMap<TensorId, TensorValue>,
    /// FLOPs / saved-tensor recorder shared with the ops.
    pub debug: DebugRecorder,
    /// The DAG in topological order (producers before consumers); walked in
    /// reverse by `backward`. Retained afterwards for the debug tables/graphs.
    pub topo: Vec<OpCallRef>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Create an engine with an initially empty recorder.
    ///
    /// Forward observation is explicit: use [`Self::with_recording`] around
    /// the execution whose operations belong to this engine's trace.
    pub fn new() -> Self {
        let debug = DebugRecorder::new();
        Self {
            grads: HashMap::new(),
            debug,
            topo: vec![],
        }
    }

    /// Observe operations and save events produced during `f`.
    ///
    /// Scopes may nest across engines. The previous recorder is restored on
    /// normal return and during panic unwinding.
    pub fn with_recording<R>(&self, f: impl FnOnce() -> R) -> R {
        crate::debug::with_recorder(&self.debug, f)
    }

    /// Compute gradients of `loss` w.r.t. every leaf tensor that required grad.
    ///
    /// After this returns, each such leaf's `.grad()` holds `∂loss/∂leaf`.
    /// The pass has three steps: seed → topo-sort → reverse walk.
    pub fn backward(&mut self, loss: &Tensor) {
        let recorder = self.debug.clone();
        crate::debug::with_recorder(&recorder, || self.backward_impl(loss));
    }

    fn backward_impl(&mut self, loss: &Tensor) {
        // ── Step 1: seed the recursion ──────────────────────────────────────
        // The chain rule needs a starting gradient. `∂loss/∂loss = 1`. Backprop
        // only makes sense from a scalar objective, hence the numel==1 assert.
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

        // ── Step 2: order the DAG so every node is visited after all nodes it
        // feeds. We store producers-before-consumers, then iterate in reverse.
        let topo = self.collect_topo(loss);
        self.topo = topo.clone();

        // ── Step 3: reverse walk. Visiting consumers before producers
        // guarantees a tensor has received gradients from *all* its consumers
        // before its own producer consumes them — the multivariate chain rule.
        for call in topo.into_iter().rev() {
            // Gather the gradient that flowed into each output of this op. We
            // `remove` it (each output is consumed exactly once, here) and fall
            // back to zeros if the output turned out to be unused downstream.
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

            // Apply the local chain rule for this op: output grads -> input grads.
            let edges = call.backward.backward(&grad_outputs, &mut ctx);

            // Route each input gradient to its target, skipping pruned edges.
            for edge in edges {
                if !edge.target.requires_grad {
                    continue;
                }
                self.accumulate(edge.target, edge.grad);
            }
        }
    }

    /// Add `grad` into the running total for `target`. This is where the
    /// *summation* half of the chain rule happens: a tensor consumed by N ops
    /// receives N gradient contributions, and they must be summed.
    ///
    /// Two sinks receive the gradient:
    ///  1. `self.grads[target.id]` — the working accumulator the reverse walk
    ///     later reads when it reaches this tensor's producer.
    ///  2. `leaf.autograd.grad` — for leaves, the user-visible result exposed by
    ///     `Tensor::grad()`.
    fn accumulate(&mut self, target: GradTarget, grad: TensorValue) {
        self.debug.record_grad_accum(&target, &grad);

        // Sink 1: intermediate accumulator (sum if a contribution already arrived).
        if let Some(existing) = self.grads.get_mut(&target.id) {
            let summed = crate::ops::basic::raw_add_values(existing, &grad);
            *existing = summed;
        } else {
            self.grads.insert(target.id, grad.clone());
        }

        // Sink 2: if this input is a live leaf, also fold the gradient into the
        // tensor's own `.grad` so the caller can read it after `backward`.
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

    /// Topologically sort the DAG reachable from `loss` (producers first).
    ///
    /// Starts at the op that produced `loss` and walks `producer` links back to
    /// the leaves via DFS. The DFS naturally yields post-order, which for this
    /// producer-points-backward graph *is* topological order.
    fn collect_topo(&self, loss: &Tensor) -> Vec<OpCallRef> {
        let mut out = vec![];
        let mut seen = HashSet::new();
        let producer = loss.inner.borrow().autograd.producer.clone();
        if let Some(call) = producer {
            Self::dfs_call(&call, &mut seen, &mut out);
        }
        out
    }

    /// Post-order DFS over `OpCall`s. `seen` (keyed by `OpCallId`) dedupes nodes
    /// shared by multiple consumers, so each op is emitted exactly once. Because
    /// a node is pushed to `out` only after all its input-producers have been
    /// pushed, `out` ends up producers-before-consumers.
    fn dfs_call(call: &OpCallRef, seen: &mut HashSet<OpCallId>, out: &mut Vec<OpCallRef>) {
        if !seen.insert(call.id) {
            return; // already visited — a shared subgraph, skip it
        }
        for input in &call.inputs {
            if let Some(parent) = &input.producer {
                Self::dfs_call(parent, seen, out);
            }
        }
        out.push(call.clone());
    }

    // ── Reporting helpers ──────────────────────────────────────────────────
    // These read nothing from the gradient math; they render the retained
    // `topo` and the recorder's data for teaching/analysis. Call them after
    // `backward` has populated `self.topo`.

    /// Print one row per op: kind, name, shapes, and FLOPs.
    pub fn print_op_table(&self) {
        self.debug.print_op_table(&self.topo);
    }

    /// Print the activations saved for backward and their memory cost.
    pub fn print_saved_tensor_table(&self) {
        self.debug.print_saved_tensor_table();
    }

    /// Print the gradient-checkpointing summary (saved vs. recomputed).
    pub fn print_checkpoint_report(&self) {
        self.debug.print_checkpoint_report();
    }

    /// Emit the DAG as Graphviz DOT for visualizing the computation graph.
    pub fn write_dot(&self, path: &str) {
        self.debug.write_dot(&self.topo, path);
    }

    /// Render the DAG as Graphviz DOT for callers that need fallible I/O.
    pub fn dot_string(&self) -> String {
        self.debug.dot_string(&self.topo)
    }
}
