//! Debug recorder: op table, saved-tensor table, DOT graph.
//!
//! Book reference: Ch.9 "Profile TPU Code" (conceptual analog),
//! <https://jax-ml.github.io/scaling-book/profiling/>
//!
//! `DebugRecorder` intercepts every op call, save event, and grad accumulation
//! during a forward+backward pass and renders three reports:
//! - `print_op_table` — ordered list of ops with input/output shapes
//! - `print_saved_tensor_table` — bytes saved per tensor (materialized / borrowed / recompute)
//! - `write_dot` — Graphviz DOT graph of the op DAG
//!
//! These serve the same purpose as a TPU profiler at a conceptual level, though
//! tnsr is CPU-only and produces no timing or hardware-counter data.

use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::{GradTarget, OpCallId, OpCallRef, OpKind};
use crate::saved::{SaveRole, SaveSite};
use crate::tensor::{Shape, TensorId, TensorValue};

#[derive(Clone, Debug)]
pub struct OpCallRecord {
    pub id: OpCallId,
    pub kind: OpKind,
    pub name: String,
    pub input_shapes: Vec<Shape>,
    pub input_tensor_ids: Vec<TensorId>,
    pub input_requires_grad: Vec<bool>,
    pub output_tensor_ids: Vec<TensorId>,
    pub output_shapes: Vec<Shape>,
    pub saved_sites: Vec<SaveSite>,
}

#[derive(Clone, Debug)]
pub enum SaveEvent {
    Materialized {
        site: SaveSite,
    },
    Borrowed {
        site: SaveSite,
    },
    Recomputable {
        site: SaveSite,
        checkpoint_id: usize,
        ordinal: usize,
    },
    UnpackedMaterialized {
        site: SaveSite,
    },
    UnpackedBorrowed {
        site: SaveSite,
    },
    UnpackedRecomputed {
        site: SaveSite,
    },
}

#[derive(Clone, Debug)]
pub enum CheckpointEvent {
    Enter { id: usize, name: String },
    Exit { id: usize },
    RecomputeStart { id: usize, name: String },
    RecomputeEnd { id: usize, saved_count: usize },
}

#[derive(Clone, Debug)]
pub enum BackwardEvent {
    ApplyOp {
        id: OpCallId,
        name: String,
        kind: OpKind,
    },
}

#[derive(Clone, Debug)]
pub enum GradEvent {
    Accumulate {
        tensor_id: TensorId,
        shape: Shape,
        bytes: usize,
    },
    LeafWrite {
        tensor_id: TensorId,
    },
}

#[derive(Default)]
struct DebugStore {
    op_calls: Vec<OpCallRecord>,
    save_events: Vec<SaveEvent>,
    checkpoint_events: Vec<CheckpointEvent>,
    backward_events: Vec<BackwardEvent>,
    grad_events: Vec<GradEvent>,
}

#[derive(Clone)]
pub struct DebugRecorder {
    inner: Rc<RefCell<DebugStore>>,
}

impl Default for DebugRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugRecorder {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(DebugStore::default())),
        }
    }

    /// Clone out every forward op call recorded so far. Read-only accessor for
    /// analysis tools (e.g. the Qwen3 op-closure tracer) that want to inspect
    /// what the forward pass dispatched without touching the private store.
    pub fn op_call_records(&self) -> Vec<OpCallRecord> {
        self.inner.borrow().op_calls.clone()
    }

    /// Clone out the `OpKind` of every op applied during the backward pass, in
    /// application order. The Rust analog of counting ATen backward ops.
    pub fn backward_apply_kinds(&self) -> Vec<OpKind> {
        self.inner
            .borrow()
            .backward_events
            .iter()
            .map(|ev| match ev {
                BackwardEvent::ApplyOp { kind, .. } => *kind,
            })
            .collect()
    }

    pub fn record_op_call(&self, call: &OpCallRef) {
        let mut s = self.inner.borrow_mut();
        s.op_calls.push(OpCallRecord {
            id: call.id,
            kind: call.kind,
            name: call.name.clone(),
            input_shapes: call.inputs.iter().map(|t| t.shape.clone()).collect(),
            input_tensor_ids: call.inputs.iter().map(|t| t.id).collect(),
            input_requires_grad: call.inputs.iter().map(|t| t.requires_grad).collect(),
            output_tensor_ids: call.outputs.clone(),
            output_shapes: call.output_shapes.clone(),
            saved_sites: call.debug_saved.clone(),
        });
    }

    pub fn record_save_materialized(&self, site: &SaveSite) {
        self.inner
            .borrow_mut()
            .save_events
            .push(SaveEvent::Materialized { site: site.clone() });
    }

    pub fn record_save_borrowed(&self, site: &SaveSite) {
        self.inner
            .borrow_mut()
            .save_events
            .push(SaveEvent::Borrowed { site: site.clone() });
    }

    pub fn record_save_recomputable(&self, site: &SaveSite, checkpoint_id: usize, ordinal: usize) {
        self.inner
            .borrow_mut()
            .save_events
            .push(SaveEvent::Recomputable {
                site: site.clone(),
                checkpoint_id,
                ordinal,
            });
    }

    pub fn record_saved_unpack_materialized(&self, site: &SaveSite) {
        self.inner
            .borrow_mut()
            .save_events
            .push(SaveEvent::UnpackedMaterialized { site: site.clone() });
    }

    pub fn record_saved_unpack_borrowed(&self, site: &SaveSite) {
        self.inner
            .borrow_mut()
            .save_events
            .push(SaveEvent::UnpackedBorrowed { site: site.clone() });
    }

    pub fn record_saved_unpack_recompute(&self, site: &SaveSite) {
        self.inner
            .borrow_mut()
            .save_events
            .push(SaveEvent::UnpackedRecomputed { site: site.clone() });
    }

    pub fn record_checkpoint_enter(&self, id: usize, name: &str) {
        self.inner
            .borrow_mut()
            .checkpoint_events
            .push(CheckpointEvent::Enter {
                id,
                name: name.to_string(),
            });
    }

    pub fn record_checkpoint_exit(&self, id: usize) {
        self.inner
            .borrow_mut()
            .checkpoint_events
            .push(CheckpointEvent::Exit { id });
    }

    pub fn record_recompute_start(&self, id: usize, name: &str) {
        self.inner
            .borrow_mut()
            .checkpoint_events
            .push(CheckpointEvent::RecomputeStart {
                id,
                name: name.to_string(),
            });
    }

    pub fn record_recompute_end(&self, id: usize, saved_count: usize) {
        self.inner
            .borrow_mut()
            .checkpoint_events
            .push(CheckpointEvent::RecomputeEnd { id, saved_count });
    }

    pub fn record_backward_call(&self, call: &OpCallRef) {
        self.inner
            .borrow_mut()
            .backward_events
            .push(BackwardEvent::ApplyOp {
                id: call.id,
                name: call.name.clone(),
                kind: call.kind,
            });
    }

    pub fn record_grad_accum(&self, target: &GradTarget, _grad: &TensorValue) {
        self.inner
            .borrow_mut()
            .grad_events
            .push(GradEvent::Accumulate {
                tensor_id: target.id,
                shape: target.shape.clone(),
                bytes: target.shape.bytes_f32(),
            });
        if target.leaf.is_some() {
            self.inner
                .borrow_mut()
                .grad_events
                .push(GradEvent::LeafWrite {
                    tensor_id: target.id,
                });
        }
    }

    /// Export a stable, Rust-owned trace schema for agent tools and future
    /// viewers. The schema includes both normalized records and a replayable
    /// event stream so consumers can choose the simpler representation.
    pub fn trace_json(&self) -> serde_json::Value {
        let store = self.inner.borrow();
        let forward_ops: Vec<serde_json::Value> =
            store.op_calls.iter().map(forward_op_json).collect();

        let mut events = Vec::new();
        for call in &store.op_calls {
            events.push(serde_json::json!({
                "kind": "op",
                "detail": call.name,
                "bytes": call.output_shapes.iter().map(Shape::bytes_f32).sum::<usize>(),
                "op_id": call.id.0,
                "op_kind": op_kind_name(call.kind),
                "op_name": call.name,
                "inputs": input_tensor_json(call),
                "outputs": output_tensor_json(call),
                "saved_sites": call.saved_sites.iter().map(save_site_json).collect::<Vec<_>>(),
            }));
        }
        for ev in &store.save_events {
            events.push(save_event_json(ev));
        }
        for ev in &store.checkpoint_events {
            events.push(checkpoint_event_json(ev));
        }
        for ev in &store.backward_events {
            events.push(backward_event_json(ev));
        }
        for ev in &store.grad_events {
            events.push(grad_event_json(ev));
        }

        serde_json::json!({
            "schema": "tnsr.debug_trace",
            "schema_version": 1,
            "events": events,
            "forward_ops": forward_ops,
            "saved_sites": store.save_events.iter().filter_map(save_event_site_json).collect::<Vec<_>>(),
            "checkpoints": store.checkpoint_events.iter().map(checkpoint_event_json).collect::<Vec<_>>(),
            "backward_ops": store.backward_events.iter().map(backward_event_json).collect::<Vec<_>>(),
            "grad_accumulations": store.grad_events.iter().map(grad_event_json).collect::<Vec<_>>(),
        })
    }

    pub fn trace_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.trace_json()).expect("debug trace JSON serialization")
    }

    pub fn print_op_table(&self, op_calls: &[OpCallRef]) {
        println!(
            "{:<4} {:<18} {:<20} {:<16} saved bytes",
            "#", "kind", "name", "output shape"
        );
        println!("{}", "-".repeat(72));
        for (i, call) in op_calls.iter().enumerate() {
            let out_shape = call
                .output_shapes
                .first()
                .map(|s| s.to_string())
                .unwrap_or_default();
            let saved_bytes: usize = call.debug_saved.iter().map(|s| s.bytes).sum();
            println!(
                "{:<4} {:<18} {:<20} {:<16} {}",
                i + 1,
                format!("{:?}", call.kind),
                call.name,
                out_shape,
                saved_bytes,
            );
        }
    }

    pub fn print_saved_tensor_table(&self) {
        let store = self.inner.borrow();
        println!("\nSaved tensor events:");
        println!(
            "{:<8} {:<28} {:<14} {:<8} kind",
            "ordinal", "site", "shape", "bytes"
        );
        println!("{}", "-".repeat(72));
        let mut ordinal = 0usize;
        for ev in &store.save_events {
            match ev {
                SaveEvent::Materialized { site } => {
                    println!(
                        "{:<8} {:<28} {:<14} {:<8} materialized",
                        ordinal,
                        site.name,
                        site.shape.to_string(),
                        site.bytes
                    );
                    ordinal += 1;
                }
                SaveEvent::Borrowed { site } => {
                    println!(
                        "{:<8} {:<28} {:<14} {:<8} borrowed(param)",
                        ordinal,
                        site.name,
                        site.shape.to_string(),
                        site.bytes
                    );
                    ordinal += 1;
                }
                SaveEvent::Recomputable {
                    site,
                    checkpoint_id,
                    ordinal: ord,
                } => {
                    println!(
                        "{:<8} {:<28} {:<14} {:<8} recompute(chk={},ord={})",
                        ord,
                        site.name,
                        site.shape.to_string(),
                        0,
                        checkpoint_id,
                        ord
                    );
                }
                _ => {}
            }
        }
    }

    pub fn print_checkpoint_report(&self) {
        let store = self.inner.borrow();
        println!("\nCheckpoint events:");
        for ev in &store.checkpoint_events {
            match ev {
                CheckpointEvent::Enter { id, name } => {
                    println!("  [enter] chk={} name={}", id, name)
                }
                CheckpointEvent::Exit { id } => println!("  [exit]  chk={}", id),
                CheckpointEvent::RecomputeStart { id, name } => {
                    println!("  [recompute-start] chk={} name={}", id, name)
                }
                CheckpointEvent::RecomputeEnd { id, saved_count } => {
                    println!("  [recompute-end]   chk={} saved_count={}", id, saved_count)
                }
            }
        }
    }

    pub fn write_dot(&self, op_calls: &[OpCallRef], path: &str) {
        let mut s = String::new();
        s.push_str("digraph Autograd {\n");
        s.push_str("  rankdir=LR;\n");
        s.push_str("  node [fontname=\"Helvetica\"];\n");

        for call in op_calls {
            s.push_str(&format!(
                "  op_{} [label=\"#{} {:?}\\n{}\", shape=ellipse];\n",
                call.id.0, call.id.0, call.kind, call.name,
            ));
            for inp in &call.inputs {
                s.push_str(&format!(
                    "  tensor_{} [label=\"T{}\\n{}\", shape=box];\n",
                    inp.id.0, inp.id.0, inp.shape,
                ));
                s.push_str(&format!("  tensor_{} -> op_{};\n", inp.id.0, call.id.0));
            }
            for out_id in &call.outputs {
                s.push_str(&format!("  op_{} -> tensor_{};\n", call.id.0, out_id.0));
            }
        }

        s.push_str("}\n");
        std::fs::write(path, s).unwrap_or_else(|e| eprintln!("write_dot error: {}", e));
    }
}

thread_local! {
    static GLOBAL_RECORDER: RefCell<Option<DebugRecorder>> = const { RefCell::new(None) };
}

fn forward_op_json(call: &OpCallRecord) -> serde_json::Value {
    serde_json::json!({
        "id": call.id.0,
        "kind": op_kind_name(call.kind),
        "name": call.name,
        "display_label": format!("#{} {}\\n{}", call.id.0, op_kind_name(call.kind), call.name),
        "bytes": call.output_shapes.iter().map(Shape::bytes_f32).sum::<usize>(),
        "inputs": input_tensor_json(call),
        "outputs": output_tensor_json(call),
        "edges": {
            "inputs": call.input_tensor_ids.iter().map(|id| id.0).collect::<Vec<_>>(),
            "outputs": call.output_tensor_ids.iter().map(|id| id.0).collect::<Vec<_>>(),
        },
        "saved_sites": call.saved_sites.iter().map(save_site_json).collect::<Vec<_>>(),
    })
}

fn input_tensor_json(call: &OpCallRecord) -> Vec<serde_json::Value> {
    call.input_tensor_ids
        .iter()
        .zip(call.input_shapes.iter())
        .zip(call.input_requires_grad.iter())
        .map(|((id, shape), requires_grad)| tensor_json(*id, shape, *requires_grad))
        .collect()
}

fn output_tensor_json(call: &OpCallRecord) -> Vec<serde_json::Value> {
    call.output_tensor_ids
        .iter()
        .zip(call.output_shapes.iter())
        .map(|(id, shape)| tensor_json(*id, shape, true))
        .collect()
}

fn tensor_json(id: TensorId, shape: &Shape, requires_grad: bool) -> serde_json::Value {
    serde_json::json!({
        "id": id.0,
        "shape": shape.0,
        "requires_grad": requires_grad,
        "bytes": shape.bytes_f32(),
    })
}

fn save_site_json(site: &SaveSite) -> serde_json::Value {
    serde_json::json!({
        "op_kind": op_kind_name(site.op),
        "name": site.name,
        "role": save_role_name(site.role),
        "shape": site.shape.0,
        "bytes": site.bytes,
        "display_label": format!("{} {} {}", save_role_name(site.role), site.name, site.shape),
    })
}

fn save_event_site_json(ev: &SaveEvent) -> Option<serde_json::Value> {
    match ev {
        SaveEvent::Materialized { site }
        | SaveEvent::Borrowed { site }
        | SaveEvent::Recomputable { site, .. }
        | SaveEvent::UnpackedMaterialized { site }
        | SaveEvent::UnpackedBorrowed { site }
        | SaveEvent::UnpackedRecomputed { site } => Some(save_site_json(site)),
    }
}

fn save_event_json(ev: &SaveEvent) -> serde_json::Value {
    match ev {
        SaveEvent::Materialized { site } => {
            save_event_with_detail("saved", "materialized", site, site.bytes)
        }
        SaveEvent::Borrowed { site } => save_event_with_detail("saved", "borrowed", site, 0),
        SaveEvent::Recomputable {
            site,
            checkpoint_id,
            ordinal,
        } => {
            let mut value = save_event_with_detail("saved", "recomputable", site, 0);
            value["checkpoint_id"] = serde_json::json!(checkpoint_id);
            value["ordinal"] = serde_json::json!(ordinal);
            value
        }
        SaveEvent::UnpackedMaterialized { site } => {
            save_event_with_detail("saved_unpack", "materialized", site, site.bytes)
        }
        SaveEvent::UnpackedBorrowed { site } => {
            save_event_with_detail("saved_unpack", "borrowed", site, 0)
        }
        SaveEvent::UnpackedRecomputed { site } => {
            save_event_with_detail("saved_unpack", "recomputed", site, site.bytes)
        }
    }
}

fn save_event_with_detail(
    kind: &str,
    detail: &str,
    site: &SaveSite,
    bytes: usize,
) -> serde_json::Value {
    serde_json::json!({
        "kind": kind,
        "detail": detail,
        "bytes": bytes,
        "op_kind": op_kind_name(site.op),
        "save_site": save_site_json(site),
    })
}

fn checkpoint_event_json(ev: &CheckpointEvent) -> serde_json::Value {
    match ev {
        CheckpointEvent::Enter { id, name } => serde_json::json!({
            "kind": "checkpoint",
            "detail": "enter",
            "checkpoint_id": id,
            "name": name,
        }),
        CheckpointEvent::Exit { id } => serde_json::json!({
            "kind": "checkpoint",
            "detail": "exit",
            "checkpoint_id": id,
        }),
        CheckpointEvent::RecomputeStart { id, name } => serde_json::json!({
            "kind": "checkpoint",
            "detail": "recompute_start",
            "checkpoint_id": id,
            "name": name,
        }),
        CheckpointEvent::RecomputeEnd { id, saved_count } => serde_json::json!({
            "kind": "checkpoint",
            "detail": "recompute_end",
            "checkpoint_id": id,
            "saved_count": saved_count,
        }),
    }
}

fn backward_event_json(ev: &BackwardEvent) -> serde_json::Value {
    match ev {
        BackwardEvent::ApplyOp { id, name, kind } => serde_json::json!({
            "kind": "backward",
            "detail": "apply_op",
            "bytes": 0,
            "op_id": id.0,
            "op_kind": op_kind_name(*kind),
            "op_name": name,
            "display_label": format!("#{} {}\\n{}", id.0, op_kind_name(*kind), name),
        }),
    }
}

fn grad_event_json(ev: &GradEvent) -> serde_json::Value {
    match ev {
        GradEvent::Accumulate {
            tensor_id,
            shape,
            bytes,
        } => serde_json::json!({
            "kind": "grad_accum",
            "detail": "accumulate",
            "bytes": bytes,
            "output": tensor_json(*tensor_id, shape, true),
        }),
        GradEvent::LeafWrite { tensor_id } => serde_json::json!({
            "kind": "grad_leaf_write",
            "detail": "leaf_write",
            "bytes": 0,
            "tensor_id": tensor_id.0,
        }),
    }
}

fn op_kind_name(kind: OpKind) -> &'static str {
    match kind {
        OpKind::Add => "Add",
        OpKind::Mul => "Mul",
        OpKind::Scale => "Scale",
        OpKind::Sum => "Sum",
        OpKind::Embedding => "Embedding",
        OpKind::Linear => "Linear",
        OpKind::LayerNorm => "LayerNorm",
        OpKind::RmsNorm => "RmsNorm",
        OpKind::Gelu => "Gelu",
        OpKind::Silu => "Silu",
        OpKind::SwiGlu => "SwiGlu",
        OpKind::AttentionScores => "AttentionScores",
        OpKind::CausalMask => "CausalMask",
        OpKind::Softmax => "Softmax",
        OpKind::AttentionMix => "AttentionMix",
        OpKind::Dropout => "Dropout",
        OpKind::CrossEntropy => "CrossEntropy",
        OpKind::Reshape => "Reshape",
        OpKind::Split => "Split",
        OpKind::Rope => "Rope",
        OpKind::GqaAttention => "GqaAttention",
    }
}

fn save_role_name(role: SaveRole) -> &'static str {
    match role {
        SaveRole::Activation => "Activation",
        SaveRole::Parameter => "Parameter",
        SaveRole::AuxStat => "AuxStat",
        SaveRole::BoundaryInput => "BoundaryInput",
        SaveRole::RngState => "RngState",
    }
}

pub fn set_global_recorder(r: DebugRecorder) {
    GLOBAL_RECORDER.with(|g| *g.borrow_mut() = Some(r));
}

pub fn record_op_call_global(call: &OpCallRef) {
    GLOBAL_RECORDER.with(|g| {
        if let Some(r) = g.borrow().as_ref() {
            r.record_op_call(call);
        }
    });
}

pub fn record_op_call_global_checkpoint_enter(id: usize, name: &str) {
    GLOBAL_RECORDER.with(|g| {
        if let Some(r) = g.borrow().as_ref() {
            r.record_checkpoint_enter(id, name);
        }
    });
}

pub fn record_op_call_global_checkpoint_exit(id: usize) {
    GLOBAL_RECORDER.with(|g| {
        if let Some(r) = g.borrow().as_ref() {
            r.record_checkpoint_exit(id);
        }
    });
}

pub fn get_global_recorder() -> DebugRecorder {
    GLOBAL_RECORDER.with(|g| g.borrow().clone().unwrap_or_else(DebugRecorder::new))
}
