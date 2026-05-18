use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::{GradTarget, OpCallId, OpCallRef, OpKind};
use crate::saved::SaveSite;
use crate::tensor::{Shape, TensorId, TensorValue};

#[derive(Clone, Debug)]
pub struct OpCallRecord {
    pub id: OpCallId,
    pub kind: OpKind,
    pub name: String,
    pub input_shapes: Vec<Shape>,
    pub output_shapes: Vec<Shape>,
    pub saved_sites: Vec<SaveSite>,
}

#[derive(Clone, Debug)]
pub enum SaveEvent {
    Materialized { site: SaveSite },
    Borrowed { site: SaveSite },
    Recomputable { site: SaveSite, checkpoint_id: usize, ordinal: usize },
    UnpackedMaterialized { site: SaveSite },
    UnpackedBorrowed { site: SaveSite },
    UnpackedRecomputed { site: SaveSite },
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
    ApplyOp { id: OpCallId, name: String, kind: OpKind },
}

#[derive(Clone, Debug)]
pub enum GradEvent {
    Accumulate { tensor_id: TensorId, shape: Shape, bytes: usize },
    LeafWrite { tensor_id: TensorId },
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

impl DebugRecorder {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(DebugStore::default())),
        }
    }

    pub fn record_op_call(&self, call: &OpCallRef) {
        let mut s = self.inner.borrow_mut();
        s.op_calls.push(OpCallRecord {
            id: call.id,
            kind: call.kind,
            name: call.name.clone(),
            input_shapes: call.inputs.iter().map(|t| t.shape.clone()).collect(),
            output_shapes: call.output_shapes.clone(),
            saved_sites: call.debug_saved.clone(),
        });
    }

    pub fn record_save_materialized(&self, site: &SaveSite) {
        self.inner.borrow_mut().save_events.push(SaveEvent::Materialized { site: site.clone() });
    }

    pub fn record_save_borrowed(&self, site: &SaveSite) {
        self.inner.borrow_mut().save_events.push(SaveEvent::Borrowed { site: site.clone() });
    }

    pub fn record_save_recomputable(&self, site: &SaveSite, checkpoint_id: usize, ordinal: usize) {
        self.inner.borrow_mut().save_events.push(SaveEvent::Recomputable {
            site: site.clone(),
            checkpoint_id,
            ordinal,
        });
    }

    pub fn record_saved_unpack_materialized(&self, site: &SaveSite) {
        self.inner.borrow_mut().save_events.push(SaveEvent::UnpackedMaterialized { site: site.clone() });
    }

    pub fn record_saved_unpack_borrowed(&self, site: &SaveSite) {
        self.inner.borrow_mut().save_events.push(SaveEvent::UnpackedBorrowed { site: site.clone() });
    }

    pub fn record_saved_unpack_recompute(&self, site: &SaveSite) {
        self.inner.borrow_mut().save_events.push(SaveEvent::UnpackedRecomputed { site: site.clone() });
    }

    pub fn record_checkpoint_enter(&self, id: usize, name: &str) {
        self.inner.borrow_mut().checkpoint_events.push(CheckpointEvent::Enter {
            id,
            name: name.to_string(),
        });
    }

    pub fn record_checkpoint_exit(&self, id: usize) {
        self.inner.borrow_mut().checkpoint_events.push(CheckpointEvent::Exit { id });
    }

    pub fn record_recompute_start(&self, id: usize, name: &str) {
        self.inner.borrow_mut().checkpoint_events.push(CheckpointEvent::RecomputeStart {
            id,
            name: name.to_string(),
        });
    }

    pub fn record_recompute_end(&self, id: usize, saved_count: usize) {
        self.inner.borrow_mut().checkpoint_events.push(CheckpointEvent::RecomputeEnd {
            id,
            saved_count,
        });
    }

    pub fn record_backward_call(&self, call: &OpCallRef) {
        self.inner.borrow_mut().backward_events.push(BackwardEvent::ApplyOp {
            id: call.id,
            name: call.name.clone(),
            kind: call.kind,
        });
    }

    pub fn record_grad_accum(&self, target: &GradTarget, _grad: &TensorValue) {
        self.inner.borrow_mut().grad_events.push(GradEvent::Accumulate {
            tensor_id: target.id,
            shape: target.shape.clone(),
            bytes: target.shape.bytes_f32(),
        });
        if target.leaf.is_some() {
            self.inner.borrow_mut().grad_events.push(GradEvent::LeafWrite {
                tensor_id: target.id,
            });
        }
    }

    pub fn print_op_table(&self, op_calls: &[OpCallRef]) {
        println!("{:<4} {:<18} {:<20} {:<16} {}", "#", "kind", "name", "output shape", "saved bytes");
        println!("{}", "-".repeat(72));
        for (i, call) in op_calls.iter().enumerate() {
            let out_shape = call.output_shapes.first().map(|s| s.to_string()).unwrap_or_default();
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
        println!("{:<8} {:<28} {:<14} {:<8} {}", "ordinal", "site", "shape", "bytes", "kind");
        println!("{}", "-".repeat(72));
        let mut ordinal = 0usize;
        for ev in &store.save_events {
            match ev {
                SaveEvent::Materialized { site } => {
                    println!("{:<8} {:<28} {:<14} {:<8} materialized", ordinal, site.name, site.shape.to_string(), site.bytes);
                    ordinal += 1;
                }
                SaveEvent::Borrowed { site } => {
                    println!("{:<8} {:<28} {:<14} {:<8} borrowed(param)", ordinal, site.name, site.shape.to_string(), site.bytes);
                    ordinal += 1;
                }
                SaveEvent::Recomputable { site, checkpoint_id, ordinal: ord } => {
                    println!("{:<8} {:<28} {:<14} {:<8} recompute(chk={},ord={})", ord, site.name, site.shape.to_string(), 0, checkpoint_id, ord);
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
                CheckpointEvent::Enter { id, name } => println!("  [enter] chk={} name={}", id, name),
                CheckpointEvent::Exit { id } => println!("  [exit]  chk={}", id),
                CheckpointEvent::RecomputeStart { id, name } => println!("  [recompute-start] chk={} name={}", id, name),
                CheckpointEvent::RecomputeEnd { id, saved_count } => println!("  [recompute-end]   chk={} saved_count={}", id, saved_count),
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
    static GLOBAL_RECORDER: RefCell<Option<DebugRecorder>> = RefCell::new(None);
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
    GLOBAL_RECORDER.with(|g| {
        g.borrow()
            .clone()
            .unwrap_or_else(DebugRecorder::new)
    })
}
