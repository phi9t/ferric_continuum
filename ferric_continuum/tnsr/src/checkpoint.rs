//! Gradient checkpointing / rematerialization.
//!
//! Book reference: Ch.4 "Miscellaneous Math" + Ch.5 "Parallelize a Transformer",
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! `checkpoint()` is the Rust analog of `jax.remat` / `jax.checkpoint`.
//! Two policies are provided:
//!
//! - **`WholeBlockCheckpoint`** — saves only the boundary input (block input x),
//!   recomputes all activations during backward.  Matches the book's "Block
//!   Remat" strategy: ~8·N FLOPs/token vs 6·N without remat.
//!
//! - **`TransformerSelectivePolicy`** — saves softmax output (if small) and
//!   attention intermediates (Q, K); recomputes GELU and LayerNorm.  Inspired
//!   by the book's "Big Matmuls Only" strategy but saves inputs rather than
//!   outputs, so it is an analog, not an exact match.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::debug::DebugRecorder;
use crate::saved::{default_save, RecomputeHandle, SaveRole, SaveSite, SavedTensor};
use crate::tensor::{Tensor, TensorId, TensorValue};

static CHECKPOINT_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn fresh_checkpoint_id() -> usize {
    CHECKPOINT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointDecision {
    MustSave,
    PreferSave,
    PreferRecompute,
    MustRecompute,
}

pub trait CheckpointPolicy {
    fn decide(&self, site: &SaveSite) -> CheckpointDecision;
}

pub struct WholeBlockCheckpoint;

impl CheckpointPolicy for WholeBlockCheckpoint {
    fn decide(&self, site: &SaveSite) -> CheckpointDecision {
        match site.role {
            SaveRole::Parameter => CheckpointDecision::MustSave,
            SaveRole::BoundaryInput => CheckpointDecision::MustSave,
            SaveRole::RngState => CheckpointDecision::MustSave,
            SaveRole::Activation | SaveRole::AuxStat => CheckpointDecision::MustRecompute,
        }
    }
}

pub struct TransformerSelectivePolicy {
    pub save_softmax_under_bytes: usize,
    pub recompute_activation_over_bytes: usize,
}

impl CheckpointPolicy for TransformerSelectivePolicy {
    fn decide(&self, site: &SaveSite) -> CheckpointDecision {
        use crate::autograd::OpKind::*;

        if site.role == SaveRole::Parameter {
            return CheckpointDecision::MustSave;
        }

        match site.op {
            Softmax if site.bytes <= self.save_softmax_under_bytes => CheckpointDecision::MustSave,

            Gelu | Silu | SwiGlu | LayerNorm | RmsNorm => CheckpointDecision::PreferRecompute,

            Linear if site.bytes >= self.recompute_activation_over_bytes => {
                CheckpointDecision::PreferRecompute
            }

            AttentionScores | AttentionMix => CheckpointDecision::PreferSave,

            _ => CheckpointDecision::PreferRecompute,
        }
    }
}

pub struct SaveRecord {
    pub ordinal: usize,
    pub original_tensor_id: TensorId,
    pub site: SaveSite,
    pub decision: CheckpointDecision,
}

pub struct CheckpointContext {
    pub id: usize,
    pub name: String,
    pub input_values: Vec<TensorValue>,
    pub run: Rc<dyn Fn(Vec<Tensor>) -> Tensor>,
    pub policy: Rc<dyn CheckpointPolicy>,
    pub original_saves: Vec<SaveRecord>,
    pub recompute_cache: HashMap<usize, TensorValue>,
    pub is_recomputing: bool,
    pub recompute_cursor: usize,
}

impl CheckpointContext {
    pub fn record_original_save(
        &mut self,
        t: &Tensor,
        site: SaveSite,
        decision: CheckpointDecision,
        debug: &DebugRecorder,
    ) -> SavedTensor {
        let ordinal = self.original_saves.len();
        let original_tensor_id = t.inner.borrow().id;

        self.original_saves.push(SaveRecord {
            ordinal,
            original_tensor_id,
            site: site.clone(),
            decision,
        });

        debug.record_save_recomputable(&site, self.id, ordinal);

        SavedTensor::Recompute {
            handle: RecomputeHandle {
                checkpoint_id: self.id,
                ordinal,
                original_tensor_id,
            },
            site,
        }
    }

    pub fn record_recomputed_save(&mut self, t: &Tensor, site: SaveSite) {
        let ordinal = self.recompute_cursor;
        self.recompute_cursor += 1;

        if ordinal < self.original_saves.len() {
            let expected = &self.original_saves[ordinal];
            assert_eq!(
                expected.site.name, site.name,
                "checkpoint recompute save-site mismatch at ordinal {}: expected {:?}, got {:?}",
                ordinal, expected.site.name, site.name
            );
        }

        self.recompute_cache
            .insert(ordinal, t.inner.borrow().value.clone());
    }
}

thread_local! {
    static CHECKPOINT_STACK: RefCell<Vec<Rc<RefCell<CheckpointContext>>>> =
        const { RefCell::new(vec![]) };

    static CHECKPOINT_REGISTRY: RefCell<HashMap<usize, Rc<RefCell<CheckpointContext>>>> =
        RefCell::new(HashMap::new());
}

fn lookup_checkpoint(id: usize) -> Rc<RefCell<CheckpointContext>> {
    CHECKPOINT_REGISTRY.with(|reg| {
        reg.borrow()
            .get(&id)
            .cloned()
            .unwrap_or_else(|| panic!("checkpoint id {} not found in registry", id))
    })
}

pub fn is_recording_recompute_saves() -> bool {
    CHECKPOINT_STACK.with(|stack| {
        stack
            .borrow()
            .last()
            .map(|c| c.borrow().is_recomputing)
            .unwrap_or(false)
    })
}

pub fn checkpoint<F>(
    name: impl Into<String>,
    policy: Rc<dyn CheckpointPolicy>,
    inputs: &[Tensor],
    f: F,
) -> Tensor
where
    F: Fn(Vec<Tensor>) -> Tensor + 'static,
{
    let id = fresh_checkpoint_id();
    let name: String = name.into();

    let input_values: Vec<TensorValue> = inputs
        .iter()
        .map(|t| t.inner.borrow().value.clone())
        .collect();

    let run: Rc<dyn Fn(Vec<Tensor>) -> Tensor> = Rc::new(f);

    let ctx = Rc::new(RefCell::new(CheckpointContext {
        id,
        name: name.clone(),
        input_values,
        run: run.clone(),
        policy,
        original_saves: vec![],
        recompute_cache: HashMap::new(),
        is_recomputing: false,
        recompute_cursor: 0,
    }));

    CHECKPOINT_REGISTRY.with(|reg| {
        reg.borrow_mut().insert(id, ctx.clone());
    });

    // Record enter event via global debug
    crate::debug::record_op_call_global_checkpoint_enter(id, &name);

    CHECKPOINT_STACK.with(|stack| {
        stack.borrow_mut().push(ctx);
    });

    let out = run(inputs.to_vec());

    CHECKPOINT_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });

    crate::debug::record_op_call_global_checkpoint_exit(id);

    out
}

pub fn save_tensor_through_current_policy(t: &Tensor, site: SaveSite) -> SavedTensor {
    let debug = crate::debug::get_global_recorder();

    CHECKPOINT_STACK.with(|stack| {
        let current = stack.borrow().last().cloned();

        match current {
            None => default_save(t, site, &debug),

            Some(ctx_ref) => {
                let is_recomputing = ctx_ref.borrow().is_recomputing;

                if is_recomputing {
                    // Only record saves that were originally tracked in original_saves:
                    // those whose policy decision was Recompute AND role is not Parameter.
                    let decision = ctx_ref.borrow().policy.decide(&site);
                    let should_record = matches!(
                        decision,
                        CheckpointDecision::PreferRecompute | CheckpointDecision::MustRecompute
                    ) && site.role != SaveRole::Parameter;

                    if should_record {
                        ctx_ref.borrow_mut().record_recomputed_save(t, site.clone());
                    }
                    return default_save(t, site, &debug);
                }

                let decision = ctx_ref.borrow().policy.decide(&site);

                match decision {
                    CheckpointDecision::MustSave | CheckpointDecision::PreferSave => {
                        default_save(t, site, &debug)
                    }
                    CheckpointDecision::PreferRecompute | CheckpointDecision::MustRecompute => {
                        if site.role == SaveRole::Parameter {
                            default_save(t, site, &debug)
                        } else {
                            ctx_ref
                                .borrow_mut()
                                .record_original_save(t, site, decision, &debug)
                        }
                    }
                }
            }
        }
    })
}

pub fn recompute_and_get(handle: RecomputeHandle, debug: &DebugRecorder) -> TensorValue {
    let ctx_ref = lookup_checkpoint(handle.checkpoint_id);

    let cache_hit = ctx_ref
        .borrow()
        .recompute_cache
        .contains_key(&handle.ordinal);

    if !cache_hit {
        do_recompute(ctx_ref.clone(), debug);
    }

    let result = ctx_ref
        .borrow()
        .recompute_cache
        .get(&handle.ordinal)
        .expect("recompute did not regenerate requested ordinal")
        .clone();
    result
}

fn do_recompute(ctx_ref: Rc<RefCell<CheckpointContext>>, debug: &DebugRecorder) {
    let (id, name, input_values, run) = {
        let mut ctx = ctx_ref.borrow_mut();
        ctx.recompute_cache.clear();
        ctx.recompute_cursor = 0;
        ctx.is_recomputing = true;
        (
            ctx.id,
            ctx.name.clone(),
            ctx.input_values.clone(),
            ctx.run.clone(),
        )
    };

    debug.record_recompute_start(id, &name);

    let _no_grad = crate::grad_mode::NoGradGuard::new();

    let inputs: Vec<Tensor> = input_values
        .iter()
        .cloned()
        .map(Tensor::from_value_no_grad)
        .collect();

    CHECKPOINT_STACK.with(|stack| {
        stack.borrow_mut().push(ctx_ref.clone());
    });

    let _ = run(inputs);

    CHECKPOINT_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });

    let saved_count = {
        let mut ctx = ctx_ref.borrow_mut();
        ctx.is_recomputing = false;
        ctx.recompute_cache.len()
    };

    debug.record_recompute_end(id, saved_count);
}
