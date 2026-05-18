use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::autograd::OpKind;
use crate::debug::DebugRecorder;
use crate::tensor::{Shape, Tensor, TensorId, TensorInner, TensorValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveRole {
    Activation,
    Parameter,
    AuxStat,
    BoundaryInput,
    RngState,
}

#[derive(Debug, Clone)]
pub struct SaveSite {
    pub op: OpKind,
    pub name: String,
    pub role: SaveRole,
    pub shape: Shape,
    pub bytes: usize,
}

impl SaveSite {
    pub fn new(op: OpKind, name: impl Into<String>, role: SaveRole, t: &Tensor) -> Self {
        let shape = t.inner.borrow().value.shape.clone();
        let bytes = shape.bytes_f32();
        Self {
            op,
            name: name.into(),
            role,
            shape,
            bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecomputeHandle {
    pub checkpoint_id: usize,
    pub ordinal: usize,
    pub original_tensor_id: TensorId,
}

#[derive(Clone)]
pub enum SavedTensor {
    Materialized {
        value: TensorValue,
        site: SaveSite,
    },
    Borrowed {
        tensor: Weak<RefCell<TensorInner>>,
        version: u64,
        site: SaveSite,
    },
    Recompute {
        handle: RecomputeHandle,
        site: SaveSite,
    },
}

impl SavedTensor {
    pub fn save(t: &Tensor, site: SaveSite) -> Self {
        crate::checkpoint::save_tensor_through_current_policy(t, site)
    }

    pub fn unpack(&self, debug: &DebugRecorder) -> TensorValue {
        match self {
            SavedTensor::Materialized { value, site } => {
                debug.record_saved_unpack_materialized(site);
                value.clone()
            }
            SavedTensor::Borrowed {
                tensor,
                version,
                site,
            } => {
                debug.record_saved_unpack_borrowed(site);
                let upgraded = tensor
                    .upgrade()
                    .expect("borrowed saved tensor was dropped before backward");
                let inner = upgraded.borrow();
                assert_eq!(
                    inner.version, *version,
                    "tensor version changed before backward at save site {:?}",
                    site.name
                );
                inner.value.clone()
            }
            SavedTensor::Recompute { handle, site } => {
                debug.record_saved_unpack_recompute(site);
                crate::checkpoint::recompute_and_get(*handle, debug)
            }
        }
    }
}

pub fn default_save(t: &Tensor, site: SaveSite, debug: &DebugRecorder) -> SavedTensor {
    match site.role {
        SaveRole::Parameter => {
            let inner = t.inner.borrow();
            debug.record_save_borrowed(&site);
            SavedTensor::Borrowed {
                tensor: Rc::downgrade(&t.inner),
                version: inner.version,
                site,
            }
        }
        _ => {
            debug.record_save_materialized(&site);
            SavedTensor::Materialized {
                value: t.inner.borrow().value.clone(),
                site,
            }
        }
    }
}
