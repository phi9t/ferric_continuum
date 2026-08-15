//! Generic CPU inference helpers.
//!
//! These are small, model-agnostic primitives for decode-time bookkeeping:
//! RoPE over raw `[B,T,D]` query/key values, per-sequence KV cache storage, and
//! deterministic continuous batching. Qwen3's differentiable training path
//! continues to use `ops::rope` and `ops::gqa`.

use std::collections::{HashMap, VecDeque};

use crate::tensor::{Shape, TensorValue};

/// RoPE configuration for one attention head.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeConfig {
    dim: usize,
    base: f32,
    inv_freq: Vec<f32>,
}

impl RopeConfig {
    pub fn new(dim: usize, base: f32) -> Self {
        assert!(dim > 0, "rope dim must be non-zero");
        assert_eq!(dim % 2, 0, "rope dim must be even");
        assert!(base > 1.0, "rope base must be greater than one");
        let inv_freq = (0..dim / 2)
            .map(|i| base.powf(-((2 * i) as f32) / dim as f32))
            .collect();
        Self {
            dim,
            base,
            inv_freq,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

/// Apply RoPE to `[B,T,D]` query or key values using one absolute position per
/// token.
pub fn apply_rope(input: &TensorValue, positions: &[usize], cfg: &RopeConfig) -> TensorValue {
    let shape = &input.shape.0;
    assert_eq!(shape.len(), 3, "apply_rope input must be [B,T,D]");
    let (batch, time, dim) = (shape[0], shape[1], shape[2]);
    assert_eq!(dim, cfg.dim, "apply_rope input dim must match config");
    assert_eq!(
        positions.len(),
        batch * time,
        "apply_rope positions must have one entry per [B,T] token"
    );

    let mut out = input.data.as_ref().clone();
    for b in 0..batch {
        for t in 0..time {
            let pos = positions[b * time + t];
            let offset = (b * time + t) * dim;
            for pair in 0..dim / 2 {
                let even = out[offset + 2 * pair];
                let odd = out[offset + 2 * pair + 1];
                let angle = pos as f32 * cfg.inv_freq[pair];
                let (sin, cos) = angle.sin_cos();
                out[offset + 2 * pair] = even * cos - odd * sin;
                out[offset + 2 * pair + 1] = even * sin + odd * cos;
            }
        }
    }
    TensorValue::from_vec(input.shape.clone(), out)
}

#[derive(Clone)]
pub struct KvCacheView {
    pub sequence_id: u64,
    pub len: usize,
    pub keys: TensorValue,
    pub values: TensorValue,
}

#[derive(Clone, Debug, Default)]
struct SequenceCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

/// Per-sequence KV cache for CPU inference.
#[derive(Clone, Debug)]
pub struct KvCache {
    head_dim: usize,
    sequences: HashMap<u64, SequenceCache>,
}

impl KvCache {
    pub fn new(head_dim: usize) -> Self {
        assert!(head_dim > 0, "head_dim must be non-zero");
        Self {
            head_dim,
            sequences: HashMap::new(),
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn append(&mut self, sequence_id: u64, keys: TensorValue, values: TensorValue) {
        self.append_at(sequence_id, self.position(sequence_id), keys, values);
    }

    pub fn append_at(
        &mut self,
        sequence_id: u64,
        expected_position: usize,
        keys: TensorValue,
        values: TensorValue,
    ) {
        self.validate_chunk(&keys.shape, "keys");
        self.validate_chunk(&values.shape, "values");
        assert_eq!(
            keys.shape, values.shape,
            "key/value chunk shapes must match"
        );
        assert_eq!(
            self.position(sequence_id),
            expected_position,
            "kv cache append position mismatch"
        );
        let len = keys.shape.0[0];
        let seq = self.sequences.entry(sequence_id).or_default();
        seq.keys.extend(keys.data.iter().copied());
        seq.values.extend(values.data.iter().copied());
        seq.len += len;
    }

    pub fn prefix(&self, sequence_id: u64) -> Option<KvCacheView> {
        let seq = self.sequences.get(&sequence_id)?;
        Some(KvCacheView {
            sequence_id,
            len: seq.len,
            keys: TensorValue::from_vec(Shape(vec![seq.len, self.head_dim]), seq.keys.clone()),
            values: TensorValue::from_vec(Shape(vec![seq.len, self.head_dim]), seq.values.clone()),
        })
    }

    pub fn position(&self, sequence_id: u64) -> usize {
        self.sequences.get(&sequence_id).map_or(0, |seq| seq.len)
    }

    pub fn remove(&mut self, sequence_id: u64) -> Option<KvCacheView> {
        let seq = self.sequences.remove(&sequence_id)?;
        Some(KvCacheView {
            sequence_id,
            len: seq.len,
            keys: TensorValue::from_vec(Shape(vec![seq.len, self.head_dim]), seq.keys),
            values: TensorValue::from_vec(Shape(vec![seq.len, self.head_dim]), seq.values),
        })
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    fn validate_chunk(&self, shape: &Shape, name: &str) {
        assert_eq!(shape.0.len(), 2, "{name} chunk must be [T,D]");
        assert!(shape.0[0] > 0, "{name} chunk must contain at least one token");
        assert_eq!(
            shape.0[1], self.head_dim,
            "{name} dim must match cache head_dim"
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeRequest {
    pub request_id: u64,
    pub prompt: Vec<u32>,
    pub max_len: usize,
}

impl DecodeRequest {
    pub fn new(request_id: u64, prompt: Vec<u32>, max_len: usize) -> Self {
        assert!(
            !prompt.is_empty(),
            "decode request prompt must be non-empty"
        );
        assert!(
            max_len >= prompt.len(),
            "decode request max_len must be at least prompt length"
        );
        Self {
            request_id,
            prompt,
            max_len,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttentionStep {
    Prefill {
        request_id: u64,
        token: u32,
        position: usize,
    },
    Decode {
        request_id: u64,
        token: u32,
        position: usize,
    },
}

impl AttentionStep {
    pub fn request_id(&self) -> u64 {
        match self {
            Self::Prefill { request_id, .. } | Self::Decode { request_id, .. } => *request_id,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchStats {
    pub queued: usize,
    pub active: usize,
    pub capacity: usize,
}

#[derive(Clone, Debug)]
struct ActiveRequest {
    request: DecodeRequest,
    next_prompt: usize,
    generated: usize,
    next_decode: Option<u32>,
}

impl ActiveRequest {
    fn position(&self) -> usize {
        self.next_prompt + self.generated
    }

    fn is_finished(&self) -> bool {
        self.position() >= self.request.max_len
    }
}

/// Deterministic continuous batcher for single-process CPU decoding.
#[derive(Clone, Debug)]
pub struct ContinuousBatcher {
    capacity: usize,
    queued: VecDeque<DecodeRequest>,
    active: Vec<ActiveRequest>,
}

impl ContinuousBatcher {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "batch capacity must be non-zero");
        Self {
            capacity,
            queued: VecDeque::new(),
            active: Vec::new(),
        }
    }

    pub fn enqueue(&mut self, request: DecodeRequest) {
        assert!(
            !self.contains(request.request_id),
            "duplicate request id {}",
            request.request_id
        );
        self.queued.push_back(request);
    }

    pub fn next_batch(&mut self) -> Vec<AttentionStep> {
        self.admit_queued();
        let mut steps = Vec::with_capacity(self.capacity);

        for req in &mut self.active {
            while steps.len() < self.capacity && req.next_prompt < req.request.prompt.len() {
                let position = req.next_prompt;
                let token = req.request.prompt[position];
                req.next_prompt += 1;
                steps.push(AttentionStep::Prefill {
                    request_id: req.request.request_id,
                    token,
                    position,
                });
            }
            if steps.len() == self.capacity {
                return steps;
            }
        }

        self.active.retain(|req| !req.is_finished());
        for req in &mut self.active {
            if steps.len() == self.capacity {
                break;
            }
            if req.next_prompt < req.request.prompt.len() || req.is_finished() {
                continue;
            }
            if let Some(token) = req.next_decode.take() {
                let position = req.position();
                req.generated += 1;
                steps.push(AttentionStep::Decode {
                    request_id: req.request.request_id,
                    token,
                    position,
                });
            }
        }
        steps
    }

    pub fn finish_step(&mut self, request_id: u64, next_token: Option<u32>) {
        let Some(index) = self
            .active
            .iter()
            .position(|req| req.request.request_id == request_id)
        else {
            panic!("unknown active request id {request_id}");
        };

        if let Some(token) = next_token {
            assert!(
                !self.active[index].is_finished(),
                "request {request_id} already reached max_len"
            );
            assert!(
                self.active[index].next_decode.is_none(),
                "request {request_id} already has a pending decode token"
            );
            self.active[index].next_decode = Some(token);
        } else {
            self.active.remove(index);
            self.admit_queued();
        }
    }

    pub fn stats(&self) -> BatchStats {
        BatchStats {
            queued: self.queued.len(),
            active: self.active.len(),
            capacity: self.capacity,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queued.is_empty() && self.active.is_empty()
    }

    fn admit_queued(&mut self) {
        while self.active.len() < self.capacity {
            let Some(request) = self.queued.pop_front() else {
                break;
            };
            self.active.push(ActiveRequest {
                request,
                next_prompt: 0,
                generated: 0,
                next_decode: None,
            });
        }
    }

    fn contains(&self, request_id: u64) -> bool {
        self.queued.iter().any(|req| req.request_id == request_id)
            || self
                .active
                .iter()
                .any(|req| req.request.request_id == request_id)
    }
}
