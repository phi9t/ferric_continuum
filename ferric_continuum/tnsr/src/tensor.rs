//! Tensor data structures and storage.
//!
//! Book reference: Ch.4 — activations have shape `[B,T,D]` (batch × seq × d_model),
//! weight matrices have shape `[D_in, D_out]`.
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! `Tensor` is an `Rc<RefCell<TensorInner>>` — single-threaded, interior-mutable.
//! `TensorValue` stores f32 data behind an `Rc<Vec<f32>>` for cheap shallow clones.
//! `Shape` is a newtype over `Vec<usize>` with `numel()` and `bytes_f32()` helpers.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn fresh_tensor_id() -> TensorId {
    TensorId(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn bytes_f32(&self) -> usize {
        self.numel() * 4
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

#[derive(Clone)]
pub struct TensorValue {
    pub shape: Shape,
    pub data: Rc<Vec<f32>>,
}

impl TensorValue {
    pub fn zeros(shape: Shape) -> Self {
        let n = shape.numel();
        Self {
            shape,
            data: Rc::new(vec![0.0_f32; n]),
        }
    }

    pub fn from_vec(shape: Shape, data: Vec<f32>) -> Self {
        assert_eq!(shape.numel(), data.len());
        Self {
            shape,
            data: Rc::new(data),
        }
    }

    pub fn bytes(&self) -> usize {
        self.shape.bytes_f32()
    }

    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        Rc::make_mut(&mut self.data)
    }
}

pub struct AutogradMeta {
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub grad: Option<TensorValue>,
    pub producer: Option<crate::autograd::OpCallRef>,
}

pub struct TensorInner {
    pub id: TensorId,
    pub value: TensorValue,
    pub version: u64,
    pub autograd: AutogradMeta,
}

#[derive(Clone)]
pub struct Tensor {
    pub inner: Rc<RefCell<TensorInner>>,
}

impl Tensor {
    pub fn from_value(value: TensorValue, requires_grad: bool) -> Self {
        let id = fresh_tensor_id();
        Tensor {
            inner: Rc::new(RefCell::new(TensorInner {
                id,
                value,
                version: 0,
                autograd: AutogradMeta {
                    requires_grad,
                    is_leaf: true,
                    grad: None,
                    producer: None,
                },
            })),
        }
    }

    pub fn from_value_no_grad(value: TensorValue) -> Self {
        Self::from_value(value, false)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let shape = Shape(shape.to_vec());
        Self::from_value(TensorValue::zeros(shape), false)
    }

    pub fn randn(shape: &[usize]) -> Self {
        let s = Shape(shape.to_vec());
        let n = s.numel();
        let data = xorshift_normal(n);
        Self::from_value(TensorValue::from_vec(s, data), false)
    }

    pub fn randn_scaled(shape: &[usize], scale: f32) -> Self {
        let s = Shape(shape.to_vec());
        let n = s.numel();
        let data: Vec<f32> = xorshift_normal(n).into_iter().map(|v| v * scale).collect();
        Self::from_value(TensorValue::from_vec(s, data), false)
    }

    pub fn requires_grad(self) -> Self {
        self.inner.borrow_mut().autograd.requires_grad = true;
        self
    }

    pub fn set_requires_grad(&self, req: bool) {
        self.inner.borrow_mut().autograd.requires_grad = req;
    }

    pub fn grad(&self) -> Option<TensorValue> {
        self.inner.borrow().autograd.grad.clone()
    }

    pub fn grad_stats(&self) -> Option<TensorStats> {
        let inner = self.inner.borrow();
        inner.autograd.grad.as_ref().map(tensor_value_stats)
    }

    pub fn shape(&self) -> Shape {
        self.inner.borrow().value.shape.clone()
    }

    pub fn id(&self) -> TensorId {
        self.inner.borrow().id
    }

    pub fn grad_target(&self) -> crate::autograd::GradTarget {
        let inner = self.inner.borrow();
        crate::autograd::GradTarget {
            id: inner.id,
            shape: inner.value.shape.clone(),
            requires_grad: inner.autograd.requires_grad,
            producer: inner.autograd.producer.clone(),
            leaf: if inner.autograd.is_leaf {
                Some(Rc::downgrade(&self.inner))
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub nan_count: usize,
    pub inf_count: usize,
    pub numel: usize,
}

pub fn tensor_value_stats(v: &TensorValue) -> TensorStats {
    let data = v.data.as_ref();
    let n = data.len();
    if n == 0 {
        return TensorStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            nan_count: 0,
            inf_count: 0,
            numel: 0,
        };
    }
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut sum = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in data {
        if x.is_nan() {
            nan_count += 1;
        } else if x.is_infinite() {
            inf_count += 1;
        } else {
            sum += x as f64;
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }
    }
    let mean = (sum / n as f64) as f32;
    let mut var_sum = 0.0f64;
    for &x in data {
        if x.is_finite() {
            let d = x as f64 - mean as f64;
            var_sum += d * d;
        }
    }
    let std = (var_sum / n as f64).sqrt() as f32;
    TensorStats {
        min,
        max,
        mean,
        std,
        nan_count,
        inf_count,
        numel: n,
    }
}

fn xorshift_normal(n: usize) -> Vec<f32> {
    use std::cell::Cell;
    thread_local! {
        static RNG: Cell<u64> = const { Cell::new(0x_dead_beef_cafe_1234) };
    }

    fn next(rng: &Cell<u64>) -> u64 {
        let mut x = rng.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        rng.set(x);
        x
    }

    fn to_unit(x: u64) -> f32 {
        // upper 24 bits -> [0, 1)
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    let mut data = Vec::with_capacity(n);
    RNG.with(|rng| {
        let mut i = 0;
        while i < n {
            let u1 = to_unit(next(rng)).max(1e-7);
            let u2 = to_unit(next(rng));
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();
            data.push(z0);
            if i + 1 < n {
                data.push(z1);
            }
            i += 2;
        }
        data.truncate(n);
    });
    data
}
