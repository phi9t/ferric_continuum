//! Semantic axis types layered over [`Tensor`].
//!
//! [`TypedTensor`] owns the same cheap tensor handle as the untyped API. Its
//! type parameter records what each runtime dimension *means* while the actual
//! extents remain dynamic. Attaching or erasing axes never copies tensor data
//! and never inserts an autograd operation.

use std::fmt;
use std::marker::PhantomData;

use crate::tensor::{Shape, Tensor, TensorId};

mod sealed {
    pub trait Axes {}
    pub trait SequenceKind {}
}

/// One semantic tensor dimension.
///
/// Downstream research code may define additional axes. Library operations
/// still accept exact transformer layouts, so a custom axis cannot accidentally
/// satisfy (for example) a query-head argument.
pub trait Axis: 'static {
    /// Stable machine-facing identifier. Wave C may use this at dynamic
    /// interchange boundaries.
    fn stable_id() -> String;

    /// Human-facing name used by [`TypedTensor::named_shape`].
    fn display_name() -> String {
        Self::stable_id()
    }
}

/// A supported ordered list of semantic axes.
pub trait Axes: sealed::Axes + 'static {
    /// Number of physical runtime dimensions in this ordered axis list.
    const RANK: usize;

    /// Ordered human-facing axis names.
    fn names() -> Vec<String>;
}

/// Marker implemented only by the complete- and sharded-sequence kinds.
pub trait SequenceKind: sealed::SequenceKind + 'static {
    /// Machine-facing identifier for this sequence ownership kind.
    fn stable_id() -> &'static str;

    /// Human-facing label for this sequence ownership kind.
    fn display_name() -> &'static str;
}

/// A complete sequence owned by one logical attention invocation.
pub enum Full {}

/// One equal contiguous context-parallel sequence shard.
pub enum Shard {}

impl sealed::SequenceKind for Full {}
impl sealed::SequenceKind for Shard {}

impl SequenceKind for Full {
    fn stable_id() -> &'static str {
        "sequence"
    }

    fn display_name() -> &'static str {
        "sequence"
    }
}

impl SequenceKind for Shard {
    fn stable_id() -> &'static str {
        "local_sequence"
    }

    fn display_name() -> &'static str {
        "local_sequence"
    }
}

/// The sequence axis refined by an ownership kind.
pub struct Sequence<K>(PhantomData<fn() -> K>);

impl<K: SequenceKind> Axis for Sequence<K> {
    fn stable_id() -> String {
        K::stable_id().to_owned()
    }

    fn display_name() -> String {
        K::display_name().to_owned()
    }
}

/// Semantic sequence axis for a complete sequence.
pub type FullSequence = Sequence<Full>;

/// Semantic sequence axis for one equal contiguous context shard.
pub type ShardSequence = Sequence<Shard>;

/// Batch examples processed independently.
pub enum Batch {}

/// Model hidden-feature dimension.
pub enum Hidden {}

/// Query-head dimension in grouped-query attention.
pub enum QueryHead {}

/// Shared key/value-head dimension in grouped-query attention.
pub enum KvHead {}

/// Feature dimension within one attention head.
pub enum HeadDim {}

impl Axis for Batch {
    fn stable_id() -> String {
        "batch".to_owned()
    }
}

impl Axis for Hidden {
    fn stable_id() -> String {
        "hidden".to_owned()
    }
}

impl Axis for QueryHead {
    fn stable_id() -> String {
        "query_head".to_owned()
    }
}

impl Axis for KvHead {
    fn stable_id() -> String {
        "kv_head".to_owned()
    }
}

impl Axis for HeadDim {
    fn stable_id() -> String {
        "head_dim".to_owned()
    }
}

/// Two semantic axes represented by one physical runtime dimension.
pub struct Merged<A, B>(PhantomData<fn() -> (A, B)>);

impl<A: Axis, B: Axis> Axis for Merged<A, B> {
    fn stable_id() -> String {
        format!("merged({},{})", A::stable_id(), B::stable_id())
    }

    fn display_name() -> String {
        format!("{}*{}", A::display_name(), B::display_name())
    }
}

/// A runtime extent whose semantic axis is carried by the Rust type.
///
/// `AxisExtent<QueryHead>` and `AxisExtent<KvHead>` have the same compact
/// runtime representation, but they cannot be interchanged accidentally at an
/// operation seam.
pub struct AxisExtent<A: Axis> {
    value: usize,
    axis: PhantomData<fn() -> A>,
}

impl<A: Axis> AxisExtent<A> {
    /// Attach semantic meaning to a validated runtime extent.
    pub const fn new(value: usize) -> Self {
        Self {
            value,
            axis: PhantomData,
        }
    }

    /// Return the dynamic extent.
    pub const fn get(self) -> usize {
        self.value
    }

    /// Merge two semantic axes represented by one physical dimension.
    pub fn checked_merge<B: Axis>(self, rhs: AxisExtent<B>) -> Option<AxisExtent<Merged<A, B>>> {
        Shape::checked_product(&[self.value, rhs.value]).map(AxisExtent::new)
    }
}

impl<A: Axis> From<usize> for AxisExtent<A> {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl<A: Axis> Copy for AxisExtent<A> {}

impl<A: Axis> Clone for AxisExtent<A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: Axis> PartialEq for AxisExtent<A> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<A: Axis> Eq for AxisExtent<A> {}

impl<A: Axis> fmt::Debug for AxisExtent<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(&format!("AxisExtent<{}>", A::stable_id()))
            .field(&self.value)
            .finish()
    }
}

/// Ordered semantic axis list for a rank-one tensor.
pub struct Axes1<A>(PhantomData<fn() -> A>);

/// Ordered semantic axis list for a rank-two tensor.
pub struct Axes2<A, B>(PhantomData<fn() -> (A, B)>);

/// Ordered semantic axis list for a rank-three tensor.
pub struct Axes3<A, B, C>(PhantomData<fn() -> (A, B, C)>);

/// Ordered semantic axis list for a rank-four tensor.
pub struct Axes4<A, B, C, D>(PhantomData<fn() -> (A, B, C, D)>);

impl<A: Axis> sealed::Axes for Axes1<A> {}
impl<A: Axis, B: Axis> sealed::Axes for Axes2<A, B> {}
impl<A: Axis, B: Axis, C: Axis> sealed::Axes for Axes3<A, B, C> {}
impl<A: Axis, B: Axis, C: Axis, D: Axis> sealed::Axes for Axes4<A, B, C, D> {}

impl<A: Axis> Axes for Axes1<A> {
    const RANK: usize = 1;

    fn names() -> Vec<String> {
        vec![A::display_name()]
    }
}

impl<A: Axis, B: Axis> Axes for Axes2<A, B> {
    const RANK: usize = 2;

    fn names() -> Vec<String> {
        vec![A::display_name(), B::display_name()]
    }
}

impl<A: Axis, B: Axis, C: Axis> Axes for Axes3<A, B, C> {
    const RANK: usize = 3;

    fn names() -> Vec<String> {
        vec![A::display_name(), B::display_name(), C::display_name()]
    }
}

impl<A: Axis, B: Axis, C: Axis, D: Axis> Axes for Axes4<A, B, C, D> {
    const RANK: usize = 4;

    fn names() -> Vec<String> {
        vec![
            A::display_name(),
            B::display_name(),
            C::display_name(),
            D::display_name(),
        ]
    }
}

/// A runtime tensor did not have the rank required by its declared axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisError {
    /// Rank implied by the requested semantic axis list.
    pub expected_rank: usize,
    /// Rank found on the supplied runtime tensor.
    pub actual_rank: usize,
}

impl fmt::Display for AxisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "typed tensor expected rank {} but received rank {}",
            self.expected_rank, self.actual_rank
        )
    }
}

impl std::error::Error for AxisError {}

/// A [`Tensor`] handle carrying an ordered semantic axis list in its Rust type.
///
/// `from_tensor` checks rank, but the caller supplies the meaning of each axis.
/// Once declared, operation-specific signatures preserve or transform those
/// meanings at compile time.
pub struct TypedTensor<A: Axes> {
    tensor: Tensor,
    axes: PhantomData<fn() -> A>,
}

impl<A: Axes> TypedTensor<A> {
    /// Attach caller-declared semantic axes after checking the runtime rank.
    ///
    /// Shape alone cannot infer axis meaning: choosing `A` is the caller's
    /// semantic assertion.
    pub fn from_tensor(tensor: Tensor) -> Result<Self, AxisError> {
        let actual_rank = tensor.rank();
        if actual_rank != A::RANK {
            return Err(AxisError {
                expected_rank: A::RANK,
                actual_rank,
            });
        }
        Ok(Self::from_proven_axes(tensor))
    }

    /// Tag an operation output or already-validated compatibility input.
    pub(crate) fn from_proven_axes(tensor: Tensor) -> Self {
        Self {
            tensor,
            axes: PhantomData,
        }
    }

    /// Borrow the exact underlying untyped tensor handle.
    pub fn as_tensor(&self) -> &Tensor {
        self.assert_current_rank();
        &self.tensor
    }

    /// Erase semantic axes and return the exact underlying tensor handle.
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Return a clone of the underlying runtime shape.
    pub fn shape(&self) -> Shape {
        self.assert_current_rank();
        self.tensor.shape()
    }

    /// Return the identity of the underlying tensor.
    pub fn id(&self) -> TensorId {
        self.tensor.id()
    }

    /// Pair ordered semantic display names with current runtime extents.
    pub fn named_shape(&self) -> Vec<(String, usize)> {
        A::names().into_iter().zip(self.shape().0).collect()
    }

    fn extent_at<X: Axis>(&self, index: usize) -> AxisExtent<X> {
        self.assert_current_rank();
        AxisExtent::new(self.tensor.dim(index))
    }

    fn assert_current_rank(&self) {
        let actual_rank = self.tensor.rank();
        assert_eq!(
            actual_rank,
            A::RANK,
            "typed tensor rank changed after axis attachment: expected rank {} but received rank {}",
            A::RANK,
            actual_rank
        );
    }
}

impl<A: Axes> Clone for TypedTensor<A> {
    fn clone(&self) -> Self {
        Self::from_proven_axes(self.tensor.clone())
    }
}

/// Hidden states `[Batch, Sequence<S>, Hidden]`.
pub type HiddenStates<S> = TypedTensor<Axes3<Batch, Sequence<S>, Hidden>>;

/// Projected queries `[Batch, Sequence<S>, QueryHead*HeadDim]`.
pub type ProjectedQueries<S> = TypedTensor<Axes3<Batch, Sequence<S>, Merged<QueryHead, HeadDim>>>;

/// Projected keys or values `[Batch, Sequence<S>, KvHead*HeadDim]`.
pub type ProjectedKv<S> = TypedTensor<Axes3<Batch, Sequence<S>, Merged<KvHead, HeadDim>>>;

/// Split queries `[Batch, Sequence<S>, QueryHead, HeadDim]`.
pub type QueryHeads<S> = TypedTensor<Axes4<Batch, Sequence<S>, QueryHead, HeadDim>>;

/// Split keys or values `[Batch, Sequence<S>, KvHead, HeadDim]`.
pub type KvHeads<S> = TypedTensor<Axes4<Batch, Sequence<S>, KvHead, HeadDim>>;

/// Query projection weight `[Hidden, QueryHead*HeadDim]`.
pub type QueryProjectionWeight = TypedTensor<Axes2<Hidden, Merged<QueryHead, HeadDim>>>;

/// Key or value projection weight `[Hidden, KvHead*HeadDim]`.
pub type KvProjectionWeight = TypedTensor<Axes2<Hidden, Merged<KvHead, HeadDim>>>;

/// Attention output weight `[QueryHead*HeadDim, Hidden]`.
pub type OutputProjectionWeight = TypedTensor<Axes2<Merged<QueryHead, HeadDim>, Hidden>>;

/// Per-head RMSNorm scale `[HeadDim]`.
pub type HeadScale = TypedTensor<Axes1<HeadDim>>;

/// Complete-sequence hidden states `[Batch, Sequence, Hidden]`.
pub type FullHiddenStates = HiddenStates<Full>;

/// Sequence-sharded hidden states `[Batch, LocalSequence, Hidden]`.
pub type ShardHiddenStates = HiddenStates<Shard>;

/// Complete-sequence query heads `[Batch, Sequence, QueryHead, HeadDim]`.
pub type FullQueryHeads = QueryHeads<Full>;

/// Complete-sequence KV heads `[Batch, Sequence, KvHead, HeadDim]`.
pub type FullKvHeads = KvHeads<Full>;

/// Sequence-sharded query heads `[Batch, LocalSequence, QueryHead, HeadDim]`.
pub type ShardQueryHeads = QueryHeads<Shard>;

/// Sequence-sharded KV heads `[Batch, LocalSequence, KvHead, HeadDim]`.
pub type ShardKvHeads = KvHeads<Shard>;

impl<S: SequenceKind> TypedTensor<Axes3<Batch, Sequence<S>, Hidden>> {
    /// Return the batch extent.
    pub fn batch_extent(&self) -> AxisExtent<Batch> {
        self.extent_at(0)
    }

    /// Return the complete- or local-sequence extent encoded by `S`.
    pub fn sequence_extent(&self) -> AxisExtent<Sequence<S>> {
        self.extent_at(1)
    }

    /// Return the model hidden-feature extent.
    pub fn hidden_extent(&self) -> AxisExtent<Hidden> {
        self.extent_at(2)
    }
}

impl<S: SequenceKind> TypedTensor<Axes3<Batch, Sequence<S>, Merged<QueryHead, HeadDim>>> {
    /// Return the batch extent.
    pub fn batch_extent(&self) -> AxisExtent<Batch> {
        self.extent_at(0)
    }

    /// Return the complete- or local-sequence extent encoded by `S`.
    pub fn sequence_extent(&self) -> AxisExtent<Sequence<S>> {
        self.extent_at(1)
    }

    /// Return the merged query-head-by-head-feature extent.
    pub fn flattened_query_extent(&self) -> AxisExtent<Merged<QueryHead, HeadDim>> {
        self.extent_at(2)
    }
}

impl<S: SequenceKind> TypedTensor<Axes3<Batch, Sequence<S>, Merged<KvHead, HeadDim>>> {
    /// Return the batch extent.
    pub fn batch_extent(&self) -> AxisExtent<Batch> {
        self.extent_at(0)
    }

    /// Return the complete- or local-sequence extent encoded by `S`.
    pub fn sequence_extent(&self) -> AxisExtent<Sequence<S>> {
        self.extent_at(1)
    }

    /// Return the merged KV-head-by-head-feature extent.
    pub fn flattened_kv_extent(&self) -> AxisExtent<Merged<KvHead, HeadDim>> {
        self.extent_at(2)
    }
}

impl<S: SequenceKind> TypedTensor<Axes4<Batch, Sequence<S>, QueryHead, HeadDim>> {
    /// Return the batch extent.
    pub fn batch_extent(&self) -> AxisExtent<Batch> {
        self.extent_at(0)
    }

    /// Return the complete- or local-sequence extent encoded by `S`.
    pub fn sequence_extent(&self) -> AxisExtent<Sequence<S>> {
        self.extent_at(1)
    }

    /// Return the number of query heads.
    pub fn query_heads_extent(&self) -> AxisExtent<QueryHead> {
        self.extent_at(2)
    }

    /// Return the feature extent within each query head.
    pub fn head_dim_extent(&self) -> AxisExtent<HeadDim> {
        self.extent_at(3)
    }
}

impl<S: SequenceKind> TypedTensor<Axes4<Batch, Sequence<S>, KvHead, HeadDim>> {
    /// Return the batch extent.
    pub fn batch_extent(&self) -> AxisExtent<Batch> {
        self.extent_at(0)
    }

    /// Return the complete- or local-sequence extent encoded by `S`.
    pub fn sequence_extent(&self) -> AxisExtent<Sequence<S>> {
        self.extent_at(1)
    }

    /// Return the number of shared key/value heads.
    pub fn kv_heads_extent(&self) -> AxisExtent<KvHead> {
        self.extent_at(2)
    }

    /// Return the feature extent within each key/value head.
    pub fn head_dim_extent(&self) -> AxisExtent<HeadDim> {
        self.extent_at(3)
    }
}

impl TypedTensor<Axes2<Hidden, Merged<QueryHead, HeadDim>>> {
    /// Return the projection's input hidden-feature extent.
    pub fn hidden_extent(&self) -> AxisExtent<Hidden> {
        self.extent_at(0)
    }

    /// Return the projection's merged query-head output extent.
    pub fn flattened_query_extent(&self) -> AxisExtent<Merged<QueryHead, HeadDim>> {
        self.extent_at(1)
    }
}

impl TypedTensor<Axes2<Hidden, Merged<KvHead, HeadDim>>> {
    /// Return the projection's input hidden-feature extent.
    pub fn hidden_extent(&self) -> AxisExtent<Hidden> {
        self.extent_at(0)
    }

    /// Return the projection's merged KV-head output extent.
    pub fn flattened_kv_extent(&self) -> AxisExtent<Merged<KvHead, HeadDim>> {
        self.extent_at(1)
    }
}

impl TypedTensor<Axes2<Merged<QueryHead, HeadDim>, Hidden>> {
    /// Return the output projection's merged query-head input extent.
    pub fn flattened_query_extent(&self) -> AxisExtent<Merged<QueryHead, HeadDim>> {
        self.extent_at(0)
    }

    /// Return the output projection's hidden-feature output extent.
    pub fn hidden_extent(&self) -> AxisExtent<Hidden> {
        self.extent_at(1)
    }
}

impl TypedTensor<Axes1<HeadDim>> {
    /// Return the per-head normalization scale extent.
    pub fn head_dim_extent(&self) -> AxisExtent<HeadDim> {
        self.extent_at(0)
    }
}

#[cfg(doctest)]
/// Compile-time contracts for semantic transformer axes.
///
/// Axis order is nominal. This is the legal order:
///
/// ```no_run
/// use tnsr::{
///     tensor::Tensor,
///     typed::{Axes3, Batch, FullHiddenStates, FullSequence, Hidden, TypedTensor},
/// };
/// let hidden = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, 3])).unwrap();
/// let _: FullHiddenStates = hidden;
/// ```
///
/// Swapping sequence and hidden axes is a different type:
///
/// ```compile_fail
/// use tnsr::{
///     tensor::Tensor,
///     typed::{Axes3, Batch, FullHiddenStates, FullSequence, Hidden, TypedTensor},
/// };
/// let hidden = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, 3])).unwrap();
/// let _: TypedTensor<Axes3<Batch, Hidden, FullSequence>> = hidden;
/// ```
///
/// Typed GQA accepts query heads followed by KV heads:
///
/// ```no_run
/// use tnsr::{
///     ops::gqa,
///     tensor::Tensor,
///     typed::{FullKvHeads, FullQueryHeads},
/// };
/// let q = FullQueryHeads::from_tensor(Tensor::zeros(&[1, 2, 2, 2])).unwrap();
/// let k = FullKvHeads::from_tensor(Tensor::zeros(&[1, 2, 1, 2])).unwrap();
/// let v = FullKvHeads::from_tensor(Tensor::zeros(&[1, 2, 1, 2])).unwrap();
/// let _ = gqa::gqa_attention_typed(&q, &k, &v, "gqa");
/// ```
///
/// A query-head tensor cannot masquerade as K or V:
///
/// ```compile_fail
/// use tnsr::{
///     ops::gqa,
///     tensor::Tensor,
///     typed::{FullKvHeads, FullQueryHeads},
/// };
/// let q = FullQueryHeads::from_tensor(Tensor::zeros(&[1, 2, 2, 2])).unwrap();
/// let k = FullKvHeads::from_tensor(Tensor::zeros(&[1, 2, 1, 2])).unwrap();
/// let v = FullKvHeads::from_tensor(Tensor::zeros(&[1, 2, 1, 2])).unwrap();
/// let _ = gqa::gqa_attention_typed(&q, &q, &q, "gqa");
/// ```
///
/// Context-parallel GQA accepts sequence-shard axes:
///
/// ```no_run
/// use tnsr::{
///     ops::context_parallel_gqa,
///     tensor::Tensor,
///     typed::{FullKvHeads, FullQueryHeads, ShardKvHeads, ShardQueryHeads},
/// };
/// let q = ShardQueryHeads::from_tensor(Tensor::zeros(&[1, 1, 2, 2])).unwrap();
/// let k = ShardKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let v = ShardKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let full_q = FullQueryHeads::from_tensor(Tensor::zeros(&[1, 1, 2, 2])).unwrap();
/// let full_k = FullKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let full_v = FullKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let _ = context_parallel_gqa::context_parallel_gqa_attention_typed(
///     &[q], &[k], &[v], "cp",
/// );
/// ```
///
/// Full-sequence tensors cannot be supplied as context shards:
///
/// ```compile_fail
/// use tnsr::{
///     ops::context_parallel_gqa,
///     tensor::Tensor,
///     typed::{FullKvHeads, FullQueryHeads, ShardKvHeads, ShardQueryHeads},
/// };
/// let q = ShardQueryHeads::from_tensor(Tensor::zeros(&[1, 1, 2, 2])).unwrap();
/// let k = ShardKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let v = ShardKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let full_q = FullQueryHeads::from_tensor(Tensor::zeros(&[1, 1, 2, 2])).unwrap();
/// let full_k = FullKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let full_v = FullKvHeads::from_tensor(Tensor::zeros(&[1, 1, 1, 2])).unwrap();
/// let _ = context_parallel_gqa::context_parallel_gqa_attention_typed(
///     &[full_q], &[full_k], &[full_v], "cp",
/// );
/// ```
///
/// Query projection accepts its corresponding weight axes:
///
/// ```no_run
/// use tnsr::{
///     ops::linear,
///     tensor::Tensor,
///     typed::{FullHiddenStates, KvProjectionWeight, QueryProjectionWeight},
/// };
/// let x = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, 4])).unwrap();
/// let query_weight = QueryProjectionWeight::from_tensor(Tensor::zeros(&[4, 4])).unwrap();
/// let kv_weight = KvProjectionWeight::from_tensor(Tensor::zeros(&[4, 4])).unwrap();
/// let _ = linear::project_queries(&x, &query_weight, "q_proj");
/// ```
///
/// KV projection weights cannot be passed to query projection:
///
/// ```compile_fail
/// use tnsr::{
///     ops::linear,
///     tensor::Tensor,
///     typed::{FullHiddenStates, KvProjectionWeight, QueryProjectionWeight},
/// };
/// let x = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, 4])).unwrap();
/// let query_weight = QueryProjectionWeight::from_tensor(Tensor::zeros(&[4, 4])).unwrap();
/// let kv_weight = KvProjectionWeight::from_tensor(Tensor::zeros(&[4, 4])).unwrap();
/// let _ = linear::project_queries(&x, &kv_weight, "q_proj");
/// ```
///
/// Head splitting accepts evidence for the exact semantic axes:
///
/// ```no_run
/// use tnsr::{
///     ops::shape,
///     tensor::Tensor,
///     typed::{AxisExtent, Full, HeadDim, ProjectedQueries, QueryHead},
/// };
/// let q = ProjectedQueries::<Full>::from_tensor(Tensor::zeros(&[1, 2, 4])).unwrap();
/// let _ = shape::split_query_heads(
///     &q,
///     AxisExtent::<QueryHead>::new(2),
///     AxisExtent::<HeadDim>::new(2),
///     "q_split",
/// );
/// ```
///
/// KV-head evidence cannot select a query-head layout:
///
/// ```compile_fail
/// use tnsr::{
///     ops::shape,
///     tensor::Tensor,
///     typed::{AxisExtent, Full, HeadDim, KvHead, ProjectedQueries},
/// };
/// let q = ProjectedQueries::<Full>::from_tensor(Tensor::zeros(&[1, 2, 4])).unwrap();
/// let _ = shape::split_query_heads(
///     &q,
///     AxisExtent::<KvHead>::new(2),
///     AxisExtent::<HeadDim>::new(2),
///     "q_split",
/// );
/// ```
///
/// RoPE accepts explicit query-head axes:
///
/// ```no_run
/// use tnsr::{
///     ops::rope::{self, RopeConfig},
///     tensor::Tensor,
///     typed::{FullHiddenStates, FullQueryHeads},
/// };
/// let q = FullQueryHeads::from_tensor(Tensor::zeros(&[1, 2, 2, 2])).unwrap();
/// let x = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, 4])).unwrap();
/// let _ = rope::rotate_queries(&q, RopeConfig { base: 10_000.0, start_pos: 0 }, "q_rope");
/// ```
///
/// Hidden states have the wrong rank and axes for RoPE:
///
/// ```compile_fail
/// use tnsr::{
///     ops::rope::{self, RopeConfig},
///     tensor::Tensor,
///     typed::{FullHiddenStates, FullQueryHeads},
/// };
/// let q = FullQueryHeads::from_tensor(Tensor::zeros(&[1, 2, 2, 2])).unwrap();
/// let x = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, 4])).unwrap();
/// let _ = rope::rotate_queries(&x, RopeConfig { base: 10_000.0, start_pos: 0 }, "q_rope");
/// ```
pub struct TypedTensorCompileContracts;
