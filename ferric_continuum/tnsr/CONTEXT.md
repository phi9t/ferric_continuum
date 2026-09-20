# tnsr

`tnsr` is the inspectable tensor and transformer context used to study model
semantics, automatic differentiation, execution, and foundation-model
parallelism.

## Language

**Causal GQA semantics**:
The mathematical rules for grouped query-head mapping, scaled scores, causal
visibility, stable normalization, value mixing, and their gradients, independent
of sequence ownership.
_Avoid_: GQA kernel, attention implementation

**Attention layout**:
The arrangement and ownership of query, key, and value sequence positions
presented to causal GQA semantics. Ordinary and context-parallel attention are
distinct attention layouts.
_Avoid_: GQA mode, attention format

**Query block**:
A contiguous interval of globally positioned queries evaluated against the
causally visible portion of global keys and values.
_Avoid_: local attention, rank attention

**Executable derivation**:
A transformer computation whose code follows the mathematical derivation in
order, keeping shapes, ownership, intermediate quantities, and gradients visible
where they are used.
_Avoid_: generic math pipeline, strategy machinery

**Named dimension**:
The semantic role of a tensor axis—such as batch, sequence, hidden, query head,
KV head, or head dimension—represented independently from its runtime extent.
_Avoid_: axis number, positional dimension

**Operation variant**:
A distinct mathematical operation with its own readable implementation, chosen
by the caller or type system rather than by runtime mode flags inside a generic
operation.
_Avoid_: operation mode, strategy branch
