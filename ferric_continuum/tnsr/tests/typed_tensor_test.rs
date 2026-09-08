use std::rc::Rc;

use tnsr::{
    autograd::{Engine, GraphObservation, GraphOpIndex, GraphTensorId, OpKind},
    ops::{basic, context_parallel_gqa, gqa, linear, norm, rope, shape},
    qwen3::{Qwen3Attention, Qwen3Config},
    tensor::{Shape, Tensor, TensorValue},
    typed::{
        Axes, Axes3, AxisExtent, Batch, Full, FullHiddenStates, FullKvHeads, FullQueryHeads,
        FullSequence, HeadDim, HeadScale, Hidden, HiddenStates, KvHead, KvProjectionWeight,
        OutputProjectionWeight, ProjectedKv, ProjectedQueries, QueryHead, QueryProjectionWeight,
        ShardHiddenStates, ShardKvHeads, ShardQueryHeads,
    },
};

fn deterministic_value(shape: &[usize], scale: f32, offset: f32) -> TensorValue {
    let count: usize = shape.iter().product();
    let data = (0..count)
        .map(|i| (((i * 19 + 7) % 31) as f32 - 15.0) * scale + offset)
        .collect();
    TensorValue::from_vec(Shape(shape.to_vec()), data)
}

fn clone_leaf(tensor: &Tensor) -> Tensor {
    Tensor::from_value(tensor.value(), true)
}

fn clone_attention(source: &Qwen3Attention) -> Qwen3Attention {
    Qwen3Attention {
        n_q_heads: source.n_q_heads,
        n_kv_heads: source.n_kv_heads,
        head_dim: source.head_dim,
        rope_cfg: source.rope_cfg,
        wq: clone_leaf(&source.wq),
        wk: clone_leaf(&source.wk),
        wv: clone_leaf(&source.wv),
        wo: clone_leaf(&source.wo),
        q_norm: clone_leaf(&source.q_norm),
        k_norm: clone_leaf(&source.k_norm),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "mismatch at {index}: actual={actual} expected={expected}"
        );
    }
}

fn split_hidden(value: &TensorValue, shards: usize) -> Vec<TensorValue> {
    let shape = &value.shape.0;
    let (batch, sequence, hidden) = (shape[0], shape[1], shape[2]);
    assert_eq!(sequence % shards, 0);
    let local_sequence = sequence / shards;
    let source = value.data.as_ref();

    (0..shards)
        .map(|rank| {
            let mut data = vec![0.0; batch * local_sequence * hidden];
            for batch_index in 0..batch {
                for local_position in 0..local_sequence {
                    let global_position = rank * local_sequence + local_position;
                    let source_start = (batch_index * sequence + global_position) * hidden;
                    let target_start = (batch_index * local_sequence + local_position) * hidden;
                    data[target_start..target_start + hidden]
                        .copy_from_slice(&source[source_start..source_start + hidden]);
                }
            }
            TensorValue::from_vec(Shape(vec![batch, local_sequence, hidden]), data)
        })
        .collect()
}

fn split_heads(value: &TensorValue, shards: usize) -> Vec<TensorValue> {
    let shape = &value.shape.0;
    let (batch, sequence, heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(sequence % shards, 0);
    let local_sequence = sequence / shards;

    (0..shards)
        .map(|rank| {
            let mut data = vec![0.0; batch * local_sequence * heads * head_dim];
            for batch_index in 0..batch {
                for local_position in 0..local_sequence {
                    let global_position = rank * local_sequence + local_position;
                    for head in 0..heads {
                        let source_start =
                            ((batch_index * sequence + global_position) * heads + head) * head_dim;
                        let target_start =
                            ((batch_index * local_sequence + local_position) * heads + head)
                                * head_dim;
                        data[target_start..target_start + head_dim]
                            .copy_from_slice(&value.data[source_start..source_start + head_dim]);
                    }
                }
            }
            TensorValue::from_vec(Shape(vec![batch, local_sequence, heads, head_dim]), data)
        })
        .collect()
}

fn join_hidden(shards: &[TensorValue]) -> TensorValue {
    let local_shape = &shards[0].shape.0;
    let (batch, local_sequence, hidden) = (local_shape[0], local_shape[1], local_shape[2]);
    let sequence = local_sequence * shards.len();
    let mut data = vec![0.0; batch * sequence * hidden];

    for (rank, shard) in shards.iter().enumerate() {
        for batch_index in 0..batch {
            for local_position in 0..local_sequence {
                let global_position = rank * local_sequence + local_position;
                let source_start = (batch_index * local_sequence + local_position) * hidden;
                let target_start = (batch_index * sequence + global_position) * hidden;
                data[target_start..target_start + hidden]
                    .copy_from_slice(&shard.data[source_start..source_start + hidden]);
            }
        }
    }
    TensorValue::from_vec(Shape(vec![batch, sequence, hidden]), data)
}

fn join_heads(shards: &[TensorValue]) -> TensorValue {
    let shape = &shards[0].shape.0;
    let (batch, local_sequence, heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let sequence = local_sequence * shards.len();
    let mut data = vec![0.0; batch * sequence * heads * head_dim];

    for (rank, shard) in shards.iter().enumerate() {
        for batch_index in 0..batch {
            for local_position in 0..local_sequence {
                let global_position = rank * local_sequence + local_position;
                for head in 0..heads {
                    let source_start =
                        ((batch_index * local_sequence + local_position) * heads + head) * head_dim;
                    let target_start =
                        ((batch_index * sequence + global_position) * heads + head) * head_dim;
                    data[target_start..target_start + head_dim]
                        .copy_from_slice(&shard.data[source_start..source_start + head_dim]);
                }
            }
        }
    }

    TensorValue::from_vec(Shape(vec![batch, sequence, heads, head_dim]), data)
}

fn summed_raw_shard_loss(outputs: &[Tensor], prefix: &str) -> Tensor {
    let mut loss = basic::sum(&outputs[0], &format!("{prefix}.sum0"));
    for (rank, output) in outputs.iter().enumerate().skip(1) {
        let shard_loss = basic::sum(output, &format!("{prefix}.sum{rank}"));
        loss = basic::add(&loss, &shard_loss, &format!("{prefix}.add{rank}"));
    }
    loss
}

fn summed_typed_head_loss(outputs: &[ShardQueryHeads], prefix: &str) -> Tensor {
    let raw_outputs: Vec<Tensor> = outputs
        .iter()
        .map(|output| output.as_tensor().clone())
        .collect();
    summed_raw_shard_loss(&raw_outputs, prefix)
}

fn summed_typed_shard_loss(outputs: &[ShardHiddenStates]) -> Tensor {
    let mut loss = basic::sum(outputs[0].as_tensor(), "typed.cp.loss0");
    for (rank, output) in outputs.iter().enumerate().skip(1) {
        let shard_loss = basic::sum(output.as_tensor(), &format!("typed.cp.loss{rank}"));
        loss = basic::add(&loss, &shard_loss, &format!("typed.cp.add{rank}"));
    }
    loss
}

#[test]
fn typed_tensor_exposes_named_dimensions_without_changing_tensor_identity() {
    let input = Tensor::randn(&[2, 3, 4]).requires_grad();
    let weight = Tensor::randn(&[4, 4]).requires_grad();
    let projected = linear::linear(&input, &weight, "project");
    let producer_before = projected.producer_id().expect("linear producer");

    let typed = FullHiddenStates::from_tensor(projected.clone()).expect("rank-three tensor");
    let producer_after = typed
        .as_tensor()
        .producer_id()
        .expect("same linear producer");

    assert_eq!(
        typed.named_shape(),
        vec![
            ("batch".to_owned(), 2),
            ("sequence".to_owned(), 3),
            ("hidden".to_owned(), 4),
        ]
    );
    assert_eq!(typed.id(), projected.id());
    assert!(typed.as_tensor().shares_storage_with(&projected));
    assert_eq!(producer_before, producer_after);

    let cloned = typed.clone();
    assert!(cloned.as_tensor().shares_storage_with(&projected));
    let erased = cloned.into_tensor();
    assert!(erased.shares_storage_with(&projected));
}

#[test]
fn canonical_transformer_layouts_expose_typed_runtime_extents() {
    let hidden = FullHiddenStates::from_tensor(Tensor::zeros(&[2, 3, 5])).unwrap();
    let projected_queries =
        ProjectedQueries::<Full>::from_tensor(Tensor::zeros(&[2, 3, 8])).unwrap();
    let projected_kv = ProjectedKv::<Full>::from_tensor(Tensor::zeros(&[2, 3, 4])).unwrap();
    let query_heads = FullQueryHeads::from_tensor(Tensor::zeros(&[2, 3, 4, 2])).unwrap();
    let kv_heads = FullKvHeads::from_tensor(Tensor::zeros(&[2, 3, 2, 2])).unwrap();
    let query_weight = QueryProjectionWeight::from_tensor(Tensor::zeros(&[5, 8])).unwrap();
    let kv_weight = KvProjectionWeight::from_tensor(Tensor::zeros(&[5, 4])).unwrap();
    let output_weight = OutputProjectionWeight::from_tensor(Tensor::zeros(&[8, 5])).unwrap();
    let scale = HeadScale::from_tensor(Tensor::zeros(&[2])).unwrap();

    assert_eq!(hidden.batch_extent().get(), 2);
    assert_eq!(hidden.sequence_extent().get(), 3);
    assert_eq!(hidden.hidden_extent().get(), 5);
    assert_eq!(projected_queries.flattened_query_extent().get(), 8);
    assert_eq!(projected_kv.flattened_kv_extent().get(), 4);
    assert_eq!(query_heads.query_heads_extent().get(), 4);
    assert_eq!(query_heads.head_dim_extent().get(), 2);
    assert_eq!(kv_heads.kv_heads_extent().get(), 2);
    assert_eq!(kv_heads.head_dim_extent().get(), 2);
    assert_eq!(query_weight.hidden_extent().get(), 5);
    assert_eq!(query_weight.flattened_query_extent().get(), 8);
    assert_eq!(kv_weight.hidden_extent().get(), 5);
    assert_eq!(kv_weight.flattened_kv_extent().get(), 4);
    assert_eq!(output_weight.flattened_query_extent().get(), 8);
    assert_eq!(output_weight.hidden_extent().get(), 5);
    assert_eq!(scale.head_dim_extent().get(), 2);
}

#[test]
fn typed_extents_preserve_axis_meaning_while_merging() {
    let query_heads = AxisExtent::<QueryHead>::new(4);
    let head_dim = AxisExtent::<HeadDim>::new(2);
    let merged = query_heads.checked_merge(head_dim).unwrap();

    assert_eq!(merged.get(), 8);
    assert_eq!(query_heads, AxisExtent::<QueryHead>::new(4));
    assert_eq!(
        AxisExtent::<QueryHead>::new(usize::MAX).checked_merge(AxisExtent::<HeadDim>::new(2)),
        None
    );
}

#[test]
fn head_splitting_accepts_named_extent_evidence() {
    let projected_queries =
        ProjectedQueries::<Full>::from_tensor(Tensor::zeros(&[1, 3, 4])).unwrap();
    let projected_kv = ProjectedKv::<Full>::from_tensor(Tensor::zeros(&[1, 3, 2])).unwrap();

    let queries = shape::split_query_heads(
        &projected_queries,
        AxisExtent::<QueryHead>::new(2),
        AxisExtent::<HeadDim>::new(2),
        "typed.q_split",
    );
    let keys = shape::split_kv_heads(
        &projected_kv,
        AxisExtent::<KvHead>::new(1),
        AxisExtent::<HeadDim>::new(2),
        "typed.k_split",
    );

    assert_eq!(queries.shape().0, vec![1, 3, 2, 2]);
    assert_eq!(keys.shape().0, vec![1, 3, 1, 2]);
}

#[test]
fn typed_tensor_rejects_the_wrong_runtime_rank() {
    let error = FullHiddenStates::from_tensor(Tensor::zeros(&[2, 3]))
        .err()
        .expect("rank mismatch");

    assert_eq!(error.expected_rank, 3);
    assert_eq!(error.actual_rank, 2);
    assert_eq!(
        error.to_string(),
        "typed tensor expected rank 3 but received rank 2"
    );
}

#[test]
#[should_panic(expected = "typed tensor rank changed after axis attachment")]
fn typed_tensor_detects_rank_mutation_through_a_legacy_alias() {
    let raw = Tensor::zeros(&[1, 2, 3]);
    let typed = FullHiddenStates::from_tensor(raw.clone()).unwrap();
    raw.inner.borrow_mut().value = TensorValue::zeros(Shape(vec![2, 3]));

    let _ = typed.hidden_extent();
}

#[test]
fn generic_hidden_state_alias_accepts_a_valid_sequence_kind() {
    let typed: HiddenStates<Full> =
        HiddenStates::from_tensor(Tensor::zeros(&[1, 2, 3])).expect("full hidden states");
    assert_eq!(typed.shape().0, vec![1, 2, 3]);
}

#[test]
fn axes_expose_their_ordered_display_names() {
    type FullHiddenAxes = Axes3<Batch, FullSequence, Hidden>;

    assert_eq!(
        FullHiddenAxes::names(),
        vec![
            "batch".to_owned(),
            "sequence".to_owned(),
            "hidden".to_owned(),
        ]
    );
}

#[test]
fn typed_attention_operations_make_every_axis_transform_explicit() {
    let (batch, sequence, hidden, query_heads, kv_heads, head_dim) = (1, 3, 4, 2, 1, 2);
    let input_raw = Tensor::randn(&[batch, sequence, hidden]).requires_grad();
    let query_weight_raw = Tensor::randn(&[hidden, query_heads * head_dim]).requires_grad();
    let kv_weight_raw = Tensor::randn(&[hidden, kv_heads * head_dim]).requires_grad();
    let value_weight_raw = Tensor::randn(&[hidden, kv_heads * head_dim]).requires_grad();
    let output_weight_raw = Tensor::randn(&[query_heads * head_dim, hidden]).requires_grad();
    let query_scale_raw = Tensor::randn(&[head_dim]).requires_grad();
    let key_scale_raw = Tensor::randn(&[head_dim]).requires_grad();

    let input = FullHiddenStates::from_tensor(input_raw.clone()).unwrap();
    let query_weight = QueryProjectionWeight::from_tensor(query_weight_raw.clone()).unwrap();
    let kv_weight = KvProjectionWeight::from_tensor(kv_weight_raw.clone()).unwrap();
    let value_weight = KvProjectionWeight::from_tensor(value_weight_raw.clone()).unwrap();
    let output_weight = OutputProjectionWeight::from_tensor(output_weight_raw.clone()).unwrap();
    let query_scale = HeadScale::from_tensor(query_scale_raw.clone()).unwrap();
    let key_scale = HeadScale::from_tensor(key_scale_raw.clone()).unwrap();

    let projected_queries = linear::project_queries(&input, &query_weight, "typed.q_proj");
    let projected_keys = linear::project_keys(&input, &kv_weight, "typed.k_proj");
    let projected_values = linear::project_values(&input, &value_weight, "typed.v_proj");
    assert_eq!(
        projected_queries.named_shape(),
        vec![
            ("batch".to_owned(), batch),
            ("sequence".to_owned(), sequence),
            ("query_head*head_dim".to_owned(), query_heads * head_dim),
        ]
    );

    let queries =
        shape::split_query_heads(&projected_queries, query_heads, head_dim, "typed.q_split");
    let keys = shape::split_kv_heads(&projected_keys, kv_heads, head_dim, "typed.k_split");
    let values = shape::split_kv_heads(&projected_values, kv_heads, head_dim, "typed.v_split");
    let queries = norm::normalize_queries(&queries, &query_scale, "typed.q_norm");
    let keys = norm::normalize_keys(&keys, &key_scale, "typed.k_norm");
    let queries = rope::rotate_queries(
        &queries,
        rope::RopeConfig {
            base: 10_000.0,
            start_pos: 0,
        },
        "typed.q_rope",
    );
    let keys: FullKvHeads = rope::rotate_keys(
        &keys,
        rope::RopeConfig {
            base: 10_000.0,
            start_pos: 0,
        },
        "typed.k_rope",
    );
    let attention: FullQueryHeads = gqa::gqa_attention_typed(&queries, &keys, &values, "typed.gqa");
    assert_eq!(
        attention.named_shape(),
        vec![
            ("batch".to_owned(), batch),
            ("sequence".to_owned(), sequence),
            ("query_head".to_owned(), query_heads),
            ("head_dim".to_owned(), head_dim),
        ]
    );
    let attention_graph = GraphObservation::from_outputs(&[attention.as_tensor()]);
    let producer = attention_graph
        .producer_of(attention.as_tensor())
        .unwrap()
        .unwrap();
    assert_eq!(
        attention_graph
            .operation(producer)
            .expect("observed GQA producer")
            .kind,
        OpKind::GqaAttention
    );

    let flattened = shape::merge_query_heads(&attention, "typed.merge_heads");
    let output = linear::project_attention_output(&flattened, &output_weight, "typed.o_proj");
    assert_eq!(output.shape().0, vec![batch, sequence, hidden]);

    let mut engine = Engine::new();
    engine.backward(&basic::sum(output.as_tensor(), "typed.loss"));
    for raw in [
        &input_raw,
        &query_weight_raw,
        &kv_weight_raw,
        &value_weight_raw,
        &output_weight_raw,
        &query_scale_raw,
        &key_scale_raw,
    ] {
        assert!(
            raw.grad().is_some(),
            "typed path must preserve autograd edges"
        );
    }
}

#[test]
#[should_panic(expected = "split_query_heads: flattened head extent mismatch")]
fn query_head_split_checks_the_merged_extent_even_for_an_empty_batch() {
    let projected = ProjectedQueries::<Full>::from_tensor(Tensor::zeros(&[0, 2, 4])).unwrap();

    let _ = shape::split_query_heads(&projected, 2, 3, "invalid.q_split");
}

#[test]
#[should_panic(expected = "split_kv_heads: flattened head extent mismatch")]
fn kv_head_split_checks_the_merged_extent_even_for_an_empty_batch() {
    let projected = ProjectedKv::<Full>::from_tensor(Tensor::zeros(&[0, 2, 4])).unwrap();

    let _ = shape::split_kv_heads(&projected, 1, 3, "invalid.kv_split");
}

#[test]
#[should_panic(expected = "project_queries: Hidden extent must match weight input")]
fn query_projection_consumes_named_hidden_extents_before_erasure() {
    let hidden = FullHiddenStates::from_tensor(Tensor::zeros(&[0, 2, 3])).unwrap();
    let weight = QueryProjectionWeight::from_tensor(Tensor::zeros(&[4, 6])).unwrap();

    let _ = linear::project_queries(&hidden, &weight, "typed.q_proj");
}

#[test]
#[should_panic(expected = "rotate_queries: HeadDim must be positive")]
fn query_rope_consumes_named_head_extent_before_erasure() {
    let queries = FullQueryHeads::from_tensor(Tensor::zeros(&[0, 1, 2, 0])).unwrap();

    let _ = rope::rotate_queries(&queries, rope::RopeConfig::default(), "typed.q_rope");
}

#[test]
#[should_panic(expected = "normalize_queries: HeadScale extent must equal Dh")]
fn query_normalization_checks_head_scale_extent_even_for_an_empty_batch() {
    let queries = FullQueryHeads::from_tensor(Tensor::zeros(&[0, 2, 2, 2])).unwrap();
    let scale = HeadScale::from_tensor(Tensor::zeros(&[3])).unwrap();

    let _ = norm::normalize_queries(&queries, &scale, "invalid.q_norm");
}

#[test]
#[should_panic(expected = "normalize_keys: HeadScale extent must equal Dh")]
fn key_normalization_checks_head_scale_extent_even_for_an_empty_batch() {
    let keys = FullKvHeads::from_tensor(Tensor::zeros(&[0, 2, 1, 2])).unwrap();
    let scale = HeadScale::from_tensor(Tensor::zeros(&[3])).unwrap();

    let _ = norm::normalize_keys(&keys, &scale, "invalid.k_norm");
}

#[test]
fn typed_context_parallel_gqa_preserves_the_shared_multi_output_producer() {
    let q0 = ShardQueryHeads::from_tensor(Tensor::randn(&[1, 1, 2, 2]).requires_grad()).unwrap();
    let q1 = ShardQueryHeads::from_tensor(Tensor::randn(&[1, 1, 2, 2]).requires_grad()).unwrap();
    let k0 = ShardKvHeads::from_tensor(Tensor::randn(&[1, 1, 1, 2]).requires_grad()).unwrap();
    let k1 = ShardKvHeads::from_tensor(Tensor::randn(&[1, 1, 1, 2]).requires_grad()).unwrap();
    let v0 = ShardKvHeads::from_tensor(Tensor::randn(&[1, 1, 1, 2]).requires_grad()).unwrap();
    let v1 = ShardKvHeads::from_tensor(Tensor::randn(&[1, 1, 1, 2]).requires_grad()).unwrap();

    let outputs = context_parallel_gqa::context_parallel_gqa_attention_typed(
        &[q0, q1],
        &[k0, k1],
        &[v0, v1],
        "typed.cp_gqa",
    );

    assert_eq!(outputs.len(), 2);
    let graph = GraphObservation::from_outputs(&[outputs[0].as_tensor(), outputs[1].as_tensor()]);
    assert_eq!(graph.operations().len(), 1);
    assert_eq!(
        graph.producer_of(outputs[0].as_tensor()),
        Ok(Some(GraphOpIndex(0)))
    );
    assert_eq!(
        graph.producer_of(outputs[1].as_tensor()),
        Ok(Some(GraphOpIndex(0)))
    );
    assert_eq!(
        graph.operations()[0].kind,
        OpKind::ContextParallelGqaAttention
    );
}

#[test]
fn typed_context_parallel_gqa_is_transparent_for_p1_p2_p4_and_b2() {
    let (batch, sequence, query_heads, kv_heads, head_dim) = (2, 4, 4, 2, 2);
    let q_value = deterministic_value(&[batch, sequence, query_heads, head_dim], 0.023, -0.07);
    let k_value = deterministic_value(&[batch, sequence, kv_heads, head_dim], 0.037, 0.11);
    let v_value = deterministic_value(&[batch, sequence, kv_heads, head_dim], 0.041, -0.13);

    for cp in [1, 2, 4] {
        let raw_q: Vec<Tensor> = split_heads(&q_value, cp)
            .into_iter()
            .map(|value| Tensor::from_value(value, true))
            .collect();
        let raw_k: Vec<Tensor> = split_heads(&k_value, cp)
            .into_iter()
            .map(|value| Tensor::from_value(value, true))
            .collect();
        let raw_v: Vec<Tensor> = split_heads(&v_value, cp)
            .into_iter()
            .map(|value| Tensor::from_value(value, true))
            .collect();
        let typed_q_raw: Vec<Tensor> = split_heads(&q_value, cp)
            .into_iter()
            .map(|value| Tensor::from_value(value, true))
            .collect();
        let typed_k_raw: Vec<Tensor> = split_heads(&k_value, cp)
            .into_iter()
            .map(|value| Tensor::from_value(value, true))
            .collect();
        let typed_v_raw: Vec<Tensor> = split_heads(&v_value, cp)
            .into_iter()
            .map(|value| Tensor::from_value(value, true))
            .collect();
        let typed_q: Vec<ShardQueryHeads> = typed_q_raw
            .iter()
            .cloned()
            .map(|tensor| ShardQueryHeads::from_tensor(tensor).unwrap())
            .collect();
        let typed_k: Vec<ShardKvHeads> = typed_k_raw
            .iter()
            .cloned()
            .map(|tensor| ShardKvHeads::from_tensor(tensor).unwrap())
            .collect();
        let typed_v: Vec<ShardKvHeads> = typed_v_raw
            .iter()
            .cloned()
            .map(|tensor| ShardKvHeads::from_tensor(tensor).unwrap())
            .collect();

        let mut raw_engine = Engine::new();
        let raw_outputs = context_parallel_gqa::context_parallel_gqa_attention(
            &raw_q,
            &raw_k,
            &raw_v,
            "raw.cp_gqa",
        );
        let raw_loss = summed_raw_shard_loss(&raw_outputs, "raw.cp.loss");
        let mut typed_engine = Engine::new();
        let typed_outputs = context_parallel_gqa::context_parallel_gqa_attention_typed(
            &typed_q,
            &typed_k,
            &typed_v,
            "typed.cp_gqa",
        );
        let typed_loss = summed_typed_head_loss(&typed_outputs, "typed.cp.loss");

        let raw_output_values: Vec<TensorValue> = raw_outputs.iter().map(Tensor::value).collect();
        let typed_output_values: Vec<TensorValue> = typed_outputs
            .iter()
            .map(|output| output.as_tensor().value())
            .collect();
        assert_close(
            join_heads(&typed_output_values).data.as_ref(),
            join_heads(&raw_output_values).data.as_ref(),
            0.0,
        );

        let roots: Vec<&Tensor> = typed_outputs
            .iter()
            .map(|output| output.as_tensor())
            .collect();
        let graph = GraphObservation::from_outputs(&roots);
        assert_eq!(graph.operations().len(), 1);
        assert!(typed_outputs
            .iter()
            .all(|output| { graph.producer_of(output.as_tensor()) == Ok(Some(GraphOpIndex(0))) }));
        assert_eq!(
            graph.operations()[0]
                .inputs
                .iter()
                .map(|input| input.id)
                .collect::<Vec<_>>(),
            (0..typed_q_raw.len() + typed_k_raw.len() + typed_v_raw.len())
                .map(GraphTensorId)
                .collect::<Vec<_>>()
        );

        raw_engine.backward(&raw_loss);
        typed_engine.backward(&typed_loss);
        for (raw_group, typed_group) in [
            (&raw_q, &typed_q_raw),
            (&raw_k, &typed_k_raw),
            (&raw_v, &typed_v_raw),
        ] {
            let raw_gradients: Vec<TensorValue> = raw_group
                .iter()
                .map(|tensor| tensor.grad().expect("raw shard gradient"))
                .collect();
            let typed_gradients: Vec<TensorValue> = typed_group
                .iter()
                .map(|tensor| tensor.grad().expect("typed shard gradient"))
                .collect();
            assert_close(
                join_heads(&typed_gradients).data.as_ref(),
                join_heads(&raw_gradients).data.as_ref(),
                0.0,
            );
        }
    }
}

#[test]
fn typed_qwen3_attention_matches_the_independent_legacy_derivation() {
    let config = Qwen3Config::tiny();
    let seed = Qwen3Attention::new(&config);
    let legacy_attention = clone_attention(&seed);
    let typed_attention = clone_attention(&seed);
    let input_value = deterministic_value(&[2, 3, config.hidden_size], 0.0125, -0.05);
    let legacy_input = Tensor::from_value(input_value.clone(), true);
    let typed_input_raw = Tensor::from_value(input_value, true);
    let typed_input = FullHiddenStates::from_tensor(typed_input_raw.clone()).unwrap();

    let mut legacy_engine = Engine::new();
    let legacy_output = legacy_attention.forward(&legacy_input);
    let legacy_trace = GraphObservation::from_outputs(&[&legacy_output]);
    legacy_engine.backward(&basic::sum(&legacy_output, "legacy.loss"));

    let mut typed_engine = Engine::new();
    let typed_output = typed_attention.forward_typed(&typed_input);
    let typed_trace = GraphObservation::from_outputs(&[typed_output.as_tensor()]);
    typed_engine.backward(&basic::sum(typed_output.as_tensor(), "typed.loss"));

    assert_close(
        typed_output.as_tensor().value().data.as_ref(),
        legacy_output.value().data.as_ref(),
        1e-6,
    );
    assert_close(
        typed_input_raw.grad().unwrap().data.as_ref(),
        legacy_input.grad().unwrap().data.as_ref(),
        1e-6,
    );
    for (typed_parameter, legacy_parameter) in typed_attention
        .parameters()
        .into_iter()
        .zip(legacy_attention.parameters())
    {
        assert_close(
            typed_parameter.grad().unwrap().data.as_ref(),
            legacy_parameter.grad().unwrap().data.as_ref(),
            1e-6,
        );
    }
    assert_eq!(typed_trace, legacy_trace);
}

#[test]
fn typed_qwen3_context_parallel_matches_full_forward_and_gradients() {
    let config = Qwen3Config::tiny();
    let seed = Qwen3Attention::new(&config);
    let full_attention = clone_attention(&seed);
    let cp_attention = clone_attention(&seed);
    let input_value = deterministic_value(&[2, 4, config.hidden_size], 0.01, -0.025);
    let full_input = Tensor::from_value(input_value.clone(), true);
    let shard_inputs_raw: Vec<Tensor> = split_hidden(&input_value, 2)
        .into_iter()
        .map(|value| Tensor::from_value(value, true))
        .collect();
    let shard_inputs: Vec<ShardHiddenStates> = shard_inputs_raw
        .iter()
        .cloned()
        .map(|tensor| ShardHiddenStates::from_tensor(tensor).unwrap())
        .collect();

    let mut full_engine = Engine::new();
    let full_output = full_attention.forward(&full_input);
    full_engine.backward(&basic::sum(&full_output, "full.loss"));

    let mut cp_engine = Engine::new();
    let cp_outputs = cp_attention.forward_context_parallel_typed(&shard_inputs);
    cp_engine.backward(&summed_typed_shard_loss(&cp_outputs));

    let cp_output_values: Vec<TensorValue> = cp_outputs
        .iter()
        .map(|tensor| tensor.as_tensor().value())
        .collect();
    assert_close(
        join_hidden(&cp_output_values).data.as_ref(),
        full_output.value().data.as_ref(),
        1e-5,
    );

    let cp_input_grads: Vec<TensorValue> = shard_inputs_raw
        .iter()
        .map(|tensor| tensor.grad().unwrap())
        .collect();
    assert_close(
        join_hidden(&cp_input_grads).data.as_ref(),
        full_input.grad().unwrap().data.as_ref(),
        2e-5,
    );
    for (cp_parameter, full_parameter) in cp_attention
        .parameters()
        .into_iter()
        .zip(full_attention.parameters())
    {
        assert_close(
            cp_parameter.grad().unwrap().data.as_ref(),
            full_parameter.grad().unwrap().data.as_ref(),
            2e-5,
        );
    }
}

#[test]
fn typed_qwen3_context_parallel_matches_the_independent_legacy_trace() {
    let config = Qwen3Config::tiny();
    let seed = Qwen3Attention::new(&config);
    let legacy_attention = clone_attention(&seed);
    let typed_attention = clone_attention(&seed);
    let input_value = deterministic_value(&[1, 4, config.hidden_size], 0.011, 0.025);
    let shard_values = split_hidden(&input_value, 2);
    let legacy_inputs: Vec<Tensor> = shard_values
        .iter()
        .cloned()
        .map(|value| Tensor::from_value(value, true))
        .collect();
    let typed_inputs_raw: Vec<Tensor> = shard_values
        .into_iter()
        .map(|value| Tensor::from_value(value, true))
        .collect();
    let typed_inputs: Vec<ShardHiddenStates> = typed_inputs_raw
        .iter()
        .cloned()
        .map(|tensor| ShardHiddenStates::from_tensor(tensor).unwrap())
        .collect();

    let mut legacy_engine = Engine::new();
    let legacy_outputs = legacy_attention.forward_context_parallel(&legacy_inputs);
    let legacy_roots: Vec<&Tensor> = legacy_outputs.iter().collect();
    let legacy_trace = GraphObservation::from_outputs(&legacy_roots);
    let legacy_loss = summed_raw_shard_loss(&legacy_outputs, "legacy.cp.loss");

    let mut typed_engine = Engine::new();
    let typed_outputs = typed_attention.forward_context_parallel_typed(&typed_inputs);
    let typed_roots: Vec<&Tensor> = typed_outputs
        .iter()
        .map(|output| output.as_tensor())
        .collect();
    let typed_trace = GraphObservation::from_outputs(&typed_roots);
    let typed_loss = summed_typed_shard_loss(&typed_outputs);

    assert_eq!(typed_outputs.len(), legacy_outputs.len());
    for (typed, legacy) in typed_outputs.iter().zip(&legacy_outputs) {
        assert_close(
            typed.as_tensor().value().data.as_ref(),
            legacy.value().data.as_ref(),
            1e-6,
        );
    }
    assert_eq!(typed_trace, legacy_trace);

    legacy_engine.backward(&legacy_loss);
    typed_engine.backward(&typed_loss);
    let legacy_input_gradients: Vec<TensorValue> = legacy_inputs
        .iter()
        .map(|input| input.grad().expect("legacy CP input gradient"))
        .collect();
    let typed_input_gradients: Vec<TensorValue> = typed_inputs_raw
        .iter()
        .map(|input| input.grad().expect("typed CP input gradient"))
        .collect();
    assert_close(
        join_hidden(&typed_input_gradients).data.as_ref(),
        join_hidden(&legacy_input_gradients).data.as_ref(),
        1e-6,
    );
    for (typed_parameter, legacy_parameter) in typed_attention
        .parameters()
        .into_iter()
        .zip(legacy_attention.parameters())
    {
        assert_close(
            typed_parameter.grad().unwrap().data.as_ref(),
            legacy_parameter.grad().unwrap().data.as_ref(),
            1e-6,
        );
    }
}

#[test]
fn typed_qwen3_rebuilds_validated_state_after_valid_public_mutation() {
    let config = Qwen3Config::tiny();
    let mut attention = Qwen3Attention::new(&config);
    let original_parameters = vec![
        attention.wq.clone(),
        attention.wk.clone(),
        attention.wv.clone(),
        attention.wo.clone(),
        attention.q_norm.clone(),
        attention.k_norm.clone(),
    ];
    let first_input =
        FullHiddenStates::from_tensor(Tensor::randn(&[1, 2, config.hidden_size]).requires_grad())
            .unwrap();
    let _ = attention.forward_typed(&first_input);

    attention.n_q_heads = 2;
    attention.n_kv_heads = 1;
    attention.head_dim = 6;
    attention.wq = Tensor::randn(&[config.hidden_size, 12]).requires_grad();
    attention.wk = Tensor::randn(&[config.hidden_size, 6]).requires_grad();
    attention.wv = Tensor::randn(&[config.hidden_size, 6]).requires_grad();
    attention.wo = Tensor::randn(&[12, config.hidden_size]).requires_grad();
    attention.q_norm = Tensor::randn(&[6]).requires_grad();
    attention.k_norm = Tensor::randn(&[6]).requires_grad();
    attention.rope_cfg = rope::RopeConfig {
        base: 37.0,
        start_pos: 7,
    };
    let replacement_parameters: Vec<Tensor> = attention
        .parameters()
        .into_iter()
        .map(Clone::clone)
        .collect();
    let legacy_attention = clone_attention(&attention);
    let mut stale_rope_attention = clone_attention(&attention);
    stale_rope_attention.rope_cfg.base = rope::RopeConfig::default().base;

    let second_input_value = deterministic_value(&[1, 2, config.hidden_size], 0.031, -0.09);
    let legacy_input = Tensor::from_value(second_input_value.clone(), true);
    let stale_rope_input = Tensor::from_value_no_grad(second_input_value.clone());
    let second_input =
        FullHiddenStates::from_tensor(Tensor::from_value(second_input_value, true)).unwrap();
    let legacy_output = legacy_attention.forward(&legacy_input);
    let stale_rope_output = stale_rope_attention.forward(&stale_rope_input);
    let output = attention.forward_typed(&second_input);
    assert_close(
        output.as_tensor().value().data.as_ref(),
        legacy_output.value().data.as_ref(),
        1e-6,
    );
    assert!(
        output
            .as_tensor()
            .value()
            .data
            .iter()
            .zip(stale_rope_output.value().data.iter())
            .any(|(fresh, stale)| (fresh - stale).abs() > 1e-5),
        "fixture must detect a stale RoPE base"
    );
    let graph = GraphObservation::from_outputs(&[output.as_tensor()]);
    let query_reshape = graph
        .operations()
        .iter()
        .find(|operation| operation.name == "q_reshape")
        .expect("typed query reshape");
    assert_eq!(query_reshape.outputs[0].shape, Shape(vec![1, 2, 2, 6]));
    let key_reshape = graph
        .operations()
        .iter()
        .find(|operation| operation.name == "k_reshape")
        .expect("typed key reshape");
    assert_eq!(key_reshape.outputs[0].shape, Shape(vec![1, 2, 1, 6]));

    let mut engine = Engine::new();
    engine.backward(&basic::sum(output.as_tensor(), "mutated.loss"));
    for parameter in replacement_parameters {
        assert!(parameter.grad().is_some());
    }
    for parameter in original_parameters {
        assert!(parameter.grad().is_none());
    }
}

#[test]
#[should_panic(expected = "Qwen3Attention typed: wk must have shape [D,Hkv*Dh]")]
fn typed_qwen3_validates_public_parameter_shapes_before_constructing_state() {
    let config = Qwen3Config::tiny();
    let mut attention = Qwen3Attention::new(&config);
    attention.wk = Tensor::zeros(&[
        config.hidden_size,
        config.num_key_value_heads * config.head_dim + 1,
    ]);
    let input = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, config.hidden_size])).unwrap();

    let _ = attention.forward_typed(&input);
}

#[test]
#[should_panic(expected = "Qwen3Attention typed: query head extent overflow")]
fn typed_qwen3_rejects_head_extent_overflow_before_running_operations() {
    let config = Qwen3Config::tiny();
    let mut attention = Qwen3Attention::new(&config);
    attention.n_q_heads = usize::MAX - 1;
    attention.head_dim = 2;
    let input = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 1, config.hidden_size])).unwrap();

    let _ = attention.forward_typed(&input);
}

#[test]
#[should_panic(expected = "Qwen3Attention typed: n_q_heads must be positive")]
fn typed_qwen3_rejects_zero_query_heads() {
    let config = Qwen3Config::tiny();
    let mut attention = Qwen3Attention::new(&config);
    attention.n_q_heads = 0;
    attention.wq = Tensor::zeros(&[config.hidden_size, 0]);
    attention.wo = Tensor::zeros(&[0, config.hidden_size]);
    let input = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 1, config.hidden_size])).unwrap();

    let _ = attention.forward_typed(&input);
}

#[test]
#[should_panic(expected = "Qwen3Attention typed: RoPE position overflow")]
fn typed_qwen3_checks_complete_sequence_rope_arithmetic() {
    let config = Qwen3Config::tiny();
    let mut attention = Qwen3Attention::new(&config);
    attention.rope_cfg.start_pos = usize::MAX;
    let input = FullHiddenStates::from_tensor(Tensor::zeros(&[1, 2, config.hidden_size])).unwrap();

    let _ = attention.forward_typed(&input);
}

#[test]
#[should_panic(expected = "Qwen3Attention typed context parallel: global sequence length overflow")]
fn typed_qwen3_context_parallel_checks_global_length_arithmetic() {
    let config = Qwen3Config::tiny();
    let attention = Qwen3Attention::new(&config);
    let impossible_value = TensorValue {
        shape: Shape(vec![1, usize::MAX, config.hidden_size]),
        data: Rc::new(Vec::new()),
    };
    let first =
        ShardHiddenStates::from_tensor(Tensor::from_value_no_grad(impossible_value.clone()))
            .unwrap();
    let second =
        ShardHiddenStates::from_tensor(Tensor::from_value_no_grad(impossible_value)).unwrap();

    let _ = attention.forward_context_parallel_typed(&[first, second]);
}

#[test]
#[should_panic(expected = "Qwen3Attention typed context parallel: RoPE start position overflow")]
fn typed_qwen3_context_parallel_checks_rank_offset_arithmetic() {
    let config = Qwen3Config::tiny();
    let mut attention = Qwen3Attention::new(&config);
    attention.rope_cfg.start_pos = usize::MAX;
    let first = ShardHiddenStates::from_tensor(Tensor::zeros(&[1, 1, config.hidden_size])).unwrap();
    let second =
        ShardHiddenStates::from_tensor(Tensor::zeros(&[1, 1, config.hidden_size])).unwrap();

    let _ = attention.forward_context_parallel_typed(&[first, second]);
}

#[test]
fn typed_qwen3_context_parallel_rejects_rope_overflow_before_recording_operations() {
    let config = Qwen3Config::tiny();
    let mut attention = Qwen3Attention::new(&config);
    attention.rope_cfg.start_pos = usize::MAX;
    let first = ShardHiddenStates::from_tensor(Tensor::zeros(&[1, 1, config.hidden_size])).unwrap();
    let second =
        ShardHiddenStates::from_tensor(Tensor::zeros(&[1, 1, config.hidden_size])).unwrap();
    let engine = Engine::new();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.with_recording(|| attention.forward_context_parallel_typed(&[first, second]))
    }));

    assert!(result.is_err());
    assert!(
        engine.debug.op_call_records().is_empty(),
        "RoPE position arithmetic must be validated before building a partial graph"
    );
}
