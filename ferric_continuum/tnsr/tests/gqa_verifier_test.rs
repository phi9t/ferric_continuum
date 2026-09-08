//! Independent mathematical and structural verifiers for ordinary causal GQA.
//!
//! The scalar oracle in this file deliberately does not call any `tnsr`
//! attention helper. It is a second derivation against which the fused
//! implementation can be checked.

use tnsr::{
    autograd::{GraphObservation, GraphTensor, GraphTensorId, OpKind},
    ops::gqa,
    tensor::{Shape, Tensor, TensorValue},
};

fn deterministic_value(shape: &[usize], multiplier: usize, scale: f32, offset: f32) -> TensorValue {
    let count: usize = shape.iter().product();
    let data = (0..count)
        .map(|index| (((index * multiplier + 3) % 37) as f32 - 18.0) * scale + offset)
        .collect();
    TensorValue::from_vec(Shape(shape.to_vec()), data)
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

/// Direct transcription of
///
/// `O[b,t,h,:] = sum(s <= t, softmax(QK/sqrt(Dh))[s] * V[b,s,kv(h),:])`.
fn scalar_causal_gqa(q: &TensorValue, k: &TensorValue, v: &TensorValue) -> TensorValue {
    let q_shape = &q.shape.0;
    let k_shape = &k.shape.0;
    let (batch, sequence, query_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let kv_heads = k_shape[2];
    let queries_per_kv_head = query_heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; batch * sequence * query_heads * head_dim];

    let query_offset = |batch_index, token, head| {
        ((batch_index * sequence + token) * query_heads + head) * head_dim
    };
    let kv_offset =
        |batch_index, token, head| ((batch_index * sequence + token) * kv_heads + head) * head_dim;

    for batch_index in 0..batch {
        for query_token in 0..sequence {
            for query_head in 0..query_heads {
                let kv_head = query_head / queries_per_kv_head;
                let query_start = query_offset(batch_index, query_token, query_head);

                let mut scores = Vec::with_capacity(query_token + 1);
                for key_token in 0..=query_token {
                    let key_start = kv_offset(batch_index, key_token, kv_head);
                    let dot = (0..head_dim)
                        .map(|dimension| {
                            q.data[query_start + dimension] * k.data[key_start + dimension]
                        })
                        .sum::<f32>();
                    scores.push(dot * scale);
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let normalizer = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - max_score).exp();
                        *score
                    })
                    .sum::<f32>();

                for (key_token, unnormalized_probability) in scores.iter().enumerate() {
                    let probability = unnormalized_probability / normalizer;
                    let value_start = kv_offset(batch_index, key_token, kv_head);
                    for dimension in 0..head_dim {
                        output[query_start + dimension] +=
                            probability * v.data[value_start + dimension];
                    }
                }
            }
        }
    }

    TensorValue::from_vec(q.shape.clone(), output)
}

fn changed_at(
    value: &TensorValue,
    indices: impl Iterator<Item = usize>,
    delta: f32,
) -> TensorValue {
    let mut data = value.data.as_ref().clone();
    for index in indices {
        data[index] += delta;
    }
    TensorValue::from_vec(value.shape.clone(), data)
}

#[test]
fn fused_gqa_matches_an_independent_scalar_oracle_for_asymmetric_batches_and_heads() {
    let (batch, sequence, query_heads, kv_heads, head_dim) = (2, 4, 6, 2, 3);
    let q = deterministic_value(&[batch, sequence, query_heads, head_dim], 11, 0.031, -0.07);
    let k = deterministic_value(&[batch, sequence, kv_heads, head_dim], 17, 0.043, 0.11);
    let v = deterministic_value(&[batch, sequence, kv_heads, head_dim], 23, 0.037, -0.19);
    let expected = scalar_causal_gqa(&q, &k, &v);

    let actual = gqa::gqa_attention(
        &Tensor::from_value_no_grad(q),
        &Tensor::from_value_no_grad(k),
        &Tensor::from_value_no_grad(v),
        "verified.gqa",
    );

    assert_close(actual.value().data.as_ref(), expected.data.as_ref(), 1e-6);
}

#[test]
fn qwen3_32_to_8_head_mapping_matches_the_scalar_oracle() {
    let (batch, sequence, query_heads, kv_heads, head_dim) = (1, 3, 32, 8, 2);
    let q = deterministic_value(&[batch, sequence, query_heads, head_dim], 7, 0.017, -0.03);
    let k = deterministic_value(&[batch, sequence, kv_heads, head_dim], 13, 0.023, 0.05);
    let v = deterministic_value(&[batch, sequence, kv_heads, head_dim], 19, 0.029, -0.07);
    let expected = scalar_causal_gqa(&q, &k, &v);

    let actual = gqa::gqa_attention(
        &Tensor::from_value_no_grad(q),
        &Tensor::from_value_no_grad(k),
        &Tensor::from_value_no_grad(v),
        "qwen3_heads.gqa",
    );

    assert_close(actual.value().data.as_ref(), expected.data.as_ref(), 1e-6);
}

#[test]
fn future_keys_and_values_are_independently_invisible_to_earlier_queries() {
    let (batch, sequence, query_heads, kv_heads, head_dim) = (1, 4, 4, 2, 2);
    let q = deterministic_value(&[batch, sequence, query_heads, head_dim], 7, 0.029, 0.03);
    let k = deterministic_value(&[batch, sequence, kv_heads, head_dim], 13, 0.041, -0.13);
    let v = deterministic_value(&[batch, sequence, kv_heads, head_dim], 19, 0.053, 0.17);
    let evaluate = |keys: TensorValue, values: TensorValue| {
        gqa::gqa_attention(
            &Tensor::from_value_no_grad(q.clone()),
            &Tensor::from_value_no_grad(keys),
            &Tensor::from_value_no_grad(values),
            "causal.gqa",
        )
        .value()
    };
    let baseline = evaluate(k.clone(), v.clone());
    let future_token = sequence - 1;
    let future_start = future_token * kv_heads * head_dim;
    let future_indices = future_start..future_start + kv_heads * head_dim;
    let changed_k = evaluate(changed_at(&k, future_indices.clone(), 9.0), v.clone());
    let changed_v = evaluate(k, changed_at(&v, future_indices, -7.0));
    let earlier_output_count = future_token * query_heads * head_dim;

    assert_close(
        &changed_k.data[..earlier_output_count],
        &baseline.data[..earlier_output_count],
        0.0,
    );
    assert_close(
        &changed_v.data[..earlier_output_count],
        &baseline.data[..earlier_output_count],
        0.0,
    );
    assert!(
        changed_k.data[earlier_output_count..]
            .iter()
            .zip(&baseline.data[earlier_output_count..])
            .any(|(changed, original)| (changed - original).abs() > 1e-5),
        "the changed key must remain visible to its own query position"
    );
    assert!(
        changed_v.data[earlier_output_count..]
            .iter()
            .zip(&baseline.data[earlier_output_count..])
            .any(|(changed, original)| (changed - original).abs() > 1e-5),
        "the changed value must remain visible to its own query position"
    );
}

#[test]
fn each_batch_uses_only_its_own_keys_and_values() {
    let (batch, sequence, query_heads, kv_heads, head_dim) = (2, 3, 4, 2, 2);
    let q = deterministic_value(&[batch, sequence, query_heads, head_dim], 5, 0.033, -0.04);
    let k = deterministic_value(&[batch, sequence, kv_heads, head_dim], 11, 0.047, 0.08);
    let v = deterministic_value(&[batch, sequence, kv_heads, head_dim], 17, 0.039, -0.12);
    let evaluate = |keys: TensorValue, values: TensorValue| {
        gqa::gqa_attention(
            &Tensor::from_value_no_grad(q.clone()),
            &Tensor::from_value_no_grad(keys),
            &Tensor::from_value_no_grad(values),
            "batch.gqa",
        )
        .value()
    };
    let baseline = evaluate(k.clone(), v.clone());
    let kv_batch_size = sequence * kv_heads * head_dim;
    let second_batch = kv_batch_size..2 * kv_batch_size;
    let changed = evaluate(
        changed_at(&k, second_batch.clone(), 5.0),
        changed_at(&v, second_batch, -3.0),
    );
    let output_batch_size = sequence * query_heads * head_dim;

    assert_close(
        &changed.data[..output_batch_size],
        &baseline.data[..output_batch_size],
        0.0,
    );
    assert!(
        changed.data[output_batch_size..]
            .iter()
            .zip(&baseline.data[output_batch_size..])
            .any(|(changed, original)| (changed - original).abs() > 1e-5),
        "mutating batch 1 must still affect batch 1"
    );
}

#[test]
fn fused_gqa_records_one_named_node_with_q_k_v_edges_in_equation_order() {
    let q = Tensor::randn(&[1, 2, 4, 2]).requires_grad();
    let k = Tensor::randn(&[1, 2, 2, 2]).requires_grad();
    let v = Tensor::randn(&[1, 2, 2, 2]).requires_grad();
    let output = gqa::gqa_attention(&q, &k, &v, "topology.gqa");
    let graph = GraphObservation::from_outputs(&[&output]);
    let operations = graph.operations();

    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].kind, OpKind::GqaAttention);
    assert_eq!(operations[0].name, "topology.gqa");
    assert_eq!(operations[0].inputs.len(), 3);
    assert_eq!(
        operations[0].inputs,
        [
            GraphTensor {
                id: GraphTensorId(0),
                shape: q.shape(),
            },
            GraphTensor {
                id: GraphTensorId(1),
                shape: k.shape(),
            },
            GraphTensor {
                id: GraphTensorId(2),
                shape: v.shape(),
            },
        ]
    );
    assert_eq!(
        operations[0].outputs,
        [GraphTensor {
            id: GraphTensorId(3),
            shape: output.shape(),
        }]
    );
}

#[test]
#[should_panic(expected = "gqa: Hq must be positive")]
fn fused_gqa_rejects_zero_query_heads() {
    let q = Tensor::zeros(&[1, 1, 0, 2]);
    let k = Tensor::zeros(&[1, 1, 1, 2]);
    let v = Tensor::zeros(&[1, 1, 1, 2]);

    let _ = gqa::gqa_attention(&q, &k, &v, "invalid.gqa");
}

#[test]
#[should_panic(expected = "gqa: Hk must be positive")]
fn fused_gqa_rejects_zero_kv_heads() {
    let q = Tensor::zeros(&[1, 1, 2, 2]);
    let k = Tensor::zeros(&[1, 1, 0, 2]);
    let v = Tensor::zeros(&[1, 1, 0, 2]);

    let _ = gqa::gqa_attention(&q, &k, &v, "invalid.gqa");
}

#[test]
#[should_panic(expected = "gqa: Dh must be positive")]
fn fused_gqa_rejects_zero_head_dimension() {
    let q = Tensor::zeros(&[1, 1, 2, 0]);
    let k = Tensor::zeros(&[1, 1, 1, 0]);
    let v = Tensor::zeros(&[1, 1, 1, 0]);

    let _ = gqa::gqa_attention(&q, &k, &v, "invalid.gqa");
}
