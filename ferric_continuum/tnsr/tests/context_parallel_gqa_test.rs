//! Correctness tests for logical context-parallel grouped-query attention.

use std::rc::Rc;

use tnsr::{
    autograd::{Engine, GraphObservation, GraphOpIndex, OpKind},
    ops::{
        basic,
        context_parallel_gqa::{
            context_parallel_gqa_attention, raw_context_parallel_gqa_backward,
            raw_context_parallel_gqa_forward,
        },
        gqa,
    },
    qwen3::{Qwen3Attention, Qwen3Config},
    tensor::{Shape, Tensor, TensorValue},
};

fn deterministic_value(shape: &[usize], scale: f32, offset: f32) -> TensorValue {
    let n: usize = shape.iter().product();
    let data = (0..n)
        .map(|i| (((i * 17 + 5) % 29) as f32 - 14.0) * scale + offset)
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

/// Split `[B,T,H,Dh]` into equal contiguous sequence shards while preserving
/// the batch-major layout within every shard.
fn split_sequence(value: &TensorValue, cp: usize) -> Vec<TensorValue> {
    let shape = &value.shape.0;
    assert_eq!(shape.len(), 4);
    let (b, t, h, dh) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(t % cp, 0);
    let local_t = t / cp;
    let src = value.data.as_ref();

    (0..cp)
        .map(|rank| {
            let mut shard = vec![0.0f32; b * local_t * h * dh];
            for bi in 0..b {
                for local_ti in 0..local_t {
                    let global_ti = rank * local_t + local_ti;
                    for hi in 0..h {
                        for di in 0..dh {
                            let src_i = ((bi * t + global_ti) * h + hi) * dh + di;
                            let dst_i = ((bi * local_t + local_ti) * h + hi) * dh + di;
                            shard[dst_i] = src[src_i];
                        }
                    }
                }
            }
            TensorValue::from_vec(Shape(vec![b, local_t, h, dh]), shard)
        })
        .collect()
}

/// Reassemble equal `[B,S,H,Dh]` shards into batch-major `[B,P*S,H,Dh]`.
fn join_sequence(shards: &[TensorValue]) -> TensorValue {
    assert!(!shards.is_empty());
    let shape = &shards[0].shape.0;
    let (b, local_t, h, dh) = (shape[0], shape[1], shape[2], shape[3]);
    let global_t = local_t * shards.len();
    let mut full = vec![0.0f32; b * global_t * h * dh];

    for (rank, shard) in shards.iter().enumerate() {
        let src = shard.data.as_ref();
        for bi in 0..b {
            for local_ti in 0..local_t {
                let global_ti = rank * local_t + local_ti;
                for hi in 0..h {
                    for di in 0..dh {
                        let src_i = ((bi * local_t + local_ti) * h + hi) * dh + di;
                        let dst_i = ((bi * global_t + global_ti) * h + hi) * dh + di;
                        full[dst_i] = src[src_i];
                    }
                }
            }
        }
    }

    TensorValue::from_vec(Shape(vec![b, global_t, h, dh]), full)
}

fn split_hidden_sequence(value: &TensorValue, cp: usize) -> Vec<TensorValue> {
    let shape = &value.shape.0;
    assert_eq!(shape.len(), 3);
    let (b, t, d) = (shape[0], shape[1], shape[2]);
    assert_eq!(t % cp, 0);
    let local_t = t / cp;
    let src = value.data.as_ref();
    (0..cp)
        .map(|rank| {
            let mut shard = vec![0.0f32; b * local_t * d];
            for bi in 0..b {
                for local_ti in 0..local_t {
                    let global_ti = rank * local_t + local_ti;
                    let src_base = (bi * t + global_ti) * d;
                    let dst_base = (bi * local_t + local_ti) * d;
                    shard[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
                }
            }
            TensorValue::from_vec(Shape(vec![b, local_t, d]), shard)
        })
        .collect()
}

fn join_hidden_sequence(shards: &[TensorValue]) -> TensorValue {
    assert!(!shards.is_empty());
    let shape = &shards[0].shape.0;
    let (b, local_t, d) = (shape[0], shape[1], shape[2]);
    let global_t = local_t * shards.len();
    let mut full = vec![0.0f32; b * global_t * d];
    for (rank, shard) in shards.iter().enumerate() {
        let src = shard.data.as_ref();
        for bi in 0..b {
            for local_ti in 0..local_t {
                let global_ti = rank * local_t + local_ti;
                let src_base = (bi * local_t + local_ti) * d;
                let dst_base = (bi * global_t + global_ti) * d;
                full[dst_base..dst_base + d].copy_from_slice(&src[src_base..src_base + d]);
            }
        }
    }
    TensorValue::from_vec(Shape(vec![b, global_t, d]), full)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "value mismatch at {i}: actual={a:.7} expected={e:.7} diff={:.7}",
            (a - e).abs()
        );
    }
}

fn dot_loss(outputs: &[TensorValue], dout: &[TensorValue]) -> f32 {
    outputs
        .iter()
        .zip(dout.iter())
        .map(|(out, grad)| {
            out.data
                .iter()
                .zip(grad.data.iter())
                .map(|(x, g)| x * g)
                .sum::<f32>()
        })
        .sum()
}

fn perturbed(value: &TensorValue, index: usize, delta: f32) -> TensorValue {
    let mut data = value.data.as_ref().clone();
    data[index] += delta;
    TensorValue::from_vec(value.shape.clone(), data)
}

fn summed_loss(outputs: &[Tensor], prefix: &str) -> Tensor {
    assert!(!outputs.is_empty());
    let mut loss = basic::sum(&outputs[0], &format!("{prefix}.sum0"));
    for (rank, output) in outputs.iter().enumerate().skip(1) {
        let shard_loss = basic::sum(output, &format!("{prefix}.sum{rank}"));
        loss = basic::add(&loss, &shard_loss, &format!("{prefix}.add{rank}"));
    }
    loss
}

#[test]
fn raw_forward_matches_unsharded_gqa_for_multiple_cp_degrees_and_batches() {
    // B=2 catches the tempting but wrong implementation that concatenates flat
    // shard buffers instead of gathering each batch's sequence rows.
    let (b, t, hq, hk, dh) = (2, 4, 4, 2, 4);
    let q = deterministic_value(&[b, t, hq, dh], 0.025, -0.1);
    let k = deterministic_value(&[b, t, hk, dh], 0.035, 0.05);
    let v = deterministic_value(&[b, t, hk, dh], 0.045, -0.2);

    let reference = gqa::gqa_attention(
        &Tensor::from_value_no_grad(q.clone()),
        &Tensor::from_value_no_grad(k.clone()),
        &Tensor::from_value_no_grad(v.clone()),
        "reference",
    );
    let reference_value = reference.value();

    for cp in [1usize, 2, 4] {
        let q_shards = split_sequence(&q, cp);
        let k_shards = split_sequence(&k, cp);
        let v_shards = split_sequence(&v, cp);
        let (outputs, saved) = raw_context_parallel_gqa_forward(&q_shards, &k_shards, &v_shards);
        let actual = join_sequence(&outputs);

        assert_eq!(saved.shape.cp, cp);
        assert_eq!(saved.shape.global_t, t);
        assert_eq!(actual.shape, reference_value.shape);
        assert_close(actual.data.as_ref(), reference_value.data.as_ref(), 1e-6);
    }
}

#[test]
fn raw_backward_matches_finite_differences_for_every_shard_input() {
    let (b, t, hq, hk, dh, cp) = (1, 4, 2, 1, 2, 2);
    let q = deterministic_value(&[b, t, hq, dh], 0.035, -0.1);
    let k = deterministic_value(&[b, t, hk, dh], 0.045, 0.2);
    let v = deterministic_value(&[b, t, hk, dh], 0.055, -0.15);
    let q_shards = split_sequence(&q, cp);
    let k_shards = split_sequence(&k, cp);
    let v_shards = split_sequence(&v, cp);
    let (outputs, saved) = raw_context_parallel_gqa_forward(&q_shards, &k_shards, &v_shards);
    let dout: Vec<TensorValue> = outputs
        .iter()
        .enumerate()
        .map(|(rank, output)| {
            deterministic_value(&output.shape.0, 0.025 + rank as f32 * 0.01, -0.05)
        })
        .collect();
    let analytic =
        raw_context_parallel_gqa_backward(&dout, &q_shards, &k_shards, &v_shards, &saved);

    let eps = 1e-3f32;
    let tol = 4e-3f32;
    for group in 0..3 {
        let (base, grads) = match group {
            0 => (&q_shards, &analytic.dq),
            1 => (&k_shards, &analytic.dk),
            _ => (&v_shards, &analytic.dv),
        };
        for rank in 0..cp {
            for index in 0..base[rank].shape.numel() {
                let mut plus = base.clone();
                plus[rank] = perturbed(&base[rank], index, eps);
                let mut minus = base.clone();
                minus[rank] = perturbed(&base[rank], index, -eps);

                let eval = |candidate: &[TensorValue]| {
                    let (q_eval, k_eval, v_eval) = match group {
                        0 => (candidate, k_shards.as_slice(), v_shards.as_slice()),
                        1 => (q_shards.as_slice(), candidate, v_shards.as_slice()),
                        _ => (q_shards.as_slice(), k_shards.as_slice(), candidate),
                    };
                    let (out, _) = raw_context_parallel_gqa_forward(q_eval, k_eval, v_eval);
                    dot_loss(&out, &dout)
                };
                let numerical = (eval(&plus) - eval(&minus)) / (2.0 * eps);
                let actual = grads[rank].data[index];
                assert!(
                    (actual - numerical).abs() <= tol,
                    "gradient mismatch group={group} rank={rank} index={index}: \
                     analytic={actual:.6} numerical={numerical:.6} diff={:.6}",
                    (actual - numerical).abs()
                );
            }
        }
    }
}

#[test]
fn raw_backward_matches_unsharded_gqa_for_nonuniform_upstream_gradients_at_every_degree() {
    let (b, t, hq, hk, dh) = (2, 4, 4, 2, 2);
    let q = deterministic_value(&[b, t, hq, dh], 0.025, -0.1);
    let k = deterministic_value(&[b, t, hk, dh], 0.035, 0.05);
    let v = deterministic_value(&[b, t, hk, dh], 0.045, -0.2);
    let dout = deterministic_value(&[b, t, hq, dh], 0.019, 0.07);

    let q_full = Tensor::from_value(q.clone(), true);
    let k_full = Tensor::from_value(k.clone(), true);
    let v_full = Tensor::from_value(v.clone(), true);
    let output = gqa::gqa_attention(&q_full, &k_full, &v_full, "weighted.reference");
    let weighted = basic::mul(
        &output,
        &Tensor::from_value_no_grad(dout.clone()),
        "weighted.reference.multiply",
    );
    let mut engine = Engine::new();
    engine.backward(&basic::sum(&weighted, "weighted.reference.loss"));
    let expected_dq = q_full.grad().unwrap();
    let expected_dk = k_full.grad().unwrap();
    let expected_dv = v_full.grad().unwrap();

    for cp in [1usize, 2, 4] {
        let q_shards = split_sequence(&q, cp);
        let k_shards = split_sequence(&k, cp);
        let v_shards = split_sequence(&v, cp);
        let dout_shards = split_sequence(&dout, cp);
        let (_, saved) = raw_context_parallel_gqa_forward(&q_shards, &k_shards, &v_shards);
        let actual = raw_context_parallel_gqa_backward(
            &dout_shards,
            &q_shards,
            &k_shards,
            &v_shards,
            &saved,
        );

        assert_close(
            join_sequence(&actual.dq).data.as_ref(),
            expected_dq.data.as_ref(),
            2e-6,
        );
        assert_close(
            join_sequence(&actual.dk).data.as_ref(),
            expected_dk.data.as_ref(),
            2e-6,
        );
        assert_close(
            join_sequence(&actual.dv).data.as_ref(),
            expected_dv.data.as_ref(),
            2e-6,
        );
    }
}

#[test]
fn autograd_shard_gradients_match_unsharded_gqa() {
    // B=2 and one token per rank jointly exercise batch-major gradient
    // accumulation and owner scattering across every sequence boundary.
    let (b, t, hq, hk, dh, cp) = (2, 4, 4, 2, 2, 4);
    let q_value = deterministic_value(&[b, t, hq, dh], 0.025, -0.1);
    let k_value = deterministic_value(&[b, t, hk, dh], 0.035, 0.05);
    let v_value = deterministic_value(&[b, t, hk, dh], 0.045, -0.2);

    let q_full = Tensor::from_value(q_value.clone(), true);
    let k_full = Tensor::from_value(k_value.clone(), true);
    let v_full = Tensor::from_value(v_value.clone(), true);
    let mut reference_engine = Engine::new();
    let reference_output = gqa::gqa_attention(&q_full, &k_full, &v_full, "reference");
    reference_engine.backward(&basic::sum(&reference_output, "reference.loss"));

    let q_shards: Vec<Tensor> = split_sequence(&q_value, cp)
        .into_iter()
        .map(|v| Tensor::from_value(v, true))
        .collect();
    let k_shards: Vec<Tensor> = split_sequence(&k_value, cp)
        .into_iter()
        .map(|v| Tensor::from_value(v, true))
        .collect();
    let v_shards: Vec<Tensor> = split_sequence(&v_value, cp)
        .into_iter()
        .map(|v| Tensor::from_value(v, true))
        .collect();
    let mut cp_engine = Engine::new();
    let outputs =
        context_parallel_gqa_attention(&q_shards, &k_shards, &v_shards, "context_parallel");
    cp_engine.backward(&summed_loss(&outputs, "context_parallel.loss"));

    let q_grads: Vec<TensorValue> = q_shards.iter().map(|t| t.grad().unwrap()).collect();
    let k_grads: Vec<TensorValue> = k_shards.iter().map(|t| t.grad().unwrap()).collect();
    let v_grads: Vec<TensorValue> = v_shards.iter().map(|t| t.grad().unwrap()).collect();
    assert_close(
        join_sequence(&q_grads).data.as_ref(),
        q_full.grad().unwrap().data.as_ref(),
        1e-6,
    );
    assert_close(
        join_sequence(&k_grads).data.as_ref(),
        k_full.grad().unwrap().data.as_ref(),
        1e-6,
    );
    assert_close(
        join_sequence(&v_grads).data.as_ref(),
        v_full.grad().unwrap().data.as_ref(),
        1e-6,
    );
}

#[test]
fn cross_shard_causal_visibility_has_the_correct_direction() {
    let (b, t, hq, hk, dh, cp) = (1, 4, 2, 1, 2, 2);
    let q = deterministic_value(&[b, t, hq, dh], 0.03, 0.1);
    let k = deterministic_value(&[b, t, hk, dh], 0.04, -0.1);
    let v = deterministic_value(&[b, t, hk, dh], 0.05, 0.2);
    let q_shards = split_sequence(&q, cp);
    let k_shards = split_sequence(&k, cp);
    let v_shards = split_sequence(&v, cp);
    let (baseline, _) = raw_context_parallel_gqa_forward(&q_shards, &k_shards, &v_shards);

    let mut future_v = v_shards.clone();
    let mut future_data = future_v[1].data.as_ref().clone();
    for x in &mut future_data {
        *x += 7.0;
    }
    future_v[1] = TensorValue::from_vec(future_v[1].shape.clone(), future_data);
    let (future_changed, _) = raw_context_parallel_gqa_forward(&q_shards, &k_shards, &future_v);
    assert_close(
        baseline[0].data.as_ref(),
        future_changed[0].data.as_ref(),
        0.0,
    );

    let mut past_v = v_shards.clone();
    let mut past_data = past_v[0].data.as_ref().clone();
    for x in &mut past_data {
        *x += 7.0;
    }
    past_v[0] = TensorValue::from_vec(past_v[0].shape.clone(), past_data);
    let (past_changed, _) = raw_context_parallel_gqa_forward(&q_shards, &k_shards, &past_v);
    assert!(
        baseline[1]
            .data
            .iter()
            .zip(past_changed[1].data.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "later queries must see values owned by earlier ranks"
    );
}

#[test]
fn unused_output_contributes_zero_query_gradient() {
    let (b, t, hq, hk, dh, cp) = (1, 4, 2, 1, 2, 2);
    let q = deterministic_value(&[b, t, hq, dh], 0.03, 0.1);
    let k = deterministic_value(&[b, t, hk, dh], 0.04, -0.1);
    let v = deterministic_value(&[b, t, hk, dh], 0.05, 0.2);
    let q_shards: Vec<Tensor> = split_sequence(&q, cp)
        .into_iter()
        .map(|x| Tensor::from_value(x, true))
        .collect();
    let k_shards: Vec<Tensor> = split_sequence(&k, cp)
        .into_iter()
        .map(|x| Tensor::from_value(x, true))
        .collect();
    let v_shards: Vec<Tensor> = split_sequence(&v, cp)
        .into_iter()
        .map(|x| Tensor::from_value(x, true))
        .collect();

    let mut engine = Engine::new();
    let outputs = context_parallel_gqa_attention(&q_shards, &k_shards, &v_shards, "cp");
    engine.backward(&basic::sum(&outputs[0], "only_rank0"));

    let unused_q_grad = q_shards[1].grad().expect("rank 1 Q gradient");
    assert!(unused_q_grad.data.iter().all(|&x| x == 0.0));
}

#[test]
fn debug_records_one_context_parallel_multi_output_operation() {
    let q0 = Tensor::randn(&[1, 1, 2, 2]).requires_grad();
    let q1 = Tensor::randn(&[1, 1, 2, 2]).requires_grad();
    let k0 = Tensor::randn(&[1, 1, 1, 2]).requires_grad();
    let k1 = Tensor::randn(&[1, 1, 1, 2]).requires_grad();
    let v0 = Tensor::randn(&[1, 1, 1, 2]).requires_grad();
    let v1 = Tensor::randn(&[1, 1, 1, 2]).requires_grad();
    let mut engine = Engine::new();
    let outputs = context_parallel_gqa_attention(&[q0, q1], &[k0, k1], &[v0, v1], "cp.debug");
    let graph = GraphObservation::from_outputs(&[&outputs[0], &outputs[1]]);
    assert_eq!(graph.operations().len(), 1);
    assert_eq!(
        graph.operations()[0].kind,
        OpKind::ContextParallelGqaAttention
    );
    assert_eq!(graph.operations()[0].name, "cp.debug");
    assert_eq!(graph.operations()[0].outputs.len(), 2);
    assert_eq!(graph.producer_of(&outputs[0]), Ok(Some(GraphOpIndex(0))));
    assert_eq!(graph.producer_of(&outputs[1]), Ok(Some(GraphOpIndex(0))));

    engine.backward(&basic::sum(&outputs[0], "loss"));
}

#[test]
#[should_panic(expected = "context_parallel_gqa: at least one shard is required")]
fn empty_shards_are_rejected() {
    raw_context_parallel_gqa_forward(&[], &[], &[]);
}

#[test]
#[should_panic(expected = "context_parallel_gqa: Q and K shard counts must match")]
fn mismatched_shard_counts_are_rejected() {
    let q = deterministic_value(&[1, 2, 2, 2], 0.1, 0.0);
    raw_context_parallel_gqa_forward(&[q], &[], &[]);
}

#[test]
#[should_panic(expected = "context_parallel_gqa: every Q shard must have the same shape")]
fn unequal_local_sequence_lengths_are_rejected() {
    let q0 = deterministic_value(&[1, 2, 2, 2], 0.1, 0.0);
    let q1 = deterministic_value(&[1, 1, 2, 2], 0.1, 0.0);
    let k0 = deterministic_value(&[1, 2, 1, 2], 0.1, 0.0);
    let k1 = deterministic_value(&[1, 1, 1, 2], 0.1, 0.0);
    let v0 = k0.clone();
    let v1 = k1.clone();
    raw_context_parallel_gqa_forward(&[q0, q1], &[k0, k1], &[v0, v1]);
}

#[test]
#[should_panic(expected = "context_parallel_gqa: Hq must be divisible by Hk")]
fn incompatible_gqa_head_counts_are_rejected() {
    let q = deterministic_value(&[1, 2, 3, 2], 0.1, 0.0);
    let k = deterministic_value(&[1, 2, 2, 2], 0.1, 0.0);
    let v = k.clone();
    raw_context_parallel_gqa_forward(&[q], &[k], &[v]);
}

#[test]
#[should_panic(expected = "context_parallel_gqa: Hq must be positive")]
fn zero_query_heads_are_rejected() {
    let q = deterministic_value(&[1, 1, 0, 2], 0.1, 0.0);
    let k = deterministic_value(&[1, 1, 1, 2], 0.1, 0.0);
    let v = k.clone();

    raw_context_parallel_gqa_forward(&[q], &[k], &[v]);
}

#[test]
#[should_panic(expected = "context_parallel_gqa: Dh must be positive")]
fn zero_head_dimension_is_rejected() {
    let q = deterministic_value(&[1, 1, 2, 0], 0.1, 0.0);
    let k = deterministic_value(&[1, 1, 1, 0], 0.1, 0.0);
    let v = k.clone();

    raw_context_parallel_gqa_forward(&[q], &[k], &[v]);
}

#[test]
#[should_panic(expected = "context_parallel_gqa: global sequence length overflow")]
fn global_sequence_length_overflow_is_rejected() {
    let impossible_q = TensorValue {
        shape: Shape(vec![1, usize::MAX, 2, 2]),
        data: Rc::new(Vec::new()),
    };
    let impossible_kv = TensorValue {
        shape: Shape(vec![1, usize::MAX, 1, 2]),
        data: Rc::new(Vec::new()),
    };

    raw_context_parallel_gqa_forward(
        &[impossible_q.clone(), impossible_q],
        &[impossible_kv.clone(), impossible_kv.clone()],
        &[impossible_kv.clone(), impossible_kv],
    );
}

#[test]
fn qwen3_attention_context_parallel_forward_matches_unsharded() {
    let cfg = Qwen3Config::tiny();
    let attention = Qwen3Attention::new(&cfg);
    let x_value = deterministic_value(&[2, 4, cfg.hidden_size], 0.015, -0.05);
    let reference = attention.forward(&Tensor::from_value_no_grad(x_value.clone()));
    let x_shards: Vec<Tensor> = split_hidden_sequence(&x_value, 2)
        .into_iter()
        .map(Tensor::from_value_no_grad)
        .collect();
    let outputs = attention.forward_context_parallel(&x_shards);
    let output_values: Vec<TensorValue> = outputs.iter().map(Tensor::value).collect();
    let actual = join_hidden_sequence(&output_values);
    assert_close(actual.data.as_ref(), reference.value().data.as_ref(), 1e-5);
}

#[test]
fn qwen3_attention_context_parallel_backward_matches_unsharded() {
    let cfg = Qwen3Config::tiny();
    let seed = Qwen3Attention::new(&cfg);
    let full_attention = clone_attention(&seed);
    let cp_attention = clone_attention(&seed);
    let x_value = deterministic_value(&[2, 4, cfg.hidden_size], 0.015, -0.05);
    let full_input = Tensor::from_value(x_value.clone(), true);
    let x_shards: Vec<Tensor> = split_hidden_sequence(&x_value, 2)
        .into_iter()
        .map(|x| Tensor::from_value(x, true))
        .collect();

    let mut full_engine = Engine::new();
    let full_output = full_attention.forward(&full_input);
    full_engine.backward(&basic::sum(&full_output, "qwen.full.loss"));

    let mut cp_engine = Engine::new();
    let cp_outputs = cp_attention.forward_context_parallel(&x_shards);
    cp_engine.backward(&summed_loss(&cp_outputs, "qwen.cp.loss"));

    let cp_input_grads: Vec<TensorValue> = x_shards
        .iter()
        .map(|input| input.grad().expect("context-parallel input gradient"))
        .collect();
    assert_close(
        join_hidden_sequence(&cp_input_grads).data.as_ref(),
        full_input
            .grad()
            .expect("unsharded input gradient")
            .data
            .as_ref(),
        2e-5,
    );
    for (cp_parameter, full_parameter) in cp_attention
        .parameters()
        .into_iter()
        .zip(full_attention.parameters())
    {
        assert_close(
            cp_parameter
                .grad()
                .expect("context-parallel parameter gradient")
                .data
                .as_ref(),
            full_parameter
                .grad()
                .expect("unsharded parameter gradient")
                .data
                .as_ref(),
            2e-5,
        );
    }
}
