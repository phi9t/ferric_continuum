use tnsr::{
    inference::{apply_rope, AttentionStep, ContinuousBatcher, DecodeRequest, KvCache, RopeConfig},
    tensor::{Shape, TensorValue},
};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn tv(shape: &[usize], data: &[f32]) -> TensorValue {
    TensorValue::from_vec(Shape(shape.to_vec()), data.to_vec())
}

#[test]
fn apply_rope_uses_absolute_positions_for_3d_values() {
    let cfg = RopeConfig::new(2, 10_000.0);
    let input = tv(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let out = apply_rope(&input, &[0, 1], &cfg);

    assert_eq!(out.shape, Shape(vec![1, 2, 2]));
    assert!(approx_eq(out.data[0], 1.0, 1e-6));
    assert!(approx_eq(out.data[1], 0.0, 1e-6));

    let (sin, cos) = 1.0_f32.sin_cos();
    assert!(approx_eq(out.data[2], -sin, 1e-6));
    assert!(approx_eq(out.data[3], cos, 1e-6));
}

#[test]
fn kv_cache_appends_prefixes_and_removes_sequences() {
    let mut cache = KvCache::new(2);
    assert!(cache.is_empty());
    assert_eq!(cache.head_dim(), 2);

    cache.append(7, tv(&[1, 2], &[1.0, 2.0]), tv(&[1, 2], &[10.0, 20.0]));
    assert_eq!(cache.position(7), 1);

    cache.append_at(
        7,
        1,
        tv(&[2, 2], &[3.0, 4.0, 5.0, 6.0]),
        tv(&[2, 2], &[30.0, 40.0, 50.0, 60.0]),
    );
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.position(7), 3);

    let prefix = cache.prefix(7).expect("prefix");
    assert_eq!(prefix.sequence_id, 7);
    assert_eq!(prefix.len, 3);
    assert_eq!(prefix.keys.shape, Shape(vec![3, 2]));
    assert_eq!(prefix.keys.data.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        prefix.values.data.as_ref(),
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );

    let removed = cache.remove(7).expect("removed");
    assert_eq!(removed.len, 3);
    assert!(cache.is_empty());
    assert!(cache.prefix(7).is_none());
}

#[test]
fn continuous_batcher_prefills_then_decodes_deterministically() {
    let mut batcher = ContinuousBatcher::new(2);
    batcher.enqueue(DecodeRequest::new(1, vec![10, 11], 4));
    batcher.enqueue(DecodeRequest::new(2, vec![20], 2));

    assert_eq!(batcher.stats().queued, 2);
    assert_eq!(
        batcher.next_batch(),
        vec![
            AttentionStep::Prefill {
                request_id: 1,
                token: 10,
                position: 0,
            },
            AttentionStep::Prefill {
                request_id: 1,
                token: 11,
                position: 1,
            },
        ]
    );

    assert_eq!(
        batcher.next_batch(),
        vec![AttentionStep::Prefill {
            request_id: 2,
            token: 20,
            position: 0,
        }]
    );

    batcher.finish_step(1, Some(12));
    batcher.finish_step(2, Some(21));
    assert_eq!(
        batcher.next_batch(),
        vec![
            AttentionStep::Decode {
                request_id: 1,
                token: 12,
                position: 2,
            },
            AttentionStep::Decode {
                request_id: 2,
                token: 21,
                position: 1,
            },
        ]
    );

    batcher.finish_step(1, Some(13));
    assert_eq!(
        batcher.next_batch(),
        vec![AttentionStep::Decode {
            request_id: 1,
            token: 13,
            position: 3,
        }]
    );
    assert!(batcher.next_batch().is_empty());
    assert!(batcher.is_empty());
}
