use tnsr::dtensor::harness::TrainingStepScenario;
use tnsr::dtensor::{
    CollectiveKind, Layout, MeshAxis, MeshError, MeshTrace, MeshTraceEvent, ParallelDims5D,
    Placement, RankCoord5D, ShardError, ShardMap, TrainingPhase, MESH_SIM_TRACE_SCHEMA,
};
use tnsr::tensor::{Shape, TensorValue};

#[test]
fn unsharded_reference_training_step_is_deterministic_shape_wise() {
    let scenario = TrainingStepScenario::tiny_dense();
    let reference = scenario.run_unsharded_reference();

    assert!(reference.loss.is_finite());
    assert_eq!(reference.output_shape.0, vec![2, 4, 8]);
    assert_eq!(reference.parameter_grad_shapes.len(), 11);
    assert!(reference
        .parameter_grad_shapes
        .iter()
        .all(|shape| shape.numel() > 0));
}

#[test]
fn parallel_dims_validate_torchtitan_dense_product() {
    let dims = ParallelDims5D::new(2, 2, 3, 4, 5);
    assert_eq!(dims.world_size(), 240);
    assert_eq!(dims.batch_size(), 6);
    assert_eq!(dims.loss_size(), 24);
    assert_eq!(dims.fsdp_size(), 12);
    assert_eq!(dims.axis_size(MeshAxis::DpShard), 3);
    assert_eq!(dims.validate_world_size(240), Ok(()));
    assert_eq!(
        dims.validate_world_size(241),
        Err(MeshError::WorldSizeMismatch {
            expected: 240,
            got: 241
        })
    );
}

#[test]
fn parallel_dims_coords_are_row_major_with_tp_fastest() {
    let dims = ParallelDims5D::new(2, 2, 2, 2, 2);
    assert_eq!(
        dims.coord(0).unwrap(),
        RankCoord5D {
            pp: 0,
            dp_replicate: 0,
            dp_shard: 0,
            cp: 0,
            tp: 0
        }
    );
    assert_eq!(
        dims.coord(31).unwrap(),
        RankCoord5D {
            pp: 1,
            dp_replicate: 1,
            dp_shard: 1,
            cp: 1,
            tp: 1
        }
    );
    assert_eq!(
        dims.coord(32),
        Err(MeshError::RankOutOfRange {
            rank: 32,
            world_size: 32
        })
    );
}

#[test]
fn shard_map_slices_and_reconstructs_divisible_tensor() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let value = TensorValue::from_vec(
        Shape(vec![2, 4]),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
    );
    let map = ShardMap::new(value.shape.clone(), layout, dims).unwrap();

    assert_eq!(map.local_shape_for_rank(0).unwrap(), Shape(vec![2, 2]));
    let shards: Vec<_> = (0..dims.world_size())
        .map(|rank| map.shard_tensor_for_rank(&value, rank).unwrap())
        .collect();
    assert_eq!(shards[0].data.as_ref(), &vec![0.0, 1.0, 10.0, 11.0]);
    assert_eq!(shards[1].data.as_ref(), &vec![2.0, 3.0, 12.0, 13.0]);

    let reconstructed = map.reconstruct_from_rank_shards(&shards).unwrap();
    assert_eq!(reconstructed.shape, value.shape);
    assert_eq!(reconstructed.data.as_ref(), value.data.as_ref());
}

#[test]
fn layout_rejects_ambiguous_duplicate_shard_dim() {
    assert!(Layout::new(vec![
        (MeshAxis::Tp, Placement::Shard(1)),
        (MeshAxis::Cp, Placement::Shard(1)),
    ])
    .is_err());
}

#[test]
fn shard_tensor_rejects_out_of_range_shard_dim_without_panicking() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(2))]).unwrap();
    let value = TensorValue::from_vec(
        Shape(vec![2, 4]),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
    );
    let map = ShardMap::new(value.shape.clone(), layout, dims).unwrap();

    let err = match map.shard_tensor_for_rank(&value, 0) {
        Ok(_) => panic!("out-of-range shard dim unexpectedly produced a shard"),
        Err(err) => err,
    };
    assert_eq!(err, ShardError::ShardDimOutOfRange { dim: 2, rank: 0 });
}

#[test]
fn reconstruct_rejects_wrong_local_shard_shape_without_panicking() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let map = ShardMap::new(Shape(vec![2, 4]), layout, dims).unwrap();
    let shards = vec![
        TensorValue::from_vec(Shape(vec![2, 2]), vec![0.0, 1.0, 10.0, 11.0]),
        TensorValue::from_vec(Shape(vec![2, 3]), vec![2.0, 3.0, 4.0, 12.0, 13.0, 14.0]),
    ];

    let err = match map.reconstruct_from_rank_shards(&shards) {
        Ok(_) => panic!("wrong local shard shape unexpectedly reconstructed"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        ShardError::ShapeMismatch {
            expected: Shape(vec![2, 2]),
            got: Shape(vec![2, 3])
        }
    );
}

#[test]
fn mesh_trace_schema_and_events_are_stable() {
    let mut trace = MeshTrace::default();
    trace.record(MeshTraceEvent::Collective {
        phase: TrainingPhase::Backward,
        kind: CollectiveKind::AllReduce,
        axis: MeshAxis::DpReplicate,
        bytes: 128,
        ranks: vec![0, 1],
    });

    assert_eq!(trace.schema, MESH_SIM_TRACE_SCHEMA);
    assert_eq!(trace.events.len(), 1);
    assert_eq!(
        trace.events[0],
        MeshTraceEvent::Collective {
            phase: TrainingPhase::Backward,
            kind: CollectiveKind::AllReduce,
            axis: MeshAxis::DpReplicate,
            bytes: 128,
            ranks: vec![0, 1]
        }
    );
}
