use tnsr::dtensor::harness::TrainingStepScenario;
use tnsr::dtensor::{
    CollectiveError, CollectiveKind, CollectiveSimulator, InjectionEvent, InjectionKind,
    InjectionPlan, Layout, MeshAxis, MeshError, MeshTrace, MeshTraceEvent, ParallelDims5D,
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
fn common_substrate_smoke_connects_reference_mesh_collective_and_injection() {
    let scenario = TrainingStepScenario::tiny_dense();
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let plan = InjectionPlan::new(vec![InjectionEvent {
        label: "missing_rank".to_string(),
        phase: TrainingPhase::Backward,
        rank: 0,
        axis: Some(MeshAxis::DpReplicate),
        kind: InjectionKind::MissingParticipant,
    }]);

    let report = scenario.run_common_substrate_smoke(dims, Some(&plan));

    assert!(report.reference_loss.is_finite());
    assert_eq!(report.world_size, 4);
    assert_eq!(report.trace_events, 1);
    assert_eq!(report.failures, 1);
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

#[test]
fn collective_all_reduce_sums_axis_groups_and_records_trace() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let shards = vec![
        TensorValue::from_vec(Shape(vec![2]), vec![1.0, 2.0]),
        TensorValue::from_vec(Shape(vec![2]), vec![10.0, 20.0]),
        TensorValue::from_vec(Shape(vec![2]), vec![100.0, 200.0]),
        TensorValue::from_vec(Shape(vec![2]), vec![1000.0, 2000.0]),
    ];
    let mut sim = CollectiveSimulator::new(dims);
    let out = sim
        .all_reduce_sum(MeshAxis::DpReplicate, TrainingPhase::Backward, &shards)
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(out[0].data.as_ref(), &vec![101.0, 202.0]);
    assert_eq!(out[1].data.as_ref(), &vec![1010.0, 2020.0]);
    assert_eq!(out[2].data.as_ref(), &vec![101.0, 202.0]);
    assert_eq!(out[3].data.as_ref(), &vec![1010.0, 2020.0]);
    assert_eq!(sim.trace.events.len(), 1);
}

#[test]
fn collective_all_gather_gathers_axis_groups_and_records_trace() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let shards = vec![
        TensorValue::from_vec(Shape(vec![1]), vec![1.0]),
        TensorValue::from_vec(Shape(vec![1]), vec![10.0]),
        TensorValue::from_vec(Shape(vec![1]), vec![100.0]),
        TensorValue::from_vec(Shape(vec![1]), vec![1000.0]),
    ];
    let mut sim = CollectiveSimulator::new(dims);
    let out = sim
        .all_gather(MeshAxis::DpReplicate, TrainingPhase::Forward, &shards)
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(out[0].shape, Shape(vec![2, 1]));
    assert_eq!(out[0].data.as_ref(), &vec![1.0, 100.0]);
    assert_eq!(out[1].data.as_ref(), &vec![10.0, 1000.0]);
    assert_eq!(out[2].data.as_ref(), &vec![1.0, 100.0]);
    assert_eq!(out[3].data.as_ref(), &vec![10.0, 1000.0]);
    assert_eq!(
        sim.trace.events[0],
        MeshTraceEvent::Collective {
            phase: TrainingPhase::Forward,
            kind: CollectiveKind::AllGather,
            axis: MeshAxis::DpReplicate,
            bytes: 2,
            ranks: vec![0, 2, 1, 3],
        }
    );
}

#[test]
fn collective_rejects_bad_rank_count_before_grouping() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let shards = vec![TensorValue::from_vec(Shape(vec![1]), vec![1.0])];
    let mut sim = CollectiveSimulator::new(dims);

    let err = match sim.all_reduce_sum(MeshAxis::DpReplicate, TrainingPhase::Backward, &shards) {
        Ok(_) => panic!("wrong rank count unexpectedly produced collective output"),
        Err(err) => err,
    };

    assert_eq!(
        err,
        CollectiveError::RankCountMismatch {
            expected: 4,
            got: 1
        }
    );
}

#[test]
fn injection_plan_corrupts_and_reports_deterministically() {
    let plan = InjectionPlan::new(vec![
        InjectionEvent {
            label: "rank0_nan".to_string(),
            phase: TrainingPhase::Backward,
            rank: 0,
            axis: Some(MeshAxis::DpReplicate),
            kind: InjectionKind::InjectNan { offset: 1 },
        },
        InjectionEvent {
            label: "rank0_missing".to_string(),
            phase: TrainingPhase::Backward,
            rank: 0,
            axis: Some(MeshAxis::DpReplicate),
            kind: InjectionKind::MissingParticipant,
        },
    ]);
    let mut shard = TensorValue::from_vec(Shape(vec![2]), vec![1.0, 2.0]);
    let failures = plan.apply_to_shard(TrainingPhase::Backward, 0, &mut shard);

    assert!(shard.data[1].is_nan());
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].rank, Some(0));
    assert_eq!(failures[0].axis, Some(MeshAxis::DpReplicate));
}
