use tnsr::dtensor::harness::TrainingStepScenario;
use tnsr::dtensor::{MeshAxis, MeshError, ParallelDims5D, RankCoord5D};

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
