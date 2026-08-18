use tnsr::dtensor::harness::TrainingStepScenario;

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
