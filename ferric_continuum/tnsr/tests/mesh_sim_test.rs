use tnsr::dtensor::harness::TrainingStepScenario;
use tnsr::dtensor::{
    redistribute_native_training_step_boundary, run_tiny_dense_mesh_simulation, CollectiveError,
    CollectiveKind, CollectiveSimulator, DTensor, DtError, DualPipeChunk, DualPipeComponentKind,
    DualPipeSchedule, InjectionEvent, InjectionKind, InjectionPlan, Layout, MeshAxis, MeshError,
    MeshSimCollectiveRecord, MeshSimPlan, MeshSimStep, MeshTrace, MeshTraceEvent,
    NativeLayoutError, ParallelDims5D, Placement, PlanError, RankCoord5D, ReduceOp, ShardError,
    ShardMap, TensorLayoutMeta, TrainingPhase, MESH_SIM_TRACE_SCHEMA,
};
use tnsr::ops::basic;
use tnsr::tensor::{Shape, Tensor, TensorValue};

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
        collective: Some(CollectiveKind::AllReduce),
        kind: InjectionKind::MissingParticipant,
    }]);

    let report = scenario.run_common_substrate_smoke(dims, Some(&plan));

    assert!(report.reference_loss.is_finite());
    assert_eq!(report.world_size, 4);
    assert_eq!(report.trace_events, 3);
    assert_eq!(report.failures, 1);
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Injection { label, .. } if label == "missing_rank")
    ));
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Failure(record) if record.message == "missing_rank")
    ));
}

#[test]
fn wrapper_dtensor_harness_runs_minimum_slice_and_matches_reference_shape() {
    let scenario = TrainingStepScenario::tiny_dense();
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);

    let report = scenario
        .run_wrapper_dtensor_tracer_bullet(dims, None)
        .unwrap();

    assert!(report.reference_loss.is_finite());
    assert_eq!(report.reference_output_shape, Shape(vec![2, 4, 8]));
    assert_eq!(report.activation_full_shape, Shape(vec![2, 4, 8]));
    assert!(report.activation_matches_reference);
    assert_eq!(report.activation_tp_shard_shape, Shape(vec![2, 4, 4]));
    assert_eq!(report.partial_dp_shard_shape, Shape(vec![2, 2]));
    assert_eq!(report.partial_replicated_shape, Shape(vec![2, 4]));
    assert_eq!(report.loss_reduction_supported, false);
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllGather,
                axis: MeshAxis::Tp,
                ..
            }
        )
    }));
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::ReduceScatter,
                axis: MeshAxis::DpShard,
                ..
            }
        )
    }));
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllReduce,
                axis: MeshAxis::DpReplicate,
                ..
            }
        )
    }));
}

#[test]
fn wrapper_dtensor_harness_injection_records_failure_context() {
    let scenario = TrainingStepScenario::tiny_dense();
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);
    let plan = InjectionPlan::new(vec![InjectionEvent {
        label: "wrapper_missing_reduce_scatter_rank0".to_string(),
        phase: TrainingPhase::Backward,
        rank: 0,
        axis: Some(MeshAxis::DpShard),
        collective: Some(CollectiveKind::ReduceScatter),
        kind: InjectionKind::MissingParticipant,
    }]);

    let report = scenario
        .run_wrapper_dtensor_tracer_bullet(dims, Some(&plan))
        .unwrap();

    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].phase, TrainingPhase::Backward);
    assert_eq!(report.failures[0].rank, Some(0));
    assert_eq!(report.failures[0].axis, Some(MeshAxis::DpShard));
    assert_eq!(
        report.failures[0].collective,
        Some(CollectiveKind::ReduceScatter)
    );
    assert_eq!(
        report.failures[0].tensor,
        Some("wrapper_partial".to_string())
    );
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Injection { label, .. } if label == "wrapper_missing_reduce_scatter_rank0")
    ));
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Failure(record) if record.message == "wrapper_missing_reduce_scatter_rank0")
    ));
}

#[test]
fn tensor_layout_meta_attaches_without_changing_value() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let value = TensorValue::from_vec(Shape(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]);
    let tensor = Tensor::from_value_no_grad(value.clone());
    let original_id = tensor.id();
    let meta = TensorLayoutMeta {
        global_shape: Shape(vec![2, 4]),
        layout: layout.clone(),
        dims,
        rank: 1,
    };

    let tensor = tensor.with_layout_meta(meta.clone());

    assert_eq!(tensor.id(), original_id);
    assert_eq!(tensor.shape(), value.shape);
    assert_eq!(
        tensor.inner.borrow().value.data.as_ref(),
        value.data.as_ref()
    );
    assert_eq!(tensor.layout_meta(), Some(meta));
}

#[test]
fn tensor_without_layout_meta_preserves_existing_local_behavior() {
    let lhs = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![2, 2]),
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let rhs = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![2, 2]),
        vec![10.0, 20.0, 30.0, 40.0],
    ));

    let out = basic::add(&lhs, &rhs, "local_add_stays_local");

    assert_eq!(out.shape(), Shape(vec![2, 2]));
    assert_eq!(
        out.inner.borrow().value.data.as_ref(),
        &vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(lhs.layout_meta(), None);
    assert_eq!(rhs.layout_meta(), None);
    assert_eq!(out.layout_meta(), None);
}

#[test]
fn local_op_drops_layout_meta_without_explicit_native_boundary() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let layout = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)]).unwrap();
    let meta = TensorLayoutMeta {
        global_shape: Shape(vec![2, 2]),
        layout,
        dims,
        rank: 0,
    };
    let lhs = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![2, 2]),
        vec![1.0, 2.0, 3.0, 4.0],
    ))
    .with_layout_meta(meta);
    let rhs = Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![2, 2]),
        vec![10.0, 20.0, 30.0, 40.0],
    ));

    let out = basic::add(&lhs, &rhs, "add_does_not_propagate_layout");

    assert_eq!(
        out.inner.borrow().value.data.as_ref(),
        &vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(out.layout_meta(), None);
}

#[test]
fn native_layout_training_boundary_consumes_metadata_and_traces() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let src_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)]).unwrap();
    let dst_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let value = TensorValue::from_vec(
        Shape(vec![2, 4]),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
    );
    let tensors: Vec<Tensor> = (0..dims.world_size())
        .map(|rank| {
            Tensor::from_value_no_grad(value.clone()).with_layout_meta(TensorLayoutMeta {
                global_shape: value.shape.clone(),
                layout: src_layout.clone(),
                dims,
                rank,
            })
        })
        .collect();
    let mut sim = CollectiveSimulator::new(dims);

    let sharded = redistribute_native_training_step_boundary(
        &tensors,
        dst_layout.clone(),
        &mut sim,
        TrainingPhase::Forward,
        "native_activation",
    )
    .unwrap();

    assert_eq!(sharded.len(), 2);
    assert_eq!(sharded[0].shape(), Shape(vec![2, 2]));
    assert_eq!(
        sharded[0].inner.borrow().value.data.as_ref(),
        &vec![0.0, 1.0, 10.0, 11.0]
    );
    assert_eq!(sharded[1].shape(), Shape(vec![2, 2]));
    assert_eq!(
        sharded[1].inner.borrow().value.data.as_ref(),
        &vec![2.0, 3.0, 12.0, 13.0]
    );
    assert_eq!(
        sharded[1].layout_meta(),
        Some(TensorLayoutMeta {
            global_shape: Shape(vec![2, 4]),
            layout: dst_layout,
            dims,
            rank: 1,
        })
    );
    assert!(sim.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::LayoutTransition {
                phase: TrainingPhase::Forward,
                tensor,
                ..
            } if tensor == "native_activation"
        )
    }));
}

#[test]
fn native_layout_unsupported_propagation_fails_loudly() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let src_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let dst_layout =
        Layout::new(vec![(MeshAxis::DpShard, Placement::Partial(ReduceOp::Sum))]).unwrap();
    let value = TensorValue::from_vec(Shape(vec![2, 2]), vec![0.0, 1.0, 10.0, 11.0]);
    let tensors: Vec<Tensor> = (0..dims.world_size())
        .map(|rank| {
            Tensor::from_value_no_grad(value.clone()).with_layout_meta(TensorLayoutMeta {
                global_shape: Shape(vec![2, 4]),
                layout: src_layout.clone(),
                dims,
                rank,
            })
        })
        .collect();
    let mut sim = CollectiveSimulator::new(dims);

    let err = match redistribute_native_training_step_boundary(
        &tensors,
        dst_layout,
        &mut sim,
        TrainingPhase::Backward,
        "unsupported_native_boundary",
    ) {
        Ok(_) => panic!("unsupported native layout propagation unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        NativeLayoutError::UnsupportedPropagation { .. }
    ));
    assert!(sim.trace.events.is_empty());
}

#[test]
fn native_layout_rejects_simulator_dims_mismatch_with_same_world_size() {
    let meta_dims = ParallelDims5D::new(1, 1, 1, 1, 4);
    let sim_dims = ParallelDims5D::new(1, 1, 2, 1, 2);
    assert_eq!(meta_dims.world_size(), sim_dims.world_size());
    let src_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)]).unwrap();
    let dst_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let value = TensorValue::from_vec(
        Shape(vec![2, 4]),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
    );
    let tensors: Vec<Tensor> = (0..meta_dims.world_size())
        .map(|rank| {
            Tensor::from_value_no_grad(value.clone()).with_layout_meta(TensorLayoutMeta {
                global_shape: value.shape.clone(),
                layout: src_layout.clone(),
                dims: meta_dims,
                rank,
            })
        })
        .collect();
    let mut sim = CollectiveSimulator::new(sim_dims);

    let err = match redistribute_native_training_step_boundary(
        &tensors,
        dst_layout,
        &mut sim,
        TrainingPhase::Forward,
        "native_mismatched_mesh",
    ) {
        Ok(_) => panic!("native boundary accepted a simulator with mismatched mesh factorization"),
        Err(err) => err,
    };

    assert_eq!(
        err,
        NativeLayoutError::SimulatorDimsMismatch {
            metadata: meta_dims,
            simulator: sim_dims,
        }
    );
    assert!(sim.trace.events.is_empty());
}

#[test]
fn native_layout_harness_runs_minimum_slice_and_matches_reference_shape() {
    let scenario = TrainingStepScenario::tiny_dense();
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);

    let report = scenario
        .run_native_layout_tracer_bullet(dims, None)
        .unwrap();

    assert!(report.reference_loss.is_finite());
    assert_eq!(report.reference_output_shape, Shape(vec![2, 4, 8]));
    assert_eq!(report.activation_full_shape, Shape(vec![2, 4, 8]));
    assert!(report.activation_matches_reference);
    assert_eq!(report.activation_tp_shard_shape, Shape(vec![2, 4, 4]));
    assert_eq!(report.partial_dp_shard_shape, Shape(vec![2, 2]));
    assert_eq!(report.partial_replicated_shape, Shape(vec![2, 4]));
    assert_eq!(report.loss_reduction_supported, false);
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllGather,
                axis: MeshAxis::Tp,
                ..
            }
        )
    }));
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::ReduceScatter,
                axis: MeshAxis::DpShard,
                ..
            }
        )
    }));
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllReduce,
                axis: MeshAxis::DpReplicate,
                ..
            }
        )
    }));
}

#[test]
fn native_layout_harness_injection_records_failure_context() {
    let scenario = TrainingStepScenario::tiny_dense();
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);
    let plan = InjectionPlan::new(vec![InjectionEvent {
        label: "native_missing_reduce_scatter_rank0".to_string(),
        phase: TrainingPhase::Backward,
        rank: 0,
        axis: Some(MeshAxis::DpShard),
        collective: Some(CollectiveKind::ReduceScatter),
        kind: InjectionKind::MissingParticipant,
    }]);

    let report = scenario
        .run_native_layout_tracer_bullet(dims, Some(&plan))
        .unwrap();

    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].phase, TrainingPhase::Backward);
    assert_eq!(report.failures[0].rank, Some(0));
    assert_eq!(report.failures[0].axis, Some(MeshAxis::DpShard));
    assert_eq!(
        report.failures[0].collective,
        Some(CollectiveKind::ReduceScatter)
    );
    assert_eq!(
        report.failures[0].tensor,
        Some("native_partial".to_string())
    );
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Injection { label, .. } if label == "native_missing_reduce_scatter_rank0")
    ));
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Failure(record) if record.message == "native_missing_reduce_scatter_rank0")
    ));
}

#[test]
fn plan_first_tiny_dense_training_records_planned_collectives() {
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);

    let plan = MeshSimPlan::tiny_dense_training(dims).unwrap();

    assert_eq!(plan.dims, dims);
    assert!(plan.steps.iter().any(|step| {
        matches!(
            step,
            MeshSimStep::Tensor {
                name,
                global_shape,
                layout,
            } if name == "plan_activation"
                && *global_shape == Shape(vec![2, 4, 8])
                && layout.placement(MeshAxis::Tp) == Some(Placement::Replicate)
        )
    }));
    assert!(plan.steps.iter().any(|step| {
        matches!(
            step,
            MeshSimStep::Redistribute {
                tensor,
                phase: TrainingPhase::Forward,
                ..
            } if tensor == "plan_activation"
        )
    }));
    let planned_collectives: Vec<_> = plan
        .steps
        .iter()
        .filter_map(|step| match step {
            MeshSimStep::Collective {
                kind,
                axis,
                tensor,
                phase,
            } => Some((*kind, *axis, tensor.as_str(), *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(
        planned_collectives,
        vec![
            (
                CollectiveKind::AllGather,
                MeshAxis::Tp,
                "plan_activation",
                TrainingPhase::Forward,
            ),
            (
                CollectiveKind::ReduceScatter,
                MeshAxis::DpShard,
                "plan_grad_partial",
                TrainingPhase::Backward,
            ),
            (
                CollectiveKind::AllReduce,
                MeshAxis::DpReplicate,
                "plan_ddp_partial",
                TrainingPhase::Backward,
            ),
        ]
    );
}

#[test]
fn plan_first_execution_emits_common_mesh_trace() {
    let scenario = TrainingStepScenario::tiny_dense();
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);
    let plan = MeshSimPlan::tiny_dense_training(dims).unwrap();

    let report = plan.execute(None).unwrap();

    let reference = scenario.run_unsharded_reference();
    assert!(report.reference_loss.is_finite());
    assert_eq!(report.reference_output_shape, reference.output_shape);
    assert_eq!(report.reference_output_shape, Shape(vec![2, 4, 8]));
    assert!(report.activation_matches_reference);
    assert_eq!(report.activation_tp_shard_shape, Shape(vec![2, 4, 4]));
    assert_eq!(report.partial_dp_shard_shape, Shape(vec![2, 2]));
    assert_eq!(report.partial_replicated_shape, Shape(vec![2, 4]));
    assert_eq!(report.planned_collectives, 3);
    assert_eq!(report.executed_collectives, 3);
    assert_eq!(
        report.planned_collective_records,
        report.executed_collective_records
    );
    assert_eq!(
        report.planned_collective_records,
        collective_records_from_trace(&report.trace)
    );
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::LayoutTransition {
                phase: TrainingPhase::Forward,
                tensor,
                src,
                dst,
            } if tensor == "plan_activation" && src.contains("Replicate") && dst.contains("Shard(2)")
        )
    }));
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllGather,
                axis: MeshAxis::Tp,
                ..
            }
        )
    }));
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::ReduceScatter,
                axis: MeshAxis::DpShard,
                ..
            }
        )
    }));
    assert!(report.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllReduce,
                axis: MeshAxis::DpReplicate,
                ..
            }
        )
    }));
}

#[test]
fn plan_first_injection_is_represented_in_plan_and_trace() {
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);
    let mut plan = MeshSimPlan::tiny_dense_training(dims).unwrap();
    plan.steps.insert(
        6,
        MeshSimStep::Injection {
            event: InjectionEvent {
                label: "plan_missing_reduce_scatter_rank0".to_string(),
                phase: TrainingPhase::Backward,
                rank: 0,
                axis: Some(MeshAxis::DpShard),
                collective: Some(CollectiveKind::ReduceScatter),
                kind: InjectionKind::MissingParticipant,
            },
        },
    );

    let report = plan.execute(None).unwrap();

    assert!(plan.steps.iter().any(
        |step| matches!(step, MeshSimStep::Injection { event } if event.label == "plan_missing_reduce_scatter_rank0")
    ));
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].phase, TrainingPhase::Backward);
    assert_eq!(report.failures[0].rank, Some(0));
    assert_eq!(report.failures[0].axis, Some(MeshAxis::DpShard));
    assert_eq!(
        report.failures[0].collective,
        Some(CollectiveKind::ReduceScatter)
    );
    assert_eq!(
        report.failures[0].tensor,
        Some("plan_grad_partial".to_string())
    );
    assert_eq!(
        report.failures[0].layout,
        Some("DpShard:Partial(sum)".to_string())
    );
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Injection { label, .. } if label == "plan_missing_reduce_scatter_rank0")
    ));
    assert!(report.trace.events.iter().any(
        |event| matches!(event, MeshTraceEvent::Failure(record) if record.message == "plan_missing_reduce_scatter_rank0")
    ));
    assert_eq!(
        report.planned_collective_records,
        report.executed_collective_records
    );
}

#[test]
fn canonical_plan_first_entrypoint_runs_tiny_dense_training() {
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);

    let report = run_tiny_dense_mesh_simulation(dims, None).unwrap();

    assert_eq!(report.trace.schema, MESH_SIM_TRACE_SCHEMA);
    assert_eq!(report.planned_collectives, 3);
    assert_eq!(report.executed_collectives, 3);
    assert!(report.activation_matches_reference);
    assert!(report
        .stable_vocabulary
        .mesh_axes
        .contains(&"pp,dp_replicate,dp_shard,cp,tp".to_string()));
}

#[test]
fn dualpipe_forward_chunk_exposes_component_order() {
    let schedule = DualPipeSchedule::tiny_dense_training();

    assert_eq!(
        schedule.component_kinds(DualPipeChunk::Forward(0)),
        vec![
            DualPipeComponentKind::Attention,
            DualPipeComponentKind::AllToAllDispatch,
            DualPipeComponentKind::Mlp,
            DualPipeComponentKind::AllToAllCombine,
            DualPipeComponentKind::PpCommunication,
        ]
    );
}

#[test]
fn dualpipe_backward_chunk_splits_input_and_weight_gradients() {
    let schedule = DualPipeSchedule::tiny_dense_training();

    assert_eq!(
        schedule.component_kinds(DualPipeChunk::Backward(0)),
        vec![
            DualPipeComponentKind::AttentionInputGradient,
            DualPipeComponentKind::AttentionWeightGradient,
            DualPipeComponentKind::AllToAllDispatchGradient,
            DualPipeComponentKind::MlpInputGradient,
            DualPipeComponentKind::MlpWeightGradient,
            DualPipeComponentKind::AllToAllCombineGradient,
            DualPipeComponentKind::PpCommunication,
        ]
    );
}

#[test]
fn dualpipe_schedule_allows_one_paired_forward_backward_overlap() {
    let schedule = DualPipeSchedule::tiny_dense_training();

    schedule.validate().unwrap();
    assert!(schedule.legal_overlaps().iter().any(|overlap| {
        overlap.forward_component == DualPipeComponentKind::Mlp
            && overlap.backward_component == DualPipeComponentKind::AttentionInputGradient
    }));
}

#[test]
fn dualpipe_schedule_rejects_dependency_ordering_overlap_with_failure_context() {
    let mut schedule = DualPipeSchedule::tiny_dense_training();
    schedule.add_overlap(
        DualPipeChunk::Forward(0),
        DualPipeComponentKind::Attention,
        DualPipeChunk::Forward(0),
        DualPipeComponentKind::Mlp,
    );

    let err = match schedule.validate() {
        Ok(_) => panic!("illegal same-chunk dependency overlap unexpectedly validated"),
        Err(err) => err,
    };

    match err {
        PlanError::Schedule(failure) => {
            assert_eq!(failure.phase, TrainingPhase::Forward);
            assert_eq!(failure.tensor, Some("dualpipe.forward[0].mlp".to_string()));
            assert_eq!(
                failure.layout,
                Some("dualpipe.forward[0].attention".to_string())
            );
            assert_eq!(failure.collective, None);
            assert!(failure.message.contains("dependency"));
        }
        other => panic!("expected structured schedule failure, got {other:?}"),
    }
}

#[test]
fn plan_first_report_text_is_deterministic() {
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);

    let first = run_tiny_dense_mesh_simulation(dims, None)
        .unwrap()
        .deterministic_run_report();
    let second = run_tiny_dense_mesh_simulation(dims, None)
        .unwrap()
        .deterministic_run_report();

    assert_eq!(first, second);
    assert!(first.contains("entrypoint: run_tiny_dense_mesh_simulation"));
    assert!(first.contains("dualpipe.forward[0]: attention -> all_to_all_dispatch -> mlp"));
    assert!(first.contains("collective: backward reduce_scatter dp_shard plan_grad_partial"));
}

fn collective_records_from_trace(trace: &MeshTrace) -> Vec<MeshSimCollectiveRecord> {
    trace
        .events
        .iter()
        .filter_map(|event| match event {
            MeshTraceEvent::Collective {
                phase,
                kind,
                axis,
                tensor: Some(tensor),
                ..
            } => Some(MeshSimCollectiveRecord {
                kind: *kind,
                axis: *axis,
                tensor: tensor.clone(),
                phase: *phase,
            }),
            _ => None,
        })
        .collect()
}

#[test]
fn wrapper_dtensor_replicate_shard_reconstructs_and_traces() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let src_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)]).unwrap();
    let dst_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let value = TensorValue::from_vec(
        Shape(vec![2, 4]),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
    );
    let dtensor = DTensor::from_replicated(value.clone(), src_layout, dims).unwrap();
    let mut sim = CollectiveSimulator::new(dims);

    let sharded = dtensor
        .redistribute(dst_layout, &mut sim, TrainingPhase::Forward)
        .unwrap();

    assert_eq!(sharded.global_shape, Shape(vec![2, 4]));
    assert_eq!(sharded.shards[0].shape, Shape(vec![2, 2]));
    assert_eq!(sharded.shards[0].data.as_ref(), &vec![0.0, 1.0, 10.0, 11.0]);
    assert_eq!(sharded.shards[1].shape, Shape(vec![2, 2]));
    assert_eq!(sharded.shards[1].data.as_ref(), &vec![2.0, 3.0, 12.0, 13.0]);
    let reconstructed = sharded.full_tensor().unwrap();
    assert_eq!(reconstructed.shape, value.shape);
    assert_eq!(reconstructed.data.as_ref(), value.data.as_ref());
    assert!(sim.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::LayoutTransition {
                phase: TrainingPhase::Forward,
                ..
            }
        )
    }));
}

#[test]
fn wrapper_dtensor_partial_to_shard_uses_dimension_aware_reduce_scatter() {
    let dims = ParallelDims5D::new(1, 1, 2, 1, 1);
    let src_layout =
        Layout::new(vec![(MeshAxis::DpShard, Placement::Partial(ReduceOp::Sum))]).unwrap();
    let dst_layout = Layout::new(vec![(MeshAxis::DpShard, Placement::Shard(1))]).unwrap();
    let shards = vec![
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        ),
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0, 3000.0, 4000.0],
        ),
    ];
    let dtensor = DTensor {
        global_shape: Shape(vec![2, 4]),
        layout: src_layout,
        dims,
        shards,
    };
    let mut sim = CollectiveSimulator::new(dims);

    let sharded = dtensor
        .redistribute(dst_layout, &mut sim, TrainingPhase::Backward)
        .unwrap();

    assert_eq!(sharded.shards.len(), 2);
    assert_eq!(sharded.shards[0].shape, Shape(vec![2, 2]));
    assert_eq!(
        sharded.shards[0].data.as_ref(),
        &vec![101.0, 202.0, 1010.0, 2020.0]
    );
    assert_eq!(sharded.shards[1].shape, Shape(vec![2, 2]));
    assert_eq!(
        sharded.shards[1].data.as_ref(),
        &vec![303.0, 404.0, 3030.0, 4040.0]
    );
    assert!(matches!(
        sim.trace.events.first(),
        Some(MeshTraceEvent::Collective {
            kind: CollectiveKind::ReduceScatter,
            axis: MeshAxis::DpShard,
            ..
        })
    ));
}

#[test]
fn wrapper_dtensor_shard_and_partial_replicate_paths_preserve_full_tensor() {
    let shard_dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let replicated_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)]).unwrap();
    let sharded_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let value = TensorValue::from_vec(
        Shape(vec![2, 4]),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
    );
    let sharded = DTensor::from_replicated(value.clone(), sharded_layout, shard_dims).unwrap();
    let mut gather_sim = CollectiveSimulator::new(shard_dims);

    let gathered = sharded
        .redistribute(replicated_layout, &mut gather_sim, TrainingPhase::Forward)
        .unwrap();

    assert_eq!(gathered.shards[0].shape, value.shape);
    assert_eq!(gathered.shards[0].data.as_ref(), value.data.as_ref());
    assert_eq!(gathered.shards[1].data.as_ref(), value.data.as_ref());
    assert!(gather_sim.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllGather,
                axis: MeshAxis::Tp,
                ..
            }
        )
    }));

    let partial_dims = ParallelDims5D::new(1, 2, 1, 1, 1);
    let partial_layout = Layout::new(vec![(
        MeshAxis::DpReplicate,
        Placement::Partial(ReduceOp::Sum),
    )])
    .unwrap();
    let replicate_layout =
        Layout::new(vec![(MeshAxis::DpReplicate, Placement::Replicate)]).unwrap();
    let partial = DTensor {
        global_shape: Shape(vec![2, 2]),
        layout: partial_layout,
        dims: partial_dims,
        shards: vec![
            TensorValue::from_vec(Shape(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]),
            TensorValue::from_vec(Shape(vec![2, 2]), vec![10.0, 20.0, 30.0, 40.0]),
        ],
    };
    let mut reduce_sim = CollectiveSimulator::new(partial_dims);

    let reduced = partial
        .redistribute(replicate_layout, &mut reduce_sim, TrainingPhase::Backward)
        .unwrap();

    assert_eq!(reduced.shards[0].shape, Shape(vec![2, 2]));
    assert_eq!(
        reduced.shards[0].data.as_ref(),
        &vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(
        reduced.shards[1].data.as_ref(),
        &vec![11.0, 22.0, 33.0, 44.0]
    );
    assert!(reduce_sim.trace.events.iter().any(|event| {
        matches!(
            event,
            MeshTraceEvent::Collective {
                kind: CollectiveKind::AllReduce,
                axis: MeshAxis::DpReplicate,
                ..
            }
        )
    }));
}

#[test]
fn wrapper_dtensor_identical_layout_clones_without_trace_event() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let layout = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)]).unwrap();
    let value = TensorValue::from_vec(Shape(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]);
    let dtensor = DTensor::from_replicated(value.clone(), layout.clone(), dims).unwrap();
    let mut sim = CollectiveSimulator::new(dims);

    let same = dtensor
        .redistribute(layout, &mut sim, TrainingPhase::Forward)
        .unwrap();

    assert_eq!(same, dtensor);
    assert!(sim.trace.events.is_empty());
}

#[test]
fn wrapper_dtensor_from_replicated_rejects_partial_layout() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 1);
    let partial_layout = Layout::new(vec![(
        MeshAxis::DpReplicate,
        Placement::Partial(ReduceOp::Sum),
    )])
    .unwrap();
    let value = TensorValue::from_vec(Shape(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]);

    let err = match DTensor::from_replicated(value, partial_layout, dims) {
        Ok(_) => panic!("from_replicated unexpectedly accepted a partial layout"),
        Err(err) => err,
    };

    assert!(matches!(err, DtError::UnsupportedRedistribution { .. }));
}

#[test]
fn wrapper_dtensor_unsupported_redistribution_returns_typed_error() {
    let dims = ParallelDims5D::new(1, 1, 1, 1, 2);
    let src_layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(1))]).unwrap();
    let dst_layout =
        Layout::new(vec![(MeshAxis::DpShard, Placement::Partial(ReduceOp::Sum))]).unwrap();
    let value = TensorValue::from_vec(
        Shape(vec![2, 4]),
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
    );
    let dtensor = DTensor::from_replicated(value, src_layout, dims).unwrap();
    let mut sim = CollectiveSimulator::new(dims);

    let err = match dtensor.redistribute(dst_layout, &mut sim, TrainingPhase::Forward) {
        Ok(_) => panic!("unsupported redistribution unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(matches!(err, DtError::UnsupportedRedistribution { .. }));
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
        tensor: None,
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
            tensor: None,
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
            tensor: None,
            bytes: 2,
            ranks: vec![0, 2, 1, 3],
        }
    );
}

#[test]
fn collective_reduce_scatter_sums_then_scatters_axis_groups_and_records_trace() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let shards = vec![
        TensorValue::from_vec(Shape(vec![4]), vec![1.0, 2.0, 3.0, 4.0]),
        TensorValue::from_vec(Shape(vec![4]), vec![10.0, 20.0, 30.0, 40.0]),
        TensorValue::from_vec(Shape(vec![4]), vec![100.0, 200.0, 300.0, 400.0]),
        TensorValue::from_vec(Shape(vec![4]), vec![1000.0, 2000.0, 3000.0, 4000.0]),
    ];
    let mut sim = CollectiveSimulator::new(dims);
    let out = sim
        .reduce_scatter_sum(MeshAxis::DpReplicate, 0, TrainingPhase::Backward, &shards)
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(out[0].shape, Shape(vec![2]));
    assert_eq!(out[0].data.as_ref(), &vec![101.0, 202.0]);
    assert_eq!(out[1].data.as_ref(), &vec![1010.0, 2020.0]);
    assert_eq!(out[2].data.as_ref(), &vec![303.0, 404.0]);
    assert_eq!(out[3].data.as_ref(), &vec![3030.0, 4040.0]);
    assert_eq!(
        sim.trace.events[0],
        MeshTraceEvent::Collective {
            phase: TrainingPhase::Backward,
            kind: CollectiveKind::ReduceScatter,
            axis: MeshAxis::DpReplicate,
            tensor: None,
            bytes: 8,
            ranks: vec![0, 2, 1, 3],
        }
    );
}

#[test]
fn collective_reduce_scatter_slices_requested_tensor_dim_across_axis_groups() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let shards = vec![
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        ),
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![
                1000.0, 2000.0, 3000.0, 4000.0, 10000.0, 20000.0, 30000.0, 40000.0,
            ],
        ),
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0, 3000.0, 4000.0],
        ),
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![
                100000.0, 200000.0, 300000.0, 400000.0, 1000000.0, 2000000.0, 3000000.0, 4000000.0,
            ],
        ),
    ];
    let mut sim = CollectiveSimulator::new(dims);

    let out = sim
        .reduce_scatter_sum(MeshAxis::DpReplicate, 1, TrainingPhase::Backward, &shards)
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(out[0].shape, Shape(vec![2, 2]));
    assert_eq!(out[0].data.as_ref(), &vec![101.0, 202.0, 1010.0, 2020.0]);
    assert_eq!(out[2].shape, Shape(vec![2, 2]));
    assert_eq!(out[2].data.as_ref(), &vec![303.0, 404.0, 3030.0, 4040.0]);
    assert_eq!(out[1].shape, Shape(vec![2, 2]));
    assert_eq!(
        out[1].data.as_ref(),
        &vec![101000.0, 202000.0, 1010000.0, 2020000.0]
    );
    assert_eq!(out[3].shape, Shape(vec![2, 2]));
    assert_eq!(
        out[3].data.as_ref(),
        &vec![303000.0, 404000.0, 3030000.0, 4040000.0]
    );
    assert_eq!(
        sim.trace.events[0],
        MeshTraceEvent::Collective {
            phase: TrainingPhase::Backward,
            kind: CollectiveKind::ReduceScatter,
            axis: MeshAxis::DpReplicate,
            tensor: None,
            bytes: 16,
            ranks: vec![0, 2, 1, 3],
        }
    );
}

#[test]
fn collective_reduce_scatter_rejects_out_of_range_shard_dim() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 1);
    let shards = vec![
        TensorValue::from_vec(Shape(vec![2, 4]), vec![0.0; 8]),
        TensorValue::from_vec(Shape(vec![2, 4]), vec![1.0; 8]),
    ];
    let mut sim = CollectiveSimulator::new(dims);

    let err =
        match sim.reduce_scatter_sum(MeshAxis::DpReplicate, 2, TrainingPhase::Backward, &shards) {
            Ok(_) => panic!("out-of-range shard dim unexpectedly reduced"),
            Err(err) => err,
        };

    assert_eq!(err, CollectiveError::SliceDimOutOfRange { dim: 2, rank: 0 });
}

#[test]
fn collective_reduce_scatter_rejects_uneven_requested_dim() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 1);
    let shards = vec![
        TensorValue::from_vec(Shape(vec![2, 3]), vec![0.0; 6]),
        TensorValue::from_vec(Shape(vec![2, 3]), vec![1.0; 6]),
    ];
    let mut sim = CollectiveSimulator::new(dims);

    let err =
        match sim.reduce_scatter_sum(MeshAxis::DpReplicate, 1, TrainingPhase::Backward, &shards) {
            Ok(_) => panic!("uneven shard dim unexpectedly reduced"),
            Err(err) => err,
        };

    assert_eq!(
        err,
        CollectiveError::UnevenLocalSlice {
            dim: 1,
            size: 3,
            parts: 2,
        }
    );
}

#[test]
fn collective_broadcast_copies_axis_group_root_and_records_trace() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let shards = vec![
        TensorValue::from_vec(Shape(vec![2]), vec![1.0, 2.0]),
        TensorValue::from_vec(Shape(vec![2]), vec![10.0, 20.0]),
        TensorValue::from_vec(Shape(vec![2]), vec![100.0, 200.0]),
        TensorValue::from_vec(Shape(vec![2]), vec![1000.0, 2000.0]),
    ];
    let mut sim = CollectiveSimulator::new(dims);
    let out = sim
        .broadcast(MeshAxis::DpReplicate, TrainingPhase::Forward, 1, &shards)
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(out[0].data.as_ref(), &vec![100.0, 200.0]);
    assert_eq!(out[1].data.as_ref(), &vec![1000.0, 2000.0]);
    assert_eq!(out[2].data.as_ref(), &vec![100.0, 200.0]);
    assert_eq!(out[3].data.as_ref(), &vec![1000.0, 2000.0]);
    assert_eq!(
        sim.trace.events[0],
        MeshTraceEvent::Collective {
            phase: TrainingPhase::Forward,
            kind: CollectiveKind::Broadcast,
            axis: MeshAxis::DpReplicate,
            tensor: None,
            bytes: 4,
            ranks: vec![0, 2, 1, 3],
        }
    );
}

#[test]
fn collective_local_slice_selects_axis_local_chunk_and_records_layout_transition() {
    let dims = ParallelDims5D::new(1, 2, 1, 1, 2);
    let shards = vec![
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
        ),
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0],
        ),
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![40.0, 41.0, 42.0, 43.0, 50.0, 51.0, 52.0, 53.0],
        ),
        TensorValue::from_vec(
            Shape(vec![2, 4]),
            vec![60.0, 61.0, 62.0, 63.0, 70.0, 71.0, 72.0, 73.0],
        ),
    ];
    let mut sim = CollectiveSimulator::new(dims);
    let out = sim
        .local_slice(
            MeshAxis::DpReplicate,
            TrainingPhase::Forward,
            "activation",
            1,
            &shards,
        )
        .unwrap();

    assert_eq!(out.len(), 4);
    assert_eq!(out[0].shape, Shape(vec![2, 2]));
    assert_eq!(out[0].data.as_ref(), &vec![0.0, 1.0, 10.0, 11.0]);
    assert_eq!(out[1].data.as_ref(), &vec![20.0, 21.0, 30.0, 31.0]);
    assert_eq!(out[2].data.as_ref(), &vec![42.0, 43.0, 52.0, 53.0]);
    assert_eq!(out[3].data.as_ref(), &vec![62.0, 63.0, 72.0, 73.0]);
    assert_eq!(
        sim.trace.events[0],
        MeshTraceEvent::LayoutTransition {
            phase: TrainingPhase::Forward,
            tensor: "activation".to_string(),
            src: "Replicate".to_string(),
            dst: "Local".to_string(),
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
            collective: None,
            kind: InjectionKind::InjectNan { offset: 1 },
        },
        InjectionEvent {
            label: "rank0_missing".to_string(),
            phase: TrainingPhase::Backward,
            rank: 0,
            axis: Some(MeshAxis::DpReplicate),
            collective: Some(CollectiveKind::ReduceScatter),
            kind: InjectionKind::MissingParticipant,
        },
    ]);
    let mut shard = TensorValue::from_vec(Shape(vec![2]), vec![1.0, 2.0]);
    let failures = plan.apply_to_shard(TrainingPhase::Backward, 0, &mut shard);

    assert!(shard.data[1].is_nan());
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].rank, Some(0));
    assert_eq!(failures[0].axis, Some(MeshAxis::DpReplicate));
    assert_eq!(failures[0].collective, Some(CollectiveKind::ReduceScatter));
}
