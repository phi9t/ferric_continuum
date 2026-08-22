use crate::autograd::Engine;
use crate::ops::{linear, loss};
use crate::tensor::{Shape, Tensor, TensorValue};
use crate::transformer::{TransformerBlock, TransformerConfig};

use super::trace::{MeshTrace, MeshTraceEvent};
use super::{
    CollectiveSimulator, InjectionPlan, Layout, MeshAxis, ParallelDims5D, Placement, ShardMap,
    TrainingPhase,
};

pub struct TrainingStepScenario {
    pub cfg: TransformerConfig,
    pub vocab: usize,
    pub targets: Vec<usize>,
}

pub struct TrainingStepReference {
    pub loss: f32,
    pub output_shape: Shape,
    pub parameter_grad_shapes: Vec<Shape>,
}

pub struct HarnessSmokeReport {
    pub reference_loss: f32,
    pub world_size: usize,
    pub trace_events: usize,
    pub trace: MeshTrace,
    pub failures: usize,
}

impl TrainingStepScenario {
    pub fn tiny_dense() -> Self {
        let cfg = TransformerConfig {
            batch: 2,
            seq: 4,
            d_model: 8,
            d_ff: 16,
            n_heads: 1,
        };
        let vocab = 11;
        let targets = (0..cfg.batch * cfg.seq).map(|i| i % vocab).collect();
        Self {
            cfg,
            vocab,
            targets,
        }
    }

    pub fn run_unsharded_reference(&self) -> TrainingStepReference {
        let mut engine = Engine::new();
        let block = TransformerBlock::new(TransformerConfig {
            batch: self.cfg.batch,
            seq: self.cfg.seq,
            d_model: self.cfg.d_model,
            d_ff: self.cfg.d_ff,
            n_heads: self.cfg.n_heads,
        });
        let x = Tensor::randn_scaled(&[self.cfg.batch, self.cfg.seq, self.cfg.d_model], 0.02)
            .requires_grad();
        let logits_w = Tensor::randn_scaled(&[self.cfg.d_model, self.vocab], 0.02).requires_grad();

        let hidden = block.forward(&x);
        let logits = linear::linear(&hidden, &logits_w, "mesh_sim_logits");
        let loss = loss::cross_entropy(&logits, &self.targets, "mesh_sim_loss");
        engine.backward(&loss);

        let mut parameter_grad_shapes: Vec<Shape> = block
            .parameters()
            .into_iter()
            .filter_map(|p| p.grad().map(|g| g.shape))
            .collect();
        parameter_grad_shapes.push(logits_w.grad().expect("logits grad").shape);
        let loss_value = loss.inner.borrow().value.data[0];

        TrainingStepReference {
            loss: loss_value,
            output_shape: hidden.shape(),
            parameter_grad_shapes,
        }
    }

    pub fn reference_activation_value(&self) -> TensorValue {
        TensorValue::from_vec(
            Shape(vec![self.cfg.batch, self.cfg.seq, self.cfg.d_model]),
            (0..self.cfg.batch * self.cfg.seq * self.cfg.d_model)
                .map(|i| i as f32)
                .collect(),
        )
    }

    pub fn run_common_substrate_smoke(
        &self,
        dims: ParallelDims5D,
        injection_plan: Option<&InjectionPlan>,
    ) -> HarnessSmokeReport {
        let reference = self.run_unsharded_reference();
        let global = crate::tensor::TensorValue::from_vec(
            crate::tensor::Shape(vec![self.cfg.batch, self.cfg.seq, self.cfg.d_model]),
            vec![1.0; self.cfg.batch * self.cfg.seq * self.cfg.d_model],
        );
        let layout = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(2))]).unwrap();
        let shard_map = ShardMap::new(global.shape.clone(), layout, dims).unwrap();
        let mut shards: Vec<_> = (0..dims.world_size())
            .map(|rank| shard_map.shard_tensor_for_rank(&global, rank).unwrap())
            .collect();

        let mut simulator = CollectiveSimulator::new(dims);
        let mut failures = 0usize;
        if let Some(plan) = injection_plan {
            for (rank, shard) in shards.iter_mut().enumerate() {
                let matching = plan.matching_events(TrainingPhase::Backward, rank);
                for event in &matching {
                    simulator.trace.record(MeshTraceEvent::Injection {
                        phase: event.phase,
                        label: event.label.clone(),
                        rank: event.rank,
                    });
                }
                let records = plan.apply_to_shard(TrainingPhase::Backward, rank, shard);
                failures += records.len();
                for record in records {
                    simulator.trace.record(MeshTraceEvent::Failure(record));
                }
            }
        }

        let _ = simulator
            .all_reduce_sum(MeshAxis::DpReplicate, TrainingPhase::Backward, &shards)
            .unwrap();

        HarnessSmokeReport {
            reference_loss: reference.loss,
            world_size: dims.world_size(),
            trace_events: simulator.trace.events.len(),
            trace: simulator.trace,
            failures,
        }
    }
}
