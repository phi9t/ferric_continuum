use crate::autograd::Engine;
use crate::ops::{linear, loss};
use crate::tensor::{Shape, Tensor};
use crate::transformer::{TransformerBlock, TransformerConfig};

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
        Self { cfg, vocab, targets }
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
        let logits_w =
            Tensor::randn_scaled(&[self.cfg.d_model, self.vocab], 0.02).requires_grad();

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
}
