use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::saved::{SaveRole, SaveSite, SavedTensor};
use crate::tensor::{Shape, Tensor, TensorValue};

const EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// LayerNorm helpers
// ---------------------------------------------------------------------------
// x: [..., D], gamma: [D], beta: [D]
// mean = mean(x, axis=-1), var = var(x, axis=-1)
// x_hat = (x - mean) / sqrt(var + eps)
// y = x_hat * gamma + beta

pub struct LayerNormStats {
    pub mean: Vec<f32>,     // [batch]
    pub var: Vec<f32>,      // [batch]
    pub x_hat: TensorValue, // same shape as x
}

fn layer_norm_forward(
    x: &TensorValue,
    gamma: &TensorValue,
    beta: &TensorValue,
) -> (TensorValue, LayerNormStats) {
    let xd = &x.shape.0;
    let d = *xd.last().unwrap();
    let batch: usize = xd[..xd.len() - 1].iter().product();

    let xr = x.data.as_ref();
    let gr = gamma.data.as_ref();
    let br = beta.data.as_ref();

    let mut means = vec![0.0f32; batch];
    let mut vars = vec![0.0f32; batch];
    let mut x_hat_data = vec![0.0f32; x.shape.numel()];
    let mut out_data = vec![0.0f32; x.shape.numel()];

    for b in 0..batch {
        let row = &xr[b * d..(b + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let std_inv = (var + EPS).sqrt().recip();

        means[b] = mean;
        vars[b] = var;

        for i in 0..d {
            let xh = (xr[b * d + i] - mean) * std_inv;
            x_hat_data[b * d + i] = xh;
            out_data[b * d + i] = xh * gr[i] + br[i];
        }
    }

    let stats = LayerNormStats {
        mean: means,
        var: vars,
        x_hat: TensorValue::from_vec(x.shape.clone(), x_hat_data),
    };

    (TensorValue::from_vec(x.shape.clone(), out_data), stats)
}

// Returns (dx, dgamma, dbeta)
fn layer_norm_backward(
    dy: &TensorValue,
    x: &TensorValue,
    gamma: &TensorValue,
    stats: &LayerNormStats,
) -> (TensorValue, TensorValue, TensorValue) {
    let xd = &x.shape.0;
    let d = *xd.last().unwrap();
    let batch: usize = xd[..xd.len() - 1].iter().product();

    let dyr = dy.data.as_ref();
    let xr = x.data.as_ref();
    let gr = gamma.data.as_ref();
    let xhr = stats.x_hat.data.as_ref();

    let mut dx = vec![0.0f32; x.shape.numel()];
    let mut dgamma = vec![0.0f32; d];
    let mut dbeta = vec![0.0f32; d];

    for b in 0..batch {
        let std_inv = (stats.var[b] + EPS).sqrt().recip();
        let mean = stats.mean[b];

        // dgamma and dbeta sum over batch dimension
        for i in 0..d {
            dgamma[i] += dyr[b * d + i] * xhr[b * d + i];
            dbeta[i] += dyr[b * d + i];
        }

        // dx computation
        let dx_hat: Vec<f32> = (0..d).map(|i| dyr[b * d + i] * gr[i]).collect();

        let dvar_sum: f32 = (0..d)
            .map(|i| dx_hat[i] * (xr[b * d + i] - mean))
            .sum::<f32>();
        let dvar = -0.5 * dvar_sum * std_inv.powi(3);

        let dmean_sum: f32 = dx_hat.iter().sum::<f32>();
        let dmean = -std_inv * dmean_sum
            + dvar * (-2.0 / d as f32) * (0..d).map(|i| xr[b * d + i] - mean).sum::<f32>();

        for i in 0..d {
            dx[b * d + i] = dx_hat[i] * std_inv
                + dvar * 2.0 * (xr[b * d + i] - mean) / d as f32
                + dmean / d as f32;
        }
    }

    (
        TensorValue::from_vec(x.shape.clone(), dx),
        TensorValue::from_vec(Shape(vec![d]), dgamma),
        TensorValue::from_vec(Shape(vec![d]), dbeta),
    )
}

// ---------------------------------------------------------------------------
// LayerNorm op
// ---------------------------------------------------------------------------

struct LayerNormBackward {
    x: SavedTensor,
    gamma: SavedTensor,
    mean: Vec<f32>,
    var: Vec<f32>,
    x_hat: SavedTensor,
    x_target: GradTarget,
    gamma_target: GradTarget,
    beta_target: GradTarget,
}

impl BackwardRecipe for LayerNormBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let x = ctx.unpack(&self.x);
        let gamma = ctx.unpack(&self.gamma);
        let x_hat = ctx.unpack(&self.x_hat);

        let stats = LayerNormStats {
            mean: self.mean.clone(),
            var: self.var.clone(),
            x_hat,
        };

        let (dx, dgamma, dbeta) = layer_norm_backward(dy, &x, &gamma, &stats);

        vec![
            GradEdge {
                target: self.x_target.clone(),
                grad: dx,
            },
            GradEdge {
                target: self.gamma_target.clone(),
                grad: dgamma,
            },
            GradEdge {
                target: self.beta_target.clone(),
                grad: dbeta,
            },
        ]
    }
}

pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, name: &str) -> Tensor {
    let (out_value, stats) = {
        let xv = x.inner.borrow();
        let gv = gamma.inner.borrow();
        let bv = beta.inner.borrow();
        layer_norm_forward(&xv.value, &gv.value, &bv.value)
    };

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x, gamma, beta])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let x_site = SaveSite::new(
            OpKind::LayerNorm,
            format!("{name}.input"),
            SaveRole::Activation,
            x,
        );
        let g_site = SaveSite::new(
            OpKind::LayerNorm,
            format!("{name}.gamma"),
            SaveRole::Parameter,
            gamma,
        );
        let xh_site = SaveSite::new(
            OpKind::LayerNorm,
            format!("{name}.x_hat"),
            SaveRole::AuxStat,
            x,
        );

        let saved_x = SavedTensor::save(x, x_site.clone());
        let saved_g = SavedTensor::save(gamma, g_site.clone());

        // x_hat is derived; save it as a materialized value
        let x_hat_tensor = Tensor::from_value_no_grad(stats.x_hat.clone());
        let saved_xh = SavedTensor::save(&x_hat_tensor, xh_site.clone());

        let r: Box<dyn BackwardRecipe> = Box::new(LayerNormBackward {
            x: saved_x,
            gamma: saved_g,
            mean: stats.mean,
            var: stats.var,
            x_hat: saved_xh,
            x_target: x.grad_target(),
            gamma_target: gamma.grad_target(),
            beta_target: beta.grad_target(),
        });
        (Some(r), vec![x_site, g_site, xh_site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(
        OpKind::LayerNorm,
        name,
        &[x, gamma, beta],
        out_value,
        recipe,
        debug_saved,
    )
}

// ---------------------------------------------------------------------------
// RMSNorm: y = x / rms(x) * gamma, rms(x) = sqrt(mean(x^2) + eps)
// ---------------------------------------------------------------------------

fn rms_norm_forward(x: &TensorValue, gamma: &TensorValue) -> (TensorValue, Vec<f32>) {
    let xd = &x.shape.0;
    let d = *xd.last().unwrap();
    let batch: usize = xd[..xd.len() - 1].iter().product();
    let xr = x.data.as_ref();
    let gr = gamma.data.as_ref();

    let mut rms_inv = vec![0.0f32; batch];
    let mut out_data = vec![0.0f32; x.shape.numel()];

    for b in 0..batch {
        let ms: f32 = xr[b * d..(b + 1) * d].iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let ri = (ms + EPS).sqrt().recip();
        rms_inv[b] = ri;
        for i in 0..d {
            out_data[b * d + i] = xr[b * d + i] * ri * gr[i];
        }
    }

    (TensorValue::from_vec(x.shape.clone(), out_data), rms_inv)
}

fn rms_norm_backward(
    dy: &TensorValue,
    x: &TensorValue,
    gamma: &TensorValue,
    rms_inv: &[f32],
) -> (TensorValue, TensorValue) {
    let xd = &x.shape.0;
    let d = *xd.last().unwrap();
    let batch: usize = xd[..xd.len() - 1].iter().product();

    let dyr = dy.data.as_ref();
    let xr = x.data.as_ref();
    let gr = gamma.data.as_ref();

    let mut dx = vec![0.0f32; x.shape.numel()];
    let mut dgamma = vec![0.0f32; d];

    for b in 0..batch {
        let ri = rms_inv[b];
        let ri3 = ri * ri * ri;

        for i in 0..d {
            dgamma[i] += dyr[b * d + i] * xr[b * d + i] * ri;
        }

        let dot: f32 = (0..d).map(|i| dyr[b * d + i] * gr[i] * xr[b * d + i]).sum();
        let scale = -ri3 / d as f32;

        for i in 0..d {
            dx[b * d + i] = dyr[b * d + i] * gr[i] * ri + scale * dot * xr[b * d + i];
        }
    }

    (
        TensorValue::from_vec(x.shape.clone(), dx),
        TensorValue::from_vec(Shape(vec![d]), dgamma),
    )
}

struct RmsNormBackward {
    x: SavedTensor,
    gamma: SavedTensor,
    rms_inv: Vec<f32>,
    x_target: GradTarget,
    gamma_target: GradTarget,
}

impl BackwardRecipe for RmsNormBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let x = ctx.unpack(&self.x);
        let gamma = ctx.unpack(&self.gamma);
        let (dx, dgamma) = rms_norm_backward(dy, &x, &gamma, &self.rms_inv);
        vec![
            GradEdge {
                target: self.x_target.clone(),
                grad: dx,
            },
            GradEdge {
                target: self.gamma_target.clone(),
                grad: dgamma,
            },
        ]
    }
}

pub fn rms_norm(x: &Tensor, gamma: &Tensor, name: &str) -> Tensor {
    let (out_value, rms_inv) = {
        let xv = x.inner.borrow();
        let gv = gamma.inner.borrow();
        rms_norm_forward(&xv.value, &gv.value)
    };

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x, gamma])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let x_site = SaveSite::new(
            OpKind::RmsNorm,
            format!("{name}.input"),
            SaveRole::Activation,
            x,
        );
        let g_site = SaveSite::new(
            OpKind::RmsNorm,
            format!("{name}.gamma"),
            SaveRole::Parameter,
            gamma,
        );
        let saved_x = SavedTensor::save(x, x_site.clone());
        let saved_g = SavedTensor::save(gamma, g_site.clone());
        let r: Box<dyn BackwardRecipe> = Box::new(RmsNormBackward {
            x: saved_x,
            gamma: saved_g,
            rms_inv,
            x_target: x.grad_target(),
            gamma_target: gamma.grad_target(),
        });
        (Some(r), vec![x_site, g_site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(
        OpKind::RmsNorm,
        name,
        &[x, gamma],
        out_value,
        recipe,
        debug_saved,
    )
}
