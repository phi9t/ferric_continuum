from __future__ import annotations

import numpy as np
import pytest

from ferric_continuum.optimizers.muon import muon_update


def _orthogonalize(update: np.ndarray,
                   ns_steps: int,
                   coeff_a: float,
                   coeff_b: float,
                   coeff_c: float) -> np.ndarray:
    x = update.copy()
    frob_norm = np.linalg.norm(x)
    if frob_norm > 0.0:
        x *= np.sqrt(x.shape[0]) / frob_norm

    for _ in range(ns_steps):
        gram = x @ x.T
        gram2 = gram @ gram
        x = coeff_a * x + (coeff_b * gram + coeff_c * gram2) @ x
    return x


def _muon_reference(params: np.ndarray,
                    grads: np.ndarray,
                    momentum: np.ndarray,
                    *,
                    learning_rate: float,
                    beta: float,
                    nesterov: bool,
                    ns_steps: int,
                    coeff_a: float,
                    coeff_b: float,
                    coeff_c: float,
                    weight_decay: float) -> tuple[np.ndarray, np.ndarray]:
    momentum = beta * momentum + (1.0 - beta) * grads
    update = beta * momentum + (1.0 - beta) * grads if nesterov else momentum
    ortho = _orthogonalize(update, ns_steps, coeff_a, coeff_b, coeff_c)
    params = params - learning_rate * (ortho + weight_decay * params)
    return params, momentum


def test_muon_update_matches_reference() -> None:
    params = np.array([[1.0, -2.0, 3.0], [0.5, -1.5, 2.5]], dtype=np.float64)
    grads = np.array([[0.1, -0.3, 0.2], [0.05, -0.15, 0.25]], dtype=np.float64)
    momentum = np.zeros_like(params)

    ref_params = params.copy()
    ref_momentum = momentum.copy()

    muon_update(
        params,
        grads,
        momentum,
        learning_rate=0.1,
        beta=0.9,
        nesterov=True,
        ns_steps=5,
        ns_coeff_a=3.4445,
        ns_coeff_b=-4.775,
        ns_coeff_c=2.0315,
        weight_decay=0.01,
    )

    ref_params, ref_momentum = _muon_reference(
        ref_params,
        grads,
        ref_momentum,
        learning_rate=0.1,
        beta=0.9,
        nesterov=True,
        ns_steps=5,
        coeff_a=3.4445,
        coeff_b=-4.775,
        coeff_c=2.0315,
        weight_decay=0.01,
    )

    np.testing.assert_allclose(params, ref_params, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(momentum, ref_momentum, rtol=1e-12, atol=1e-12)


def test_muon_requires_2d_params() -> None:
    params = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    grads = np.array([0.1, -0.3, 0.2], dtype=np.float64)
    momentum = np.zeros_like(params)

    with pytest.raises(ValueError, match="2D array"):
        muon_update(
            params,
            grads,
            momentum,
            learning_rate=0.1,
            beta=0.9,
            nesterov=True,
            ns_steps=5,
            ns_coeff_a=3.4445,
            ns_coeff_b=-4.775,
            ns_coeff_c=2.0315,
            weight_decay=0.0,
        )


def test_muon_shape_mismatch_raises() -> None:
    params = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    grads = np.array([[0.1, -0.3, 0.2], [0.05, -0.15, 0.25]], dtype=np.float64)
    momentum = np.zeros_like(params)

    with pytest.raises(ValueError, match="identical shapes"):
        muon_update(
            params,
            grads,
            momentum,
            learning_rate=0.1,
            beta=0.9,
            nesterov=False,
            ns_steps=3,
            ns_coeff_a=3.4445,
            ns_coeff_b=-4.775,
            ns_coeff_c=2.0315,
            weight_decay=0.0,
        )

