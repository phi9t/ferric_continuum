from __future__ import annotations

import numpy as np

from . import _muon_cc


def muon_update(
    params: np.ndarray,
    grads: np.ndarray,
    momentum: np.ndarray,
    *,
    learning_rate: float,
    beta: float = 0.9,
    nesterov: bool = True,
    ns_steps: int = 5,
    ns_coeff_a: float = 3.4445,
    ns_coeff_b: float = -4.775,
    ns_coeff_c: float = 2.0315,
    weight_decay: float = 0.0,
) -> np.ndarray:
    """Applies an in-place Muon update to 2D parameters and returns params."""
    return _muon_cc.muon_update(
        params,
        grads,
        momentum,
        learning_rate,
        beta,
        nesterov,
        ns_steps,
        ns_coeff_a,
        ns_coeff_b,
        ns_coeff_c,
        weight_decay,
    )

