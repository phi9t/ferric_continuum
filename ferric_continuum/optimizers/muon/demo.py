from __future__ import annotations

import numpy as np

from ferric_continuum.optimizers.muon import muon_update


def main() -> None:
    params = np.array([[1.0, -2.0, 3.0], [0.5, -1.5, 2.5]], dtype=np.float64)
    grads = np.array([[0.1, -0.3, 0.2], [0.05, -0.15, 0.25]], dtype=np.float64)
    momentum = np.zeros_like(params)

    for step in range(1, 4):
        muon_update(
            params,
            grads,
            momentum,
            learning_rate=0.1,
            beta=0.9,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.01,
        )
        print(f"step={step} params={params}")


if __name__ == "__main__":
    main()

