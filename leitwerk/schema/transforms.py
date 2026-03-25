from __future__ import annotations

import numpy as np


def _softplus(value: np.ndarray | float) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(value, dtype=float))


def _softplus_to_latent(value: np.ndarray | float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    return values + np.log1p(-np.exp(-values))
