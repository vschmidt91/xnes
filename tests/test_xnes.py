from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from xnes.xnes import _default_eta_B

from xnes import XNES, XNESStatus


def _ranking(scores: np.ndarray) -> list[int]:
    return sorted(range(scores.size), key=lambda idx: float(scores[idx]), reverse=True)


def test_xnes_rank_invariance_under_monotonic_transform() -> None:
    dim = 4
    n = 24
    steps = 6

    xnes_a = XNES(np.zeros(dim), np.eye(dim))
    xnes_b = XNES(np.zeros(dim), np.eye(dim))
    for xnes in (xnes_a, xnes_b):
        xnes.csa_enabled = False
        xnes.eta_mu = 1.0
        xnes.eta_sigma = 0.8
        xnes.eta_B = 0.2

    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)

    for _ in range(steps):
        z_a = xnes_a.ask(n, rng_a)
        x_a = xnes_a.transform(z_a)
        z_b = xnes_b.ask(n, rng_b)
        assert np.allclose(z_a, z_b)
        assert np.allclose(x_a, xnes_a.transform(z_a))

        raw_scores = -np.sum(x_a**2, axis=0) + 1e-12 * np.arange(n)
        transformed_scores = 3.0 * raw_scores + 2.0

        ranking_raw = _ranking(raw_scores)
        ranking_transformed = _ranking(transformed_scores)
        assert ranking_raw == ranking_transformed

        xnes_a.tell(z_a, ranking_raw)
        xnes_b.tell(z_b, ranking_transformed)

    assert np.allclose(xnes_a.mu, xnes_b.mu)
    assert np.allclose(xnes_a.scale, xnes_b.scale)


def test_xnes_linear_invariance_with_stress_values() -> None:
    n = 30
    steps = 5

    mu = np.array([1e10, -1e10, 5e9], dtype=float)
    scale = np.array(
        [
            [1.0e-10, 2.0e-11, -1.0e-11],
            [0.0, 3.0e-10, 2.0e-11],
            [0.0, 0.0, 8.0e-11],
        ],
        dtype=float,
    )

    transform = np.array(
        [
            [2.0e4, 4.0e2, 0.0],
            [0.0, 3.0e-4, 8.0],
            [5.0, -2.0, 4.0e1],
        ],
        dtype=float,
    )
    shift = np.array([-2.0e9, 4.0e9, -3.0e8], dtype=float)

    xnes_x = XNES(mu, scale)
    xnes_y = XNES(transform @ mu + shift, transform @ scale)
    for xnes in (xnes_x, xnes_y):
        xnes.csa_enabled = False
        xnes.eta_mu = 1.0
        xnes.eta_sigma = 0.7
        xnes.eta_B = 0.3

    rng_x = np.random.default_rng(9)
    rng_y = np.random.default_rng(9)
    score_projection = np.array([1.7, -0.2, 3.3], dtype=float)

    for _ in range(steps):
        z_x = xnes_x.ask(n, rng_x)
        z_y = xnes_y.ask(n, rng_y)
        assert np.allclose(z_x, z_y)

        scores = score_projection @ z_x + 1e-12 * np.arange(n)
        ranking = _ranking(scores)

        status_x = xnes_x.tell(z_x, ranking)
        status_y = xnes_y.tell(z_y, ranking)
        assert status_x == status_y

        assert np.all(np.isfinite(xnes_x.mu))
        assert np.isfinite(xnes_x.sigma)
        assert np.all(np.isfinite(xnes_x.scale))

        assert np.all(np.isfinite(xnes_y.mu))
        assert np.isfinite(xnes_y.sigma)
        assert np.all(np.isfinite(xnes_y.scale))

        assert np.allclose(xnes_y.mu, transform @ xnes_x.mu + shift, rtol=1e-10, atol=3e-3)
        assert np.allclose(xnes_y.scale, transform @ xnes_x.scale, rtol=1e-10, atol=1e-12)


def test_xnes_eta_B_scales_dimension_dependent_shape_rate() -> None:
    xnes = XNES(np.zeros(3), np.eye(3))
    xnes.csa_enabled = False
    xnes.eta_mu = 0.0
    xnes.eta_sigma = 0.0
    xnes.eta_B = 0.2

    samples = np.array(
        [
            [1.0, 0.0, -1.0, 0.5],
            [0.5, -1.0, 0.0, 1.0],
            [-0.5, 1.0, 0.5, -1.0],
        ],
        dtype=float,
    )
    ranking = [0, 1, 2, 3]

    n = samples.shape[1]
    d = xnes.dim
    w_pos = np.maximum(0.0, np.log(n / 2 + 1) - np.log(np.arange(1, n + 1)))
    w_pos /= float(np.sum(w_pos))
    w_active = w_pos - (1.0 / n)
    z_sorted = samples[:, ranking]
    grad_M = (z_sorted * w_active) @ z_sorted.T
    grad_sigma = float(np.trace(grad_M) / d)
    grad_B_shape = grad_M - grad_sigma * np.eye(d)

    expected_B = expm(0.5 * 0.2 * _default_eta_B(d) * grad_B_shape)
    sign, logdet = np.linalg.slogdet(expected_B)
    assert sign > 0
    expected_B *= np.exp(-logdet / d)

    status = xnes.tell(samples, ranking)
    assert status is XNESStatus.LOC_STEP_MIN
    assert np.allclose(xnes.mu, np.zeros(3))
    assert np.isclose(xnes.sigma, 1.0)
    assert np.allclose(xnes.B, expected_B)
