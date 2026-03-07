from __future__ import annotations

import numpy as np

from xnes import XNES


def _ranking(scores: np.ndarray) -> list[int]:
    return sorted(range(scores.size), key=lambda idx: float(scores[idx]), reverse=True)


def test_xnes_rank_invariance_under_monotonic_transform() -> None:
    dim = 4
    n = 24
    steps = 6

    xnes_a = XNES(np.zeros(dim), np.eye(dim), csa_enabled=False, eta_mu=1.0, eta_sigma=0.8, eta_B=0.2)
    xnes_b = XNES(np.zeros(dim), np.eye(dim), csa_enabled=False, eta_mu=1.0, eta_sigma=0.8, eta_B=0.2)

    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)

    for _ in range(steps):
        z_a, x_a = xnes_a.ask(n, rng_a)
        z_b, _ = xnes_b.ask(n, rng_b)
        assert np.allclose(z_a, z_b)

        raw_scores = -np.sum(x_a**2, axis=0) + 1e-12 * np.arange(n)
        transformed_scores = 3.0 * raw_scores + 2.0

        ranking_raw = _ranking(raw_scores)
        ranking_transformed = _ranking(transformed_scores)
        assert ranking_raw == ranking_transformed

        xnes_a.tell(z_a, ranking_raw)
        xnes_b.tell(z_b, ranking_transformed)

    assert np.allclose(xnes_a.loc, xnes_b.loc)
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

    xnes_x = XNES(mu, scale, csa_enabled=False, eta_mu=1.0, eta_sigma=0.7, eta_B=0.3)
    xnes_y = XNES(transform @ mu + shift, transform @ scale, csa_enabled=False, eta_mu=1.0, eta_sigma=0.7, eta_B=0.3)

    rng_x = np.random.default_rng(9)
    rng_y = np.random.default_rng(9)
    score_projection = np.array([1.7, -0.2, 3.3], dtype=float)

    for _ in range(steps):
        z_x, _ = xnes_x.ask(n, rng_x)
        z_y, _ = xnes_y.ask(n, rng_y)
        assert np.allclose(z_x, z_y)

        scores = score_projection @ z_x + 1e-12 * np.arange(n)
        ranking = _ranking(scores)

        stop_x = xnes_x.tell(z_x, ranking)
        stop_y = xnes_y.tell(z_y, ranking)
        assert stop_x == stop_y

        assert np.all(np.isfinite(xnes_x.loc))
        assert np.isfinite(xnes_x.sigma)
        assert np.all(np.isfinite(xnes_x.scale))

        assert np.all(np.isfinite(xnes_y.loc))
        assert np.isfinite(xnes_y.sigma)
        assert np.all(np.isfinite(xnes_y.scale))

        assert np.allclose(xnes_y.loc, transform @ xnes_x.loc + shift, rtol=1e-10, atol=3e-3)
        assert np.allclose(xnes_y.scale, transform @ xnes_x.scale, rtol=1e-10, atol=1e-12)
