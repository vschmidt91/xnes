from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import cond, norm
from scipy.linalg import expm, qr


@dataclass(frozen=True)
class XNESState:
    names: list[str]
    loc: np.ndarray
    scale: np.ndarray
    p_sigma: np.ndarray
    batch_z: np.ndarray | None = None
    batch_x: np.ndarray | None = None
    results: list[tuple[float, ...]] | None = None


def _default_eta_B(dim: int) -> float:
    if dim <= 0:
        return 1.0
    return float(0.6 * (3.0 + np.log(dim)) / (dim * np.sqrt(dim)))


class XNES:
    def __init__(
        self,
        x0: np.ndarray,
        sigma0: np.ndarray | float,
        p_sigma: np.ndarray | None = None,
        *,
        csa_enabled: bool = True,
        eta_mu: float = 1.0,
        eta_sigma: float = 1.0,
        eta_B: float | None = None,
    ) -> None:
        self.loc = np.asarray(x0, dtype=float)
        self.sigma: float
        self.B: np.ndarray
        self.p_sigma: np.ndarray
        self.csa_enabled = bool(csa_enabled)
        self.eta_mu = _validate_positive_finite(eta_mu, "eta_mu")
        self.eta_sigma = _validate_positive_finite(eta_sigma, "eta_sigma")
        self.eta_B = None if eta_B is None else _validate_positive_finite(eta_B, "eta_B")

        if self.dim == 0:
            self.sigma = 1.0
            self.B = np.eye(0)
            self.p_sigma = np.zeros(0)
            return

        scale0 = np.asarray(sigma0, dtype=float)
        if scale0.ndim == 0:
            scale0 = np.repeat(scale0, self.dim)
        if scale0.ndim == 1:
            scale0 = np.diag(scale0)
        if scale0.shape != (self.dim, self.dim):
            msg = f"Expected scale shape {(self.dim, self.dim)}, got {scale0.shape}"
            raise ValueError(msg)

        sign, logdet = np.linalg.slogdet(scale0)
        if sign <= 0 or not np.isfinite(logdet):
            msg = "Scale matrix must have a positive finite determinant."
            raise ValueError(msg)

        self.sigma = max(float(np.exp(logdet / self.dim)), 1e-30)
        self.B = scale0 / self.sigma
        if p_sigma is None:
            self.p_sigma = np.zeros(self.dim)
        else:
            p_sigma_vec = np.asarray(p_sigma, dtype=float)
            if p_sigma_vec.shape != (self.dim,):
                msg = f"Expected p_sigma shape {(self.dim,)}, got {p_sigma_vec.shape}"
                raise ValueError(msg)
            self.p_sigma = p_sigma_vec

    @property
    def dim(self) -> int:
        return int(self.loc.size)

    @property
    def scale(self) -> np.ndarray:
        return self.sigma * self.B

    def ask(
        self,
        num_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.dim == 0:
            n = int(num_samples) if num_samples is not None else 4
            return np.zeros((0, n)), np.zeros((0, n))

        n = int(num_samples) if num_samples is not None else (4 + int(3 * np.log(self.dim)))
        if n <= 1:
            n = 2
        if n % 2 == 1:
            n += 1

        n_half = n // 2
        rng = rng or np.random.default_rng()

        z_half = np.empty((self.dim, n_half))
        for start in range(0, n_half, self.dim):
            end = min(start + self.dim, n_half)
            k = end - start
            raw = rng.standard_normal((self.dim, k))
            lengths = np.sqrt(rng.chisquare(self.dim, k))
            basis, _ = qr(raw, mode="economic")
            z_half[:, start:end] = basis * lengths

        z = np.hstack([z_half, -z_half])
        x = self.loc[:, None] + self.scale @ z
        return z, x

    def tell(self, samples: np.ndarray, ranking: list[int], eps: float = 1e-10) -> bool:
        if self.dim == 0:
            return True

        n = samples.shape[1]
        d = self.dim
        if samples.shape[0] != d:
            msg = f"Sample shape mismatch, expected {d} rows, got {samples.shape[0]}"
            raise ValueError(msg)
        if len(ranking) != n or sorted(ranking) != list(range(n)):
            msg = "ranking must be a permutation matching sample count."
            raise ValueError(msg)
        if not np.all(np.isfinite(samples)):
            msg = "samples must be finite."
            raise ValueError(msg)

        w_pos = np.maximum(0.0, np.log(n / 2 + 1) - np.log(np.arange(1, n + 1)))
        w_sum = float(np.sum(w_pos))
        if w_sum <= 0.0:
            msg = "Invalid utility weights: positive weight sum must be > 0."
            raise ValueError(msg)
        w_pos /= w_sum
        mu_eff_pos = 1.0 / np.sum(w_pos**2)
        w_active = w_pos - (1.0 / n)
        z_sorted = samples[:, ranking]

        grad_mu = z_sorted @ w_active
        grad_mu_pos = z_sorted @ w_pos
        grad_M = (z_sorted * w_active) @ z_sorted.T
        grad_sigma = float(np.trace(grad_M) / d)
        grad_B_shape = grad_M - grad_sigma * np.eye(d)

        loc_step = self.eta_mu * self.sigma * (self.B @ grad_mu)
        self.loc += loc_step

        if self.csa_enabled:
            c_sigma = (mu_eff_pos + 2.0) / (d + mu_eff_pos + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff_pos - 1.0) / (d + 1.0)) - 1.0) + c_sigma
            expected_norm = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            self.p_sigma = (1.0 - c_sigma) * self.p_sigma + np.sqrt(
                c_sigma * (2.0 - c_sigma) * mu_eff_pos
            ) * grad_mu_pos
            sigma_log_step = (c_sigma / d_sigma) * (np.linalg.norm(self.p_sigma) / expected_norm - 1.0)
            sigma_log_step = self.eta_sigma * float(np.clip(sigma_log_step, -0.5, 0.5))
        else:
            sigma_log_step = 0.5 * self.eta_sigma * grad_sigma
            sigma_log_step = float(np.clip(sigma_log_step, -50.0, 50.0))

        self.sigma *= float(np.exp(sigma_log_step))

        eta_B = self.eta_B if self.eta_B is not None else _default_eta_B(d)
        self.B = self.B @ expm(0.5 * eta_B * grad_B_shape)

        sign, logdet = np.linalg.slogdet(self.B)
        if sign <= 0 or not np.isfinite(logdet):
            return True
        self.B *= np.exp(-logdet / d)

        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            return True
        if (
            not np.all(np.isfinite(self.loc))
            or not np.all(np.isfinite(self.B))
            or not np.all(np.isfinite(self.p_sigma))
        ):
            return True

        scale = self.scale
        cond_scale = cond(scale)
        if not np.isfinite(cond_scale):
            return True
        return bool(self.sigma < eps or norm(scale, 2) < eps or norm(loc_step, 2) < eps or cond_scale * eps > 1)


def _validate_positive_finite(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        msg = f"{name} must be a positive finite float."
        raise ValueError(msg)
    return out
