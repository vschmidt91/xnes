"""Core xNES distribution update implementation.

The class maintains the search distribution in factored form `sigma * B`,
generates mirrored orthogonal samples, and applies canonical xNES updates with
optional CSA step-size control.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numpy.linalg import cond, norm
from scipy.linalg import expm, qr


def _default_eta_B(dim: int) -> float:
    """Return the built-in dimension-dependent shape learning-rate factor."""

    if dim <= 0:
        return 1.0
    return float(0.6 * (3.0 + np.log(dim)) / (dim * np.sqrt(dim)))


def _normalize_scale_matrix(scale: np.ndarray | float, dim: int) -> np.ndarray:
    scale0 = np.asarray(scale, dtype=float)
    if scale0.ndim == 0:
        return np.diag(np.repeat(scale0, dim))
    if scale0.ndim == 1:
        return np.diag(scale0)
    if scale0.shape != (dim, dim):
        msg = f"Expected scale shape {(dim, dim)}, got {scale0.shape}"
        raise ValueError(msg)
    return scale0


def _validated_samples(samples: np.ndarray, dim: int) -> np.ndarray:
    z = np.asarray(samples, dtype=float)
    if z.ndim != 2:
        msg = "samples must have shape (dim, n)."
        raise ValueError(msg)
    if z.shape[0] != dim:
        msg = f"Sample shape mismatch, expected {dim} rows, got {z.shape[0]}"
        raise ValueError(msg)
    if not np.all(np.isfinite(z)):
        msg = "samples must be finite."
        raise ValueError(msg)
    return z


def _default_sample_count(num_samples: int | None, dim: int) -> int:
    n = int(num_samples) if num_samples is not None else (4 + int(3 * np.log(dim)))
    if n <= 1:
        n = 2
    if n % 2 == 1:
        n += 1
    return n


def _utility_weights(sample_count: int) -> tuple[np.ndarray, np.ndarray, float]:
    w_pos = np.maximum(0.0, np.log(sample_count / 2 + 1) - np.log(np.arange(1, sample_count + 1)))
    w_sum = float(np.sum(w_pos))
    if w_sum <= 0.0:
        msg = "Invalid utility weights: positive weight sum must be > 0."
        raise ValueError(msg)
    w_pos /= w_sum
    mu_eff_pos = float(1.0 / np.sum(w_pos**2))
    return w_pos, w_pos - (1.0 / sample_count), mu_eff_pos


class XNESStatus(Enum):
    """Outcome of one `XNES.tell` update step."""

    OK = auto()
    SIGMA_MIN = auto()
    SIGMA_MAX = auto()
    SIGMA_INF = auto()
    LOC_INF = auto()
    SCALE_INF = auto()
    STEP_SIZE_PATH_INF = auto()
    SCALE_COND_INF = auto()
    SCALE_COND_ERROR = auto()
    SCALE_COND_MAX = auto()
    LOC_STEP_MIN = auto()
    SCALE_NORM_MIN = auto()


class XNES:
    """Exponential Natural Evolution Strategies distribution state.

    The distribution is stored in factored form as ``sigma * B`` and updated
    from ranked standardized samples.

    Args:
        x0: Initial mean vector.
        sigma0: Initial scale, either scalar, diagonal vector, or full matrix.
        p_sigma: Optional CSA evolution path.

    Runtime attributes `csa_enabled`, `eta_mu`, `eta_sigma`, and `eta_B` are
    initialized to built-in defaults and may be reassigned directly. `eta_B`
    acts as a multiplier on the built-in dimension-dependent shape learning
    rate heuristic.

    Raises:
        ValueError: If the supplied shapes are inconsistent, the scale matrix is
            not positive with finite determinant.
    """

    MIN_SIGMA = 1e-20
    MAX_SIGMA = 1e20
    MAX_CONDITION = 1e14

    def __init__(self, x0: np.ndarray, sigma0: np.ndarray | float, p_sigma: np.ndarray | None = None) -> None:
        self.mu = np.asarray(x0, dtype=float)
        self.sigma: float
        self.B: np.ndarray
        self.p_sigma: np.ndarray
        self.csa_enabled = True
        self.eta_mu = 1.0
        self.eta_sigma = 1.0
        self.eta_B = 1.0

        if self.dim == 0:
            self.sigma = 1.0
            self.B = np.eye(0)
            self.p_sigma = np.zeros(0)
            return

        scale0 = _normalize_scale_matrix(sigma0, self.dim)

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
        """Dimension of the search space."""

        return int(self.mu.size)

    @property
    def scale(self) -> np.ndarray:
        """Current full scale matrix `sigma * B`."""

        return self.sigma * self.B

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Map standardized samples `z` into current distribution coordinates."""

        z = _validated_samples(samples, self.dim)
        return self.mu[:, None] + self.scale @ z

    def ask(
        self,
        num_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample a mirrored standardized batch.

        Args:
            num_samples: Optional batch size. Values below two are clamped, and
                odd values are rounded up to keep mirrored pairs.
            rng: Optional NumPy random generator.

        Returns:
            Standardized samples `z`.
        """

        if self.dim == 0:
            n = int(num_samples) if num_samples is not None else 4
            return np.zeros((0, n))

        n = _default_sample_count(num_samples, self.dim)
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

        return np.hstack([z_half, -z_half])

    def tell(self, samples: np.ndarray, ranking: list[int], eps: float = 1e-10) -> XNESStatus:
        """Apply one xNES update from ranked standardized samples.

        Args:
            samples: Standardized sample matrix with shape `(dim, n)`.
            ranking: Permutation of sample indices ordered from best to worst.
            eps: Numerical stopping threshold.

        Returns:
            An `XNESStatus` indicating whether the update succeeded or hit a numerical or convergence stop condition.

        Raises:
            ValueError: If sample shapes are inconsistent, samples are not
                finite, or the ranking is not a valid permutation.
        """

        if self.dim == 0:
            return XNESStatus.SCALE_NORM_MIN

        samples = _validated_samples(samples, self.dim)
        n = samples.shape[1]
        d = self.dim
        if len(ranking) != n or sorted(ranking) != list(range(n)):
            msg = "ranking must be a permutation matching sample count."
            raise ValueError(msg)
        w_pos, w_active, mu_eff_pos = _utility_weights(n)
        z_sorted = samples[:, ranking]

        grad_mu = z_sorted @ w_active
        grad_mu_pos = z_sorted @ w_pos
        grad_M = (z_sorted * w_active) @ z_sorted.T
        grad_sigma = float(np.trace(grad_M) / d)
        grad_B_shape = grad_M - grad_sigma * np.eye(d)

        mu_step = self.eta_mu * self.sigma * (self.B @ grad_mu)
        self.mu += mu_step

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
        if not np.isfinite(self.sigma):
            return XNESStatus.SIGMA_INF

        min_sigma = max(self.MIN_SIGMA, eps)
        if self.sigma < min_sigma:
            return XNESStatus.SIGMA_MIN
        if self.sigma > self.MAX_SIGMA:
            return XNESStatus.SIGMA_MAX

        eta_B = self.eta_B * _default_eta_B(d)
        self.B = self.B @ expm(0.5 * eta_B * grad_B_shape)

        sign, logdet = np.linalg.slogdet(self.B)
        if sign <= 0 or not np.isfinite(logdet):
            return XNESStatus.SCALE_INF
        self.B *= np.exp(-logdet / d)

        if not np.all(np.isfinite(self.mu)):
            return XNESStatus.LOC_INF
        if not np.all(np.isfinite(self.B)):
            return XNESStatus.SCALE_INF
        if not np.all(np.isfinite(self.p_sigma)):
            return XNESStatus.STEP_SIZE_PATH_INF

        scale = self.scale
        if not np.all(np.isfinite(scale)):
            return XNESStatus.SCALE_INF
        if norm(scale, 2) < eps:
            return XNESStatus.SCALE_NORM_MIN
        if norm(mu_step, 2) < eps:
            return XNESStatus.LOC_STEP_MIN

        try:
            cond_scale = float(cond(scale))
        except np.linalg.LinAlgError:
            return XNESStatus.SCALE_COND_ERROR
        if not np.isfinite(cond_scale):
            return XNESStatus.SCALE_COND_INF

        max_condition = min(self.MAX_CONDITION, 1.0 / eps)
        if cond_scale > max_condition:
            return XNESStatus.SCALE_COND_MAX
        return XNESStatus.OK
