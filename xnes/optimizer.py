"""Named-parameter optimizer built on top of the xNES update rule.

The wrapper manages a dynamic registry of scalar parameters, preserves a stable
lexicographic parameter order, and exposes a serializable ask/tell style loop
through mutable :class:`Parameter` views.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from .xnes import XNES


@dataclass
class Parameter:
    """Mutable scalar parameter view exposed to the user.

    Attributes:
        name: Stable parameter identifier.
        value: Current sample value for the active candidate.
    """

    name: str
    value: float


@dataclass(frozen=True)
class ParameterInfo:
    """Immutable snapshot of a registered parameter."""

    name: str
    value: float
    loc: float
    scale: float
    prior_loc: float
    prior_scale: float


@dataclass(frozen=True)
class _Prior:
    loc: float
    scale: float


class Optimizer:
    """Maximizing optimizer with dynamic named parameters.

    Parameters are registered by name and sampled in lexicographic order so the
    optimization state is independent of registration order. Calls to `tell`
    consume one result for the current sample and advance the batch until an
    xNES update is applied.

    Args:
        pop_size: Optional batch size. Odd values are rounded up to the next
            even value.
        csa_enabled: Enable cumulative step-size adaptation.
        eta_mu: Learning rate for the mean update.
        eta_sigma: Learning rate for the global scale update.
        eta_B: Optional learning rate for the normalized shape matrix update.

    Raises:
        ValueError: If `pop_size` is non-positive or any learning rate is not a
            positive finite float.
    """

    _RESTART_ON_FAILURE = True
    _MIN_SIGMA = 1e-20
    _MAX_SIGMA = 1e20
    _MAX_CONDITION = 1e14
    _DEFAULT_REEVALUATION_FRACTION = 1.0 / 3.0
    _REEVALUATION_Z_TARGET = 1.64
    _MAX_REEVALUATIONS_PER_GENERATION = 32

    def __init__(
        self,
        pop_size: int | None = None,
        *,
        csa_enabled: bool = True,
        eta_mu: float = 1.0,
        eta_sigma: float = 1.0,
        eta_B: float | None = None,
        reevaluation_fraction: float = _DEFAULT_REEVALUATION_FRACTION,
        reevaluation_confidence: float = 0.0,
    ) -> None:
        if pop_size is not None and pop_size <= 0:
            msg = "pop_size must be positive when provided."
            raise ValueError(msg)

        self.pop_size = pop_size
        self.csa_enabled = bool(csa_enabled)
        self.eta_mu = _positive_finite(eta_mu, "eta_mu")
        self.eta_sigma = _positive_finite(eta_sigma, "eta_sigma")
        self.eta_B = None if eta_B is None else _positive_finite(eta_B, "eta_B")
        self.reevaluation_fraction = _fraction_in_unit_interval(reevaluation_fraction, "reevaluation_fraction")
        self.reevaluation_confidence = _nonnegative_finite(
            reevaluation_confidence,
            "reevaluation_confidence",
        )

        self._rng = np.random.default_rng()
        self._registry: dict[str, Parameter] = {}
        self._priors: dict[str, _Prior] = {}

        self._xnes = self._new_xnes(np.zeros(0), np.eye(0), np.zeros(0))
        self._state_names: list[str] = []
        self._batch_z = np.zeros((0, 0))
        self._batch_x = np.zeros((0, 0))
        self._batch_observations: list[list[tuple[float, ...]]] = []
        self._evaluation_order: list[int] = []
        self._fresh_evaluations = 0
        self._in_reevaluation = False
        self._current_sample_index = 0
        self._reset_batch()

    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> Parameter:
        """Register a parameter or return the existing one.

        Args:
            name: Unique parameter name.
            loc: Initial mean used for new parameters.
            scale: Initial standard deviation used for new parameters.

        Returns:
            The mutable parameter view stored in the registry.

        Raises:
            ValueError: If `scale <= 0`.
        """

        if scale <= 0:
            msg = "scale must be > 0."
            raise ValueError(msg)

        existing = self._registry.get(name)
        if existing is not None:
            return existing

        parameter = Parameter(name=name, value=float(loc))
        self._registry[name] = parameter
        self._priors[name] = _Prior(loc=float(loc), scale=float(scale))
        self._reconcile_after_registry_change()
        return parameter

    def remove(self, name: str) -> None:
        """Remove a parameter from the registry and reconcile optimizer state.

        Args:
            name: Registered parameter name.

        Raises:
            KeyError: If the parameter is unknown.
        """

        if name not in self._registry:
            msg = f"Unknown parameter '{name}'."
            raise KeyError(msg)
        del self._registry[name]
        del self._priors[name]
        self._reconcile_after_registry_change()

    def save(self) -> dict[str, object]:
        """Serialize the current optimizer state.

        Returns:
            JSON-compatible snapshot of registry order, xNES state, batch data, results, and RNG state.
        """

        return {
            "names": list(self._state_names),
            "loc": self._xnes.loc.tolist(),
            "scale": self._xnes.scale.tolist(),
            "p_sigma": self._xnes.p_sigma.tolist(),
            "batch_z": self._batch_z.tolist(),
            "batch_x": self._batch_x.tolist(),
            "results": [list(item) for item in self._ordered_results()],
            "observations": [
                [[float(value) for value in result] for result in sample] for sample in self._batch_observations
            ],
            "evaluation_order": list(self._evaluation_order),
            "fresh_evaluations": self._fresh_evaluations,
            "in_reevaluation": self._in_reevaluation,
            "current_sample_index": self._current_sample_index,
            "rng_state": dict(self._rng.bit_generator.state),
        }

    def load(self, state: object) -> None:
        """Restore optimizer state from `save`.

        If no parameters are registered yet, the registry is reconstructed from
        the serialized names and diagonal scale entries. Non-mapping inputs are
        ignored.

        Args:
            state: Serialized optimizer state.
        """

        if not isinstance(state, Mapping):
            return
        state_obj = cast(Mapping[str, object], state)

        names = cast(list[str], state_obj["names"])
        loc = np.asarray(state_obj["loc"], dtype=float)
        scale = np.asarray(state_obj["scale"], dtype=float)
        p_sigma = np.asarray(state_obj["p_sigma"], dtype=float)
        batch_z = np.asarray(state_obj["batch_z"], dtype=float)
        batch_x = np.asarray(state_obj["batch_x"], dtype=float)
        result_rows = cast(Sequence[Sequence[float]], state_obj.get("results", []))
        results = [tuple(float(value) for value in row) for row in result_rows]
        raw_observations = cast(Sequence[Sequence[Sequence[float]]], state_obj.get("observations", []))
        observations = [[tuple(float(value) for value in result) for result in sample] for sample in raw_observations]
        evaluation_order = [int(idx) for idx in cast(Sequence[int], state_obj.get("evaluation_order", []))]
        fresh_evaluations = _coerce_int(state_obj.get("fresh_evaluations"), len(results))
        in_reevaluation = bool(state_obj.get("in_reevaluation", False))
        default_sample_index = min(len(results), max(batch_x.shape[1] - 1, 0))
        current_sample_index = _coerce_int(state_obj.get("current_sample_index"), default_sample_index)

        raw_rng_state = state_obj.get("rng_state")
        if isinstance(raw_rng_state, Mapping):
            self._rng.bit_generator.state = dict(raw_rng_state)

        if not self._registry:
            diag = np.abs(np.diag(np.asarray(scale, dtype=float)))
            for idx, name in enumerate(names):
                scale_value = float(diag[idx]) if idx < diag.size else 1.0
                if not np.isfinite(scale_value) or scale_value <= 0.0:
                    scale_value = 1.0
                self._registry[name] = Parameter(name=name, value=float(loc[idx]))
                self._priors[name] = _Prior(loc=float(loc[idx]), scale=scale_value)

        self._reconcile_state(
            old_names=names,
            old_loc=loc,
            old_scale=scale,
            old_p_sigma=p_sigma,
            old_batch_z=batch_z,
            old_batch_x=batch_x,
            old_results=results,
            old_observations=observations,
            old_evaluation_order=evaluation_order,
            old_fresh_evaluations=fresh_evaluations,
            old_in_reevaluation=in_reevaluation,
            old_current_sample_index=current_sample_index,
        )

    def tell(self, result: float | Sequence[float] | np.ndarray) -> bool:
        """Submit one objective result for the current sample.

        Scalar results are treated as one-element tuples. Sequence results are
        ranked lexicographically, with larger tuples considered better.

        Args:
            result: Objective value for the current sample.

        Returns:
            `True` when the current batch has been fully consumed and an update or restart step completed.

        Raises:
            TypeError: If `result` is neither a scalar nor a numeric sequence.
            ValueError: If `result` is an empty sequence.
        """

        normalized = _normalize_result(result)
        if self._batch_observations:
            self._batch_observations[self._current_sample_index].append(normalized)
        self._evaluation_order.append(self._current_sample_index)

        if not self._in_reevaluation:
            self._fresh_evaluations += 1
            if self._fresh_evaluations < self._batch_z.shape[1]:
                self._current_sample_index = self._fresh_evaluations
                self._apply_sample_values(self._current_sample_index)
                return False
            self._in_reevaluation = True

        if self._should_finish_generation():
            self._finalize_generation()
            return True

        next_sample_index = self._select_reevaluation_candidate()
        if next_sample_index is None:
            self._finalize_generation()
            return True

        self._current_sample_index = next_sample_index
        self._apply_sample_values(self._current_sample_index)
        return False

    def set_best(self) -> None:
        """Set all registered parameter values to the current population mean.

        This mutates exposed :class:`Parameter` views in place without changing
        optimizer state. It is intended for evaluation/inference after training.
        To continue training afterwards, restore a previously saved state so the
        ask/tell sample cursor remains aligned.
        """

        for row, name in enumerate(self._state_names):
            self._registry[name].value = float(self._xnes.loc[row])

    def get_info(self) -> list[ParameterInfo]:
        """Return immutable snapshots of all registered parameters.

        Returns:
            A new list of :class:`ParameterInfo` items in lexicographic name
            order. `value` is the current sampled parameter value, `loc` is the
            current xNES mean, `scale` is the corresponding diagonal entry of
            the current xNES scale matrix, and `prior_*` values come from the
            parameter registration.
        """

        scale_diag = np.diag(self._xnes.scale)
        return [
            ParameterInfo(
                name=name,
                value=float(self._registry[name].value),
                loc=float(self._xnes.loc[row]),
                scale=float(scale_diag[row]),
                prior_loc=self._priors[name].loc,
                prior_scale=self._priors[name].scale,
            )
            for row, name in enumerate(self._state_names)
        ]

    def _ordered_names(self) -> list[str]:
        return sorted(self._registry)

    def _build_initial_state(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        loc = np.array([self._priors[name].loc for name in names], dtype=float)
        scale_diag = np.array([self._priors[name].scale for name in names], dtype=float)
        return loc, np.diag(scale_diag)

    def _resolve_population_size(self, dim: int) -> int | None:
        if self.pop_size is None:
            return None
        n = self.pop_size
        if n % 2 == 1:
            n += 1
        if dim > 0 and n < 2:
            n = 2
        return n

    def _new_xnes(self, loc: np.ndarray, scale: np.ndarray, p_sigma: np.ndarray) -> XNES:
        return XNES(
            loc,
            scale,
            p_sigma=p_sigma,
            csa_enabled=self.csa_enabled,
            eta_mu=self.eta_mu,
            eta_sigma=self.eta_sigma,
            eta_B=self.eta_B,
        )

    def _reconcile_after_registry_change(self) -> None:
        self._reconcile_state(
            old_names=self._state_names,
            old_loc=self._xnes.loc.copy(),
            old_scale=self._xnes.scale.copy(),
            old_p_sigma=self._xnes.p_sigma.copy(),
            old_batch_z=np.zeros((0, 0)),
            old_batch_x=np.zeros((0, 0)),
            old_results=[],
            old_observations=[],
            old_evaluation_order=[],
            old_fresh_evaluations=0,
            old_in_reevaluation=False,
            old_current_sample_index=0,
        )

    def _reconcile_state(
        self,
        old_names: list[str],
        old_loc: np.ndarray,
        old_scale: np.ndarray,
        old_p_sigma: np.ndarray,
        old_batch_z: np.ndarray,
        old_batch_x: np.ndarray,
        old_results: list[tuple[float, ...]],
        old_observations: list[list[tuple[float, ...]]],
        old_evaluation_order: list[int],
        old_fresh_evaluations: int,
        old_in_reevaluation: bool,
        old_current_sample_index: int,
    ) -> None:
        new_names = self._ordered_names()
        new_loc, new_scale = self._build_initial_state(new_names)
        restored_p_sigma = np.zeros(len(new_names), dtype=float)
        old_index = {name: idx for idx, name in enumerate(old_names)}
        curr_idx: list[int] = []
        prev_idx: list[int] = []
        for i, name in enumerate(new_names):
            j = old_index.get(name)
            if j is not None:
                curr_idx.append(i)
                prev_idx.append(j)

        if curr_idx:
            new_loc[curr_idx] = old_loc[prev_idx]
            new_scale[np.ix_(curr_idx, curr_idx)] = old_scale[np.ix_(prev_idx, prev_idx)]
            restored_p_sigma[curr_idx] = old_p_sigma[prev_idx]

        self._xnes = self._new_xnes(new_loc, new_scale, restored_p_sigma)
        self._state_names = new_names

        if old_batch_z.shape[1] > 0:
            restored_batch_z = np.zeros((len(new_names), old_batch_z.shape[1]), dtype=float)
            restored_batch_x = np.zeros((len(new_names), old_batch_z.shape[1]), dtype=float)
            restored_batch_x[:] = new_loc[:, None]
            if curr_idx:
                restored_batch_z[curr_idx, :] = old_batch_z[prev_idx, :]
                restored_batch_x[curr_idx, :] = old_batch_x[prev_idx, :]
            self._batch_z = restored_batch_z
            self._batch_x = restored_batch_x
            self._restore_generation_state(
                old_results,
                old_observations,
                old_evaluation_order,
                old_fresh_evaluations,
                old_in_reevaluation,
                old_current_sample_index,
            )
        else:
            self._reset_batch()

    def _reset_batch(self) -> None:
        pop_size = self._resolve_population_size(self._xnes.dim)
        self._batch_z, self._batch_x = self._xnes.ask(pop_size, self._rng)
        self._batch_observations = [[] for _ in range(self._batch_z.shape[1])]
        self._evaluation_order = []
        self._fresh_evaluations = 0
        self._in_reevaluation = False
        self._current_sample_index = 0
        self._apply_sample_values(self._current_sample_index)

    def _apply_sample_values(self, sample_index: int) -> None:
        if self._batch_x.shape[1] == 0:
            return
        idx = sample_index % self._batch_x.shape[1]
        for row, name in enumerate(self._state_names):
            self._registry[name].value = float(self._batch_x[row, idx])

    def _restore_generation_state(
        self,
        old_results: list[tuple[float, ...]],
        old_observations: list[list[tuple[float, ...]]],
        old_evaluation_order: list[int],
        old_fresh_evaluations: int,
        old_in_reevaluation: bool,
        old_current_sample_index: int,
    ) -> None:
        n = self._batch_z.shape[1]
        if len(old_observations) == n:
            self._batch_observations = [list(sample) for sample in old_observations]
        else:
            self._batch_observations = [[] for _ in range(n)]
            for idx, result in enumerate(old_results[:n]):
                self._batch_observations[idx].append(result)

        self._evaluation_order = [idx for idx in old_evaluation_order if 0 <= idx < n]
        if not self._evaluation_order and old_results:
            self._evaluation_order = list(range(min(len(old_results), n)))

        self._fresh_evaluations = min(max(old_fresh_evaluations, 0), n)
        if self._fresh_evaluations == 0:
            self._fresh_evaluations = sum(1 for sample in self._batch_observations if sample)
        self._in_reevaluation = bool(old_in_reevaluation or self._fresh_evaluations >= n)

        if n == 0:
            self._current_sample_index = 0
            return

        if not 0 <= old_current_sample_index < n:
            if not self._in_reevaluation and self._fresh_evaluations < n:
                old_current_sample_index = self._fresh_evaluations
            elif self._evaluation_order:
                old_current_sample_index = self._evaluation_order[-1]
            else:
                old_current_sample_index = 0

        self._current_sample_index = old_current_sample_index
        self._apply_sample_values(self._current_sample_index)

    def _ordered_results(self) -> list[tuple[float, ...]]:
        return [
            self._batch_observations[sample_idx][occurrence_idx]
            for sample_idx, occurrence_idx in self._ordered_result_keys()
        ]

    def _ordered_result_keys(self) -> list[tuple[int, int]]:
        counts = [0 for _ in self._batch_observations]
        ordered: list[tuple[int, int]] = []
        for sample_idx in self._evaluation_order:
            if 0 <= sample_idx < len(self._batch_observations):
                occurrence_idx = counts[sample_idx]
                if occurrence_idx < len(self._batch_observations[sample_idx]):
                    ordered.append((sample_idx, occurrence_idx))
                    counts[sample_idx] += 1
        return ordered

    def _should_finish_generation(self) -> bool:
        n = self._batch_z.shape[1]
        if n <= 1:
            return True
        if self.reevaluation_confidence <= 0.0:
            return True
        if self._reevaluation_count() >= self._MAX_REEVALUATIONS_PER_GENERATION:
            return True
        if not self._supports_scalar_confidence():
            return True

        ranking = self._ranking_from_observations()
        k = self._elite_count(n)
        left = ranking[k - 1]
        right = ranking[k]
        if len(self._batch_observations[left]) < 2 or len(self._batch_observations[right]) < 2:
            return False

        delta, standard_error = self._boundary_separation(left, right)
        if standard_error <= 0.0:
            return True
        return bool(delta / standard_error >= self.reevaluation_confidence)

    def _supports_scalar_confidence(self) -> bool:
        return bool(self._batch_observations) and all(
            sample and len(sample[0]) == 1 for sample in self._batch_observations
        )

    def _select_reevaluation_candidate(self) -> int | None:
        n = self._batch_z.shape[1]
        if n <= 1:
            return None

        ranking = self._ranking_from_observations()
        k = self._elite_count(n)
        left = ranking[k - 1]
        right = ranking[k]
        left_count = len(self._batch_observations[left])
        right_count = len(self._batch_observations[right])

        if left_count < 2 or right_count < 2:
            if left_count <= right_count:
                return left
            return right

        if not self._supports_scalar_confidence():
            return None

        left_gain = self._variance_reduction_gain(left)
        right_gain = self._variance_reduction_gain(right)
        if left_gain > right_gain:
            return left
        if right_gain > left_gain:
            return right
        if left_count <= right_count:
            return left
        return right

    def _finalize_generation(self) -> None:
        ranking = self._ranking_from_observations()
        stopped = bool(self._xnes.tell(self._batch_z, ranking))

        restarted = False
        if stopped:
            restarted = self._handle_instability()
        if not restarted:
            restarted = self._stabilize_runtime()
        if not restarted:
            self._reset_batch()

    def _ranking_from_observations(self) -> list[int]:
        aggregates = [self._aggregate_sample_results(sample) for sample in self._batch_observations]
        return sorted(range(len(aggregates)), key=lambda idx: aggregates[idx], reverse=True)

    def _aggregate_sample_results(self, observations: list[tuple[float, ...]]) -> tuple[float, ...]:
        if not observations:
            return tuple()
        values = np.asarray(observations, dtype=float)
        return tuple(float(value) for value in np.mean(values, axis=0))

    def _elite_count(self, population_size: int) -> int:
        return max(1, min(population_size - 1, math.ceil(population_size * self.reevaluation_fraction)))

    def _reevaluation_count(self) -> int:
        return max(0, len(self._evaluation_order) - self._fresh_evaluations)

    def _boundary_separation(self, left: int, right: int) -> tuple[float, float]:
        left_values = np.asarray(self._batch_observations[left], dtype=float).reshape(-1)
        right_values = np.asarray(self._batch_observations[right], dtype=float).reshape(-1)
        left_mean = float(np.mean(left_values))
        right_mean = float(np.mean(right_values))
        left_var = self._sample_variance(left_values)
        right_var = self._sample_variance(right_values)
        standard_error = math.sqrt(left_var / left_values.size + right_var / right_values.size)
        return left_mean - right_mean, standard_error

    def _variance_reduction_gain(self, sample_index: int) -> float:
        values = np.asarray(self._batch_observations[sample_index], dtype=float).reshape(-1)
        count = values.size
        if count < 2:
            return math.inf
        variance = self._sample_variance(values)
        return variance / (count * (count + 1))

    def _sample_variance(self, values: np.ndarray) -> float:
        if values.size < 2:
            return math.inf
        return float(np.var(values, ddof=1))

    def _handle_instability(self) -> bool:
        if not self._RESTART_ON_FAILURE:
            return False
        self._restart_distribution()
        return True

    def _restart_distribution(self) -> None:
        names = self._ordered_names()
        loc, scale = self._build_initial_state(names)
        self._xnes = self._new_xnes(loc, scale, np.zeros(len(names), dtype=float))
        self._state_names = names
        self._reset_batch()

    def _stabilize_runtime(self) -> bool:
        sigma = float(self._xnes.sigma)
        if not np.isfinite(sigma) or not self._MIN_SIGMA <= sigma <= self._MAX_SIGMA:
            return self._handle_instability()
        if not np.all(np.isfinite(self._xnes.loc)) or not np.all(np.isfinite(self._xnes.B)):
            return self._handle_instability()

        try:
            cond_value = float(np.linalg.cond(self._xnes.scale))
        except np.linalg.LinAlgError:
            return self._handle_instability()
        if not np.isfinite(cond_value):
            return self._handle_instability()
        if cond_value > self._MAX_CONDITION:
            return self._handle_instability()
        return False


def _normalize_result(result: float | Sequence[float] | np.ndarray) -> tuple[float, ...]:
    if isinstance(result, (int, float)):
        return (float(result),)
    if isinstance(result, np.ndarray):
        values = tuple(float(item) for item in result.reshape(-1))
        if not values:
            msg = "Sequence result cannot be empty."
            raise ValueError(msg)
        return values
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        values = tuple(float(item) for item in result)
        if not values:
            msg = "Sequence result cannot be empty."
            raise ValueError(msg)
        return values
    msg = f"Unsupported result type: {type(result)!r}"
    raise TypeError(msg)


def _positive_finite(value: float, field: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        msg = f"{field} must be a positive finite float."
        raise ValueError(msg)
    return out


def _nonnegative_finite(value: float, field: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        msg = f"{field} must be a non-negative finite float."
        raise ValueError(msg)
    return out


def _fraction_in_unit_interval(value: float, field: str) -> float:
    out = float(value)
    if not np.isfinite(out) or not 0.0 < out < 1.0:
        msg = f"{field} must be a finite float in (0, 1)."
        raise ValueError(msg)
    return out


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
