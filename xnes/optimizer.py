"""Named-parameter optimizer built on top of the xNES update rule.

The wrapper manages a dynamic registry of scalar parameters, preserves a stable
lexicographic parameter order, and exposes a serializable ask/tell style loop
through mutable :class:`Parameter` views.
"""

from __future__ import annotations

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

    def __init__(
        self,
        pop_size: int | None = None,
        *,
        csa_enabled: bool = True,
        eta_mu: float = 1.0,
        eta_sigma: float = 1.0,
        eta_B: float | None = None,
    ) -> None:
        if pop_size is not None and pop_size <= 0:
            msg = "pop_size must be positive when provided."
            raise ValueError(msg)

        self.pop_size = pop_size
        self.csa_enabled = bool(csa_enabled)
        self.eta_mu = _positive_finite(eta_mu, "eta_mu")
        self.eta_sigma = _positive_finite(eta_sigma, "eta_sigma")
        self.eta_B = None if eta_B is None else _positive_finite(eta_B, "eta_B")

        self._rng = np.random.default_rng()
        self._registry: dict[str, Parameter] = {}
        self._priors: dict[str, _Prior] = {}

        self._xnes = self._new_xnes(np.zeros(0), np.eye(0), np.zeros(0))
        self._state_names: list[str] = []
        self._batch_z = np.zeros((0, 0))
        self._batch_x = np.zeros((0, 0))
        self._results: list[tuple[float, ...]] = []
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
            "results": [list(item) for item in self._results],
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

        self._results.append(_normalize_result(result))

        done = len(self._results) >= self._batch_z.shape[1]
        if done:
            ranking = sorted(range(len(self._results)), key=lambda idx: self._results[idx], reverse=True)
            stopped = bool(self._xnes.tell(self._batch_z, ranking))

            restarted = False
            if stopped:
                restarted = self._handle_instability()
            if not restarted:
                restarted = self._stabilize_runtime()
            if not restarted:
                self._reset_batch()
            return True

        self._apply_sample_values(len(self._results))
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

    def get_values(self) -> Mapping[str, float]:
        """Return a mapping of current parameter values by name.

        Returns:
            A new name-to-value dictionary in lexicographic name order.
        """

        return {name: float(self._registry[name].value) for name in self._ordered_names()}

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
            self._results = list(old_results)
            sample_idx = min(len(self._results), restored_batch_x.shape[1] - 1)
            self._apply_sample_values(sample_idx)
        else:
            self._reset_batch()

    def _reset_batch(self) -> None:
        pop_size = self._resolve_population_size(self._xnes.dim)
        self._batch_z, self._batch_x = self._xnes.ask(pop_size, self._rng)
        self._results = []
        self._apply_sample_values(0)

    def _apply_sample_values(self, sample_index: int) -> None:
        if self._batch_x.shape[1] == 0:
            return
        idx = sample_index % self._batch_x.shape[1]
        for row, name in enumerate(self._state_names):
            self._registry[name].value = float(self._batch_x[row, idx])

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
