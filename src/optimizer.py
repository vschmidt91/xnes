from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np

from src.xnes import XNES

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
Result: TypeAlias = float | Sequence[float]


@dataclass
class Parameter:
    name: str
    value: float


@dataclass(frozen=True)
class _Prior:
    loc: float
    scale: float


class Optimizer:
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

        self._xnes: XNES | None = None
        self._state_names: list[str] = []
        self._batch_z: np.ndarray | None = None
        self._batch_x: np.ndarray | None = None
        self._results: list[tuple[float, ...]] = []

    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> Parameter:
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
        if name not in self._registry:
            msg = f"Unknown parameter '{name}'."
            raise KeyError(msg)
        del self._registry[name]
        del self._priors[name]
        self._reconcile_after_registry_change()

    def save(self) -> JSON:
        self._ensure_runtime_ready()
        assert self._xnes is not None

        return {
            "names": list(self._state_names),
            "loc": self._xnes.loc.tolist(),
            "scale": self._xnes.scale.tolist(),
            "p_sigma": self._xnes.p_sigma.tolist(),
            "batch_z": None if self._batch_z is None else self._batch_z.tolist(),
            "batch_x": None if self._batch_x is None else self._batch_x.tolist(),
            "results": [list(item) for item in self._results],
            "rng_state": cast(JSON, self._rng.bit_generator.state),
        }

    def load(self, state: JSON) -> None:
        if not isinstance(state, Mapping):
            return
        state_obj = cast(Mapping[str, JSON], state)

        names = cast(list[str], state_obj["names"])
        loc = np.asarray(state_obj["loc"], dtype=float)
        scale = np.asarray(state_obj["scale"], dtype=float)
        p_sigma = np.asarray(state_obj["p_sigma"], dtype=float)

        raw_batch_z = state_obj.get("batch_z")
        raw_batch_x = state_obj.get("batch_x")
        batch_z = None if raw_batch_z is None else np.asarray(raw_batch_z, dtype=float)
        batch_x = None if raw_batch_x is None else np.asarray(raw_batch_x, dtype=float)

        raw_results = state_obj.get("results")
        if raw_results is None:
            results: list[tuple[float, ...]] = []
        else:
            result_rows = cast(Sequence[Sequence[float]], raw_results)
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

    def tell(self, result: Result) -> bool:
        self._ensure_runtime_ready()
        assert self._xnes is not None
        assert self._batch_z is not None

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

    def _ensure_runtime_ready(self) -> None:
        if self._xnes is None:
            self._reconcile_state(None, None, None, None, None, None, [])
        if self._batch_x is None:
            self._reset_batch()

    def _reconcile_after_registry_change(self) -> None:
        if self._xnes is None:
            self._reconcile_state(None, None, None, None, None, None, [])
            return
        self._reconcile_state(
            old_names=list(self._state_names),
            old_loc=self._xnes.loc.copy(),
            old_scale=self._xnes.scale.copy(),
            old_p_sigma=self._xnes.p_sigma.copy(),
            old_batch_z=None,
            old_batch_x=None,
            old_results=[],
        )

    def _reconcile_state(
        self,
        old_names: list[str] | None,
        old_loc: np.ndarray | None,
        old_scale: np.ndarray | None,
        old_p_sigma: np.ndarray | None,
        old_batch_z: np.ndarray | None,
        old_batch_x: np.ndarray | None,
        old_results: list[tuple[float, ...]],
    ) -> None:
        new_names = self._ordered_names()
        new_loc, new_scale = self._build_initial_state(new_names)
        restored_p_sigma = np.zeros(len(new_names), dtype=float)
        restored_batch_z: np.ndarray | None = None
        restored_batch_x: np.ndarray | None = None
        restored_results: list[tuple[float, ...]] = []

        if old_names is not None and old_loc is not None and old_scale is not None and old_p_sigma is not None:
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

            if old_batch_z is not None and old_batch_x is not None:
                n_samples = old_batch_z.shape[1]
                restored_batch_z = np.zeros((len(new_names), n_samples), dtype=float)
                restored_batch_x = np.tile(new_loc[:, None], (1, n_samples))
                if curr_idx:
                    restored_batch_z[curr_idx, :] = old_batch_z[prev_idx, :]
                    restored_batch_x[curr_idx, :] = old_batch_x[prev_idx, :]
                restored_results = list(old_results)

        self._xnes = self._new_xnes(new_loc, new_scale, restored_p_sigma)
        self._state_names = new_names

        if restored_batch_z is not None and restored_batch_x is not None:
            self._batch_z = restored_batch_z
            self._batch_x = restored_batch_x
            self._results = restored_results
            sample_idx = min(len(self._results), restored_batch_x.shape[1] - 1) if restored_batch_x.shape[1] > 0 else 0
            self._apply_sample_values(sample_idx)
        else:
            self._reset_batch()

    def _reset_batch(self) -> None:
        assert self._xnes is not None
        pop_size = self._resolve_population_size(self._xnes.dim)
        self._batch_z, self._batch_x = self._xnes.ask(pop_size, self._rng)
        self._results = []
        self._apply_sample_values(0)

    def _apply_sample_values(self, sample_index: int) -> None:
        if self._batch_x is None or self._batch_x.shape[1] == 0:
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
        assert self._xnes is not None
        sigma = float(self._xnes.sigma)
        if not np.isfinite(sigma):
            return self._handle_instability()
        if sigma < self._MIN_SIGMA:
            return self._handle_instability()
        if sigma > self._MAX_SIGMA:
            return self._handle_instability()
        if not np.all(np.isfinite(self._xnes.loc)) or not np.all(np.isfinite(self._xnes.B)):
            return self._handle_instability()

        cond_value = self._safe_condition_number()
        if not np.isfinite(cond_value):
            return self._handle_instability()
        if cond_value > self._MAX_CONDITION:
            return self._handle_instability()
        return False

    def _safe_condition_number(self) -> float:
        if self._xnes is None:
            return float("nan")
        try:
            return float(np.linalg.cond(self._xnes.scale))
        except Exception:
            return float("inf")


def _normalize_result(result: Result) -> tuple[float, ...]:
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
