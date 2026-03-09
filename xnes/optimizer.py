"""Named-parameter optimizer built on top of the xNES update rule.

The wrapper is designed for a strict checkpointed workflow:

1. create `Optimizer`
2. register parameters with `add`
3. call `load` with `None` for a fresh run or a previously saved state
4. optionally call `set_context`
5. read `Parameter.value`
6. call `tell` exactly once for the current evaluation
7. call `save`

The registry is fixed once `load` has been called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from ._scheduler import BatchScheduler
from .xnes import XNES, XNESStatus


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
    """Immutable snapshot of a registered parameter.

    Attributes:
        name: Stable parameter identifier.
        value: Current sampled value exposed through `Parameter`.
        loc: Current xNES population mean for this parameter.
        scale: Current diagonal entry of the xNES scale matrix.
        prior_loc: Registration-time mean used for fresh runs.
        prior_scale: Registration-time standard deviation used for fresh runs.
    """

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


@dataclass(frozen=True)
class Report:
    """Outcome of one `Optimizer.tell` call.

    Attributes:
        completed_batch: Whether this result completed the current batch.
        matched_context: Whether sample selection used a mirrored context match.
        status: xNES status returned after the batch update.
        restarted: Whether the wrapper restarted from priors after the update.
    """

    completed_batch: bool
    matched_context: bool
    status: XNESStatus
    restarted: bool


@dataclass(frozen=True)
class LoadResult:
    """Outcome of one `Optimizer.load` call.

    Attributes:
        parameters_added: Registered parameter names not present in the loaded state.
        parameters_removed: Loaded parameter names not present in the current registry.
    """

    parameters_added: list[str]
    parameters_removed: list[str]


class Optimizer:
    """Maximizing optimizer with named scalar parameters.

    Parameters are registered by name and sampled in lexicographic order so the
    optimization state is independent of registration order.

    Intended call flow:

    1. create the optimizer
    2. register all parameters with `add`
    3. call `load`
    4. optionally call `set_context`
    5. evaluate the current `Parameter.value` values
    6. call `tell`
    7. call `save`

    After `load`, the registry is fixed and `add` is no longer
    allowed.

    Runtime configuration lives on the instance attributes `csa_enabled`,
    `eta_mu`, `eta_sigma`, and `eta_B`. Leaving any of them as `None`
    preserves the default chosen by `XNES`; assigning a concrete value
    overrides that default for newly created internal xNES states. `eta_B`
    scales the built-in dimension-dependent shape learning rate
    multiplicatively.

    Batch size is configured via the instance attribute `pop_size`. Leaving it
    as `None` keeps the xNES default; assigning an odd value rounds up to the
    next even batch size when a new batch is created.
    """

    def __init__(self) -> None:
        self.pop_size: int | None = None
        self.csa_enabled: bool | None = None
        self.eta_mu: float | None = None
        self.eta_sigma: float | None = None
        self.eta_B: float | None = None

        self._rng = np.random.default_rng()
        self._registry: dict[str, Parameter] = {}
        self._priors: dict[str, _Prior] = {}
        self._loaded = False

        self._xnes: XNES = self._new_xnes(np.zeros(0), np.eye(0), np.zeros(0))
        self._state_names: list[str] = []
        self._scheduler = BatchScheduler()

    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> Parameter:
        """Register a parameter or return the existing one.

        This is a setup-time operation. It must happen before `load`.

        Args:
            name: Unique parameter name.
            loc: Initial mean used for new parameters.
            scale: Initial standard deviation used for new parameters.

        Returns:
            The mutable parameter view stored in the registry.

        Raises:
            ValueError: If `scale <= 0`.
            RuntimeError: If called after `load`.
        """

        if scale <= 0:
            msg = "scale must be > 0."
            raise ValueError(msg)

        existing = self._registry.get(name)
        if existing is not None:
            return existing
        if self._loaded:
            msg = "Cannot add parameters after load()."
            raise RuntimeError(msg)

        parameter = Parameter(name=name, value=float(loc))
        self._registry[name] = parameter
        self._priors[name] = _Prior(loc=float(loc), scale=float(scale))
        return parameter

    def save(self) -> dict[str, object]:
        """Serialize the current optimizer state after `tell`.

        Under the intended workflow, callers save only after the current
        evaluation result has been reported via `tell`.

        Returns:
            JSON-compatible snapshot of registry order, xNES state, batch data, results, and RNG state.

        Raises:
            RuntimeError: If called after `set_context` but before `tell`.
        """

        if self._scheduler.active_context is not None:
            msg = "Cannot save after set_context() before tell()."
            raise RuntimeError(msg)

        return {
            "names": list(self._state_names),
            "loc": self._xnes.mu.tolist(),
            "scale": self._xnes.scale.tolist(),
            "step_size_path": self._xnes.p_sigma.tolist(),
            "batch_z": self._scheduler.batch_z.tolist(),
            "batch_x": self._scheduler.batch_x.tolist(),
            "results": [None if item is None else list(item) for item in self._scheduler.results],
            "context_waiting": dict(self._scheduler.context_waiting),
            "rng_state": dict(self._rng.bit_generator.state),
        }

    def load(self, state: object) -> LoadResult:
        """Restore optimizer state or initialize a fresh run.

        ``load(None)`` initializes a fresh optimizer state from the registered
        parameter priors. ``load(state)`` restores a previously saved state. If
        the registered parameter set changed, shared coordinates keep their
        learned state, added parameters start from their priors, removed
        parameters are dropped, and the in-flight batch is reconciled into the
        new parameter space.

        Args:
            state: `None` for a fresh run, or a serialized state produced by
                `save`.

        Returns:
            A `LoadResult` describing added and removed parameters.

        Raises:
            RuntimeError: If no parameters have been registered yet.
        """

        if not self._registry:
            msg = "Register parameters with add() before load()."
            raise RuntimeError(msg)
        expected_names = self._ordered_names()
        if state is None:
            self._reset_from_priors()
            self._loaded = True
            return LoadResult(parameters_added=list(expected_names), parameters_removed=[])

        state_obj = cast(Mapping[str, object], state)

        names = cast(list[str], state_obj["names"])
        loc = np.asarray(state_obj["loc"], dtype=float)
        scale = np.asarray(state_obj["scale"], dtype=float)
        step_size_path_json = state_obj["step_size_path"]
        step_size_path = np.asarray(step_size_path_json, dtype=float)
        batch_z = np.asarray(state_obj["batch_z"], dtype=float)
        batch_x = np.asarray(state_obj["batch_x"], dtype=float)
        result_rows = cast(Sequence[Sequence[float] | None], state_obj["results"])
        results = [None if row is None else tuple(float(value) for value in row) for row in result_rows]
        context_waiting = dict(cast(Mapping[str, int], state_obj["context_waiting"]))

        parameters_added = [name for name in expected_names if name not in names]
        parameters_removed = [name for name in names if name not in expected_names]
        loc, scale, step_size_path = self._reconcile_distribution_state(
            names,
            expected_names,
            loc,
            scale,
            step_size_path,
        )
        batch_z, batch_x = self._reconcile_batch_state(names, expected_names, batch_z, batch_x)

        self._rng.bit_generator.state = dict(cast(Mapping[str, object], state_obj["rng_state"]))
        self._xnes = self._new_xnes(loc, scale, step_size_path)
        self._state_names = expected_names
        self._scheduler.restore(batch_z, batch_x, results, context_waiting)
        self._apply_active_sample_values()
        self._loaded = True
        return LoadResult(
            parameters_added=parameters_added,
            parameters_removed=parameters_removed,
        )

    def _reset_from_priors(self) -> None:
        names = self._ordered_names()
        loc, scale = self._build_initial_state(names)
        self._xnes = self._new_xnes(loc, scale, np.zeros(len(names), dtype=float))
        self._state_names = names
        self._reset_batch()

    def set_context(self, context: str) -> bool:
        """Retarget the current sample selection using a string context.

        If the same context string was already seen for one side of a mirrored
        pair in the current batch, the mirrored sample is selected. Otherwise
        the current pending sample stays selected.

        This method is optional. If it is never called, sampling proceeds in the
        default batch order.

        Args:
            context: Human-readable objective-context identifier.

        Returns:
            `True` iff sample selection used the mirrored partner of a previously seen matching context.

        Raises:
            RuntimeError: If called before `load`.
            TypeError: If `context` is not a string.
        """
        if not self._loaded:
            msg = "Call load() before set_context()."
            raise RuntimeError(msg)
        if self._scheduler.active_sample_index is None:
            return False
        if not isinstance(context, str):
            msg = "context must be a string."
            raise TypeError(msg)

        matched_context = self._scheduler.set_context(context)
        self._apply_active_sample_values()
        return matched_context

    def tell(self, result: float | Sequence[float] | np.ndarray) -> Report:
        """Submit the objective result for the current sample.

        Scalar results are treated as one-element tuples. Sequence results are
        ranked lexicographically, with larger tuples considered better.

        Args:
            result: Objective value for the current sample.

        Returns:
            A `Report` describing batch completion, context matching, xNES status, and whether a restart happened.

        Raises:
            TypeError: If `result` is neither a scalar nor a numeric sequence.
            ValueError: If `result` is an empty sequence.
            RuntimeError: If called before `load`.
        """

        if self._scheduler.active_sample_index is None:
            msg = "No active sample available. Call load() first."
            raise RuntimeError(msg)

        completed_batch, matched_context = self._scheduler.record_result(_normalize_result(result))
        if completed_batch:
            results = self._scheduler.completed_results()
            ranking = sorted(range(len(results)), key=lambda idx: results[idx], reverse=True)
            status = self._xnes.tell(self._scheduler.batch_z, ranking)

            restarted = status is not XNESStatus.OK
            if restarted:
                self._restart_distribution()
            else:
                self._reset_batch()
            return Report(True, matched_context, status, restarted)

        self._apply_active_sample_values()
        return Report(False, matched_context, XNESStatus.OK, False)

    def set_best(self) -> None:
        """Set all registered parameter values to the current population mean.

        This mutates exposed `Parameter` views in place without changing
        optimizer state. It is intended for evaluation/inference after training.
        To continue training afterwards, restore a previously saved state and
        resume the normal `load -> optional set_context -> evaluate -> tell ->
        save` flow.
        """

        for row, name in enumerate(self._state_names):
            self._registry[name].value = float(self._xnes.mu[row])

    def get_info(self) -> list[ParameterInfo]:
        """Return immutable snapshots of all registered parameters.

        Returns:
            Snapshots in lexicographic name order with current sampled values, xNES means/scales, and priors.
        """

        scale_diag = np.diag(self._xnes.scale)
        return [
            ParameterInfo(
                name=name,
                value=float(self._registry[name].value),
                loc=float(self._xnes.mu[row]),
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

    def _reconcile_distribution_state(
        self,
        saved_names: list[str],
        current_names: list[str],
        loc: np.ndarray,
        scale: np.ndarray,
        step_size_path: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reconciled_loc, reconciled_scale = self._build_initial_state(current_names)
        reconciled_step_size_path = np.zeros(len(current_names), dtype=float)

        saved_index = {name: idx for idx, name in enumerate(saved_names)}
        shared_current_indices: list[int] = []
        shared_saved_indices: list[int] = []

        for current_idx, name in enumerate(current_names):
            saved_idx = saved_index.get(name)
            if saved_idx is None:
                continue
            shared_current_indices.append(current_idx)
            shared_saved_indices.append(saved_idx)
            reconciled_loc[current_idx] = float(loc[saved_idx])
            reconciled_step_size_path[current_idx] = float(step_size_path[saved_idx])

        if shared_current_indices:
            reconciled_scale[np.ix_(shared_current_indices, shared_current_indices)] = scale[
                np.ix_(shared_saved_indices, shared_saved_indices)
            ]

        return reconciled_loc, reconciled_scale, reconciled_step_size_path

    def _reconcile_batch_state(
        self,
        saved_names: list[str],
        current_names: list[str],
        batch_z: np.ndarray,
        batch_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_count = batch_x.shape[1]
        reconciled_batch_z = np.zeros((len(current_names), sample_count), dtype=float)
        reconciled_batch_x = np.tile(
            np.array([self._priors[name].loc for name in current_names], dtype=float).reshape(-1, 1),
            (1, sample_count),
        )

        saved_index = {name: idx for idx, name in enumerate(saved_names)}
        for current_idx, name in enumerate(current_names):
            saved_idx = saved_index.get(name)
            if saved_idx is None:
                continue
            reconciled_batch_z[current_idx, :] = batch_z[saved_idx, :]
            reconciled_batch_x[current_idx, :] = batch_x[saved_idx, :]

        return reconciled_batch_z, reconciled_batch_x

    def _new_xnes(self, loc: np.ndarray, scale: np.ndarray, step_size_path: np.ndarray) -> XNES:
        xnes = XNES(
            loc,
            scale,
            p_sigma=step_size_path,
        )
        if self.csa_enabled is not None:
            xnes.csa_enabled = self.csa_enabled
        if self.eta_mu is not None:
            xnes.eta_mu = self.eta_mu
        if self.eta_sigma is not None:
            xnes.eta_sigma = self.eta_sigma
        if self.eta_B is not None:
            xnes.eta_B = self.eta_B
        return xnes

    def _reset_batch(self) -> None:
        batch_z, batch_x = self._xnes.ask(self.pop_size, self._rng)
        self._scheduler.reset(batch_z, batch_x)
        self._apply_active_sample_values()

    def _apply_active_sample_values(self) -> None:
        sample_index = self._scheduler.active_sample_index
        if sample_index is None or self._scheduler.batch_x.shape[1] == 0:
            return
        for row, name in enumerate(self._state_names):
            self._registry[name].value = float(self._scheduler.batch_x[row, sample_index])

    def _restart_distribution(self) -> None:
        names = self._ordered_names()
        loc, scale = self._build_initial_state(names)
        self._xnes = self._new_xnes(loc, scale, np.zeros(len(names), dtype=float))
        self._state_names = names
        self._reset_batch()


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
