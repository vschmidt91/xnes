"""Named-parameter optimizer built on top of the xNES update rule.

The wrapper is designed for a strict checkpointed workflow:

1. create `Optimizer`
2. register parameters with `add`
3. call `load` with `None` for a fresh run or a previously saved state
4. call `ask` (optionally with `context=...`)
5. read sampled values from `Parameters` (e.g. `params["x"]`)
6. call `tell(params, result)` exactly once for that reservation
7. call `save`

The registered parameter set is fixed once `load` has been called.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np

from ._scheduler import BatchScheduler
from .xnes import XNES, XNESStatus


@dataclass(frozen=True)
class _Prior:
    loc: float
    scale: float


@dataclass(frozen=True)
class TellResult:
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
        parameters_removed: Loaded parameter names not present in the current parameter set.
    """

    parameters_added: list[str]
    parameters_removed: list[str]


@dataclass(frozen=True)
class Parameters(Mapping[str, float]):
    """Reserved sample values and reservation metadata returned by `ask`.

    This is a mapping over sampled parameter values, so values can be read
    directly via `params["name"]`.
    """

    id: int
    sample_index: int
    params: Mapping[str, float]
    context: str | None
    matched_context: bool

    def __getitem__(self, key: str) -> float:
        return self.params[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)


@dataclass(frozen=True)
class _Claim:
    id: int
    sample_index: int
    context: str | None
    matched_context: bool


class Optimizer:
    """Maximizing optimizer with named scalar parameters.

    Parameters are registered by name and sampled in lexicographic order so the
    optimization state is independent of registration order.

    Intended call flow:

    1. create the optimizer
    2. register all parameters with `add`
    3. call `load`
    4. call `ask`
    5. evaluate sampled parameter values
    6. call `tell` with that `Parameters` instance
    7. call `save`

    After `load`, the registered parameter set is fixed and `add` is no longer
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
        self._priors: dict[str, _Prior] = {}
        self._loaded = False

        self._xnes: XNES = self._new_xnes(np.zeros(0), np.eye(0), np.zeros(0))
        self._state_names: list[str] = []
        self._scheduler = BatchScheduler()
        self._claims_by_id: dict[int, _Claim] = {}
        self._claimed_sample_indices: set[int] = set()
        self._next_claim_id = 0

    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> None:
        """Register a parameter name.

        This is a setup-time operation. It must happen before `load`.

        Args:
            name: Unique parameter name.
            loc: Initial mean used for new parameters.
            scale: Initial standard deviation used for new parameters.

        Raises:
            ValueError: If `scale <= 0` or `name` is already registered.
            RuntimeError: If called after `load`.
        """

        if scale <= 0:
            msg = "scale must be > 0."
            raise ValueError(msg)

        if self._loaded:
            msg = "Cannot add parameters after load()."
            raise RuntimeError(msg)
        if name in self._priors:
            msg = f"Parameter '{name}' is already registered."
            raise ValueError(msg)

        self._priors[name] = _Prior(loc=float(loc), scale=float(scale))

    def save(self) -> dict[str, object]:
        """Serialize the current optimizer state."""
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

        if not self._priors:
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
        self._clear_claims()
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

    def ask(self, context: str | None = None) -> Parameters:
        """Reserve a sample for one evaluation run."""
        if not self._loaded:
            msg = "Call load() before ask()."
            raise RuntimeError(msg)

        sample_index, matched_context = self._scheduler.pick_sample(context, self._claimed_sample_indices)
        if sample_index is None:
            if self._claimed_sample_indices:
                msg = "No unclaimed sample available. Pending trials must be told first."
                raise RuntimeError(msg)
            if all(item is not None for item in self._scheduler.results):
                self._complete_batch()
                sample_index, matched_context = self._scheduler.pick_sample(context, self._claimed_sample_indices)
            if sample_index is None:
                msg = "No sample available."
                raise RuntimeError(msg)

        claim_id = self._next_claim_id
        self._next_claim_id += 1
        self._claimed_sample_indices.add(sample_index)
        claim = _Claim(claim_id, sample_index, context, matched_context)
        self._claims_by_id[claim_id] = claim

        params = {name: float(self._scheduler.batch_x[row, sample_index]) for row, name in enumerate(self._state_names)}
        return Parameters(
            id=claim_id,
            sample_index=sample_index,
            params=MappingProxyType(params),
            context=context,
            matched_context=matched_context,
        )

    def tell(self, params: Parameters, result: float | Sequence[float] | np.ndarray) -> TellResult:
        """Submit the objective result for reserved parameters."""
        claim = self._claims_by_id.pop(params.id, None)
        if claim is None:
            msg = "Unknown parameters reservation."
            raise RuntimeError(msg)
        self._claimed_sample_indices.discard(claim.sample_index)

        completed_batch = self._scheduler.record_result(claim.sample_index, claim.context, _normalize_result(result))
        if completed_batch:
            status, restarted = self._complete_batch()
            return TellResult(True, claim.matched_context, status, restarted)
        return TellResult(False, claim.matched_context, XNESStatus.OK, False)

    def _ordered_names(self) -> list[str]:
        return sorted(self._priors)

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
        self._clear_claims()

    def _restart_distribution(self) -> None:
        names = self._ordered_names()
        loc, scale = self._build_initial_state(names)
        self._xnes = self._new_xnes(loc, scale, np.zeros(len(names), dtype=float))
        self._state_names = names
        self._reset_batch()

    def _complete_batch(self) -> tuple[XNESStatus, bool]:
        results = self._scheduler.completed_results()
        ranking = sorted(range(len(results)), key=lambda idx: results[idx], reverse=True)
        status = self._xnes.tell(self._scheduler.batch_z, ranking)
        restarted = status is not XNESStatus.OK
        if restarted:
            self._restart_distribution()
        else:
            self._reset_batch()
        return status, restarted

    def _clear_claims(self) -> None:
        self._claims_by_id.clear()
        self._claimed_sample_indices.clear()


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
