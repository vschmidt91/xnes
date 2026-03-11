"""Schema-first optimizer wrapper built on top of the xNES update rule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

import numpy as np

from ._scheduler import BatchCompletion, BatchReservation, BatchScheduler
from .schema import Prior, SchemaSpec, parse_schema
from .xnes import XNES, XNESStatus

T = TypeVar("T")


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
        parameters_added: Current schema leaf names not present in the loaded state.
        parameters_removed: Loaded schema leaf names not present in the current schema.
    """

    parameters_added: list[str]
    parameters_removed: list[str]


@dataclass(frozen=True)
class Sample(Generic[T]):
    """Typed parameter values and reservation metadata returned by `ask` or `ask_best`.

    `params` is an instance of the root schema dataclass passed to `Optimizer`,
    including any nested dataclass subtrees. Samples returned by `ask_best()`
    have `sample_id=None` and cannot be passed to `tell()`.
    """

    sample_id: int | None
    params: T
    context: str | None
    matched_context: bool
    _reservation: BatchReservation | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class Optimizer(Generic[T]):
    """Maximizing optimizer over dataclass schemas.

    The schema must be a dataclass tree whose internal nodes are dataclasses
    and whose optimized leaves are declared as `Annotated[float, Prior(...)]`.
    The wrapper exposes typed runtime values via `Sample.params` while keeping
    optimizer state keyed by stable dotted leaf names. Field ordering is
    lexicographic by leaf name rather than dataclass declaration order.
    """

    def __init__(self, schema_type: type[T]) -> None:
        self.pop_size: int | None = None
        self.csa_enabled: bool | None = None
        self.eta_mu: float | None = None
        self.eta_sigma: float | None = None
        self.eta_B: float | None = None

        self._schema: SchemaSpec[T] = parse_schema(schema_type)
        self._rng = np.random.default_rng()
        self._priors: dict[str, Prior] = {field_spec.name: field_spec.prior for field_spec in self._schema.fields}
        self._loaded = False

        self._xnes: XNES = self._new_xnes(np.zeros(0), np.eye(0), np.zeros(0))
        self._state_names: list[str] = []
        self._scheduler = BatchScheduler()

    def save(self) -> dict[str, object]:
        """Serialize the current optimizer state into a JSON-compatible mapping."""
        return {
            "names": list(self._state_names),
            "loc": self._xnes.mu.tolist(),
            "scale": self._xnes.scale.tolist(),
            "step_size_path": self._xnes.p_sigma.tolist(),
            "batch_z": self._scheduler.batch_z.tolist(),
            "results": [None if item is None else list(item) for item in self._scheduler.results],
            "context_waiting": dict(self._scheduler.context_waiting),
            "rng_state": dict(self._rng.bit_generator.state),
        }

    def load(self, state: object) -> LoadResult:
        """Restore optimizer state or initialize a fresh run from schema priors.

        Passing `None` starts a new run from the schema priors and reports all
        current schema leaf names as added. Loading a previous snapshot
        reconciles added and removed schema leaves by name while preserving
        shared learned state and any unfinished batch results.
        """

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
        batch_z = self._reconcile_batch_state(names, expected_names, batch_z)

        self._rng.bit_generator.state = dict(cast(Mapping[str, object], state_obj["rng_state"]))
        self._xnes = self._new_xnes(loc, scale, step_size_path)
        self._state_names = expected_names
        self._scheduler.restore(batch_z, results, context_waiting)
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

    def ask(self, context: str | None = None) -> Sample[T]:
        """Reserve one sampled parameter set for an evaluation run.

        The returned `Sample` contains the typed root schema instance on
        `sample.params`, including any nested dataclass values, plus
        reservation metadata required by `tell()`.
        """
        if not self._loaded:
            msg = "Call load() before ask()."
            raise RuntimeError(msg)

        reservation = self._reserve(context)
        sample = self._xnes.transform(self._scheduler.batch_z[:, [reservation.sample_index]])[:, 0]
        return Sample(
            sample_id=reservation.sample_index,
            params=self._build_params(sample),
            context=reservation.context,
            matched_context=reservation.matched_context,
            _reservation=reservation,
        )

    def ask_best(self) -> Sample[T]:
        """Return a deterministic, context-free snapshot of the current means.

        This does not reserve a sample and the returned `Sample` is not
        tellable.
        """
        if not self._loaded:
            msg = "Call load() before ask_best()."
            raise RuntimeError(msg)

        return Sample(
            sample_id=None,
            params=self._build_params(self._xnes.mu),
            context=None,
            matched_context=False,
        )

    def tell(self, sample: Sample[T], result: float | Sequence[float] | np.ndarray) -> TellResult:
        """Submit the objective result for sampled parameters returned by `ask`.

        Results use maximize semantics. Scalars are treated as one-element
        tuples, and sequence results are ranked lexicographically.
        """
        if sample.sample_id is None:
            msg = "Samples from ask_best() were not sampled and cannot be told."
            raise RuntimeError(msg)
        reservation = sample._reservation
        if reservation is None:
            msg = "Unknown sample reservation."
            raise RuntimeError(msg)

        completion = self._scheduler.record_result(reservation, _normalize_result(result))
        if completion is not None:
            status, restarted = self._complete_batch(completion)
            return TellResult(True, reservation.matched_context, status, restarted)
        return TellResult(False, reservation.matched_context, XNESStatus.OK, False)

    def _ordered_names(self) -> list[str]:
        return [field_spec.name for field_spec in self._schema.fields]

    def _build_initial_state(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        loc = np.array([self._priors[name].mean for name in names], dtype=float)
        scale_diag = np.array([self._priors[name].sigma for name in names], dtype=float)
        return loc, np.diag(scale_diag)

    def _build_params(self, values: np.ndarray) -> T:
        leaf_values = {
            field_spec.path: float(value) for field_spec, value in zip(self._schema.fields, values, strict=True)
        }
        return self._schema.instantiate(leaf_values)

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
    ) -> np.ndarray:
        sample_count = batch_z.shape[1]
        reconciled_batch_z = np.zeros((len(current_names), sample_count), dtype=float)

        saved_index = {name: idx for idx, name in enumerate(saved_names)}
        for current_idx, name in enumerate(current_names):
            saved_idx = saved_index.get(name)
            if saved_idx is None:
                continue
            reconciled_batch_z[current_idx, :] = batch_z[saved_idx, :]

        return reconciled_batch_z

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
        batch_z = self._xnes.ask(self.pop_size, self._rng)
        self._scheduler.reset(batch_z)

    def _restart_distribution(self) -> None:
        names = self._ordered_names()
        loc, scale = self._build_initial_state(names)
        self._xnes = self._new_xnes(loc, scale, np.zeros(len(names), dtype=float))
        self._state_names = names
        self._reset_batch()

    def _complete_batch(self, completion: BatchCompletion) -> tuple[XNESStatus, bool]:
        status = self._xnes.tell(self._scheduler.batch_z, completion.ranking)
        restarted = status is not XNESStatus.OK
        if restarted:
            self._restart_distribution()
        else:
            self._reset_batch()
        return status, restarted

    def _reserve(self, context: str | None) -> BatchReservation:
        result = self._scheduler.reserve(context)
        if result is None:
            msg = "No unclaimed sample available. Pending trials must be told first."
            raise RuntimeError(msg)
        if isinstance(result, BatchCompletion):
            self._complete_batch(result)
            result = self._scheduler.reserve(context)
        if not isinstance(result, BatchReservation):
            msg = "No sample available."
            raise RuntimeError(msg)
        return result


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
