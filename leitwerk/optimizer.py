"""Schema-first optimizer wrapper built on top of the xNES update rule."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar, cast

import numpy as np

from .scheduler import BatchScheduler, Reservation
from .schema import Parameter, SchemaDiff, SchemaSpec, parse_schema
from .xnes import XNES, XNESStatus

T = TypeVar("T")
JSON: TypeAlias = Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
_ResultRecord = tuple[float, ...] | None
_SUCCESSFUL_TERMINATION_STATUSES = frozenset(
    {
        XNESStatus.SIGMA_MIN,
        XNESStatus.LOC_STEP_MIN,
        XNESStatus.SCALE_NORM_MIN,
    }
)


@dataclass(frozen=True)
class TellResult:
    """Outcome of one `Optimizer.tell` call.

    Attributes:
        completed_batch: Whether this result completed the current batch.
        matched_context: Whether sample selection used a mirrored context match.
        status: xNES status returned after the batch update.
        restarted: Whether the wrapper restarted with a fresh distribution after the update.
    """

    completed_batch: bool
    matched_context: bool
    status: XNESStatus
    restarted: bool


class Optimizer(Generic[T]):
    """Schema-first xNES optimizer over dataclass or mapping schemas.

    Dataclass schemas must be dataclass trees whose optimized leaves are
    declared as `Annotated[float, Parameter(...)]`.
    Mapping schemas must be nested mappings with string keys and
    `Parameter(...)` leaves.
    The `ask()` / `tell()` interface is strictly sequential and keeps the
    pending reservation inside the optimizer. `load()` and `save()` are only
    allowed at idle boundaries, i.e. when no `ask()` is pending. Optimizer
    state is keyed by stable dotted leaf names plus persisted parameter
    definitions. Field ordering is lexicographic by leaf name rather than
    declaration or insertion order.

    Results are ranked for maximization by default. Pass `minimize=True` to
    rank lower results as better instead. This optimization direction is
    runtime configuration and is not persisted by `save()`.
    """

    def __init__(
        self,
        schema_type: type[T] | Mapping[str, object],
        pop_size: int | None = None,
        minimize: bool = False,
        csa_enabled: bool = True,
        eta_mu: float = 1.0,
        eta_sigma: float = 1.0,
        eta_B: float = 1.0,
    ) -> None:
        self.pop_size = pop_size
        self.minimize = minimize
        self.csa_enabled = csa_enabled
        self.eta_mu = eta_mu
        self.eta_sigma = eta_sigma
        self.eta_B = eta_B

        self._schema: SchemaSpec[T] = cast(SchemaSpec[T], parse_schema(schema_type))
        self._rng = np.random.default_rng()
        self._scheduler = BatchScheduler()
        self._pending_reservation: Reservation | None = None
        self._xnes: XNES
        self._reset_distribution()

    def save(self) -> dict[str, object]:
        """Serialize the current optimizer state into a JSON-compatible mapping."""
        self._require_idle("save")
        return {
            "schema": self._schema.state_schema(),
            "loc": self._xnes.mu.tolist(),
            "scale": self._xnes.scale.tolist(),
            "step_size_path": self._xnes.p_sigma.tolist(),
            "batch": self._scheduler.batch.tolist(),
            "results": _serialize_results(self._scheduler.results),
            "context_pending": dict(self._scheduler.context_pending),
            "rng_state": dict(self._rng.bit_generator.state),
        }

    def load(self, state: object) -> SchemaDiff:
        """Restore optimizer state from a previous snapshot.

        Loading a previous snapshot reconciles added, removed, and changed
        schema leaves by persisted transform compatibility while preserving
        shared learned state.
        """
        self._require_idle("load")
        if state is None:
            msg = "state must be a saved optimizer snapshot, not None."
            raise TypeError(msg)

        state_obj = cast(Mapping[str, object], state)

        schema_json = cast(Mapping[str, object], state_obj["schema"])
        saved_schema = _deserialize_schema(schema_json)
        saved_names = sorted(saved_schema)
        loc = np.asarray(state_obj["loc"], dtype=float)
        scale = np.asarray(state_obj["scale"], dtype=float)
        step_size_path = np.asarray(state_obj["step_size_path"], dtype=float)
        batch = _as_batch_matrix(state_obj["batch"], len(saved_names))
        results = _deserialize_results(state_obj["results"])
        context_pending = dict(cast(Mapping[str, int], state_obj["context_pending"]))

        schema_diff = self._schema.diff(saved_schema)
        loc, scale, step_size_path = self._reconcile_distribution_state(
            saved_names,
            schema_diff.unchanged,
            loc,
            scale,
            step_size_path,
        )

        self._rng.bit_generator.state = dict(cast(Mapping[str, object], state_obj["rng_state"]))
        self._xnes = self._new_xnes(loc, scale, step_size_path)
        self._pending_reservation = None
        if batch.shape[1] == 0:
            self._sample_batch()
        else:
            batch = self._reconcile_batch_state(saved_names, schema_diff, batch, results)
            self._scheduler.restore(batch, results, context_pending)
        return schema_diff

    def _reset_distribution(self, loc: np.ndarray | None = None) -> None:
        reset_loc, scale, step_size_path = self._schema.initial_distribution()
        if loc is not None:
            reset_loc = np.array(loc, dtype=float, copy=True)
        self._xnes = self._new_xnes(reset_loc, scale, step_size_path)
        self._sample_batch()

    def ask(self, context: JSON = None) -> T:
        """Reserve one sampled parameter set for one evaluation.

        Context matching uses exact string equality. Non-string contexts are
        normalized into canonical JSON text before matching and persistence.
        """
        self._require_idle("ask")
        self._pending_reservation = self._reserve(_normalize_context(context))
        return self._params_for(self._pending_reservation)

    def ask_best(self) -> T:
        """Return the current mean parameters in the schema's runtime shape."""
        return self._schema.build_params(self._xnes.mu)

    def tell(self, result: float | Sequence[float] | np.ndarray) -> TellResult:
        """Submit the objective result for the pending sample."""
        reservation = self._require_pending()
        tell_result = self._tell_reservation(reservation, result)
        self._pending_reservation = None
        return tell_result

    def _tell_reservation(
        self,
        reservation: Reservation,
        result: float | Sequence[float] | np.ndarray,
    ) -> TellResult:
        completed_batch = self._scheduler.record_result(reservation, _normalize_result(result))
        if completed_batch:
            status, restarted = self._complete_batch()
            return TellResult(True, reservation.matched_context, status, restarted)
        return TellResult(False, reservation.matched_context, XNESStatus.OK, False)

    def _params_for(self, reservation: Reservation) -> T:
        latent_sample = self._xnes.transform(self._scheduler.batch[:, [reservation.sample_id]])[:, 0]
        return self._schema.build_params(latent_sample)

    def _reconcile_distribution_state(
        self,
        saved_names: list[str],
        unchanged_names: list[str],
        loc: np.ndarray,
        scale: np.ndarray,
        step_size_path: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reconciled_loc, reconciled_scale, reconciled_step_size_path = self._schema.initial_distribution()

        saved_index = {name: idx for idx, name in enumerate(saved_names)}
        current_index = self._schema.index_by_name()
        shared_indices = [(current_index[name], saved_index[name]) for name in unchanged_names]
        for current_idx, saved_idx in shared_indices:
            reconciled_loc[current_idx] = float(loc[saved_idx])
            reconciled_step_size_path[current_idx] = float(step_size_path[saved_idx])

        if shared_indices:
            shared_current_indices, shared_saved_indices = zip(*shared_indices, strict=True)
            reconciled_scale[np.ix_(shared_current_indices, shared_current_indices)] = scale[
                np.ix_(shared_saved_indices, shared_saved_indices)
            ]

        return reconciled_loc, reconciled_scale, reconciled_step_size_path

    def _reconcile_batch_state(
        self,
        saved_names: list[str],
        schema_diff: SchemaDiff,
        batch: np.ndarray,
        results: list[tuple[float, ...] | None],
    ) -> np.ndarray:
        sample_count = batch.shape[1]
        reconciled_batch = np.zeros((self._schema.dim, sample_count), dtype=float)
        completed_mask = np.zeros(sample_count, dtype=bool)
        for idx, result in enumerate(results[:sample_count]):
            completed_mask[idx] = result is not None
        pending_mask = ~completed_mask
        mirror_index = _mirror_indices(sample_count)
        # For changed dimensions, keep exactly those samples whose whole mirror
        # pair is still pending; zero all others. This keeps each pair either
        # both old or both zero, so changed coordinates remain exact mirrors.
        mirror_pending_mask = pending_mask & pending_mask[mirror_index]

        saved_index = {name: idx for idx, name in enumerate(saved_names)}
        current_index = self._schema.index_by_name()
        for name in schema_diff.unchanged:
            current_idx = current_index[name]
            saved_idx = saved_index[name]
            reconciled_batch[current_idx, :] = batch[saved_idx, :]
        for name in schema_diff.changed:
            current_idx = current_index[name]
            saved_idx = saved_index[name]
            reconciled_batch[current_idx, mirror_pending_mask] = batch[saved_idx, mirror_pending_mask]

        return reconciled_batch

    def _new_xnes(self, loc: np.ndarray, scale: np.ndarray, step_size_path: np.ndarray) -> XNES:
        xnes = XNES(
            loc,
            scale,
            p_sigma=step_size_path,
        )
        xnes.csa_enabled = self.csa_enabled
        xnes.eta_mu = self.eta_mu
        xnes.eta_sigma = self.eta_sigma
        xnes.eta_B = self.eta_B
        return xnes

    def _sample_batch(self) -> None:
        batch = self._xnes.ask(self.pop_size, self._rng)
        self._pending_reservation = None
        self._scheduler.reset(batch)

    def _complete_batch(self) -> tuple[XNESStatus, bool]:
        status = self._xnes.tell(self._scheduler.batch, self._ranking())
        restarted = status is not XNESStatus.OK
        if restarted:
            restart_loc = self._xnes.mu if status in _SUCCESSFUL_TERMINATION_STATUSES else None
            self._reset_distribution(restart_loc)
        else:
            self._sample_batch()
        return status, restarted

    def _reserve(self, context: str | None) -> Reservation:
        result = self._scheduler.reserve(context)
        if result is None:
            if not self._scheduler.is_complete():
                msg = "No sample available."
                raise RuntimeError(msg)
            self._complete_batch()
            result = self._scheduler.reserve(context)
            if result is None:
                msg = "No sample available."
                raise RuntimeError(msg)
        return result

    def _ranking(self) -> list[int]:
        assert self._scheduler.is_complete()
        results = [cast(tuple[float, ...], item) for item in self._scheduler.results]
        return sorted(range(len(results)), key=lambda idx: results[idx], reverse=not self.minimize)

    def _require_idle(self, action: str) -> None:
        if self._pending_reservation is not None:
            msg = f"Pending ask must be told before calling {action}()."
            raise RuntimeError(msg)

    def _require_pending(self) -> Reservation:
        if self._pending_reservation is None:
            msg = "No pending ask."
            raise RuntimeError(msg)
        return self._pending_reservation


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


def _normalize_context(context: JSON) -> str | None:
    if context is None or isinstance(context, str):
        return context
    try:
        return json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = "context must be a string or JSON-serializable value."
        raise TypeError(msg) from exc


def _deserialize_schema(schema_json: Mapping[str, object]) -> dict[str, Parameter]:
    return {str(name): Parameter.from_state(spec) for name, spec in schema_json.items()}


def _serialize_results(results: Sequence[_ResultRecord]) -> list[list[float] | None]:
    return [None if row is None else list(row) for row in results]


def _deserialize_results(result_rows: object) -> list[_ResultRecord]:
    rows = cast(Sequence[Sequence[float] | None], result_rows)
    return [None if row is None else tuple(float(value) for value in row) for row in rows]


def _as_batch_matrix(batch_json: object, dim: int) -> np.ndarray:
    batch = np.asarray(batch_json, dtype=float)
    if batch.ndim == 1 and batch.size == 0:
        return np.zeros((dim, 0), dtype=float)
    return batch


def _mirror_indices(sample_count: int) -> np.ndarray:
    mirror_index = np.arange(sample_count)
    half = sample_count // 2
    if half:
        mirror_index[:half] += half
        mirror_index[half:] -= half
    return mirror_index
