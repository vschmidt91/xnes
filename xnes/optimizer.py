"""Schema-first optimizer wrapper built on top of the xNES update rule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

import numpy as np

from .scheduler import BatchCompletion, BatchScheduler, BatchTrial
from .schema import Parameter, SchemaDiff, SchemaSpec, parse_schema
from .xnes import XNES, XNESStatus

T = TypeVar("T")


@dataclass(frozen=True)
class TellResult:
    """Outcome of one `Optimizer.tell` call.

    Attributes:
        completed_batch: Whether this result completed the current batch.
        matched_context: Whether sample selection used a mirrored context match.
        status: xNES status returned after the batch update.
        restarted: Whether the wrapper restarted from schema metadata after the update.
    """

    completed_batch: bool
    matched_context: bool
    status: XNESStatus
    restarted: bool


@dataclass(frozen=True)
class Trial:
    """Trial metadata returned by `ask()` and consumed by `tell()`."""

    sample_id: int
    context: str | None
    matched_context: bool
    _trial: BatchTrial | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class Optimizer(Generic[T]):
    """Maximizing optimizer over dataclass schemas.

    The schema must be a dataclass tree whose internal nodes are dataclasses
    and whose optimized leaves are declared as `Annotated[float, Parameter(...)]`.
    The wrapper exposes typed runtime values separately from trial handles
    while keeping optimizer state keyed by stable dotted leaf names
    plus persisted parameter definitions. Field ordering is lexicographic by
    leaf name rather than dataclass declaration order.
    """

    def __init__(self, schema_type: type[T]) -> None:
        self.pop_size: int | None = None
        self.csa_enabled: bool | None = None
        self.eta_mu: float | None = None
        self.eta_sigma: float | None = None
        self.eta_B: float | None = None

        self._schema: SchemaSpec[T] = parse_schema(schema_type)
        self._rng = np.random.default_rng()
        self._loaded = False

        loc, scale, step_size_path = self._schema.initial_distribution()
        self._xnes: XNES = self._new_xnes(loc, scale, step_size_path)
        self._scheduler = BatchScheduler()

    def save(self) -> dict[str, object]:
        """Serialize the current optimizer state into a JSON-compatible mapping."""
        return {
            "schema": self._schema.state_schema(),
            "loc": self._xnes.mu.tolist(),
            "scale": self._xnes.scale.tolist(),
            "step_size_path": self._xnes.p_sigma.tolist(),
            "batch": self._scheduler.batch.tolist(),
            "results": [None if item is None else list(item) for item in self._scheduler.results],
            "context_pending": dict(self._scheduler.context_pending),
            "rng_state": dict(self._rng.bit_generator.state),
        }

    def load(self, state: object) -> SchemaDiff:
        """Restore optimizer state or initialize a fresh run from schema metadata.

        Passing `None` starts a new run from the schema metadata and reports all
        current schema leaf names as added. Loading a previous snapshot
        reconciles added, removed, and changed schema leaves by persisted
        parameter definition while preserving shared learned state.
        """

        expected_names = list(self._schema.names)
        if state is None:
            self._reset_from_schema()
            self._loaded = True
            return SchemaDiff(added=expected_names, removed=[], changed=[], unchanged=[])

        state_obj = cast(Mapping[str, object], state)

        schema_json = cast(Mapping[str, object], state_obj["schema"])
        saved_schema = {str(name): Parameter.from_state(spec) for name, spec in schema_json.items()}
        saved_names = sorted(saved_schema)
        loc = np.asarray(state_obj["loc"], dtype=float)
        scale = np.asarray(state_obj["scale"], dtype=float)
        step_size_path_json = state_obj["step_size_path"]
        step_size_path = np.asarray(step_size_path_json, dtype=float)
        batch = np.asarray(state_obj["batch"], dtype=float)
        if batch.ndim == 1 and batch.size == 0:
            batch = np.zeros((len(saved_names), 0), dtype=float)
        result_rows = cast(Sequence[Sequence[float] | None], state_obj["results"])
        results = [None if row is None else tuple(float(value) for value in row) for row in result_rows]
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
        if batch.shape[1] == 0:
            self._sample_batch()
        else:
            batch = self._reconcile_batch_state(saved_names, schema_diff, batch, results)
            self._scheduler.restore(batch, results, context_pending)
        self._loaded = True
        return schema_diff

    def _reset_from_schema(self) -> None:
        loc, scale, step_size_path = self._schema.initial_distribution()
        self._xnes = self._new_xnes(loc, scale, step_size_path)
        self._sample_batch()

    def ask(self, context: str | None = None) -> tuple[Trial, T]:
        """Reserve one sampled parameter set for an evaluation run.

        Returns a pair `(trial, params)` where `params` is an instance of
        the root schema dataclass passed to `Optimizer`, including any nested
        dataclass subtrees. The `trial` must be passed back to `tell()`
        exactly once.
        """
        if not self._loaded:
            msg = "Call load() before ask()."
            raise RuntimeError(msg)

        batch_trial = self._reserve(context)
        latent_sample = self._xnes.transform(self._scheduler.batch[:, [batch_trial.sample_index]])[:, 0]
        return (
            Trial(
                sample_id=batch_trial.sample_index,
                context=batch_trial.context,
                matched_context=batch_trial.matched_context,
                _trial=batch_trial,
            ),
            self._schema.build_params(latent_sample),
        )

    def ask_best(self) -> T:
        """Return a deterministic, context-free snapshot of the current means."""
        if not self._loaded:
            msg = "Call load() before ask_best()."
            raise RuntimeError(msg)

        return self._schema.build_params(self._xnes.mu)

    def tell(self, trial: Trial, result: float | Sequence[float] | np.ndarray) -> TellResult:
        """Submit the objective result for one trial returned by `ask()`.

        Results use maximize semantics. Scalars are treated as one-element
        tuples, and sequence results are ranked lexicographically.
        """
        batch_trial = trial._trial
        if batch_trial is None:
            msg = "Unknown trial."
            raise RuntimeError(msg)

        completion = self._scheduler.record_result(batch_trial, _normalize_result(result))
        if completion is not None:
            status, restarted = self._complete_batch(completion)
            return TellResult(True, batch_trial.matched_context, status, restarted)
        return TellResult(False, batch_trial.matched_context, XNESStatus.OK, False)

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
        shared_current_indices: list[int] = [current_index[name] for name in unchanged_names]
        shared_saved_indices: list[int] = []

        for name in unchanged_names:
            current_idx = current_index[name]
            saved_idx = saved_index[name]
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
        half = sample_count // 2
        mirror_index = np.arange(sample_count)
        if half:
            mirror_index[:half] += half
            mirror_index[half:] -= half
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
        if self.csa_enabled is not None:
            xnes.csa_enabled = self.csa_enabled
        if self.eta_mu is not None:
            xnes.eta_mu = self.eta_mu
        if self.eta_sigma is not None:
            xnes.eta_sigma = self.eta_sigma
        if self.eta_B is not None:
            xnes.eta_B = self.eta_B
        return xnes

    def _sample_batch(self) -> None:
        batch = self._xnes.ask(self.pop_size, self._rng)
        self._scheduler.reset(batch)

    def _restart_distribution(self) -> None:
        loc, scale, step_size_path = self._schema.initial_distribution()
        self._xnes = self._new_xnes(loc, scale, step_size_path)
        self._sample_batch()

    def _complete_batch(self, completion: BatchCompletion) -> tuple[XNESStatus, bool]:
        status = self._xnes.tell(self._scheduler.batch, completion.ranking)
        restarted = status is not XNESStatus.OK
        if restarted:
            self._restart_distribution()
        else:
            self._sample_batch()
        return status, restarted

    def _reserve(self, context: str | None) -> BatchTrial:
        result = self._scheduler.reserve(context)
        if result is None:
            msg = "No unclaimed sample available. Pending trials must be told first."
            raise RuntimeError(msg)
        if isinstance(result, BatchCompletion):
            self._complete_batch(result)
            result = self._scheduler.reserve(context)
        if not isinstance(result, BatchTrial):
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
