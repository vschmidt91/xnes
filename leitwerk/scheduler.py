from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class Reservation:
    sample_id: int
    context: str | None
    matched_context: bool


class BatchScheduler:
    """Batch state with optional context matching support."""

    def __init__(self) -> None:
        self.batch: np.ndarray = np.zeros((0, 0))
        self.results: list[tuple[float, ...] | None] = []
        self.context_pending: dict[str, int] = {}
        self._reserved_indices: set[int] = set()

    def reset(self, batch: np.ndarray) -> None:
        self.batch = batch
        self.results = [None] * self.batch.shape[1]
        self.context_pending = {}
        self._reserved_indices = set()

    def restore(
        self,
        batch: np.ndarray,
        results: list[tuple[float, ...] | None],
        context_pending: dict[str, int],
    ) -> None:
        self.batch = batch
        sample_count = self.batch.shape[1]
        self.results = [None] * sample_count
        for idx, item in enumerate(results[:sample_count]):
            self.results[idx] = item
        self.context_pending = {
            context: sample_idx
            for context, sample_idx in context_pending.items()
            if 0 <= sample_idx < sample_count
            and self.results[sample_idx] is not None
            and self.results[self._mirror_index(sample_idx)] is None
        }
        self._reserved_indices = set()

    def reserve(self, context: str | None) -> Reservation | None:
        sample_index, matched_context = self._pick_sample(context)
        if sample_index is not None:
            trial = Reservation(
                sample_id=sample_index,
                context=context,
                matched_context=matched_context,
            )
            self._reserved_indices.add(sample_index)
            return trial
        if self._reserved_indices or self.is_complete():
            return None
        msg = "Scheduler has no reservable sample and no completed batch."
        raise RuntimeError(msg)

    def record_result(self, reservation: Reservation, result: tuple[float, ...]) -> bool:
        sample_index = reservation.sample_id
        if not 0 <= sample_index < len(self.results):
            msg = "Sample index out of range."
            raise RuntimeError(msg)
        if sample_index not in self._reserved_indices:
            msg = "Unknown trial."
            raise RuntimeError(msg)
        if self.results[sample_index] is not None:
            msg = "Sample already has a recorded result."
            raise RuntimeError(msg)

        self._reserved_indices.remove(sample_index)
        self.results[sample_index] = result
        self._update_context_pending(sample_index, reservation.context)
        return self.is_complete()

    def has_reserved_trials(self) -> bool:
        return bool(self._reserved_indices)

    def is_complete(self) -> bool:
        return all(item is not None for item in self.results)

    def _pick_sample(self, context: str | None) -> tuple[int | None, bool]:
        if context is not None:
            pending_index = self.context_pending.get(context)
            if pending_index is not None:
                mirror_index = self._mirror_index(pending_index)
                if self.results[mirror_index] is None and mirror_index not in self._reserved_indices:
                    return mirror_index, True

        sample_index = next(
            (idx for idx, result in enumerate(self.results) if result is None and idx not in self._reserved_indices),
            None,
        )
        return sample_index, False

    def _mirror_index(self, sample_index: int) -> int:
        half = self.batch.shape[1] // 2
        return sample_index + half if sample_index < half else sample_index - half

    def _update_context_pending(self, sample_index: int, context: str | None) -> None:
        if context is None:
            return

        pending_index = self.context_pending.get(context)
        if pending_index is None:
            mirror_index = self._mirror_index(sample_index)
            if self.results[mirror_index] is None:
                self.context_pending[context] = sample_index
            return

        if self._mirror_index(pending_index) == sample_index:
            del self.context_pending[context]
