from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass(frozen=True)
class BatchReservation:
    sample_index: int
    context: str | None
    matched_context: bool


@dataclass(frozen=True)
class BatchCompletion:
    ranking: list[int]


class BatchScheduler:
    """Batch state with optional context matching support."""

    def __init__(self) -> None:
        self.batch_z: np.ndarray = np.zeros((0, 0))
        self.results: list[tuple[float, ...] | None] = []
        self.context_waiting: dict[str, int] = {}
        self._reserved_indices: set[int] = set()

    def reset(self, batch_z: np.ndarray) -> None:
        self.batch_z = batch_z
        self.results = [None] * self.batch_z.shape[1]
        self.context_waiting = {}
        self._reserved_indices = set()

    def restore(
        self,
        batch_z: np.ndarray,
        results: list[tuple[float, ...] | None],
        context_waiting: dict[str, int],
    ) -> None:
        self.batch_z = batch_z
        sample_count = self.batch_z.shape[1]
        self.results = [None] * sample_count
        for idx, item in enumerate(results[:sample_count]):
            self.results[idx] = item
        self.context_waiting = {
            context: sample_idx
            for context, sample_idx in context_waiting.items()
            if 0 <= sample_idx < sample_count
            and self.results[sample_idx] is not None
            and self.results[self._mirror_index(sample_idx)] is None
        }
        self._reserved_indices = set()

    def reserve(self, context: str | None) -> BatchReservation | BatchCompletion | None:
        sample_index, matched_context = self._pick_sample(context)
        if sample_index is not None:
            reservation = BatchReservation(
                sample_index=sample_index,
                context=context,
                matched_context=matched_context,
            )
            self._reserved_indices.add(sample_index)
            return reservation
        if self._reserved_indices:
            return None
        if self.is_complete():
            return BatchCompletion(self._ranking())
        msg = "Scheduler has no reservable sample and no completed batch."
        raise RuntimeError(msg)

    def record_result(self, reservation: BatchReservation, result: tuple[float, ...]) -> BatchCompletion | None:
        sample_index = reservation.sample_index
        if not 0 <= sample_index < len(self.results):
            msg = "Sample index out of range."
            raise RuntimeError(msg)
        if sample_index not in self._reserved_indices:
            msg = "Unknown sample reservation."
            raise RuntimeError(msg)
        if self.results[sample_index] is not None:
            msg = "Sample already has a recorded result."
            raise RuntimeError(msg)

        self._reserved_indices.remove(sample_index)
        self.results[sample_index] = result
        self._register_context_match(sample_index, reservation.context)
        if not self.is_complete():
            return None
        return BatchCompletion(self._ranking())

    def is_complete(self) -> bool:
        return all(item is not None for item in self.results)

    def _ranking(self) -> list[int]:
        assert self.is_complete()
        results = [cast(tuple[float, ...], item) for item in self.results]
        return sorted(range(len(results)), key=lambda idx: results[idx], reverse=True)

    def _pick_sample(self, context: str | None) -> tuple[int | None, bool]:
        if context is not None:
            waiting_index = self.context_waiting.get(context)
            if waiting_index is not None:
                mirror_index = self._mirror_index(waiting_index)
                if self.results[mirror_index] is None and mirror_index not in self._reserved_indices:
                    return mirror_index, True

        sample_index = next(
            (idx for idx, result in enumerate(self.results) if result is None and idx not in self._reserved_indices),
            None,
        )
        return sample_index, False

    def _mirror_index(self, sample_index: int) -> int:
        half = self.batch_z.shape[1] // 2
        return sample_index + half if sample_index < half else sample_index - half

    def _register_context_match(self, sample_index: int, context: str | None) -> None:
        if context is None:
            return

        waiting_index = self.context_waiting.get(context)
        if waiting_index is None:
            mirror_index = self._mirror_index(sample_index)
            if self.results[mirror_index] is None:
                self.context_waiting[context] = sample_index
            return

        if self._mirror_index(waiting_index) == sample_index:
            del self.context_waiting[context]
