from __future__ import annotations

import numpy as np


class BatchScheduler:
    """Batch state with optional context matching support."""

    def __init__(self) -> None:
        self.batch_z: np.ndarray = np.zeros((0, 0))
        self.batch_x: np.ndarray = np.zeros((0, 0))
        self.results: list[tuple[float, ...] | None] = []
        self.context_waiting: dict[str, int] = {}

    def reset(self, batch_z: np.ndarray, batch_x: np.ndarray) -> None:
        self.batch_z = batch_z
        self.batch_x = batch_x
        self.results = [None] * self.batch_x.shape[1]
        self.context_waiting = {}

    def restore(
        self,
        batch_z: np.ndarray,
        batch_x: np.ndarray,
        results: list[tuple[float, ...] | None],
        context_waiting: dict[str, int],
    ) -> None:
        self.batch_z = batch_z
        self.batch_x = batch_x
        sample_count = self.batch_x.shape[1]
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

    def pick_sample(self, context: str | None, claimed_indices: set[int]) -> tuple[int | None, bool]:
        if context is not None:
            waiting_index = self.context_waiting.get(context)
            if waiting_index is not None:
                mirror_index = self._mirror_index(waiting_index)
                if self.results[mirror_index] is None and mirror_index not in claimed_indices:
                    return mirror_index, True

        sample_index = next(
            (idx for idx, result in enumerate(self.results) if result is None and idx not in claimed_indices),
            None,
        )
        return sample_index, False

    def record_result(self, sample_index: int, context: str | None, result: tuple[float, ...]) -> bool:
        if not 0 <= sample_index < len(self.results):
            msg = "Sample index out of range."
            raise RuntimeError(msg)
        if self.results[sample_index] is not None:
            msg = "Sample already has a recorded result."
            raise RuntimeError(msg)

        self.results[sample_index] = result
        self._register_context_match(sample_index, context)
        return all(item is not None for item in self.results)

    def completed_results(self) -> list[tuple[float, ...]]:
        return [item for item in self.results if item is not None]

    def _mirror_index(self, sample_index: int) -> int:
        half = self.batch_x.shape[1] // 2
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
