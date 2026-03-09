from __future__ import annotations

import hashlib
import json
from typing import TypeAlias

import numpy as np

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | tuple["JSONValue", ...] | dict[str, "JSONValue"]


def _stable_hash(obj: JSONValue) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(data.encode(), digest_size=16).hexdigest()


class BatchScheduler:
    """Sequential interface over a mirrored batch with optional context matching."""

    def __init__(self) -> None:
        self.batch_z: np.ndarray = np.zeros((0, 0))
        self.batch_x: np.ndarray = np.zeros((0, 0))
        self.results: list[tuple[float, ...] | None] = []
        self.active_sample_index: int | None = None
        self.active_context_hash: str | None = None
        self.active_context_matched = False
        self.context_waiting: dict[str, int] = {}

    def reset(self, batch_z: np.ndarray, batch_x: np.ndarray) -> None:
        self.batch_z = batch_z
        self.batch_x = batch_x
        self.results = [None] * self.batch_x.shape[1]
        self.context_waiting = {}
        self._clear_active()
        self._activate_next_sample()

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
            context_hash: sample_idx
            for context_hash, sample_idx in context_waiting.items()
            if 0 <= sample_idx < sample_count
            and self.results[sample_idx] is not None
            and self.results[self._mirror_index(sample_idx)] is None
        }
        self._clear_active()
        self._activate_next_sample()

    def set_context(self, context: JSONValue) -> bool:
        if self.active_sample_index is None:
            return False

        context_hash = _stable_hash(context)
        sample_index, matched_context = self._select_sample_index(context_hash)
        self._set_active_sample(sample_index, context_hash, matched_context)
        return matched_context

    def record_result(self, result: tuple[float, ...]) -> tuple[bool, bool]:
        if self.active_sample_index is None:
            msg = "No active sample available."
            raise RuntimeError(msg)

        sample_index = self.active_sample_index
        matched_context = self.active_context_matched
        self.results[sample_index] = result
        self._register_context_match(sample_index, self.active_context_hash)
        self._clear_active()

        completed_batch = all(item is not None for item in self.results)
        if not completed_batch:
            self._activate_next_sample()
        return completed_batch, matched_context

    def completed_results(self) -> list[tuple[float, ...]]:
        return [item for item in self.results if item is not None]

    def _mirror_index(self, sample_index: int) -> int:
        half = self.batch_x.shape[1] // 2
        return sample_index + half if sample_index < half else sample_index - half

    def _activate_next_sample(self) -> None:
        sample_index = next((idx for idx, result in enumerate(self.results) if result is None), None)
        if sample_index is None:
            self._clear_active()
            return
        self._set_active_sample(sample_index)

    def _set_active_sample(
        self,
        sample_index: int,
        context_hash: str | None = None,
        matched_context: bool = False,
    ) -> None:
        self.active_sample_index = sample_index
        self.active_context_hash = context_hash
        self.active_context_matched = matched_context

    def _clear_active(self) -> None:
        self.active_sample_index = None
        self.active_context_hash = None
        self.active_context_matched = False

    def _select_sample_index(self, context_hash: str) -> tuple[int, bool]:
        current_index = self.active_sample_index
        current_available = current_index is not None and self.results[current_index] is None

        waiting_index = self.context_waiting.get(context_hash)
        if waiting_index is not None:
            mirror_index = self._mirror_index(waiting_index)
            if self.results[mirror_index] is None:
                return mirror_index, True
            del self.context_waiting[context_hash]

        sample_index = (
            current_index
            if current_available and current_index is not None
            else next((idx for idx, result in enumerate(self.results) if result is None), None)
        )
        if sample_index is None:
            msg = "Current batch is already fully assigned."
            raise RuntimeError(msg)
        return sample_index, False

    def _register_context_match(self, sample_index: int, context_hash: str | None) -> None:
        if context_hash is None:
            return

        waiting_index = self.context_waiting.get(context_hash)
        if waiting_index is None:
            mirror_index = self._mirror_index(sample_index)
            if self.results[mirror_index] is None:
                self.context_waiting[context_hash] = sample_index
            return

        if self._mirror_index(waiting_index) == sample_index:
            del self.context_waiting[context_hash]
