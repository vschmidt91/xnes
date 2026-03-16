"""Filesystem-backed optimizer session wrapper."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np

from . import SchemaDiff
from .optimizer import JSONLike, JSONObject, Optimizer, OptimizerReport, OptimizerSettings

T = TypeVar("T")


class OptimizerSession(Generic[T]):
    """Persisted optimizer workflow around an in-memory `Optimizer`."""

    def __init__(
        self,
        path: str | Path,
        schema_type: type[T] | Mapping[str, object],
        settings: OptimizerSettings | None = None,
    ) -> None:

        session_path = Path(path)
        optimizer = Optimizer(schema_type, settings=settings)

        schema_diff = None
        if session_path.exists():
            state = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                msg = "Persisted optimizer state must be a JSON object."
                raise TypeError(msg)
            schema_diff = optimizer.load(cast(JSONObject, state))

        self._path = session_path
        self._optimizer = optimizer
        self._schema_diff = schema_diff

    @property
    def restored(self) -> bool:
        """Whether this session loaded an existing checkpoint."""
        return self._schema_diff is not None

    @property
    def settings(self) -> OptimizerSettings:
        """Effective runtime optimizer configuration."""
        return self._optimizer.settings

    @property
    def schema_diff(self) -> SchemaDiff | None:
        """Reconciled difference between persisted and current schema definitions."""
        return self._schema_diff

    @property
    def mean(self) -> T:
        """Current optimizer mean parameters."""
        return self._optimizer.mean

    def ask(self, context: JSONLike = None) -> T:
        """Reserve one sampled parameter set for evaluation."""
        return self._optimizer.ask(context)

    def tell(self, result: float | Sequence[float] | np.ndarray) -> OptimizerReport:
        """Record one result and atomically persist the updated optimizer state."""
        report = self._optimizer.tell(result)
        self.flush()
        return report

    def flush(self) -> None:
        """Persist the current committed optimizer state."""
        _write_json_atomically(self._path, self._optimizer.save())


def _write_json_atomically(path: Path, payload: JSONObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
