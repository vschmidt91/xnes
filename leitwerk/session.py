"""Filesystem-backed optimizer session wrapper."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np

from .optimizer import JSONLike, JSONObject, Optimizer, OptimizerReport, OptimizerSettings
from .schema import SchemaDiff

T = TypeVar("T")


class OptimizerSession(Generic[T]):
    """Persisted optimizer workflow around an in-memory `Optimizer`."""

    def __init__(self, path: Path, optimizer: Optimizer[T], schema_diff: SchemaDiff | None = None) -> None:
        self._path = path
        self._optimizer = optimizer
        self.schema_diff = schema_diff
        self._dirty = False

    @classmethod
    def open(
        cls,
        path: str | Path,
        schema_type: type[T] | Mapping[str, object],
        *,
        settings: OptimizerSettings | None = None,
    ) -> OptimizerSession[T]:
        """Create a persisted session and eagerly restore if the checkpoint exists."""
        session_path = Path(path)
        optimizer = Optimizer(schema_type, settings=settings)
        schema_diff = None
        if session_path.exists():
            state = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                msg = "Persisted optimizer state must be a JSON object."
                raise TypeError(msg)
            schema_diff = optimizer.load(cast(JSONObject, state))
        return cls(session_path, optimizer, schema_diff)

    @property
    def restored(self) -> bool:
        """Whether this session loaded an existing checkpoint."""
        return self.schema_diff is not None

    @property
    def settings(self) -> OptimizerSettings:
        """Effective runtime optimizer configuration."""
        return self._optimizer.settings

    @property
    def mean(self) -> T:
        """Current optimizer mean parameters."""
        return self._optimizer.mean

    def ask(self, context: JSONLike = None) -> T:
        """Reserve one sampled parameter set for evaluation."""
        self._require_clean()
        return self._optimizer.ask(context)

    def tell(self, result: float | Sequence[float] | np.ndarray) -> OptimizerReport:
        """Record one result and atomically persist the updated optimizer state."""
        tell_result = self._optimizer.tell(result)
        self._dirty = True
        self.flush()
        return tell_result

    def flush(self) -> None:
        """Persist the current committed optimizer state."""
        self._dirty = True
        _write_json_atomically(self._path, self._optimizer.save())
        self._dirty = False

    def _require_clean(self) -> None:
        if self._dirty:
            msg = "Session has unflushed state. Call flush() before ask()."
            raise RuntimeError(msg)


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
