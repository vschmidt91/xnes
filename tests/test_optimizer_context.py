from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest
from leitwerk import OptimizerReport, XNESStatus

from ._optimizer_helpers import (
    _initialized_optimizer,
    _make_identity_schema,
    _read_pending_context_matches,
)


def test_context_reuses_mirror_on_repeat() -> None:
    schema = _make_identity_schema("ContextParams", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)
    context = "arena:zerg"

    first_params = optimizer.ask(context=context)
    first = np.array([first_params.x, first_params.y], dtype=float)
    assert optimizer.tell(1.0) == OptimizerReport(False, False, XNESStatus.OK, False)

    optimizer.ask()
    assert optimizer.tell(0.0) == OptimizerReport(False, False, XNESStatus.OK, False)

    mirror_params = optimizer.ask(context=context)
    mirror = np.array([mirror_params.x, mirror_params.y], dtype=float)
    assert np.allclose(mirror, -first)
    assert optimizer.tell(-1.0) == OptimizerReport(False, True, XNESStatus.OK, False)


def test_json_context_reuses_mirror_on_canonical_repeat() -> None:
    schema = _make_identity_schema("JsonContextParams", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    first_params = optimizer.ask(context={"b": 2, "a": 1})
    first = np.array([first_params.x, first_params.y], dtype=float)
    assert optimizer.tell(1.0) == OptimizerReport(False, False, XNESStatus.OK, False)

    optimizer.ask()
    assert optimizer.tell(0.0) == OptimizerReport(False, False, XNESStatus.OK, False)

    mirror_params = optimizer.ask(context={"a": 1, "b": 2})
    mirror = np.array([mirror_params.x, mirror_params.y], dtype=float)
    assert np.allclose(mirror, -first)
    assert optimizer.tell(-1.0) == OptimizerReport(False, True, XNESStatus.OK, False)


def test_json_context_is_persisted_as_a_json_string_key() -> None:
    schema = _make_identity_schema("SavedJsonContext", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    optimizer.ask(context={"b": 2, "a": 1})
    optimizer.tell(1.0)

    saved_matches = _read_pending_context_matches(optimizer.save())
    assert list(saved_matches.values()) == [0]
    [saved_context] = saved_matches
    assert json.loads(saved_context) == {"a": 1, "b": 2}


def test_pending_context_matches_is_cleared_when_mirror_is_taken_without_match() -> None:
    schema = _make_identity_schema("StaleContextPending", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    optimizer.ask(context="arena:zerg")
    optimizer.tell(1.0)

    optimizer.ask()
    optimizer.tell(0.0)

    optimizer.ask(context="arena:protoss")
    optimizer.tell(-1.0)

    assert _read_pending_context_matches(optimizer.save()) == {}


def test_ask_rejects_non_json_context() -> None:
    class Opaque:
        pass

    schema = _make_identity_schema("OpaqueContext", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    with pytest.raises(TypeError, match="JSON-serializable"):
        optimizer.ask(context=cast(Any, Opaque()))

    with pytest.raises(RuntimeError, match="No pending ask"):
        optimizer.tell(0.0)


def test_serial_ask_raises_when_a_sample_is_pending() -> None:
    schema = _make_identity_schema("PendingSerialAsk", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    optimizer.ask()

    with pytest.raises(RuntimeError, match="Pending"):
        optimizer.ask()


def test_tell_raises_without_pending_ask() -> None:
    schema = _make_identity_schema("NoPendingTell", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    with pytest.raises(RuntimeError, match="No pending ask"):
        optimizer.tell(0.0)
