from __future__ import annotations

import json
from dataclasses import make_dataclass
from pathlib import Path
from typing import Annotated, Any

import leitwerk.session as session_module
import pytest
from leitwerk import Optimizer, OptimizerSession, OptimizerSettings, Parameter, SchemaDiff

_TEST_SEED = 12345


def _make_schema(schema_name: str, **parameters: tuple[float, float]) -> type[Any]:
    return make_dataclass(
        schema_name,
        [
            (field_name, Annotated[float, Parameter(mean=mean, scale=scale)])
            for field_name, (mean, scale) in parameters.items()
        ],
        frozen=True,
        slots=True,
    )


def _read_state(path: Path) -> dict[str, object]:
    state = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(state, dict)
    return state


def test_session_flush_persists_initial_state_and_restores(tmp_path: Path) -> None:
    schema = _make_schema("SessionParams", beta=(-1.0, 2.0), alpha=(2.0, 1.5))
    settings = OptimizerSettings(population_size=6, seed=_TEST_SEED)
    path = tmp_path / "session.json"
    expected_settings = OptimizerSettings(population_size=6, seed=_TEST_SEED)

    session = OptimizerSession(path, schema, settings=settings)

    assert session.restored is False
    assert session.dirty is False
    assert session.schema_diff == SchemaDiff(added=["alpha", "beta"], removed=[], changed=[], unchanged=[])
    assert session.settings == expected_settings
    assert session.mean.__class__ is schema
    assert session.scale_marginal.__class__ is schema
    assert session.scale_marginal.alpha == 1.5
    assert session.scale_marginal.beta == 2.0

    session.flush()
    assert session.dirty is False

    expected = Optimizer(schema, settings=settings).save()
    assert _read_state(path) == expected

    restored = OptimizerSession(path, schema)

    assert restored.restored is True
    assert restored.dirty is False
    assert restored.settings == expected_settings
    assert restored.schema_diff == SchemaDiff(added=[], removed=[], changed=[], unchanged=["alpha", "beta"])
    assert restored.mean == session.mean
    assert restored.scale_marginal == session.scale_marginal


def test_session_tell_persists_committed_progress(tmp_path: Path) -> None:
    schema = _make_schema("TellSessionParams", x=(2.0, 1.5), y=(-1.0, 0.7))
    settings = OptimizerSettings(population_size=4, seed=_TEST_SEED)
    path = tmp_path / "session.json"
    session = OptimizerSession(path, schema, settings=settings)
    optimizer: Optimizer[Any] = Optimizer(schema, settings=settings)

    session_params = session.ask()
    optimizer_params = optimizer.ask()
    assert session_params == optimizer_params

    result = -(session_params.x**2 + 0.5 * session_params.y**2)
    assert session.tell(result) == optimizer.tell(result)
    assert _read_state(path) == optimizer.save()

    restored = OptimizerSession(path, schema, settings=settings)
    assert restored.ask() == session.ask()


def test_session_reports_schema_diff_on_restore(tmp_path: Path) -> None:
    base_schema = _make_schema("BaseSessionParams", y=(-1.0, 0.7), x=(2.0, 1.5))
    path = tmp_path / "session.json"
    settings = OptimizerSettings(population_size=4, seed=_TEST_SEED)

    OptimizerSession(path, base_schema, settings=settings).flush()

    changed_schema = _make_schema("ChangedSessionParams", z=(3.0, 2.0), x=(2.0, 1.5), y=(-1.0, 0.7))
    restored = OptimizerSession(path, changed_schema, settings=settings)

    assert restored.restored is True
    assert restored.schema_diff == SchemaDiff(added=["z"], removed=[], changed=[], unchanged=["x", "y"])
    assert restored.mean.__class__ is changed_schema


def test_session_failed_tell_flush_marks_dirty_and_blocks_ask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _make_schema("DirtySessionParams", x=(2.0, 1.5))
    path = tmp_path / "session.json"
    session = OptimizerSession(path, schema, settings=OptimizerSettings(population_size=4, seed=_TEST_SEED))

    def fail_write(path: Path, payload: dict[str, object]) -> None:
        del path, payload
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(session_module, "_write_json_atomically", fail_write)

    params = session.ask()

    with pytest.raises(OSError, match="disk full"):
        session.tell(-(params.x**2))

    assert session.dirty is True
    with pytest.raises(RuntimeError, match=r"flush\(\) before ask"):
        session.ask()


def test_session_flush_recovers_dirty_state_after_transient_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _make_schema("DirtyRecoveryParams", x=(2.0, 1.5))
    path = tmp_path / "session.json"
    session = OptimizerSession(path, schema, settings=OptimizerSettings(population_size=4, seed=_TEST_SEED))
    real_write = session_module._write_json_atomically

    def fail_write(path: Path, payload: dict[str, object]) -> None:
        del path, payload
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(session_module, "_write_json_atomically", fail_write)

    params = session.ask()

    with pytest.raises(OSError, match="disk full"):
        session.tell(-(params.x**2))

    assert session.dirty is True

    monkeypatch.setattr(session_module, "_write_json_atomically", real_write)
    session.flush()

    assert session.dirty is False
    assert path.exists()
    assert session.ask().__class__ is schema


def test_session_runtime_settings_override_persisted_baseline(tmp_path: Path) -> None:
    schema = _make_schema("SessionSettings", x=(2.0, 1.5))
    path = tmp_path / "session.json"
    OptimizerSession(
        path,
        schema,
        settings=OptimizerSettings(population_size=6, seed=_TEST_SEED, minimize=True, eta_mean=0.9),
    ).flush()

    restored = OptimizerSession(path, schema, settings=OptimizerSettings(seed=999, eta_scale_global=0.3))

    assert restored.restored is True
    assert restored.settings == OptimizerSettings(
        population_size=6,
        seed=999,
        minimize=True,
        eta_mean=0.9,
        eta_scale_global=0.3,
    )

    restored.flush()

    reloaded = OptimizerSession(path, schema)
    assert reloaded.settings == restored.settings


def test_session_rejects_non_object_checkpoint(tmp_path: Path) -> None:
    schema = _make_schema("InvalidCheckpointParams", x=(0.0, 1.0))
    path = tmp_path / "session.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="JSON object"):
        OptimizerSession(path, schema)
