from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import make_dataclass
from typing import Annotated, Any, cast

import numpy as np
import pytest
from leitwerk import Optimizer, OptimizerSettings, Parameter
from leitwerk.state import JSONObject

_TEST_SEED = 12345


def _make_schema(schema_name: str, **parameters: Parameter) -> type[Any]:
    return make_dataclass(
        schema_name,
        [(field_name, Annotated[float, parameter]) for field_name, parameter in parameters.items()],
        frozen=True,
        slots=True,
    )


def _make_identity_schema(schema_name: str, **parameters: tuple[float, float]) -> type[Any]:
    return _make_schema(
        schema_name,
        **{field_name: Parameter(mean=mean, scale=scale) for field_name, (mean, scale) in parameters.items()},
    )


def _make_mapping_schema(**parameters: object) -> dict[str, object]:
    return dict(parameters)


def _initialized_optimizer(
    schema: type[Any] | Mapping[str, object],
    *,
    batch_size: int,
    minimize: bool | None = None,
) -> Optimizer[Any]:
    return _optimizer(schema, batch_size=batch_size, minimize=minimize)


def _optimizer(
    schema: type[Any] | Mapping[str, object],
    *,
    batch_size: int | None = None,
    seed: int | None = _TEST_SEED,
    minimize: bool | None = None,
    eta_mean: float | None = None,
    eta_scale_global: float | None = None,
    eta_scale_shape: float | None = None,
) -> Optimizer[Any]:
    return Optimizer(
        schema,
        settings=OptimizerSettings(
            batch_size=batch_size,
            seed=seed,
            minimize=minimize,
            eta_mean=eta_mean,
            eta_scale_global=eta_scale_global,
            eta_scale_shape=eta_scale_shape,
        ),
    )


def _initialized_state(schema: type[Any] | Mapping[str, object], *, batch_size: int) -> JSONObject:
    state = _initialized_optimizer(schema, batch_size=batch_size).save()
    assert isinstance(state, dict)
    return state


def _parameter_spec(
    *,
    mean: float | None,
    scale: float = 1.0,
    min: float | None = None,
    max: float | None = None,
) -> dict[str, object]:
    return {"mean": mean, "scale": scale, "min": min, "max": max}


def _read_schema(state: object) -> dict[str, dict[str, object]]:
    assert isinstance(state, dict)
    schema_json = state["schema"]
    assert isinstance(schema_json, dict)
    assert all(isinstance(name, str) for name in schema_json)
    for spec in schema_json.values():
        assert isinstance(spec, Mapping)
    return {
        str(name): {str(key): value for key, value in cast(Mapping[str, object], spec).items()}
        for name, spec in schema_json.items()
    }


def _read_schema_names(state: object) -> list[str]:
    return list(_read_schema(state))


def _read_mean(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    mean_json = state["mean"]
    assert isinstance(mean_json, list)
    return np.asarray(mean_json, dtype=float)


def _read_scale(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    scale_json = state["scale"]
    assert isinstance(scale_json, list)
    return np.asarray(scale_json, dtype=float)


def _read_batch(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    batch_json = state["batch"]
    assert isinstance(batch_json, list)
    return np.asarray(batch_json, dtype=float)


def _read_batch_latent_points(state: object) -> np.ndarray:
    mean = _read_mean(state)
    scale = _read_scale(state)
    batch = _read_batch(state)
    return mean[:, None] + scale @ batch


def _read_results(state: object) -> list[tuple[float, ...] | None]:
    assert isinstance(state, dict)
    results_json = state["results"]
    assert isinstance(results_json, list)
    return [None if row is None else tuple(float(value) for value in row) for row in results_json]


def _read_pending_context_matches(state: object) -> dict[str, int]:
    assert isinstance(state, dict)
    pending_context_matches_json = state["pending_context_matches"]
    assert isinstance(pending_context_matches_json, dict)
    assert all(isinstance(context, str) for context in pending_context_matches_json)
    return {str(context): int(sample_idx) for context, sample_idx in pending_context_matches_json.items()}


def _read_settings(state: object) -> dict[str, int | float | bool | None]:
    assert isinstance(state, dict)
    settings_json = state["settings"]
    assert isinstance(settings_json, Mapping)
    batch_size = settings_json["batch_size"]
    seed = settings_json["seed"]
    minimize = settings_json["minimize"]
    eta_mean = settings_json["eta_mean"]
    eta_scale_global = settings_json["eta_scale_global"]
    eta_scale_shape = settings_json["eta_scale_shape"]
    return {
        "batch_size": None if batch_size is None else int(batch_size),
        "seed": None if seed is None else int(seed),
        "minimize": None if minimize is None else bool(minimize),
        "eta_mean": None if eta_mean is None else float(eta_mean),
        "eta_scale_global": None if eta_scale_global is None else float(eta_scale_global),
        "eta_scale_shape": None if eta_scale_shape is None else float(eta_scale_shape),
    }


def _read_status(state: object) -> dict[str, int | float]:
    assert isinstance(state, dict)
    status_json = state["status"]
    assert isinstance(status_json, Mapping)
    return {
        "num_samples": int(status_json["num_samples"]),
        "num_batches": int(status_json["num_batches"]),
        "num_restarts": int(status_json["num_restarts"]),
        "num_parameters": int(status_json["num_parameters"]),
        "axis_ratio": float(status_json["axis_ratio"]),
        "scale_global": float(status_json["scale_global"]),
        "batch_progress": float(status_json["batch_progress"]),
        "batch_size": int(status_json["batch_size"]),
    }


def _assert_same_status(actual: dict[str, int | float], expected: dict[str, int | float]) -> None:
    for key in ("num_samples", "num_batches", "num_restarts", "num_parameters", "batch_size"):
        assert actual[key] == expected[key]
    for key in ("axis_ratio", "scale_global", "batch_progress"):
        assert actual[key] == pytest.approx(expected[key])


def _assert_same_settings(
    actual: dict[str, int | float | bool | None],
    expected: dict[str, int | float | bool | None],
) -> None:
    assert actual.keys() == expected.keys()
    for key in ("batch_size", "seed", "minimize"):
        assert actual[key] == expected[key]
    for key in ("eta_mean", "eta_scale_global", "eta_scale_shape"):
        if expected[key] is None:
            assert actual[key] is None
        else:
            assert actual[key] == pytest.approx(expected[key])


def _softplus_inverse(value: float) -> float:
    return float(value + np.log1p(-np.exp(-value)))


def _run_function_optimization(
    objective: Callable[[np.ndarray], float],
    *,
    init_mean: float,
    init_scale: float,
    dim: int,
    batch_size: int,
    evaluations: int,
    minimize: bool = False,
) -> tuple[float, float]:
    schema = _make_identity_schema(
        "SphereParams",
        **{f"x{i}": (init_mean, init_scale) for i in range(dim)},
    )
    optimizer = _optimizer(schema, batch_size=batch_size, minimize=minimize)

    initial_mean = _read_mean(optimizer.save())
    initial_value = objective(initial_mean)

    for _ in range(evaluations):
        params = optimizer.ask()
        point = np.array([getattr(params, f"x{i}") for i in range(dim)], dtype=float)
        result = objective(point) if minimize else -objective(point)
        optimizer.tell(result)

    final_mean = _read_mean(optimizer.save())
    final_value = objective(final_mean)
    return initial_value, final_value
