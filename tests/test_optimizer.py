from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import make_dataclass
from typing import Annotated, Any, cast

import numpy as np
import pytest
from leitwerk import Optimizer, Parameter, SchemaDiff, TellResult, XNESStatus
from leitwerk.optimizer import JSONObject


def _make_schema(schema_name: str, **parameters: Parameter) -> type[Any]:
    return make_dataclass(
        schema_name,
        [(field_name, Annotated[float, parameter]) for field_name, parameter in parameters.items()],
        frozen=True,
    )


def _make_identity_schema(schema_name: str, **parameters: tuple[float, float]) -> type[Any]:
    return _make_schema(
        schema_name,
        **{field_name: Parameter(loc=loc, scale=scale) for field_name, (loc, scale) in parameters.items()},
    )


def _make_mapping_schema(**parameters: object) -> dict[str, object]:
    return dict(parameters)


def _make_mapping_identity_schema(**parameters: tuple[float, float]) -> dict[str, object]:
    return _make_mapping_schema(
        **{field_name: Parameter(loc=loc, scale=scale) for field_name, (loc, scale) in parameters.items()},
    )


def _initialized_optimizer(
    schema: type[Any] | Mapping[str, object],
    *,
    population_size: int,
    minimize: bool = False,
) -> Optimizer[Any]:
    return Optimizer(schema, population_size=population_size, minimize=minimize)


def _initialized_state(schema: type[Any] | Mapping[str, object], *, population_size: int) -> JSONObject:
    state = _initialized_optimizer(schema, population_size=population_size).save()
    assert isinstance(state, dict)
    return state


def _parameter_spec(
    *,
    loc: float | None,
    scale: float = 1.0,
    min: float | None = None,
    max: float | None = None,
) -> dict[str, object]:
    return {"loc": loc, "scale": scale, "min": min, "max": max}


def _read_schema(state: object) -> dict[str, dict[str, object]]:
    assert isinstance(state, dict)
    schema_json = state["schema"]
    assert isinstance(schema_json, dict)
    assert all(isinstance(name, str) for name in schema_json)
    for spec in schema_json.values():
        assert isinstance(spec, Mapping)
        assert list(spec) == ["loc", "scale", "min", "max"]
    return {
        str(name): {str(key): value for key, value in cast(Mapping[str, object], spec).items()}
        for name, spec in schema_json.items()
    }


def _read_schema_names(state: object) -> list[str]:
    return list(_read_schema(state))


def _read_loc(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    loc_json = state["loc"]
    assert isinstance(loc_json, list)
    return np.asarray(loc_json, dtype=float)


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
    loc = _read_loc(state)
    scale = _read_scale(state)
    batch = _read_batch(state)
    return loc[:, None] + scale @ batch


def _read_results(state: object) -> list[tuple[float, ...] | None]:
    assert isinstance(state, dict)
    results_json = state["results"]
    assert isinstance(results_json, list)
    return [None if row is None else tuple(float(value) for value in row) for row in results_json]


def _read_context_pending(state: object) -> dict[str, int]:
    assert isinstance(state, dict)
    context_pending_json = state["context_pending"]
    assert isinstance(context_pending_json, dict)
    assert all(isinstance(context, str) for context in context_pending_json)
    return {str(context): int(sample_idx) for context, sample_idx in context_pending_json.items()}


def _read_status(state: object) -> dict[str, int | float | bool | None]:
    assert isinstance(state, dict)
    status_json = state["status"]
    assert isinstance(status_json, Mapping)
    assert list(status_json) == [
        "total_samples",
        "num_batches",
        "num_restarts",
        "num_parameters",
        "axis_ratio",
        "step_size",
        "batch_progress",
        "batch_size",
        "population_size",
        "minimize",
        "eta_mu",
        "eta_sigma",
        "eta_B",
    ]
    return {
        "total_samples": int(status_json["total_samples"]),
        "num_batches": int(status_json["num_batches"]),
        "num_restarts": int(status_json["num_restarts"]),
        "num_parameters": int(status_json["num_parameters"]),
        "axis_ratio": float(status_json["axis_ratio"]),
        "step_size": float(status_json["step_size"]),
        "batch_progress": float(status_json["batch_progress"]),
        "population_size": None if status_json["population_size"] is None else int(status_json["population_size"]),
        "minimize": bool(status_json["minimize"]),
        "eta_mu": float(status_json["eta_mu"]),
        "eta_sigma": float(status_json["eta_sigma"]),
        "eta_B": float(status_json["eta_B"]),
    }


def _assert_same_status(
    actual: dict[str, int | float | bool | None],
    expected: dict[str, int | float | bool | None],
) -> None:
    for key in ("total_samples", "num_batches", "num_restarts", "num_parameters", "population_size", "minimize"):
        assert actual[key] == expected[key]
    for key in ("axis_ratio", "step_size", "batch_progress", "eta_mu", "eta_sigma", "eta_B"):
        assert actual[key] == pytest.approx(expected[key])


def _softplus_inverse(value: float) -> float:
    return float(value + np.log1p(-np.exp(-value)))


def _run_function_optimization(
    objective: Callable[[np.ndarray], float],
    *,
    init_loc: float,
    init_scale: float,
    dim: int,
    population_size: int,
    evaluations: int,
    minimize: bool = False,
) -> tuple[float, float]:
    schema = _make_identity_schema(
        "SphereParams",
        **{f"x{i}": (init_loc, init_scale) for i in range(dim)},
    )
    optimizer: Optimizer[Any] = Optimizer(schema, population_size=population_size, minimize=minimize)

    initial_loc = _read_loc(optimizer.save())
    initial_value = objective(initial_loc)

    for _ in range(evaluations):
        params = optimizer.ask()
        point = np.array([getattr(params, f"x{i}") for i in range(dim)], dtype=float)
        result = objective(point) if minimize else -objective(point)
        optimizer.tell(result)

    final_loc = _read_loc(optimizer.save())
    final_value = objective(final_loc)
    return initial_value, final_value


def test_optimizer_improves_sphere() -> None:
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    initial, final = _run_function_optimization(
        sphere,
        init_loc=3.0,
        init_scale=2.0,
        dim=4,
        population_size=28,
        evaluations=1400,
    )
    assert final < 0.15 * initial


def test_optimizer_improves_sphere_in_minimization_mode() -> None:
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    initial, final = _run_function_optimization(
        sphere,
        init_loc=3.0,
        init_scale=2.0,
        dim=4,
        population_size=28,
        evaluations=1400,
        minimize=True,
    )
    assert final < 0.15 * initial


def test_state_save_load_roundtrip() -> None:
    schema_a = _make_identity_schema("RoundtripA", alpha=(2.0, 1.5), beta=(-1.0, 2.0))
    opt_a = Optimizer(schema_a, population_size=20)

    for _ in range(37):
        params = opt_a.ask()
        alpha = params.alpha
        beta = params.beta
        opt_a.tell(-(alpha**2 + beta**2))

    state = opt_a.save()
    assert isinstance(state, dict)
    assert "step_size_path" not in state

    schema_b = _make_identity_schema("RoundtripB", beta=(-1.0, 2.0), alpha=(2.0, 1.5))
    opt_b = Optimizer(schema_b, population_size=20)
    load_result = opt_b.load(state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=[], unchanged=["alpha", "beta"])
    loaded = opt_b.save()

    assert _read_schema(loaded) == _read_schema(state)
    assert np.allclose(_read_loc(loaded), _read_loc(state))
    assert np.allclose(_read_scale(loaded), _read_scale(state))
    _assert_same_status(_read_status(loaded), _read_status(state))


def test_status_block_exposes_basic_diagnostics_and_roundtrips() -> None:
    schema = _make_identity_schema("BasicDiagnostics", alpha=(2.0, 1.5), beta=(-1.0, 1.5))
    optimizer = _initialized_optimizer(schema, population_size=4)

    initial_status = _read_status(optimizer.save())
    assert initial_status == {
        "total_samples": 0,
        "num_batches": 0,
        "num_restarts": 0,
        "num_parameters": 2,
        "axis_ratio": 1.0,
        "step_size": 1.5,
        "batch_progress": 0,
        "population_size": 4,
        "minimize": False,
        "eta_mu": 1.0,
        "eta_sigma": 1.0,
        "eta_B": 1.0,
    }

    params = optimizer.ask()
    optimizer.tell(-(params.alpha**2 + params.beta**2))
    progressed_state = optimizer.save()
    progressed_status = _read_status(progressed_state)
    assert progressed_status["total_samples"] == 1
    assert progressed_status["num_batches"] == 0
    assert progressed_status["num_restarts"] == 0
    assert progressed_status["num_parameters"] == 2
    assert progressed_status["axis_ratio"] == pytest.approx(1.0)
    assert progressed_status["step_size"] == pytest.approx(1.5)
    assert progressed_status["batch_progress"] == 1
    assert progressed_status["population_size"] == 4
    assert progressed_status["minimize"] is False
    assert progressed_status["eta_mu"] == pytest.approx(1.0)
    assert progressed_status["eta_sigma"] == pytest.approx(1.0)
    assert progressed_status["eta_B"] == pytest.approx(1.0)

    restored = _initialized_optimizer(schema, population_size=4)
    restored.load(progressed_state)
    _assert_same_status(_read_status(restored.save()), progressed_status)


def test_schema_state_is_human_readable_parameter_specs() -> None:
    schema = _make_schema(
        "ReadableState",
        alpha=Parameter(loc=1.5, scale=0.25, max=3.0),
        beta=Parameter(loc=2.0, scale=3.0, min=0.0),
        gamma=Parameter(scale=1.0, min=0.0, max=1.0),
    )
    state = _initialized_state(schema, population_size=4)

    assert _read_schema(state) == {
        "alpha": _parameter_spec(loc=1.5, scale=0.25, max=3.0),
        "beta": _parameter_spec(loc=2.0, scale=3.0, min=0.0),
        "gamma": _parameter_spec(loc=None, scale=1.0, min=0.0, max=1.0),
    }


def test_load_reconciles_added_and_removed_parameters() -> None:
    base_schema = _make_identity_schema("BaseSchema", x=(2.0, 1.5), y=(-1.0, 0.7))
    base = Optimizer(base_schema, population_size=4)

    for _ in range(5):
        params = base.ask()
        x = params.x
        y = params.y
        base.tell(-(x**2 + 0.5 * y**2))

    base_state = base.save()
    base_loc = _read_loc(base_state)
    base_scale = _read_scale(base_state)
    base_batch = _read_batch(base_state)
    base_batch_latent_points = _read_batch_latent_points(base_state)
    base_results = _read_results(base_state)
    assert any(result is not None for result in base_results)

    added_schema = _make_identity_schema("AddedSchema", x=(2.0, 1.5), y=(-1.0, 0.7), z=(3.0, 2.0))
    added = Optimizer(added_schema, population_size=4)
    load_result = added.load(base_state)
    assert load_result == SchemaDiff(added=["z"], removed=[], changed=[], unchanged=["x", "y"])
    added_state = added.save()

    assert _read_schema_names(added_state) == ["x", "y", "z"]
    assert np.allclose(_read_loc(added_state), np.array([base_loc[0], base_loc[1], 3.0]))
    assert np.allclose(
        _read_scale(added_state),
        np.array(
            [
                [base_scale[0, 0], base_scale[0, 1], 0.0],
                [base_scale[1, 0], base_scale[1, 1], 0.0],
                [0.0, 0.0, 2.0],
            ]
        ),
    )
    assert np.allclose(_read_batch(added_state), np.vstack([base_batch, np.zeros((1, base_batch.shape[1]))]))
    assert np.allclose(
        _read_batch_latent_points(added_state),
        np.vstack([base_batch_latent_points, np.full((1, base_batch_latent_points.shape[1]), 3.0)]),
    )
    assert _read_results(added_state) == base_results

    added.ask()
    added.tell(-1.0)
    added_partial_state = added.save()
    added_loc = _read_loc(added_partial_state)
    added_scale = _read_scale(added_partial_state)
    added_batch = _read_batch(added_partial_state)
    added_batch_latent_points = _read_batch_latent_points(added_partial_state)
    added_results = _read_results(added_partial_state)
    assert any(result is not None for result in added_results)

    removed_schema = _make_identity_schema("RemovedSchema", x=(2.0, 1.5), y=(-1.0, 0.7))
    removed = Optimizer(removed_schema, population_size=4)
    load_result = removed.load(added_partial_state)
    assert load_result == SchemaDiff(added=[], removed=["z"], changed=[], unchanged=["x", "y"])
    removed_state = removed.save()

    assert _read_schema_names(removed_state) == ["x", "y"]
    assert np.allclose(_read_loc(removed_state), added_loc[:2])
    assert np.allclose(_read_scale(removed_state), added_scale[:2, :2])
    assert np.allclose(_read_batch(removed_state), added_batch[:2, :])
    assert np.allclose(_read_batch_latent_points(removed_state), added_batch_latent_points[:2, :])
    assert _read_results(removed_state) == added_results


def test_load_reconciles_bounds_changes_and_selectively_preserves_batch() -> None:
    base_schema = _make_schema(
        "BaseChangedSchema",
        x=Parameter(loc=0.25, scale=1.0, min=0.0),
        y=Parameter(loc=-1.0, scale=0.7),
    )
    base = Optimizer(base_schema, population_size=4)

    for _ in range(5):
        params = base.ask()
        x = params.x
        y = params.y
        base.tell(-((x - 0.2) ** 2 + 0.5 * y**2))

    base_state = base.save()
    base_loc = _read_loc(base_state)
    base_scale = _read_scale(base_state)
    base_batch = _read_batch(base_state)
    base_results = _read_results(base_state)
    completed_mask = np.array([result is not None for result in base_results], dtype=bool)
    pending_mask = ~completed_mask
    half = base_batch.shape[1] // 2
    mirror_index = np.arange(base_batch.shape[1])
    mirror_index[:half] += half
    mirror_index[half:] -= half
    mirror_pending_mask = pending_mask & pending_mask[mirror_index]
    assert completed_mask.any()
    assert pending_mask.any()

    changed_schema = _make_schema(
        "ChangedSchema",
        x=Parameter(loc=0.25, scale=1.0, min=0.0, max=1.0),
        y=Parameter(loc=-1.0, scale=0.7),
    )
    changed = Optimizer(changed_schema, population_size=4)
    load_result = changed.load(base_state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=["x"], unchanged=["y"])
    changed_state = changed.save()
    fresh_state = _initialized_state(changed_schema, population_size=4)

    changed_loc = _read_loc(changed_state)
    changed_scale = _read_scale(changed_state)
    changed_batch = _read_batch(changed_state)

    assert np.allclose(changed_loc, np.array([_read_loc(fresh_state)[0], base_loc[1]]))
    assert np.allclose(
        changed_scale,
        np.array(
            [
                [_read_scale(fresh_state)[0, 0], 0.0],
                [0.0, base_scale[1, 1]],
            ]
        ),
    )
    assert np.allclose(changed_batch[1, :], base_batch[1, :])
    assert np.allclose(changed_batch[0, completed_mask], 0.0)
    assert np.allclose(changed_batch[0, mirror_pending_mask], base_batch[0, mirror_pending_mask])
    assert np.allclose(changed_batch[0, ~mirror_pending_mask], 0.0)
    assert _read_results(changed_state) == base_results
    assert _read_context_pending(changed_state) == _read_context_pending(base_state)


def test_loc_and_scale_changes_do_not_trigger_schema_changeover() -> None:
    base_schema = _make_identity_schema("StrictBase", x=(2.0, 1.5), y=(-1.0, 0.5))
    base_state = _initialized_state(base_schema, population_size=4)

    changed_schema = _make_schema(
        "StrictChanged",
        x=Parameter(loc=4.0, scale=1.5),
        y=Parameter(loc=-1.0, scale=2.0),
    )
    changed = Optimizer(changed_schema, population_size=4)
    load_result = changed.load(base_state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=[], unchanged=["x", "y"])

    changed_state = changed.save()
    assert _read_schema(changed_state) == {
        "x": _parameter_spec(loc=4.0, scale=1.5),
        "y": _parameter_spec(loc=-1.0, scale=2.0),
    }
    assert np.allclose(_read_loc(changed_state), _read_loc(base_state))
    assert np.allclose(_read_scale(changed_state), _read_scale(base_state))
    assert np.allclose(_read_batch(changed_state), _read_batch(base_state))
    assert _read_results(changed_state) == _read_results(base_state)


def test_schema_order_is_lexicographic() -> None:
    first_schema = _make_schema(
        "FirstSchema",
        zeta=Parameter(loc=3.0, scale=1.0, min=0.0),
        alpha=Parameter(loc=1.0, scale=2.0),
        mu=Parameter(loc=2.0, scale=3.0, min=-1.0, max=5.0),
    )
    first = _initialized_optimizer(first_schema, population_size=18)
    state_first = first.save()

    second_schema = _make_schema(
        "SecondSchema",
        mu=Parameter(loc=2.0, scale=3.0, min=-1.0, max=5.0),
        zeta=Parameter(loc=3.0, scale=1.0, min=0.0),
        alpha=Parameter(loc=1.0, scale=2.0),
    )
    second = _initialized_optimizer(second_schema, population_size=18)
    state_second = second.save()

    assert _read_schema_names(state_first) == ["alpha", "mu", "zeta"]
    assert _read_schema_names(state_second) == ["alpha", "mu", "zeta"]
    assert np.allclose(_read_loc(state_first), _read_loc(state_second))


def test_nested_schema_flattens_leaf_names_and_rebuilds_dataclasses() -> None:
    combat = make_dataclass(
        "CombatParameters",
        [
            ("retreat_threshold", Annotated[float, Parameter(loc=-1.0, scale=2.0)]),
            ("attack_threshold", Annotated[float, Parameter(loc=2.0, scale=3.0, min=0.0)]),
        ],
        frozen=True,
    )
    mining = make_dataclass(
        "MiningParameters",
        [("gas_priority", Annotated[float, Parameter(loc=0.5, scale=1.0, min=0.0, max=1.0)])],
        frozen=True,
    )
    schema = make_dataclass(
        "NestedParameters",
        [
            ("mining", mining),
            ("alpha", Annotated[float, Parameter(loc=1.5, scale=0.25, max=3.0)]),
            ("combat", combat),
        ],
        frozen=True,
    )

    expected_names = ["alpha", "combat.attack_threshold", "combat.retreat_threshold", "mining.gas_priority"]
    optimizer: Optimizer[Any] = Optimizer(schema, population_size=6)
    assert _read_schema_names(optimizer.save()) == expected_names

    best = optimizer.ask_best()

    assert best.__class__ is schema
    assert best.combat.__class__ is combat
    assert best.mining.__class__ is mining
    assert best.alpha == 1.5
    assert best.combat.attack_threshold == 2.0
    assert best.combat.retreat_threshold == -1.0
    assert best.mining.gas_priority == 0.5

    params = optimizer.ask()
    assert params.__class__ is schema
    assert params.combat.__class__ is combat
    assert params.mining.__class__ is mining
    assert isinstance(params.alpha, float)
    assert isinstance(params.combat.attack_threshold, float)
    assert isinstance(params.mining.gas_priority, float)
    assert params.combat.attack_threshold > 0.0
    assert 0.0 < params.mining.gas_priority < 1.0
    assert params.alpha < 3.0


def test_nested_mapping_schema_flattens_leaf_names_and_rebuilds_plain_dicts() -> None:
    schema = {
        "mining": {
            "gas_priority": Parameter(loc=0.5, scale=1.0, min=0.0, max=1.0),
        },
        "alpha": Parameter(loc=1.5, scale=0.25, max=3.0),
        "combat": {
            "retreat_threshold": Parameter(loc=-1.0, scale=2.0),
            "attack_threshold": Parameter(loc=2.0, scale=3.0, min=0.0),
        },
    }

    expected_names = ["alpha", "combat.attack_threshold", "combat.retreat_threshold", "mining.gas_priority"]
    optimizer = _initialized_optimizer(schema, population_size=6)
    assert _read_schema_names(optimizer.save()) == expected_names

    best = optimizer.ask_best()

    assert isinstance(best, dict)
    assert isinstance(best["combat"], dict)
    assert isinstance(best["mining"], dict)
    assert best["alpha"] == 1.5
    assert best["combat"]["attack_threshold"] == 2.0
    assert best["combat"]["retreat_threshold"] == -1.0
    assert best["mining"]["gas_priority"] == 0.5

    params = optimizer.ask()
    assert isinstance(params, dict)
    assert isinstance(params["combat"], dict)
    assert isinstance(params["mining"], dict)
    assert isinstance(params["alpha"], float)
    assert isinstance(params["combat"]["attack_threshold"], float)
    assert isinstance(params["mining"]["gas_priority"], float)
    assert params["combat"]["attack_threshold"] > 0.0
    assert 0.0 < params["mining"]["gas_priority"] < 1.0
    assert params["alpha"] < 3.0


def test_mapping_schema_order_is_lexicographic_and_independent_of_insertion_order() -> None:
    first_schema = _make_mapping_schema(
        zeta=Parameter(loc=3.0, scale=1.0, min=0.0),
        alpha=Parameter(loc=1.0, scale=2.0),
        branch={
            "mu": Parameter(loc=2.0, scale=3.0, min=-1.0, max=5.0),
        },
    )
    first = _initialized_optimizer(first_schema, population_size=18)
    state_first = first.save()

    second_schema = _make_mapping_schema(
        branch={
            "mu": Parameter(loc=2.0, scale=3.0, min=-1.0, max=5.0),
        },
        zeta=Parameter(loc=3.0, scale=1.0, min=0.0),
        alpha=Parameter(loc=1.0, scale=2.0),
    )
    second = _initialized_optimizer(second_schema, population_size=18)
    state_second = second.save()

    assert _read_schema_names(state_first) == ["alpha", "branch.mu", "zeta"]
    assert _read_schema_names(state_second) == ["alpha", "branch.mu", "zeta"]
    assert np.allclose(_read_loc(state_first), _read_loc(state_second))


def test_mapping_schema_state_save_load_roundtrip() -> None:
    schema_a = {
        "beta": Parameter(loc=-1.0, scale=2.0),
        "block": {
            "alpha": Parameter(loc=2.0, scale=1.5),
        },
    }
    opt_a: Optimizer[Any] = Optimizer(schema_a, population_size=20)

    for _ in range(37):
        params = opt_a.ask()
        block = cast(dict[str, object], params["block"])
        alpha = cast(float, block["alpha"])
        beta = cast(float, params["beta"])
        opt_a.tell(-(alpha**2 + beta**2))

    state = opt_a.save()
    assert isinstance(state, dict)

    schema_b = {
        "block": {
            "alpha": Parameter(loc=2.0, scale=1.5),
        },
        "beta": Parameter(loc=-1.0, scale=2.0),
    }
    opt_b: Optimizer[Any] = Optimizer(schema_b, population_size=20)
    load_result = opt_b.load(state)
    assert load_result == SchemaDiff(added=[], removed=[], changed=[], unchanged=["beta", "block.alpha"])
    loaded = opt_b.save()

    assert _read_schema(loaded) == _read_schema(state)
    assert np.allclose(_read_loc(loaded), _read_loc(state))
    assert np.allclose(_read_scale(loaded), _read_scale(state))


def test_mapping_schema_rejects_non_parameter_leaf_values() -> None:
    with pytest.raises(TypeError, match=r"must be Parameter\(\.\.\.\) or a nested mapping"):
        Optimizer({"x": 1.0})


def test_schema_requires_parameter_annotated_float_fields() -> None:
    bad_schema = make_dataclass("BadSchema", [("x", float)], frozen=True)

    with pytest.raises(TypeError, match=r"Annotated\[float, Parameter"):
        Optimizer(bad_schema)


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (Parameter(loc=0.0, scale=0.0), r"scale must be a positive finite float"),
        (Parameter(loc=1.0, min=1.0, max=1.0), r"min < max"),
        (Parameter(loc=2.0, min=1.0, max=2.0), r"min < loc < max"),
        (Parameter(loc=1.0, min=1.0), r"loc > min"),
        (Parameter(loc=1.0, max=1.0), r"loc < max"),
    ],
)
def test_schema_rejects_invalid_parameter_domains(parameter: Parameter, message: str) -> None:
    schema = _make_schema("InvalidSchema", x=parameter)

    with pytest.raises(ValueError, match=message):
        Optimizer(schema)


def test_omitted_loc_uses_canonical_latent_zero_center() -> None:
    schema = _make_schema(
        "CanonicalLocs",
        unbounded=Parameter(),
        lower=Parameter(min=2.0),
        upper=Parameter(max=5.0),
        interval=Parameter(min=2.0, max=6.0),
    )
    optimizer = _initialized_optimizer(schema, population_size=6)

    state = optimizer.save()
    assert _read_schema(state) == {
        "interval": _parameter_spec(loc=None, scale=1.0, min=2.0, max=6.0),
        "lower": _parameter_spec(loc=None, scale=1.0, min=2.0),
        "unbounded": _parameter_spec(loc=None, scale=1.0),
        "upper": _parameter_spec(loc=None, scale=1.0, max=5.0),
    }
    assert np.allclose(_read_loc(state), np.zeros(4))

    best = optimizer.ask_best()
    assert best.interval == pytest.approx(4.0)
    assert best.lower == pytest.approx(2.0 + np.log(2.0))
    assert best.unbounded == pytest.approx(0.0)
    assert best.upper == pytest.approx(5.0 - np.log(2.0))


def test_legacy_parameter_constructors_are_not_exposed() -> None:
    assert not hasattr(Parameter, "unbounded")
    assert not hasattr(Parameter, "between")
    assert not hasattr(Parameter, "above")
    assert not hasattr(Parameter, "below")
    assert not hasattr(Parameter, "above_exponential")
    assert not hasattr(Parameter, "below_exponential")


def test_ask_returns_typed_sample() -> None:
    schema = _make_identity_schema("TypedSample", b=(2.0, 1.0), a=(-1.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=6)

    params = optimizer.ask()

    assert params.__class__ is schema
    assert isinstance(params.a, float)
    assert isinstance(params.b, float)


def test_ask_best_returns_current_user_space_locs_without_context() -> None:
    schema = _make_schema(
        "BestParams",
        b=Parameter(scale=1.0, min=0.0, max=1.0),
        a=Parameter(scale=1.0, min=-2.0),
    )
    optimizer = _initialized_optimizer(schema, population_size=6)

    best = optimizer.ask_best()
    assert best.__class__ is schema
    assert best.a == pytest.approx(-2.0 + np.log(2.0))
    assert best.b == pytest.approx(0.5)


def test_context_reuses_mirror_on_repeat() -> None:
    schema = _make_identity_schema("ContextParams", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)
    context = "arena:zerg"

    first_params = optimizer.ask(context=context)
    first = np.array([first_params.x, first_params.y], dtype=float)
    first_report = optimizer.tell(1.0)
    assert first_report == TellResult(False, False, XNESStatus.OK, False)

    optimizer.ask()
    second_report = optimizer.tell(0.0)
    assert second_report == TellResult(False, False, XNESStatus.OK, False)

    mirror_params = optimizer.ask(context=context)
    mirror = np.array([mirror_params.x, mirror_params.y], dtype=float)
    assert np.allclose(mirror, -first)
    third_report = optimizer.tell(-1.0)
    assert third_report == TellResult(False, True, XNESStatus.OK, False)


def test_json_context_reuses_mirror_on_canonical_repeat() -> None:
    schema = _make_identity_schema("JsonContextParams", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    first_params = optimizer.ask(context={"b": 2, "a": 1})
    first = np.array([first_params.x, first_params.y], dtype=float)
    first_report = optimizer.tell(1.0)
    assert first_report == TellResult(False, False, XNESStatus.OK, False)

    optimizer.ask()
    second_report = optimizer.tell(0.0)
    assert second_report == TellResult(False, False, XNESStatus.OK, False)

    mirror_params = optimizer.ask(context={"a": 1, "b": 2})
    mirror = np.array([mirror_params.x, mirror_params.y], dtype=float)
    assert np.allclose(mirror, -first)
    third_report = optimizer.tell(-1.0)
    assert third_report == TellResult(False, True, XNESStatus.OK, False)


def test_json_context_is_saved_as_canonical_string() -> None:
    schema = _make_identity_schema("SavedJsonContext", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    optimizer.ask(context={"b": 2, "a": 1})
    optimizer.tell(1.0)

    assert _read_context_pending(optimizer.save()) == {'{"a":1,"b":2}': 0}


def test_ask_rejects_non_json_context() -> None:
    class Opaque:
        pass

    schema = _make_identity_schema("OpaqueContext", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)
    context = cast(Any, Opaque())

    with pytest.raises(TypeError, match="JSON-serializable"):
        optimizer.ask(context=context)

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


def test_load_raises_with_pending_ask() -> None:
    schema = _make_identity_schema("PendingLoad", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)
    state = optimizer.save()

    optimizer.ask()

    with pytest.raises(RuntimeError, match="Pending ask"):
        optimizer.load(state)


def test_save_raises_with_pending_ask() -> None:
    schema = _make_identity_schema("PendingSave", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)

    optimizer.ask()

    with pytest.raises(RuntimeError, match="Pending ask"):
        optimizer.save()


def test_load_discards_unsaved_local_progress_at_idle_boundary() -> None:
    schema = _make_identity_schema("DiscardUnsavedProgress", x=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=4)
    initial_state = optimizer.save()

    params = optimizer.ask()
    optimizer.tell(params.x)
    progressed_state = optimizer.save()
    assert sum(result is not None for result in _read_results(progressed_state)) == 1

    optimizer.load(initial_state)
    restored_state = optimizer.save()
    assert _read_results(restored_state) == _read_results(initial_state)
    assert np.allclose(_read_loc(restored_state), _read_loc(initial_state))
    assert np.allclose(_read_scale(restored_state), _read_scale(initial_state))
    assert np.allclose(_read_batch(restored_state), _read_batch(initial_state))


def test_runtime_config_is_reported_in_status_but_not_loaded() -> None:
    schema = _make_identity_schema("RuntimeConfig", x=(4.0, 2.0))
    opt_a = Optimizer(
        schema,
        population_size=14,
        minimize=True,
        eta_mu=0.9,
        eta_sigma=0.7,
        eta_B=0.2,
    )
    for _ in range(20):
        params = opt_a.ask()
        x = params.x
        opt_a.tell(x**2)

    state = opt_a.save()
    assert isinstance(state, dict)
    assert "context_pending" in state
    assert "status" in state
    assert "minimize" not in state
    assert "step_size_path" not in state
    assert _read_status(state)["population_size"] == 14
    assert _read_status(state)["minimize"] is True
    assert _read_status(state)["eta_mu"] == pytest.approx(0.9)
    assert _read_status(state)["eta_sigma"] == pytest.approx(0.7)
    assert _read_status(state)["eta_B"] == pytest.approx(0.2)

    opt_b = Optimizer(schema, population_size=4)
    opt_b.load(state)
    loaded = opt_b.save()
    assert isinstance(loaded, dict)
    assert "context_pending" in loaded
    assert "status" in loaded
    assert opt_b.minimize is False
    assert opt_b.eta_mu == 1.0
    assert opt_b.eta_sigma == 1.0
    assert opt_b.eta_B == 1.0
    assert _read_status(loaded)["population_size"] == 4
    assert _read_status(loaded)["minimize"] is False
    assert _read_status(loaded)["eta_mu"] == pytest.approx(1.0)
    assert _read_status(loaded)["eta_sigma"] == pytest.approx(1.0)
    assert _read_status(loaded)["eta_B"] == pytest.approx(1.0)


def test_load_allows_switching_optimization_direction_mid_batch() -> None:
    schema = _make_identity_schema("SwitchDirection", x=(0.0, 1.0))
    base = Optimizer(schema, population_size=4, minimize=True)

    first_params = base.ask()
    base.tell(first_params.x)
    state = base.save()
    saved_results = _read_results(state)
    completed_indices = [idx for idx, result in enumerate(saved_results) if result is not None]
    assert completed_indices == [0]
    assert saved_results[0] == pytest.approx((first_params.x,))

    minimizing = Optimizer(schema, population_size=4, minimize=True)
    maximizing = Optimizer(schema, population_size=4)
    minimizing.load(state)
    maximizing.load(state)

    for optimizer in (minimizing, maximizing):
        for _ in range(3):
            params = optimizer.ask()
            optimizer.tell(params.x)

    assert minimizing.ask_best().x < 0.0
    assert maximizing.ask_best().x > 0.0


def test_restart_on_conditioning_failure() -> None:
    schema = _make_identity_schema("RestartSchema", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = _initialized_optimizer(schema, population_size=10)

    state = optimizer.save()
    assert isinstance(state, dict)
    state["scale"] = [[1e-10, 0.0], [0.0, 1e10]]

    restored = Optimizer(schema, population_size=10)
    restored.load(state)

    for _ in range(10):
        params = restored.ask()
        x = params.x
        y = params.y
        restored.tell(-(x**2 + y**2))

    conditioned = restored.save()
    assert np.allclose(_read_loc(conditioned), np.array([0.0, 0.0]))
    cond = float(np.linalg.cond(_read_scale(conditioned)))
    assert cond < 1e14


def test_successful_termination_restarts_from_final_mu_with_fresh_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _make_identity_schema("RestartSuccess", x=(2.0, 1.5), y=(-1.0, 0.7))
    optimizer = _initialized_optimizer(schema, population_size=4)
    final_mu = np.array([4.25, -3.5])

    def fake_tell(samples: np.ndarray, ranking: list[int]) -> XNESStatus:
        optimizer._xnes.mu = final_mu.copy()
        optimizer._xnes.sigma = 6.0
        optimizer._xnes.B = np.array([[1.5, 0.2], [0.0, 0.5]])
        return XNESStatus.LOC_STEP_MIN

    monkeypatch.setattr(optimizer._xnes, "tell", fake_tell)

    for _ in range(3):
        optimizer.ask()
        report = optimizer.tell(0.0)
        assert report.completed_batch is False

    optimizer.ask()
    report = optimizer.tell(0.0)

    assert report.completed_batch is True
    assert report.status is XNESStatus.LOC_STEP_MIN
    assert report.restarted is True

    state = optimizer.save()
    assert np.allclose(_read_loc(state), final_mu)
    assert np.allclose(_read_scale(state), np.diag([1.5, 0.7]))
    assert _read_status(state)["total_samples"] == 4
    assert _read_status(state)["num_batches"] == 1
    assert _read_status(state)["num_restarts"] == 1
    assert _read_status(state)["batch_progress"] == 0.0


def test_failed_termination_restarts_from_schema_loc_with_fresh_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _make_identity_schema("RestartFailure", x=(2.0, 1.5), y=(-1.0, 0.7))
    optimizer = _initialized_optimizer(schema, population_size=4)

    def fake_tell(samples: np.ndarray, ranking: list[int]) -> XNESStatus:
        optimizer._xnes.mu = np.array([4.25, -3.5])
        optimizer._xnes.sigma = 6.0
        optimizer._xnes.B = np.array([[1.5, 0.2], [0.0, 0.5]])
        return XNESStatus.SCALE_COND_MAX

    monkeypatch.setattr(optimizer._xnes, "tell", fake_tell)

    for _ in range(3):
        optimizer.ask()
        report = optimizer.tell(0.0)
        assert report.completed_batch is False

    optimizer.ask()
    report = optimizer.tell(0.0)

    assert report.completed_batch is True
    assert report.status is XNESStatus.SCALE_COND_MAX
    assert report.restarted is True

    state = optimizer.save()
    assert np.allclose(_read_loc(state), np.array([2.0, -1.0]))
    assert np.allclose(_read_scale(state), np.diag([1.5, 0.7]))
    assert _read_status(state)["total_samples"] == 4
    assert _read_status(state)["num_batches"] == 1
    assert _read_status(state)["num_restarts"] == 1
    assert _read_status(state)["batch_progress"] == 0.0


def test_save_load_preserves_optimizer_state() -> None:
    schema = _make_identity_schema("PreserveState", x=(2.0, 1.5), y=(-1.0, 0.7))
    direct = _initialized_optimizer(schema, population_size=12)

    recreated_state = direct.save()
    for _ in range(40):
        direct_params = direct.ask()
        direct_x = direct_params.x
        direct_y = direct_params.y
        direct_result = -(direct_x**2 + 0.5 * direct_y**2)
        direct.tell(direct_result)

        recreated = Optimizer(schema, population_size=12)
        recreated.load(recreated_state)
        recreated_params = recreated.ask()
        recreated_x = recreated_params.x
        recreated_y = recreated_params.y
        recreated_result = -(recreated_x**2 + 0.5 * recreated_y**2)
        recreated.tell(recreated_result)
        recreated_state = recreated.save()

    direct_state = direct.save()
    assert _read_schema(direct_state) == _read_schema(recreated_state)
    assert np.allclose(_read_loc(direct_state), _read_loc(recreated_state))
    assert np.allclose(_read_scale(direct_state), _read_scale(recreated_state))


def test_transformed_parameters_use_latent_state_but_emit_user_space_values() -> None:
    schema = _make_schema(
        "TransformedParams",
        a=Parameter(loc=1.2, scale=3.4),
        b=Parameter(loc=2.9, scale=1.0, min=2.3, max=3.4),
        c=Parameter(loc=3.0, scale=1.0, min=0.0),
        d=Parameter(loc=3.0, scale=1.0, max=4.0),
    )
    optimizer = Optimizer(schema, population_size=6)

    state = optimizer.save()
    expected_latent_loc = np.array(
        [
            1.2,
            np.log((2.9 - 2.3) / (3.4 - 2.9)),
            _softplus_inverse(3.0),
            -_softplus_inverse(1.0),
        ]
    )
    assert np.allclose(_read_loc(state), expected_latent_loc)
    assert np.allclose(np.diag(_read_scale(state)), np.array([3.4, 1.0, 1.0, 1.0]))

    best = optimizer.ask_best()
    assert best.a == pytest.approx(1.2)
    assert best.b == pytest.approx(2.9)
    assert best.c == pytest.approx(3.0)
    assert best.d == pytest.approx(3.0)

    for _ in range(6):
        params = optimizer.ask()
        assert 2.3 < params.b < 3.4
        assert params.c > 0.0
        assert params.d < 4.0
        optimizer.tell(0.0)

    reloaded = Optimizer(schema, population_size=6)
    reloaded.load(state)
    reloaded_best = reloaded.ask_best()
    assert reloaded_best.a == pytest.approx(best.a)
    assert reloaded_best.b == pytest.approx(best.b)
    assert reloaded_best.c == pytest.approx(best.c)
    assert reloaded_best.d == pytest.approx(best.d)
