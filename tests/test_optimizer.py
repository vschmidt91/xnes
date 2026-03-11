from __future__ import annotations

from collections.abc import Callable
from dataclasses import make_dataclass
from typing import Annotated, Any

import numpy as np
import pytest

from xnes import LoadResult, Optimizer, Parameter, TellResult, XNESStatus


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


def _read_loc(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    loc_json = state["loc"]
    assert isinstance(loc_json, list)
    return np.asarray(loc_json, dtype=float)


def _read_names(state: object) -> list[str]:
    assert isinstance(state, dict)
    names_json = state["names"]
    assert isinstance(names_json, list)
    assert all(isinstance(name, str) for name in names_json)
    return list(names_json)


def _read_scale(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    scale_json = state["scale"]
    assert isinstance(scale_json, list)
    return np.asarray(scale_json, dtype=float)


def _read_step_size_path(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    step_size_path_json = state["step_size_path"]
    assert isinstance(step_size_path_json, list)
    return np.asarray(step_size_path_json, dtype=float)


def _read_batch_z(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    batch_z_json = state["batch_z"]
    assert isinstance(batch_z_json, list)
    return np.asarray(batch_z_json, dtype=float)


def _read_batch_latent_points(state: object) -> np.ndarray:
    loc = _read_loc(state)
    scale = _read_scale(state)
    batch_z = _read_batch_z(state)
    return loc[:, None] + scale @ batch_z


def _read_results(state: object) -> list[tuple[float, ...] | None]:
    assert isinstance(state, dict)
    results_json = state["results"]
    assert isinstance(results_json, list)
    return [None if row is None else tuple(float(value) for value in row) for row in results_json]


def _softplus_inverse(value: float) -> float:
    return float(value + np.log1p(-np.exp(-value)))


def _run_function_optimization(
    objective: Callable[[np.ndarray], float],
    *,
    init_loc: float,
    init_scale: float,
    dim: int,
    pop_size: int,
    evaluations: int,
) -> tuple[float, float]:
    schema = _make_identity_schema(
        "SphereParams",
        **{f"x{i}": (init_loc, init_scale) for i in range(dim)},
    )
    optimizer: Optimizer[Any] = Optimizer(schema)
    optimizer.pop_size = pop_size
    optimizer.load(None)

    initial_loc = _read_loc(optimizer.save())
    initial_value = objective(initial_loc)

    for _ in range(evaluations):
        sample = optimizer.ask()
        point = np.array([getattr(sample.params, f"x{i}") for i in range(dim)], dtype=float)
        optimizer.tell(sample, -objective(point))

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
        pop_size=28,
        evaluations=1400,
    )
    assert final < 0.15 * initial


def test_state_save_load_roundtrip() -> None:
    schema_a = _make_identity_schema("RoundtripA", alpha=(2.0, 1.5), beta=(-1.0, 2.0))
    opt_a = Optimizer(schema_a)
    opt_a.pop_size = 20
    load_result = opt_a.load(None)
    assert load_result == LoadResult(["alpha", "beta"], [])

    for _ in range(37):
        sample = opt_a.ask()
        alpha = sample.params.alpha
        beta = sample.params.beta
        opt_a.tell(sample, -(alpha**2 + beta**2))

    state = opt_a.save()
    assert isinstance(state, dict)

    schema_b = _make_identity_schema("RoundtripB", beta=(-999.0, 0.5), alpha=(999.0, 0.5))
    opt_b = Optimizer(schema_b)
    opt_b.pop_size = 20
    load_result = opt_b.load(state)
    assert load_result == LoadResult([], [])
    loaded = opt_b.save()

    assert _read_names(loaded) == _read_names(state)
    assert np.allclose(_read_loc(loaded), _read_loc(state))
    assert np.allclose(_read_scale(loaded), _read_scale(state))


def test_load_reconciles_added_and_removed_parameters() -> None:
    base_schema = _make_identity_schema("BaseSchema", x=(2.0, 1.5), y=(-1.0, 0.7))
    base = Optimizer(base_schema)
    base.pop_size = 4
    load_result = base.load(None)
    assert load_result == LoadResult(["x", "y"], [])

    for _ in range(5):
        sample = base.ask()
        x = sample.params.x
        y = sample.params.y
        base.tell(sample, -(x**2 + 0.5 * y**2))

    base_state = base.save()
    base_loc = _read_loc(base_state)
    base_scale = _read_scale(base_state)
    base_step_size_path = _read_step_size_path(base_state)
    base_batch_z = _read_batch_z(base_state)
    base_batch_latent_points = _read_batch_latent_points(base_state)
    base_results = _read_results(base_state)
    assert any(result is not None for result in base_results)

    added_schema = _make_identity_schema("AddedSchema", x=(999.0, 9.0), y=(999.0, 9.0), z=(3.0, 2.0))
    added = Optimizer(added_schema)
    added.pop_size = 4
    load_result = added.load(base_state)
    assert load_result == LoadResult(["z"], [])
    added_state = added.save()

    assert _read_names(added_state) == ["x", "y", "z"]
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
    assert np.allclose(
        _read_step_size_path(added_state),
        np.array([base_step_size_path[0], base_step_size_path[1], 0.0]),
    )
    assert np.allclose(_read_batch_z(added_state), np.vstack([base_batch_z, np.zeros((1, base_batch_z.shape[1]))]))
    assert np.allclose(
        _read_batch_latent_points(added_state),
        np.vstack([base_batch_latent_points, np.full((1, base_batch_latent_points.shape[1]), 3.0)]),
    )
    assert _read_results(added_state) == base_results

    sample = added.ask()
    added.tell(sample, -1.0)
    added_partial_state = added.save()
    assert any(result is not None for result in _read_results(added_partial_state))
    added_loc = _read_loc(added_state)
    added_scale = _read_scale(added_state)
    added_step_size_path = _read_step_size_path(added_state)
    added_batch_z = _read_batch_z(added_partial_state)
    added_batch_latent_points = _read_batch_latent_points(added_partial_state)
    added_results = _read_results(added_partial_state)

    removed_schema = _make_identity_schema("RemovedSchema", x=(-999.0, 5.0), y=(-999.0, 5.0))
    removed = Optimizer(removed_schema)
    removed.pop_size = 4
    load_result = removed.load(added_partial_state)
    assert load_result == LoadResult([], ["z"])
    removed_state = removed.save()

    assert _read_names(removed_state) == ["x", "y"]
    assert np.allclose(_read_loc(removed_state), added_loc[:2])
    assert np.allclose(_read_scale(removed_state), added_scale[:2, :2])
    assert np.allclose(_read_step_size_path(removed_state), added_step_size_path[:2])
    assert np.allclose(_read_batch_z(removed_state), added_batch_z[:2, :])
    assert np.allclose(_read_batch_latent_points(removed_state), added_batch_latent_points[:2, :])
    assert _read_results(removed_state) == added_results


def test_schema_order_is_lexicographic() -> None:
    first_schema = _make_schema(
        "FirstSchema",
        zeta=Parameter.above(lower=0.0, loc=3.0, scale=1.0),
        alpha=Parameter(loc=1.0, scale=2.0),
        mu=Parameter.between(lower=-1.0, upper=5.0, loc=2.0, scale=3.0),
    )
    first = Optimizer(first_schema)
    first.pop_size = 18
    first.load(None)
    state_first = first.save()

    second_schema = _make_schema(
        "SecondSchema",
        mu=Parameter.between(lower=-1.0, upper=5.0, loc=2.0, scale=3.0),
        zeta=Parameter.above(lower=0.0, loc=3.0, scale=1.0),
        alpha=Parameter(loc=1.0, scale=2.0),
    )
    second = Optimizer(second_schema)
    second.pop_size = 18
    second.load(None)
    state_second = second.save()

    assert _read_names(state_first) == ["alpha", "mu", "zeta"]
    assert _read_names(state_second) == ["alpha", "mu", "zeta"]
    assert np.allclose(_read_loc(state_first), _read_loc(state_second))


def test_nested_schema_flattens_leaf_names_and_rebuilds_dataclasses() -> None:
    combat = make_dataclass(
        "CombatParameters",
        [
            ("retreat_threshold", Annotated[float, Parameter(loc=-1.0, scale=2.0)]),
            ("attack_threshold", Annotated[float, Parameter.above(lower=0.0, loc=2.0, scale=3.0)]),
        ],
        frozen=True,
    )
    mining = make_dataclass(
        "MiningParameters",
        [("gas_priority", Annotated[float, Parameter.between(lower=0.0, upper=1.0, loc=0.5, scale=1.0)])],
        frozen=True,
    )
    schema = make_dataclass(
        "NestedParameters",
        [
            ("mining", mining),
            ("alpha", Annotated[float, Parameter.below(upper=3.0, loc=1.5, scale=0.25)]),
            ("combat", combat),
        ],
        frozen=True,
    )

    optimizer: Optimizer[Any] = Optimizer(schema)
    optimizer.pop_size = 6
    load_result = optimizer.load(None)

    expected_names = ["alpha", "combat.attack_threshold", "combat.retreat_threshold", "mining.gas_priority"]
    assert load_result == LoadResult(expected_names, [])
    assert _read_names(optimizer.save()) == expected_names

    best = optimizer.ask_best()

    assert best.params.__class__ is schema
    assert best.params.combat.__class__ is combat
    assert best.params.mining.__class__ is mining
    assert best.params.alpha == 1.5
    assert best.params.combat.attack_threshold == 2.0
    assert best.params.combat.retreat_threshold == -1.0
    assert best.params.mining.gas_priority == 0.5

    sample = optimizer.ask()
    assert sample.params.__class__ is schema
    assert sample.params.combat.__class__ is combat
    assert sample.params.mining.__class__ is mining
    assert isinstance(sample.params.alpha, float)
    assert isinstance(sample.params.combat.attack_threshold, float)
    assert isinstance(sample.params.mining.gas_priority, float)
    assert sample.params.combat.attack_threshold > 0.0
    assert 0.0 < sample.params.mining.gas_priority < 1.0
    assert sample.params.alpha < 3.0


def test_schema_requires_parameter_annotated_float_fields() -> None:
    bad_schema = make_dataclass("BadSchema", [("x", float)], frozen=True)

    with pytest.raises(TypeError, match=r"Annotated\[float, Parameter"):
        Optimizer(bad_schema)


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (Parameter(loc=0.0, scale=0.0), r"scale must be a positive finite float"),
        (Parameter.between(lower=1.0, upper=1.0, loc=1.0), r"lower < upper"),
        (Parameter.between(lower=1.0, upper=2.0, loc=2.0), r"lower < loc < upper"),
        (Parameter.above(lower=1.0, loc=1.0), r"loc > lower"),
        (Parameter.below(upper=1.0, loc=1.0), r"loc < upper"),
        (Parameter.above_exponential(lower=2.0, loc=2.0), r"loc > lower"),
        (Parameter.below_exponential(upper=2.0, loc=2.0), r"loc < upper"),
    ],
)
def test_schema_rejects_invalid_parameter_domains(parameter: Parameter, message: str) -> None:
    schema = _make_schema("InvalidSchema", x=parameter)

    with pytest.raises(ValueError, match=message):
        Optimizer(schema)


def test_ask_returns_typed_sample() -> None:
    schema = _make_identity_schema("TypedSample", b=(2.0, 1.0), a=(-1.0, 1.0))
    optimizer = Optimizer(schema)
    optimizer.pop_size = 6
    optimizer.load(None)

    sample = optimizer.ask()

    assert sample.sample_id is not None
    assert sample.context is None
    assert sample.matched_context is False
    assert sample.params.__class__ is schema
    assert isinstance(sample.params.a, float)
    assert isinstance(sample.params.b, float)


def test_ask_best_returns_current_user_space_locs_without_context() -> None:
    schema = _make_schema(
        "BestParams",
        b=Parameter.between(lower=0.0, upper=1.0, loc=0.25, scale=1.0),
        a=Parameter.above(lower=-2.0, loc=1.0, scale=1.0),
    )
    optimizer = Optimizer(schema)
    optimizer.pop_size = 6
    optimizer.load(None)

    best = optimizer.ask_best()

    assert best.sample_id is None
    assert best.context is None
    assert best.matched_context is False
    assert best.params.__class__ is schema
    assert best.params.a == 1.0
    assert best.params.b == 0.25


def test_tell_rejects_ask_best_sample() -> None:
    schema = _make_identity_schema("SingleParam", x=(0.0, 1.0))
    optimizer = Optimizer(schema)
    optimizer.pop_size = 4
    optimizer.load(None)

    best = optimizer.ask_best()

    with pytest.raises(RuntimeError, match="ask_best"):
        optimizer.tell(best, 1.0)


def test_context_reuses_mirror_on_repeat() -> None:
    schema = _make_identity_schema("ContextParams", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = Optimizer(schema)
    optimizer.pop_size = 4
    optimizer.load(None)
    context = "arena:zerg"

    first_sample = optimizer.ask(context=context)
    first = np.array([first_sample.params.x, first_sample.params.y], dtype=float)
    first_report = optimizer.tell(first_sample, 1.0)
    assert first_report == TellResult(False, False, XNESStatus.OK, False)

    second_sample = optimizer.ask()
    second_report = optimizer.tell(second_sample, 0.0)
    assert second_report == TellResult(False, False, XNESStatus.OK, False)

    mirror_sample = optimizer.ask(context=context)
    mirror = np.array([mirror_sample.params.x, mirror_sample.params.y], dtype=float)
    assert np.allclose(mirror, -first)
    third_report = optimizer.tell(mirror_sample, -1.0)
    assert third_report == TellResult(False, True, XNESStatus.OK, False)


def test_ask_allows_out_of_order_tells() -> None:
    schema = _make_identity_schema("OutOfOrder", x=(0.0, 1.0))
    optimizer = Optimizer(schema)
    optimizer.pop_size = 4
    optimizer.load(None)

    first = optimizer.ask()
    second = optimizer.ask()

    r2 = optimizer.tell(second, 0.0)
    r1 = optimizer.tell(first, 1.0)
    assert r2.completed_batch is False
    assert r1.completed_batch is False


def test_ask_raises_when_batch_is_fully_claimed() -> None:
    schema = _make_identity_schema("ClaimedBatch", x=(0.0, 1.0))
    optimizer = Optimizer(schema)
    optimizer.pop_size = 4
    optimizer.load(None)

    samples = [optimizer.ask() for _ in range(4)]
    assert len(samples) == 4
    with pytest.raises(RuntimeError, match="Pending"):
        optimizer.ask()


def test_stale_tell_is_rejected() -> None:
    schema = _make_identity_schema("StaleTell", x=(0.0, 1.0))
    optimizer = Optimizer(schema)
    optimizer.pop_size = 4
    optimizer.load(None)

    sample = optimizer.ask()
    optimizer.tell(sample, 1.0)

    with pytest.raises(RuntimeError, match="Unknown"):
        optimizer.tell(sample, 1.0)


def test_runtime_config_is_not_persisted_or_loaded() -> None:
    schema = _make_identity_schema("RuntimeConfig", x=(4.0, 2.0))
    opt_a = Optimizer(schema)
    opt_a.pop_size = 14
    opt_a.csa_enabled = False
    opt_a.eta_mu = 0.9
    opt_a.eta_sigma = 0.7
    opt_a.eta_B = 0.2
    opt_a.load(None)
    for _ in range(20):
        sample = opt_a.ask()
        x = sample.params.x
        opt_a.tell(sample, -(x**2))

    state = opt_a.save()
    assert isinstance(state, dict)
    assert "context_waiting" in state

    opt_b = Optimizer(schema)
    opt_b.pop_size = 4
    opt_b.load(state)
    loaded = opt_b.save()
    assert isinstance(loaded, dict)
    assert "context_waiting" in loaded
    assert opt_b.csa_enabled is None
    assert opt_b.eta_mu is None
    assert opt_b.eta_sigma is None
    assert opt_b.eta_B is None


def test_restart_on_conditioning_failure() -> None:
    schema = _make_identity_schema("RestartSchema", x=(0.0, 1.0), y=(0.0, 1.0))
    optimizer = Optimizer(schema)
    optimizer.pop_size = 10
    optimizer.load(None)

    state = optimizer.save()
    assert isinstance(state, dict)
    state["scale"] = [[1e-10, 0.0], [0.0, 1e10]]

    restored = Optimizer(schema)
    restored.pop_size = 10
    restored.load(state)

    for _ in range(10):
        sample = restored.ask()
        x = sample.params.x
        y = sample.params.y
        restored.tell(sample, -(x**2 + y**2))

    conditioned = restored.save()
    assert np.allclose(_read_loc(conditioned), np.array([0.0, 0.0]))
    cond = float(np.linalg.cond(_read_scale(conditioned)))
    assert cond < 1e14


def test_save_load_preserves_optimizer_state() -> None:
    schema = _make_identity_schema("PreserveState", x=(2.0, 1.5), y=(-1.0, 0.7))
    direct = Optimizer(schema)
    direct.pop_size = 12
    direct.load(None)

    recreated_state = direct.save()
    for _ in range(40):
        direct_sample = direct.ask()
        direct_x = direct_sample.params.x
        direct_y = direct_sample.params.y
        direct_result = -(direct_x**2 + 0.5 * direct_y**2)
        direct.tell(direct_sample, direct_result)

        recreated = Optimizer(schema)
        recreated.pop_size = 12
        recreated.load(recreated_state)
        recreated_sample = recreated.ask()
        recreated_x = recreated_sample.params.x
        recreated_y = recreated_sample.params.y
        recreated_result = -(recreated_x**2 + 0.5 * recreated_y**2)
        recreated.tell(recreated_sample, recreated_result)
        recreated_state = recreated.save()

    direct_state = direct.save()
    assert _read_names(direct_state) == _read_names(recreated_state)
    assert np.allclose(_read_loc(direct_state), _read_loc(recreated_state))
    assert np.allclose(_read_scale(direct_state), _read_scale(recreated_state))
    assert np.allclose(_read_step_size_path(direct_state), _read_step_size_path(recreated_state))


def test_transformed_parameters_use_latent_state_but_emit_user_space_values() -> None:
    schema = _make_schema(
        "TransformedParams",
        a=Parameter(loc=1.2, scale=3.4),
        b=Parameter.between(lower=2.3, upper=3.4, loc=2.9, scale=1.0),
        c=Parameter.above(lower=0.0, loc=3.0, scale=1.0),
        d=Parameter.below(upper=4.0, loc=3.0, scale=1.0),
        e=Parameter.above_exponential(lower=0.0, loc=3.0, scale=1.0),
        f=Parameter.below_exponential(upper=4.0, loc=3.0, scale=1.0),
    )
    optimizer = Optimizer(schema)
    optimizer.pop_size = 6
    load_result = optimizer.load(None)

    assert load_result == LoadResult(["a", "b", "c", "d", "e", "f"], [])

    state = optimizer.save()
    expected_latent_loc = np.array(
        [
            1.2,
            np.log((2.9 - 2.3) / (3.4 - 2.9)),
            _softplus_inverse(3.0),
            -_softplus_inverse(1.0),
            np.log(3.0),
            -np.log(1.0),
        ]
    )
    assert np.allclose(_read_loc(state), expected_latent_loc)
    assert np.allclose(np.diag(_read_scale(state)), np.array([3.4, 1.0, 1.0, 1.0, 1.0, 1.0]))

    best = optimizer.ask_best()
    assert best.params.a == pytest.approx(1.2)
    assert best.params.b == pytest.approx(2.9)
    assert best.params.c == pytest.approx(3.0)
    assert best.params.d == pytest.approx(3.0)
    assert best.params.e == pytest.approx(3.0)
    assert best.params.f == pytest.approx(3.0)

    samples = [optimizer.ask() for _ in range(6)]
    for sample in samples:
        assert 2.3 < sample.params.b < 3.4
        assert sample.params.c > 0.0
        assert sample.params.d < 4.0
        assert sample.params.e > 0.0
        assert sample.params.f < 4.0
    for sample in samples:
        optimizer.tell(sample, 0.0)

    reloaded = Optimizer(schema)
    reloaded.pop_size = 6
    reloaded.load(state)
    reloaded_best = reloaded.ask_best().params
    assert reloaded_best.a == pytest.approx(best.params.a)
    assert reloaded_best.b == pytest.approx(best.params.b)
    assert reloaded_best.c == pytest.approx(best.params.c)
    assert reloaded_best.d == pytest.approx(best.params.d)
    assert reloaded_best.e == pytest.approx(best.params.e)
    assert reloaded_best.f == pytest.approx(best.params.f)
