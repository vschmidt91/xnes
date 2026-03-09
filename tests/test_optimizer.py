from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from xnes._scheduler import JSONValue

from xnes import LoadResult, Optimizer, ParameterInfo, Report, XNESStatus


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


def _read_batch_x(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    batch_x_json = state["batch_x"]
    assert isinstance(batch_x_json, list)
    return np.asarray(batch_x_json, dtype=float)


def _read_results(state: object) -> list[tuple[float, ...] | None]:
    assert isinstance(state, dict)
    results_json = state["results"]
    assert isinstance(results_json, list)
    return [None if row is None else tuple(float(value) for value in row) for row in results_json]


def _run_function_optimization(
    objective: Callable[[np.ndarray], float],
    *,
    init_loc: float,
    init_scale: float,
    dim: int,
    pop_size: int,
    evaluations: int,
) -> tuple[float, float]:
    optimizer = Optimizer()
    optimizer.pop_size = pop_size
    params = [optimizer.add(f"x{i}", loc=init_loc, scale=init_scale) for i in range(dim)]
    optimizer.load(None)

    initial_loc = _read_loc(optimizer.save())
    initial_value = objective(initial_loc)

    for _ in range(evaluations):
        point = np.array([parameter.value for parameter in params], dtype=float)
        optimizer.tell(-objective(point))

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
    opt_a = Optimizer()
    opt_a.pop_size = 20
    p1 = opt_a.add("alpha", loc=2.0, scale=1.5)
    p2 = opt_a.add("beta", loc=-1.0, scale=2.0)
    load_result = opt_a.load(None)
    assert load_result == LoadResult(["alpha", "beta"], [])

    for _ in range(37):
        opt_a.tell(-(p1.value**2 + p2.value**2))

    state = opt_a.save()
    assert isinstance(state, dict)
    assert "config" not in state

    opt_b = Optimizer()
    opt_b.pop_size = 20
    opt_b.add("beta", loc=-999.0, scale=0.5)
    opt_b.add("alpha", loc=999.0, scale=0.5)
    load_result = opt_b.load(state)
    assert load_result == LoadResult([], [])
    loaded = opt_b.save()

    assert _read_names(loaded) == _read_names(state)
    assert np.allclose(_read_loc(loaded), _read_loc(state))
    assert np.allclose(_read_scale(loaded), _read_scale(state))


def test_load_reconciles_added_and_removed_parameters() -> None:
    base = Optimizer()
    base.pop_size = 4
    x = base.add("x", loc=2.0, scale=1.5)
    y = base.add("y", loc=-1.0, scale=0.7)
    load_result = base.load(None)
    assert load_result == LoadResult(["x", "y"], [])

    for _ in range(5):
        base.tell(-(x.value**2 + 0.5 * y.value**2))

    base_state = base.save()
    base_loc = _read_loc(base_state)
    base_scale = _read_scale(base_state)
    base_step_size_path = _read_step_size_path(base_state)
    base_batch_z = _read_batch_z(base_state)
    base_batch_x = _read_batch_x(base_state)
    base_results = _read_results(base_state)
    assert any(result is not None for result in _read_results(base_state))

    added = Optimizer()
    added.pop_size = 4
    added.add("x", loc=999.0, scale=9.0)
    added.add("y", loc=999.0, scale=9.0)
    added.add("z", loc=3.0, scale=2.0)
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
    assert np.allclose(_read_batch_x(added_state), np.vstack([base_batch_x, np.full((1, base_batch_x.shape[1]), 3.0)]))
    assert _read_results(added_state) == base_results

    z = next(item for item in added.get_info() if item.name == "z")
    assert z.prior_loc == 3.0
    assert z.prior_scale == 2.0

    added.tell(-1.0)
    added_partial_state = added.save()
    assert any(result is not None for result in _read_results(added_partial_state))
    added_loc = _read_loc(added_state)
    added_scale = _read_scale(added_state)
    added_step_size_path = _read_step_size_path(added_state)
    added_batch_z = _read_batch_z(added_partial_state)
    added_batch_x = _read_batch_x(added_partial_state)
    added_results = _read_results(added_partial_state)

    removed = Optimizer()
    removed.pop_size = 4
    removed.add("x", loc=-999.0, scale=5.0)
    removed.add("y", loc=-999.0, scale=5.0)
    load_result = removed.load(added_partial_state)
    assert load_result == LoadResult([], ["z"])
    removed_state = removed.save()

    assert _read_names(removed_state) == ["x", "y"]
    assert np.allclose(_read_loc(removed_state), added_loc[:2])
    assert np.allclose(_read_scale(removed_state), added_scale[:2, :2])
    assert np.allclose(_read_step_size_path(removed_state), added_step_size_path[:2])
    assert np.allclose(_read_batch_z(removed_state), added_batch_z[:2, :])
    assert np.allclose(_read_batch_x(removed_state), added_batch_x[:2, :])
    assert _read_results(removed_state) == added_results


def test_registration_order_is_lexicographic() -> None:
    first = Optimizer()
    first.pop_size = 18
    first.add("zeta", loc=3.0, scale=1.0)
    first.add("alpha", loc=1.0, scale=2.0)
    first.add("mu", loc=2.0, scale=3.0)
    first.load(None)
    state_first = first.save()

    second = Optimizer()
    second.pop_size = 18
    second.add("mu", loc=2.0, scale=3.0)
    second.add("zeta", loc=3.0, scale=1.0)
    second.add("alpha", loc=1.0, scale=2.0)
    second.load(None)
    state_second = second.save()

    assert _read_names(state_first) == ["alpha", "mu", "zeta"]
    assert _read_names(state_second) == ["alpha", "mu", "zeta"]
    assert np.allclose(_read_loc(state_first), _read_loc(state_second))


def test_get_info_reports_current_state() -> None:
    optimizer = Optimizer()
    optimizer.pop_size = 12
    alpha = optimizer.add("alpha", loc=-2.0, scale=2.0)
    zeta = optimizer.add("zeta", loc=3.0, scale=1.0)
    optimizer.load(None)

    values = optimizer.get_info()
    assert isinstance(values, list)
    assert all(isinstance(item, ParameterInfo) for item in values)
    assert [item.name for item in values] == ["alpha", "zeta"]
    assert [item.value for item in values] == [alpha.value, zeta.value]
    assert [item.loc for item in values] == [-2.0, 3.0]
    assert [item.scale for item in values] == [2.0, 1.0]
    assert [item.prior_loc for item in values] == [-2.0, 3.0]
    assert [item.prior_scale for item in values] == [2.0, 1.0]


def test_context_reuses_mirror_on_repeat() -> None:
    optimizer = Optimizer()
    optimizer.pop_size = 4
    x = optimizer.add("x", loc=0.0, scale=1.0)
    y = optimizer.add("y", loc=0.0, scale=1.0)
    optimizer.load(None)
    context: JSONValue = {"map": "arena", "opponent": {"race": "zerg"}}

    first = np.array([x.value, y.value], dtype=float)
    assert optimizer.set_context(context) is False
    first_report = optimizer.tell(1.0)
    assert first_report == Report(False, False, XNESStatus.OK, False)

    second_report = optimizer.tell(0.0)
    assert second_report == Report(False, False, XNESStatus.OK, False)

    assert optimizer.set_context(context) is True
    mirror = np.array([x.value, y.value], dtype=float)
    assert np.allclose(mirror, -first)
    third_report = optimizer.tell(-1.0)
    assert third_report == Report(False, True, XNESStatus.OK, False)


def test_save_requires_tell_after_context() -> None:
    optimizer = Optimizer()
    optimizer.pop_size = 4
    optimizer.add("x", loc=0.0, scale=1.0)
    optimizer.load(None)
    optimizer.set_context({"job": 1})

    try:
        optimizer.save()
    except RuntimeError as exc:
        assert "tell" in str(exc)
    else:
        raise AssertionError("save() should reject in-flight context state")


def test_set_context_rejects_non_json_serializable_value() -> None:
    optimizer = Optimizer()
    optimizer.pop_size = 4
    optimizer.add("x", loc=0.0, scale=1.0)
    optimizer.load(None)

    try:
        optimizer.set_context(cast(JSONValue, {"bad": object()}))
    except TypeError:
        pass
    else:
        raise AssertionError("set_context() should reject non-JSON-serializable values")


def test_runtime_config_is_not_persisted_or_loaded() -> None:
    opt_a = Optimizer()
    opt_a.pop_size = 14
    opt_a.csa_enabled = False
    opt_a.eta_mu = 0.9
    opt_a.eta_sigma = 0.7
    opt_a.eta_B = 0.2
    param = opt_a.add("x", loc=4.0, scale=2.0)
    opt_a.load(None)
    for _ in range(20):
        opt_a.tell(-(param.value**2))

    state = opt_a.save()
    assert isinstance(state, dict)
    assert "config" not in state

    opt_b = Optimizer()
    opt_b.pop_size = 4
    opt_b.add("x", loc=4.0, scale=2.0)
    opt_b.load(state)
    loaded = opt_b.save()
    assert isinstance(loaded, dict)
    assert "config" not in loaded
    assert opt_b.csa_enabled is None
    assert opt_b.eta_mu is None
    assert opt_b.eta_sigma is None
    assert opt_b.eta_B is None


def test_restart_on_conditioning_failure() -> None:
    optimizer = Optimizer()
    optimizer.pop_size = 10
    x = optimizer.add("x", loc=0.0, scale=1.0)
    y = optimizer.add("y", loc=0.0, scale=1.0)
    optimizer.load(None)

    state = optimizer.save()
    assert isinstance(state, dict)
    state["scale"] = [[1e-10, 0.0], [0.0, 1e10]]

    restored = Optimizer()
    restored.pop_size = 10
    restored.add("x", loc=0.0, scale=1.0)
    restored.add("y", loc=0.0, scale=1.0)
    restored.load(state)

    for _ in range(10):
        restored.tell(-(x.value**2 + y.value**2))

    conditioned = restored.save()
    assert np.allclose(_read_loc(conditioned), np.array([0.0, 0.0]))
    cond = float(np.linalg.cond(_read_scale(conditioned)))
    assert cond < 1e14


def test_save_load_preserves_optimizer_state() -> None:
    direct = Optimizer()
    direct.pop_size = 12
    direct_x = direct.add("x", loc=2.0, scale=1.5)
    direct_y = direct.add("y", loc=-1.0, scale=0.7)
    direct.load(None)

    recreated_state = direct.save()
    for _ in range(40):
        direct_result = -(direct_x.value**2 + 0.5 * direct_y.value**2)
        direct.tell(direct_result)

        recreated = Optimizer()
        recreated.pop_size = 12
        recreated_x = recreated.add("x", loc=2.0, scale=1.5)
        recreated_y = recreated.add("y", loc=-1.0, scale=0.7)
        recreated.load(recreated_state)
        recreated_result = -(recreated_x.value**2 + 0.5 * recreated_y.value**2)
        recreated.tell(recreated_result)
        recreated_state = recreated.save()

    direct_state = direct.save()
    assert _read_names(direct_state) == _read_names(recreated_state)
    assert np.allclose(_read_loc(direct_state), _read_loc(recreated_state))
    assert np.allclose(_read_scale(direct_state), _read_scale(recreated_state))
    assert np.allclose(_read_step_size_path(direct_state), _read_step_size_path(recreated_state))
