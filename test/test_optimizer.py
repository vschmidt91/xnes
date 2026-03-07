from __future__ import annotations

from collections.abc import Callable

import numpy as np

from xnes import Optimizer


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


def _read_p_sigma(state: object) -> np.ndarray:
    assert isinstance(state, dict)
    p_sigma_json = state["p_sigma"]
    assert isinstance(p_sigma_json, list)
    return np.asarray(p_sigma_json, dtype=float)


def _read_optional_matrix(state: object, key: str) -> np.ndarray | None:
    assert isinstance(state, dict)
    raw = state[key]
    if raw is None:
        return None
    assert isinstance(raw, list)
    return np.asarray(raw, dtype=float)


def _run_function_optimization(
    objective: Callable[[np.ndarray], float],
    *,
    init_loc: float,
    init_scale: float,
    dim: int,
    pop_size: int,
    evaluations: int,
) -> tuple[float, float]:
    optimizer = Optimizer(pop_size=pop_size)
    params = [optimizer.add(f"x{i}", loc=init_loc, scale=init_scale) for i in range(dim)]

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


def test_optimizer_improves_ellipsoid() -> None:
    def ellipsoid(x: np.ndarray) -> float:
        weights = 10.0 ** np.linspace(0.0, 3.0, num=x.size)
        return float(np.sum(weights * x**2))

    initial, final = _run_function_optimization(
        ellipsoid,
        init_loc=2.5,
        init_scale=2.0,
        dim=5,
        pop_size=36,
        evaluations=2400,
    )
    assert final < 0.2 * initial


def test_optimizer_improves_rosenbrock() -> None:
    def rosenbrock(x: np.ndarray) -> float:
        left = 100.0 * (x[1:] - x[:-1] ** 2) ** 2
        right = (1.0 - x[:-1]) ** 2
        return float(np.sum(left + right))

    initial, final = _run_function_optimization(
        rosenbrock,
        init_loc=-2.0,
        init_scale=3.0,
        dim=3,
        pop_size=36,
        evaluations=3500,
    )
    assert final < 0.4 * initial


def test_state_save_load_roundtrip() -> None:
    opt_a = Optimizer(pop_size=20)
    p1 = opt_a.add("alpha", loc=2.0, scale=1.5)
    p2 = opt_a.add("beta", loc=-1.0, scale=2.0)

    for _ in range(37):
        opt_a.tell(-(p1.value**2 + p2.value**2))

    state = opt_a.save()
    assert isinstance(state, dict)
    assert "version" not in state
    assert "priors" not in state
    assert "config" not in state

    opt_b = Optimizer(pop_size=20)
    opt_b.add("beta", loc=-999.0, scale=0.5)
    opt_b.add("alpha", loc=999.0, scale=0.5)
    opt_b.load(state)
    loaded = opt_b.save()

    assert _read_names(loaded) == _read_names(state)
    assert np.allclose(_read_loc(loaded), _read_loc(state))


def test_registration_order_is_lexicographic() -> None:
    first = Optimizer(pop_size=18)
    first.add("zeta", loc=3.0, scale=1.0)
    first.add("alpha", loc=1.0, scale=2.0)
    first.add("mu", loc=2.0, scale=3.0)
    state_first = first.save()

    second = Optimizer(pop_size=18)
    second.add("mu", loc=2.0, scale=3.0)
    second.add("zeta", loc=3.0, scale=1.0)
    second.add("alpha", loc=1.0, scale=2.0)
    state_second = second.save()

    assert _read_names(state_first) == ["alpha", "mu", "zeta"]
    assert _read_names(state_second) == ["alpha", "mu", "zeta"]
    assert np.allclose(_read_loc(state_first), _read_loc(state_second))


def test_get_values_returns_plain_name_to_value_mapping() -> None:
    optimizer = Optimizer(pop_size=12)
    zeta = optimizer.add("zeta", loc=3.0, scale=1.0)
    alpha = optimizer.add("alpha", loc=-2.0, scale=2.0)

    values = optimizer.get_values()
    assert isinstance(values, dict)
    assert list(values) == ["alpha", "zeta"]
    assert values["alpha"] == alpha.value
    assert values["zeta"] == zeta.value


def test_add_remove_between_operations() -> None:
    optimizer = Optimizer(pop_size=16)
    a = optimizer.add("a", loc=1.0, scale=2.0)
    b = optimizer.add("b", loc=-2.0, scale=3.0)

    for _ in range(10):
        optimizer.tell(-(a.value**2 + b.value**2))

    optimizer.remove("b")
    c = optimizer.add("c", loc=0.5, scale=1.0)

    for _ in range(20):
        optimizer.tell(-(a.value**2 + c.value**2))

    state = optimizer.save()
    assert _read_names(state) == ["a", "c"]
    assert _read_loc(state).shape == (2,)


def test_runtime_config_is_not_persisted_or_loaded() -> None:
    opt_a = Optimizer(
        pop_size=14,
        csa_enabled=False,
        eta_mu=0.9,
        eta_sigma=0.7,
        eta_B=0.2,
    )
    param = opt_a.add("x", loc=4.0, scale=2.0)
    for _ in range(20):
        opt_a.tell(-(param.value**2))

    state = opt_a.save()
    assert isinstance(state, dict)
    assert "config" not in state

    opt_b = Optimizer(pop_size=4, csa_enabled=True, eta_mu=1.0, eta_sigma=1.0, eta_B=None)
    opt_b.load(state)
    loaded = opt_b.save()
    assert isinstance(loaded, dict)
    assert "config" not in loaded
    assert opt_b.csa_enabled is True
    assert opt_b.eta_mu == 1.0
    assert opt_b.eta_sigma == 1.0
    assert opt_b.eta_B is None


def test_restart_on_conditioning_failure() -> None:
    optimizer = Optimizer(pop_size=10)
    x = optimizer.add("x", loc=0.0, scale=1.0)
    y = optimizer.add("y", loc=0.0, scale=1.0)

    state = optimizer.save()
    assert isinstance(state, dict)
    state["scale"] = [[1e-10, 0.0], [0.0, 1e10]]
    optimizer.load(state)

    for _ in range(10):
        optimizer.tell(-(x.value**2 + y.value**2))

    conditioned = optimizer.save()
    assert np.allclose(_read_loc(conditioned), np.array([0.0, 0.0]))
    cond = float(np.linalg.cond(_read_scale(conditioned)))
    assert cond < 1e14


def test_optimizer_can_be_recreated_between_iterations() -> None:
    initializer = Optimizer(pop_size=12)
    initializer.add("x", loc=2.0, scale=1.5)
    initializer.add("y", loc=-1.0, scale=0.7)
    state = initializer.save()

    direct = Optimizer(pop_size=12)
    direct.load(state)
    direct_x = direct.add("x")
    direct_y = direct.add("y")

    recreated_state = state
    for _ in range(40):
        direct_result = -(direct_x.value**2 + 0.5 * direct_y.value**2)
        direct.tell(direct_result)

        recreated = Optimizer(pop_size=12)
        recreated.load(recreated_state)
        recreated_x = recreated.add("x")
        recreated_y = recreated.add("y")
        recreated_result = -(recreated_x.value**2 + 0.5 * recreated_y.value**2)
        recreated.tell(recreated_result)
        recreated_state = recreated.save()

    direct_state = direct.save()
    assert _read_names(direct_state) == _read_names(recreated_state)
    assert np.allclose(_read_loc(direct_state), _read_loc(recreated_state))
    assert np.allclose(_read_scale(direct_state), _read_scale(recreated_state))
    assert np.allclose(_read_p_sigma(direct_state), _read_p_sigma(recreated_state))

    direct_batch_z = _read_optional_matrix(direct_state, "batch_z")
    recreated_batch_z = _read_optional_matrix(recreated_state, "batch_z")
    assert (direct_batch_z is None) == (recreated_batch_z is None)
    if direct_batch_z is not None and recreated_batch_z is not None:
        assert np.allclose(direct_batch_z, recreated_batch_z)

    direct_batch_x = _read_optional_matrix(direct_state, "batch_x")
    recreated_batch_x = _read_optional_matrix(recreated_state, "batch_x")
    assert (direct_batch_x is None) == (recreated_batch_x is None)
    if direct_batch_x is not None and recreated_batch_x is not None:
        assert np.allclose(direct_batch_x, recreated_batch_x)

    assert isinstance(direct_state, dict)
    assert isinstance(recreated_state, dict)
    direct_results = np.asarray(direct_state["results"], dtype=float)
    recreated_results = np.asarray(recreated_state["results"], dtype=float)
    assert direct_results.shape == recreated_results.shape
    assert np.allclose(direct_results, recreated_results)
    assert direct_state["rng_state"] == recreated_state["rng_state"]
