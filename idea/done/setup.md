# Initial Repository Setup

This project will hold already written code to make it maintainable and reusable.
The optimizer is a tried-and-tested implementation of resources/exponential_natural_evolution_strategies.pdf
The only addition is a dynamic step size adaptation based on path accumulation inspired by CMA-ES.
Existing code is separated into core algorithm and wrapper.
Wrapper is an extended ask-and-tell interface to allow dynamically adding and removing parameters between iterations

## Tasks

1. rewrite the code from resources/ into src/
2. add linting with ruff+mypy and a Makefile with `fix` and `check` commands
3. add tests described below
4. extend README.md

## Interface

```py

@dataclass
class Parameter:
    name: str
    value: float
    
type JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
type Result = float | Sequence[float]

class Optimizer:
    def __init__(self, pop_size: int | None = None) -> None:
    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> Parameter:
    def save(self) -> JSON:
    def load(self, state: JSON) -> None:
    def tell(self, result: Result):

```

User workflow is

```
opt = Optimizer()
param1 = opt.add("coeff_1", loc=2.0, scale=3.0)
param2 = opt.add("coeff_2")
opt.load(state)

...

result = f(param1.value, np.exp(param2.value))

...

opt.tell(result)


```

## Notes

- rename ParameterOptimizer to Optimizer
- ignore MatchupParameterProvider and _ContextualParameterOptimizer
- multiobjective optimization is not in scope yet
- omit the parameter transformations (softplus, sigmoid, quadratic...)
- just plain multinormal distribution for now
- priority is short functional starting point
- keep the algorithm largely untouched but simplify the wrapper as much as possible

## Test Cases

- optimizer can find optimum of analytical test functions (sphere, ellipsoid, rosenbrock)
- optimizer state can be saved and loaded
- parameters can be registered in any order
- optimizer can add new parameters and removes between operations

## Questions

1. would it be better if loading happens before parameter registering?
2. not sure if all orderings of save/load/register should be supported or if there is a simpler
3. is there a better type for the returned `Parameter`? maybe `Callable[[], float]`?
4. what is the best way to handle result telling? ideally we support all comparable types without having to store them.

# === codex ===

## Writedown (March 6, 2026)
- `Optimizer.tell` should accept `Result = float | Sequence[float]` and normalize to a tuple for ranking.
- Ranking rule is lexicographical tuple order.
  Scalar objectives are treated as 1-tuples and vector objectives as explicit tuples `(r1, r2, ..., rn)`.
- Registration order is not semantically significant; name ordering is lexicographical as AGENTS notes.
- `Parameter` should remain a small dataclass with at least `name` and `value`; do not switch to callable form.
- Save/load policy:
  - Primary supported flow:
    `add()` parameters then `load(state)` or `load(state)` on empty optimizer to restore full state.
  - Do not support undefined mixed ordering flows unless explicitly implemented behind a strict validation rule.

## Execution (March 6, 2026)
- Implemented in `src/`:
  - `XNES` core update logic with mirrored sampling and dynamic step-size adaptation.
  - `Optimizer` wrapper with `add`, `remove`, `save`, `load`, `tell`.
  - Lexicographical name ordering independent of registration order.
  - `Result` normalization to tuple and lexicographical ranking (maximize semantics).
- Added tooling:
  - `pyproject.toml` with `ruff`, `mypy`, `pytest` config.
  - `Makefile` with `fix` and `check`.
- Added tests for:
  - sphere/ellipsoid/rosenbrock improvement
  - save/load
  - registration order invariance
  - add/remove between operations
- Extended `README.md` with interface, semantics, and usage.

## Contradictions Resolved
- `Optimizer.__init__` mismatch is now resolved in this file as `pop_size: int | None = None`.
- "multiobjective not in scope" vs `Result = Sequence[float]` is implemented as lexicographical ordered tuple
  ranking, not Pareto optimization.

## Contradictions Remaining (Need User Decision)
- None at the moment.

## Execution Addendum (March 6, 2026)
- Implemented optional canonical xNES step-size update with `csa_enabled=False`.
- Kept current CSA-inspired behavior as default with `csa_enabled=True`.
- Exposed learning rates:
  - `eta_mu`
  - `eta_sigma`
  - `eta_B`
- Persisted optimizer config and diagnostics inside `save()` / `load(state)`.
- Added safeguards and restart handling:
  - sigma bounds (`min_sigma`, `max_sigma`)
  - condition-number bound (`max_condition`)
  - `restart_on_failure`
  - diagnostics via `diagnostics()`
- Added invariance/stability regression tests:
  - monotonic fitness transformation invariance
  - linear-coordinate invariance with stress values (`mu` around `1e10`, scale around `1e-10`)

## Refactor Note (March 7, 2026)
- Removed optimizer diagnostics fields/method and their persistence payload.
- Kept runtime restart checks but moved thresholds to internal constants:
  - `_MIN_SIGMA`
  - `_MAX_SIGMA`
  - `_MAX_CONDITION`
- Added `rng_state` to optimizer save/load state so optimizer recreation between iterations is exact.
- Added test coverage for recreate-per-evaluation workflow and state equivalence.

## Flexibility Note (March 7, 2026)
- Removed `version` and `priors` from persisted optimizer state.
- `load(state)` is now permissive/best-effort and avoids strict validation exceptions.
- Config precedence is now explicit:
  - constructor-specified `csa_enabled`, `eta_mu`, `eta_sigma`, `eta_B` always override loaded config values
  - when these are not specified in constructor, load may adopt values from state `config`

## Simplification Note (March 7, 2026)
- Removed persisted `config` from optimizer state entirely.
- Removed constructor-intent lock/override tracking.
- `load(state)` now assumes state is produced by this implementation and applies direct minimal parsing.
