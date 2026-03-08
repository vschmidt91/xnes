# xnes
Evolutionary black-box optimizer with serialization and dynamic named parameters.

## Overview
- Core algorithm: canonical xNES update with optional CSA step-size adaptation.
- Wrapper: named parameter interface with `add`, `load`, optional `set_context`, `tell`, and `save`.
- Parameter ordering is lexicographical by name, independent of registration order.

## Call Flow

The wrapper is intentionally strict. The supported flow is:

1. create `Optimizer`
2. register all parameters with `add(...)`
3. call `load(None)` for a fresh run, or `load(state)` to resume
4. optionally call `set_context(context_id)`
5. read the current `Parameter.value` values
6. evaluate once
7. call `tell(result)`
8. call `save()`

Important constraints:

- `add()` is setup-only and must happen before `load()`
- `save()` must happen after `tell()`, not after `set_context()` and before `tell()`
- if `set_context()` is never called, the optimizer uses its default batch order
- `set_context()` hashes the provided object immediately and does not store the original context object

## Quickstart
```python
import numpy as np

from xnes import Optimizer

opt = Optimizer(pop_size=32)
coeff_1 = opt.add("coeff_1", loc=2.0, scale=3.0)
coeff_2 = opt.add("coeff_2")

# Fresh run:
opt.load(None)
# Or restore:
# opt.load(state)

for _ in range(500):
    # Optional: context-aware mirror routing for this evaluation.
    # opt.set_context(opponent_id)
    value = coeff_1.value + np.exp(coeff_2.value)
    # `tell` maximizes; minimize `f` via `-f`
    opt.tell(-value**2)
    state = opt.save()
```

## Interface
```python
import numpy as np

from dataclasses import dataclass
from collections.abc import Hashable, Sequence

@dataclass
class Parameter:
    name: str
    value: float

@dataclass
class ParameterInfo:
    name: str
    value: float
    loc: float
    scale: float
    prior_loc: float
    prior_scale: float

class XNESStatus(Enum): ...

@dataclass
class Report:
    completed_batch: bool
    matched_context: bool
    status: XNESStatus
    restarted: bool

class Optimizer:
    def __init__(self, pop_size: int | None = None) -> None: ...
    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> Parameter: ...
    def get_info(self) -> list[ParameterInfo]: ...
    def set_best(self) -> None: ...
    def save(self) -> dict[str, object]: ...
    def load(self, state: object) -> None: ...
    def set_context(self, context: Hashable) -> bool: ...
    def tell(self, result: float | Sequence[float] | np.ndarray) -> Report: ...

class XNES:
    def __init__(
        self,
        x0: np.ndarray,
        sigma0: np.ndarray | float,
        p_sigma: np.ndarray | None = None,
    ) -> None: ...
    def ask(self, num_samples: int | None = None, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]: ...
    def tell(self, samples: np.ndarray, ranking: list[int], eps: float = 1e-10) -> XNESStatus: ...
```

Runtime tuning is done via attributes:

```python
opt = Optimizer(pop_size=32)
opt.csa_enabled = False
opt.eta_mu = 0.9
opt.eta_sigma = 0.7
opt.eta_B = 0.2  # 20% of the default shape-learning step
```

Leaving `Optimizer.csa_enabled`, `Optimizer.eta_mu`, `Optimizer.eta_sigma`, or
`Optimizer.eta_B` as `None` keeps the defaults defined by `XNES`. A bare
`XNES(...)` instance starts with `csa_enabled = True`, `eta_mu = 1.0`,
`eta_sigma = 1.0`, and `eta_B = 1.0`, which applies the built-in shape-learning
rate `0.6 * (3 + log(dim)) / (dim * sqrt(dim))` for `dim > 0`. Smaller
`eta_B` values damp that rate multiplicatively.

## Result Ordering
- Scalar results are treated as 1-tuples.
- Sequence results are compared lexicographically (no multiobjective/pareto optimization)
- Higher tuples are better (maximize semantics).

## Training vs Testing
- During training, call `load(None)` or `load(state)`, optionally call `set_context(...)`, evaluate the current `Parameter.value` values, then pass the result to `tell` and persist with `save()`.
- For testing/inference, call `set_best()` to overwrite `Parameter.value` with the current population mean.
- If you want to resume training after testing, save state before `set_best()` and later `load` that state.

## Development
- `make fix`: apply Ruff fixes and format.
- `make check`: run Ruff, mypy, and pytest.
