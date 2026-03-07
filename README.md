# xnes
Evolutionary black-box optimizer with serialization and dynamic named parameters.

## Overview
- Core algorithm: canonical xNES update with optional CSA step-size adaptation.
- Wrapper: named parameter interface with `add`, `remove`, `tell`, `save`, and `load`.
- Parameter ordering is lexicographical by name, independent of registration order.

## Quickstart
```python
import numpy as np

from xnes import Optimizer

opt = Optimizer(pop_size=32)
coeff_1 = opt.add("coeff_1", loc=2.0, scale=3.0)
coeff_2 = opt.add("coeff_2")

# Optional state restore
# opt.load(state)

for _ in range(500):
    value = coeff_1.value + np.exp(coeff_2.value)
    # `tell` maximizes; minimize `f` via `-f`
    opt.tell(-value**2)

state = opt.save()
```

## Interface
```python
import numpy as np

from dataclasses import dataclass
from collections.abc import Sequence

@dataclass
class Parameter:
    name: str
    value: float

class Optimizer:
    def __init__(
        self,
        pop_size: int | None = None,
        *,
        csa_enabled: bool = True,
        eta_mu: float = 1.0,
        eta_sigma: float = 1.0,
        eta_B: float | None = None,
    ) -> None: ...
    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> Parameter: ...
    def remove(self, name: str) -> None: ...
    def save(self) -> dict[str, object]: ...
    def load(self, state: object) -> None: ...
    def tell(self, result: float | Sequence[float] | np.ndarray) -> bool: ...

class XNES:
    def __init__(
        self,
        x0: np.ndarray,
        sigma0: np.ndarray | float,
        p_sigma: np.ndarray | None = None,
        *,
        csa_enabled: bool = True,
        eta_mu: float = 1.0,
        eta_sigma: float = 1.0,
        eta_B: float | None = None,
    ) -> None: ...
    def ask(self, num_samples: int | None = None, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]: ...
    def tell(self, samples: np.ndarray, ranking: list[int], eps: float = 1e-10) -> bool: ...
```

Public imports come from `xnes`, not `src`.

## Result Ordering
- Scalar results are treated as 1-tuples.
- Sequence results are compared lexicographically.
- Higher tuples are better (maximize semantics).

## Development
- `make fix`: apply Ruff fixes and format.
- `make check`: run Ruff, mypy, and pytest.
