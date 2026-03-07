# xnes
Evolutionary black-box optimizer with serialization and dynamic named parameters.

## Overview
- Core algorithm: canonical xNES update with optional CSA step-size adaptation.
- Wrapper: named parameter interface with `add`, `remove`, `tell`, `save`, and `load`.
- Parameter ordering is lexicographical by name, independent of registration order.

## Quickstart
```python
import numpy as np

from src import Optimizer

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
from dataclasses import dataclass
from collections.abc import Sequence

@dataclass
class Parameter:
    name: str
    value: float

type JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
type Result = float | Sequence[float]

class Optimizer:
    def __init__(
        self,
        pop_size: int | None = None,
        *,
        csa_enabled: bool = True,
        eta_mu: float = 1.0,
        eta_sigma: float = 1.0,
        eta_B: float | None = None,
        min_sigma: float = 1e-20,
        max_sigma: float = 1e20,
        max_condition: float = 1e14,
    ) -> None: ...
    def add(self, name: str, loc: float = 0.0, scale: float = 1.0) -> Parameter: ...
    def remove(self, name: str) -> None: ...
    def save(self) -> JSON: ...
    def load(self, state: JSON) -> None: ...
    def tell(self, result: Result) -> bool: ...
    def diagnostics(self) -> dict[str, JSON]: ...
```

## Result Ordering
- Scalar results are treated as 1-tuples.
- Sequence results are compared lexicographically.
- Higher tuples are better (maximize semantics).

## Development
- `make fix`: apply Ruff fixes and format.
- `make check`: run Ruff, mypy, and pytest.
