"""Public package interface for the schema-first xNES optimizer wrapper.

Schemas may be flat dataclasses or nested dataclass trees whose optimized
leaves are declared as `Annotated[float, Parameter(...)]`.

Example:
    ```python
    from dataclasses import dataclass
    from typing import Annotated

    from xnes import Optimizer, Parameter

    @dataclass(frozen=True)
    class Params:
        coeff: Annotated[float, Parameter(loc=1.0, scale=0.5, min=0.0)]
        ratio: Annotated[float, Parameter(min=0.0, max=1.0)]

    opt = Optimizer(Params, pop_size=32, minimize=True)

    for _ in range(100):
        trial, params = opt.ask()
        value = (params.coeff - 3.0) ** 2 + (params.ratio - 0.25) ** 2
        opt.tell(trial, value)

    best = opt.ask_best()
    print(best.coeff, best.ratio)
    ```
"""

from .optimizer import Optimizer, TellResult, Trial
from .schema import Parameter, SchemaDiff
from .xnes import XNES, XNESStatus

__all__ = ["Optimizer", "Parameter", "SchemaDiff", "TellResult", "Trial", "XNES", "XNESStatus"]
