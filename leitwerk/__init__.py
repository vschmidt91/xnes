"""Public package interface for the schema-first xNES optimizer wrapper.

Schemas may be dataclass trees whose optimized leaves are declared as
`Annotated[float, Parameter(...)]`, or nested mappings whose leaves are
`Parameter(...)` values.

Example:
    ```python
    from dataclasses import dataclass
    from typing import Annotated

    from leitwerk import Optimizer, OptimizerSettings, Parameter

    @dataclass(frozen=True)
    class Params:
        coeff: Annotated[float, Parameter(loc=1.0, scale=0.5, min=0.0)]
        ratio: Annotated[float, Parameter(min=0.0, max=1.0)]

    opt = Optimizer(
        Params,
        OptimizerSettings(population_size=32, minimize=True),
    )

    for _ in range(100):
        params = opt.ask()
        value = (params.coeff - 3.0) ** 2 + (params.ratio - 0.25) ** 2
        opt.tell(value)

    mean = opt.mean
    print(mean.coeff, mean.ratio)
    ```
"""

from .optimizer import Optimizer, OptimizerReport, OptimizerSettings
from .schema import Parameter, SchemaDiff
from .session import OptimizerSession
from .xnes import XNES, XNESStatus

__all__ = [
    "Optimizer",
    "OptimizerSession",
    "OptimizerSettings",
    "Parameter",
    "SchemaDiff",
    "OptimizerReport",
    "XNES",
    "XNESStatus",
]
