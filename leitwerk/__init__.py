"""Public package interface for the schema-based xNES optimizer wrapper.

Schemas may be dataclass trees whose optimized leaves are declared as
`Annotated[float, Parameter(...)]`, or nested mappings whose leaves are
`Parameter(...)` values.

Example:
    ```python
    from dataclasses import dataclass
    from typing import Annotated

    from leitwerk import Optimizer, Parameter

    @dataclass(frozen=True)
    class Params:
        coeff: Annotated[float, Parameter(mean=1.0, scale=0.5, min=0.0)]
        ratio: Annotated[float, Parameter(min=0.0, max=1.0)]

    opt = Optimizer(Params, batch_size=32)

    for _ in range(100):
        params = opt.ask()
        loss = (params.coeff - 3.0) ** 2 + (params.ratio - 0.25) ** 2
        opt.tell(-loss)

    mean = opt.mean
    print(mean.coeff, mean.ratio)
    ```
"""

from .optimizer import Optimizer, OptimizerReport
from .schema import Parameter, SchemaDiff
from .session import OptimizerSession
from .xnes import XNES, XNESStatus

__all__ = [
    "Optimizer",
    "OptimizerSession",
    "Parameter",
    "SchemaDiff",
    "OptimizerReport",
    "XNES",
    "XNESStatus",
]
