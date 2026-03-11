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
        coeff: Annotated[float, Parameter.above(lower=0.0, loc=1.0, scale=0.5)]
        ratio: Annotated[float, Parameter.between(lower=0.0, upper=1.0)]

    opt = Optimizer(Params)
    opt.pop_size = 32
    opt.load(None)

    for _ in range(100):
        sample = opt.ask()
        value = (sample.params.coeff - 3.0) ** 2 + (sample.params.ratio - 0.25) ** 2
        opt.tell(sample, -value)

    best = opt.ask_best()
    print(best.params.coeff, best.params.ratio)
    ```
"""

from .optimizer import LoadResult, Optimizer, Sample, TellResult
from .schema import Parameter
from .xnes import XNES, XNESStatus

__all__ = ["Optimizer", "Parameter", "Sample", "TellResult", "LoadResult", "XNES", "XNESStatus"]
