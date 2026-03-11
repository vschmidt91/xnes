"""Public package interface for the schema-first xNES optimizer wrapper.

Schemas may be flat dataclasses or nested dataclass trees whose optimized
leaves are declared as `Annotated[float, Prior(...)]`.

Example:
    ```python
    from dataclasses import dataclass
    from typing import Annotated

    from xnes import Optimizer, Prior

    @dataclass(frozen=True)
    class Params:
        coeff: Annotated[float, Prior(mean=1.0, sigma=0.5)]
        bias: Annotated[float, Prior()]

    opt = Optimizer(Params)
    opt.pop_size = 32
    opt.load(None)

    for _ in range(100):
        sample = opt.ask()
        value = (sample.params.coeff - 3.0) ** 2 + sample.params.bias**2
        opt.tell(sample, -value)

    best = opt.ask_best()
    print(best.params.coeff, best.params.bias)
    ```
"""

from .optimizer import LoadResult, Optimizer, Sample, TellResult
from .schema import Prior
from .xnes import XNES, XNESStatus

__all__ = ["Optimizer", "Prior", "Sample", "TellResult", "LoadResult", "XNES", "XNESStatus"]
