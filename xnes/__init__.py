"""Public package interface for the xNES optimizer.

Example:
    ```python
    from xnes import Optimizer

    opt = Optimizer()
    opt.pop_size = 32
    opt.add("coeff", loc=1.0, scale=0.5)
    opt.load(None)

    for _ in range(100):
        params = opt.ask()
        opt.tell(params, -(params["coeff"] - 3.0) ** 2)

    best = opt.ask_best()
    print(best["coeff"])
    ```
"""

from .optimizer import LoadResult, Optimizer, Parameters, TellResult
from .xnes import XNES, XNESStatus

__all__ = ["Optimizer", "Parameters", "TellResult", "LoadResult", "XNES", "XNESStatus"]
