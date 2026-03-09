"""Public package interface for the xNES optimizer.

Example:
    ```python
    from xnes import Optimizer

    opt = Optimizer()
    opt.pop_size = 32
    coeff = opt.add("coeff", loc=1.0, scale=0.5)
    opt.load(None)

    for _ in range(100):
        opt.tell(-(coeff.value - 3.0) ** 2)
    ```
"""

from .optimizer import Optimizer, Parameter, ParameterInfo, Report
from .xnes import XNES, XNESStatus

__all__ = ["Optimizer", "Parameter", "ParameterInfo", "Report", "XNES", "XNESStatus"]
