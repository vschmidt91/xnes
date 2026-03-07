"""Public package interface for the xNES optimizer.

Example:
    ```python
    from src import Optimizer

    opt = Optimizer(pop_size=32)
    coeff = opt.add("coeff", loc=1.0, scale=0.5)

    for _ in range(100):
        opt.tell(-(coeff.value - 3.0) ** 2)
    ```
"""

from src.optimizer import JSON, Optimizer, Parameter, Result

__all__ = ["JSON", "Optimizer", "Parameter", "Result"]
