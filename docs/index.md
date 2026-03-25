# leitwerk

`leitwerk` is a schema-based evolutionary optimizer for long-running black-box
evaluation loops.

Use it when you want to tune scalar parameters, checkpoint progress, and keep
training across schema changes.
Core ideas:

- runtime parameters are exposed directly as dataclasses or nested dicts
- training is single-flight: `ask()`, evaluate once, then `tell()`
- `OptimizerSession` adds JSON persistence on top of `Optimizer`
- `mean` gives the current deterministic center without sampling
