# Integration Guide

This page shows the typical pattern for wiring `leitwerk` into your own
evaluation loop.

## 1. Define a parameter schema

```py
from dataclasses import dataclass
from typing import Annotated

from leitwerk import Parameter


@dataclass(frozen=True)
class MyParams:
    attack_threshold: Annotated[float, Parameter()]
    worker_limit: Annotated[float, Parameter(mean=66, scale=10, min=12)]
```

`Parameter(...)` defines the prior for each scalar:

- `mean`: initial best guess
- `scale`: initial spread
- `min` and `max`: optional bounds

Nested dataclasses are supported if you want to group parameters.

## 2. Create the optimizer

Use `Optimizer` for in-memory runs and `OptimizerSession` for automatic JSON
persistence.

```py
from leitwerk import OptimizerSession, OptimizerSettings

settings = OptimizerSettings(population_size=10)
opt = OptimizerSession("params.json", MyParams, settings)
```

Available settings:

- `population_size`
- `seed`
- `minimize`
- `eta_mean`
- `eta_scale_global`
- `eta_scale_shape`

All settings are optional. Omit fields you do not want to override.

If the session file already exists, it is loaded automatically and reconciled
against the current schema.

```pycon
>>> opt.restored
True
>>> opt.schema_diff
SchemaDiff(added=[], removed=[], changed=[], unchanged=['attack_threshold', 'worker_limit'])
```

## 3. Sample one candidate

```py
context = {"opponent_race": "Protoss"}  # optional
params = opt.ask(context)
```

```pycon
>>> params
MyParams(attack_threshold=-0.8312413125179872, worker_limit=59.407519238244)
```

For deterministic evaluation, use the current mean instead of sampling:

```pycon
>>> opt.mean
MyParams(attack_threshold=0.0, worker_limit=66.0)
```

Training is single-flight: call `ask()`, evaluate once, then call `tell()`.
A second `ask()` before `tell()` raises, and `tell()` without a pending `ask()`
raises.

## 4. Report the result

You can report either a scalar score or a tuple of scores.

```py
result = +1 if win else 0
report = opt.tell(result)

# or use lexicographic tie-breakers
report = opt.tell((result, calc_heuristic()))
```

```pycon
>>> report
OptimizerReport(completed_batch=False, matched_context=False, status=<XNESStatus.OK: 1>, restarted=False)
```

Result handling:

- `opt.tell(x)` uses a single scalar objective
- `opt.tell((a, b, c))` ranks results lexicographically
- the first item is the main objective, later items are tie-breakers
- only relative ranking matters, not absolute numeric values

If your natural objective is a loss, either negate it yourself or set
`OptimizerSettings(minimize=True)`.

## 5. Persist and resume

If you use `Optimizer`, call `save()` and `load()` yourself. If you use
`OptimizerSession`, `tell()` persists the updated committed state automatically
and `flush()` lets you force a write.

Persistence rules:

- `save()` snapshots committed state only; a pending `ask()` is not serialized
- `load()` replaces the current state and cancels any pending `ask()`
- if you save after `ask()` and later load that snapshot, the in-flight sample is gone
- for exact restart semantics, checkpoint after `tell()`, not after `ask()`

## 6. Use context when it matters

`ask(context=...)` lets mirrored samples land in the same environment when
possible, which helps keep the gradient estimate centered in stateful systems.

Useful contexts include:

- opponent race
- map name
- evaluation shard
- opponent id

Context values are matched by exact equality after JSON normalization.

For schema changes, objective design, and optimizer background, see the
[FAQ](faq.md).
