# Integration Guide

This page shows the typical pattern for wiring `leitwerk` into your own evaluation loop.

The intended call flow is sequential:

![Flowchart](flowchart.png)

---

## 1. Define a parameter schema

*~ Preflight Check ~*

```py
from dataclasses import dataclass
from typing import Annotated

from leitwerk import Parameter


@dataclass
class MyParams:
    attack_threshold: Annotated[float, Parameter()]
    worker_limit: Annotated[float, Parameter(mean=66, scale=10, min=12)]
```

`Parameter(...)` defines the prior distribution for each value:

- `mean`: initial best guess
- `scale`: initial spread
- `min` and `max`: optional bounds

!!! tip
    Dictionary schemas are supported if you prefer string-based lookup:
    ```py
    MyParams = {"a": Parameter(), "b": Parameter()}
    ```

!!! tip
    Use nested schemas to group parameters together.
    `leitwerk` understands tree structures.

## 2. Create the optimizer

*~ Engine Ignition ~*

For automatic JSON persistence, use the `OptimizerSession` wrapper:

```py
from leitwerk import OptimizerSession

opt = OptimizerSession("params.json", MyParams)
```

If the session file already exists, it is loaded and reconciled automatically, see [What happens when the schema changes?](faq.md#what-happens-when-the-schema-changes)

```pycon
>>> opt.restored
True
>>> opt.schema_diff
SchemaDiff(added=[], removed=[], changed=[], unchanged=['attack_threshold', 'worker_limit'])
```

For an in-memory run, use `Optimizer` directly:

```py
from leitwerk import Optimizer

opt = Optimizer(MyParams)
```

If you are using other means of persistence, you can optionally restore it from state as well:

```py
schema_diff = opt.load(state)
```

Both `Optimizer` and `OptimizerSession` accept `batch_size` and `seed` directly:

```py
from leitwerk import Optimizer

opt = Optimizer(MyParams, batch_size=10, seed=1234)
```

Available constructor arguments:

- `batch_size`: number of samples per batch / optimizer step
- `seed`: root seed used to deterministically derive future batches

## 3. Sample a candidate

*~ Liftoff ~*

```py
params = opt.ask()
```

```pycon
>>> params
MyParams(attack_threshold=-0.8312413125179872, worker_limit=59.407519238244)
```

For deterministic evaluation, use the optimized mean instead of sampling:

```pycon
>>> opt.mean
MyParams(attack_threshold=0.0, worker_limit=66.0)
```

Optionally, provide a JSON-valued context for the current sample:

```py
context = {"opponent_race": "Protoss"}  # optional
params = opt.ask(context)
```

!!! info
    [What context should I provide?](faq.md#what-context-should-i-provide)

## 4. Report the result

*~ Landing ~*

After evaluation, encode the outcome as a scalar or tuple:

```py
result = +1 if win else 0
report = opt.tell(result)
```

Binary results carry little information and can slow down training.
It is usually better to provide additional tie-breakers:

```py
report = opt.tell((result, calc_heuristic()))
```

Result handling:

- `opt.tell((a, b, c))` ranks results lexicographically with higher = better
- the first item is the main objective, later items are tie-breakers
- maximize by default, flip the sign for loss objectives


!!! info
    [How should I choose the objective?](faq.md#how-should-i-choose-the-objective)

```pycon
>>> report
OptimizerReport(completed_batch=False, matched_context=False, status=<XNESStatus.OK: 1>, restarted=False)
```

When using `OptimizerSession`, the JSON file is updated atomically on `tell`.
An `Optimizer` can be serialized with:

```py
state = opt.save()
```
