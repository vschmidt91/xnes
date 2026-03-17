# Quickstart

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np

from leitwerk import Optimizer, OptimizerSettings, Parameter


@dataclass(frozen=True)
class Params:
    coeff_1: Annotated[float, Parameter(loc=2.0, scale=3.0, min=0.0)]
    coeff_2: Annotated[float, Parameter()]


state_path = Path("optimizer-state.json")
opt = Optimizer(Params, OptimizerSettings(population_size=32))
if state_path.exists():
    state = json.loads(state_path.read_text())
    load_result = opt.load(state)

for _ in range(500):
    params = opt.ask(context="validation:shard-0")
    value = params.coeff_1 + np.exp(params.coeff_2)
    opt.tell(-value**2)
    state_path.write_text(json.dumps(opt.save()))

mean = opt.mean
print(mean.coeff_1, mean.coeff_2)
```

`tell` uses maximize semantics. To minimize an objective `f(x)`, pass `-f(x)`.

Current schema requirements:

- the schema must be a dataclass
- optimized fields must be `Annotated[float, Parameter(...)]`
- persisted state is ordered lexicographically by field name

If you resume with a changed schema, shared learned state is preserved, new
fields start from parameter defaults, removed fields are dropped, and the current
unfinished batch is reconciled rather than discarded.

Training loop: call `ask`, evaluate `params.field`, then call `tell(result)`.

`ask()` / `tell()` is single-flight: at most one sample may be pending.
Calling `ask()` twice in a row raises, and `tell()` without a pending `ask()`
raises.

Persistence edge cases:

- `save()` may be called while an `ask()` is pending, but it snapshots only
  committed state. The pending reservation is not serialized.
- `load()` may be called while an `ask()` is pending; it cancels that pending
  sample and replaces the current state.
- if you `save()` after `ask()` and later `load()` that snapshot, the in-flight
  sample is gone and its later `tell()` will fail with `No pending ask`
- `load()` may intentionally discard unsaved local progress from earlier
  `tell()` calls
- for exact restart semantics, checkpoint after `tell()`, not after `ask()`

For deterministic inference, read `mean`. If you want the means from a
saved run rather than a fresh optimizer, call `load(...)` first.

