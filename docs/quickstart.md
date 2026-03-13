# Quickstart

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np

from leitwerk import Optimizer, Parameter


@dataclass(frozen=True)
class Params:
    coeff_1: Annotated[float, Parameter(loc=2.0, scale=3.0, min=0.0)]
    coeff_2: Annotated[float, Parameter()]


state_path = Path("optimizer-state.json")
opt = Optimizer(Params, population_size=32)
if state_path.exists():
    state = json.loads(state_path.read_text())
    load_result = opt.load(state)

for _ in range(500):
    params = opt.ask(context="validation:shard-0")
    value = params.coeff_1 + np.exp(params.coeff_2)
    opt.tell(-value**2)
    state_path.write_text(json.dumps(opt.save()))

best = opt.ask_best()
print(best.coeff_1, best.coeff_2)
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

`ask()` / `tell()` is strictly sequential. After `ask()` returns, the next
mutating call must be `tell()`. `save()` and `load()` are only supported at
idle boundaries, i.e. when no `ask()` is pending.

`load()` at an idle boundary may intentionally discard unsaved local progress
from earlier `tell()` calls.

For deterministic inference, call `ask_best()`. If you want the means from a
saved run rather than a fresh optimizer, call `load(...)` first.

