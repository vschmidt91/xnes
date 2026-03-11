# Quickstart

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np

from xnes import Optimizer, Prior


@dataclass(frozen=True)
class Params:
    coeff_1: Annotated[float, Prior(mean=2.0, sigma=3.0)]
    coeff_2: Annotated[float, Prior()]


state_path = Path("optimizer-state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else None

opt = Optimizer(Params)
opt.pop_size = 32
load_result = opt.load(state)

for _ in range(500):
    sample = opt.ask(context="validation:shard-0")
    params = sample.params
    value = params.coeff_1 + np.exp(params.coeff_2)
    opt.tell(sample, -value**2)
    state_path.write_text(json.dumps(opt.save()))

best = opt.ask_best()
print(best.params.coeff_1, best.params.coeff_2)
```

`tell` uses maximize semantics. To minimize an objective `f(x)`, pass `-f(x)`.

Current schema requirements:

- the schema must be a dataclass
- optimized fields must be `Annotated[float, Prior(...)]`
- persisted state is ordered lexicographically by field name

If you resume with a changed schema, shared learned state is preserved, new
fields start from priors, removed fields are dropped, and the current
unfinished batch is reconciled rather than discarded. On `load(None)`, all
current schema fields are reported as added.

Training loop: call `ask`, evaluate `sample.params.field`, then call
`tell(sample, result)`.

For deterministic inference, call `ask_best()` after `load(...)`. It returns a
context-free snapshot of the current means with `sample_id=None` and cannot be
passed to `tell()`.
