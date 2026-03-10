# Quickstart

```python
import json
from pathlib import Path

import numpy as np

from xnes import Optimizer

state_path = Path("optimizer-state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else None

opt = Optimizer()
opt.pop_size = 32
opt.add("coeff_1", loc=2.0, scale=3.0)
opt.add("coeff_2")
load_result = opt.load(state)

for _ in range(500):
    params = opt.ask(context="validation:shard-0")
    value = params["coeff_1"] + np.exp(params["coeff_2"])
    opt.tell(params, -value**2)
    state_path.write_text(json.dumps(opt.save()))

best = opt.ask_best()
print(best["coeff_1"], best["coeff_2"])
```

`tell` uses maximize semantics. To minimize an objective `f(x)`, pass `-f(x)`.

If you resume with a changed parameter set, shared learned state is preserved,
new parameters start from priors, removed parameters are dropped, and the
current unfinished batch is reconciled rather than discarded.
On `load(None)`, all currently registered parameters are reported as added.

Training loop: call `ask`, evaluate `params[...]`, then call `tell(params, result)`.

For deterministic inference, call `ask_best()` after `load(...)`. It returns a
context-free snapshot of the current means and cannot be passed to `tell()`.
