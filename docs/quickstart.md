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
    trial = opt.ask(context="validation:shard-0")
    value = trial.params["coeff_1"] + np.exp(trial.params["coeff_2"])
    opt.tell(trial, -value**2)
    state_path.write_text(json.dumps(opt.save()))

opt.set_best()  # switch parameter views to current population mean for testing
```

`tell` uses maximize semantics. To minimize an objective `f(x)`, pass `-f(x)`.

If you resume with a changed parameter set, shared learned state is preserved,
new parameters start from priors, removed parameters are dropped, and the
current unfinished batch is reconciled rather than discarded.
On `load(None)`, all currently registered parameters are reported as added.

When switching between training and testing:
- Training: call `ask`, evaluate `trial.params`, then call `tell(trial, result)`.
- Testing: call `set_best()` and evaluate without `tell`.
- Resume training: `load(state)` saved before `set_best()`.
