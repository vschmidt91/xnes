# Quickstart

```python
import numpy as np

from xnes import Optimizer

opt = Optimizer(pop_size=32)
coeff_1 = opt.add("coeff_1", loc=2.0, scale=3.0)
coeff_2 = opt.add("coeff_2")

for _ in range(500):
    value = coeff_1.value + np.exp(coeff_2.value)
    opt.tell(-value**2)

state = opt.save()
```

`tell` uses maximize semantics. To minimize an objective `f(x)`, pass `-f(x)`.
