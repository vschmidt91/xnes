<p align="center">
  <img src="docs/logo.png" alt="Leitwerk">
</p>

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with mathematics</em>
</p>

`leitwerk` is an evolutionary optimizer with typed paramers and persistence.

- **Simple** : Start from existing values without much setup
- **Persistent** : The optimizer lives inside a JSON file
- **Dynamic** : Keep developing without losing progress

---

Requires: Python 3.11+

For a minimal setup, run:

```sh
$ pip install .
```

or for a full setup with tests/linting, notebook support and `burnysc2`:

```sh
$ pip install -e .[dev,docs,benchmark]
```

## Example 1

At base level, `leitwerk` is an ask-and-tell blackbox optimizer.

```py
from leitwerk import Optimizer, Parameter

opt = Optimizer({"a": Parameter(), "b": Parameter()}, minimize=True)
for _ in range(500):
    x = opt.ask()
    opt.tell((x["a"] - 1) ** 2 + (x["b"] - 2) ** 2)
    
print(opt.mean)
# {'a': 1.0000000001945673, 'b': 2.0000000008038628}
```

## Example 2 - Starcraft II Bot

For a more complete example with typed schema and persistence, run:

```sh
$ python examples/train_sc2_bot.py
```

This trains a simple probe rush with two parameters against the hardest built-in AI.
The optimizer state is stored in `./data/params.json` and is somewhat human-readable, so have a look.

---

# User Guide

To use `leitwerk` in an existing setup, you usually have to split the loop and use `load()` / `save()` for persistence (if necessary).

## 1. Preflight Check

Define your parameter schema as an annotated dataclass:

```py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from leitwerk import Optimizer, Parameter

@dataclass
class CombatParams:
    attack_threshold: Annotated[float, Parameter()]                     # standard normal N(0, 1)
    skirmish_range: Annotated[float, Parameter(loc=5.0, scale=1.5)]     # or mean +/- std

@dataclass
class MacroParams:
    army_priority: Annotated[float, Parameter(min=0, max=1)]            # or lower/upper bounds
    worker_limit: Annotated[float, Parameter(loc=66, scale=10, min=12)] # or a mix

@dataclass
class Params:
    combat: CombatParams                                                # nested dataclasses for grouping (optional)
    macro: MacroParams
```

This tells the optimizer how to seed the initial population:

- `loc` is the initial best guess (prior median)
- `scale` is the initial spread/uncertainty  (prior standard deviation in latent space)
- `min` and `max` are hard bounds (modeled as soft-plus/sigmoid activations)
- most combinations work, see [Optimizer details](#optimization-details)

### 2. Takeoff

Create the optimizer with your schema and old state (if present):

```py
opt = Optimizer(Params)

params_file = Path("params.json")
if params_file.exists():
    with params_file.open() as f:
        schema_diff = opt.load(json.load(f))
        
params = opt.ask()
```

`Optimizer(Params)` takes a few optional arguments:

- `minimize` : switch to minimization mode (default: `False`)
- `population_size` : number of samples per batch (default: `4 + int(3 * log(num_params))`)
- `eta_mu`, `eta_sigma`, `eta_B` : xNES learning rates [^1] (default: `1.0`)

> [!NOTE]
> These are runtime settings, so your code is the single source of truth for config.
> The persisted state is strictly for optimization progress.

Parameters are typed for proper IntelliSense and type-checking:

```py
print(params.combat)
# CombatParams(attack_threshold=-0.35633796355282366, skirmish_range=3.427746197055024)
print(params.macro)
# MacroParams(army_priority=0.8041272429029714, worker_limit=74.90326494070536)
```

Alternatively, use dictionaries and string-based access, see the [example above](#example-1).


 When `Optimizer.load()` detects a schema change, state is reconciled per parameter:
 - Parameters are identified by flattened name
 - Changes to `min`/`max` trigger a reset
 - Changes to `loc`/`scale` do not, but will be used for future resets
 - `load()` returns a `schema_diff` which reports parameters that were added/removed/changed/unchanged.

> [!WARNING]
> Call ordering is strict:
> - after `ask()`, the next mutating call must be `tell()`
> - `save()` is only valid while idle, i.e. after `tell()` and before the next `ask()`
> - `load()` cancels any pending sample and replaces the current in-memory state
> - `load()` replaces the current in-memory progress with the snapshot you pass in

> [!TIP]
> When the _meaning_ of a parameter changes, the optimizer won't know.
> It might take a long time to adapt if the new optimum is far away.
> Consider renaming the parameter and providing a new `loc`/ `scale` for a targeted reset.

### 3. Landing

When you see the result, encode it as one or more numbers:

```py
result = +1 if win else 0
opt.tell(result)

# better:
# opt.tell([result, calc_heuristic()])

with params_file.open("w") as f:
    json.dump(opt.save(), f)
```

- Results are normalized to `tuple[float, ...]` and ranked lexicographically
- This is simple tie-breaking, not actual multi-objective optimization (TBD)
- Only ranking matters, not numerical objective values


> [!WARNING]
> Make sure the sign of the results and `Optimizer.minimize` match up!


> [!TIP]
> The right objective function is key.
> - Binary win/loss makes sense to start with, but is not very informative on its own
> - Smooth gradients help: army value, income, cost-effectiveness, ...
> - Changing the objective later on can work, but at your own discretion
> - Consider multiple optimizers/objectives for different parameters

---

## Local Development

### Commands

- `make check` : linting + tests
- `make fix`: auto-format
- `make docs`: build docs
- `make docs-serve`: serve docs locally

---

## Context Matching (optional)

You can help `leitwerk` to make the sampling a bit more efficient by sorting
evaluation runs into categories. Context values may be strings or any
JSON-compatible value.
Only context equality after canonical JSON normalization matters - the actual
content is not parsed.

Examples:

- `opt.ask(context=self.enemy_race.name)`
- `opt.ask(context=self.opponent_id)`
- `opt.ask(context=self.game_info.map_name)`

If context is provided, `leitwerk` uses it for mirror sampling: [^3]

- Samples are generated in pairs: for every search direction `d`, also try `-d`
- Ideally, pairs are evaluated in the same context
- This helps to keept the gradient estimate centered/unbiased


> [!NOTE]
> This still evolves a single set of parameters.
> For actual per-matchup evolution, use multiple Optimizers.

> [!TIP]
> Rule of thumb: the population should be large enough to hold two of each unique context.
>
> For AIArena authors:
> - `context=self.enemy_race` : `population_size >= 8`
> - `context=self.opponent_id` : `population_size >= 2 * division_size`

## Optimization Details

- Search distribution: `z ~ N(mu, Sigma)` in latent space, with full covariance `Sigma`
- User parameters are smooth bijections of latent coordinates:
  - one-sided bounds: soft-plus map activation `(min, inf)` or `(-inf, max)`
  - two-sided bounds: sigmoid activation `(min, max)`
- `loc` is interpreted in user space; if omitted, the latent center `0` is used
- `scale` always lives in latent space, i.e. it sets search spread before applying the bound mapping
- Fitness shaping makes the search invariant under strictly monotone transformations of the objective
- Tuple results are ordered lexicographically
  - This is tie-breaking by secondary keys
  - Pareto optimization TBD
- xNES hyperparameters:
  - `population_size` is essentially abstracted away, finetune if you want
  - `eta_mu = 1.0`
  - `eta_sigma = 1.0`
  - `eta_B = 1.0` with the canonical dimension factor [^1]

## License

This project is licensed under the terms of the MIT license.

[^1]: https://people.idsia.ch/~tom/publications/xnes.pdf
[^2]: https://github.com/CMA-ES/pycma
[^3]: https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies
[^4]: https://numbbo.github.io/coco/testsuites/bbob

