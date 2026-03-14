<p align="center">
  <img src="docs/logo.png" alt="Leitwerk">
</p>

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with evolution</em>
</p>

`leitwerk` is an evolutionary optimizer with typed paramers and serialization.

- **Simple** : Start from existing values without much setup
- **Persistent** : The state lives inside a human-readable JSON representation
- **Dynamic** : Keep developing without losing optimization progress

---

Requires: Python 3.11+

For a minimal setup, run:

```sh
$ pip install .
```

For a full setup with tests/linting, notebook support and `burnysc2`, run:

```sh
$ pip install -e .[dev,docs,benchmark]
```

## Example 1

At base level, `leitwerk` is an ask-and-tell blackbox optimizer:

```py
from leitwerk import Optimizer, Parameter

opt = Optimizer({"a": Parameter(), "b": Parameter()}, minimize=True)
for _ in range(500):
    x = opt.ask()
    opt.tell((x["a"] - 1) ** 2 + (x["b"] - 2) ** 2)
```

```sh
>>> opt.mean
{'a': 1.0000000001945673, 'b': 2.0000000008038628}
```

## Example 2 - Starcraft II Bot

For a more complete example with typed parameters and persistence, run:

```sh
$ python examples/train_sc2_bot.py
```

The script trains a very simple probe rush with two parameters against the built-in AI.
It takes a few hours and don't expect fancy micro, this is purely proof-of-concept.
The optimizer state and results are persisted in files:

- `data/params.json` : optimizer state in somewhat human-readable format, have a look
- `data/plot.png` : winrate, K/D and parameters over time
- `data/history.json` : helper file for result history

---

# Integration Guide

To use this in an existing setup, the optimization loop will probably need to be split up into parts.
For serialization, use `Optimizer.load(state: JSON)` and `Optimizer.save() -> JSON` and persist to disk as necessary.

## 1. Preflight Check

Define your parameter schema as an annotated dataclass:

```py
from dataclasses import dataclass
from typing import Annotated
from leitwerk import Parameter

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
    combat: CombatParams                                                # nested structure (optional)
    macro: MacroParams
```

Alternatively, create a dictionary schema for string-based access, as in the [minimal example above](#example-1).

The schema tells the optimizer how to seed the population:

- `loc` is the initial best guess (prior median)
- `scale` is the initial spread/uncertainty  (prior standard deviation in latent space)
- `min` and `max` are hard bounds (modeled as soft-plus/sigmoid activations)
- most combinations work

> [!TIP]
> The parser understands full tree structures, so you can group your parameters for easy handover to other components.

## 2. Takeoff

Create the optimizer with your schema and restore previous state, if present:

```py
import json
from pathlib import Path
from leitwerk import Optimizer

opt = Optimizer(Params)

params_file = Path("params.json")
if params_file.exists():
    with params_file.open() as f:
        schema_diff = opt.load(json.load(f))
```

`Optimizer(Params, ...)` takes several optional arguments:

- `minimize` : switch to minimization mode (default: `False`)
- `population_size` : number of samples per batch (default: depends on the number of parameters)
- `eta_mu`, `eta_sigma`, `eta_B` : xNES [^1] learning rates (default: `1.0`)

> [!NOTE]
> These are runtime settings, they can change across runs without restriction.
> Persisted state is strictly for optimization progress.

When `Optimizer.load(state)` encounters a schema change, state is reconciled per parameter:
- Parameters are identified by flattened name, so renaming triggers a reset
- Changes to `min`/`max` trigger a reset
- Changes to `loc`/`scale` don't, but will be stored for future resets
- `schema_diff` reports which parameters were added, removed or changed

> [!TIP]
> You can "manually" reset a parameter by renaming it.
> When you do, consider providing a new `loc`/ `scale`.
> This can make sense if the meaning of the parameter changed considerably.

Request a sample and for this run:

```py
params = opt.ask()
```

For the demerministic population mean (test mode, tournament play, ...), use:

```py
params = opt.mean
```

Parameters are typed for proper IntelliSense and type-checking:

```sh
>>> params.combat
CombatParams(attack_threshold=-0.35633796355282366, skirmish_range=3.427746197055024)

>>> params.macro
MacroParams(army_priority=0.8041272429029714, worker_limit=74.90326494070536)
```

## 3. Landing

When you see the result, encode it as one or more numbers:

```py
result = +1 if win else 0
opt.tell(result)

# better:
# opt.tell([result, calc_heuristic()])

with params_file.open("w") as f:
    json.dump(opt.save(), f)
```

> [!WARNING]
> Sampling is strictly sequential, so you must tell a result before asking again (or reload).
> Both functions raise errors when they detect an ask / tell cycle overlap.

When every sample of the population is evaluated, the optimizer updates its distribution and samples a new batch under the hood.

- `opt.tell(result)` returns a small report for monitoring
- Results are normalized to `tuple[float, ...]` and ranked lexicographically
- This is simple tie-breaking, not actual multi-objective optimization (TBD)
- Fitness Shaping: only ranking matters, not numerical objective values [^1]

> [!NOTE]
> Make sure the sign of the results and `Optimizer.minimize` match your intent


The right objective function is key:
- Binary win/loss makes sense to prioritize, but is not very informative on its own
- Smooth gradients can help: army value, income, cost-effectiveness, ...
- Changing the objective later on _may_ work, but at least the current batch will have mixed signals
- Consider multiple optimizers/objectives for different parameters

---

## Developer Commands

- `make check` : linting + tests
- `make fix`: auto-format
- `make docs`: build docs
- `make docs-serve`: serve docs locally

## Context Matching (optional)

You can help the sample selection to be a bit more efficient by providing JSON-serializable context.

Examples:

- `self.enemy_race.name`
- `self.opponent_id`
- `self.game_info.map_name`

The actual content is not parsed, this is purely equality-based.
When provided, `opt.ask(context)` uses it for mirror sample matching: [^3]

- Samples are generated in pairs: for every search direction `d`, also try `-d`
- Ideally, pairs are evaluated in the same context
- This helps to keept the gradient estimate centered/unbiased

> [!NOTE]
> This is a small stabilizing effect, it still evolves a single set of parameters.
> For actual context-dependent evolution, use multiple optimizers.

> [!TIP]
> Rule of thumb: the population should be large enough to hold two of each unique context.
> 
> Concretely for AIArena authors:
> - `context = self.enemy_race` : `population_size` >= 8 (6 if you delay sampling until first scout)
> - `context = self.opponent_id` : `population_size` >= 2 * division_size

[^1]: https://people.idsia.ch/~tom/publications/xnes.pdf
[^2]: https://github.com/CMA-ES/pycma
[^3]: https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies
[^4]: https://numbbo.github.io/coco/testsuites/bbob

## License

This project is licensed under the terms of the MIT license.