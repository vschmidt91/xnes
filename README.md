<p align="center">
  <img src="docs/logo.png" alt="Leitwerk">
</p>

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with stochastic optimization.</em>
</p>

`leitwerk` is a schema-first evolutionary optimizer for long-running evaluation loops.

It gives you:

- typed parameter objects in the code
- JSON checkpoints you can inspect, diff, and resume
- schema reconciliation, so you can keep iterating without throwing away learned state

---

## Install

Requires: Python 3.11+

Library only:

```sh
pip install .
```

Development setup with the SC2 example, tests, docs, plotting, and benchmark tooling:

```sh
pip install -e .[dev,docs,benchmark]
```

## Core API

```py
from leitwerk import Optimizer, Parameter

opt = Optimizer({"a": Parameter(), "b": Parameter()}, minimize=True)
for _ in range(500):
    x = opt.ask()
    opt.tell((x["a"] - 1) ** 2 + (x["b"] - 2) ** 2)

print(opt.mean)
# {'a': 1.0000000001945673, 'b': 2.0000000008038628}
```

## Quickstart: Run the Example Bot

```sh
python examples/train_sc2_bot.py
```

The example in [examples/train_sc2_bot.py](examples/train_sc2_bot.py):

- runs a simple probe rush against the built-in AI
- samples one parameter set per game
- scores each game with win/loss and a secondary efficiency signal
- saves state after every completed evaluation

It writes:

- `data/params.json`: optimizer state
- `data/history.json`: flattened result history and parameter values
- `data/plot.png`: rolling plots for outcome, efficiency, and learned parameters

For SC2 bot authors, reading this file is probably enough to get started.

---

## Integration Guide

The intended loop is for one evaluation: load > ask > tell > save

### 1. Define a Typed Parameter Schema

```py
from dataclasses import dataclass
from typing import Annotated

from leitwerk import Parameter


@dataclass
class Params:
    attack_threshold: Annotated[float, Parameter()]
    worker_limit: Annotated[float, Parameter(loc=66, scale=10, min=12)]
```

`Parameter(...)` defines how each value is initialized:

- `loc`: initial best guess
- `scale`: initial spread
- `min` and `max`: optional bounds

> [!TIP]
> Nested dataclasses are supported if you want to group parameters together.

### 2. Create or Resume the Optimizer

```py
import json
from pathlib import Path

from leitwerk import Optimizer


opt = Optimizer(Params)

params_file = Path("data/params.json")
if params_file.exists():
    schema_diff = opt.load(json.loads(params_file.read_text()))
```

`schema_diff` tells you which flattened parameter names were added, removed or changed.

`Optimizer(Params, ...)` takes additional arguments:

- `minimize`: rank lower results as better
- `population_size`: number of evaluations per batch
- `eta_mu`, `eta_sigma`, `eta_B`: xNES learning rates

> [!NOTE]
> Optimizer arguments are runtime settings, they are not restored on `load`.

### 3. Ask For a Sample

```py
context = {"enemy_race": "Protoss"}     # optional context
params = opt.ask(context)
print(params)
# Params(attack_threshold=-0.8312413125179872, worker_limit=59.407519238244)
```

For deterministic sampling, use:

```py
params = opt.mean
print(params)
# Params(attack_threshold=0.0, worker_limit=66.0)
```

### 4. Tell the Result and Save

```py
result = +1 if win else 0
report = opt.tell(result)

# better:
# report = opt.tell((result, calc_heuristic()))

params_file.parent.mkdir(parents=True, exist_ok=True)
params_file.write_text(json.dumps(opt.save(), indent=2))
```

Result handling is simple:

- `opt.tell(x)` uses a single scalar objective
- `opt.tell((a, b, c))` ranks results lexicographically
- the first item is the main objective, later items act as tie-breakers
- only relative ranking matters, not absolute numeric values

`report` tells you:

- whether the batch completed
- whether context matching found a mirrored sample
- which xNES status was returned
- whether the optimizer restarted with a fresh distribution.

---

## What Happens When the Schema Changes

This is one of the main reasons to use `leitwerk` in an active project.

- parameters are identified by flattened names and reconciled individually
- renaming a parameter resets that parameter
- changing `min` or `max` resets that parameter
- changing `loc` or `scale` defines future resets, but does not trigger one

In practice, you can keep developing, adding knobs, and updating priors without resetting the whole learned state every time.


## Choosing Objectives

For efficient training, the objective often matters more than the optimizer.

- put win rate first if that is the real target
- add tie-breakers such as army value, income, or cost efficiency
- this is NOT multi-objective / pareto optimization
- keep objective semantics stable for long term training
- use separate optimizers if parameters actually belong to separate objectives

## Context Matching

`opt.ask(context=...)` lets the scheduler match mirrored samples in the same context when possible.

Useful contexts for SC2 bots:

- `self.enemy_race.name`
- `self.opponent_id`
- `self.game_info.map_name`

The context value is matched by exact equality.
Non-string values are serialized to canonical JSON strings before matching and persistence.

Why this helps:

- samples are generated in mirrored pairs
- matching both sides of a pair to the same context keeps the gradient estimate centered

This is a stabilizing trick, not actual context-aware optimization.

## More Reading

- [docs/quickstart.md](docs/quickstart.md)
- [docs/reference/api.md](docs/reference/api.md)

## Developer Commands

- `make fix`: auto-format
- `make check`: lint and test
- `make docs`: build docs
- `make docs-serve`: serve docs locally

## References

- [xNES paper](https://people.idsia.ch/~tom/publications/xnes.pdf)
- [Mirrored sampling paper](https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies)

## License

This project is licensed under the terms of the MIT license.
