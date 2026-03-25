<p align="center">
  <img src="docs/logo.png" alt="Leitwerk">
</p>

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with stochastic optimization.</em>
</p>

`leitwerk` is a schema-based evolutionary optimizer for long-running evaluation loops.

It gives you:

- typed parameters with priors in your code
- JSON checkpoints you can inspect, diff, and resume
- schema reconciliation, so you can keep developing and preserve training progress

---

## Install

Requires: Python >=3.11,<3.14

Library only:

```sh
pip install .
```

Development setup with the SC2 example, tests, docs, plotting, and benchmark tooling:

```sh
pip install -e .[dev,docs,benchmark]
```

## Quickstart: Run the Example SC2 Bot

```sh
python examples/train_sc2_bot.py
```

The example in [examples/train_sc2_bot.py](examples/train_sc2_bot.py):

- runs a simple probe rush against the built-in AI
- samples one parameter set per game
- scores each game with win/loss and a secondary efficiency signal
- saves state after every completed evaluation

It writes progress into:

- `data/params.json`: optimizer state
- `data/history.json`: result history and parameter values
- `data/plot.png`: rolling plots for outcome, efficiency, and learned parameters

---

## Integration Guide

### 1. Define a Parameter Schema

```py
from dataclasses import dataclass
from typing import Annotated

from leitwerk import Parameter


@dataclass
class MyParams:
    attack_threshold: Annotated[float, Parameter()]
    worker_limit: Annotated[float, Parameter(mean=66, scale=10, min=12)]
```

`Parameter(...)` defines how each value is initialized, i.e. the prior distribution:

- `mean`: initial best guess
- `scale`: initial spread
- `min` and `max`: optional bounds

> [!TIP]
> Nested schemas are supported if you want to group parameters together.

### 2. Create the Optimizer

For file persistence, use the `OptimizerSession` wrapper:

```py
from leitwerk import OptimizerSession, OptimizerSettings

settings = OptimizerSettings(population_size=10)
opt = OptimizerSession("params.json", MyParams, settings)
```

Optimizer settings are optional, the available arguments are:

- `population_size`: number of evaluations per batch
- `seed`: for reproducible sampling
- `minimize`: rank lower results as better
- `eta_mean`, `eta_scale_global`, `eta_scale_shape`: xNES learning rates

> [!NOTE]
> These settings are persisted as fallback, runtime values override.

If `params.json` already exists, the optimizer will [reconcile](#what-happens-when-the-schema-changes):

```pycon
>>> opt.schema_diff
SchemaDiff(added=[], removed=[], changed=[], unchanged=['attack_threshold', 'worker_limit'])
```

### 3. Sample

```py
context = {"opponent_race": "Protoss"}  # optional
params = opt.ask(context)
```

```pycon
>>> params
MyParams(attack_threshold=-0.8312413125179872, worker_limit=59.407519238244)
```

For deterministic evaluation, use the distribution mean:

```pycon
>>> opt.mean
MyParams(attack_threshold=0.0, worker_limit=66.0)
```

### 4. Evaluate and Tell the Result

Encode the objective as one or more numbers:

```py
result = +1 if win else 0
report = opt.tell(result)   # saves to params.json atomically

# better:
# report = opt.tell((result, calc_heuristic()))
```

```pycon
>>> report
OptimizerReport(completed_batch=False, matched_context=False, status=<XNESStatus.OK: 1>, restarted=False)
```

Result handling is simple:

- `opt.tell(x)` uses a single scalar objective
- `opt.tell((a, b, c))` ranks results lexicographically
- the first item is the main objective, later items act as tie-breakers
- only relative ranking matters, not absolute numeric values

---

## Minimal Example

The core API of `leitwerk` is an ask-and-tell black-box optimizer:

```py
from leitwerk import Optimizer, OptimizerSettings, Parameter

opt = Optimizer({"a": Parameter(), "b": Parameter()}, OptimizerSettings(minimize=True))
for _ in range(500):
    x = opt.ask()
    opt.tell((x["a"] - 1) ** 2 + (x["b"] - 2) ** 2)
```

```pycon
>>> opt.mean
{'a': 1.0000000001945673, 'b': 2.0000000008038628}
```

## What Happens When the Schema Changes

This is the main reason to use `leitwerk` - it handles the changes so you can keep iterating.

- parameters are identified by flattened names
- renaming a parameter resets that parameter
- changing `min` or `max` resets that parameter
- changing `mean` or `scale` defines the new reset target, but does not trigger one

In practice, this means you can add and remove parameters without resetting the whole state.

## Choosing Objectives

For efficient training, the objective often matters more than the optimizer.

- put win rate first if that is the real target
- add tie-breakers such as army value, income, or cost efficiency
- keep objective semantics stable for long-term training
- use multiple optimizers if parameters actually belong to separate objectives

This is not multi-objective / Pareto optimization.

## Context Matching

`opt.ask(context=...)` lets the scheduler match mirrored samples in the same context when possible.

Useful contexts for SC2 bots include:

- opponent race
- own race (for random bots)
- map name
- opponent id

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
