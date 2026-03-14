![Leitwerk](docs/logo.png)

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with mathematics</em>
</p>

---

`leitwerk` is an evolutionary optimizer that you can strap onto your self play loop.
Handcraft some parameters, set up the training and let it fly.

## Features

- **Easy** : Start from existing values without much setup
- **Persistent** : State lives in human-readable JSON
- **Dynamic** : Keep developing without losing progress
- **Efficient** : Canonical xNES [^1] implementation, benchmarked against `cma` [^2] on BBOB [^4]

Requires Python 3.11+ and not much else.

## API

```py
class Parameter:
    def __init__(self, loc=0.0, scale=1.0, min=None, max=None):
      
class Optimizer[T]:
    def __init__(self, Schema, ...):
    def load(self, state: JSON) -> SchemaDiff:
    def ask(self, context: JSON = None) -> T:
    def tell(self, result: float | Sequence[float]) -> TellResult:
    def save(self) -> JSON:
```

## Example

If persistence is not required, `leitwerk` is a standard `ask`/`tell` function optimizer:

```py
from leitwerk import Optimizer, Parameter

def f(x1, x2):
    return (x1 - 1)**2 + (x2 - 1)**2  # minimum is at (1, 1)

opt = Optimizer({"x1": Parameter(), "x2": Parameter()}, minimize=True)

for _ in range(100):
    x = opt.ask()
    if opt.tell(f(**x)).restarted:

print(opt.ask_best())
# {'x1': 1.007115753775713, 'x2': 0.9922700335131514}
```

# Usage

## 1. Parameter Schema

**Preflight Check** : Describe your tuning knobs as an annotated dataclass:

```py
@dataclass
class Params:
    army_priority: Annotated[float, Parameter(loc=3.0, scale=0.5)]      # mean + std
    skirmish_range: Annotated[float, Parameter(min=5, max=10)]          # or lower/upper bounds
    worker_target: Annotated[float, Parameter(loc=50, scale=10, min=1)] # or mixed
```

This is the type you will receive as samples, which plays nice with IntelliSense and type-checks.
Alternatively, use plain dictionaries, see the [example below](#example-1-function-minimization)

- `loc` and `scale` are your best guess and spread/uncertainty, used for initial population and resets.
- `min` and `max` are hard limits
- most combinations work, see [Optimizer details](#optimization-details)

> [!TIP]
> Use nested schemas to group the parameters into blocks, it understands tree structures.

## 2. Configuration

**Engine Ignition** : Start the optimizer with the schema and old state if there is one:

```py
def on_start(self):
    self.optimizer = Optimizer(Params)
    with open("./data/params.json") as f:
        schema_diff = self.optimizer.load(json.load(f))
    self.params = self.optimizer.ask()
```

> [!NOTE]
> Schema changes can be reconciled a degree:
> - Parameters are identified by flattened names
> - When `min`/`max` change, the Parameter is reset using `loc` and `scale`
> - `SchemaDiff` is a report of added/removed/changed parameters

> [!TIP]
> When you change how the parameter is used, `leitwerk` cannot know. It might adapt anyway - or consider renaming it.

## 3. Objective Function

**Liftoff** : When you can see the result, encode it as one or more numbers:

```py
def on_end(self, result):
    # your evaluation logic
    elo = {Result.Victory: +4, Result.Tie: 0, Result.Defeat: -4}[result]
    heuristic = self.state.score.total_damage_dealt_life / max(1, self.state.score.total_damage_taken_life)
    result = self.opt.tell((elo, heuristic))
    with open("./data/params.json", "w") as f:
        json.dump(opt.save(), f)
```

This will select towards wins and use a heuristic as secondary ranking.


> [!IMPORTANT]
> The objective function is your secret sauce.
> - Binary loss is fine, but slow
> - Unit Counts? Army Value? Income?
> - If the objective changes drastically, consider a reset
> - Result tuples are compared lexicographically with higher=better (enable minimization with `minimize=True`)

That's it. `leitwerk` is ready to crunch the numbers while you do other things.
The state is somewhat human-readable, so have a look.

---

```

## Example 2 - Starcraft II Bot

This example is a worker rush bot with very simple attack/retreat logic and two parameters.
It tunes itself to beat the hardest built-in AI in about a hundred games.

Manual changes could do this much easier - this is just a proof of concept.

```py
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from loguru import logger
from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot, Computer

from leitwerk import Optimizer, Parameter
from sc2.bot_ai import BotAI
from sc2.data import Result, Race, Difficulty


DATA_PATH = Path("./data")
PARAMS_FILE = DATA_PATH / "params.json"


@dataclass
class BotParams:
    attack_threshold: Annotated[float, Parameter(.5, min=0, max=1)]
    retreat_threshold: Annotated[float, Parameter(.5, min=0, max=1)]


class LearningBot(BotAI):

    async def on_start(self):
        # restore state from disk
        self.optimizer = Optimizer(BotParams)
        DATA_PATH.mkdir(exist_ok=True)
        if PARAMS_FILE.exists():
            with PARAMS_FILE.open() as f:
                state = json.load(f)
            diff = self.optimizer.load(state)
            logger.info(diff)
        context = self.enemy_race.name  # optional: matchup-based mirror sampling
        self.params = self.optimizer.ask(context)
        logger.info(self.params)

    async def on_step(self, iteration):
        mineral_patch = self.mineral_field.closest_to(self.start_location)
        for worker in self.workers:
            if worker.shield_percentage > self.params.attack_threshold:
                if self.enemy_structures:
                    worker.attack(self.enemy_structures.random.position)
                else:
                    worker.attack(self.enemy_start_locations[0])
            elif worker.shield_health_percentage < self.params.retreat_threshold:
                worker.gather(mineral_patch)
        if self.supply_used == 0:
            await self.client.debug_kill_unit(self.structures)

    async def on_end(self, game_result: Result) -> None:
        # primary objective: win
        win_loss = {
            Result.Victory: +1,
            Result.Tie: 0,
            Result.Defeat: -1,
        }[game_result]
        # secondary objective: be cost-effective
        efficiency = self.state.score.total_damage_dealt_life / max(1, self.state.score.total_damage_taken_life)
        score = (win_loss, efficiency)
        logger.info(score)
        tell_result = self.optimizer.tell(score)
        logger.info(tell_result)
        state = self.optimizer.save()
        with PARAMS_FILE.open("w") as f:
            json.dump(state, f, indent=2)


def main():
    while True:
        run_game(
            maps.get("TorchesAIE_v4"),
            [Bot(Race.Protoss, LearningBot()), Computer(Race.Protoss, Difficulty.CheatInsane)],
            realtime=False,
        )


if __name__ == "__main__":
    main()
```

---

## Local Install

Requires Python 3.11+

```sh
poetry install --all-extras
```

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
- Only ranks matter, not objective magnitudes
  - fitness shaping makes the search invariant under strictly monotone transformations of the objective
- Tuple results are ordered lexicographically
  - This is tie-breaking by secondary keys
  - Pareto optimization TBD
- xNES hyperparameters:
  - `population_size` is essentially abstracted away, but finetuned if you want
  - `eta_mu = 1.0`
  - `eta_sigma = 1.0`
  - `eta_B = 1.0` with the canonical dimension factor [^1]

## License

This project is licensed under the terms of the MIT license.

[^1]: https://people.idsia.ch/~tom/publications/xnes.pdf
[^2]: https://github.com/CMA-ES/pycma
[^3]: https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies
[^4]: https://numbbo.github.io/coco/testsuites/bbob

