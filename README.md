![Leitwerk](docs/logo.png)

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with mathematics</em>
</p>

---

`leitwerk` is an evolutionary strategy with file persistence and schema reconciliation.
Define the parameters, set up a loop in the training environment and let it learn.
It is aimed at noisy, expensive black-box settings like simulators, bots and game AI with a few handcrafted parameters.

## Features

- **Easy** : Start from existing values without much setup
- **Persistent** : State lives in human-readable JSON
- **Dynamic** : Keep developing without losing progress
- **Efficient** : Canonical xNES [^1] implementation on par with `cma` [^2] on BBOB [^4]

## Installation

Requires Python 3.11+

```sh
poetry install --all-extras
```

## Quickstart

### Step 1 - Define the parameters

`leitwerk` uses annotated dataclasses to define the initial starting point:

```py
@dataclass
class MyParams:
    param1: Annotated[float, Parameter(loc=1.2, scale=3.4)] # loc: center guess, scale: spread/uncertainty
    army_priority: Annotated[float, Parameter(loc=1, min=0)]   # optional lower bounds
```

Samples will be provided with the proper type for IntelliSense and type checking.
Alternatively, you can use plain dictionaries, see the [example below](#example-1-function-minimization)

> [!TIP]
> Use nested schemas to group the parameters into sensible blocks.
> `leitwerk` handles full tree structures.

### Step 2 - Set up the Loop

On Startup, create the optimizer and restore state if there is one:

```py
opt = Optimizer(Params, pop_size=10)

# optional:
with open("params.json") as f:
    schema_diff = opt.load(json.load(f))
```

`schema_diff` reports about parameters that were added/removed/changed compared to the old state.

Run any number of training cycles:

```py
for _ in range(32):
    params = opt.ask()
    result = ...
    opt.tell(result)
```

Batching and result aggregation happen under the hood.

Persist state:

```py
state = opt.save()
```

> [!WARNING]
> `ask`/`tell` cycles are only supported inside a `load`/`save` block.
> Don't restore the optimizer and tell it about old samples!

---

## Example 1 - Function Minimization

If persistence is not required, `leitwerk` can be used as a standard `ask`/`tell` function optimizer:

```py
from leitwerk import Optimizer, Parameter

def f(x1, x2):
    return (x1 - 1)**2 + (x2 - 1)**2  # minimum is at (1, 1)

opt = Optimizer({"x1": Parameter(), "x2": Parameter()}, minimize=True)

for _ in range(100):
    x = opt.ask()
    opt.tell(f(**x))

print(opt.ask_best())
# {'x1': 1.007115753775713, 'x2': 0.9922700335131514}
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

## Context Matching (optional)

You can help `leitwerk` to make the sampling a bit more efficient by sorting the trial runs into categories.
These are arbitrary strings that depend on the problem you are solving.
Only context equality matters - the actual content is not parsed.

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
> - `context=self.enemy_race` : `pop_size >= 8`
> - `context=self.opponent_id` : `pop_size >= 2 * division_size`

## Optimizer Details (advanced)

- Population is modelled as unbounded multivariate gaussian distribution
- Bounded parameters are implemented via smooth bijective mappings:
  1. `min=None`, `max=None`: identity
  2. `min=a`, `max=None`: scale + soft-plus + offset
  3. `min=None`, `max=b`: scale + soft-plus + mirror + offset
  4. `min=a`, `max=b`: scale/offset + sigmoid + scale/offset
- Only ranking of results matters, not numerical values
  - fitness shaping is used internally
  - makes optimization invariant under monotonic objective transformations
- Objective values can be sequences/tuples
  - ranking is lexicographic
  - this implements simple tie-breaking, not pareto/multi-objective optimization
- Learning rates use the canonical decomposition [^1] and can be adapted:
  - `eta_mu=1.0`
  - `eta_sigma=1.0`
  - `eta_B=1.0` (will be combined with dimensionality factor)

## License

This project is licensed under the terms of the MIT license.

[^1]: https://people.idsia.ch/~tom/publications/xnes.pdf
[^2]: https://github.com/CMA-ES/pycma
[^3]: https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies
[^4]: https://numbbo.github.io/coco/testsuites/bbob
