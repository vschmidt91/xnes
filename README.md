![Leitwerk](docs/logo.png)

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with the power of evolution</em>
</p>

---

`leitwerk` is a covariance adaptation optimizer with schema reconciliation.
Define the parameters, set up a training loop and let it learn.

It is aimed at noisy, expensive black-box settings:

- Bots
- Game AIs
- Simulations
- Any parameterized system with outputs to be optimized

## Features

- **Easy**: Start from existing values without much setup
- **Dynamic**: Keep developing without losing progress
- **Serializable**: Optimizer state lives in human-readable JSON
- **Efficient** Custom implementation of xNES [^1], on par with CMA-ES [^2] on the BBOB [^4]

## Usage

There are two ways to define parameters in `leitwerk`.
This is mainly syntax, the optimization behaves exactly the same.

### Option 1 - `dict[str, Parameter]`

```py
PARAMS = {
  "param1": Parameter(1.2, scale=3.4),
  "param2": Parameter(min=0, max=1)
}
```

Easy setup and string-based access.

### Option 2 - Annotated `dataclass`

```py
@dataclass
class Params:
    param1: Annotated[float, Parameter(1.2, scale=3.4)]
    param2: Annotated[float, Parameter(min=0, max=1)]
```

Samples will be typed as `Params`. Good for IntelliSense and type checking.


> [!TIP]
> Use nested schemas to group the parameters logically.
> `leitwerk` handles full tree structures - just don't mix dictionaries and dataclasses.

> [!TIP]
> Schemas change are reconciled in `Optimizer.load(state)`.
> It reports a `SchemaDiff` and leaves unchanged parameters intact.

### Evolutionary Cycle

Initialize the Optimizer and (optionally) restore state:

```py
opt = Optimizer(Params, pop_size=10)
# schema_diff = opt.load(state)
```

Run one or more training cycles:

```py
for _ in range(10):
    trial, params = opt.ask()
    result = ...
    opt.tell(trial, result)
```

Persist state:

```py
state = opt.save()
```

---

## Example 1 - Function Minimization

```py
from leitwerk import Optimizer, Parameter

def f(x, y):
    return (x - 1)**2 + (y - 1)**2  # minimum is at (1, 1)

opt = Optimizer({"x": Parameter(), "y": Parameter()}, minimize=True)

for _ in range(100):
    trial, params = opt.ask()
    opt.tell(trial, f(**params))

print(opt.ask_best())
# {'x': 1.007115753775713, 'y': 0.9922700335131514}
```

## Example 2 - Starcraft II Bot

This example is a worker rush bot with very simple combat logic and two parameters.
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
        self.trial, self.params = self.optimizer.ask(context)
        logger.info(self.trial)
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
        tell_result = self.optimizer.tell(self.trial, score)
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

If context is provided, `leitwerk` uses context-matched mirror sampling: [^3]

- Samples are generated in pairs: for every search direction `d`, also try `-d`
- Ideally, pairs are evaluated in the same context
- This helps to keept the gradient estimate centered/unbiased

The number of unique contexts is linked to the ideal population size.

> [!NOTE]
> This still evolves a single set of parameters.
> For actual per-matchup evolution, create multiple Optimizers.

> [!TIP]
> For AIArena authors:
> - Random bots: consider including `self.race.name` as well
> - if matching on opponent ID: use `pop_size = 2 * division_size`

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
[^2]: https://en.wikipedia.org/wiki/CMA-ES
[^3]: https://www.researchgate.net/publication/266087889_Mirrored_Orthogonal_Sampling_with_Pairwise_Selection_in_Evolution_Strategies
[^4]: https://numbbo.github.io/coco/testsuites/bbob