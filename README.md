![Leitwerk](docs/logo.png)

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with mathematics</em>
</p>

---

`leitwerk` is an evolutionary optimizer you can strap onto your Bot, Simulation or Game AI.
Handcraft some parameters, set up the loop and let it fly.

## Features

- **Easy** : Start from existing values without much setup
- **Persistent** : Optimizer lives in a JSON file
- **Dynamic** : Keep developing without losing progress
- **Efficient** : Canonical xNES [^1] implementation, benchmarked against `cma` [^2] on BBOB [^4]

## Installation

Requires Python 3.11+

```sh
poetry install --all-extras
```

## Example 1

On a base level, `leitwerk` is a function optimizer with an `ask`/`tell` interface:

```py
from leitwerk import Optimizer, Parameter

def f(x1, x2):
    return (x1 - 1)**2 + (x2 - 1)**2  # minimum at (1, 1)

opt = Optimizer({"x1": Parameter(), "x2": Parameter()}, minimize=True)

for _ in range(100):
    x = opt.ask()
    opt.tell(f(**x))

print(opt.ask_best())
# {'x1': 0.9943998488500848, 'x2': 1.0003564039700967}
```

## Example 2 - Worker Rush Bot

To see persistence in action, run:

```sh
poetry run python examples/train_sc2_bot.py
```

This trains a simple probe rush bot with two parameters against the hardest built-in AI.
Optimizer state is persisted in `./data/params.json`.
It is somewhat human-readable, so have a look.

## API Summary

```py
class Parameter:
    def __init__(self, loc=None, scale=1.0, min=None, max=None):
      
class Optimizer[T]:
    def __init__(self, Schema, ...):
    def load(self, state: JSON) -> SchemaDiff:
    def ask(self, context: JSON = None) -> T:
    def tell(self, result: float | Sequence[float]) -> TellResult:
    def save(self) -> JSON:
```

## Integration Guide

### 1. Preflight Check

Define your parameter schema as an annotated dataclass:

```py
@dataclass
class Params:
    attack_threshold: Annotated[float, Parameter()]                     # standard normal N(0, 1)
    army_priority: Annotated[float, Parameter(loc=3.0, scale=0.5)]      # or mean + std
    skirmish_range: Annotated[float, Parameter(min=5, max=10)]          # or lower/upper bounds
    worker_target: Annotated[float, Parameter(loc=50, scale=10, min=1)] # or a mix
```

This tells the optimizer how to seed the population:

- `loc` : initial best guess (prior median in user-space)
- `scale` : initial spread/uncertainty  (prior standard deviation in latent space)
- `min` and `max` : asymptotic bounds (enforced as soft-plus/sigmoid transformations)
- most combinations work, see [Optimizer details](#optimization-details)

Samples are typed, so IntelliSense and type-checking work.
Alternatively, you can use plain dictionaries, see the [example below](#example-1-function-minimization)

> [!TIP]
> Use nested schemas to group the parameters into blocks, `leitwerk` understands tree structures.

### 2. Takeoff

Create the optimizer with your schema and old state (if present):

```py
opt = Optimizer(Params)
params_file = Path("params.json")
if params_file.exists():
    with params_file.open() as f:
        schema_diff = opt.load(json.load(f))  # load restores learned state, not constructor options
params = opt.ask()
```

Optional optimizer arguments:

- `population_size` : number of samples per optimizer step
- `minimize` : switches to minimization mode
- `eta_mu`, `eta_sigma`, `eta_B` : tunable learning rates [^1]

> [!NOTE]
> When the schema changes, state is reconciled per parameter.
> - Parameters are identified by their flattened name
> - `schema_diff` is a report of added/removed/changed/unchanged parameters
> - Changes to `min`/`max` trigger a reset

> [!TIP]
> When the _meaning_ of a parameter changes, the optimizer cannot know.
> It will adapt eventually - alternatively, rename the parameter to trigger a selective reset.

### 3. Landing

When you see the result, encode it as one or more numbers:

```py
prio1 = +1 if win else 0
prio2 = calc_heuristic()
report = opt.tell((prio1, prio2))
with params_file.open("w") as f:
    json.dump(opt.save(), f)
```

> [!NOTE]
> - Result tuples are compared lexicographically with higher=better
> - Only ranking matters, not objective magnitudes
> - `Optimizer(..., minimize=True)` switches to minimization mode

> [!IMPORTANT]


The objective function shapes the problem landscape:
- Binary Win/Loss is fine, but slow
- Unit Counts? Army Value? Income?
- When the objective changes drastically, consider a reset
- You might want to set up multiple optimizers with different objectives

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

