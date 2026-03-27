<p align="center">
  <img src="docs/logo.png" alt="Leitwerk">
</p>

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with stochastic optimization.</em>
</p>

`leitwerk` is a schema-based evolutionary optimizer for long-running training loops. It comes with:

- parameters as plain dataclasses in your code
- JSON checkpoints you can inspect and resume
- schema reconciliation, so you can keep developing and retain progress

## Links

- [Documentation](https://phantomsc2.github.io/leitwerk/)
- [Integration Guide](https://phantomsc2.github.io/leitwerk/guide/)
- [API Reference](https://phantomsc2.github.io/leitwerk/reference/api/)

## Installation

Requires: Python >=3.11,<3.14

```sh
pip install .
```

For a development setup with tests, docs, benchmarks and `python-sc2`:

```sh
pip install -e .[dev,docs,benchmark]
```

## Minimal Example

At base level, `leitwerk` is a sequential ask/tell function maximizer:

```py
from leitwerk import Optimizer, Parameter

opt = Optimizer({"x": Parameter()})
for _ in range(100):
    x = opt.ask()["x"]
    fx = -(x - 1)**2
    opt.tell(fx)
```

```pycon
>>> opt.mean
{'x': 1.0007710964577097}
```

What happens between `ask` and `tell` is a black box for `leitwerk` - it can be a simple one-liner or a complex simulation.

## Example - StarCraft II Bot

For a real training loop with file persistence, see: [examples/train_sc2_bot.py](examples/train_sc2_bot.py)

- a simple worker rush bot with parameters `simulation_time` and `retreat_threshold`
- runs games continually against the built-in AI
- scores each game using win/loss and a combat heuristic
- persists state with `OptimizerSession` and plot results

Run it directly if you have StarCraft II installed and want to observe the games:

```sh
python examples/train_sc2_bot.py
```

Alternatively, use the headless docker setup:

```sh
cd examples
docker compose up --build
```

On the first run, this will download a 3.8GB installation of SC2 into the container.
Afterward, it will run significantly faster than the rendered game on the host.

### Training Results

The example saves progress in `examples/data` after each game:

- `params.json`: optimizer state
- `plot.png`: graphs showing parameter samples and result values and over time
- `history.json`: helper file

<p align="center">
  <img src="docs/example_plot.png" alt="Leitwerk">
</p>

Initially, it explores a wide range of values before narrowing the search.
After about 100 games, it achieves a perfect winrate again the ingame AI.

## Developer Commands

- `make fix`: auto-format
- `make check`: lint and test
- `make docs`: build docs
- `make docs-serve`: serve docs locally
