<p align="center">
  <img src="docs/logo.png" alt="Leitwerk">
</p>

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with stochastic optimization.</em>
</p>

`leitwerk` is a schema-based evolutionary optimizer that offers:

- typed parameters as plain dataclasses
- JSON checkpoints you can inspect and resume
- schema reconciliation, so you can keep developing and preserve progress

[Documentation](https://phantomsc2.github.io/leitwerk/)

## Links

- [StarCraft II Example Bot](https://phantomsc2.github.io/leitwerk/quickstart/)
- [Integration Guide](https://phantomsc2.github.io/leitwerk/integration/)
- [API Reference](https://phantomsc2.github.io/leitwerk/reference/api/)

## Installation

Requires: Python >=3.11,<3.14

```sh
pip install .
```

Full development setup for the SC2 example, tests, docs, and benchmarks:

```sh
pip install -e .[dev,docs,benchmark]
```

## Minimal Example

The core API of `leitwerk` is an ask/tell black-box optimizer:

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

## Developer Commands

- `make fix`: auto-format
- `make check`: lint and test
- `make docs`: build docs
- `make docs-serve`: serve docs locally
