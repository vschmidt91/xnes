# xnes

Small Python library for black-box optimization with named scalar parameters, strict checkpointing, and a
lightweight xNES implementation.

It is aimed at expensive, stateful evaluation loops: tuning heuristic weights in simulators, control systems, game
agents, or other black-box programs where runs are noisy, resumable, and often organized around recurring contexts.

## Highlights

- Canonical xNES core with optional CSA step-size adaptation.
- Argument-free `Optimizer()` wrapper with sensible defaults inherited from `XNES`.
- Named scalar parameters via `add(...)`, with lexicographic ordering independent of registration order.
- JSON-compatible optimizer state through `save()` and `load(...)`.
- Optional mirrored-sample routing through `ask(context=...)` using human-readable string contexts.
- Deterministic inference through `ask_best()`, which returns the current means without sampling.

## Requirements

- Python `>=3.11,<3.14`
- Runtime dependencies: NumPy and SciPy

## Installation

Install the package locally:

```bash
python -m pip install -e .
```

Install with development tools:

```bash
python -m pip install -e ".[dev]"
```

Install with development tools and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

## Quickstart

```python
import json
from pathlib import Path

import numpy as np

from xnes import Optimizer

state_path = Path("optimizer-state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else None

opt = Optimizer()
opt.pop_size = 32

opt.add("coeff_1", loc=2.0, scale=3.0)
opt.add("coeff_2")

load_result = opt.load(state)

for _ in range(500):
    params = opt.ask(context="validation:shard-0")
    value = params["coeff_1"] + np.exp(params["coeff_2"])
    report = opt.tell(params, -value**2)  # `tell` maximizes, so minimize via `-f`
    state_path.write_text(json.dumps(opt.save()))

    if report.completed_batch:
        pass

best = opt.ask_best()
print(best["coeff_1"], best["coeff_2"])
```

## Workflow

The wrapper is intentionally strict. The expected loop is:

1. Create `Optimizer()`.
2. Register parameters with `add(...)`.
3. Call `load(None)` for a fresh run or `load(state)` to resume.
4. Call `ask(context=...)` to reserve one parameter sample.
5. Read sampled values directly from `params[...]`.
6. Evaluate exactly once.
7. Call `tell(params, result)`.
8. Persist with `save()`.

For deterministic inference after training:

1. Create `Optimizer()`.
2. Register parameters with `add(...)`.
3. Call `load(state)`.
4. Call `ask_best()`.
5. Read mean values directly from `params[...]`.

Important constraints:

- `add()` is setup-only and must happen before `load()`.
- The registered parameter set is fixed after `load()`.
- When `load(state)` sees a changed parameter set, shared learned state is reconciled, added parameters start from
  priors, removed parameters are dropped, and any in-flight batch is reconciled rather than discarded.
- `load(None)` reports all currently registered parameters as added.
- `ask()` creates runtime-only reservations; these claims are not persisted.
- `ask_best()` is context-free and returns a deterministic snapshot of the current means.
- `ask_best()` returns parameters with `sample_id=None`; they are not sampled reservations.
- If all samples in a batch are reserved and unresolved, `ask()` raises.
- If `ask(context=...)` is never used, evaluation proceeds in the default batch order.
- `tell()` only accepts sampled parameters returned by `ask()`.

## Core API

- `Optimizer.add(name, loc=0.0, scale=1.0) -> None`
  Register a named scalar parameter.
- `Optimizer.load(state) -> LoadResult`
  Initialize from priors with `None`, or restore and reconcile a previous snapshot from `save()`.
- `Optimizer.ask(context=None) -> Parameters`
  Reserve one sample and return its parameter mapping plus reservation metadata.
- `Optimizer.ask_best() -> Parameters`
  Return a deterministic snapshot of the current means. This does not reserve a sample.
- `Optimizer.tell(params, result) -> TellResult`
  Submit one scalar or tuple-like objective result for sampled parameters returned by `ask()`.
- `Optimizer.save() -> dict[str, object]`
  Return a JSON-compatible snapshot of optimizer state.
- `LoadResult`
  Reports parameters added and parameters removed.
- `Parameters`
  Parameter mapping returned by `ask()` or `ask_best()`. Sampled parameters carry a concrete `sample_id`; `ask_best()`
  returns `sample_id=None`.

## Configuration

`Optimizer()` intentionally takes no constructor arguments. Optional tuning lives on instance attributes:

```python
opt = Optimizer()
opt.pop_size = 32
opt.csa_enabled = False
opt.eta_mu = 0.9
opt.eta_sigma = 0.7
opt.eta_B = 0.2
```

Behavior:

- Leaving `pop_size` as `None` keeps the xNES default batch size.
- Odd `pop_size` values are rounded up to the next even value when a new batch is created.
- Leaving `csa_enabled`, `eta_mu`, `eta_sigma`, or `eta_B` as `None` keeps the defaults defined by `XNES`.
- A bare `XNES(...)` instance starts with `csa_enabled = True`, `eta_mu = 1.0`, `eta_sigma = 1.0`, and
  `eta_B = 1.0`.
- `eta_B` scales the built-in dimension-dependent shape-learning rate multiplicatively.

## Objective Semantics

- `tell()` uses maximize semantics.
- Scalar results are treated as one-element tuples.
- Sequence results are ranked lexicographically.
- Higher tuples are better.
- This is not a Pareto or multiobjective optimizer.

## Training Loop

- During training, follow `load -> ask -> evaluate -> tell -> save`.

## Inference

- For deterministic inference, follow `load -> ask_best`.
- `ask_best()` ignores context because it does not sample.
- Outputs from `ask_best()` are read-only snapshots of current means and must not be passed to `tell()`.

## Development

- `make fix`: apply Ruff fixes and format the codebase.
- `make check`: run Ruff, Ruff format checks, mypy, and pytest.
- `make docs`: build the MkDocs site.
- `make docs-serve`: serve the MkDocs site locally.
- `make fix check`: common local pass to auto-fix, lint, type-check, and test.
