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
- Optional mirrored-sample routing through `set_context(...)` using JSON-serializable context values.

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

coeff_1 = opt.add("coeff_1", loc=2.0, scale=3.0)
coeff_2 = opt.add("coeff_2")

load_result = opt.load(state)

for _ in range(500):
    # Optional: use any JSON-serializable context to match mirrored samples.
    # opt.set_context({"task": "validation", "shard": 0})

    value = coeff_1.value + np.exp(coeff_2.value)
    report = opt.tell(-value**2)  # `tell` maximizes, so minimize via `-f`
    state_path.write_text(json.dumps(opt.save()))

    if report.completed_batch:
        pass
```

## Workflow

The wrapper is intentionally strict. The expected loop is:

1. Create `Optimizer()`.
2. Register parameters with `add(...)`.
3. Call `load(None)` for a fresh run or `load(state)` to resume.
4. Optionally call `set_context(context)` for the current evaluation.
5. Read the current `Parameter.value` values.
6. Evaluate exactly once.
7. Call `tell(result)`.
8. Persist with `save()`.

Important constraints:

- `add()` is setup-only and must happen before `load()`.
- The registry is fixed after `load()`.
- When `load(state)` sees a changed parameter set, shared learned state is reconciled, added parameters start from
  priors, removed parameters are dropped, and any in-flight batch is discarded.
- `load(None)` reports all currently registered parameters as added.
- `save()` must happen after `tell()`, not after `set_context()` and before `tell()`.
- If `set_context()` is never called, evaluation proceeds in the default batch order.
- `set_context()` accepts JSON-serializable values and stores only a stable hash, not the original object.

## Core API

- `Optimizer.add(name, loc=0.0, scale=1.0) -> Parameter`
  Register a named scalar parameter and get its mutable sampled view.
- `Optimizer.load(state) -> LoadResult`
  Initialize from priors with `None`, or restore and reconcile a previous snapshot from `save()`.
- `Optimizer.tell(result) -> Report`
  Submit one scalar or tuple-like objective result for the current sample.
- `Optimizer.save() -> dict[str, object]`
  Return a JSON-compatible snapshot of optimizer state.
- `Optimizer.set_context(context) -> bool`
  Retarget the current sample using a JSON-serializable context value.
- `Optimizer.get_info() -> list[ParameterInfo]`
  Inspect current values, means, scales, and registration priors.
- `Optimizer.set_best() -> None`
  Overwrite exposed parameter views with the current population mean for evaluation or inference.
- `LoadResult`
  Reports parameters added, parameters removed, and whether loading discarded an unfinished batch.

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

## Training And Inference

- During training, follow the normal `load -> optional set_context -> evaluate -> tell -> save` loop.
- For evaluation or inference, call `set_best()` to expose the current population mean through each
  `Parameter.value`.
- If you want to resume training after `set_best()`, save state before calling it and later restore with `load(...)`.

## Development

- `make fix`: apply Ruff fixes and format the codebase.
- `make check`: run Ruff, Ruff format checks, mypy, and pytest.
- `make docs`: build the MkDocs site.
- `make docs-serve`: serve the MkDocs site locally.
- `make fix check`: common local pass to auto-fix, lint, type-check, and test.
