# xnes

Small Python library for black-box optimization with a canonical xNES core,
strict checkpointing, and a typed schema-first wrapper.

It is aimed at expensive, stateful evaluation loops: tuning heuristic weights
in simulators, control systems, game agents, or other black-box programs where
runs are noisy, resumable, and sometimes organized around recurring contexts.

## Highlights

- Canonical xNES core with optional CSA step-size adaptation.
- Schema-first `Optimizer(Schema)` wrapper returning typed dataclass params.
- Parameter metadata colocated with fields via `Annotated[float, Parameter(...)]`.
- JSON-compatible optimizer state with human-readable parameter schema definitions through `save()` and `load(...)`.
- Optional mirrored-sample routing through `ask(context=...)`.
- Deterministic inference through `ask_best()`.

## Requirements

- Python `>=3.11,<3.14`
- Runtime dependencies: NumPy and SciPy

## Installation

Install the package locally:

```bash
python -m pip install -e .
```

Install development tools with Poetry:

```bash
poetry install --with dev
```

Install development tools and documentation dependencies:

```bash
poetry install --with dev,docs
```

## Quickstart

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np

from xnes import Optimizer, Parameter


@dataclass(frozen=True)
class Params:
    coeff_1: Annotated[float, Parameter(loc=2.0, scale=3.0, min=0.0)]
    coeff_2: Annotated[float, Parameter()]


state_path = Path("optimizer-state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else None

opt = Optimizer(Params)
opt.pop_size = 32
load_result = opt.load(state)

for _ in range(500):
    trial, params = opt.ask(context="validation:shard-0")
    value = params.coeff_1 + np.exp(params.coeff_2)
    report = opt.tell(trial, -value**2)  # `tell` maximizes, so minimize via `-f`
    state_path.write_text(json.dumps(opt.save()))

    if report.completed_batch:
        pass

best = opt.ask_best()
print(best.coeff_1, best.coeff_2)
```

## Schema Model

The public wrapper is schema-first:

1. Define a dataclass schema.
2. Annotate each optimized field as `Annotated[float, Parameter(...)]`.
3. Construct `Optimizer(Schema)`.

Current schema constraints:

- The schema must be a dataclass type.
- Optimized fields must currently be `Annotated[float, Parameter(...)]`.
- Fields must be `init=True`.
- State layout is ordered lexicographically by field name, not by declaration order.

## Workflow

The wrapper is intentionally strict. The expected training loop is:

1. Define a schema dataclass.
2. Construct `Optimizer(Schema)`.
3. Call `load(None)` for a fresh run or `load(state)` to resume.
4. Call `ask(context=...)` to reserve one sampled parameter set.
5. Read runtime values from `params.field`.
6. Evaluate exactly once.
7. Call `tell(trial, result)`.
8. Persist with `save()`.

For deterministic inference after training:

1. Construct `Optimizer(Schema)`.
2. Call `load(state)`.
3. Call `ask_best()`.
4. Read mean values from `best.field`.

Important constraints:

- `load()` must be called before `ask()` or `ask_best()`.
- `load(None)` reports all current schema fields as added.
- `load(state)` reconciles changed schemas by persisted parameter definition.
- Shared learned state is preserved for unchanged fields.
- Added fields start from their parameter defaults.
- Removed fields are dropped.
- Any unfinished batch is reconciled rather than discarded.
- `ask()` creates runtime-only trials; claims are not persisted.
- `ask_best()` is context-free and returns a deterministic snapshot of current means as `T`.
- If all samples in a batch are reserved and unresolved, `ask()` raises.
- If `ask(context=...)` is never used, evaluation proceeds in the default batch order.

## Core API

- `Parameter(loc=0.0, scale=1.0, min=None, max=None)`
  Parameter metadata attached to one schema field. Add `min` and/or `max` for one-sided softplus or two-sided sigmoid bounds.
- `Optimizer(schema_type)`
  Construct a maximizing optimizer over a dataclass schema.
- `Optimizer.load(state) -> SchemaDiff`
  Initialize from schema defaults with `None`, or restore and reconcile a previous snapshot from `save()`.
- `Optimizer.ask(context=None) -> tuple[Trial, T]`
  Reserve one sample and return `(trial, params)`.
- `Optimizer.ask_best() -> T`
  Return a deterministic snapshot of the current means.
- `Optimizer.tell(trial, result) -> TellResult`
  Submit one scalar or tuple-like objective result for a trial returned by `ask()`.
- `Optimizer.save() -> dict[str, object]`
  Return a JSON-compatible snapshot of optimizer state, including readable per-field parameter definitions.
- `Trial`
  Runtime-only handle returned by `ask()` with `sample_id`, `context`, and `matched_context`.
- `SchemaDiff`
  Reports schema fields added, removed, changed, and unchanged when loading.
- `TellResult`
  Reports whether a batch completed, whether context matching occurred, and whether xNES restarted.

## Configuration

`Optimizer(Schema)` keeps optional tuning on instance attributes:

```python
opt = Optimizer(Params)
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
- `eta_B` scales the built-in dimension-dependent shape learning rate multiplicatively.

## Objective Semantics

- `tell()` uses maximize semantics.
- Scalar results are treated as one-element tuples.
- Sequence results are ranked lexicographically.
- Higher tuples are better.
- This is not a Pareto or multiobjective optimizer.

## Development

- `make fix`: apply Ruff fixes and format the codebase.
- `make check`: run Ruff, Ruff format checks, mypy, and pytest.
- `make docs`: build the MkDocs site.
- `make docs-serve`: serve the MkDocs site locally.
- `make fix check`: common local pass to auto-fix, lint, type-check, and test.
