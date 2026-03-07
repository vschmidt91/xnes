# Documentation

These should be created automatically from the source code and docstrings.
`mkdocs` should be able to do this.

# Task

1. add docstrings
2. generate doc folder from docstrings

# Notes

- generation ideally just with a make command, otherwise a script

# === GPT-5 ===

Goal: generate docs automatically from source + docstrings, with minimal manual maintenance.

## 1) Add real docstrings first
- Keep docstrings in `src/` as Google-style so `mkdocstrings` can render cleanly.
- Add/extend these docs:
  - module docs: `src/__init__.py`, `src/optimizer.py`, `src/xnes.py`
  - public API docs: `Optimizer`, `Parameter`, `XNES`, `add`, `remove`, `tell`, `save`, `load`
- Include at least:
  - short summary
  - args and return values
  - failure modes (`ValueError`, shape/ordering constraints)
  - small usage example for `Optimizer` in module-level docstring

## 2) Add MkDocs config for auto API extraction
- Add `mkdocs.yml` at repo root.
- Use plugins:
  - `mkdocs-material`
  - `mkdocstrings[python]`
  - `mkdocs-autorefs`
- Use structure:
  - `docs/index.md` as landing page
  - `docs/reference/api.md` with `::: src`
  - optionally `::: src.optimizer`, `::: src.xnes`, `::: src.__init__`
- Recommended `mkdocs.yml` fragments:
  - `site_name: xnes`
  - `site_url: ""` (or repo URL later)
  - `plugins: - mkdocstrings` with `handlers.python.options.docstring_style: google`, `show_root_heading: true`, `show_source: false`
  - `nav` entries for Overview / Quickstart / API Reference

## 3) Make docs one-command
- Extend `Makefile`:
  - `docs:` -> `$(PYTHON) -m mkdocs build`
  - `docs-serve:` -> `$(PYTHON) -m mkdocs serve`
  - optional `docs-clean:` -> `Remove-Item -Recurse -Force .\\site` (PowerShell), with guard on path

## 4) Add dependency group
- Add to `pyproject.toml` under `[project.optional-dependencies]`:
  - `docs = ["mkdocs>=1.6", "mkdocs-material>=9.5", "mkdocstrings[python]>=0.26", "mkdocs-autorefs>=1.0"]`
- This keeps docs tooling optional.

## 5) Add guardrails
- Add a lightweight doc-build check in CI:
  - run `make docs` in a docs job
  - fail early on docstring/API breakage
- Keep docs source under `/docs` and generated output in `/site` only.

## 6) Nice-to-have later
- Add `examples/` page with a one-click snippet using `Optimizer` state save/load.
- Add an `architecture.md` with update loop (`ask`/`tell`/`reconcile`) so behavior is easier to review than reading code alone.

## Report
- Implemented:
  - public API docstrings in `src/__init__.py`, `src/optimizer.py`, `src/xnes.py`
  - `mkdocs.yml`
  - `docs/index.md`
  - `docs/quickstart.md`
  - `docs/reference/api.md`
  - `Makefile` targets: `docs`, `docs-serve`
  - optional docs dependency set in `pyproject.toml`
- Executed successfully:
  - `.\.venv\Scripts\python -m ruff check src test`
  - `.\.venv\Scripts\python -m mypy src test`
  - `.\.venv\Scripts\python -m pytest -q`
- Build status:
  - `make docs` currently fails in the local environment
  - concrete blocker: `mkdocs-material` is not installed
  - additional missing packages: `mkdocstrings`, `mkdocs-autorefs`
- Conclusion:
  - source-driven docs setup is in place
  - final generation works once the docs extras are installed into the venv
