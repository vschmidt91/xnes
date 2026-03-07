.PHONY: fix check docs docs-serve
PYTHON ?= python

fix:
	$(PYTHON) -m ruff check --fix src test
	$(PYTHON) -m ruff format src test

check:
	$(PYTHON) -m ruff check src test
	$(PYTHON) -m ruff format --check src test
	$(PYTHON) -m mypy src test
	$(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m mkdocs build

docs-serve:
	$(PYTHON) -m mkdocs serve
