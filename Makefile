.PHONY: fix check docs docs-serve
PYTHON ?= python

fix:
	$(PYTHON) -m ruff check --fix xnes test
	$(PYTHON) -m ruff format xnes test

check:
	$(PYTHON) -m ruff check xnes test
	$(PYTHON) -m ruff format --check xnes test
	$(PYTHON) -m mypy xnes test
	$(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m mkdocs build

docs-serve:
	$(PYTHON) -m mkdocs serve
