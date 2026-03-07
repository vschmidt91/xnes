.PHONY: fix check
PYTHON := .\.venv\Scripts\python

fix:
	$(PYTHON) -m ruff check --fix src test
	$(PYTHON) -m ruff format src test

check:
	$(PYTHON) -m ruff check src test
	$(PYTHON) -m ruff format --check src test
	$(PYTHON) -m mypy src test
	$(PYTHON) -m pytest -q
