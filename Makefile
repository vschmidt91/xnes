.PHONY: fix check docs docs-serve

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

PYTHON := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python)
SOURCES = leitwerk tests examples

fix:
	$(PYTHON) -m ruff check --fix --unsafe-fixes $(SOURCES)
	$(PYTHON) -m ruff format $(SOURCES)

check:
	$(PYTHON) -m ruff check $(SOURCES)
	$(PYTHON) -m ruff format --check $(SOURCES)
	$(PYTHON) -m mypy $(SOURCES)
	$(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m mkdocs build

docs-serve:
	$(PYTHON) -m mkdocs serve
