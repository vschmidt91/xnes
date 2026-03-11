.PHONY: fix check docs docs-serve
POETRY ?= poetry
SOURCES = xnes tests

fix:
	$(POETRY) run ruff format $(SOURCES)
	$(POETRY) run ruff check --fix --unsafe-fixes $(SOURCES)

check:
	$(POETRY) run ruff check $(SOURCES)
	$(POETRY) run ruff format --check $(SOURCES)
	$(POETRY) run mypy $(SOURCES)
	$(POETRY) run pytest -q

docs:
	$(POETRY) run mkdocs build

docs-serve:
	$(POETRY) run mkdocs serve
