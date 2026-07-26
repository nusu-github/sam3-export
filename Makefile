UV ?= uv
QUALITY_PATHS := src tests scripts

.PHONY: help lock sync toolchain format format-check lint typecheck test quality build clean

help:
	@printf '%s\n' \
	  'lock          Resolve the development lockfile' \
	  'sync          Install the development environment' \
	  'format        Apply safe Ruff fixes and format source and tests' \
	  'quality       Run formatting, lint, type, test, and build gates' \
	  'build         Build wheel and source distributions with uv'

lock:
	$(UV) lock

sync:
	$(UV) sync --all-groups --inexact

toolchain:
	$(UV) --version
	$(UV) run --no-sync ruff --version
	$(UV) run --no-sync pyrefly --version
	$(UV) run --no-sync pytest --version

format:
	$(UV) run --no-sync ruff check --fix $(QUALITY_PATHS)
	$(UV) run --no-sync ruff format $(QUALITY_PATHS)

format-check:
	$(UV) run --no-sync ruff format --check $(QUALITY_PATHS)

lint:
	$(UV) run --no-sync ruff check $(QUALITY_PATHS)

typecheck:
	$(UV) run --no-sync pyrefly check --config pyrefly.toml

test:
	$(UV) run --no-sync pytest

quality: toolchain format-check lint typecheck test build

build:
	$(UV) build --no-sources

clean:
	rm -rf .coverage .pytest_cache .pyrefly_cache .ruff_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
