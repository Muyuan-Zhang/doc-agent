# Requires GNU Make.
# Windows: winget install GnuWin32.Make  (or choco install make)
# macOS/Linux: make is available by default.

.DEFAULT_GOAL := help

.PHONY: help install up down dev test test-integration lint format logs clean

help:
	@echo "doc-agent development targets"
	@echo ""
	@echo "  install          pip install -e .[dev]"
	@echo "  up               start infra (postgres / redis / milvus) and wait until healthy"
	@echo "  down             stop infra containers"
	@echo "  dev              up + run uvicorn dev server (foreground, reload enabled)"
	@echo "  test             unit tests only (no integration marker)"
	@echo "  test-integration all tests including @pytest.mark.integration"
	@echo "  lint             ruff check + mypy"
	@echo "  format           ruff format + ruff --fix"
	@echo "  logs             tail all container logs (Ctrl+C to stop)"
	@echo "  clean            remove __pycache__, .pytest_cache, *.pyc, logs/"

install:
	pip install -e ".[dev]"

up:
	docker compose up -d --wait

down:
	docker compose down

dev: up
	python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	python -m pytest tests/ -m "not integration" --cov=app --cov-report=term-missing

test-integration:
	python -m pytest tests/ --cov=app --cov-report=term-missing

lint:
	python -m ruff check .
	python -m mypy app/

format:
	python -m ruff format .
	python -m ruff check --fix .

logs:
	docker compose logs -f

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
	python -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]"
	python -c "import shutil; shutil.rmtree('logs', ignore_errors=True)"
