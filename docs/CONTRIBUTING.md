# Contributing Guide

## Prerequisites

- Python ≥ 3.11
- [Hatch](https://hatch.pypa.io/) or plain `pip` with a virtual environment
- Running infrastructure for integration tests: PostgreSQL, Redis, Milvus

## Setup

```bash
# 1. Clone and enter the project
git clone <repo>
cd doc-agent

# 2. Install runtime + dev dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env — at minimum set POSTGRES_URL, REDIS_URL, OPENAI_API_KEY

# 4. Apply database migrations
psql "$POSTGRES_URL" -f migrations/m1_schema.sql
psql "$POSTGRES_URL" -f migrations/m5_schema.sql

# 5. Start the service
uvicorn app:create_app --factory --reload
```

---

## Commands

<!-- AUTO-GENERATED from pyproject.toml -->

| Command | Description |
|---------|-------------|
| `pip install -e ".[dev]"` | Install all dependencies including dev tools |
| `uvicorn app:create_app --factory --reload` | Start dev server with hot reload on port 8000 |
| `python -m pytest tests/ -q` | Run full test suite (unit + e2e, no real infra needed) |
| `python -m pytest tests/ -q --ignore=tests/e2e` | Unit tests only |
| `python -m pytest -m integration` | Integration tests (requires live infra) |
| `python -m pytest tests/ --cov=app --cov-report=term-missing` | Tests with coverage report |
| `python -m pytest tests/memory/ --cov=app/memory --cov-fail-under=80` | Module-level coverage gate |

<!-- END AUTO-GENERATED -->

---

## Testing

### Philosophy

TDD is mandatory — tests before implementation. Each module follows the ECC pipeline:

```
/ecc:plan → /ecc:tdd-workflow → /ecc:code-review → /ecc:security-scan → /e2e → merge
```

### Coverage Gates

| Module | Minimum |
|--------|---------|
| M0 (`app/core/`, `app/clients/`) | 90% |
| All other modules | 80% |

### Test types

- **Unit tests** (`tests/<module>/`) — all network calls use `AsyncMock`; no real infra
- **E2E tests** (`tests/e2e/`) — full ASGI stack via `httpx.AsyncClient`; still mocked at the client boundary
- **Integration tests** — marked `@pytest.mark.integration`; require live PostgreSQL/Redis/Milvus

### Writing tests

Follow the AAA pattern (Arrange → Act → Assert). Use the helper factories already established in each test module (`_make_pg()`, `_make_redis()`, `_make_milvus()`, etc.) for consistency.

---

## Code Style

- Python 3.11+ syntax (`X | Y` unions, `list[X]` generics — no `from typing import`)
- Pydantic v2 with `ConfigDict(frozen=True)` for all data schemas
- SQLAlchemy Core only — no ORM `Base`/`Session`/`relationship`
- All SQL via `text()` with named parameter dicts
- Milvus access exclusively through `MilvusClient` — no direct `pymilvus` imports in business code
- Functions ≤ 50 lines, files ≤ 800 lines
- No inline comments unless the WHY is non-obvious

---

## PR Checklist

- [ ] Tests written before implementation (TDD)
- [ ] Coverage ≥ 80% (≥ 90% for M0)
- [ ] No hardcoded secrets or connection strings
- [ ] `CLAUDE.md` module boundary respected (no cross-module imports outside defined deps)
- [ ] Migration SQL added if schema changes
- [ ] `.env.example` updated if new env vars added
- [ ] `docs/ENV.md` updated to reflect new settings
