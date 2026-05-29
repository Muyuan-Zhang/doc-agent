"""E2E tests for F2 engineering scripts.

Infra-free tier: lint and clean run via Python directly (same commands that
Makefile delegates to).  Integration tier: verifies the health endpoint is
reachable after docker compose brings the full stack up — mirrors what
start.bat does end-to-end.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
MAKE = shutil.which("make")


# ---------------------------------------------------------------------------
# Infra-free — lint / format pipeline (same commands as `make lint` / `make format`)
# ---------------------------------------------------------------------------

class TestLintPipeline:
    def test_ruff_is_installed_and_runnable(self):
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode == 0, "ruff not installed or broken"
        assert "ruff" in result.stdout.lower()

    def test_ruff_check_invocation_is_valid(self):
        """ruff check . must exit with code 0 (no errors) or 1 (lint hits).
        Exit code 2 means misconfiguration — that would be a F2 regression."""
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode in (0, 1), (
            f"ruff exited {result.returncode} (misconfiguration):\n{result.stderr}"
        )

    def test_mypy_is_installed_and_runnable(self):
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode == 0, "mypy not installed or broken"
        assert "mypy" in result.stdout.lower()


class TestCleanPipeline:
    def test_python_clean_pycache_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import shutil, pathlib; "
             "[shutil.rmtree(p, ignore_errors=True) "
             "for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode == 0

    def test_python_clean_pytest_cache_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-c", "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode == 0

    def test_python_clean_pyc_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# GNU Make smoke tests — skip when make is not installed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(MAKE is None, reason="GNU make not installed")
class TestMakeHelp:
    def test_help_exits_zero(self):
        result = subprocess.run([MAKE, "help"], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0

    def test_help_lists_required_targets(self):
        result = subprocess.run([MAKE, "help"], cwd=ROOT, capture_output=True, text=True)
        for target in ("install", "up", "down", "dev", "test", "lint", "format", "logs", "clean"):
            assert target in result.stdout, f"'{target}' missing from make help"

    def test_bare_make_shows_help(self):
        result = subprocess.run([MAKE], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0
        assert "doc-agent" in result.stdout


@pytest.mark.skipif(MAKE is None, reason="GNU make not installed")
class TestMakeClean:
    def test_clean_target_exits_zero(self):
        result = subprocess.run([MAKE, "clean"], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0

    def test_clean_removes_root_pycache(self):
        subprocess.run([MAKE, "clean"], cwd=ROOT, capture_output=True)
        assert not (ROOT / "__pycache__").exists()


# ---------------------------------------------------------------------------
# Integration — full stack lifecycle (requires docker compose + running infra)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullStackHealth:
    """Verify the health endpoint is reachable when docker compose stack is up.

    These tests are only meaningful when run via `make test-integration` after
    `make up` (or after `start.bat`).  They confirm the exact sequence that
    start.bat establishes actually results in a healthy API.
    """
    async def test_liveness_returns_200(self):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=5.0) as c:
            r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    async def test_readiness_returns_200_when_all_services_healthy(self):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=5.0) as c:
            r = await c.get("/health/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    async def test_all_five_services_reported_ok(self):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=5.0) as c:
            r = await c.get("/health/ready")
        checks = r.json()["checks"]
        for svc in ("postgres", "redis", "milvus", "mq", "llm"):
            assert checks.get(svc) == "ok", f"{svc} not healthy: {checks.get(svc)}"

    async def test_openapi_schema_accessible(self):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=5.0) as c:
            r = await c.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert schema.get("info", {}).get("title") == "doc-agent"
