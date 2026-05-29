"""Structural tests for F2 engineering scripts: Makefile, start.bat, stop.bat."""
from pathlib import Path

ROOT = Path(__file__).parent.parent

MAKEFILE = ROOT / "Makefile"
START_BAT = ROOT / "start.bat"
STOP_BAT = ROOT / "stop.bat"

REQUIRED_MAKE_TARGETS = [
    "help",
    "install",
    "up",
    "down",
    "dev",
    "test",
    "test-integration",
    "lint",
    "format",
    "logs",
    "clean",
]


class TestMakefileExists:
    def test_makefile_is_present(self):
        assert MAKEFILE.exists(), "Makefile not found at repo root"


class TestMakefileTargets:
    def _content(self) -> str:
        return MAKEFILE.read_text(encoding="utf-8")

    def test_has_help_target(self):
        assert "help:" in self._content()

    def test_has_install_target(self):
        assert "install:" in self._content()

    def test_has_up_target(self):
        assert "up:" in self._content()

    def test_has_down_target(self):
        assert "down:" in self._content()

    def test_has_dev_target(self):
        assert "dev:" in self._content()

    def test_has_test_target(self):
        assert "test:" in self._content()

    def test_has_test_integration_target(self):
        assert "test-integration:" in self._content()

    def test_has_lint_target(self):
        assert "lint:" in self._content()

    def test_has_format_target(self):
        assert "format:" in self._content()

    def test_has_logs_target(self):
        assert "logs:" in self._content()

    def test_has_clean_target(self):
        assert "clean:" in self._content()


class TestMakefileCommands:
    def _content(self) -> str:
        return MAKEFILE.read_text(encoding="utf-8")

    def test_install_uses_dev_extras(self):
        assert ".[dev]" in self._content()

    def test_up_uses_docker_compose_wait(self):
        content = self._content()
        assert "docker compose up" in content
        assert "--wait" in content

    def test_down_uses_docker_compose_down(self):
        assert "docker compose down" in content() if False else "docker compose down" in self._content()

    def test_test_excludes_integration_by_default(self):
        assert "not integration" in self._content()

    def test_lint_includes_ruff(self):
        assert "ruff" in self._content()

    def test_lint_includes_mypy(self):
        assert "mypy" in self._content()

    def test_format_uses_ruff_format(self):
        assert "ruff format" in self._content()

    def test_help_is_default_target(self):
        content = self._content()
        assert ".DEFAULT_GOAL" in content or content.strip().startswith("help:")

    def test_phony_declaration_present(self):
        assert ".PHONY" in self._content()


class TestStartBatExists:
    def test_start_bat_is_present(self):
        assert START_BAT.exists(), "start.bat not found at repo root"


class TestStartBatContent:
    def _content(self) -> str:
        return START_BAT.read_text(encoding="utf-8")

    def test_has_echo_off(self):
        assert "@echo off" in self._content().lower()

    def test_checks_for_dot_env(self):
        assert ".env" in self._content()

    def test_copies_from_env_example(self):
        assert ".env.example" in self._content()

    def test_uses_docker_compose_up(self):
        assert "docker compose up" in self._content()

    def test_waits_for_healthy_services(self):
        assert "--wait" in self._content()

    def test_saves_pid_to_file(self):
        assert ".uvicorn.pid" in self._content()

    def test_starts_uvicorn(self):
        assert "uvicorn" in self._content()

    def test_creates_logs_directory(self):
        assert "logs" in self._content()

    def test_prints_api_url(self):
        assert "8000" in self._content()

    def test_exits_nonzero_when_docker_fails(self):
        content = self._content()
        assert "exit /B 1" in content or "exit /b 1" in content.lower()


class TestStopBatExists:
    def test_stop_bat_is_present(self):
        assert STOP_BAT.exists(), "stop.bat not found at repo root"


class TestStopBatContent:
    def _content(self) -> str:
        return STOP_BAT.read_text(encoding="utf-8")

    def test_has_echo_off(self):
        assert "@echo off" in self._content().lower()

    def test_reads_pid_file(self):
        assert ".uvicorn.pid" in self._content()

    def test_kills_process(self):
        content = self._content()
        assert "taskkill" in content.lower()

    def test_runs_docker_compose_down(self):
        assert "docker compose down" in self._content()

    def test_cleans_up_pid_file(self):
        content = self._content()
        assert "del" in content.lower() and ".uvicorn.pid" in content

    def test_handles_missing_pid_gracefully(self):
        content = self._content()
        assert "if exist" in content.lower() or "if not exist" in content.lower()
