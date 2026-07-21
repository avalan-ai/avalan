"""Exercise the database and common structured-input gate harnesses."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from subprocess import CompletedProcess
from sys import executable, modules
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    """Return one repository harness script as a module."""
    module_name = f"_input_contract_harness_{name}"
    spec = spec_from_file_location(
        module_name, _ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DATABASE = _load_script("task_pgsql_test_database")
_GATE = _load_script("run_input_contract_gate")


def test_admin_dsn_runner_script_uses_ephemeral_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the runner and generated test DSN through admin mode."""
    operations: list[tuple[str, str, str | None]] = []
    observed: dict[str, Any] = {}
    monkeypatch.setattr(_DATABASE, "_require_runtime_modules", lambda: None)
    monkeypatch.setattr(
        _DATABASE, "_database_name", lambda prefix: f"{prefix}_fixed"
    )
    monkeypatch.setattr(
        _DATABASE,
        "_create_database",
        lambda dsn, name: operations.append(("create", dsn, name)),
    )
    monkeypatch.setattr(
        _DATABASE,
        "_drop_database",
        lambda dsn, name: operations.append(("drop", dsn, name)),
    )

    def run_child(
        command: tuple[str, ...],
        *,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        observed["command"] = command
        observed["check"] = check
        observed["dsn"] = env["AVALAN_TASK_TEST_POSTGRESQL_DSN"]
        return CompletedProcess(command, 0)

    monkeypatch.setattr(_DATABASE, "run", run_child)
    code = _DATABASE._run_with_admin_dsn(
        "postgresql://admin:secret@db.example/postgres?sslmode=require",
        "contract",
        ("--through-phase", "0"),
        runner_script="scripts/run_input_contract_gate.py",
    )
    assert code == 0
    assert observed["command"] == (
        executable,
        "scripts/run_input_contract_gate.py",
        "--through-phase",
        "0",
    )
    assert observed["check"] is False
    assert (
        observed["dsn"]
        == "postgresql://admin:secret@db.example/contract_fixed?sslmode=require"
    )
    assert operations == [
        (
            "create",
            "postgresql://admin:secret@db.example/postgres?sslmode=require",
            "contract_fixed",
        ),
        (
            "drop",
            "postgresql://admin:secret@db.example/postgres?sslmode=require",
            "contract_fixed",
        ),
    ]


def test_main_selects_admin_dsn_and_runner_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer the explicit admin DSN and preserve child arguments."""
    observed: dict[str, object] = {}

    def run_admin(
        dsn: str,
        prefix: str,
        child_args: tuple[str, ...],
        *,
        runner_script: str | None,
    ) -> int:
        observed.update(
            dsn=dsn,
            prefix=prefix,
            child_args=child_args,
            runner_script=runner_script,
        )
        return 4

    monkeypatch.setattr(_DATABASE, "_run_with_admin_dsn", run_admin)
    monkeypatch.setattr(
        _DATABASE,
        "_run_with_docker",
        lambda *args, **kwargs: pytest.fail("Docker must not be selected"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "task_pgsql_test_database.py",
            "--admin-dsn",
            "postgresql://explicit/postgres",
            "--database-prefix",
            "selected",
            "--runner-script",
            "scripts/run_input_contract_gate.py",
            "--",
            "--through-phase",
            "0",
        ],
    )
    assert _DATABASE.main() == 4
    assert observed == {
        "dsn": "postgresql://explicit/postgres",
        "prefix": "selected",
        "child_args": ("--through-phase", "0"),
        "runner_script": "scripts/run_input_contract_gate.py",
    }


def test_docker_forwards_runner_and_always_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the runner through Docker and stop the started container."""
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(_DATABASE, "_require_runtime_modules", lambda: None)
    monkeypatch.setattr(
        _DATABASE, "_docker_container_name", lambda: "container"
    )
    monkeypatch.setattr(_DATABASE, "token_urlsafe", lambda size: "password")
    monkeypatch.setattr(_DATABASE, "_free_tcp_port", lambda: 55432)
    monkeypatch.setattr(
        _DATABASE,
        "_start_docker_postgres",
        lambda **values: events.append(("start", values)),
    )
    monkeypatch.setattr(
        _DATABASE,
        "_wait_for_database",
        lambda dsn, timeout: events.append(("wait", dsn, timeout)),
    )

    def run_admin(
        dsn: str,
        prefix: str,
        child_args: tuple[str, ...],
        *,
        runner_script: str | None,
    ) -> int:
        events.append(("admin", dsn, prefix, child_args, runner_script))
        return 6

    monkeypatch.setattr(_DATABASE, "_run_with_admin_dsn", run_admin)
    monkeypatch.setattr(
        _DATABASE,
        "_stop_docker_container",
        lambda name: events.append(("stop", name)),
    )
    code = _DATABASE._run_with_docker(
        "contract",
        ("--through-phase", "0"),
        image="postgres:16-alpine",
        timeout_seconds=12.0,
        runner_script="scripts/run_input_contract_gate.py",
    )
    assert code == 6
    assert events[0] == (
        "start",
        {
            "image": "postgres:16-alpine",
            "name": "container",
            "password": "password",
            "port": 55432,
        },
    )
    assert events[1] == (
        "wait",
        "postgresql://postgres:password@127.0.0.1:55432/postgres",
        12.0,
    )
    assert events[2] == (
        "admin",
        "postgresql://postgres:password@127.0.0.1:55432/postgres",
        "contract",
        ("--through-phase", "0"),
        "scripts/run_input_contract_gate.py",
    )
    assert events[3] == ("stop", "container")


def test_common_gate_cleans_partial_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete stale input and partial output when verification fails."""
    stale = tmp_path / ".coverage.stale"
    stale.write_text("stale", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def run_command(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> CompletedProcess[str]:
        assert cwd == tmp_path
        assert check is False
        assert "PYTHONPATH" not in env
        assert not stale.exists()
        calls.append(command)
        if len(calls) == 1:
            (tmp_path / ".coverage").write_text("partial", encoding="utf-8")
            (tmp_path / "coverage.json").write_text(
                "partial", encoding="utf-8"
            )
            (tmp_path / "coverage.xml").write_text("partial", encoding="utf-8")
            return CompletedProcess(command, 0)
        return CompletedProcess(command, 5)

    monkeypatch.setattr(_GATE, "run", run_command)
    monkeypatch.setenv("PYTHONPATH", "ambient")
    assert _GATE.run_coverage_gate(repo_root=tmp_path) == 5
    assert len(calls) == 2
    assert calls[0][1:3] == ("-m", "pytest")
    assert calls[1][1:] == ("scripts/verify_src_coverage.py",)
    assert not any(
        (tmp_path / name).exists()
        for name in (".coverage", "coverage.json", "coverage.xml")
    )


def test_common_gate_skips_acceptance_after_coverage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return the coverage failure without starting acceptance."""
    monkeypatch.setattr(_GATE, "run_coverage_gate", lambda *, repo_root: 11)
    monkeypatch.setattr(
        _GATE,
        "run",
        lambda *args, **kwargs: pytest.fail("acceptance must not run"),
    )
    assert _GATE.run_gate(0, repo_root=tmp_path) == 11
