"""Load checked-in reasoning-summary scripts for direct unit testing."""

import sys
from hashlib import sha256
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from os import environ
from pathlib import Path
from subprocess import TimeoutExpired
from subprocess import run as run_process
from threading import RLock
from types import ModuleType
from typing import Any, cast

_MISSING_MODULE = object()
_LOADER_PATH = Path(__file__).resolve()
_LOADER_CONTENT_SHA256 = sha256(_LOADER_PATH.read_bytes()).hexdigest()
_HELPER_PATH = (
    _LOADER_PATH.parents[1] / "scripts" / "reasoning_summary_json.py"
).resolve()
_HELPER_CONTENT_SHA256 = sha256(_HELPER_PATH.read_bytes()).hexdigest()
_BENCHMARK_PATH = (
    _LOADER_PATH.parents[1] / "scripts" / "benchmark_reasoning_summary.py"
).resolve()
_PHASE9_SUBPROCESS_TIMEOUT_SECONDS = 60
_PHASE9_SUBPROCESS_MAXIMUM_ATTEMPTS = 3
_PHASE9_EXPECTED_HARD_GATE_EXIT_CODE = 1
_PHASE9_TIMING_FAILURE_REASONS = frozenset(
    {
        "median 8192-to-4096 work ratio exceeded 2.5",
        "p95 8192-to-4096 work ratio exceeded 2.5",
        "heartbeat maximum drift exceeded 100 milliseconds",
    }
)
_PHASE9_REPORT_KEYS = frozenset(
    {
        "generated_at",
        "hard_gate",
        "machine",
        "phase9_budgets",
        "phase9_metrics",
        "platform",
        "processor",
        "protocol",
        "python",
        "python_implementation",
        "schema_version",
        "stream_performance_budget",
        "suite",
        "workloads",
    }
)
_PATH_IDENTITY = f"{_LOADER_PATH}\0{_HELPER_PATH}"
_PROCESS_STATE_KEY = (
    "_avalan_reasoning_summary_loader_state_"
    f"{sha256(_PATH_IDENTITY.encode('utf-8')).hexdigest()}_"
    f"{_LOADER_CONTENT_SHA256}"
)


def _new_process_state() -> dict[str, Any]:
    """Return an uninitialized process-shared loader registry."""
    return {
        "lock": RLock(),
        "loader_path": str(_LOADER_PATH),
        "loader_content_sha256": _LOADER_CONTENT_SHA256,
        "script_load_sequence": 0,
        "helper": None,
        "helper_path": None,
        "helper_content_sha256": None,
        "helper_module_name": None,
    }


def _load_process_state() -> dict[str, Any]:
    """Return one registry shared by every physical loader alias."""
    state = cast(
        dict[str, Any],
        sys.__dict__.setdefault(_PROCESS_STATE_KEY, _new_process_state()),
    )
    lock = state.get("lock")
    assert lock is not None
    with lock:
        assert state.get("loader_path") == str(_LOADER_PATH)
        assert state.get("loader_content_sha256") == _LOADER_CONTENT_SHA256
        assert type(state.get("script_load_sequence")) is int
    return state


_PROCESS_STATE = _load_process_state()
_PROCESS_LOCK = cast(Any, _PROCESS_STATE["lock"])


def _module_registry() -> dict[str, Any]:
    """Return ``sys.modules`` with its runtime-permitted ``None`` values."""
    return cast(dict[str, Any], sys.modules)


def _module_entry(name: str) -> tuple[bool, object]:
    """Capture one module entry without conflating missing and ``None``."""
    registry = _module_registry()
    if name in registry:
        return True, registry[name]
    return False, _MISSING_MODULE


def _restore_module_entry(
    name: str,
    previous: tuple[bool, object],
) -> None:
    """Restore one module entry exactly to its captured state."""
    existed, value = previous
    registry = _module_registry()
    if existed:
        registry[name] = value
    else:
        registry.pop(name, None)


def _fresh_module_name(base_name: str) -> str:
    """Return an unused module name without replacing an existing entry."""
    registry = _module_registry()
    candidate = base_name
    collision_sequence = 0
    while candidate in registry:
        collision_sequence += 1
        candidate = f"{base_name}_{collision_sequence}"
    return candidate


def _execute_module(spec: ModuleSpec, module: ModuleType) -> None:
    """Execute a module through its checked import specification."""
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _load_reasoning_summary_json() -> ModuleType:
    """Load and retain the physically verified shared JSON helper."""
    path = _HELPER_PATH
    content_sha256 = sha256(path.read_bytes()).hexdigest()
    assert content_sha256 == _HELPER_CONTENT_SHA256
    with _PROCESS_LOCK:
        existing = _PROCESS_STATE.get("helper")
        if existing is not None:
            assert isinstance(existing, ModuleType)
            assert _PROCESS_STATE.get("helper_path") == str(path)
            assert (
                _PROCESS_STATE.get("helper_content_sha256") == content_sha256
            )
            existing_path = getattr(existing, "__file__", None)
            assert existing_path is not None
            assert Path(existing_path).resolve() == path
            return existing

        path_digest = sha256(str(path).encode("utf-8")).hexdigest()[:16]
        module_name = _fresh_module_name(
            "_avalan_reasoning_summary_json_"
            f"{path_digest}_{content_sha256[:16]}"
        )
        spec = spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        registry = _module_registry()
        registry[module_name] = module
        try:
            _execute_module(spec, module)
        except BaseException:
            registry.pop(module_name, None)
            raise
        loaded_path = getattr(module, "__file__", None)
        assert loaded_path is not None
        assert Path(loaded_path).resolve() == path
        assert sha256(path.read_bytes()).hexdigest() == content_sha256
        _PROCESS_STATE["helper"] = module
        _PROCESS_STATE["helper_path"] = str(path)
        _PROCESS_STATE["helper_content_sha256"] = content_sha256
        _PROCESS_STATE["helper_module_name"] = module_name
        return module


_REASONING_SUMMARY_JSON = _load_reasoning_summary_json()
StrictJsonError = cast(
    type[ValueError],
    cast(Any, _REASONING_SUMMARY_JSON).StrictJsonError,
)
DuplicateJsonObjectNameError = cast(
    type[ValueError],
    cast(Any, _REASONING_SUMMARY_JSON).DuplicateJsonObjectNameError,
)
NonFiniteJsonNumberError = cast(
    type[ValueError],
    cast(Any, _REASONING_SUMMARY_JSON).NonFiniteJsonNumberError,
)


class Phase9BenchmarkSubprocessError(RuntimeError):
    """Indicate that the isolated Phase 9 benchmark did not complete."""


def strict_json_loads(source: str) -> object:
    """Parse JSON through the shared duplicate-rejecting helper."""
    return cast(Any, _REASONING_SUMMARY_JSON).strict_json_loads(source)


def canonical_json_pointer(path: tuple[str | int, ...]) -> str:
    """Return a canonical pointer through the shared JSON helper."""
    return cast(
        str, cast(Any, _REASONING_SUMMARY_JSON).canonical_json_pointer(path)
    )


def typed_json_path(
    path: tuple[str | int, ...],
) -> tuple[tuple[str, str | int], ...]:
    """Return a typed path identity through the shared JSON helper."""
    return cast(
        tuple[tuple[str, str | int], ...],
        cast(Any, _REASONING_SUMMARY_JSON).typed_json_path(path),
    )


def json_mapping_entries(
    value: object,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str, tuple[str | int, ...], tuple[str, ...]], ...]:
    """Return mapping entries through the shared JSON helper."""
    return cast(
        tuple[tuple[str, tuple[str | int, ...], tuple[str, ...]], ...],
        cast(Any, _REASONING_SUMMARY_JSON).json_mapping_entries(value, path),
    )


def _phase9_subprocess_environment() -> dict[str, str]:
    """Return an environment without pytest or coverage instrumentation."""
    sanitized = {
        key: value
        for key, value in environ.items()
        if key.upper() != "PYTHONPATH"
        and not key.upper().startswith(("COVERAGE_", "COV_CORE_", "PYTEST_"))
    }
    sanitized["PYTHONNOUSERSITE"] = "1"
    return sanitized


def _phase9_complete_report_hard_gate(
    payload: object,
) -> tuple[bool, tuple[str, ...]] | None:
    """Return the hard gate from one structurally complete Phase 9 report."""
    if not isinstance(payload, dict) or set(payload) != _PHASE9_REPORT_KEYS:
        return None
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("suite") != "reasoning-summary-phase9-performance"
    ):
        return None
    for field_name in (
        "generated_at",
        "machine",
        "platform",
        "processor",
        "python",
        "python_implementation",
    ):
        if not isinstance(payload.get(field_name), str):
            return None
    for field_name in (
        "phase9_budgets",
        "protocol",
        "stream_performance_budget",
    ):
        if not isinstance(payload.get(field_name), dict):
            return None

    workloads = payload.get("workloads")
    if not isinstance(workloads, list) or len(workloads) != 2:
        return None
    for workload in workloads:
        if (
            not isinstance(workload, dict)
            or not isinstance(workload.get("deterministic"), dict)
            or not isinstance(workload.get("sample_microseconds"), list)
            or len(workload["sample_microseconds"]) != 20
        ):
            return None
    if {workload.get("delta_count") for workload in workloads} != {
        4096,
        8192,
    }:
        return None

    metrics = payload.get("phase9_metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "heartbeat",
        "queue_pressure",
        "responses_coalescing",
    }:
        return None
    heartbeat = metrics.get("heartbeat")
    responses_coalescing = metrics.get("responses_coalescing")
    queue_pressure = metrics.get("queue_pressure")
    if (
        not isinstance(heartbeat, dict)
        or not isinstance(responses_coalescing, dict)
        or not isinstance(queue_pressure, dict)
        or not queue_pressure
        or not all(
            isinstance(value, dict) for value in queue_pressure.values()
        )
    ):
        return None

    hard_gate = payload.get("hard_gate")
    if not isinstance(hard_gate, dict) or set(hard_gate) != {
        "failure_reasons",
        "passed",
    }:
        return None
    passed = hard_gate.get("passed")
    reasons = hard_gate.get("failure_reasons")
    if (
        type(passed) is not bool
        or not isinstance(reasons, list)
        or not all(isinstance(reason, str) and reason for reason in reasons)
        or passed is bool(reasons)
    ):
        return None
    return passed, tuple(reasons)


def _phase9_subprocess_exit_error(
    returncode: int,
    stdout: str,
    stderr: str,
) -> Phase9BenchmarkSubprocessError:
    """Return a bounded subprocess-exit error with captured diagnostics."""
    detail = "\n".join(
        value for value in (stdout.strip(), stderr.strip()) if value
    )
    message = f"Phase 9 benchmark subprocess exited with code {returncode}"
    if detail:
        message = f"{message}: {detail[-2000:]}"
    return Phase9BenchmarkSubprocessError(message)


def run_phase9_benchmark_subprocess() -> dict[str, object]:
    """Run the Phase 9 CLI outside pytest and coverage instrumentation."""
    command = [sys.executable, str(_BENCHMARK_PATH), "--phase9"]
    timing_failures: list[tuple[str, ...]] = []
    for attempt in range(1, _PHASE9_SUBPROCESS_MAXIMUM_ATTEMPTS + 1):
        try:
            completed = run_process(
                command,
                capture_output=True,
                check=False,
                cwd=_LOADER_PATH.parents[1],
                env=_phase9_subprocess_environment(),
                text=True,
                timeout=_PHASE9_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except TimeoutExpired:
            raise Phase9BenchmarkSubprocessError(
                "Phase 9 benchmark subprocess timed out after "
                f"{_PHASE9_SUBPROCESS_TIMEOUT_SECONDS} seconds"
            ) from None

        if completed.returncode not in {
            0,
            _PHASE9_EXPECTED_HARD_GATE_EXIT_CODE,
        }:
            raise _phase9_subprocess_exit_error(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
        try:
            payload = strict_json_loads(completed.stdout)
        except ValueError as error:
            if completed.returncode != 0:
                raise _phase9_subprocess_exit_error(
                    completed.returncode,
                    completed.stdout,
                    completed.stderr,
                ) from error
            raise Phase9BenchmarkSubprocessError(
                "Phase 9 benchmark subprocess returned invalid JSON"
            ) from error

        gate = _phase9_complete_report_hard_gate(payload)
        if gate is None:
            raise Phase9BenchmarkSubprocessError(
                "Phase 9 benchmark subprocess returned an invalid report"
            )
        passed, failure_reasons = gate
        if completed.returncode == 0:
            if not passed:
                raise Phase9BenchmarkSubprocessError(
                    "Phase 9 benchmark subprocess returned an invalid report"
                )
            return cast(dict[str, object], payload)

        if passed or not failure_reasons:
            raise Phase9BenchmarkSubprocessError(
                "Phase 9 benchmark subprocess returned an invalid report"
            )
        if not set(failure_reasons) <= _PHASE9_TIMING_FAILURE_REASONS:
            raise Phase9BenchmarkSubprocessError(
                "Phase 9 benchmark hard gate failed: "
                + "; ".join(failure_reasons)
            )
        timing_failures.append(failure_reasons)
        if attempt < _PHASE9_SUBPROCESS_MAXIMUM_ATTEMPTS:
            continue
        diagnostics = " | ".join(
            f"attempt {index}: {', '.join(reasons)}"
            for index, reasons in enumerate(timing_failures, start=1)
        )
        raise Phase9BenchmarkSubprocessError(
            "Phase 9 benchmark timing confirmation failed after 3 attempts: "
            f"{diagnostics}"
        )
    raise AssertionError("unreachable Phase 9 benchmark attempt state")


def load_reasoning_summary_script(name: str) -> ModuleType:
    """Load one repository script without making ``scripts`` a package."""
    assert isinstance(name, str)
    assert name in {
        "benchmark_reasoning_summary",
        "verify_reasoning_summary_acceptance",
    }

    path = (_LOADER_PATH.parents[1] / "scripts" / f"{name}.py").resolve()
    path_digest = sha256(str(path).encode("utf-8")).hexdigest()[:16]
    content_sha256 = sha256(path.read_bytes()).hexdigest()
    with _PROCESS_LOCK:
        assert _PROCESS_STATE.get("helper") is _REASONING_SUMMARY_JSON
        assert _PROCESS_STATE.get("helper_path") == str(
            Path(cast(str, _REASONING_SUMMARY_JSON.__file__)).resolve()
        )
        assert sha256(
            Path(cast(str, _REASONING_SUMMARY_JSON.__file__)).read_bytes()
        ).hexdigest() == _PROCESS_STATE.get("helper_content_sha256")

        registry = _module_registry()
        while True:
            sequence = cast(int, _PROCESS_STATE["script_load_sequence"]) + 1
            _PROCESS_STATE["script_load_sequence"] = sequence
            module_name = (
                f"_avalan_test_{name}_{path_digest}_"
                f"{content_sha256[:16]}_{sequence}"
            )
            if module_name not in registry:
                break

        spec = spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        script_entry = _module_entry(module_name)
        assert not script_entry[0]
        helper_module_name = cast(str, _PROCESS_STATE["helper_module_name"])
        private_helper_entry = _module_entry(helper_module_name)
        public_helper_entry = _module_entry("reasoning_summary_json")
        registry[module_name] = module
        registry[helper_module_name] = _REASONING_SUMMARY_JSON
        registry["reasoning_summary_json"] = _REASONING_SUMMARY_JSON
        try:
            _execute_module(spec, module)
        except BaseException:
            _restore_module_entry(module_name, script_entry)
            raise
        finally:
            _restore_module_entry(helper_module_name, private_helper_entry)
            _restore_module_entry(
                "reasoning_summary_json", public_helper_entry
            )
        return module
