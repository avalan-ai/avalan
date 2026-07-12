"""Test strict reasoning-summary acceptance inventory enforcement."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from subprocess import CompletedProcess
from sys import modules
from threading import Barrier
from types import ModuleType
from typing import Any, cast

import pytest
from reasoning_summary_script_loader import (
    DuplicateJsonObjectNameError,
    load_reasoning_summary_script,
)

_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)
AcceptanceVerificationError = cast(
    type[RuntimeError],
    _ACCEPTANCE_SCRIPT.AcceptanceVerificationError,
)
load_manifest = cast(Any, _ACCEPTANCE_SCRIPT.load_manifest)
verify_acceptance = cast(Any, _ACCEPTANCE_SCRIPT.verify_acceptance)


def _write_suite(
    root: Path,
    source: str,
    *,
    node_id: str = "test_acceptance.py::test_required",
    conftest: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "test_acceptance.py").write_text(source, encoding="utf-8")
    if conftest is not None:
        (root / "conftest.py").write_text(conftest, encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        dumps(
            {
                "schema_version": 1,
                "feature": "reasoning_summary",
                "active_phase": 0,
                "dimensions": {"temporary": [node_id]},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _assert_rejected(
    tmp_path: Path,
    source: str,
    *,
    message: str,
    node_id: str = "test_acceptance.py::test_required",
    conftest: str | None = None,
) -> None:
    manifest = _write_suite(
        tmp_path,
        source,
        node_id=node_id,
        conftest=conftest,
    )
    with pytest.raises(AcceptanceVerificationError, match=message):
        verify_acceptance(manifest, repo_root=tmp_path)


def test_acceptance_runner_collects_and_executes_exact_node(
    tmp_path: Path,
) -> None:
    manifest = _write_suite(
        tmp_path,
        "def test_required():\n    assert True\n",
    )

    result = verify_acceptance(manifest, repo_root=tmp_path)

    assert result.node_ids == ("test_acceptance.py::test_required",)


def test_acceptance_runner_rejects_missing_uncollected_node(
    tmp_path: Path,
) -> None:
    _assert_rejected(
        tmp_path,
        "def test_present():\n    assert True\n",
        node_id="test_acceptance.py::test_missing",
        message="collection|collected|probe",
    )


def test_acceptance_runner_rejects_deselected_node(tmp_path: Path) -> None:
    _assert_rejected(
        tmp_path,
        "def test_required():\n    assert True\n",
        conftest=(
            "def pytest_collection_modifyitems(config, items):\n"
            "    removed = list(items)\n"
            "    items[:] = []\n"
            "    config.hook.pytest_deselected(items=removed)\n"
        ),
        message="deselected",
    )


def test_acceptance_runner_rejects_nonpassing_runtime_outcomes(
    tmp_path: Path,
) -> None:
    cases = (
        (
            (
                "import pytest\n@pytest.mark.skip(reason='static')\n"
                "def test_required():\n    assert True\n"
            ),
            "disallowed markers",
        ),
        (
            (
                "import pytest\n@pytest.mark.skipif(True, reason='static')\n"
                "def test_required():\n    assert True\n"
            ),
            "disallowed markers",
        ),
        (
            (
                "import pytest\ndef test_required():\n"
                "    pytest.skip('dynamic')\n"
            ),
            "outcome was skipped",
        ),
        (
            (
                "import pytest\ndef test_required():\n"
                "    pytest.xfail('dynamic')\n"
            ),
            "xfail/xpass|disallowed markers",
        ),
        (
            (
                "import pytest\n@pytest.mark.xfail(reason='unexpected pass')\n"
                "def test_required():\n    assert True\n"
            ),
            "disallowed markers|xfail/xpass",
        ),
        (
            (
                "import pytest\n@pytest.fixture\ndef broken():\n"
                "    raise RuntimeError('setup failed')\n"
                "def test_required(broken):\n    assert True\n"
            ),
            "not exactly once fully executed|outcome was failed",
        ),
        (
            (
                "import pytest\n@pytest.fixture\ndef broken():\n"
                "    yield\n    pytest.skip('teardown skip')\n"
                "def test_required(broken):\n    assert True\n"
            ),
            "outcome was skipped",
        ),
        (
            (
                "import pytest\n@pytest.fixture\ndef broken():\n"
                "    yield\n    raise RuntimeError('teardown failed')\n"
                "def test_required(broken):\n    assert True\n"
            ),
            "outcome was failed",
        ),
    )
    for index, (source, message) in enumerate(cases):
        _assert_rejected(
            tmp_path / f"runtime-{index}",
            source,
            message=message,
        )


def test_acceptance_runner_rejects_importorskip_collection(
    tmp_path: Path,
) -> None:
    _assert_rejected(
        tmp_path,
        "import pytest\n"
        "pytest.importorskip('reasoning_summary_missing_dependency')\n"
        "def test_required():\n    assert True\n",
        message="collection|collected|probe",
    )


def test_acceptance_manifest_rejects_duplicate_nodes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    node_id = "test_acceptance.py::test_required"
    manifest.write_text(
        dumps(
            {
                "schema_version": 1,
                "feature": "reasoning_summary",
                "active_phase": 0,
                "dimensions": {"one": [node_id], "two": [node_id]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceVerificationError, match="duplicate"):
        load_manifest(manifest)


def test_acceptance_probe_rejects_nonzero_process_with_payload() -> None:
    for invalid_returncode in (1, False, 0.0, "0", None):
        completed = CompletedProcess(
            args=["python"],
            returncode=cast(Any, invalid_returncode),
            stdout="__SENTINEL__{}\n",
            stderr="probe failed\n",
        )

        with pytest.raises(
            AcceptanceVerificationError,
            match="process exited",
        ):
            _ACCEPTANCE_SCRIPT._probe_payload(completed, "__SENTINEL__")


def test_acceptance_collection_rejects_nonzero_pytest_exit() -> None:
    for invalid_exit_code in (1, False, True, 0.0, 1.0, "0", None):
        payload: dict[str, object] = {
            "exit_code": invalid_exit_code,
            "items": [],
            "deselected": [],
            "collection_reports": [],
            "probe_stdout": "",
            "probe_stderr": "",
        }

        with pytest.raises(
            AcceptanceVerificationError,
            match="collection exited",
        ):
            _ACCEPTANCE_SCRIPT._verify_collection((), payload)

    node_id = "test_acceptance.py::test_required"
    valid_payload: dict[str, object] = {
        "exit_code": 0,
        "items": [{"nodeid": node_id, "markers": []}],
        "deselected": [],
        "collection_reports": [],
        "probe_stdout": "",
        "probe_stderr": "",
    }
    _ACCEPTANCE_SCRIPT._verify_collection((node_id,), valid_payload)

    unknown_payload = deepcopy(valid_payload)
    unknown_payload["unexpected"] = None
    with pytest.raises(AcceptanceVerificationError, match="invalid keys"):
        _ACCEPTANCE_SCRIPT._verify_collection((node_id,), unknown_payload)

    unknown_item = deepcopy(valid_payload)
    unknown_items = cast(list[dict[str, object]], unknown_item["items"])
    unknown_items[0]["unexpected"] = None
    with pytest.raises(AcceptanceVerificationError, match="invalid keys"):
        _ACCEPTANCE_SCRIPT._verify_collection((node_id,), unknown_item)

    invalid_markers = deepcopy(valid_payload)
    marker_items = cast(list[dict[str, object]], invalid_markers["items"])
    marker_items[0]["markers"] = [False]
    with pytest.raises(AcceptanceVerificationError, match="invalid fields"):
        _ACCEPTANCE_SCRIPT._verify_collection((node_id,), invalid_markers)

    unknown_collection_report = deepcopy(valid_payload)
    unknown_collection_report["collection_reports"] = [
        {
            "nodeid": node_id,
            "outcome": "failed",
            "detail": "failure",
            "unexpected": None,
        }
    ]
    with pytest.raises(AcceptanceVerificationError, match="invalid keys"):
        _ACCEPTANCE_SCRIPT._verify_collection(
            (node_id,),
            unknown_collection_report,
        )


def test_acceptance_runner_rejects_node_paths_outside_root(
    tmp_path: Path,
) -> None:
    node_ids = (
        "../outside_test.py::test_required",
        "/tmp/outside_test.py::test_required",
        "tests\\outside_test.py::test_required",
    )
    for index, node_id in enumerate(node_ids):
        root = tmp_path / f"path-{index}"
        manifest = _write_suite(
            root,
            "def test_required():\n    assert True\n",
            node_id=node_id,
        )
        with pytest.raises(AcceptanceVerificationError, match="escapes"):
            verify_acceptance(manifest, repo_root=root)


def test_acceptance_runner_rejects_manifest_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    manifest = _write_suite(
        tmp_path / "outside",
        "def test_required():\n    assert True\n",
    )
    root.mkdir()

    with pytest.raises(AcceptanceVerificationError, match="manifest"):
        verify_acceptance(manifest, repo_root=root)


def test_acceptance_manifest_rejects_invalid_schema(tmp_path: Path) -> None:
    payloads: tuple[dict[str, object], ...] = (
        {},
        {"schema_version": 2},
        {"schema_version": False},
        {"schema_version": True},
        {"schema_version": 1.0},
        {
            "schema_version": 1,
            "feature": "other",
            "active_phase": 0,
            "dimensions": {"one": ["test_a.py::test_a"]},
        },
        {
            "schema_version": 1,
            "feature": "reasoning_summary",
            "active_phase": True,
            "dimensions": {"one": ["test_a.py::test_a"]},
        },
        {
            "schema_version": 1,
            "feature": "reasoning_summary",
            "active_phase": 0.0,
            "dimensions": {"one": ["test_a.py::test_a"]},
        },
        {
            "schema_version": 1,
            "feature": "reasoning_summary",
            "active_phase": 0,
            "dimensions": {},
        },
        {
            "schema_version": 1,
            "feature": "reasoning_summary",
            "active_phase": 0,
            "dimensions": {"one": ["test_a.py::test_a"]},
            "unexpected": None,
        },
    )
    for index, payload in enumerate(payloads):
        manifest = tmp_path / f"manifest-{index}.json"
        manifest.write_text(dumps(payload), encoding="utf-8")
        with pytest.raises(AcceptanceVerificationError):
            load_manifest(manifest)

    duplicate_documents = (
        (
            '{"schema_version":false,"schema_version":1,'
            '"feature":"reasoning_summary","active_phase":0,'
            '"dimensions":{"one":["test_a.py::test_a"]}}'
        ),
        (
            '{"schema_version":1,"feature":"reasoning_summary",'
            '"active_phase":0,"dimensions":{'
            '"one":false,"one":["test_a.py::test_a"]}}'
        ),
    )
    for index, source in enumerate(duplicate_documents):
        manifest = tmp_path / f"duplicate-manifest-{index}.json"
        manifest.write_text(source, encoding="utf-8")
        with pytest.raises(
            AcceptanceVerificationError,
            match="duplicate JSON object name",
        ):
            load_manifest(manifest)

    nonfinite_constants = ("NaN", "Infinity", "-Infinity")
    nonfinite_documents = tuple(
        source
        for constant in nonfinite_constants
        for source in (
            (
                f'{{"__nonfinite__":{constant},"schema_version":1,'
                '"feature":"reasoning_summary","active_phase":0,'
                '"dimensions":{"one":["test_a.py::test_a"]}}'
            ),
            (
                '{"schema_version":1,"feature":"reasoning_summary",'
                f'"active_phase":0,"dimensions":{{"__nonfinite__":{constant},'
                '"one":["test_a.py::test_a"]}}'
            ),
        )
    )
    for index, source in enumerate(nonfinite_documents):
        manifest = tmp_path / f"nonfinite-manifest-{index}.json"
        manifest.write_text(source, encoding="utf-8")
        with pytest.raises(
            AcceptanceVerificationError,
            match="non-finite JSON number",
        ):
            load_manifest(manifest)


def _passing_execution_payload() -> dict[str, object]:
    node_id = "test_acceptance.py::test_required"
    return {
        "exit_code": 0,
        "items": [node_id],
        "deselected": [],
        "collection_reports": [],
        "probe_stdout": "",
        "probe_stderr": "",
        "reports": [
            {
                "nodeid": node_id,
                "when": phase,
                "outcome": "passed",
                "wasxfail": "",
                "detail": "",
            }
            for phase in ("setup", "call", "teardown")
        ],
    }


def test_acceptance_execution_rejects_nonzero_and_xfail_xpass() -> None:
    node_id = "test_acceptance.py::test_required"
    for invalid_exit_code in (1, False, True, 0.0, 1.0, "0", None):
        nonzero = _passing_execution_payload()
        nonzero["exit_code"] = invalid_exit_code
        with pytest.raises(
            AcceptanceVerificationError,
            match="execution exited",
        ):
            _ACCEPTANCE_SCRIPT._verify_execution((node_id,), nonzero)

    valid = _passing_execution_payload()
    _ACCEPTANCE_SCRIPT._verify_execution((node_id,), valid)

    unknown_payload = _passing_execution_payload()
    unknown_payload["unexpected"] = None
    with pytest.raises(AcceptanceVerificationError, match="invalid keys"):
        _ACCEPTANCE_SCRIPT._verify_execution((node_id,), unknown_payload)

    unknown_report = _passing_execution_payload()
    unknown_reports = cast(
        list[dict[str, object]],
        unknown_report["reports"],
    )
    unknown_reports[0]["unexpected"] = None
    with pytest.raises(AcceptanceVerificationError, match="invalid keys"):
        _ACCEPTANCE_SCRIPT._verify_execution((node_id,), unknown_report)

    invalid_report_fields = {
        "nodeid": (False,),
        "when": (False, "cleanup"),
        "outcome": (False, "unknown"),
        "wasxfail": (False, None),
        "detail": (False, None),
    }
    for field_name, invalid_values in invalid_report_fields.items():
        for invalid_value in invalid_values:
            invalid_report = _passing_execution_payload()
            reports = cast(
                list[dict[str, object]],
                invalid_report["reports"],
            )
            reports[0][field_name] = invalid_value
            with pytest.raises(AcceptanceVerificationError):
                _ACCEPTANCE_SCRIPT._verify_execution(
                    (node_id,),
                    invalid_report,
                )

    for missing_field in ("wasxfail", "detail"):
        missing_report = _passing_execution_payload()
        reports = cast(
            list[dict[str, object]],
            missing_report["reports"],
        )
        reports[0].pop(missing_field)
        with pytest.raises(AcceptanceVerificationError, match="invalid keys"):
            _ACCEPTANCE_SCRIPT._verify_execution(
                (node_id,),
                missing_report,
            )

    for outcome in ("passed", "skipped"):
        payload = _passing_execution_payload()
        reports = cast(list[dict[str, object]], payload["reports"])
        reports[1]["outcome"] = outcome
        reports[1]["wasxfail"] = "reason"
        with pytest.raises(AcceptanceVerificationError, match="xfail/xpass"):
            _ACCEPTANCE_SCRIPT._verify_execution((node_id,), payload)


def test_acceptance_execution_rejects_unexpected_and_duplicate_reports() -> (
    None
):
    node_id = "test_acceptance.py::test_required"
    unexpected = _passing_execution_payload()
    reports = cast(list[dict[str, object]], unexpected["reports"])
    reports.append(
        {
            "nodeid": "test_acceptance.py::test_unexpected",
            "when": "call",
            "outcome": "passed",
            "wasxfail": "",
            "detail": "",
        }
    )
    with pytest.raises(AcceptanceVerificationError, match="unexpected"):
        _ACCEPTANCE_SCRIPT._verify_execution((node_id,), unexpected)

    duplicate = _passing_execution_payload()
    duplicate_reports = cast(list[dict[str, object]], duplicate["reports"])
    duplicate_reports.append(dict(duplicate_reports[1]))
    with pytest.raises(AcceptanceVerificationError, match="exactly once"):
        _ACCEPTANCE_SCRIPT._verify_execution((node_id,), duplicate)

    missing = _passing_execution_payload()
    missing_reports = cast(list[dict[str, object]], missing["reports"])
    missing_reports.pop()
    with pytest.raises(AcceptanceVerificationError, match="exactly once"):
        _ACCEPTANCE_SCRIPT._verify_execution((node_id,), missing)

    unknown = _passing_execution_payload()
    unknown_reports = cast(list[dict[str, object]], unknown["reports"])
    unknown_reports.append(
        {
            "nodeid": node_id,
            "when": "cleanup",
            "outcome": "passed",
            "wasxfail": "",
            "detail": "",
        }
    )
    with pytest.raises(
        AcceptanceVerificationError,
        match="invalid acceptance execution phase",
    ):
        _ACCEPTANCE_SCRIPT._verify_execution((node_id,), unknown)


def test_acceptance_probe_rejects_malformed_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = (
        "",
        "__SENTINEL__{}\n__SENTINEL__{}\n",
        "__SENTINEL__{not-json}\n",
        "__SENTINEL__[]\n",
    )
    for stdout in cases:
        completed = CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=stdout,
            stderr="",
        )
        with pytest.raises(AcceptanceVerificationError):
            _ACCEPTANCE_SCRIPT._probe_payload(completed, "__SENTINEL__")

    probe_payloads = (
        (
            _ACCEPTANCE_SCRIPT._COLLECT_SENTINEL,
            {
                "exit_code": 0,
                "items": [],
                "deselected": [],
                "collection_reports": [],
                "unexpected": None,
            },
        ),
        (
            _ACCEPTANCE_SCRIPT._EXECUTE_SENTINEL,
            {
                "exit_code": 0,
                "items": [],
                "deselected": [],
                "collection_reports": [],
                "reports": [],
                "unexpected": None,
            },
        ),
    )
    for sentinel, payload in probe_payloads:
        completed = CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=sentinel + dumps(payload) + "\n",
            stderr="",
        )
        with pytest.raises(AcceptanceVerificationError, match="invalid keys"):
            _ACCEPTANCE_SCRIPT._probe_payload(completed, sentinel)

    duplicate_probe_documents = (
        (
            _ACCEPTANCE_SCRIPT._COLLECT_SENTINEL,
            (
                '{"exit_code":false,"exit_code":0,"items":[],'
                '"deselected":[],"collection_reports":[]}'
            ),
        ),
        (
            _ACCEPTANCE_SCRIPT._EXECUTE_SENTINEL,
            (
                '{"exit_code":0,"items":[],"deselected":[],'
                '"collection_reports":[],"reports":[{'
                '"nodeid":"test_a.py::test_a","when":"setup",'
                '"outcome":"passed","wasxfail":"",'
                '"detail":false,"detail":""}]}'
            ),
        ),
    )
    for sentinel, source in duplicate_probe_documents:
        completed = CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=sentinel + source + "\n",
            stderr="",
        )
        with pytest.raises(
            AcceptanceVerificationError,
            match="duplicate JSON object name",
        ):
            _ACCEPTANCE_SCRIPT._probe_payload(completed, sentinel)

    nonfinite_probe_documents = tuple(
        (sentinel, source)
        for constant in ("NaN", "Infinity", "-Infinity")
        for sentinel, source in (
            (
                _ACCEPTANCE_SCRIPT._COLLECT_SENTINEL,
                (
                    f'{{"__nonfinite__":{constant},"exit_code":0,'
                    '"items":[],"deselected":[],"collection_reports":[]}'
                ),
            ),
            (
                _ACCEPTANCE_SCRIPT._EXECUTE_SENTINEL,
                (
                    '{"exit_code":0,"items":[],"deselected":[],'
                    '"collection_reports":[],"reports":[{'
                    '"nodeid":"test_a.py::test_a","when":"setup",'
                    '"outcome":"passed","wasxfail":"","detail":"",'
                    f'"__nonfinite__":{constant}}}]}}'
                ),
            ),
        )
    )
    for sentinel, source in nonfinite_probe_documents:
        completed = CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=sentinel + source + "\n",
            stderr="",
        )
        with pytest.raises(
            AcceptanceVerificationError,
            match="non-finite JSON number",
        ):
            _ACCEPTANCE_SCRIPT._probe_payload(completed, sentinel)

    loader_path = Path(__file__).with_name(
        "reasoning_summary_script_loader.py"
    )

    def import_loader_alias(module_name: str) -> ModuleType:
        assert module_name not in modules
        spec = spec_from_file_location(module_name, loader_path)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        monkeypatch.setitem(modules, module_name, module)
        spec.loader.exec_module(module)
        return module

    first_loader_alias = import_loader_alias(
        "_reasoning_summary_script_loader_alias_one"
    )
    second_loader_alias = import_loader_alias(
        "_reasoning_summary_script_loader_alias_two"
    )
    process_state = cast(dict[str, Any], first_loader_alias._PROCESS_STATE)
    assert second_loader_alias._PROCESS_STATE is process_state
    assert (
        first_loader_alias._REASONING_SUMMARY_JSON
        is second_loader_alias._REASONING_SUMMARY_JSON
    )
    assert (
        first_loader_alias.DuplicateJsonObjectNameError
        is DuplicateJsonObjectNameError
    )
    assert (
        second_loader_alias.DuplicateJsonObjectNameError
        is DuplicateJsonObjectNameError
    )

    concurrent_load_count = 8
    concurrent_barrier = Barrier(concurrent_load_count)

    def concurrent_load(index: int) -> tuple[str, type[ValueError]]:
        concurrent_barrier.wait()
        loader_alias = (
            first_loader_alias if index % 2 == 0 else second_loader_alias
        )
        loaded = cast(
            ModuleType,
            loader_alias.load_reasoning_summary_script(
                "verify_reasoning_summary_acceptance"
            ),
        )
        return loaded.__name__, cast(type[ValueError], loaded.StrictJsonError)

    with ThreadPoolExecutor(max_workers=concurrent_load_count) as executor:
        concurrent_results = tuple(
            executor.map(concurrent_load, range(concurrent_load_count))
        )
    concurrent_names = tuple(name for name, _ in concurrent_results)
    assert len(concurrent_names) == len(set(concurrent_names))
    assert all(
        error_type is first_loader_alias.StrictJsonError
        for _, error_type in concurrent_results
    )

    alias = "reasoning_summary_json"
    with monkeypatch.context() as none_context:
        none_context.setitem(modules, alias, None)
        first_loader_alias.load_reasoning_summary_script(
            "verify_reasoning_summary_acceptance"
        )
        assert alias in modules and modules[alias] is None

    helper_module_name = cast(str, process_state["helper_module_name"])
    captured_helper = cast(ModuleType, process_state["helper"])
    with monkeypatch.context() as deleted_helper_context:
        deleted_helper_context.delitem(
            modules, helper_module_name, raising=True
        )
        deleted_helper_load = cast(
            ModuleType,
            second_loader_alias.load_reasoning_summary_script(
                "verify_reasoning_summary_acceptance"
            ),
        )
        assert helper_module_name not in modules
        assert deleted_helper_load.StrictJsonError is (
            first_loader_alias.StrictJsonError
        )

    permissive_same_file = ModuleType(helper_module_name)
    permissive_same_file.__file__ = captured_helper.__file__
    setattr(permissive_same_file, "strict_json_loads", loads)
    with monkeypatch.context() as replaced_helper_context:
        replaced_helper_context.setitem(
            modules, helper_module_name, permissive_same_file
        )
        replaced_helper_load = cast(
            ModuleType,
            first_loader_alias.load_reasoning_summary_script(
                "verify_reasoning_summary_acceptance"
            ),
        )
        assert modules[helper_module_name] is permissive_same_file
        assert replaced_helper_load.StrictJsonError is (
            first_loader_alias.StrictJsonError
        )

    benchmark_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_reasoning_summary.py"
    ).resolve()
    benchmark_path_digest = sha256(
        str(benchmark_path).encode("utf-8")
    ).hexdigest()[:16]
    benchmark_content_sha256 = sha256(benchmark_path.read_bytes()).hexdigest()
    colliding_sequence = cast(int, process_state["script_load_sequence"]) + 1
    colliding_module_name = (
        "_avalan_test_benchmark_reasoning_summary_"
        f"{benchmark_path_digest}_{benchmark_content_sha256[:16]}_"
        f"{colliding_sequence}"
    )
    collision_sentinel = ModuleType(colliding_module_name)
    with monkeypatch.context() as collision_context:
        collision_context.setitem(
            modules, colliding_module_name, collision_sentinel
        )
        collision_load = cast(
            ModuleType,
            second_loader_alias.load_reasoning_summary_script(
                "benchmark_reasoning_summary"
            ),
        )
        assert modules[colliding_module_name] is collision_sentinel
        assert collision_load.__name__ != colliding_module_name

    normal_loader = modules["reasoning_summary_script_loader"]
    injected_failure = RuntimeError("injected script execution failure")
    loaded_script_prefix = "_avalan_test_verify_reasoning_summary_acceptance_"
    script_names_before_failure = {
        module_name
        for module_name in modules
        if module_name.startswith(loaded_script_prefix)
    }

    def fail_script_execution(
        _spec: object,
        _module: ModuleType,
    ) -> None:
        raise injected_failure

    exception_private_sentinel = ModuleType(helper_module_name)
    exception_private_sentinel.__file__ = captured_helper.__file__
    with monkeypatch.context() as exception_context:
        exception_context.setitem(modules, alias, None)
        exception_context.setitem(
            modules,
            helper_module_name,
            exception_private_sentinel,
        )
        exception_context.setattr(
            normal_loader,
            "_execute_module",
            fail_script_execution,
        )
        with pytest.raises(RuntimeError) as execution_error:
            normal_loader.load_reasoning_summary_script(
                "verify_reasoning_summary_acceptance"
            )
        assert execution_error.value is injected_failure
        assert modules[alias] is None
        assert modules[helper_module_name] is exception_private_sentinel
        assert {
            module_name
            for module_name in modules
            if module_name.startswith(loaded_script_prefix)
        } == script_names_before_failure

    monkeypatch.delitem(modules, alias, raising=False)
    second_acceptance = load_reasoning_summary_script(
        "verify_reasoning_summary_acceptance"
    )
    assert second_acceptance is not _ACCEPTANCE_SCRIPT
    assert alias not in modules

    permissive = ModuleType(alias)
    setattr(permissive, "strict_json_loads", loads)
    monkeypatch.setitem(modules, alias, permissive)
    third_acceptance = load_reasoning_summary_script(
        "verify_reasoning_summary_acceptance"
    )
    second_benchmark = load_reasoning_summary_script(
        "benchmark_reasoning_summary"
    )
    assert third_acceptance is not second_acceptance
    assert modules[alias] is permissive

    duplicate_manifest = tmp_path / "alias-duplicate-manifest.json"
    duplicate_manifest.write_text(
        '{"schema_version":false,"schema_version":1,'
        '"feature":"reasoning_summary","active_phase":0,'
        '"dimensions":{"one":["test_a.py::test_a"]}}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate JSON object name"):
        third_acceptance.load_manifest(duplicate_manifest)

    duplicate_probe = CompletedProcess(
        args=["python"],
        returncode=0,
        stdout=(
            third_acceptance._COLLECT_SENTINEL
            + '{"exit_code":false,"exit_code":0,"items":[],'
            '"deselected":[],"collection_reports":[]}\n'
        ),
        stderr="",
    )
    with pytest.raises(RuntimeError, match="duplicate JSON object name"):
        third_acceptance._probe_payload(
            duplicate_probe,
            third_acceptance._COLLECT_SENTINEL,
        )

    duplicate_contract = tmp_path / "alias-duplicate-contract.json"
    contract_source = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "reasoning_summary"
        / "phase0_contract.json"
    ).read_text(encoding="utf-8")
    duplicate_contract.write_text(
        contract_source.replace(
            '"schema_version": 1',
            '"schema_version": false, "schema_version": 1',
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        second_benchmark,
        "contract_path",
        lambda: duplicate_contract,
    )
    with pytest.raises(DuplicateJsonObjectNameError) as duplicate_error:
        second_benchmark.benchmark_protocol()
    assert type(duplicate_error.value) is DuplicateJsonObjectNameError

    strict_contract = loads(contract_source)
    protocol = second_benchmark._benchmark_protocol_from_payload(
        strict_contract
    )
    duplicate_memory = CompletedProcess(
        args=["python"],
        returncode=0,
        stdout=(
            second_benchmark._MEMORY_SENTINEL
            + '{"peak_processing_bytes_excluding_source_fixture":false,'
            '"peak_processing_bytes_excluding_source_fixture":1,'
            '"current_retained_bytes_including_source_fixture":1,'
            '"peak_total_bytes_including_source_fixture":1}\n'
        ),
        stderr="",
    )
    monkeypatch.setattr(
        second_benchmark,
        "run",
        lambda *_args, **_kwargs: duplicate_memory,
    )
    with pytest.raises(DuplicateJsonObjectNameError) as memory_error:
        second_benchmark._isolated_memory_probe(1, "x", protocol)
    assert type(memory_error.value) is DuplicateJsonObjectNameError


def test_acceptance_runner_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test_escape.py").write_text(
        "def test_required():\n    assert True\n",
        encoding="utf-8",
    )
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    manifest = _write_suite(
        root,
        "def test_required():\n    assert True\n",
        node_id="linked/test_escape.py::test_required",
    )

    with pytest.raises(AcceptanceVerificationError, match="escapes"):
        verify_acceptance(manifest, repo_root=root)


def test_acceptance_runner_sanitizes_ambient_pytest_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k never_matches")
    monkeypatch.setenv("PYTEST_PLUGINS", "missing_host_plugin")
    monkeypatch.setenv("PYTHONPATH", "/outside/python/path")
    manifest = _write_suite(
        tmp_path,
        "def test_required():\n    assert True\n",
    )

    result = verify_acceptance(manifest, repo_root=tmp_path)

    assert result.node_ids == ("test_acceptance.py::test_required",)
