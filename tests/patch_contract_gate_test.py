"""Exercise the public Phase 0 patch contract gate boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from os import environ, pathsep
from pathlib import Path, PurePosixPath
from subprocess import CompletedProcess, run
from sys import executable, modules
from sys import path as sys_path
from types import ModuleType, SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "patch"


def _load_script(name: str) -> ModuleType:
    """Load one standalone gate script with its repository dependencies."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    spec = spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_GATE = _load_script("run_patch_contract_gate")
_CONTRACT_GATE = modules["contract_gate"]
_PLUGIN = _load_script("patch_contract_gate_plugin")
_ACCEPTANCE = _load_script("verify_patch_acceptance")
_TYPES = _load_script("verify_patch_types")


def _patch_startup_environment() -> dict[str, str]:
    """Return one hardened child environment with the real Patch audit."""
    python_path = pathsep.join(
        (
            str(_ROOT / "scripts" / "contract_startup"),
            str(_ROOT / "src"),
            str(_ROOT / "scripts"),
        )
    )
    environment = dict(environ)
    environment.update(
        {
            "PYTHONSAFEPATH": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": python_path,
            "AVALAN_CONTRACT_ALLOWED_PYTHONPATH": python_path,
        }
    )
    environment.update(_GATE._patch_artifact_guard_environment(_ROOT))
    return environment


def _copy_fixtures(destination: Path) -> None:
    """Copy the complete patch fixture bundle into one disposable directory."""
    destination.mkdir()
    for source in _FIXTURES.glob("*.json"):
        (destination / source.name).write_bytes(source.read_bytes())


def _resign(payload: dict[str, object], field: str) -> None:
    """Recalculate one deterministic patch fixture digest after mutation."""
    if field == "catalog_sha256":
        canonical = {
            "record_layout": payload["record_layout"],
            "requirements": payload["requirements"],
        }
    else:
        canonical = {
            key: value for key, value in payload.items() if key != field
        }
    payload[field] = _ACCEPTANCE.canonical_sha256(canonical)


def _write(path: Path, payload: object) -> None:
    """Write one disposable JSON fixture with canonical terminal newline."""
    path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")


def _minimal_repository(root: Path) -> None:
    """Create the minimum sealed repository inventory for runner tests."""
    for relative, content in (
        ("Makefile", "all:\n\ttrue\n"),
        ("pyproject.toml", "[project]\nname = 'patch-gate'\n"),
        ("poetry.lock", "# lock\n"),
        ("src/sample.py", "VALUE = 1\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _facts() -> str:
    """Return one passing full-suite outcome payload."""
    return (
        dumps(
            {
                "schema_version": 1,
                "collected": 3,
                "passed": 3,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "collection_skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
                "deselected": 0,
                "warnings": 0,
                "leak_warnings": 0,
                "exitstatus": 0,
            }
        )
        + "\n"
    )


def test_patch_gate_contract_is_current() -> None:
    """Require the current selection, registration, and gate wiring."""
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "PATCH_PHASE is required" in makefile
    assert "test-patch-exact:" in makefile
    assert "test-patch-pgsql-exact:" in makefile
    assert "patch phases 0 through 7 require test-patch-exact" in makefile
    assert "scripts/run_patch_contract_gate.py --preflight" in makefile
    patch_target = makefile.split("test-patch-exact:", 1)[1].split(
        "test-patch-pgsql-exact:", 1
    )[0]
    assert "python -m pip install $(TASK_PGSQL_TEST_DEPS)" in patch_target
    assert (_ROOT / "src" / "avalan" / "patch").is_dir()
    assert not (_ROOT / "src" / "avalan" / "tool" / "patch.py").exists()
    baseline = loads(
        (_FIXTURES / "baseline_evidence.json").read_text(encoding="utf-8")
    )
    assert isinstance(baseline, dict)
    assert baseline["phase"] == 13
    assert baseline["patch_tools"] == []
    facts = baseline["section2_facts"]
    assert isinstance(facts, list)
    fact_ids: list[object] = []
    for fact in facts:
        assert isinstance(fact, dict)
        fact_ids.append(fact["id"])
    assert fact_ids == [f"PATCH-S2-{index:03d}" for index in range(1, 12)]
    advertisement = baseline["runtime_patch_advertisement"]
    assert advertisement == {
        "patch_toolset_path": "src/avalan/patch/toolset.py",
        "runtime_probe": [
            "toolmanager_default_and_configured_discovery",
            "cli_selectors_and_commands",
            "mcp_a2a_server_openapi_routes",
            "flow_task_orchestrator_nodes",
            "target_handshake_profile_selectors",
            "provider_capability_catalog",
        ],
    }
    assert not (_ROOT / "src" / "avalan" / "tool" / "patch.py").exists()
    assert _GATE._PATCH_DATABASE_PHASE == 8
    assert _GATE._PATCH_CURRENT_PHASE == 13


@pytest.mark.parametrize(
    "path", ("src/unowned_patch.py", "tests/unowned_patch_test.py")
)
def test_patch_type_gate_rejects_changed_python_bypass(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Reject unowned changed source and test paths before type execution."""
    monkeypatch.setattr(
        _TYPES,
        "_repository_python_paths",
        lambda root: (PurePosixPath(path),),
    )

    with pytest.raises(
        _TYPES.PatchTypeContractError, match="changed or untracked Python path"
    ):
        _TYPES._verify_strict_sources(_ROOT, (), ())


@pytest.mark.parametrize(
    "case", ("incomplete", "digest", "missing_path", "extra")
)
def test_patch_type_gate_rejects_invalid_sealed_ownership_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Reject incomplete, tampered, missing, and extra mirror ownership."""
    owned = tmp_path / "scripts" / "owned.py"
    owned.parent.mkdir()
    owned.write_text("VALUE = 1\n", encoding="utf-8")
    paths = (PurePosixPath("scripts/owned.py"),)
    environment = _TYPES.repository_python_ownership_environment(paths)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    if case == "incomplete":
        monkeypatch.delenv(_TYPES.PATCH_PYTHON_OWNERSHIP_SHA256_ENV)
        match = "inventory state is incomplete"
    elif case == "digest":
        monkeypatch.setenv(_TYPES.PATCH_PYTHON_OWNERSHIP_SHA256_ENV, "0" * 64)
        match = "inventory integrity is invalid"
    elif case == "missing_path":
        owned.unlink()
        match = "inventory path is missing"
    else:
        extra = tmp_path / "tests" / "extra_test.py"
        extra.parent.mkdir()
        extra.write_text("VALUE = 1\n", encoding="utf-8")
        environment = _TYPES.repository_python_ownership_environment(
            (
                *paths,
                PurePosixPath("tests/extra_test.py"),
            )
        )
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        match = "changed or untracked Python path is not owned"

    with pytest.raises(_TYPES.PatchTypeContractError, match=match):
        if case == "extra":
            _TYPES._verify_strict_sources(tmp_path, (), ())
        else:
            _TYPES._repository_python_paths(tmp_path)


def test_patch_strict_source_mypy_is_hermetic_and_keeps_owned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disable cache/import expansion while checking every owned path."""
    observed: tuple[str, ...] | None = None

    def run_mypy(command: tuple[str, ...], **kwargs: object) -> object:
        nonlocal observed
        observed = command
        assert kwargs["cwd"] == tmp_path
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_TYPES, "run", run_mypy)
    owned = ("scripts/owned.py", "tests/owned_test.py")

    _TYPES._run_strict_source_mypy(tmp_path, owned)

    assert observed is not None
    assert "--follow-imports=silent" in observed
    assert "--cache-dir=/dev/null" in observed
    assert observed[-len(owned) :] == owned


def test_patch_type_gate_limits_whole_module_mypy_to_complete_sources() -> (
    None
):
    """Keep integration hunks under exact structural ownership only."""
    sources = (
        _TYPES.StrictSource(
            identifier="PATCH-TS-TEST-001",
            path="src/owned.py",
            scope="patch_domain",
            symbols=("module",),
            source_sha256="0" * 64,
        ),
        _TYPES.StrictSource(
            identifier="PATCH-TS-TEST-002",
            path="tests/legacy_test.py",
            scope="integration_hunk",
            symbols=("COMPATIBILITY_HEAD",),
            source_sha256="1" * 64,
        ),
    )

    assert _TYPES._strict_mypy_paths(sources) == ("src/owned.py",)


def test_patch_type_gate_keeps_integration_hunks_under_exact_structural(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require digest, symbol, and AST checks for every integration hunk."""
    source_path = tmp_path / "src" / "compatibility.py"
    source_path.parent.mkdir()
    payload = b'COMPATIBILITY_HEAD = "current"\n'
    source_path.write_bytes(payload)
    source = _TYPES.StrictSource(
        identifier="PATCH-TS-TEST-003",
        path="src/compatibility.py",
        scope="integration_hunk",
        symbols=("COMPATIBILITY_HEAD",),
        source_sha256=sha256(payload).hexdigest(),
    )
    observed: tuple[str, ...] | None = None

    def run_mypy(root: Path, paths: tuple[str, ...]) -> None:
        nonlocal observed
        assert root == tmp_path
        observed = paths

    monkeypatch.setattr(_TYPES, "_repository_python_paths", lambda root: ())
    monkeypatch.setattr(_TYPES, "_run_strict_source_mypy", run_mypy)

    _TYPES._verify_strict_sources(tmp_path, (source,), ())

    assert observed == ()
    source_path.write_text(
        'COMPATIBILITY_HEAD = "changed"\n', encoding="utf-8"
    )
    with pytest.raises(
        _TYPES.PatchTypeContractError,
        match="patch strict source digest changed",
    ):
        _TYPES._verify_strict_sources(tmp_path, (source,), ())

    cast_payload = b'COMPATIBILITY_HEAD = cast(str, "current")\n'
    source_path.write_bytes(cast_payload)
    cast_source = _TYPES.StrictSource(
        identifier="PATCH-TS-TEST-004",
        path="src/compatibility.py",
        scope="integration_hunk",
        symbols=("COMPATIBILITY_HEAD",),
        source_sha256=sha256(cast_payload).hexdigest(),
    )
    with pytest.raises(_TYPES.PatchTypeContractError, match="cast"):
        _TYPES._verify_strict_sources(tmp_path, (cast_source,), ())

    source_path.write_text('OTHER = "current"\n', encoding="utf-8")
    missing_source = _TYPES.StrictSource(
        identifier="PATCH-TS-TEST-005",
        path="src/compatibility.py",
        scope="integration_hunk",
        symbols=("COMPATIBILITY_HEAD",),
        source_sha256=sha256(source_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(
        _TYPES.PatchTypeContractError,
        match="patch strict source symbol is missing",
    ):
        _TYPES._verify_strict_sources(tmp_path, (missing_source,), ())


def test_patch_fixture_mypy_is_hermetic_and_scopes_owned_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold-check fixtures while retaining only their own diagnostics."""
    observed: tuple[str, ...] | None = None
    fixture_path = "tests/owned_fixture.py"

    def run_mypy(
        command: tuple[str, ...], **kwargs: object
    ) -> CompletedProcess[str]:
        nonlocal observed
        observed = command
        assert kwargs["cwd"] == tmp_path
        return CompletedProcess(
            command,
            1,
            stdout=(
                "tests/owned_fixture.py:4:1: error: owned failure "
                "[assignment]\n"
                "src/legacy.py:8:1: error: unowned failure "
                "[explicit-any]\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(_TYPES, "run", run_mypy)
    cache_path = tmp_path / "cold-mypy-cache"
    completed = _TYPES._run_fixture_mypy(
        tmp_path,
        fixture_path,
        {},
        cache_path,
    )

    assert observed is not None
    assert "--follow-imports=silent" in observed
    assert f"--cache-dir={cache_path}" in observed
    assert observed[-1] == fixture_path
    assert completed.returncode == 1
    assert _TYPES._fixture_mypy_diagnostics(
        completed.stdout + completed.stderr, fixture_path
    ) == ("tests/owned_fixture.py:4:1: error: owned failure [assignment]",)


def test_patch_fixture_mypy_cache_is_shared_and_removed_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Share one cache per invocation and remove it after fixture failure."""
    fixture_root = tmp_path / "tests" / "patch_type_contracts"
    fixture_root.mkdir(parents=True)
    first_path = fixture_root / "first.py"
    second_path = fixture_root / "second.py"
    first_path.write_text("value: int = 1\n", encoding="utf-8")
    second_path.write_text("value: int = 2\n", encoding="utf-8")
    first = _TYPES.TypeFixture(
        identifier="PATCH-TEST-001",
        kind="positive",
        lifecycle="active",
        active_from_phase=0,
        path="tests/patch_type_contracts/first.py",
        source_sha256=sha256(first_path.read_bytes()).hexdigest(),
        expected_diagnostics=(),
    )
    second = _TYPES.TypeFixture(
        identifier="PATCH-TEST-002",
        kind="positive",
        lifecycle="active",
        active_from_phase=0,
        path="tests/patch_type_contracts/second.py",
        source_sha256=sha256(second_path.read_bytes()).hexdigest(),
        expected_diagnostics=(),
    )
    caches: list[Path] = []

    def run_mypy(
        root: Path,
        fixture_path: str,
        environment: dict[str, str],
        cache_path: Path,
    ) -> CompletedProcess[str]:
        assert root == tmp_path
        assert environment == {}
        assert cache_path.is_dir()
        caches.append(cache_path)
        if fixture_path == second.path:
            raise _TYPES.PatchTypeContractError("fixture failure")
        return CompletedProcess((), 0, stdout="", stderr="")

    monkeypatch.setattr(_TYPES, "_run_fixture_mypy", run_mypy)

    with pytest.raises(_TYPES.PatchTypeContractError, match="fixture failure"):
        _TYPES._verify_type_fixtures(tmp_path, (first, second), {})

    assert len(caches) == 2
    assert caches[0] == caches[1]
    assert not caches[0].exists()


def test_patch_acceptance_execution_disables_pytest_output_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run native worker acceptance without pytest descriptor capture."""
    root = tmp_path / "repository"
    test_path = root / "tests" / "sample_test.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        "def test_value() -> None:\n"
        "    value = object()\n"
        "    assert value.__class__ is object\n",
        encoding="utf-8",
    )
    node_id = "tests/sample_test.py::test_value"
    junit_path = root / "pytest.xml"
    calls: list[tuple[str, ...]] = []

    def run_pytest(
        observed_root: Path,
        arguments: tuple[str, ...],
        *,
        timeout: int,
        inherited_names: tuple[str, ...] = (),
    ) -> CompletedProcess[str]:
        assert observed_root == root
        assert timeout > 0
        assert inherited_names == ()
        calls.append(arguments)
        if "--collect-only" in arguments:
            return CompletedProcess(
                arguments, 0, stdout=node_id + "\n", stderr=""
            )
        junit_path.write_text(
            '<testsuite tests="1" failures="0" errors="0" '
            'skipped="0"><testcase file="tests/sample_test.py" '
            'classname="tests.sample_test" name="test_value" />'
            "</testsuite>",
            encoding="utf-8",
        )
        return CompletedProcess(arguments, 0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr(_CONTRACT_GATE, "run_pytest", run_pytest)

    evidence = _CONTRACT_GATE.execute_pytest_nodes(
        root,
        (node_id,),
        junit_path=junit_path,
    )

    assert evidence.collected == (node_id,)
    assert evidence.executed == (node_id,)
    assert len(calls) == 2
    collection, execution = calls
    assert "-s" not in collection
    assert execution[:4] == ("-q", "-s", "-r", "xXs")
    assert "junit_family=legacy" in execution
    assert f"--junitxml={junit_path}" in execution
    assert execution[-1] == node_id


def test_patch_gate_subprocess_rejects_dynamic_ignored_artifact_open(
    tmp_path: Path,
) -> None:
    """Block constructed spec opens without colliding with parent facts."""
    child_root = tmp_path / "child"
    child_root.mkdir()
    child_facts = child_root / ".patch-contract-pytest-facts.json"
    parent_facts = tmp_path / "parent" / ".patch-contract-pytest-facts.json"
    parent_facts.parent.mkdir()
    parent_facts.write_text("parent facts sentinel\n", encoding="utf-8")
    test_path = tmp_path / "test_dynamic_patch_artifact.py"
    test_path.write_text(
        "from os import environ\n"
        "from pathlib import Path\n\n"
        "def test_dynamic_ignored_artifact_open() -> None:\n"
        "    name = ''.join(('PATCH', '.md'))\n"
        "    target = Path(environ['AVALAN_PATCH_CONTRACT_ARTIFACT_ROOT'])\n"
        "    target = target / 'specs' / name\n"
        "    target.open()\n",
        encoding="utf-8",
    )
    environment = _patch_startup_environment()
    environment["AVALAN_PATCH_CONTRACT_PYTEST_FACTS_PATH"] = str(child_facts)
    completed = run(
        (
            executable,
            "-m",
            "pytest",
            "-p",
            "patch_contract_gate_plugin",
            "-q",
            str(test_path),
        ),
        cwd=child_root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode != 0
    assert "patch contract guard rejected ignored artifact open" in (
        completed.stdout + completed.stderr
    )
    assert child_facts.is_file()
    assert (
        parent_facts.read_text(encoding="utf-8") == "parent facts sentinel\n"
    )


@pytest.mark.parametrize("case", ("digest", "relative", "outside"))
def test_patch_gate_subprocess_rejects_invalid_artifact_guard_state(
    tmp_path: Path,
    case: str,
) -> None:
    """Reject tampered, relative, and outside-root forbidden path state."""
    environment = _patch_startup_environment()
    digest_name = _GATE._PATCH_FORBIDDEN_ARTIFACTS_SHA256_ENV
    forbidden_name = _GATE._PATCH_FORBIDDEN_ARTIFACTS_ENV
    expected = "patch artifact guard paths are invalid"
    if case == "digest":
        environment[digest_name] = "0" * 64
        expected = "patch artifact guard integrity is invalid"
    else:
        forbidden = environment[forbidden_name].split(pathsep)
        forbidden[0] = (
            "specs/PATCH.md"
            if case == "relative"
            else str(tmp_path.parent / "outside-patch-artifact")
        )
        value = pathsep.join(forbidden)
        environment[forbidden_name] = value
        environment[digest_name] = sha256(value.encode("utf-8")).hexdigest()

    completed = run(
        (executable, "-c", "pass"),
        cwd=tmp_path,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode != 0
    assert expected in completed.stderr


def test_preflight_executes_no_pytest_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate manifest state before any covered test process."""
    monkeypatch.delenv(_GATE.POSTGRESQL_TEST_DSN_ENV, raising=False)
    monkeypatch.delenv(_GATE._LEGACY_POSTGRESQL_LEASE_ENV, raising=False)
    monkeypatch.setattr(
        _GATE,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "preflight must not execute pytest"
        ),
    )
    _GATE.preflight(_GATE._PATCH_CURRENT_PHASE, repo_root=_ROOT)


def test_preflight_accepts_caller_database_after_durability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject early database state and accept it after durability activates."""
    monkeypatch.setenv(
        _GATE.POSTGRESQL_TEST_DSN_ENV,
        "postgresql://test/patch_phase2",
    )

    with pytest.raises(
        _GATE.PatchContractGateError,
        match="reject caller-supplied PostgreSQL state",
    ):
        _GATE._reject_invalid_database_context(7)

    _GATE.preflight(_GATE._PATCH_CURRENT_PHASE, repo_root=_ROOT)

    monkeypatch.delenv(_GATE.POSTGRESQL_TEST_DSN_ENV)
    _GATE.preflight(_GATE._PATCH_CURRENT_PHASE, repo_root=_ROOT)


def test_baseline_section2_evidence_rejects_source_drift(
    tmp_path: Path,
) -> None:
    """Reject a re-audit fact whose pinned implementation evidence changed."""
    fixtures = tmp_path / "fixtures"
    _copy_fixtures(fixtures)
    path = fixtures / "baseline_evidence.json"
    payload = loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    facts = payload["section2_facts"]
    assert isinstance(facts, list)
    first = facts[0]
    assert isinstance(first, dict)
    markers = first["source_markers"]
    assert isinstance(markers, list)
    marker = markers[0]
    assert isinstance(marker, dict)
    marker["source_sha256"] = "0" * 64
    _resign(payload, "evidence_sha256")
    _write(path, payload)

    with pytest.raises(
        _ACCEPTANCE.PatchAcceptanceError,
        match="Section 2 source digest drifted",
    ):
        _ACCEPTANCE.load_phase0_contracts(fixtures, repo_root=_ROOT)


def test_gate_deletes_stale_artifacts_and_runs_one_coverage_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seal one fresh report and reject reuse of stale full-suite artifacts."""
    source = tmp_path / "source"
    mirror = tmp_path / "mirror"
    source.mkdir()
    mirror.mkdir()
    _minimal_repository(source)
    _minimal_repository(mirror)
    database = _GATE.PostgreSQLTestDatabase(
        dsn="postgresql://test/patch_contract",
        name="patch_contract",
    )
    for name in ("coverage.json", "coverage.xml", _GATE._PYTEST_FACTS):
        (source / name).write_text("stale\n", encoding="utf-8")
    commands: list[tuple[str, ...]] = []
    stages: list[str] = []

    def run_command(
        root: Path,
        command: tuple[str, ...],
        supplied_database: object,
        *,
        facts_path: Path | None = None,
        evidence_artifacts: tuple[Path, Path, Path] | None = None,
    ) -> int:
        assert root == mirror
        assert supplied_database == database
        commands.append(command)
        if command[1:3] == ("-m", "pytest"):
            stages.append("coverage")
            assert facts_path == mirror / _GATE._PYTEST_FACTS
            assert facts_path is not None
            (mirror / "coverage.json").write_text("{}\n", encoding="utf-8")
            (mirror / "coverage.xml").write_text(
                "<coverage/>\n", encoding="utf-8"
            )
            facts_path.write_text(_facts(), encoding="utf-8")
        else:
            stages.append("coverage-verifier")
            assert facts_path is None
            assert evidence_artifacts is None
        return 0

    def verify_current(
        root: Path,
        phase: int,
        supplied_database: object,
        sealed: tuple[object, ...],
        inventory: object,
        python_ownership: tuple[PurePosixPath, ...],
    ) -> None:
        assert root == mirror
        assert phase == 0
        assert supplied_database == database
        assert len(sealed) == len(_GATE._SEALED_ARTIFACTS)
        assert isinstance(inventory, _GATE.SealedInputInventory)
        assert inventory.sha256
        assert python_ownership == ()
        stages.append("current-contract-verifiers")

    monkeypatch.setattr(_GATE, "_run_isolated_command", run_command)
    monkeypatch.setattr(
        _GATE, "_run_current_contract_verifiers", verify_current
    )
    monkeypatch.setattr(
        _GATE, "verify_pytest_module_name_uniqueness", lambda root: None
    )
    monkeypatch.setattr(
        _GATE, "verify_report_after_inventory", lambda *args: None
    )
    monkeypatch.setattr(_GATE, "_validate_patch_phase", lambda *args: None)

    assert _GATE._run_mirrored_gate(source, mirror, 0, database, ()) == 0
    coverage_commands = [
        command for command in commands if command[1:3] == ("-m", "pytest")
    ]
    assert len(coverage_commands) == 1
    assert coverage_commands[0].count("patch_contract_gate_plugin") == 1
    assert (source / "coverage.json").read_text(encoding="utf-8") == "{}\n"
    assert (source / _GATE._PYTEST_FACTS).read_text(
        encoding="utf-8"
    ) == _facts()
    assert stages[0] == "coverage"
    assert stages[-1] == "current-contract-verifiers"
    assert stages[1:-1]
    assert all(stage == "coverage-verifier" for stage in stages[1:-1])


def test_gate_owns_database_before_coverage_until_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the gate-owned database alive through full-suite execution."""
    database = _GATE.PostgreSQLTestDatabase(
        dsn="postgresql://test/patch_contract",
        name="patch_contract",
    )
    stages: list[str] = []

    @contextmanager
    def owned_database() -> Iterator[object]:
        stages.append("database-enter")
        yield database
        stages.append("database-exit")

    def run_with_database(
        source: Path,
        phase: int,
        supplied: object,
    ) -> int:
        assert source == tmp_path
        assert phase == 0
        assert supplied == database
        stages.extend(("coverage", "current-contract-verifiers"))
        return 0

    monkeypatch.setattr(_GATE, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _GATE, "_remove_gate_artifacts", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(_GATE, "_external_database", lambda phase: None)
    monkeypatch.setattr(_GATE, "_owned_postgresql_database", owned_database)
    monkeypatch.setattr(_GATE, "_run_gate_with_database", run_with_database)

    assert _GATE.run_gate(0, repo_root=tmp_path) == 0
    assert stages == [
        "database-enter",
        "coverage",
        "current-contract-verifiers",
        "database-exit",
    ]


def test_outcome_plugin_records_all_required_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record passed, skipped, xfail, xpass, warning, and leak evidence."""
    facts_path = tmp_path / ".patch-contract-pytest-facts.json"
    warning = SimpleNamespace(category=ResourceWarning)
    collection_skip = _PLUGIN.CollectReport(
        nodeid="tests/optional.py",
        outcome="skipped",
        longrepr="optional dependency is unavailable",
        result=[],
        sections=(),
        duration=0,
    )
    reporter = SimpleNamespace(
        stats={
            "passed": [
                SimpleNamespace(wasxfail=None),
                SimpleNamespace(wasxfail="x"),
            ],
            "skipped": [
                SimpleNamespace(wasxfail=None),
                collection_skip,
                SimpleNamespace(wasxfail="x"),
            ],
            "deselected": [SimpleNamespace()],
            "warnings": [warning],
        }
    )
    session = SimpleNamespace(
        testscollected=4,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(
                get_plugin=lambda name: (
                    reporter if name == "terminalreporter" else None
                )
            )
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "AVALAN_PATCH_CONTRACT_PYTEST_FACTS_PATH", str(facts_path)
    )
    _PLUGIN.pytest_sessionfinish(session, 0)
    assert loads(facts_path.read_text(encoding="utf-8")) == {
        "collected": 4,
        "collection_skipped": 1,
        "deselected": 1,
        "errors": 0,
        "exitstatus": 0,
        "failed": 0,
        "leak_warnings": 1,
        "passed": 1,
        "schema_version": 1,
        "skipped": 2,
        "warnings": 1,
        "xfailed": 1,
        "xpassed": 1,
    }


def test_outcome_balance_counts_collection_skips_outside_collected_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Balance `collected N / 1 skipped` using pytest's terminal model."""
    facts_path = tmp_path / ".patch-contract-pytest-facts.json"
    collection_skip = _PLUGIN.CollectReport(
        nodeid="tests/optional.py",
        outcome="skipped",
        longrepr="optional dependency is unavailable",
        result=[],
        sections=(),
        duration=0,
    )
    reporter = SimpleNamespace(
        stats={
            "passed": [SimpleNamespace(), SimpleNamespace()],
            "skipped": [SimpleNamespace(), collection_skip],
        }
    )
    session = SimpleNamespace(
        testscollected=3,
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(
                get_plugin=lambda name: (
                    reporter if name == "terminalreporter" else None
                )
            )
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "AVALAN_PATCH_CONTRACT_PYTEST_FACTS_PATH", str(facts_path)
    )

    _PLUGIN.pytest_sessionfinish(session, 0)

    facts = _GATE._read_pytest_facts(facts_path)
    assert facts.collected == 3
    assert facts.skipped == 2
    assert facts.collection_skipped == 1
    _GATE._verify_pytest_facts(facts)


def test_nonexistent_active_e2e_fails_at_collection_before_execution(
    tmp_path: Path,
) -> None:
    """Fail an active-but-uncollectable E2E before it can count as evidence."""
    fixtures = tmp_path / "fixtures"
    _copy_fixtures(fixtures)
    manifest_path = fixtures / "acceptance_manifest.json"
    manifest = loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    nodes = manifest["nodes"]
    assert isinstance(nodes, list)
    first = nodes[0]
    assert isinstance(first, dict)
    original = first["node_id"]
    assert isinstance(original, str)
    missing = original + "[uncollectable]"
    first["node_id"] = missing
    _resign(manifest, "manifest_sha256")
    _write(manifest_path, manifest)

    for name, digest_field in (
        ("requirements_traceability.json", "catalog_sha256"),
        ("surface_conformance.json", "manifest_sha256"),
        ("phase_evidence.json", "record_sha256"),
    ):
        path = fixtures / name
        payload = loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        replacement = dumps(payload).replace(original, missing)
        mutated = loads(replacement)
        assert isinstance(mutated, dict)
        _resign(mutated, digest_field)
        _write(path, mutated)

    index_path = fixtures / "phase_evidence_index.json"
    index = loads(index_path.read_text(encoding="utf-8"))
    assert isinstance(index, dict)
    records = index["records"]
    assert isinstance(records, list)
    current = records[-1]
    assert isinstance(current, dict)
    evidence = loads(
        (fixtures / "phase_evidence.json").read_text(encoding="utf-8")
    )
    assert isinstance(evidence, dict)
    current["record_sha256"] = evidence["record_sha256"]
    current["file_sha256"] = sha256(
        (fixtures / "phase_evidence.json").read_bytes()
    ).hexdigest()
    _resign(index, "index_sha256")
    _write(index_path, index)

    completed = run(
        (
            executable,
            "scripts/verify_patch_acceptance.py",
            "--through-phase",
            str(_GATE._PATCH_CURRENT_PHASE),
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(_ROOT),
        ),
        cwd=_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 1
    assert (
        "acceptance history executable node is not exact" in completed.stderr
    )
    assert "no tests ran" not in completed.stdout


def test_public_make_target_rejects_missing_or_premature_phase() -> None:
    """Reject public target invocations that bypass Phase 0 selection rules."""
    missing = run(
        ("make", "test-patch-exact", "no-install"),
        cwd=_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert missing.returncode != 0
    assert "PATCH_PHASE is required" in missing.stderr
    premature = run(
        (
            "make",
            "test-patch-pgsql-exact",
            "no-install",
            "PATCH_PHASE=0",
        ),
        cwd=_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert premature.returncode != 0
    assert (
        "patch phases 0 through 7 require test-patch-exact" in premature.stderr
    )
