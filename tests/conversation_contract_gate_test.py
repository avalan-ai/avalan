"""Exercise reusable sealed conversation gate evidence."""

from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from os import pathsep, utime
from pathlib import Path
from subprocess import CompletedProcess, run
from sys import executable, modules
from types import ModuleType
from venv import EnvBuilder
from xml.etree.ElementTree import Element, SubElement

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MISSING_EVIDENCE_BODY = "assert 1 + 1 == 2"
_WRONG_EVIDENCE_LINES = (
    "record_property('conversation_acceptance_evidence', 'wrong')",
    "assert 1 + 1 == 2",
)
_DUPLICATE_EVIDENCE_LINES = (
    "record_property('conversation_acceptance_evidence', 'runtime')",
    "record_property('conversation_acceptance_evidence', 'runtime')",
    "assert 1 + 1 == 2",
)
_WRONG_EVIDENCE_BODY = "\n    ".join(_WRONG_EVIDENCE_LINES)
_DUPLICATE_EVIDENCE_BODY = "\n    ".join(_DUPLICATE_EVIDENCE_LINES)
_EVIDENCE_ONLY_BODY = (
    "record_property('conversation_acceptance_evidence', 'runtime')"
)


def _load_gate() -> ModuleType:
    """Return the reusable contract gate as an importable module."""
    name = "_conversation_contract_gate"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "contract_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate()


def _configured_pytest_collection(root: Path) -> CompletedProcess[str]:
    """Collect with repository-controlled pytest configuration."""
    return run(
        (executable, "-m", "pytest", "--collect-only", "-q"),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )


def _hardened_pytest_collection(root: Path) -> CompletedProcess[str]:
    """Collect with the exact shared contract-gate pytest boundary."""
    completed: CompletedProcess[str] = _GATE.run_pytest(
        root,
        ("--collect-only", "-q", "."),
        timeout=30,
    )
    return completed


def _repository(tmp_path: Path) -> Path:
    """Create the minimum measured repository inventory."""
    for relative, content in (
        ("src/sample.py", "VALUE = 1\n"),
        ("tests/sample_test.py", "def test_value() -> None:\n    pass\n"),
        ("tests/fixture.bin", "binary fixture\n"),
        ("scripts/gate.py", "VALUE = 1\n"),
        ("Makefile", "test:\n\ttrue\n"),
        ("pyproject.toml", "[project]\nname = 'sample'\n"),
        ("poetry.lock", "# lock\n"),
        ("pytest.ini", "[pytest]\n"),
        ("conftest.py", "VALUE = 1\n"),
        ("sitecustomize.py", "VALUE = 1\n"),
        ("poetry.toml", "[virtualenvs]\nin-project = true\n"),
        ("tox.ini", "[tox]\n"),
        ("setup.cfg", "[tool:pytest]\n"),
        (".coveragerc", "[run]\nbranch = true\n"),
        ("migrations/001.sql", "SELECT 1;\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def _initialize_git(root: Path) -> None:
    """Initialize one local repository for mirror enumeration."""
    run(("git", "init", "-q"), cwd=root, check=True)


def test_sealed_inventory_rejects_mid_run_mutation(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject byte mutation between pre- and post-run inventories."""
    record_property("conversation_acceptance_evidence", "runtime")
    root = _repository(tmp_path)
    before = _GATE.capture_input_inventory(root)
    (root / "tests" / "sample_test.py").write_text(
        "def test_value() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    after = _GATE.capture_input_inventory(root)
    with pytest.raises(
        _GATE.ContractGateError,
        match="mutated=.*tests/sample_test.py",
    ):
        _GATE.verify_input_inventory(before, after)


def test_sealed_inventory_accepts_unchanged_inputs(tmp_path: Path) -> None:
    """Accept identical complete pre- and post-run inventories."""
    root = _repository(tmp_path)
    inventory = _GATE.capture_input_inventory(root)
    _GATE.verify_input_inventory(inventory, inventory)
    assert inventory.entries
    assert len(inventory.sha256) == 64
    assert "tests/fixture.bin" in {entry.path for entry in inventory.entries}
    assert {
        ".coveragerc",
        "conftest.py",
        "migrations/001.sql",
        "poetry.toml",
        "pytest.ini",
        "setup.cfg",
        "sitecustomize.py",
        "tox.ini",
    } <= {entry.path for entry in inventory.entries}


@pytest.mark.parametrize("change", ("added", "removed"))
def test_sealed_inventory_rejects_path_changes(
    tmp_path: Path,
    change: str,
) -> None:
    """Reject added and removed measured paths."""
    root = _repository(tmp_path)
    before = _GATE.capture_input_inventory(root)
    changed = root / "scripts" / "new.py"
    if change == "added":
        changed.write_text("VALUE = 2\n", encoding="utf-8")
    else:
        changed = root / "scripts" / "gate.py"
        changed.unlink()
    after = _GATE.capture_input_inventory(root)
    with pytest.raises(_GATE.ContractGateError, match=f"{change}="):
        _GATE.verify_input_inventory(before, after)


def test_sealed_inventory_requires_gate_metadata(tmp_path: Path) -> None:
    """Reject an inventory missing an authoritative metadata input."""
    root = _repository(tmp_path)
    (root / "poetry.lock").unlink()
    with pytest.raises(
        _GATE.ContractGateError,
        match="required measured gate inputs are missing",
    ):
        _GATE.capture_input_inventory(root)


def test_repository_pytest_import_module_names_are_unique() -> None:
    """Keep full-suite pytest imports free of path-name collisions."""
    _GATE.verify_pytest_module_name_uniqueness(_ROOT)


def test_hardened_pytest_arguments_pin_full_suite_discovery() -> None:
    """Share one closed pytest config across preflight and full collection."""
    arguments = _GATE.hardened_pytest_arguments()
    assert arguments == (
        "-o",
        "addopts=",
        "-p",
        "avalan_contract_gate_plugin",
        "-c",
        "/dev/null",
        "--rootdir=.",
        "--noconftest",
        "--import-mode=prepend",
        "-o",
        "python_files=test_*.py *_test.py",
        "-o",
        (
            "norecursedirs=*.egg .* _darcs build CVS dist "
            "node_modules venv {arch}"
        ),
    )
    coverage = _GATE.exact_coverage_commands()[0]
    assert coverage[11 : 11 + len(arguments)] == arguments
    assert coverage[-1] == "."


def test_hardened_pytest_ignores_python_file_config_override(
    tmp_path: Path,
) -> None:
    """Ignore repository config that expands Python test file patterns."""
    sentinel = tmp_path / "tests" / "sentinel_test.py"
    specs = tuple(
        tmp_path / parent / "contract_spec.py"
        for parent in ("tests", "checks")
    )
    for path in (sentinel, *specs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\npython_files = *_spec.py\n", encoding="utf-8"
    )

    configured = _configured_pytest_collection(tmp_path)
    assert configured.returncode == 2
    assert "import file mismatch" in configured.stdout + configured.stderr
    assert _GATE._pytest_test_module_paths(tmp_path) == (sentinel,)
    _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    hardened = _hardened_pytest_collection(tmp_path)
    assert hardened.returncode == 0
    assert "tests/sentinel_test.py::test_value" in hardened.stdout
    assert "contract_spec.py" not in hardened.stdout + hardened.stderr


def test_hardened_pytest_ignores_norecursedirs_config_override(
    tmp_path: Path,
) -> None:
    """Keep default build exclusion closed against repository config."""
    module = tmp_path / "tests" / "contract_test.py"
    build_module = tmp_path / "build" / "contract_test.py"
    for path in (module, build_module):
        path.parent.mkdir(parents=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nnorecursedirs =\n", encoding="utf-8"
    )

    configured = _configured_pytest_collection(tmp_path)
    assert configured.returncode == 2
    assert "import file mismatch" in configured.stdout + configured.stderr
    assert _GATE._pytest_test_module_paths(tmp_path) == (module,)
    _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    hardened = _hardened_pytest_collection(tmp_path)
    assert hardened.returncode == 0
    assert "tests/contract_test.py::test_value" in hardened.stdout
    assert "build/contract_test.py" not in hardened.stdout + hardened.stderr


def test_hardened_pytest_disables_repository_collection_hooks(
    tmp_path: Path,
) -> None:
    """Disable conftest hooks that expand collection beyond preflight."""
    sentinel = tmp_path / "tests" / "sentinel_test.py"
    specs = tuple(
        tmp_path / parent / "contract_spec.py"
        for parent in ("tests", "checks")
    )
    for path in (sentinel, *specs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )
    (tmp_path / "conftest.py").write_text(
        "import pytest\n\n"
        "def pytest_collect_file(file_path, parent):\n"
        "    if file_path.name.endswith('_spec.py'):\n"
        "        return pytest.Module.from_parent(parent, path=file_path)\n"
        "    return None\n",
        encoding="utf-8",
    )

    configured = _configured_pytest_collection(tmp_path)
    assert configured.returncode == 2
    assert "import file mismatch" in configured.stdout + configured.stderr
    assert _GATE._pytest_test_module_paths(tmp_path) == (sentinel,)
    _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    hardened = _hardened_pytest_collection(tmp_path)
    assert hardened.returncode == 0
    assert "tests/sentinel_test.py::test_value" in hardened.stdout
    assert "contract_spec.py" not in hardened.stdout + hardened.stderr


@pytest.mark.parametrize(
    ("declaration", "preflight_rejects"),
    (
        ("pytest_plugins = ('collection_plugin',)", True),
        (
            "globals()['pytest_' + 'plugins'] = ('collection_plugin',)",
            False,
        ),
    ),
)
def test_hardened_pytest_rejects_test_module_plugin_declarations(
    tmp_path: Path,
    declaration: str,
    preflight_rejects: bool,
) -> None:
    """Block collected modules from registering later collection hooks."""
    bootstrap = tmp_path / "test_00_bootstrap.py"
    sentinel = tmp_path / "tests" / "sentinel_test.py"
    specs = tuple(
        tmp_path / parent / "contract_spec.py"
        for parent in ("zchecks", "ztests")
    )
    for path in (sentinel, *specs):
        path.parent.mkdir(parents=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )
    bootstrap.write_text(
        f"{declaration}\n\ndef test_bootstrap() -> None:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "collection_plugin.py").write_text(
        "import pytest\n\n"
        "def pytest_collect_file(file_path, parent):\n"
        "    if file_path.name.endswith('_spec.py'):\n"
        "        return pytest.Module.from_parent(parent, path=file_path)\n"
        "    return None\n",
        encoding="utf-8",
    )

    configured = _configured_pytest_collection(tmp_path)
    assert configured.returncode == 2
    assert "import file mismatch" in configured.stdout + configured.stderr
    if preflight_rejects:
        with pytest.raises(
            _GATE.ContractGateError,
            match="must not declare pytest_plugins",
        ):
            _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    else:
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    hardened = _hardened_pytest_collection(tmp_path)
    output = hardened.stdout + hardened.stderr
    assert hardened.returncode == 2
    assert "module-level pytest_plugins are disabled" in output
    assert "import file mismatch" not in output


def test_pytest_module_preflight_rejects_out_of_tree_duplicates(
    tmp_path: Path,
) -> None:
    """Reject the exact collision discovered by bare repository pytest."""
    modules = tuple(
        tmp_path / parent / "contract_test.py"
        for parent in ("tests", "checks")
    )
    for path in modules:
        path.parent.mkdir(parents=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    assert "import file mismatch" in collection.stdout + collection.stderr
    with pytest.raises(
        _GATE.ContractGateError,
        match="pytest import module names are duplicated",
    ) as error:
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    assert "checks/contract_test.py" in str(error.value)
    assert "tests/contract_test.py" in str(error.value)


def test_pytest_module_preflight_honors_default_recursion_exclusions(
    tmp_path: Path,
) -> None:
    """Skip only pytest's default ignored and virtual environment trees."""
    root = tmp_path / "repository"
    module = root / "tests" / "contract_test.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "def test_value() -> None:\n    pass\n", encoding="utf-8"
    )
    for relative in (
        ".git",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "package.egg",
        "venv",
    ):
        ignored = root / relative / "contract_test.py"
        ignored.parent.mkdir(parents=True)
        ignored.write_text(
            "def test_ignored() -> None:\n    pass\n", encoding="utf-8"
        )
    virtual_environment = root / "custom_environment"
    (virtual_environment / "pyvenv.cfg").parent.mkdir(parents=True)
    (virtual_environment / "pyvenv.cfg").write_text(
        "home = /tmp\n", encoding="utf-8"
    )
    (virtual_environment / "contract_test.py").write_text(
        "def test_ignored() -> None:\n    pass\n", encoding="utf-8"
    )
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    assert _GATE._pytest_test_module_paths(root) == (module,)
    collection = _hardened_pytest_collection(root)
    assert collection.returncode == 0
    assert "tests/contract_test.py::test_value" in collection.stdout
    assert "test_ignored" not in collection.stdout + collection.stderr
    _GATE.verify_pytest_module_name_uniqueness(root)


@pytest.mark.parametrize("directory", ("builder", "htmlcov"))
def test_pytest_module_preflight_recurses_near_exclusion_names(
    tmp_path: Path,
    directory: str,
) -> None:
    """Keep similarly named and non-pytest artifact directories in scope."""
    first = tmp_path / "tests" / "contract_test.py"
    second = tmp_path / directory / "contract_test.py"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    assert "import file mismatch" in collection.stdout + collection.stderr
    with pytest.raises(
        _GATE.ContractGateError,
        match="pytest import module names are duplicated",
    ):
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)


def test_pytest_module_preflight_rejects_collectable_symlink_directory(
    tmp_path: Path,
) -> None:
    """Fail closed instead of following a collectable symlink tree."""
    root = tmp_path / "repository"
    module = root / "tests" / "contract_test.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "def test_value() -> None:\n    pass\n", encoding="utf-8"
    )
    external = tmp_path / "external"
    external.mkdir()
    (external / "contract_test.py").write_text(
        "def test_value() -> None:\n    pass\n", encoding="utf-8"
    )
    ignored = root / ".venv"
    ignored.symlink_to(external, target_is_directory=True)
    _GATE.verify_pytest_module_name_uniqueness(root)
    ignored_collection = _hardened_pytest_collection(root)
    assert ignored_collection.returncode == 0
    assert "tests/contract_test.py::test_value" in ignored_collection.stdout
    ignored.unlink()
    (root / "checks").symlink_to(external, target_is_directory=True)

    collection = _hardened_pytest_collection(root)
    assert collection.returncode == 2
    assert "import file mismatch" in collection.stdout + collection.stderr
    with pytest.raises(
        _GATE.ContractGateError,
        match="collection directories must not be symbolic links",
    ):
        _GATE.verify_pytest_module_name_uniqueness(root)


def test_pytest_module_name_preflight_rejects_plain_duplicates(
    tmp_path: Path,
) -> None:
    """Reject plain duplicates while allowing package-qualified modules."""
    first = tmp_path / "tests" / "first" / "contract_test.py"
    second = tmp_path / "tests" / "second" / "contract_test.py"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    assert "import file mismatch" in collection.stdout + collection.stderr
    with pytest.raises(
        _GATE.ContractGateError,
        match="pytest import module names are duplicated",
    ):
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)

    (second.parent / "__init__.py").write_text("", encoding="utf-8")
    _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    assert _hardened_pytest_collection(tmp_path).returncode == 0

    (first.parent / "__init__.py").write_text("", encoding="utf-8")
    assert _GATE._pytest_import_module_name(first) == "first.contract_test"
    assert _GATE._pytest_import_module_name(second) == "second.contract_test"
    _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    assert _hardened_pytest_collection(tmp_path).returncode == 0


def test_pytest_preflight_rejects_colliding_package_prefixes(
    tmp_path: Path,
) -> None:
    """Reject distinct leaf modules whose package roots bind differently."""
    modules = (
        tmp_path / "tests" / "one" / "pkg" / "alpha_test.py",
        tmp_path / "tests" / "two" / "pkg" / "beta_test.py",
    )
    for path in modules:
        path.parent.mkdir(parents=True)
        (path.parent / "__init__.py").write_text("", encoding="utf-8")
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    assert {_GATE._pytest_import_module_name(path) for path in modules} == {
        "pkg.alpha_test",
        "pkg.beta_test",
    }
    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    with pytest.raises(
        _GATE.ContractGateError,
        match="shadow package prefixes",
    ) as error:
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    assert '"pkg"' in str(error.value)
    assert "package:tests/one/pkg" in str(error.value)
    assert "package:tests/two/pkg" in str(error.value)


def test_pytest_preflight_allows_shared_physical_package_prefix(
    tmp_path: Path,
) -> None:
    """Allow multiple leaf modules below one physical package directory."""
    package = tmp_path / "tests" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    modules = (package / "alpha_test.py", package / "beta_test.py")
    for path in modules:
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 0
    assert "pkg/alpha_test.py::test_value" in collection.stdout
    assert "pkg/beta_test.py::test_value" in collection.stdout


def test_pytest_preflight_unifies_plain_modules_and_package_prefixes(
    tmp_path: Path,
) -> None:
    """Reject one symbol bound as both a module and a package."""
    module = tmp_path / "tests" / "pkg_test.py"
    package_module = (
        tmp_path / "tests" / "other" / "pkg_test" / "alpha_test.py"
    )
    for path in (module, package_module):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )
    (package_module.parent / "__init__.py").write_text("", encoding="utf-8")

    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    with pytest.raises(
        _GATE.ContractGateError,
        match="shadow package prefixes",
    ) as error:
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    assert "module:tests/pkg_test.py" in str(error.value)
    assert "package:tests/other/pkg_test" in str(error.value)


def test_pytest_preflight_rejects_nested_package_prefix_collisions(
    tmp_path: Path,
) -> None:
    """Bind every initialized component of a nested package chain."""
    modules = (
        tmp_path / "tests" / "one" / "outer" / "inner" / "alpha_test.py",
        tmp_path / "tests" / "two" / "outer" / "inner" / "beta_test.py",
    )
    for path in modules:
        path.parent.mkdir(parents=True)
        (path.parent.parent / "__init__.py").write_text("", encoding="utf-8")
        (path.parent / "__init__.py").write_text("", encoding="utf-8")
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    with pytest.raises(_GATE.ContractGateError) as error:
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)
    message = str(error.value)
    assert '"outer"' in message
    assert '"outer.inner"' in message
    assert "package:tests/one/outer/inner" in message
    assert "package:tests/two/outer/inner" in message


@pytest.mark.parametrize(
    ("parents", "initialized_boundary"),
    (
        (("first-invalid", "second-invalid"), True),
        (("first_boundary", "second_boundary"), False),
    ),
    ids=("invalid-identifier", "uninitialized"),
)
def test_pytest_preflight_bounds_package_prefixes_at_invalid_boundaries(
    tmp_path: Path,
    parents: tuple[str, str],
    initialized_boundary: bool,
) -> None:
    """Stop package symbols at invalid and uninitialized boundaries."""
    modules = tuple(
        tmp_path / "tests" / parent / "pkg" / leaf
        for parent, leaf in zip(
            parents,
            ("alpha_test.py", "beta_test.py"),
            strict=True,
        )
    )
    for path in modules:
        path.parent.mkdir(parents=True)
        if initialized_boundary:
            (path.parent.parent / "__init__.py").write_text(
                "", encoding="utf-8"
            )
        (path.parent / "__init__.py").write_text("", encoding="utf-8")
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    bindings = tuple(_GATE._pytest_import_bindings(path) for path in modules)
    assert tuple(binding[0][0] for binding in bindings) == ("pkg", "pkg")
    assert tuple(binding[-1][0] for binding in bindings) == (
        "pkg.alpha_test",
        "pkg.beta_test",
    )
    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    with pytest.raises(
        _GATE.ContractGateError,
        match="shadow package prefixes",
    ):
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)


def test_pytest_module_preflight_rejects_invalid_package_names(
    tmp_path: Path,
) -> None:
    """Treat initialized non-identifiers as pytest's plain modules."""
    modules = tuple(
        tmp_path / "tests" / parent / "contract_test.py"
        for parent in ("first-invalid", "second-invalid")
    )
    for path in modules:
        path.parent.mkdir(parents=True)
        (path.parent / "__init__.py").write_text("", encoding="utf-8")
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    assert {_GATE._pytest_import_module_name(path) for path in modules} == {
        "contract_test"
    }
    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    assert "import file mismatch" in collection.stdout + collection.stderr
    with pytest.raises(
        _GATE.ContractGateError,
        match="pytest import module names are duplicated",
    ):
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)


def test_pytest_module_preflight_stops_at_invalid_package_parent(
    tmp_path: Path,
) -> None:
    """Keep valid nested package names below invalid parents bounded."""
    modules = tuple(
        tmp_path / "tests" / parent / "shared_package" / "contract_test.py"
        for parent in ("first-invalid", "second-invalid")
    )
    for path in modules:
        path.parent.mkdir(parents=True)
        (path.parent.parent / "__init__.py").write_text("", encoding="utf-8")
        (path.parent / "__init__.py").write_text("", encoding="utf-8")
        path.write_text(
            "def test_value() -> None:\n    pass\n", encoding="utf-8"
        )

    assert {_GATE._pytest_import_module_name(path) for path in modules} == {
        "shared_package.contract_test"
    }
    collection = _hardened_pytest_collection(tmp_path)
    assert collection.returncode == 2
    assert "import file mismatch" in collection.stdout + collection.stderr
    with pytest.raises(
        _GATE.ContractGateError,
        match="pytest import module names are duplicated",
    ):
        _GATE.verify_pytest_module_name_uniqueness(tmp_path)


def test_coverage_report_must_follow_sealed_inputs(tmp_path: Path) -> None:
    """Reject a report older than the sealed input inventory."""
    root = _repository(tmp_path)
    report = root / "coverage.json"
    report.write_text("{}\n", encoding="utf-8")
    inventory = _GATE.capture_input_inventory(root)
    old = max(0, inventory.newest_mtime_ns - 1)
    utime(report, ns=(old, old))
    with pytest.raises(_GATE.ContractGateError, match="predates the sealed"):
        _GATE.verify_report_after_inventory(report, inventory)
    fresh = inventory.newest_mtime_ns + 1
    utime(report, ns=(fresh, fresh))
    _GATE.verify_report_after_inventory(report, inventory)


@pytest.mark.parametrize(
    "body",
    (
        "pass",
        "...",
        "return",
        "return None",
        "return True",
        "assert True",
        "assert 1",
        _EVIDENCE_ONLY_BODY,
        _EVIDENCE_ONLY_BODY + "\n    pass",
        _EVIDENCE_ONLY_BODY + "\n    assert True",
    ),
)
def test_active_node_rejects_placeholder_only_body(
    tmp_path: Path,
    body: str,
) -> None:
    """Reject pass-only and ellipsis-only acceptance nodes."""
    root = _repository(tmp_path)
    (root / "tests" / "sample_test.py").write_text(
        f"def test_value() -> None:\n    {body}\n",
        encoding="utf-8",
    )

    with pytest.raises(_GATE.ContractGateError, match="placeholder-only"):
        _GATE.execute_pytest_nodes(
            root,
            ("tests/sample_test.py::test_value",),
            junit_path=root / "pytest.xml",
        )


@pytest.mark.parametrize(
    "body",
    (
        "value = 1",
        "fixture_alias = tmp_path",
        "def helper() -> None:\n        assert value",
        "class Helper:\n        assert value",
        "predicate = lambda: value",
        "assert 1 + 1 == 2",
        "observed = True\n    assert observed",
        "def helper() -> bool:\n        return True\n    assert helper",
        "def helper() -> bool:\n        return True\n    assert helper()",
        "payload = {'ok': True}\n    assert payload['ok']",
        "if False:\n        value = 1\n        assert value == 1",
    ),
    ids=(
        "assignment-only",
        "fixture-alias-only",
        "helper-definition-only",
        "class-definition-only",
        "lambda-only",
        "constant-expression-assert",
        "literal-bound-name-assert",
        "local-helper-reference",
        "local-helper-call",
        "literal-alias-subscript",
        "unreachable-assert",
    ),
)
def test_active_node_requires_positive_executable_invariant(
    tmp_path: Path,
    body: str,
) -> None:
    """Reject bodies without a reachable runtime assertion or raises block."""
    root = _repository(tmp_path)
    (root / "tests" / "sample_test.py").write_text(
        f"def test_value(tmp_path) -> None:\n    {body}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _GATE.ContractGateError,
        match="positive executable invariant",
    ):
        _GATE._validate_node_sources(
            root,
            ("tests/sample_test.py::test_value",),
            None,
        )


@pytest.mark.parametrize(
    ("preamble", "body"),
    (
        (
            "",
            "value = object()\n    assert value.__class__ is object",
        ),
        (
            "",
            (
                "value = object()\n"
                "    if True:\n"
                "        assert value.__class__ is object"
            ),
        ),
        (
            "import pytest\n\n",
            "with pytest.raises(ValueError):\n        raise ValueError",
        ),
        (
            "",
            "for value in (1,):\n        assert value.bit_length() == 1",
        ),
        (
            "",
            (
                "for value in ():\n"
                "        assert value.__class__ is object\n"
                "    else:\n"
                "        current = object()\n"
                "        assert current.__class__ is object"
            ),
        ),
    ),
    ids=(
        "runtime-assert",
        "literal-control-flow-assert",
        "pytest-raises",
        "nonempty-literal-iterable",
        "empty-iterable-orelse",
    ),
)
def test_active_node_accepts_positive_executable_invariant(
    tmp_path: Path,
    preamble: str,
    body: str,
) -> None:
    """Accept either supported positive invariant in ordinary control flow."""
    root = _repository(tmp_path)
    (root / "tests" / "sample_test.py").write_text(
        f"{preamble}def test_value() -> None:\n    {body}\n",
        encoding="utf-8",
    )

    _GATE._validate_node_sources(
        root,
        ("tests/sample_test.py::test_value",),
        None,
    )


@pytest.mark.parametrize(
    "body",
    (
        "return\n    assert object()",
        "raise RuntimeError\n    assert object()",
        "for value in ():\n        assert object()",
        "if True:\n        return\n    assert object()",
        "condition = bool()\n    if condition:\n        assert object()",
        "items = ()\n    for item in items:\n        assert object()",
        "for value in ():\n        if value:\n            assert object()",
    ),
    ids=(
        "return-before-assert",
        "raise-before-assert",
        "empty-loop",
        "literal-branch-return-before-assert",
        "runtime-conditional-branch-only",
        "literal-alias-empty-loop",
        "empty-loop-nested-assert",
    ),
)
def test_active_node_rejects_unreachable_invariant_after_evidence(
    tmp_path: Path,
    body: str,
) -> None:
    """Reject evidence followed only by unreachable positive syntax."""
    root = _repository(tmp_path)
    node_id = "tests/sample_test.py::test_value"
    (root / "tests" / "sample_test.py").write_text(
        "def test_value(record_property) -> None:\n"
        "    record_property(\n"
        "        'conversation_acceptance_evidence', 'runtime'\n"
        "    )\n"
        f"    {body}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _GATE.ContractGateError,
        match="positive executable invariant",
    ):
        _GATE._validate_node_sources(
            root,
            (node_id,),
            {node_id: "runtime"},
        )


@pytest.mark.parametrize(
    "iterable",
    (
        "()",
        "[]",
        "set()",
        "{}",
        "''",
        "b''",
        "range(0)",
        "range(2, 2)",
        "range(2, 1)",
        "range(1, 2, -1)",
    ),
)
def test_active_node_rejects_literal_empty_loop_invariant(
    tmp_path: Path,
    iterable: str,
) -> None:
    """Reject invariants reachable only through a known empty iterable."""
    root = _repository(tmp_path)
    node_id = "tests/sample_test.py::test_value"
    (root / "tests" / "sample_test.py").write_text(
        "def test_value(record_property) -> None:\n"
        "    record_property(\n"
        "        'conversation_acceptance_evidence', 'runtime'\n"
        "    )\n"
        f"    for value in {iterable}:\n"
        "        assert object()\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _GATE.ContractGateError,
        match="positive executable invariant",
    ):
        _GATE._validate_node_sources(
            root,
            (node_id,),
            {node_id: "runtime"},
        )


def test_sealed_inventory_rejects_symbolic_link(tmp_path: Path) -> None:
    """Reject a measured input whose bytes can change outside the seal."""
    root = _repository(tmp_path)
    (root / "tests" / "linked.py").symlink_to(root / "src" / "sample.py")

    with pytest.raises(
        _GATE.ContractGateError,
        match="cannot be symbolic links",
    ):
        _GATE.capture_input_inventory(root)


def test_execution_mirror_includes_current_nonignored_bytes_only(
    tmp_path: Path,
) -> None:
    """Execute from modified and untracked bytes without ignored material."""
    root = _repository(tmp_path)
    _initialize_git(root)
    (root / ".gitignore").write_text(
        "private/\nignored.py\n",
        encoding="utf-8",
    )
    run(("git", "add", "src/sample.py"), cwd=root, check=True)
    (root / "src" / "sample.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    (root / "untracked.txt").write_text("current\n", encoding="utf-8")
    (root / "ignored.py").write_text("SECRET = 1\n", encoding="utf-8")
    private = root / "private"
    private.mkdir()
    private_name = "secret" + "." + "m" + "d"
    (private / private_name).write_text("private\n", encoding="utf-8")

    with _GATE.nonignored_execution_mirror(root) as mirror:
        mirror_path = mirror
        assert (mirror / "src" / "sample.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 2\n"
        assert (mirror / "untracked.txt").read_text(
            encoding="utf-8"
        ) == "current\n"
        assert not (mirror / "ignored.py").exists()
        assert not (mirror / "private").exists()
        assert not (mirror / ".git").exists()

    assert not mirror_path.exists()


def test_execution_mirror_ignores_ambient_tmpdir_inside_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep mirror, child cwd, and child env outside an ambient TMPDIR."""
    root = _repository(tmp_path)
    _initialize_git(root)
    (root / ".gitignore").write_text("private/\n", encoding="utf-8")
    private = root / "private"
    private.mkdir()
    secret = private / "provider-secret"
    secret.write_text("secret\n", encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(private))

    with _GATE.nonignored_execution_mirror(root) as mirror:
        assert mirror.resolve() != root.resolve()
        assert not mirror.resolve().is_relative_to(root.resolve())
        assert not (mirror / "private").exists()
        with _GATE.isolated_subprocess_environment(
            mirror,
            trusted_python_root=_ROOT,
        ) as environment:
            script = (
                "from os import environ\n"
                "from pathlib import Path\n"
                "cwd = Path.cwd()\n"
                "assert cwd == Path(environ['PWD'])\n"
                "allowed = environ['PYTHONPATH'].split(':')\n"
                "assert Path(allowed[1]) == cwd / 'src'\n"
                "search_roots = (cwd, *cwd.parents)\n"
                "found = tuple(\n"
                "    root / 'private' / 'provider-secret'\n"
                "    for root in search_roots\n"
                "    if (root / 'private' / 'provider-secret').is_file()\n"
                ")\n"
                "assert not found\n"
            )
            completed = run(
                (executable, "-c", script),
                cwd=mirror,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
            assert str(private) not in environment.values()
            assert str(secret) not in environment.values()


def test_execution_mirror_rejects_source_descendant_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed before copying when the configured mirror is in source."""
    root = _repository(tmp_path)
    _initialize_git(root)
    private = root / "private"
    private.mkdir()
    monkeypatch.setattr(
        _GATE,
        "_TRUSTED_TEMPORARY_ROOT",
        private.resolve(),
    )

    with pytest.raises(
        _GATE.ContractGateError,
        match="mirror must be outside the source repository",
    ):
        with _GATE.nonignored_execution_mirror(root):
            pytest.fail("source-descendant mirror must not be yielded")


def test_sealed_artifacts_reject_fresh_byte_replacement(
    tmp_path: Path,
) -> None:
    """Reject a fresh-looking report whose verified bytes were replaced."""
    root = _repository(tmp_path)
    for name, content in (
        ("coverage.json", '{"verified":true}\n'),
        ("coverage.xml", "<verified/>\n"),
    ):
        (root / name).write_text(content, encoding="utf-8")
    sealed = _GATE.seal_artifacts(
        root,
        ("coverage.json", "coverage.xml"),
    )
    (root / "coverage.json").write_text(
        '{"tampered":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        _GATE.ContractGateError,
        match="changed after exact verification",
    ):
        _GATE.verify_artifacts(root, sealed)


def test_sanitized_environment_is_explicit_and_runtime_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allowlist child state and replace inherited writable directories."""
    root = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    root.mkdir()
    for name in (
        "HOME",
        "OLDPWD",
        "PWD",
        "PATH",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONWARNINGS",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTEST_DEBUG",
        "COVERAGE_PROCESS_START",
        "COV_CORE_SOURCE",
        "MYPY_CONFIG_FILE_DIR",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN",
    ):
        monkeypatch.setenv(name, "host-control")

    environment = _GATE.sanitized_environment(
        root,
        runtime,
        trusted_python_root=_ROOT,
    )

    assert environment["PWD"] == str(root.resolve())
    assert environment["HOME"] == str(runtime.resolve() / "home")
    assert environment["TMPDIR"] == str(runtime.resolve() / "tmp")
    allowed_python_paths = environment["PYTHONPATH"].split(":")
    assert allowed_python_paths == [
        str(runtime.resolve() / "python-startup"),
    ]
    assert (
        environment["AVALAN_CONTRACT_ALLOWED_PYTHONPATH"]
        == environment["PYTHONPATH"]
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert "host-control" not in environment["PATH"]
    assert environment["PYTEST_ADDOPTS"] == ""
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["COVERAGE_RCFILE"] == "/dev/null"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTEST_PLUGINS" not in environment
    assert "COVERAGE_PROCESS_START" not in environment
    assert "COV_CORE_SOURCE" not in environment
    assert "MYPY_CONFIG_FILE_DIR" not in environment
    assert "OLDPWD" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert "AZURE_OPENAI_API_KEY" not in environment
    assert "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN" not in environment
    for name in (
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
    ):
        assert Path(environment[name]).is_relative_to(runtime.resolve())
    with pytest.raises(
        _GATE.ContractGateError,
        match="prohibited inherited environment name",
    ):
        _GATE.sanitized_environment(
            root,
            runtime,
            inherited_names=("OPENAI_API_KEY",),
            trusted_python_root=_ROOT,
        )


def test_runtime_startup_tamper_cannot_control_fresh_child(
    tmp_path: Path,
) -> None:
    """Detect one poisoned runtime before a fresh child can reuse it."""
    root = tmp_path / "repository"
    root.mkdir()
    first_startup: Path | None = None

    with pytest.raises(
        _GATE.ContractGateError,
        match="runtime Python startup assets changed",
    ):
        with _GATE.isolated_subprocess_environment(
            root,
            trusted_python_root=_ROOT,
        ) as environment:
            first_startup = Path(environment["PYTHONPATH"].split(pathsep)[0])
            replacement = first_startup / "sitecustomize.py"
            command = (
                executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    f"Path({str(replacement)!r}).write_text("
                    "'import os\\nos._exit(0)\\n', encoding='utf-8')\n"
                ),
            )
            completed = run(command, env=environment, check=False)
            assert completed.returncode == 0

    assert first_startup is not None
    assert not first_startup.exists()
    with _GATE.isolated_subprocess_environment(
        root,
        trusted_python_root=_ROOT,
    ) as environment:
        second_startup = Path(environment["PYTHONPATH"].split(pathsep)[0])
        assert second_startup != first_startup
        completed = run(
            (executable, "-c", "raise SystemExit(97)"),
            env=environment,
            check=False,
        )
        assert completed.returncode == 97


@pytest.mark.parametrize(
    ("location", "replacement"),
    (
        ("runtime", "bytes"),
        ("runtime", "type"),
        ("trusted", "bytes"),
        ("trusted", "type"),
    ),
)
def test_isolated_startup_asset_tampering_fails_closed(
    tmp_path: Path,
    location: str,
    replacement: str,
) -> None:
    """Reject byte and type changes to runtime and trusted startup files."""
    root = tmp_path / "repository"
    trusted_root = tmp_path / "trusted"
    root.mkdir()
    trusted_startup = trusted_root / "scripts" / "contract_startup"
    trusted_startup.mkdir(parents=True)
    for name in ("sitecustomize.py", "avalan_contract_gate_plugin.py"):
        (trusted_startup / name).write_bytes(
            (_ROOT / "scripts" / "contract_startup" / name).read_bytes()
        )

    with pytest.raises(
        _GATE.ContractGateError,
        match=f"{location} Python startup",
    ):
        with _GATE.isolated_subprocess_environment(
            root,
            trusted_python_root=trusted_root,
        ) as environment:
            if location == "runtime":
                startup = Path(environment["PYTHONPATH"].split(pathsep)[0])
            else:
                startup = trusted_startup
            target = startup / "sitecustomize.py"
            if replacement == "bytes":
                target.write_text("VALUE = 1\n", encoding="utf-8")
            else:
                target.unlink()
                target.mkdir()


def test_run_pytest_rejects_runtime_startup_tampering(
    tmp_path: Path,
) -> None:
    """Surface startup mutation by a collected test before returning."""
    root = tmp_path / "repository"
    test = root / "tests" / "tamper_test.py"
    test.parent.mkdir(parents=True)
    test.write_text(
        "from os import environ, pathsep\n"
        "from pathlib import Path\n\n"
        "def test_tamper() -> None:\n"
        "    startup = Path(environ['PYTHONPATH'].split(pathsep)[0])\n"
        "    (startup / 'sitecustomize.py').write_text(\n"
        "        'VALUE = 1\\n', encoding='utf-8'\n"
        "    )\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _GATE.ContractGateError,
        match="runtime Python startup assets changed",
    ):
        _GATE.run_pytest(root, ("-q", "."), timeout=30)


def test_startup_prunes_editable_path_before_ignored_sitecustomize(
    tmp_path: Path,
) -> None:
    """Block ignored startup code reached through an editable-style pth."""
    source = _repository(tmp_path / "source")
    _initialize_git(source)
    (source / ".gitignore").write_text(
        "src/sitecustomize.py\n",
        encoding="utf-8",
    )
    marker = tmp_path / "ignored-sitecustomize-ran"
    malicious = source / "src" / "sitecustomize.py"
    malicious.write_text(
        "from os import environ\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n"
        "environ['PYTEST_ADDOPTS'] = '--help'\n"
        "environ['PYTEST_PLUGINS'] = 'ignored_plugin'\n",
        encoding="utf-8",
    )

    virtual_environment = tmp_path / "editable-environment"
    EnvBuilder(with_pip=False).create(virtual_environment)
    nested_python = virtual_environment / "bin" / "python"
    lookup = run(
        (
            str(nested_python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ),
        capture_output=True,
        check=True,
        text=True,
    )
    site_packages = Path(lookup.stdout.strip())
    (site_packages / "avalan.pth").write_text(
        f"{source / 'src'}\n",
        encoding="utf-8",
    )

    with _GATE.nonignored_execution_mirror(source) as mirror:
        assert not (mirror / "src" / "sitecustomize.py").exists()
        with _GATE.isolated_subprocess_environment(
            mirror,
            trusted_python_root=_ROOT,
        ) as environment:
            script = (
                "from os import environ\n"
                "from pathlib import Path\n"
                "from sys import path\n"
                f"forbidden = {str(source / 'src')!r}\n"
                "resolved = tuple(\n"
                "    str(Path(entry).resolve()) for entry in path if entry\n"
                ")\n"
                "assert forbidden not in resolved\n"
                "assert environ['PYTEST_ADDOPTS'] == ''\n"
                "assert 'PYTEST_PLUGINS' not in environ\n"
            )
            completed = run(
                (str(nested_python), "-c", script),
                cwd=mirror,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_run_pytest_preserves_guarded_subprocess_coverage(
    tmp_path: Path,
) -> None:
    """Keep pytest-cov startup active in a sanitized test subprocess."""
    root = tmp_path / "repository"
    source = root / "src" / "sample.py"
    test = root / "tests" / "sample_test.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    source.write_text(
        "def value() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    child = (
        "from os import environ\n"
        "from pathlib import Path\n"
        "from sys import path\n"
        f"forbidden = {str(_ROOT / 'src')!r}\n"
        "resolved = tuple(\n"
        "    str(Path(entry).resolve()) for entry in path if entry\n"
        ")\n"
        "assert forbidden not in resolved\n"
        "assert environ['PYTHONSAFEPATH'] == '1'\n"
        "assert environ['PYTHONNOUSERSITE'] == '1'\n"
        "assert environ['PYTHONPATH'] == "
        "environ['AVALAN_CONTRACT_ALLOWED_PYTHONPATH']\n"
        "assert 'COV_CORE_SOURCE' in environ\n"
        "from sample import value\n"
        "assert value() == 1\n"
    )
    test.write_text(
        "from subprocess import run\n"
        "from sys import executable\n\n"
        "def test_child() -> None:\n"
        f"    completed = run((executable, '-c', {child!r}), "
        "capture_output=True, check=False, text=True)\n"
        "    assert completed.returncode == 0, completed.stderr\n",
        encoding="utf-8",
    )

    completed = _GATE.run_pytest(
        root,
        (
            "--cov=src/",
            "--cov-config=/dev/null",
            "--cov-report=json:coverage.json",
            "-q",
            ".",
        ),
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = _GATE.strict_json_path(root / "coverage.json")
    assert isinstance(payload, dict)
    files = payload.get("files")
    assert isinstance(files, dict)
    sample = files.get("src/sample.py")
    assert isinstance(sample, dict)
    summary = sample.get("summary")
    assert isinstance(summary, dict)
    assert summary.get("missing_lines") == 0


def test_isolated_child_cannot_inherit_original_or_ignored_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove child cwd and environment omit original and ignored paths."""
    root = _repository(tmp_path)
    _initialize_git(root)
    (root / ".gitignore").write_text("private/\n", encoding="utf-8")
    private = root / "private"
    private.mkdir()
    secret = private / "provider-secret"
    secret.write_text("secret\n", encoding="utf-8")

    with _GATE.nonignored_execution_mirror(root) as mirror:
        with monkeypatch.context() as inherited:
            inherited.setenv("PWD", str(private))
            inherited.setenv("HOME", str(private))
            inherited.setenv("OLDPWD", str(root))
            inherited.setenv("PYTHONPATH", str(private))
            inherited.setenv("PATH", f"/usr/bin:/bin:{private}")
            inherited.setenv("OPENAI_API_KEY", str(secret))
            inherited.setenv("AZURE_OPENAI_API_KEY", str(secret))
            inherited.setenv(
                "AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN",
                f"postgresql://{secret}",
            )
            with _GATE.isolated_subprocess_environment(
                mirror,
                trusted_python_root=_ROOT,
            ) as environment:
                script = (
                    "from os import environ\n"
                    "from pathlib import Path\n"
                    "assert Path.cwd() == Path(environ['PWD'])\n"
                    "assert 'OLDPWD' not in environ\n"
                    "assert 'OPENAI_API_KEY' not in environ\n"
                    "assert 'AZURE_OPENAI_API_KEY' not in environ\n"
                    "assert 'AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN' "
                    "not in environ\n"
                )
                completed = run(
                    (executable, "-c", script),
                    cwd=mirror,
                    env=environment,
                    capture_output=True,
                    check=False,
                    text=True,
                )
                assert completed.returncode == 0, completed.stderr
                assert str(root) not in environment.values()
                assert str(private) not in environment.values()
                assert str(secret) not in environment.values()
                assert environment["PWD"] == str(mirror.resolve())
                assert not (mirror / "private").exists()


@pytest.mark.parametrize(
    "property_body",
    (
        _MISSING_EVIDENCE_BODY,
        _WRONG_EVIDENCE_BODY,
        _DUPLICATE_EVIDENCE_BODY,
    ),
    ids=("missing", "wrong", "duplicate"),
)
def test_active_node_requires_exact_junit_evidence_property(
    tmp_path: Path,
    property_body: str,
) -> None:
    """Reject missing, wrong, and duplicated canonical evidence syntax."""
    root = _repository(tmp_path)
    (root / "tests" / "sample_test.py").write_text(
        f"def test_value(record_property) -> None:\n    {property_body}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        _GATE.ContractGateError,
        match="exactly one canonical direct record_property call",
    ):
        _GATE.execute_pytest_nodes(
            root,
            ("tests/sample_test.py::test_value",),
            junit_path=root / "pytest.xml",
            expected_evidence={"tests/sample_test.py::test_value": "runtime"},
        )


@pytest.mark.parametrize(
    ("body", "match"),
    (
        (
            (
                "record_property(\n"
                "        name='conversation_acceptance_evidence',\n"
                "        value='runtime',\n"
                "    )\n"
                "    assert 1 + 1 == 2"
            ),
            "canonical direct record_property call",
        ),
        (
            (
                "emit = record_property\n"
                "    emit('conversation_acceptance_evidence', 'runtime')\n"
                "    assert 1 + 1 == 2"
            ),
            "canonical direct record_property call",
        ),
        (
            (
                "def emit() -> None:\n"
                "        record_property(\n"
                "            'conversation_acceptance_evidence', 'runtime'\n"
                "        )\n"
                "    emit()\n"
                "    assert 1 + 1 == 2"
            ),
            "canonical direct record_property call",
        ),
        (
            (
                "assert record_property(\n"
                "        'conversation_acceptance_evidence', 'runtime'\n"
                "    ) is None\n"
                "    assert 1 + 1 == 2"
            ),
            "canonical direct record_property call",
        ),
        (
            (
                "record_property = lambda *args: None\n"
                "    record_property(\n"
                "        'conversation_acceptance_evidence', 'runtime'\n"
                "    )\n"
                "    assert 1 + 1 == 2"
            ),
            "keywords, aliases, helpers, or nested calls",
        ),
        (
            (
                "record_property(\n"
                "        'conversation_acceptance_evidence', 'runtime'\n"
                "    )\n"
                "    assert 1 + 1 == 2"
            ),
            "canonical record_property fixture parameter",
        ),
    ),
    ids=(
        "keyword",
        "alias",
        "helper",
        "nested",
        "fixture-rebind",
        "defaulted-fixture",
    ),
)
def test_active_node_rejects_noncanonical_evidence_routes(
    tmp_path: Path,
    body: str,
    match: str,
) -> None:
    """Reject keyword, alias, helper, nested, and rebound evidence calls."""
    root = _repository(tmp_path)
    signature = (
        "record_property=lambda *args: None"
        if match == "canonical record_property fixture parameter"
        else "record_property"
    )
    (root / "tests" / "sample_test.py").write_text(
        f"def test_value({signature}) -> None:\n    {body}\n",
        encoding="utf-8",
    )

    with pytest.raises(_GATE.ContractGateError, match=match):
        _GATE._validate_node_sources(
            root,
            ("tests/sample_test.py::test_value",),
            {"tests/sample_test.py::test_value": "runtime"},
        )


def test_active_node_rejects_unresolved_ast_target(tmp_path: Path) -> None:
    """Fail closed before pytest when a manifest node cannot be resolved."""
    root = _repository(tmp_path)

    with pytest.raises(
        _GATE.ContractGateError,
        match="cannot resolve active acceptance AST target",
    ):
        _GATE._validate_node_sources(
            root,
            ("tests/sample_test.py::test_missing",),
            None,
        )


def test_evidence_only_node_is_placeholder_in_evidence_mode(
    tmp_path: Path,
) -> None:
    """Strip the one canonical evidence call before substance validation."""
    root = _repository(tmp_path)
    node_id = "tests/sample_test.py::test_value"
    (root / "tests" / "sample_test.py").write_text(
        "def test_value(record_property) -> None:\n"
        "    record_property(\n"
        "        'conversation_acceptance_evidence', 'runtime'\n"
        "    )\n",
        encoding="utf-8",
    )

    with pytest.raises(_GATE.ContractGateError, match="placeholder-only"):
        _GATE._validate_node_sources(
            root,
            (node_id,),
            {node_id: "runtime"},
        )


@pytest.mark.parametrize(
    "values",
    ((), ("wrong",), ("runtime", "runtime")),
    ids=("missing", "wrong", "duplicate"),
)
def test_runtime_junit_evidence_remains_exact(values: tuple[str, ...]) -> None:
    """Reject JUnit evidence that does not exactly match its manifest owner."""
    testcase = Element(
        "testcase",
        {
            "file": "tests/sample_test.py",
            "classname": "tests.sample_test",
            "name": "test_value",
        },
    )
    properties = SubElement(testcase, "properties")
    for value in values:
        SubElement(
            properties,
            "property",
            {
                "name": "conversation_acceptance_evidence",
                "value": value,
            },
        )

    with pytest.raises(
        _GATE.ContractGateError,
        match="exactly one matching evidence property",
    ):
        _GATE._verify_junit_evidence(
            (testcase,),
            {"tests/sample_test.py::test_value": "runtime"},
        )


def test_active_node_accepts_matching_junit_evidence_property(
    tmp_path: Path,
) -> None:
    """Accept non-vacuous evidence with its manifest-owned label."""
    root = _repository(tmp_path)
    (root / "tests" / "sample_test.py").write_text(
        "def test_value(record_property) -> None:\n"
        "    record_property(\n"
        "        'conversation_acceptance_evidence', 'runtime'\n"
        "    )\n"
        "    value = object()\n"
        "    assert value.__class__ is object\n",
        encoding="utf-8",
    )

    evidence = _GATE.execute_pytest_nodes(
        root,
        ("tests/sample_test.py::test_value",),
        junit_path=root / "pytest.xml",
        expected_evidence={"tests/sample_test.py::test_value": "runtime"},
    )

    assert evidence.executed == ("tests/sample_test.py::test_value",)
