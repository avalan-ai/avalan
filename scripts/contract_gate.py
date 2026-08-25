#!/usr/bin/env python
"""Provide reusable primitives for lifecycle-aware contract gates."""

from ast import (
    AST,
    AnnAssign,
    Assert,
    Assign,
    AsyncFor,
    AsyncFunctionDef,
    AsyncWith,
    Attribute,
    AugAssign,
    Break,
    Call,
    ClassDef,
    Constant,
    Continue,
    Expr,
    For,
    FunctionDef,
    If,
    Lambda,
    Match,
    Name,
    NamedExpr,
    Raise,
    Return,
    Store,
    Subscript,
    Try,
    TryStar,
    While,
    With,
    iter_child_nodes,
    literal_eval,
    walk,
)
from ast import parse as parse_python
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from os import environ, pathsep
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from shutil import copy2
from subprocess import CompletedProcess, run
from sys import executable
from tempfile import TemporaryDirectory
from typing import cast
from urllib.parse import SplitResult, parse_qsl, urlsplit
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import parse as parse_xml

_DYNAMIC_CODE_PATTERN = compile_regex(r"\b(?:exec|compile)\s*\(")
_NON_PASSING_SUMMARY_PATTERN = compile_regex(
    r"\b(?:skipped|xfailed|xpassed|deselected)\b"
)
_MEASURED_TOP_LEVEL = ("Makefile", "poetry.lock", "pyproject.toml")
_UNMEASURED_DIRECTORY_NAMES = (
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "htmlcov",
)
_UNMEASURED_FILE_NAMES = (
    ".coverage",
    ".patch-contract-pytest-facts.json",
    "coverage.json",
    "coverage.xml",
)
_UNMEASURED_FILE_SUFFIXES = frozenset((".pyc", ".pyo"))
_CHILD_ENVIRONMENT_ALLOWLIST = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)
_CHILD_EXECUTABLE_PATHS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)
_TRUSTED_TEMPORARY_ROOT = Path("/tmp").resolve()
_PYTEST_PLUGIN_ARGUMENTS = (
    "-p",
    "no:cacheprovider",
    "-p",
    "pytest_cov",
    "-p",
    "anyio.pytest_plugin",
)
_PYTEST_FILE_PATTERNS = ("test_*.py", "*_test.py")
_CONTRACT_ALLOWED_PYTHONPATH_ENV = "AVALAN_CONTRACT_ALLOWED_PYTHONPATH"
_CONTRACT_STARTUP_FILES = (
    "sitecustomize.py",
    "avalan_contract_gate_plugin.py",
)
_PYTEST_NORECURSE_PATTERNS = (
    "*.egg",
    ".*",
    "_darcs",
    "build",
    "CVS",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
)
POSTGRESQL_TEST_DSN_ENV = "AVALAN_TASK_TEST_POSTGRESQL_DSN"
_CHILD_INHERITED_ENVIRONMENT_ALLOWLIST = frozenset((POSTGRESQL_TEST_DSN_ENV,))
_POSTGRESQL_SCHEMES = frozenset(("postgresql", "postgresql+psycopg"))
_POSTGRESQL_SSLMODE_VALUES = (
    "allow",
    "disable",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
)


class ContractGateError(RuntimeError):
    """Report invalid or non-passing reusable gate evidence."""


class StrictJsonError(ValueError):
    """Report JSON that is ambiguous or outside the accepted grammar."""


class DuplicateJsonNameError(StrictJsonError):
    """Report a duplicate name in one JSON object."""


class NonFiniteJsonNumberError(StrictJsonError):
    """Report a non-finite JSON number."""


@dataclass(frozen=True, kw_only=True, slots=True)
class MeasuredInput:
    """Store one measured gate input and its immutable evidence."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True, kw_only=True, slots=True)
class SealedInputInventory:
    """Store a canonical inventory of every measured gate input."""

    entries: tuple[MeasuredInput, ...]
    sha256: str
    newest_mtime_ns: int


@dataclass(frozen=True, kw_only=True, slots=True)
class SealedArtifact:
    """Store the immutable bytes of one generated gate artifact."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True, kw_only=True, slots=True)
class PytestEvidence:
    """Store exact collection and successful JUnit execution evidence."""

    collected: tuple[str, ...]
    executed: tuple[str, ...]
    testcases: tuple[Element, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class _ReachableInvariantResult:
    """Store positive-invariant and bounded fallthrough analysis."""

    contains_invariant: bool
    falls_through: bool


def strict_json_loads(source: str) -> object:
    """Return a strictly decoded JSON value."""
    assert isinstance(source, str)

    def object_from_pairs(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for name, item in pairs:
            if name in value:
                raise DuplicateJsonNameError(
                    f"duplicate JSON object name: {name!r}"
                )
            value[name] = item
        return value

    def reject_constant(constant: str) -> object:
        raise NonFiniteJsonNumberError(
            f"non-finite JSON number is prohibited: {constant}"
        )

    return loads(
        source,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )


def strict_json_path(path: Path) -> object:
    """Return a strictly decoded JSON file."""
    assert isinstance(path, Path)
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError, UnicodeError) as exc:
        raise StrictJsonError(f"cannot decode {path}: {exc}") from exc


def canonical_sha256(value: object) -> str:
    """Return a stable SHA-256 digest for one JSON-compatible value."""
    return sha256(
        dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def validate_postgresql_url(dsn: str) -> None:
    """Validate a URL-style PostgreSQL DSN without identity overrides."""
    _validated_postgresql_url(dsn)


def _validated_postgresql_url(dsn: str) -> SplitResult:
    """Return strict PostgreSQL URL components or fail closed."""
    assert isinstance(dsn, str) and dsn
    try:
        parts = urlsplit(dsn)
        _ = parts.port
        query = parse_qsl(
            parts.query,
            keep_blank_values=True,
            strict_parsing=True,
        )
    except (UnicodeError, ValueError) as exc:
        raise ContractGateError("PostgreSQL DSN is malformed") from exc
    normalized_query = tuple(
        (name.casefold(), value.casefold()) for name, value in query
    )
    userinfo = parts.netloc.rsplit("@", 1)[0] if "@" in parts.netloc else None
    malformed_credentials = userinfo is not None and (
        not parts.username
        or (":" in userinfo and not parts.password)
        or any(character.isspace() for character in userinfo)
    )
    if (
        parts.scheme.casefold() not in _POSTGRESQL_SCHEMES
        or not parts.netloc
        or not parts.hostname
        or any(character.isspace() for character in parts.hostname)
        or malformed_credentials
        or bool(parts.fragment)
        or not _valid_postgresql_query(parts.query, query, normalized_query)
    ):
        raise ContractGateError("PostgreSQL DSN is malformed or ambiguous")
    return parts


def verify_pytest_module_name_uniqueness(root: Path) -> None:
    """Reject colliding pytest module names and package prefixes."""
    resolved_root = root.resolve()
    test_root = (resolved_root / "tests").resolve()
    if not test_root.is_dir():
        raise ContractGateError("pytest test root does not exist")
    symbols: dict[str, set[str]] = {}
    for path in _pytest_test_module_paths(resolved_root):
        _reject_pytest_plugin_declaration(path, resolved_root)
        for name, kind, binding in _pytest_import_bindings(path):
            relative = binding.relative_to(resolved_root).as_posix()
            symbols.setdefault(name, set()).add(f"{kind}:{relative}")
    conflicts = {
        name: sorted(bindings)
        for name, bindings in symbols.items()
        if len(bindings) > 1
    }
    if conflicts:
        raise ContractGateError(
            "pytest import module names are duplicated or shadow package "
            f"prefixes: {dumps(conflicts, sort_keys=True)}"
        )


def _pytest_test_module_paths(root: Path) -> tuple[Path, ...]:
    """Return Python test modules discovered by bare repository pytest."""
    candidates: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            paths = sorted(directory.iterdir(), reverse=True)
        except OSError as exc:
            relative = directory.relative_to(root).as_posix() or "."
            raise ContractGateError(
                f"pytest collection directory is unreadable: {relative}"
            ) from exc
        for path in paths:
            if path.is_symlink():
                if _pytest_ignores_directory_name(path.name):
                    continue
                if path.is_dir():
                    raise ContractGateError(
                        "pytest collection directories must not be symbolic "
                        f"links: {path.relative_to(root)}"
                    )
                if _pytest_matches_file(path) and path.is_file():
                    raise ContractGateError(
                        "pytest test modules must be regular files: "
                        f"{path.relative_to(root)}"
                    )
                continue
            if path.is_dir():
                if _pytest_ignores_directory(path):
                    continue
                pending.append(path)
            elif path.is_file() and _pytest_matches_file(path):
                candidates.append(path)
    return tuple(sorted(candidates))


def _reject_pytest_plugin_declaration(path: Path, root: Path) -> None:
    """Reject direct pytest plugin assignments in collected test source."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ContractGateError(
            "pytest test module is unreadable: "
            f"{path.relative_to(root).as_posix()}"
        ) from exc
    try:
        tree = parse_python(payload, filename=str(path))
    except (SyntaxError, ValueError) as exc:
        raise ContractGateError(
            "pytest test module cannot be parsed: "
            f"{path.relative_to(root).as_posix()}"
        ) from exc
    if any(
        isinstance(node, Name)
        and node.id == "pytest_plugins"
        and isinstance(node.ctx, Store)
        for node in walk(tree)
    ):
        raise ContractGateError(
            "pytest test modules must not declare pytest_plugins: "
            f"{path.relative_to(root).as_posix()}"
        )


def _pytest_ignores_directory(path: Path) -> bool:
    """Return whether default bare pytest skips one directory."""
    if _pytest_ignores_directory_name(path.name):
        return True
    try:
        return (path / "pyvenv.cfg").is_file() or (
            path / "conda-meta" / "history"
        ).is_file()
    except OSError as exc:
        raise ContractGateError(
            f"pytest virtual environment boundary is unreadable: {path}"
        ) from exc


def _pytest_ignores_directory_name(name: str) -> bool:
    """Return whether pytest's default recursion patterns skip a name."""
    return name == "__pycache__" or any(
        fnmatch(name, pattern) for pattern in _PYTEST_NORECURSE_PATTERNS
    )


def _pytest_matches_file(path: Path) -> bool:
    """Return whether pytest's default Python file patterns match a path."""
    return any(
        fnmatch(path.name, pattern) for pattern in _PYTEST_FILE_PATTERNS
    )


def _pytest_import_module_name(path: Path) -> str:
    """Return pytest's prepend-mode import name for one test module."""
    return _pytest_import_bindings(path)[-1][0]


def _pytest_import_bindings(
    path: Path,
) -> tuple[tuple[str, str, Path], ...]:
    """Return every package and module symbol bound by one test path."""
    package_directories: list[Path] = []
    parent = path.parent
    while parent.name.isidentifier() and (parent / "__init__.py").is_file():
        package_directories.append(parent)
        parent = parent.parent
    parts: list[str] = []
    bindings: list[tuple[str, str, Path]] = []
    for directory in reversed(package_directories):
        parts.append(directory.name)
        bindings.append((".".join(parts), "package", directory))
    parts.append(path.stem)
    bindings.append((".".join(parts), "module", path))
    return tuple(bindings)


def _valid_postgresql_query(
    raw_query: str,
    query: list[tuple[str, str]],
    normalized_query: tuple[tuple[str, str], ...],
) -> bool:
    """Return whether a PostgreSQL query is the tiny accepted grammar."""
    if not raw_query:
        return not query
    if len(query) != 1 or len(normalized_query) != 1:
        return False
    name, value = query[0]
    normalized_name, normalized_value = normalized_query[0]
    return (
        normalized_name == "sslmode"
        and normalized_value in _POSTGRESQL_SSLMODE_VALUES
        and name == normalized_name
        and value == normalized_value
        and raw_query == f"sslmode={value}"
    )


def capture_input_inventory(root: Path) -> SealedInputInventory:
    """Capture every runtime-affecting file in an isolated repository."""
    resolved_root = root.resolve()
    candidates: set[Path] = set()
    for path in resolved_root.rglob("*"):
        relative_parts = path.relative_to(resolved_root).parts
        if any(
            part in _UNMEASURED_DIRECTORY_NAMES or part.endswith(".egg-info")
            for part in relative_parts[:-1]
        ):
            continue
        if path.is_symlink():
            raise ContractGateError(
                "measured gate inputs cannot be symbolic links: "
                f"{path.relative_to(resolved_root)}"
            )
        if (
            path.is_file()
            and path.name not in _UNMEASURED_FILE_NAMES
            and not path.name.startswith(".coverage.")
            and path.suffix not in _UNMEASURED_FILE_SUFFIXES
        ):
            candidates.add(path)
    entries: list[MeasuredInput] = []
    newest_mtime_ns = 0
    for path in sorted(candidates):
        try:
            relative = path.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ContractGateError(
                f"measured input escapes repository root: {path}"
            ) from exc
        payload = path.read_bytes()
        stat = path.stat()
        newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
        entries.append(
            MeasuredInput(
                path=relative,
                size=len(payload),
                sha256=sha256(payload).hexdigest(),
            )
        )
    required = set(_MEASURED_TOP_LEVEL)
    observed = {entry.path for entry in entries}
    missing = sorted(required - observed)
    if missing:
        raise ContractGateError(
            f"required measured gate inputs are missing: {missing}"
        )
    if not any(entry.path.startswith("src/") for entry in entries):
        raise ContractGateError("measured source inventory is empty")
    canonical = [
        {"path": entry.path, "sha256": entry.sha256, "size": entry.size}
        for entry in entries
    ]
    return SealedInputInventory(
        entries=tuple(entries),
        sha256=canonical_sha256(canonical),
        newest_mtime_ns=newest_mtime_ns,
    )


@contextmanager
def nonignored_execution_mirror(root: Path) -> Iterator[Path]:
    """Yield an external mirror containing only nonignored current bytes."""
    resolved_root = root.resolve()
    relative_paths = _nonignored_file_paths(resolved_root)
    before = _measure_explicit_paths(resolved_root, relative_paths)
    with TemporaryDirectory(
        prefix="avalan-contract-gate-",
        dir=_TRUSTED_TEMPORARY_ROOT,
    ) as temporary:
        mirror = (Path(temporary) / "repository").resolve()
        _validate_execution_mirror(resolved_root, mirror)
        mirror.mkdir()
        for entry in before:
            source = resolved_root / entry.path
            destination = mirror / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)
        after_paths = _nonignored_file_paths(resolved_root)
        after = _measure_explicit_paths(resolved_root, after_paths)
        if before != after:
            raise ContractGateError(
                "nonignored repository inputs changed while creating the "
                "execution mirror"
            )
        mirrored = _measure_explicit_paths(
            mirror,
            tuple(entry.path for entry in before),
        )
        if before != mirrored:
            raise ContractGateError(
                "execution mirror bytes do not match the source snapshot"
            )
        try:
            yield mirror
        except BaseException as exc:
            try:
                _verify_source_snapshot(resolved_root, before)
            except ContractGateError as integrity_error:
                exc.add_note(str(integrity_error))
            raise
        else:
            _verify_source_snapshot(resolved_root, before)


def _validate_execution_mirror(source_root: Path, mirror: Path) -> None:
    """Reject an execution mirror inside its source checkout."""
    resolved_source = source_root.resolve()
    resolved_mirror = mirror.resolve()
    if resolved_mirror == resolved_source or resolved_mirror.is_relative_to(
        resolved_source
    ):
        raise ContractGateError(
            "execution mirror must be outside the source repository"
        )


def _verify_source_snapshot(
    root: Path,
    expected: tuple[MeasuredInput, ...],
) -> None:
    relative_paths = _nonignored_file_paths(root)
    observed = _measure_explicit_paths(root, relative_paths)
    if observed != expected:
        raise ContractGateError(
            "nonignored repository inputs changed during mirrored execution"
        )


def _nonignored_file_paths(root: Path) -> tuple[str, ...]:
    completed = run(
        (
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")
        raise ContractGateError(
            f"cannot enumerate nonignored repository inputs: {message}"
        )
    raw_paths = completed.stdout.split(b"\0")
    observed: list[str] = []
    for raw_path in raw_paths:
        if not raw_path:
            continue
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractGateError(
                "nonignored repository path is not valid UTF-8"
            ) from exc
        path = PurePosixPath(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ContractGateError(
                f"nonignored repository path is invalid: {relative}"
            )
        source = root / relative
        if not source.exists() and not source.is_symlink():
            continue
        if source.is_symlink() or not source.is_file():
            raise ContractGateError(
                "nonignored repository inputs must be regular files: "
                f"{relative}"
            )
        observed.append(relative)
    if len(observed) != len(set(observed)):
        raise ContractGateError("nonignored repository paths are duplicated")
    return tuple(sorted(observed))


def _measure_explicit_paths(
    root: Path,
    relative_paths: tuple[str, ...],
) -> tuple[MeasuredInput, ...]:
    entries: list[MeasuredInput] = []
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ContractGateError(
                "nonignored repository input disappeared or changed type: "
                f"{relative}"
            )
        payload = path.read_bytes()
        entries.append(
            MeasuredInput(
                path=relative,
                size=len(payload),
                sha256=sha256(payload).hexdigest(),
            )
        )
    return tuple(entries)


def verify_input_inventory(
    before: SealedInputInventory,
    after: SealedInputInventory,
) -> None:
    """Reject added, removed, or mutated measured gate inputs."""
    if before == after:
        return
    before_by_path = {entry.path: entry for entry in before.entries}
    after_by_path = {entry.path: entry for entry in after.entries}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    mutated = sorted(
        path
        for path in set(before_by_path) & set(after_by_path)
        if before_by_path[path] != after_by_path[path]
    )
    raise ContractGateError(
        "measured gate inputs changed during execution: "
        f"added={added}, removed={removed}, mutated={mutated}"
    )


def verify_report_after_inventory(
    report: Path,
    inventory: SealedInputInventory,
) -> None:
    """Reject a missing report or one older than its measured inputs."""
    if not report.is_file():
        raise ContractGateError(f"coverage report does not exist: {report}")
    if report.stat().st_mtime_ns < inventory.newest_mtime_ns:
        raise ContractGateError(
            "coverage report predates the sealed gate input inventory"
        )


def seal_artifacts(
    root: Path,
    relative_paths: tuple[str, ...],
) -> tuple[SealedArtifact, ...]:
    """Seal generated artifact bytes for later tamper verification."""
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ContractGateError(
            "artifact seal inventory is empty or duplicated"
        )
    artifacts: list[SealedArtifact] = []
    for relative in relative_paths:
        path = PurePosixPath(relative)
        if path.is_absolute() or len(path.parts) != 1:
            raise ContractGateError(f"artifact path is invalid: {relative}")
        payload_path = root / relative
        if payload_path.is_symlink() or not payload_path.is_file():
            raise ContractGateError(
                f"generated gate artifact does not exist: {relative}"
            )
        payload = payload_path.read_bytes()
        artifacts.append(
            SealedArtifact(
                path=relative,
                size=len(payload),
                sha256=sha256(payload).hexdigest(),
            )
        )
    return tuple(artifacts)


def verify_artifacts(root: Path, sealed: tuple[SealedArtifact, ...]) -> None:
    """Reject removed, replaced, or byte-mutated generated artifacts."""
    if not sealed:
        raise ContractGateError("artifact seal inventory is empty")
    observed = seal_artifacts(root, tuple(item.path for item in sealed))
    if observed != sealed:
        raise ContractGateError(
            "generated gate artifacts changed after exact verification"
        )


def _seal_python_startup_assets(
    directory: Path,
    *,
    label: str,
) -> tuple[SealedArtifact, ...]:
    """Seal required regular Python startup assets."""
    if directory.is_symlink() or not directory.is_dir():
        raise ContractGateError(f"{label} directory is invalid: {directory}")
    assets: list[SealedArtifact] = []
    for name in _CONTRACT_STARTUP_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ContractGateError(
                f"{label} asset is missing or not a regular file: {path}"
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ContractGateError(
                f"{label} asset is unreadable: {path}"
            ) from exc
        assets.append(
            SealedArtifact(
                path=name,
                size=len(payload),
                sha256=sha256(payload).hexdigest(),
            )
        )
    return tuple(assets)


def _verify_python_startup_assets(
    directory: Path,
    expected: tuple[SealedArtifact, ...],
    *,
    label: str,
) -> None:
    """Reject byte or type changes to sealed Python startup assets."""
    observed = _seal_python_startup_assets(directory, label=label)
    if observed != expected:
        raise ContractGateError(
            f"{label} assets changed during isolated subprocess execution"
        )


def sanitized_environment(
    root: Path,
    runtime_root: Path,
    *,
    inherited_names: tuple[str, ...] = (),
    trusted_python_root: Path | None = None,
) -> dict[str, str]:
    """Return an allowlisted child environment rooted outside the checkout."""
    resolved_root = root.resolve()
    resolved_runtime = runtime_root.resolve()
    if resolved_runtime == resolved_root or resolved_runtime.is_relative_to(
        resolved_root
    ):
        raise ContractGateError(
            "isolated child runtime must be outside the repository"
        )
    if (
        len(inherited_names) != len(set(inherited_names))
        or not set(inherited_names) <= _CHILD_INHERITED_ENVIRONMENT_ALLOWLIST
    ):
        raise ContractGateError(
            "isolated child requested a prohibited inherited environment name"
        )
    home = resolved_runtime / "home"
    temporary = resolved_runtime / "tmp"
    startup = resolved_runtime / "python-startup"
    xdg_root = resolved_runtime / "xdg"
    xdg_paths = {
        "XDG_CACHE_HOME": xdg_root / "cache",
        "XDG_CONFIG_HOME": xdg_root / "config",
        "XDG_DATA_HOME": xdg_root / "data",
        "XDG_RUNTIME_DIR": xdg_root / "runtime",
        "XDG_STATE_HOME": xdg_root / "state",
    }
    for path in (home, temporary, startup, *xdg_paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    trusted_root = (trusted_python_root or resolved_root).resolve()
    startup_source = trusted_root / "scripts" / "contract_startup"
    for name in _CONTRACT_STARTUP_FILES:
        source = startup_source / name
        if source.is_symlink() or not source.is_file():
            raise ContractGateError(
                f"trusted Python startup file is missing: {source}"
            )
        copy2(source, startup / name)
    allowed_python_paths = (
        startup,
        *(
            path
            for path in (
                resolved_root / "src",
                resolved_root / "scripts",
            )
            if path.is_dir()
        ),
    )
    if len(allowed_python_paths) != len(set(allowed_python_paths)) or any(
        path.is_symlink() or not path.is_dir() for path in allowed_python_paths
    ):
        raise ContractGateError("trusted Python paths are invalid")
    python_path = pathsep.join(str(path) for path in allowed_python_paths)
    environment = {
        key: environ[key]
        for key in _CHILD_ENVIRONMENT_ALLOWLIST
        if key in environ
    }
    for name in inherited_names:
        if name in environ:
            value = environ[name]
            if name == POSTGRESQL_TEST_DSN_ENV:
                if not value:
                    raise ContractGateError("PostgreSQL DSN is empty")
                validate_postgresql_url(value)
            environment[name] = value
    environment["PATH"] = pathsep.join(_CHILD_EXECUTABLE_PATHS)
    environment["PWD"] = str(resolved_root)
    environment["HOME"] = str(home)
    environment["TMPDIR"] = str(temporary)
    environment["TMP"] = str(temporary)
    environment["TEMP"] = str(temporary)
    environment.update({name: str(path) for name, path in xdg_paths.items()})
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["COVERAGE_RCFILE"] = "/dev/null"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONPATH"] = python_path
    environment[_CONTRACT_ALLOWED_PYTHONPATH_ENV] = python_path
    return environment


@contextmanager
def isolated_subprocess_environment(
    root: Path,
    *,
    inherited_names: tuple[str, ...] = (),
    trusted_python_root: Path | None = None,
) -> Iterator[dict[str, str]]:
    """Yield an allowlisted child environment with isolated writable roots."""
    resolved_trusted_root = (trusted_python_root or root.resolve()).resolve()
    trusted_startup = resolved_trusted_root / "scripts" / "contract_startup"
    trusted_snapshot = _seal_python_startup_assets(
        trusted_startup,
        label="trusted Python startup",
    )
    with TemporaryDirectory(
        prefix="avalan-contract-runtime-",
        dir="/tmp",
    ) as temporary:
        runtime_root = Path(temporary)
        environment = sanitized_environment(
            root,
            runtime_root,
            inherited_names=inherited_names,
            trusted_python_root=resolved_trusted_root,
        )
        runtime_startup = runtime_root / "python-startup"
        runtime_snapshot = _seal_python_startup_assets(
            runtime_startup,
            label="runtime Python startup",
        )
        if runtime_snapshot != trusted_snapshot:
            raise ContractGateError(
                "runtime Python startup assets differ from their trusted "
                "source"
            )
        _verify_python_startup_assets(
            trusted_startup,
            trusted_snapshot,
            label="trusted Python startup",
        )
        try:
            yield environment
        except BaseException as exc:
            try:
                _verify_python_startup_assets(
                    trusted_startup,
                    trusted_snapshot,
                    label="trusted Python startup",
                )
                _verify_python_startup_assets(
                    runtime_startup,
                    runtime_snapshot,
                    label="runtime Python startup",
                )
            except ContractGateError as integrity_error:
                exc.add_note(str(integrity_error))
            raise
        else:
            _verify_python_startup_assets(
                trusted_startup,
                trusted_snapshot,
                label="trusted Python startup",
            )
            _verify_python_startup_assets(
                runtime_startup,
                runtime_snapshot,
                label="runtime Python startup",
            )


def exact_coverage_commands() -> tuple[tuple[str, ...], ...]:
    """Return the shared full-suite exact source coverage commands."""
    return (
        (
            executable,
            "-m",
            "pytest",
            *_PYTEST_PLUGIN_ARGUMENTS,
            "--verbose",
            "-s",
            *hardened_pytest_arguments(),
            "--cov=src/",
            "--cov-config=/dev/null",
            "--cov-report=xml",
            "--cov-report=json:coverage.json",
            ".",
        ),
        (executable, "scripts/verify_src_coverage.py"),
        (
            "jq",
            "-r",
            (
                ".files | to_entries[] | select("
                ".value.summary.missing_lines != 0 or "
                ".value.summary.covered_lines != "
                ".value.summary.num_statements) | "
                '"\\(.key): " + '
                '"\\(.value.summary.percent_covered_display)%"'
            ),
            "coverage.json",
        ),
    )


def hardened_pytest_arguments() -> tuple[str, ...]:
    """Return canonical config and discovery arguments for pytest."""
    return (
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
        f"python_files={' '.join(_PYTEST_FILE_PATTERNS)}",
        "-o",
        f"norecursedirs={' '.join(_PYTEST_NORECURSE_PATTERNS)}",
    )


def remove_coverage_artifacts(
    root: Path,
    *,
    include_reports: bool,
) -> None:
    """Remove coverage runtime data and optionally generated reports."""
    artifacts = list(root.glob(".coverage.*"))
    artifacts.append(root / ".coverage")
    if include_reports:
        artifacts.extend((root / "coverage.json", root / "coverage.xml"))
    for artifact in artifacts:
        if artifact.is_file() or artifact.is_symlink():
            artifact.unlink()


def run_pytest(
    root: Path,
    arguments: tuple[str, ...],
    *,
    timeout: int,
    inherited_names: tuple[str, ...] = (),
) -> CompletedProcess[str]:
    """Run pytest with ambient plugin and option injection disabled."""
    trusted_root = Path(__file__).resolve().parents[1]
    with isolated_subprocess_environment(
        root,
        inherited_names=inherited_names,
        trusted_python_root=trusted_root,
    ) as environment:
        return run(
            (
                executable,
                "-m",
                "pytest",
                *_PYTEST_PLUGIN_ARGUMENTS,
                *hardened_pytest_arguments(),
                *arguments,
            ),
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def junit_testcase_id(testcase: Element) -> str:
    """Return one exact pytest instance ID from legacy JUnit evidence."""
    relative = _nonempty_string(
        testcase.attrib.get("file"), "JUnit testcase file"
    )
    path = PurePosixPath(relative)
    if path.is_absolute() or path.suffix != ".py" or ".." in path.parts:
        raise ContractGateError(f"JUnit testcase file is invalid: {relative}")
    name = _nonempty_string(testcase.attrib.get("name"), "JUnit testcase name")
    classname = _nonempty_string(
        testcase.attrib.get("classname"), "JUnit testcase classname"
    )
    module_name = ".".join(path.with_suffix("").parts)
    if classname == module_name:
        class_parts: tuple[str, ...] = ()
    elif classname.startswith(f"{module_name}."):
        class_parts = tuple(classname[len(module_name) + 1 :].split("."))
    else:
        raise ContractGateError(
            "JUnit testcase classname does not match its file"
        )
    if any(not part for part in class_parts):
        raise ContractGateError(
            "JUnit testcase classname has an empty component"
        )
    return "::".join((relative, *class_parts, name))


def execute_pytest_nodes(
    root: Path,
    node_ids: tuple[str, ...],
    *,
    junit_path: Path,
    collection_timeout: int = 180,
    execution_timeout: int = 900,
    expected_evidence: Mapping[str, str] | None = None,
    inherited_names: tuple[str, ...] = (),
) -> PytestEvidence:
    """Collect and execute exact nodes without non-passing outcomes."""
    if not node_ids or len(node_ids) != len(set(node_ids)):
        raise ContractGateError(
            "acceptance node inventory is empty or duplicated"
        )
    _validate_node_sources(root, node_ids, expected_evidence)
    collection = run_pytest(
        root,
        ("--collect-only", "-q", *node_ids),
        timeout=collection_timeout,
        inherited_names=inherited_names,
    )
    if collection.returncode != 0:
        raise ContractGateError(
            "pytest collection failed:"
            f"\nstdout:\n{collection.stdout[-4000:]}"
            f"\nstderr:\n{collection.stderr[-4000:]}"
        )
    collected = tuple(
        line.strip()
        for line in collection.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )
    if not collected or len(collected) != len(set(collected)):
        raise ContractGateError("pytest collection is empty or duplicated")
    for node_id in node_ids:
        if not any(
            collected_id == node_id or collected_id.startswith(f"{node_id}[")
            for collected_id in collected
        ):
            raise ContractGateError(
                f"pytest did not collect active node: {node_id}"
            )
    execution = run_pytest(
        root,
        (
            "-q",
            "-s",
            "-r",
            "xXs",
            "-o",
            "junit_family=legacy",
            f"--junitxml={junit_path}",
            *node_ids,
        ),
        timeout=execution_timeout,
        inherited_names=inherited_names,
    )
    if execution.returncode != 0:
        raise ContractGateError(
            "pytest acceptance execution failed:"
            f"\nstdout:\n{execution.stdout[-8000:]}"
            f"\nstderr:\n{execution.stderr[-4000:]}"
        )
    if _NON_PASSING_SUMMARY_PATTERN.search(execution.stdout):
        raise ContractGateError(
            "acceptance execution skipped, xfailed, xpassed, or "
            "deselected tests"
        )
    if not junit_path.is_file():
        raise ContractGateError("pytest did not write execution evidence")
    root_element = parse_xml(junit_path).getroot()
    suites = (
        tuple(root_element)
        if root_element.tag == "testsuites"
        else (root_element,)
    )
    totals = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    testcases = tuple(
        testcase for suite in suites for testcase in suite.iter("testcase")
    )
    executed = tuple(map(junit_testcase_id, testcases))
    if (
        totals["tests"] < len(collected)
        or len(executed) != len(set(executed))
        or set(executed) != set(collected)
        or any(totals[key] for key in ("failures", "errors", "skipped"))
    ):
        raise ContractGateError(
            "pytest execution evidence does not match collected instance "
            f"IDs: {totals}"
        )
    if expected_evidence is not None:
        _verify_junit_evidence(testcases, expected_evidence)
    return PytestEvidence(
        collected=collected,
        executed=executed,
        testcases=testcases,
    )


def _verify_junit_evidence(
    testcases: tuple[Element, ...],
    expected_evidence: Mapping[str, str],
) -> None:
    if not expected_evidence:
        raise ContractGateError("acceptance evidence inventory is empty")
    for testcase in testcases:
        instance_id = junit_testcase_id(testcase)
        owners = tuple(
            node_id
            for node_id in expected_evidence
            if instance_id == node_id or instance_id.startswith(f"{node_id}[")
        )
        if len(owners) != 1:
            raise ContractGateError(
                "executed acceptance instance does not have exactly one "
                f"manifest owner: {instance_id}"
            )
        properties = tuple(testcase.iter("property"))
        values = tuple(
            property_element.attrib.get("value")
            for property_element in properties
            if property_element.attrib.get("name")
            == "conversation_acceptance_evidence"
        )
        expected = expected_evidence[owners[0]]
        if values != (expected,):
            raise ContractGateError(
                "acceptance test did not emit exactly one matching evidence "
                f"property: {instance_id} expected={expected!r} "
                f"observed={values!r}"
            )


def _validate_node_sources(
    root: Path,
    node_ids: tuple[str, ...],
    expected_evidence: Mapping[str, str] | None,
) -> None:
    if expected_evidence is not None and set(expected_evidence) != set(
        node_ids
    ):
        raise ContractGateError(
            "static acceptance evidence owners differ from active nodes"
        )
    test_root = (root / "tests").resolve()
    relative_files = tuple(
        dict.fromkeys(node_id.split("::", 1)[0] for node_id in node_ids)
    )
    for relative in relative_files:
        path = (root / relative).resolve()
        if not path.is_relative_to(test_root) or not path.is_file():
            raise ContractGateError(
                f"active acceptance test does not exist: {relative}"
            )
        match = _DYNAMIC_CODE_PATTERN.search(path.read_text(encoding="utf-8"))
        if match is not None:
            raise ContractGateError(
                "active tests contain a prohibited coverage trick using "
                f"dynamic code: {relative}:{match.group(0)}"
            )
    for node_id in node_ids:
        expected = (
            expected_evidence[node_id]
            if expected_evidence is not None
            else None
        )
        _reject_placeholder_node(root, node_id, expected)


def _reject_placeholder_node(
    root: Path,
    node_id: str,
    expected_evidence: str | None,
) -> None:
    relative, *raw_parts = node_id.split("::")
    if not raw_parts:
        raise ContractGateError(
            f"active acceptance node must select a test: {node_id}"
        )
    parts = (*raw_parts[:-1], raw_parts[-1].split("[", 1)[0])
    path = root / relative
    try:
        tree = parse_python(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ContractGateError(
            f"cannot inspect active acceptance node: {node_id}"
        ) from exc
    children = tuple(tree.body)
    target: FunctionDef | AsyncFunctionDef | ClassDef | None = None
    for part in parts:
        target = next(
            (
                child
                for child in children
                if isinstance(
                    child,
                    (AsyncFunctionDef, ClassDef, FunctionDef),
                )
                and child.name == part
            ),
            None,
        )
        if target is None:
            raise ContractGateError(
                f"cannot resolve active acceptance AST target: {node_id}"
            )
        children = tuple(target.body)
    if not isinstance(target, (AsyncFunctionDef, FunctionDef)):
        raise ContractGateError(
            f"active acceptance node must select a test function: {node_id}"
        )
    body_without_docstring = tuple(
        statement
        for index, statement in enumerate(target.body)
        if not (
            index == 0
            and isinstance(statement, Expr)
            and isinstance(statement.value, Constant)
            and isinstance(statement.value.value, str)
        )
    )
    evidence_statements = tuple(
        statement
        for statement in body_without_docstring
        if _is_acceptance_evidence_statement(statement, expected_evidence)
    )
    if expected_evidence is not None:
        parameters = (*target.args.args, *target.args.kwonlyargs)
        fixture_count = sum(
            argument.arg == "record_property" for argument in parameters
        )
        default_count = len(target.args.defaults)
        positional_defaults = (
            {argument.arg for argument in target.args.args[-default_count:]}
            if default_count
            else set()
        )
        keyword_defaults = {
            argument.arg
            for argument, default in zip(
                target.args.kwonlyargs,
                target.args.kw_defaults,
                strict=True,
            )
            if default is not None
        }
        if (
            fixture_count != 1
            or "record_property" in positional_defaults
            or "record_property" in keyword_defaults
        ):
            raise ContractGateError(
                "acceptance evidence requires the canonical record_property "
                f"fixture parameter: {node_id}"
            )
        if len(evidence_statements) != 1:
            raise ContractGateError(
                "acceptance evidence requires exactly one canonical direct "
                f"record_property call: {node_id}"
            )
    for statement in body_without_docstring:
        if statement in evidence_statements:
            continue
        if _mentions_acceptance_evidence(statement):
            raise ContractGateError(
                "acceptance evidence must not use keywords, aliases, helpers, "
                f"or nested calls: {node_id}"
            )
    body = tuple(
        statement
        for statement in body_without_docstring
        if statement not in evidence_statements
    )
    if not _has_executable_acceptance_invariant(body):
        raise ContractGateError(
            "active acceptance node is placeholder-only or lacks a positive "
            f"executable invariant: {node_id}"
        )


def _is_acceptance_evidence_statement(
    statement: object,
    expected_evidence: str | None,
) -> bool:
    """Return whether a statement only records mandatory gate evidence."""
    if not isinstance(statement, Expr) or not isinstance(
        statement.value, Call
    ):
        return False
    call = statement.value
    return (
        isinstance(call.func, Name)
        and call.func.id == "record_property"
        and len(call.args) == 2
        and not call.keywords
        and isinstance(call.args[0], Constant)
        and call.args[0].value == "conversation_acceptance_evidence"
        and isinstance(call.args[1], Constant)
        and isinstance(call.args[1].value, str)
        and (
            expected_evidence is None
            or call.args[1].value == expected_evidence
        )
    )


def _mentions_acceptance_evidence(statement: object) -> bool:
    """Return whether a noncanonical statement attempts evidence emission."""
    if not isinstance(statement, AST):
        return False
    return any(
        (
            isinstance(node, Constant)
            and node.value == "conversation_acceptance_evidence"
        )
        or (isinstance(node, Name) and node.id == "record_property")
        for node in walk(statement)
    )


def _has_executable_acceptance_invariant(statements: tuple[AST, ...]) -> bool:
    """Return whether statements contain a bounded positive invariant."""
    static_names = _static_local_binding_names(statements)
    result = _analyze_reachable_block(statements, static_names)
    return result.contains_invariant


def _analyze_reachable_block(
    statements: Sequence[AST],
    static_names: frozenset[str],
) -> _ReachableInvariantResult:
    """Analyze positive invariants in one ordered, reachable block."""
    contains_invariant = False
    falls_through = True
    for statement in statements:
        if not falls_through:
            break
        result = _analyze_reachable_statement(statement, static_names)
        contains_invariant = contains_invariant or result.contains_invariant
        falls_through = result.falls_through
    return _ReachableInvariantResult(
        contains_invariant=contains_invariant,
        falls_through=falls_through,
    )


def _analyze_reachable_statement(
    node: AST,
    static_names: frozenset[str],
) -> _ReachableInvariantResult:
    """Analyze one statement with bounded control-flow reachability."""
    if isinstance(node, (AsyncFunctionDef, ClassDef, FunctionDef, Lambda)):
        return _reachable_result()
    if isinstance(node, (Break, Continue, Raise, Return)):
        return _reachable_result(falls_through=False)
    if isinstance(node, Assert):
        return _reachable_result(
            contains_invariant=_assertion_has_runtime_structure(
                node.test,
                static_names,
            )
        )
    if isinstance(node, If):
        return _analyze_reachable_if(node, static_names)
    if isinstance(node, (AsyncFor, For)):
        return _analyze_reachable_for(node, static_names)
    if isinstance(node, While):
        return _analyze_reachable_while(node, static_names)
    if isinstance(node, (AsyncWith, With)):
        return _analyze_reachable_with(node, static_names)
    if isinstance(node, Try):
        if node.handlers or node.orelse:
            return _reachable_result(falls_through=False)
        body = _analyze_reachable_block(node.body, static_names)
        finalbody = _analyze_reachable_block(node.finalbody, static_names)
        return _reachable_result(
            contains_invariant=(
                body.contains_invariant and finalbody.falls_through
            ),
            falls_through=body.falls_through and finalbody.falls_through,
        )
    if isinstance(node, (Match, TryStar)):
        return _reachable_result(falls_through=False)
    return _reachable_result()


def _analyze_reachable_if(
    node: If,
    static_names: frozenset[str],
) -> _ReachableInvariantResult:
    """Analyze reachable branches of one conditional statement."""
    truth = _literal_truth_value(node.test)
    if truth is True:
        return _analyze_reachable_block(node.body, static_names)
    if truth is False:
        return _analyze_reachable_block(node.orelse, static_names)
    body = _analyze_reachable_block(node.body, static_names)
    orelse = _analyze_reachable_block(node.orelse, static_names)
    return _reachable_result(
        falls_through=body.falls_through and orelse.falls_through,
    )


def _analyze_reachable_for(
    node: AsyncFor | For,
    static_names: frozenset[str],
) -> _ReachableInvariantResult:
    """Analyze loop blocks, excluding safely known empty iterations."""
    emptiness = (
        None
        if isinstance(node, AsyncFor)
        else _literal_finite_iterable_emptiness(node.iter)
    )
    orelse = _analyze_reachable_block(node.orelse, static_names)
    if emptiness is True:
        return orelse
    body = _analyze_reachable_block(node.body, static_names)
    if emptiness is False:
        return _reachable_result(
            contains_invariant=body.contains_invariant,
            falls_through=body.falls_through and orelse.falls_through,
        )
    return _reachable_result(
        falls_through=body.falls_through and orelse.falls_through,
    )


def _analyze_reachable_while(
    node: While,
    static_names: frozenset[str],
) -> _ReachableInvariantResult:
    """Analyze a while loop with literal condition reachability."""
    truth = _literal_truth_value(node.test)
    orelse = _analyze_reachable_block(node.orelse, static_names)
    if truth is False:
        return orelse
    body = _analyze_reachable_block(node.body, static_names)
    if truth is True:
        return _reachable_result(
            contains_invariant=body.contains_invariant,
            falls_through=False,
        )
    return _reachable_result(
        falls_through=body.falls_through and orelse.falls_through,
    )


def _analyze_reachable_with(
    node: AsyncWith | With,
    static_names: frozenset[str],
) -> _ReachableInvariantResult:
    """Analyze an ordered context-manager body."""
    is_pytest_raises = any(
        _is_pytest_raises_call(item.context_expr, static_names)
        for item in node.items
    )
    body = _analyze_reachable_block(node.body, static_names)
    return _reachable_result(
        contains_invariant=is_pytest_raises or body.contains_invariant,
        falls_through=(True if is_pytest_raises else body.falls_through),
    )


def _reachable_result(
    *,
    contains_invariant: bool = False,
    falls_through: bool = True,
) -> _ReachableInvariantResult:
    """Return one bounded positive-invariant reachability result."""
    return _ReachableInvariantResult(
        contains_invariant=contains_invariant,
        falls_through=falls_through,
    )


def _literal_finite_iterable_emptiness(node: AST) -> bool | None:
    """Return literal finite-iterable emptiness or None when unknown."""
    if (
        isinstance(node, Call)
        and isinstance(node.func, Name)
        and node.func.id == "set"
        and not node.args
        and not node.keywords
    ):
        return True
    if (
        isinstance(node, Call)
        and isinstance(node.func, Name)
        and node.func.id == "range"
        and not node.keywords
        and 1 <= len(node.args) <= 3
    ):
        values: list[int] = []
        for argument in node.args:
            try:
                value = literal_eval(argument)
            except (TypeError, ValueError):
                return None
            if type(value) is not int:
                return None
            values.append(value)
        try:
            finite_range = range(*values)
        except ValueError:
            return None
        if finite_range.step > 0:
            return finite_range.start >= finite_range.stop
        return finite_range.start <= finite_range.stop
    try:
        value = literal_eval(node)
    except (TypeError, ValueError):
        return None
    if isinstance(value, (bytes, dict, list, set, str, tuple)):
        return not value
    return None


def _assertion_has_runtime_structure(
    node: AST,
    static_names: frozenset[str],
) -> bool:
    """Return whether an assertion executes or accesses runtime state."""
    if isinstance(node, Lambda):
        return False
    if isinstance(node, (Attribute, Call, Subscript)) and not (
        _is_static_reference(node, static_names)
    ):
        return True
    return any(
        _assertion_has_runtime_structure(child, static_names)
        for child in iter_child_nodes(node)
    )


def _is_pytest_raises_call(
    node: AST,
    static_names: frozenset[str],
) -> bool:
    """Return whether a context expression is exactly pytest.raises(...)."""
    return (
        isinstance(node, Call)
        and isinstance(node.func, Attribute)
        and isinstance(node.func.value, Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "raises"
        and "pytest" not in static_names
    )


def _is_static_reference(node: AST, static_names: frozenset[str]) -> bool:
    """Return whether an access is rooted in local static-only state."""
    try:
        literal_eval(node)
    except (TypeError, ValueError):
        pass
    else:
        return True
    if isinstance(node, Lambda):
        return True
    if isinstance(node, Name):
        return node.id in static_names
    if isinstance(node, (Attribute, Subscript)):
        return _is_static_reference(node.value, static_names)
    if isinstance(node, Call):
        return _is_static_reference(node.func, static_names)
    return False


def _static_local_binding_names(
    statements: tuple[AST, ...],
) -> frozenset[str]:
    """Return names bound only to literals, lambdas, or local declarations."""
    bindings: dict[str, list[AST | None]] = {}
    dynamic_names: set[str] = set()

    def record(target: AST, value: AST | None) -> None:
        for name in _assignment_target_names(target):
            bindings.setdefault(name, []).append(value)

    def inspect(node: AST) -> None:
        if isinstance(node, (AsyncFunctionDef, ClassDef, FunctionDef)):
            bindings.setdefault(node.name, []).append(None)
            return
        if isinstance(node, Lambda):
            return
        if isinstance(node, Assign):
            for target in node.targets:
                record(target, node.value)
            inspect(node.value)
            return
        if isinstance(node, AnnAssign):
            record(node.target, node.value)
            if node.value is not None:
                inspect(node.value)
            return
        if isinstance(node, AugAssign):
            dynamic_names.update(_assignment_target_names(node.target))
            inspect(node.value)
            return
        if isinstance(node, NamedExpr):
            record(node.target, node.value)
            inspect(node.value)
            return
        for child in iter_child_nodes(node):
            inspect(child)

    for statement in statements:
        inspect(statement)
    static_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, values in bindings.items():
            if (
                name not in static_names
                and name not in dynamic_names
                and all(
                    _is_static_binding_value(value, static_names)
                    for value in values
                )
            ):
                static_names.add(name)
                changed = True
    return frozenset(static_names)


def _assignment_target_names(target: AST) -> frozenset[str]:
    """Return local names stored by one assignment target."""
    return frozenset(
        node.id
        for node in walk(target)
        if isinstance(node, Name) and isinstance(node.ctx, Store)
    )


def _is_static_binding_value(
    value: AST | None,
    static_names: set[str],
) -> bool:
    """Return whether a local binding has only static test-local state."""
    if value is None or isinstance(value, Lambda):
        return True
    try:
        literal_eval(value)
    except (TypeError, ValueError):
        return isinstance(value, Name) and value.id in static_names
    return True


def _literal_truth_value(node: AST) -> bool | None:
    """Return literal truthiness or None for a runtime expression."""
    try:
        return bool(literal_eval(node))
    except (TypeError, ValueError):
        return None


def mapping(value: object, label: str) -> dict[str, object]:
    """Return one object mapping or reject its shape."""
    if not isinstance(value, dict):
        raise ContractGateError(f"{label} must be an object")
    return cast(dict[str, object], value)


def object_list(value: object, label: str) -> list[object]:
    """Return one object list or reject its shape."""
    if not isinstance(value, list):
        raise ContractGateError(f"{label} must be a list")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractGateError(f"{label} must be a non-empty string")
    return value
