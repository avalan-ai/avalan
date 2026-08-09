#!/usr/bin/env python
"""Verify the frozen strict typing boundary for the dormant patch feature."""

from argparse import ArgumentParser, Namespace
from ast import (
    AST,
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    Attribute,
    BinOp,
    BitOr,
    Call,
    ClassDef,
    FunctionDef,
    Import,
    ImportFrom,
    Module,
    Name,
    Subscript,
    Tuple,
    walk,
)
from ast import parse as parse_python
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from os import environ, pathsep
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from subprocess import CompletedProcess, run
from sys import executable, stderr
from tempfile import TemporaryDirectory

from contract_gate import StrictJsonError, canonical_sha256, strict_json_path

_FEATURE = "patch"
_CURRENT_PHASE = 5
_FIXTURE_ROOT = PurePosixPath("tests/patch_type_contracts")
_DIAGNOSTIC_PATTERN = compile_regex(r"^.+:[0-9]+: error: .+ \[[a-z-]+\]$")
_PROHIBITED_SOURCE_PATTERN = compile_regex(
    r"\bAny\b|\bcast\s*\(|#\s*type:\s*ignore|\bdict\[str,\s*object\]"
)
_PATCH_SOURCE_SCOPES = frozenset(
    (
        "patch_script",
        "integration_hunk",
        "changed_python",
        "patch_domain",
    )
)
_UNSAFE_AUTHORITY_TOKENS = frozenset(
    (
        "approval",
        "authority",
        "capability",
        "grant",
        "plan",
        "subject",
        "action",
    )
)
_TYPE_MODULES = frozenset(("typing", "typing_extensions", "collections.abc"))
PATCH_PYTHON_OWNERSHIP_ENV = "AVALAN_PATCH_PYTHON_OWNERSHIP"
PATCH_PYTHON_OWNERSHIP_SHA256_ENV = "AVALAN_PATCH_PYTHON_OWNERSHIP_SHA256"


class PatchTypeContractError(RuntimeError):
    """Report a frozen patch type-contract violation."""


@dataclass(frozen=True, kw_only=True, slots=True)
class TypeFixture:
    """Store one positive or intentionally rejected mypy fixture."""

    identifier: str
    kind: str
    lifecycle: str
    active_from_phase: int
    path: str
    source_sha256: str
    expected_diagnostics: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class StrictSource:
    """Store one patch script or integration hunk subject to strict checks."""

    identifier: str
    path: str
    scope: str
    symbols: tuple[str, ...]
    source_sha256: str


@dataclass(frozen=True, kw_only=True, slots=True)
class TypeManifest:
    """Store the immutable patch type-fixture manifest."""

    current_phase: int
    fixtures: tuple[TypeFixture, ...]
    sources: tuple[StrictSource, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class TypingAliases:
    """Store imported spellings for the closed typing escape-hatch scan."""

    modules: frozenset[str]
    any_names: frozenset[str]
    cast_names: frozenset[str]
    awaitable_names: frozenset[str]
    mapping_names: frozenset[str]


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def default_manifest_path() -> Path:
    """Return the tracked patch type-contract manifest path."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "patch"
        / "type_contract_manifest.json"
    )


def load_manifest(path: Path) -> TypeManifest:
    """Load and validate the immutable type fixture inventory."""
    try:
        raw = strict_json_path(path)
    except StrictJsonError as exc:
        raise PatchTypeContractError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise PatchTypeContractError("patch type manifest must be an object")
    expected = {
        "schema_version",
        "feature",
        "current_phase",
        "fixtures",
        "sources",
        "manifest_sha256",
    }
    if set(raw) != expected:
        raise PatchTypeContractError("patch type manifest has invalid keys")
    if raw.get("schema_version") != 1 or raw.get("feature") != _FEATURE:
        raise PatchTypeContractError("patch type manifest header is invalid")
    current_phase = _phase(raw.get("current_phase"), "current_phase")
    if current_phase != _CURRENT_PHASE:
        raise PatchTypeContractError("patch type manifest phase is not frozen")
    fixtures_value = raw.get("fixtures")
    if not isinstance(fixtures_value, list) or not fixtures_value:
        raise PatchTypeContractError("patch type fixtures must be non-empty")
    fixtures = tuple(_fixture(item, current_phase) for item in fixtures_value)
    identifiers = tuple(item.identifier for item in fixtures)
    if len(identifiers) != len(set(identifiers)):
        raise PatchTypeContractError("patch type fixture IDs are duplicated")
    active = tuple(item for item in fixtures if item.lifecycle == "active")
    if not active:
        raise PatchTypeContractError(
            "patch type fixture inventory is inactive"
        )
    if not any(item.kind == "positive" for item in active) or not any(
        item.kind == "negative" for item in active
    ):
        raise PatchTypeContractError(
            "patch type inventory requires positive and negative evidence"
        )
    sources_value = raw.get("sources")
    if not isinstance(sources_value, list) or not sources_value:
        raise PatchTypeContractError("patch strict source inventory is empty")
    sources = tuple(_strict_source(item) for item in sources_value)
    if not _PATCH_SOURCE_SCOPES.issubset({item.scope for item in sources}):
        raise PatchTypeContractError(
            "patch strict source scopes are incomplete"
        )
    identifiers = tuple(item.identifier for item in sources)
    paths = tuple(item.path for item in sources)
    if len(identifiers) != len(set(identifiers)) or len(paths) != len(
        set(paths)
    ):
        raise PatchTypeContractError(
            "patch strict source inventory is duplicated"
        )
    canonical = {
        key: value for key, value in raw.items() if key != "manifest_sha256"
    }
    if raw.get("manifest_sha256") != canonical_sha256(canonical):
        raise PatchTypeContractError("patch type manifest digest is invalid")
    return TypeManifest(
        current_phase=current_phase,
        fixtures=fixtures,
        sources=sources,
    )


def verify_patch_types(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
) -> TypeManifest:
    """Run strict mypy against all active fixtures through one phase."""
    root = (repo_root or repository_root()).resolve()
    manifest = load_manifest(manifest_path or default_manifest_path())
    if through_phase < 0 or through_phase > manifest.current_phase:
        raise PatchTypeContractError("patch type phase is not implemented")
    selected = tuple(
        item
        for item in manifest.fixtures
        if item.lifecycle == "active"
        and item.active_from_phase <= through_phase
    )
    if not selected:
        raise PatchTypeContractError("patch type phase has no active fixtures")
    _verify_strict_sources(root, manifest.sources, manifest.fixtures)
    environment = _fixture_mypy_environment(root)
    _verify_type_fixtures(root, selected, environment)
    return manifest


def _fixture(raw: object, current_phase: int) -> TypeFixture:
    """Validate one closed type-fixture record."""
    if not isinstance(raw, dict):
        raise PatchTypeContractError("patch type fixture must be an object")
    expected = {
        "id",
        "kind",
        "lifecycle",
        "active_from_phase",
        "path",
        "source_sha256",
        "expected_diagnostics",
    }
    if set(raw) != expected:
        raise PatchTypeContractError("patch type fixture has invalid keys")
    identifier = _string(raw.get("id"), "type fixture ID")
    kind = _string(raw.get("kind"), "type fixture kind")
    lifecycle = _string(raw.get("lifecycle"), "type fixture lifecycle")
    active_from_phase = _phase(
        raw.get("active_from_phase"), "type fixture phase"
    )
    if kind not in {"positive", "negative"}:
        raise PatchTypeContractError("patch type fixture kind is invalid")
    expected_lifecycle = (
        "active" if active_from_phase <= current_phase else "planned"
    )
    if lifecycle != expected_lifecycle:
        raise PatchTypeContractError("patch type fixture lifecycle is invalid")
    path = _string(raw.get("path"), "type fixture path")
    _validate_fixture_path(path)
    source_sha256 = _sha256(
        raw.get("source_sha256"), "type fixture source digest"
    )
    diagnostics_value = raw.get("expected_diagnostics")
    if not isinstance(diagnostics_value, list) or not all(
        isinstance(item, str) and item for item in diagnostics_value
    ):
        raise PatchTypeContractError("type diagnostics must be a string list")
    diagnostics = tuple(diagnostics_value)
    if (kind == "positive" and diagnostics) or (
        kind == "negative" and not diagnostics
    ):
        raise PatchTypeContractError("type fixture diagnostic kind is invalid")
    return TypeFixture(
        identifier=identifier,
        kind=kind,
        lifecycle=lifecycle,
        active_from_phase=active_from_phase,
        path=path,
        source_sha256=source_sha256,
        expected_diagnostics=diagnostics,
    )


def _strict_source(raw: object) -> StrictSource:
    """Validate one complete patch-owned strict source inventory entry."""
    if not isinstance(raw, dict):
        raise PatchTypeContractError("patch strict source must be an object")
    expected = {"id", "path", "scope", "symbols", "source_sha256"}
    if set(raw) != expected:
        raise PatchTypeContractError("patch strict source has invalid keys")
    identifier = _string(raw.get("id"), "patch strict source ID")
    path = _string(raw.get("path"), "patch strict source path")
    _validate_source_path(path)
    scope = _string(raw.get("scope"), "patch strict source scope")
    if scope not in _PATCH_SOURCE_SCOPES:
        raise PatchTypeContractError("patch strict source scope is invalid")
    symbols_value = raw.get("symbols")
    if not isinstance(symbols_value, list) or not symbols_value:
        raise PatchTypeContractError("patch strict source symbols are invalid")
    symbols = tuple(
        _string(item, "patch strict source symbol") for item in symbols_value
    )
    if len(symbols) != len(set(symbols)):
        raise PatchTypeContractError(
            "patch strict source symbols are duplicated"
        )
    if scope in {"patch_script", "patch_domain"} and symbols != ("module",):
        raise PatchTypeContractError("patch script must inventory its module")
    if scope == "integration_hunk" and "module" in symbols:
        raise PatchTypeContractError(
            "integration hunk must name changed symbols"
        )
    if scope == "changed_python" and symbols != ("module",):
        raise PatchTypeContractError(
            "changed Python path must inventory its module"
        )
    return StrictSource(
        identifier=identifier,
        path=path,
        scope=scope,
        symbols=symbols,
        source_sha256=_sha256(
            raw.get("source_sha256"), "patch strict source digest"
        ),
    )


def _verify_strict_sources(
    root: Path,
    sources: tuple[StrictSource, ...],
    fixtures: tuple[TypeFixture, ...],
) -> None:
    """Run strict mypy and structural gates over every owned source entry."""
    owned_paths = {
        *(source.path for source in sources),
        *(fixture.path for fixture in fixtures),
    }
    discovered_paths = _repository_python_paths(root)
    unowned = tuple(
        path for path in discovered_paths if path.as_posix() not in owned_paths
    )
    if unowned:
        raise PatchTypeContractError(
            "changed or untracked Python path is not owned: "
            + ",".join(path.as_posix() for path in unowned)
        )
    paths: list[str] = []
    for source in sources:
        path = _source_path(root, source.path)
        if not path.is_file() or path.is_symlink():
            raise PatchTypeContractError(
                f"patch strict source is missing: {source.path}"
            )
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != source.source_sha256:
            raise PatchTypeContractError(
                f"patch strict source digest changed: {source.identifier}"
            )
        try:
            tree = parse_python(payload.decode("utf-8"), type_comments=True)
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise PatchTypeContractError(
                f"patch strict source is not parseable: {source.path}"
            ) from exc
        _verify_symbols(tree, source)
        _verify_strict_ast(tree, source)
        paths.append(source.path)
    _run_strict_source_mypy(root, tuple(paths))


def discover_repository_python_paths(
    root: Path,
) -> tuple[PurePosixPath, ...]:
    """Discover changed and untracked Python paths from Git state."""
    commands = (
        (
            "git",
            "diff",
            "--name-only",
            "-z",
            "HEAD",
            "--",
            "scripts",
            "src",
            "tests",
        ),
        (
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "scripts",
            "src",
            "tests",
        ),
    )
    paths: set[PurePosixPath] = set()
    for command in commands:
        completed = run(
            command,
            cwd=root,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise PatchTypeContractError(
                "cannot inspect repository Python ownership"
            )
        for raw in completed.stdout.split(b"\0"):
            if not raw:
                continue
            try:
                candidate = PurePosixPath(raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise PatchTypeContractError(
                    "repository Python path is not UTF-8"
                ) from exc
            if candidate.suffix == ".py" and not candidate.is_absolute():
                paths.add(candidate)
    return tuple(sorted(paths))


def repository_python_ownership_environment(
    paths: tuple[PurePosixPath, ...],
) -> dict[str, str]:
    """Return an integrity-bound changed Python ownership inventory."""
    value = _encode_repository_python_paths(paths)
    return {
        PATCH_PYTHON_OWNERSHIP_ENV: value,
        PATCH_PYTHON_OWNERSHIP_SHA256_ENV: (
            sha256(value.encode("utf-8")).hexdigest()
        ),
    }


def _repository_python_paths(root: Path) -> tuple[PurePosixPath, ...]:
    """Return Git paths or a sealed inventory in a Git-free mirror."""
    value = environ.get(PATCH_PYTHON_OWNERSHIP_ENV)
    digest = environ.get(PATCH_PYTHON_OWNERSHIP_SHA256_ENV)
    if value is None and digest is None:
        return discover_repository_python_paths(root)
    if value is None or digest is None:
        raise PatchTypeContractError(
            "patch Python ownership inventory state is incomplete"
        )
    if sha256(value.encode("utf-8")).hexdigest() != digest:
        raise PatchTypeContractError(
            "patch Python ownership inventory integrity is invalid"
        )
    try:
        raw = loads(value)
    except JSONDecodeError as exc:
        raise PatchTypeContractError(
            "patch Python ownership inventory is invalid JSON"
        ) from exc
    if not isinstance(raw, list) or any(
        not isinstance(item, str) for item in raw
    ):
        raise PatchTypeContractError(
            "patch Python ownership inventory is malformed"
        )
    paths = tuple(_sealed_repository_python_path(root, item) for item in raw)
    if value != _encode_repository_python_paths(paths):
        raise PatchTypeContractError(
            "patch Python ownership inventory is not canonical"
        )
    return paths


def _encode_repository_python_paths(
    paths: tuple[PurePosixPath, ...],
) -> str:
    """Encode one sorted, unique Python ownership inventory."""
    values = tuple(path.as_posix() for path in paths)
    if values != tuple(sorted(set(values))):
        raise PatchTypeContractError(
            "patch Python ownership inventory is not sorted and unique"
        )
    return dumps(values, ensure_ascii=True, separators=(",", ":"))


def _sealed_repository_python_path(root: Path, value: str) -> PurePosixPath:
    """Validate one mirror-relative Python ownership path."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.suffix != ".py"
        or not path.parts
        or path.parts[0] not in {"scripts", "src", "tests"}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PatchTypeContractError(
            "patch Python ownership inventory path is invalid"
        )
    target = root / path
    if target.is_symlink() or not target.is_file():
        raise PatchTypeContractError(
            "patch Python ownership inventory path is missing"
        )
    return path


def _verify_symbols(tree: Module, source: StrictSource) -> None:
    """Require each changed integration declaration to remain addressable."""
    if source.symbols == ("module",):
        return
    found = {
        node.name
        for node in walk(tree)
        if isinstance(node, (AsyncFunctionDef, ClassDef, FunctionDef))
    }
    found.update(
        target.id
        for node in walk(tree)
        if isinstance(node, AnnAssign) and isinstance(node.target, Name)
        for target in (node.target,)
    )
    found.update(
        target.id
        for node in walk(tree)
        if isinstance(node, Assign)
        for target in node.targets
        if isinstance(target, Name)
    )
    missing = set(source.symbols) - found
    if missing:
        raise PatchTypeContractError(
            "patch strict source symbol is missing: "
            f"{source.path}:{','.join(sorted(missing))}"
        )


def _verify_strict_ast(tree: Module, source: StrictSource) -> None:
    """Reject the closed collection of type-boundary escape hatches."""
    nodes = _strict_nodes(tree, source)
    aliases = _typing_aliases(tree)
    if source.scope in {"patch_script", "patch_domain", "changed_python"}:
        for ignored in tree.type_ignores:
            _strict_source_error(source, ignored, "type-ignore")
    for node in nodes:
        if _matches_typing_symbol(node, aliases, "Any"):
            _strict_source_error(source, node, "Any")
        if isinstance(node, Call) and _matches_typing_symbol(
            node.func, aliases, "cast"
        ):
            _strict_source_error(source, node, "cast")
        if isinstance(node, Subscript) and _is_free_form_mapping(
            node, aliases
        ):
            _strict_source_error(source, node, "free-form-mapping")
        if isinstance(node, AnnAssign) and _is_stringly_authority(node):
            _strict_source_error(source, node, "stringly-trust-state")
        if isinstance(node, ClassDef):
            _verify_immutable_authority(node, source)
        if isinstance(node, (AsyncFunctionDef, FunctionDef)):
            _verify_typed_callable(node, source)
        if isinstance(node, BinOp) and isinstance(node.op, BitOr):
            if _contains_typing_symbol(node, aliases, "Awaitable"):
                _strict_source_error(source, node, "sync-or-awaitable")


def _strict_nodes(tree: Module, source: StrictSource) -> tuple[AST, ...]:
    """Return every patch script node or only the named integration hunks."""
    if source.scope in {"patch_script", "patch_domain"}:
        return tuple(walk(tree))
    declarations = tuple(
        node
        for node in tree.body
        if _declares_strict_symbol(node, source.symbols)
    )
    return tuple(item for node in declarations for item in walk(node))


def _declares_strict_symbol(node: AST, symbols: tuple[str, ...]) -> bool:
    """Return whether one node declares an inventoried hunk symbol."""
    if isinstance(node, (AsyncFunctionDef, ClassDef, FunctionDef)):
        return node.name in symbols
    if isinstance(node, AnnAssign) and isinstance(node.target, Name):
        return node.target.id in symbols
    if isinstance(node, Assign):
        return any(
            isinstance(target, Name) and target.id in symbols
            for target in node.targets
        )
    return False


def _typing_aliases(tree: Module) -> TypingAliases:
    """Return direct, aliased, and qualified typing spellings in one module."""
    modules: set[str] = set()
    any_names = {"Any"}
    cast_names = {"cast"}
    awaitable_names = {"Awaitable"}
    mapping_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, Import):
            for imported in node.names:
                if imported.name in _TYPE_MODULES:
                    modules.add(imported.asname or imported.name)
        if isinstance(node, ImportFrom) and node.module in _TYPE_MODULES:
            for imported in node.names:
                name = imported.asname or imported.name
                match imported.name:
                    case "Any":
                        any_names.add(name)
                    case "cast":
                        cast_names.add(name)
                    case "Awaitable":
                        awaitable_names.add(name)
                    case "Mapping" if node.module != "collections.abc":
                        mapping_names.add(name)
    return TypingAliases(
        modules=frozenset(modules),
        any_names=frozenset(any_names),
        cast_names=frozenset(cast_names),
        awaitable_names=frozenset(awaitable_names),
        mapping_names=frozenset(mapping_names),
    )


def _matches_typing_symbol(
    node: AST, aliases: TypingAliases, symbol: str
) -> bool:
    """Return whether one node resolves to a closed typing escape hatch."""
    names = {
        "Any": aliases.any_names,
        "cast": aliases.cast_names,
        "Awaitable": aliases.awaitable_names,
        "Mapping": aliases.mapping_names,
    }[symbol]
    if isinstance(node, Name):
        return node.id in names
    return (
        isinstance(node, Attribute)
        and node.attr == symbol
        and isinstance(node.value, Name)
        and node.value.id in aliases.modules
        and not (symbol == "Mapping" and node.value.id == "collections.abc")
    )


def _is_free_form_mapping(node: Subscript, aliases: TypingAliases) -> bool:
    """Return whether a dictionary annotation erases a patch value type."""
    return (
        (
            isinstance(node.value, Name)
            and node.value.id == "dict"
            or _matches_typing_symbol(node.value, aliases, "Mapping")
        )
        and isinstance(node.slice, Tuple)
        and len(node.slice.elts) == 2
        and isinstance(node.slice.elts[0], Name)
        and node.slice.elts[0].id == "str"
        and isinstance(node.slice.elts[1], Name)
        and node.slice.elts[1].id == "object"
    )


def _is_stringly_authority(node: AnnAssign) -> bool:
    """Return whether an authority-bearing field is represented by ``str``."""
    if not isinstance(node.target, Name) or not isinstance(
        node.annotation, Name
    ):
        return False
    return node.annotation.id == "str" and any(
        token in node.target.id.lower() for token in _UNSAFE_AUTHORITY_TOKENS
    )


def _verify_immutable_authority(node: ClassDef, source: StrictSource) -> None:
    """Reject mutable dataclasses whose names imply mutation authority."""
    if not any(
        token in node.name.lower() for token in _UNSAFE_AUTHORITY_TOKENS
    ):
        return
    decorators = tuple(node.decorator_list)
    dataclass_calls = tuple(
        decorator
        for decorator in decorators
        if isinstance(decorator, Call)
        and isinstance(decorator.func, Name)
        and decorator.func.id == "dataclass"
    )
    for decorator in dataclass_calls:
        frozen = next(
            (
                keyword.value
                for keyword in decorator.keywords
                if keyword.arg == "frozen"
            ),
            None,
        )
        if frozen is None or not getattr(frozen, "value", False):
            _strict_source_error(source, decorator, "mutable-authority")


def _verify_typed_callable(
    node: AsyncFunctionDef | FunctionDef, source: StrictSource
) -> None:
    """Require parameters and callbacks to retain explicit type boundaries."""
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )
    if node.args.vararg is not None:
        arguments = (*arguments, node.args.vararg)
    if node.args.kwarg is not None:
        arguments = (*arguments, node.args.kwarg)
    user_arguments = tuple(
        argument
        for argument in arguments
        if argument.arg not in {"self", "cls"}
    )
    if any(argument.annotation is None for argument in user_arguments):
        _strict_source_error(source, node, "untyped-callback")
    if node.returns is None:
        _strict_source_error(source, node, "untyped-callback")


def _contains_typing_symbol(
    node: AST, aliases: TypingAliases, symbol: str
) -> bool:
    """Return whether one annotation tree resolves a typing escape hatch."""
    return any(
        _matches_typing_symbol(item, aliases, symbol) for item in walk(node)
    )


def _strict_source_error(
    source: StrictSource,
    node: AST,
    rule: str,
) -> None:
    """Raise the stable diagnostic expected by negative strict fixtures."""
    line = _node_line(node)
    raise PatchTypeContractError(
        f"patch strict source violation: {source.path}:{line}: {rule}"
    )


def _node_line(node: AST) -> int:
    """Return one AST node's required source line number."""
    line = getattr(node, "lineno", None)
    if not isinstance(line, int):
        raise PatchTypeContractError("patch type node has no line number")
    return line


def _run_strict_source_mypy(root: Path, paths: tuple[str, ...]) -> None:
    """Run strict mypy over the complete frozen source inventory."""
    environment = {
        key: value
        for key, value in environ.items()
        if key.upper() != "PYTHONPATH" and not key.upper().startswith("MYPY")
    }
    environment["MYPYPATH"] = pathsep.join(
        (
            str(root / "tests"),
            str(root / "src"),
            str(root / "scripts"),
        )
    )
    completed = run(
        (
            executable,
            "-m",
            "mypy",
            "--strict",
            "--disallow-any-explicit",
            "--follow-imports=silent",
            "--cache-dir=/dev/null",
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
            *paths,
        ),
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        raise PatchTypeContractError(
            "patch strict source mypy failed:\n"
            f"{completed.stdout}{completed.stderr}"
        )


def _fixture_mypy_environment(root: Path) -> dict[str, str]:
    """Return the isolated import environment for one type fixture."""
    environment = {
        key: value
        for key, value in environ.items()
        if key.upper() != "PYTHONPATH" and not key.upper().startswith("MYPY")
    }
    environment["MYPYPATH"] = pathsep.join(
        (
            str(root / "tests"),
            str(root / "src"),
        )
    )
    return environment


def _verify_type_fixtures(
    root: Path,
    fixtures: tuple[TypeFixture, ...],
    environment: dict[str, str],
) -> None:
    """Run all selected fixtures against one fresh shared cache."""
    with _fixture_mypy_cache() as cache_path:
        for fixture in fixtures:
            fixture_path = _fixture_path(root, fixture.path)
            source = _read_fixture(fixture_path, fixture)
            static_diagnostics = _fixture_static_diagnostics(
                source,
                fixture.path,
            )
            completed = _run_fixture_mypy(
                root,
                fixture.path,
                environment,
                cache_path,
            )
            output = completed.stdout + completed.stderr
            all_mypy_diagnostics = _mypy_diagnostics(output)
            mypy_diagnostics = _fixture_mypy_diagnostics(output, fixture.path)
            diagnostics = (*static_diagnostics, *mypy_diagnostics)
            if fixture.kind == "positive":
                if diagnostics or (
                    completed.returncode != 0 and not all_mypy_diagnostics
                ):
                    raise PatchTypeContractError(
                        "positive type fixture failed:"
                        f" {fixture.identifier}\n{output}"
                    )
                continue
            if completed.returncode == 0 and not static_diagnostics:
                raise PatchTypeContractError(
                    "negative type fixture unexpectedly passed: "
                    f"{fixture.identifier}"
                )
            if diagnostics != fixture.expected_diagnostics:
                raise PatchTypeContractError(
                    "negative type fixture diagnostics changed:"
                    f" {fixture.identifier},"
                    f" expected={fixture.expected_diagnostics},"
                    f" observed={diagnostics}\n{output}"
                )


@contextmanager
def _fixture_mypy_cache() -> Iterator[Path]:
    """Yield one invocation-scoped cache removed after fixture validation."""
    with TemporaryDirectory(prefix="avalan-patch-mypy-") as cache_directory:
        yield Path(cache_directory)


def _run_fixture_mypy(
    root: Path,
    fixture_path: str,
    environment: dict[str, str],
    cache_path: Path,
) -> CompletedProcess[str]:
    """Run one fixture with cold shared import checking and local output."""
    return run(
        (
            executable,
            "-m",
            "mypy",
            "--strict",
            "--disallow-any-explicit",
            "--follow-imports=silent",
            f"--cache-dir={cache_path}",
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
            fixture_path,
        ),
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def _mypy_diagnostics(output: str) -> tuple[str, ...]:
    """Return every normalized mypy diagnostic from one command output."""
    return tuple(
        line.strip()
        for line in output.splitlines()
        if _DIAGNOSTIC_PATTERN.match(line.strip())
    )


def _fixture_mypy_diagnostics(
    output: str, fixture_path: str
) -> tuple[str, ...]:
    """Return only diagnostics emitted for the owned type fixture path."""
    prefix = f"{fixture_path}:"
    return tuple(
        diagnostic
        for diagnostic in _mypy_diagnostics(output)
        if diagnostic.startswith(prefix)
    )


def _fixture_path(root: Path, value: str) -> Path:
    """Resolve one fixture path below the repository root."""
    path = (root / Path(*PurePosixPath(value).parts)).resolve()
    if not path.is_relative_to(root):
        raise PatchTypeContractError(
            "type fixture path escapes the repository"
        )
    return path


def _read_fixture(path: Path, fixture: TypeFixture) -> str:
    """Read and hash-check one strict type fixture."""
    if not path.is_file() or path.is_symlink():
        raise PatchTypeContractError(
            f"type fixture is missing: {fixture.path}"
        )
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != fixture.source_sha256:
        raise PatchTypeContractError(
            f"type fixture source digest changed: {fixture.identifier}"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchTypeContractError("type fixture is not UTF-8") from exc


def _fixture_static_diagnostics(source: str, path: str) -> tuple[str, ...]:
    """Return exact static escape-hatch diagnostics for one fixture source."""
    try:
        tree = parse_python(source, type_comments=True)
    except SyntaxError as exc:
        raise PatchTypeContractError(
            "patch type fixture is not parseable"
        ) from exc
    diagnostics: list[tuple[int, str]] = []
    aliases = _typing_aliases(tree)
    for ignored in tree.type_ignores:
        diagnostics.append((ignored.lineno, "type-ignore"))
    for node in walk(tree):
        diagnostic = _fixture_rule(node, aliases)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return tuple(
        f"{path}:{line}: error: patch strict type fixture forbids {rule} "
        f"[patch-{rule}]"
        for line, rule in sorted(diagnostics)
    )


def _fixture_rule(node: AST, aliases: TypingAliases) -> tuple[int, str] | None:
    """Return one forbidden static fixture rule for an AST node if present."""
    if _matches_typing_symbol(node, aliases, "Any"):
        return _node_line(node), "Any"
    if isinstance(node, Call) and _matches_typing_symbol(
        node.func, aliases, "cast"
    ):
        return node.lineno, "cast"
    if isinstance(node, Subscript) and _is_free_form_mapping(node, aliases):
        return node.lineno, "free-form-mapping"
    if isinstance(node, AnnAssign) and _is_stringly_authority(node):
        return node.lineno, "stringly-trust-state"
    if isinstance(node, ClassDef) and _has_mutable_authority(node):
        return node.lineno, "mutable-authority"
    if isinstance(
        node, (AsyncFunctionDef, FunctionDef)
    ) and _has_untyped_callable(node):
        return node.lineno, "untyped-callback"
    if (
        isinstance(node, BinOp)
        and isinstance(node.op, BitOr)
        and _contains_typing_symbol(node, aliases, "Awaitable")
    ):
        return node.lineno, "sync-or-awaitable"
    return None


def _has_mutable_authority(node: ClassDef) -> bool:
    """Return whether one authority class is a non-frozen dataclass."""
    if not any(
        token in node.name.lower() for token in _UNSAFE_AUTHORITY_TOKENS
    ):
        return False
    dataclass_calls = tuple(
        decorator
        for decorator in node.decorator_list
        if isinstance(decorator, Call)
        and isinstance(decorator.func, Name)
        and decorator.func.id == "dataclass"
    )
    if any(
        isinstance(decorator, Name) and decorator.id == "dataclass"
        for decorator in node.decorator_list
    ):
        return True
    return any(
        not any(
            keyword.arg == "frozen"
            and getattr(keyword.value, "value", False) is True
            for keyword in decorator.keywords
        )
        for decorator in dataclass_calls
    )


def _has_untyped_callable(node: AsyncFunctionDef | FunctionDef) -> bool:
    """Return whether one callable omits a parameter or return annotation."""
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )
    if node.args.vararg is not None:
        arguments = (*arguments, node.args.vararg)
    if node.args.kwarg is not None:
        arguments = (*arguments, node.args.kwarg)
    return node.returns is None or any(
        argument.annotation is None
        for argument in arguments
        if argument.arg not in {"self", "cls"}
    )


def _validate_fixture_path(value: str) -> None:
    """Reject a fixture path outside the dedicated patch directory."""
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.parent != _FIXTURE_ROOT
        or path.suffix != ".py"
    ):
        raise PatchTypeContractError(
            "type fixture path is outside patch fixtures"
        )


def _validate_source_path(value: str) -> None:
    """Reject a strict source path outside the patch gate's source boundary."""
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.suffix != ".py"
        or not path.parts
        or path.parts[0] not in {"scripts", "src", "tests"}
    ):
        raise PatchTypeContractError("patch strict source path is invalid")


def _source_path(root: Path, value: str) -> Path:
    """Resolve one strict source path below the selected repository root."""
    path = (root / Path(*PurePosixPath(value).parts)).resolve()
    if not path.is_relative_to(root):
        raise PatchTypeContractError(
            "patch strict source path escapes repository"
        )
    return path


def _phase(value: object, label: str) -> int:
    """Validate one non-negative integral phase value."""
    if type(value) is not int or value < 0:
        raise PatchTypeContractError(f"{label} must be a non-negative integer")
    return value


def _string(value: object, label: str) -> str:
    """Validate one non-empty textual value."""
    if not isinstance(value, str) or not value:
        raise PatchTypeContractError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    """Validate one lower-case SHA-256 digest."""
    result = _string(value, label)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise PatchTypeContractError(f"{label} must be a SHA-256 digest")
    return result


def _parse_args() -> Namespace:
    """Parse the strict type-gate command line."""
    parser = ArgumentParser(description="Run patch strict type contracts.")
    parser.add_argument("--through-phase", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the strict patch type gate."""
    args = _parse_args()
    try:
        verify_patch_types(through_phase=args.through_phase)
    except PatchTypeContractError as exc:
        print(f"patch type contract failed: {exc}", file=stderr)
        return 1
    print(f"patch type contract passed: through_phase={args.through_phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
