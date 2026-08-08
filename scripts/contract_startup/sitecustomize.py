"""Constrain contract-gate Python startup to declared trusted paths."""

from hashlib import sha256
from os import environ, pathsep
from os.path import commonpath, isabs, isdir, join, realpath
from sys import addaudithook, base_prefix, path, prefix
from typing import Never

_ALLOWED_PATHS_ENV = "AVALAN_CONTRACT_ALLOWED_PYTHONPATH"
_PATCH_ARTIFACT_ROOT_ENV = "AVALAN_PATCH_CONTRACT_ARTIFACT_ROOT"
_PATCH_FORBIDDEN_ARTIFACTS_ENV = "AVALAN_PATCH_CONTRACT_FORBIDDEN_ARTIFACTS"
_PATCH_FORBIDDEN_ARTIFACTS_SHA256_ENV = (
    "AVALAN_PATCH_CONTRACT_FORBIDDEN_ARTIFACTS_SHA256"
)


def _fail(message: str) -> Never:
    """Terminate startup when the contract runtime is not closed."""
    raise SystemExit(f"contract Python startup rejected: {message}")


def _is_relative_to(candidate: str, root: str) -> bool:
    """Return whether one resolved path is contained by another."""
    try:
        return commonpath((candidate, root)) == root
    except ValueError:
        return False


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return values once while preserving their first occurrence."""
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _harden_startup() -> None:
    """Remove unsafe editable and source-local import roots."""
    if environ.get("PYTHONSAFEPATH") != "1":
        _fail("PYTHONSAFEPATH is not enabled")
    if environ.get("PYTHONNOUSERSITE") != "1":
        _fail("PYTHONNOUSERSITE is not enabled")
    raw_allowed = environ.get(_ALLOWED_PATHS_ENV)
    raw_pythonpath = environ.get("PYTHONPATH")
    if not raw_allowed or not raw_pythonpath:
        _fail("trusted path state is missing")
    allowed_values = tuple(raw_allowed.split(pathsep))
    pythonpath_values = tuple(raw_pythonpath.split(pathsep))
    if (
        not allowed_values
        or allowed_values != pythonpath_values
        or any(not value or not isabs(value) for value in allowed_values)
    ):
        _fail("trusted path state is malformed")
    allowed = tuple(realpath(value) for value in allowed_values)
    if len(allowed) != len(set(allowed)) or any(
        not isdir(value) for value in allowed
    ):
        _fail("trusted paths are missing or duplicated")
    expected_sitecustomize = realpath(join(allowed[0], "sitecustomize.py"))
    if realpath(__file__) != expected_sitecustomize:
        _fail("sitecustomize did not load from the trusted startup root")

    system_roots = _deduplicate((realpath(prefix), realpath(base_prefix)))
    system_paths = _deduplicate(
        tuple(
            resolved
            for entry in path
            if entry
            for resolved in (realpath(entry),)
            if any(_is_relative_to(resolved, root) for root in system_roots)
        )
    )
    path[:] = list(_deduplicate((allowed[0], *system_paths, *allowed[1:])))


def _install_patch_artifact_guard() -> None:
    """Reject real opens of the gate-owned ignored patch artifacts."""
    root = environ.get(_PATCH_ARTIFACT_ROOT_ENV)
    forbidden = environ.get(_PATCH_FORBIDDEN_ARTIFACTS_ENV)
    digest = environ.get(_PATCH_FORBIDDEN_ARTIFACTS_SHA256_ENV)
    if root is None and forbidden is None and digest is None:
        return
    if (
        not root
        or not forbidden
        or not digest
        or not isabs(root)
        or not isdir(root)
        or realpath(root) != root
    ):
        _fail("patch artifact guard state is malformed")
    if sha256(forbidden.encode("utf-8")).hexdigest() != digest:
        _fail("patch artifact guard integrity is invalid")
    raw_paths = tuple(forbidden.split(pathsep))
    expected = tuple(realpath(value) for value in raw_paths)
    if (
        not raw_paths
        or len(raw_paths) != len(set(raw_paths))
        or any(
            not value
            or not isabs(value)
            or value != resolved
            or value == root
            or not _is_relative_to(resolved, root)
            for value, resolved in zip(raw_paths, expected, strict=True)
        )
    ):
        _fail("patch artifact guard paths are invalid")

    def reject_forbidden_open(
        event: str, arguments: tuple[object, ...]
    ) -> None:
        """Fail closed when the running process opens a forbidden artifact."""
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if isinstance(candidate, bytes):
            try:
                value = candidate.decode("utf-8")
            except UnicodeDecodeError:
                return
        elif isinstance(candidate, str):
            value = candidate
        else:
            return
        if realpath(value) in expected:
            raise RuntimeError(
                "patch contract guard rejected ignored artifact open"
            )

    addaudithook(reject_forbidden_open)


_harden_startup()
_install_patch_artifact_guard()
