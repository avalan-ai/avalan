"""Verify default patch authority behavior before toolset configuration."""

from pathlib import Path
from runpy import run_path
from subprocess import run
from sys import executable
from typing import Protocol, cast


class _DefaultPatchAuthority(Protocol):
    """Describe the closed default-deny validator surface."""

    @staticmethod
    def capability_is_issued(
        capability: object, service: object, owner: object
    ) -> bool:
        """Return whether one default capability is issued."""

    @staticmethod
    def registration_is_issued(registration: object, owner: object) -> bool:
        """Return whether one default registration is issued."""

    @staticmethod
    def registration_owns(
        registration: object, canonical_name: str, tool: object
    ) -> bool:
        """Return whether one default registration owns a tool."""

    @staticmethod
    def capability_snapshot(capability: object, service: object) -> object:
        """Return the default capability snapshot."""

    @staticmethod
    def loader_is_issued(loader: object) -> bool:
        """Return whether one default loader is issued."""

    @staticmethod
    def sandbox_endpoint_is_issued(endpoint: object) -> bool:
        """Return whether one default endpoint is issued."""


def test_patch_phase_10_runpy_authority_defaults() -> None:
    """Exercise a fresh authority namespace without replacing host hooks."""
    namespace = run_path(str(Path("src/avalan/_patch_authority.py").resolve()))
    validator = cast(
        type[_DefaultPatchAuthority], namespace["_PatchAuthorityValidator"]
    )
    capability = object()
    service = object()
    registration = object()
    owner = object()
    assert not validator.capability_is_issued(capability, service, owner)
    assert not validator.registration_is_issued(registration, owner)
    assert not validator.registration_owns(
        registration, "patch.edit", object()
    )
    assert validator.capability_snapshot(capability, service) is None
    assert not validator.loader_is_issued(object())
    assert not validator.sandbox_endpoint_is_issued(object())


def test_patch_phase_10_pristine_authority_subprocess_denies_everything() -> (
    None
):
    """Exercise defaults before a loader can install sealed authority hooks."""
    program = "\n".join(
        (
            "from avalan._patch_authority import _PatchAuthorityValidator",
            "capability = object()",
            "service = object()",
            "registration = object()",
            "owner = object()",
            "validator = _PatchAuthorityValidator",
            (
                "assert not"
                " validator.capability_is_issued(capability, service, owner)"
            ),
            "assert not validator.registration_is_issued(registration, owner)",
            (
                "assert not"
                " validator.registration_owns(registration, 'patch.edit',"
                " object())"
            ),
            (
                "assert validator.capability_snapshot(capability, service) is"
                " None"
            ),
            "assert not validator.loader_is_issued(object())",
            "assert not validator.sandbox_endpoint_is_issued(object())",
        )
    )
    result = run(
        (executable, "-c", program),
        capture_output=True,
        check=False,
        env={"PYTHONPATH": str((Path.cwd() / "src").resolve())},
        text=True,
    )
    assert result.returncode == 0, result.stderr
