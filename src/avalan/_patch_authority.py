"""Provide the read-only patch authority validation interface."""


class _PatchAuthorityValidator:
    """Read only loader-owned patch authority records."""

    @staticmethod
    def capability_is_issued(
        capability: object,
        service: object,
        owner: object | None = None,
    ) -> bool:
        """Return whether the exact capability was issued for its service."""
        del capability, service, owner
        return False

    @staticmethod
    def registration_is_issued(registration: object, owner: object) -> bool:
        """Return whether an exact toolset witness belongs to its owner."""
        del registration, owner
        return False

    @staticmethod
    def registration_owns(
        registration: object,
        canonical_name: str,
        tool: object,
    ) -> bool:
        """Return whether one issued witness owns the exact named tool."""
        del registration, canonical_name, tool
        return False

    @staticmethod
    def capability_snapshot(
        capability: object,
        service: object,
    ) -> object | None:
        """Return the immutable snapshot for one active issued capability."""
        del capability, service
        return None
