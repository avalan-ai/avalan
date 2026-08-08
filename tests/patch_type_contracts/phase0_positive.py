"""Positive strict-type anchor for the dormant patch contract."""

from typing import NewType, Protocol, assert_type

PatchRequestId = NewType("PatchRequestId", str)


class PatchScopeResolver(Protocol):
    """Resolve a frozen request identifier asynchronously."""

    async def resolve(self, identifier: PatchRequestId) -> PatchRequestId:
        """Return the same typed identifier after scope resolution."""


def verify_identity(identifier: PatchRequestId) -> None:
    """Assert the frozen public identity boundary remains typed."""
    assert_type(identifier, PatchRequestId)
