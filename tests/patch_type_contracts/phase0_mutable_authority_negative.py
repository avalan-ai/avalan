"""Reject mutable approval authority at the patch strict boundary."""

from dataclasses import dataclass


@dataclass
class PatchApprovalGrant:
    """Keep one deliberately mutable authority object for the fixture."""

    identifier: str
