"""Reject stringly typed trust state at the patch strict boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetTrace:
    """Keep one deliberately stringly action field for the fixture."""

    action: str
