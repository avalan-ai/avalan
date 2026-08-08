"""Reject cross-domain operation enumeration substitution."""

from avalan.patch.domain import OperationType
from avalan.patch.parser import RawPatchInputKind

operation: OperationType = RawPatchInputKind.APPLY_JSON
