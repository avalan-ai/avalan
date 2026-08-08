"""Assert closed raw-ingress and parser typing boundaries."""

from typing import assert_type

from avalan.patch.domain import OperationType
from avalan.patch.parser import (
    CanonicalPatchRequest,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
)


def assert_parser_types(
    ingress: RawPatchIngress,
    kind: RawPatchInputKind,
) -> None:
    """Assert raw input remains closed before parser projection."""
    result = PatchRequestParser(PatchInputLimits()).parse(ingress)
    assert_type(result, CanonicalPatchRequest)
    assert_type(kind, RawPatchInputKind)
    assert_type(result.operation, OperationType)
