"""Define the closed canonical JSON encoding for sandbox patch plans."""

from collections.abc import Mapping
from json import dumps


def canonical_sandbox_plan_bytes(value: Mapping[str, object]) -> bytes:
    """Encode every sandbox plan field into one canonical byte sequence."""
    return dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
