from .types import assert_positive_int

from collections.abc import Iterable, Mapping
from heapq import nsmallest
from typing import Any

DEFAULT_OBSERVABILITY_KEY_LIMIT = 16
DEFAULT_OBSERVABILITY_KEY_LENGTH_LIMIT = 128


def observability_key_sample(
    mapping: Mapping[Any, object],
    *,
    limit: int = DEFAULT_OBSERVABILITY_KEY_LIMIT,
    key_length_limit: int = DEFAULT_OBSERVABILITY_KEY_LENGTH_LIMIT,
) -> tuple[list[str], bool]:
    assert isinstance(mapping, Mapping)
    assert_positive_int(limit, "limit")
    assert_positive_int(key_length_limit, "key_length_limit")
    key_label_truncated = False

    def labels() -> Iterable[str]:
        nonlocal key_label_truncated
        for key in mapping:
            label, truncated = _observability_key_label(key, key_length_limit)
            key_label_truncated = key_label_truncated or truncated
            yield label

    return (
        nsmallest(limit, labels()),
        len(mapping) > limit or key_label_truncated,
    )


def _observability_key_label(
    key: object,
    key_length_limit: int,
) -> tuple[str, bool]:
    label = str(key)
    if len(label) <= key_length_limit:
        return label, False
    if key_length_limit <= 3:
        return label[:key_length_limit], True
    return f"{label[: key_length_limit - 3]}...", True
