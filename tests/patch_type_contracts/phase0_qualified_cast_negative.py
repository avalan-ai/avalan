"""Reject qualified cast-based patch type erasure."""

import typing

value: str = typing.cast(str, "value")
