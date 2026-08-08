"""Reject qualified synchronous or awaitable ambiguity."""

import typing

value: int | typing.Awaitable[int] = 1
