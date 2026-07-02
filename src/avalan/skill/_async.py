from asyncio import sleep, timeout
from collections.abc import Awaitable
from typing import TypeVar

_T = TypeVar("_T")

DEFAULT_SKILL_IO_TIMEOUT_SECONDS = 30.0


async def skill_cancellation_checkpoint() -> None:
    await sleep(0)


async def skill_bounded_await(
    awaitable: Awaitable[_T],
    *,
    timeout_seconds: float | None = DEFAULT_SKILL_IO_TIMEOUT_SECONDS,
) -> _T:
    assert timeout_seconds is None or isinstance(timeout_seconds, int | float)
    assert not isinstance(timeout_seconds, bool)
    if timeout_seconds is None:
        return await awaitable
    assert timeout_seconds > 0, "timeout_seconds must be positive"
    async with timeout(timeout_seconds):
        return await awaitable
