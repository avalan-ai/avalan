from __future__ import annotations

from typing import Callable, TypeVar

try:
    from typing import override as _override
except ImportError:  # Python < 3.12
    T = TypeVar("T", bound=Callable[..., object])

    def _override(func: T) -> T:  # type: ignore
        return func


override = _override

__all__ = ["override"]
