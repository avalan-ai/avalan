import typing
from typing import Callable, TypeVar

T = TypeVar("T", bound=Callable[..., object])

_typing_override = getattr(typing, "override", None)

if _typing_override is not None:
    _override = _typing_override
else:

    def _override(func: T) -> T:
        return func


override = _override
