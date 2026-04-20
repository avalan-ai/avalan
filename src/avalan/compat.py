import typing
from typing import Callable, TypeVar, cast

T = TypeVar("T", bound=Callable[..., object])

_typing_override = getattr(typing, "override", None)

if _typing_override is not None:
    override = cast(Callable[[T], T], _typing_override)

else:

    def override(func: T) -> T:
        return func
