from collections.abc import Callable
from typing import TypeVar

from typing_extensions import override as _override

T = TypeVar("T", bound=Callable[..., object])

override = _override
