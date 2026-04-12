from dataclasses import asdict, is_dataclass
from decimal import Decimal
from json import dumps
from logging import Logger, getLogger
from typing import Any, Sequence, TypeVar
from uuid import UUID

T = TypeVar("T")


def _lf(items: Sequence[T | None]) -> list[T]:
    return [item for item in items if item]


def _j(sep: str, items: list[str], *args: str, empty: str = "") -> str:
    real_items = _lf(items + list(args))
    return sep.join(real_items) if real_items else empty


def logger_replace(logger: Logger, logger_names: list[str]) -> None:
    assert logger and logger_names
    for logger_name in logger_names:
        updated_logger = getLogger(logger_name)
        assert updated_logger
        updated_logger.handlers = []
        for handler in logger.handlers:
            updated_logger.addHandler(handler)
        updated_logger.setLevel(logger.level)
        updated_logger.propagate = logger.propagate


def to_json(item: Any) -> str:
    def _default(o: Any) -> str | dict[str, Any]:
        if is_dataclass(o):
            assert not isinstance(o, type)
            return asdict(o)
        elif isinstance(o, (Decimal, UUID)):
            return str(o)
        raise TypeError(
            f"Object of type {type(o).__name__} is not JSON serializable"
        )

    return dumps(item, default=_default)
