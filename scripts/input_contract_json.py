"""Decode input-contract JSON without accepting ambiguous values."""

from json import JSONDecodeError, loads
from pathlib import Path


class StrictJsonError(ValueError):
    """Report JSON that is ambiguous or outside the accepted grammar."""


class DuplicateJsonNameError(StrictJsonError):
    """Report a duplicate name in one JSON object."""


class NonFiniteJsonNumberError(StrictJsonError):
    """Report a non-finite JSON number."""


def strict_json_loads(source: str) -> object:
    """Return a strictly decoded JSON value."""
    assert isinstance(source, str)

    def object_from_pairs(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for name, item in pairs:
            if name in value:
                raise DuplicateJsonNameError(
                    f"duplicate JSON object name: {name!r}"
                )
            value[name] = item
        return value

    def reject_constant(constant: str) -> object:
        raise NonFiniteJsonNumberError(
            f"non-finite JSON number is prohibited: {constant}"
        )

    return loads(
        source,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )


def strict_json_path(path: Path) -> object:
    """Return a strictly decoded JSON file."""
    assert isinstance(path, Path)
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError, UnicodeError) as exc:
        raise StrictJsonError(f"cannot decode {path}: {exc}") from exc
