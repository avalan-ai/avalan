"""Provide strict JSON and canonical shape helpers for Phase 0 artifacts."""

from json import loads
from typing import Literal, TypeAlias

JsonPathComponent: TypeAlias = str | int
JsonPath: TypeAlias = tuple[JsonPathComponent, ...]
JsonMappingEntry: TypeAlias = tuple[str, JsonPath, tuple[str, ...]]
JsonTypedPathEntry: TypeAlias = tuple[Literal["key", "index"], str | int]
JsonTypedPath: TypeAlias = tuple[JsonTypedPathEntry, ...]


class StrictJsonError(ValueError):
    """Report JSON that violates strict Phase 0 decoding rules."""


class DuplicateJsonObjectNameError(StrictJsonError):
    """Report a duplicate name within one JSON object."""


class NonFiniteJsonNumberError(StrictJsonError):
    """Report a non-finite numeric constant in JSON input."""


def strict_json_loads(source: str) -> object:
    """Parse JSON while rejecting duplicate object names."""
    assert isinstance(source, str)

    def reject_duplicate_names(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise DuplicateJsonObjectNameError(
                    f"duplicate JSON object name: {name!r}"
                )
            result[name] = value
        return result

    def reject_non_finite_number(constant: str) -> object:
        raise NonFiniteJsonNumberError(
            f"non-finite JSON number is prohibited: {constant}"
        )

    return loads(
        source,
        object_pairs_hook=reject_duplicate_names,
        parse_constant=reject_non_finite_number,
    )


def canonical_json_pointer(path: JsonPath) -> str:
    """Return the canonical RFC 6901 pointer for one JSON path."""
    return "".join(
        "/" + str(component).replace("~", "~0").replace("/", "~1")
        for component in path
    )


def typed_json_path(path: JsonPath) -> JsonTypedPath:
    """Return an identity that distinguishes object keys and list indexes."""
    identity: list[JsonTypedPathEntry] = []
    for component in path:
        if isinstance(component, str):
            identity.append(("key", component))
        else:
            assert type(component) is int
            identity.append(("index", component))
    return tuple(identity)


def json_mapping_entries(
    value: object,
    path: JsonPath = (),
) -> tuple[JsonMappingEntry, ...]:
    """Return canonical paths and exact key sets for every JSON object."""
    entries: list[JsonMappingEntry] = []
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        entries.append(
            (
                canonical_json_pointer(path),
                path,
                tuple(sorted(value)),
            )
        )
        for key, child in value.items():
            assert isinstance(key, str)
            entries.extend(json_mapping_entries(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            entries.extend(json_mapping_entries(child, (*path, index)))
    ordered = tuple(sorted(entries, key=lambda entry: entry[0]))
    pointers = tuple(entry[0] for entry in ordered)
    identities = tuple(
        (entry[0], typed_json_path(entry[1])) for entry in ordered
    )
    assert len(pointers) == len(set(pointers))
    assert len(identities) == len(set(identities))
    return ordered
