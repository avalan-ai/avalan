"""Prove every root-public input facade annotation is root-closed."""

from inspect import isclass, isfunction
from types import FunctionType
from typing import TypeVar, get_args, get_origin, get_type_hints

import avalan as avalan_root

_PUBLIC_SPECIAL_METHODS = frozenset(
    {
        "__aenter__",
        "__aexit__",
        "__call__",
        "__init__",
    }
)


def _facade_symbols() -> tuple[object, ...]:
    """Return every root export defined by the public SDK facade."""
    return tuple(
        value
        for name, value in vars(avalan_root).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == "avalan.sdk"
    )


def _annotated_targets(symbol: object) -> tuple[object, ...]:
    """Return one facade symbol and each annotated public method."""
    targets: list[object] = []
    if isclass(symbol) or isfunction(symbol):
        targets.append(symbol)
    if not isclass(symbol):
        return tuple(targets)
    for name, descriptor in vars(symbol).items():
        if name.startswith("_") and name not in _PUBLIC_SPECIAL_METHODS:
            continue
        member: object = descriptor
        if isinstance(descriptor, staticmethod | classmethod):
            member = descriptor.__func__
        if isinstance(member, FunctionType) and member.__annotations__:
            targets.append(member)
    return tuple(targets)


def _annotation_nodes(annotation: object) -> tuple[object, ...]:
    """Flatten one resolved annotation into its dependency nodes."""
    if isinstance(annotation, TypeVar):
        nodes: list[object] = []
        if annotation.__bound__ is not None:
            nodes.extend(_annotation_nodes(annotation.__bound__))
        for constraint in annotation.__constraints__:
            nodes.extend(_annotation_nodes(constraint))
        return tuple(nodes)
    origin = get_origin(annotation)
    nodes: list[object] = []
    if origin is None:
        nodes.append(annotation)
    elif origin is not annotation:
        nodes.extend(_annotation_nodes(origin))
    for argument in get_args(annotation):
        nodes.extend(_annotation_nodes(argument))
    return tuple(nodes)


def _is_root_export(value: object) -> bool:
    """Return whether the exact value is public at the Avalan root."""
    return any(
        not name.startswith("_") and candidate is value
        for name, candidate in vars(avalan_root).items()
    )


def _non_root_avalan_leaves(annotation: object) -> tuple[object, ...]:
    """Return recursively discovered Avalan leaves missing from the root."""
    missing: list[object] = []
    for dependency in _annotation_nodes(annotation):
        module = getattr(dependency, "__module__", "")
        name = getattr(dependency, "__name__", "")
        if (
            module.startswith("avalan.")
            and name
            and getattr(avalan_root, name, None) is not dependency
        ):
            missing.append(dependency)
    return tuple(missing)


def test_root_public_aliases_are_recursively_audited() -> None:
    """Audit root aliases recursively instead of treating them as leaves."""
    input_annotation = get_type_hints(avalan_root.run_agent)["input"]
    assert input_annotation is avalan_root.Input
    assert _is_root_export(input_annotation)

    dependencies = _annotation_nodes(input_annotation)
    assert (
        sum(dependency is avalan_root.Message for dependency in dependencies)
        == 2
    )
    assert _non_root_avalan_leaves(input_annotation) == ()


def test_root_public_facade_annotations_are_root_closed() -> None:
    """Resolve and root-match every Avalan dependency in facade annotations."""
    request_hints = get_type_hints(avalan_root.InputRequestView)
    question_annotation, ellipsis = get_args(request_hints["questions"])
    assert ellipsis is Ellipsis
    assert question_annotation is avalan_root.InputQuestion
    assert request_hints["state_revision"] is avalan_root.StateRevision

    facade_symbols = _facade_symbols()
    assert avalan_root.InputRequestView in facade_symbols
    assert avalan_root.run_agent in facade_symbols
    annotated_target_count = 0
    for symbol in facade_symbols:
        for target in _annotated_targets(symbol):
            hints = get_type_hints(target)
            annotated_target_count += bool(hints)
            for field, annotation in hints.items():
                missing = _non_root_avalan_leaves(annotation)
                assert missing == (), (
                    f"{target!r}.{field} leaks {missing!r} outside "
                    "the avalan root"
                )
    assert annotated_target_count > 0
