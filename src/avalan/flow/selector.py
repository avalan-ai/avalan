from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from re import Pattern, compile

_IDENTIFIER_PATTERN: Pattern[str] = compile(
    r"^([A-Za-z][A-Za-z0-9_-]*)(\[[0-9]+\])*$"
)
_INDEX_PATTERN: Pattern[str] = compile(r"\[([0-9]+)\]")
_RESERVED_ROOTS = frozenset(
    {
        "__task_files__",
        "__task_input__",
        "env",
        "environment",
        "file",
        "files",
        "fs",
        "network",
        "runtime",
        "secret",
        "secrets",
        "task",
    }
)
_UNSAFE_FRAGMENTS = frozenset(
    {
        "${",
        "$(",
        "/",
        ":",
        "\\",
        "{%",
        "{{",
        "%}",
        "}}",
        "~",
    }
)
FLOW_SELECTOR_MISSING = object()


class FlowSelectorRoot(StrEnum):
    FLOW_INPUT = "flow_input"
    NODE_OUTPUT = "node_output"


class FlowSelectorStepKind(StrEnum):
    FIELD = "field"
    INDEX = "index"


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowSelectorStep:
    kind: FlowSelectorStepKind
    value: str | int

    def __post_init__(self) -> None:
        assert isinstance(self.kind, FlowSelectorStepKind)
        if self.kind == FlowSelectorStepKind.FIELD:
            assert isinstance(self.value, str) and self.value.strip()
        else:
            assert self.kind == FlowSelectorStepKind.INDEX
            assert isinstance(self.value, int) and self.value >= 0


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowSelector:
    root: FlowSelectorRoot
    source: str
    output: str | None = None
    path: tuple[FlowSelectorStep, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.root, FlowSelectorRoot)
        assert isinstance(self.source, str) and self.source.strip()
        if self.output is not None:
            assert isinstance(self.output, str) and self.output.strip()
        assert isinstance(self.path, tuple)
        for step in self.path:
            assert isinstance(step, FlowSelectorStep)


class FlowSelectorError(ValueError):
    def __init__(self, code: str) -> None:
        assert isinstance(code, str) and code.strip()
        self.code = code
        super().__init__(code)


def parse_flow_selector(
    selector: str,
    *,
    allowed_roots: frozenset[FlowSelectorRoot] | None = None,
) -> FlowSelector:
    assert isinstance(selector, str)
    if allowed_roots is not None:
        assert isinstance(allowed_roots, frozenset)
        for root in allowed_roots:
            assert isinstance(root, FlowSelectorRoot)
    value = selector.strip()
    if not value:
        raise FlowSelectorError("flow.invalid_selector")
    _reject_unsafe_selector(value)
    raw_parts = value.split(".")
    if any(not part for part in raw_parts):
        raise FlowSelectorError("flow.invalid_selector")
    first_identifier, first_indexes = _parse_segment(raw_parts[0])
    if first_identifier in _RESERVED_ROOTS:
        raise FlowSelectorError("flow.reserved_selector")
    if first_indexes:
        raise FlowSelectorError("flow.invalid_selector")
    if first_identifier in {"input", "inputs"}:
        parsed = _parse_input_selector(raw_parts)
    else:
        parsed = _parse_node_output_selector(raw_parts)
    if allowed_roots is not None and parsed.root not in allowed_roots:
        raise FlowSelectorError("flow.invalid_selector")
    return parsed


def _parse_input_selector(raw_parts: list[str]) -> FlowSelector:
    if len(raw_parts) < 2:
        raise FlowSelectorError("flow.invalid_selector")
    source, path = _parse_selector_root(raw_parts[1], raw_parts[2:])
    return FlowSelector(
        root=FlowSelectorRoot.FLOW_INPUT,
        source=source,
        path=tuple(path),
    )


def _parse_node_output_selector(raw_parts: list[str]) -> FlowSelector:
    if len(raw_parts) < 2:
        raise FlowSelectorError("flow.invalid_selector")
    source, _ = _parse_segment(raw_parts[0])
    output, path = _parse_selector_root(raw_parts[1], raw_parts[2:])
    return FlowSelector(
        root=FlowSelectorRoot.NODE_OUTPUT,
        source=source,
        output=output,
        path=tuple(path),
    )


def _parse_selector_root(
    root_part: str,
    nested_parts: list[str],
) -> tuple[str, list[FlowSelectorStep]]:
    name, indexes = _parse_segment(root_part)
    path = [
        FlowSelectorStep(kind=FlowSelectorStepKind.INDEX, value=index)
        for index in indexes
    ]
    for part in nested_parts:
        field, field_indexes = _parse_segment(part)
        path.append(
            FlowSelectorStep(kind=FlowSelectorStepKind.FIELD, value=field)
        )
        path.extend(
            FlowSelectorStep(kind=FlowSelectorStepKind.INDEX, value=index)
            for index in field_indexes
        )
    return name, path


def _parse_segment(segment: str) -> tuple[str, tuple[int, ...]]:
    if segment.startswith("__") or segment.endswith("__"):
        raise FlowSelectorError("flow.reserved_selector")
    match = _IDENTIFIER_PATTERN.fullmatch(segment)
    if match is None:
        raise FlowSelectorError("flow.invalid_selector")
    name = match.group(1)
    indexes = tuple(int(index) for index in _INDEX_PATTERN.findall(segment))
    return name, indexes


def _reject_unsafe_selector(selector: str) -> None:
    if any(fragment in selector for fragment in _UNSAFE_FRAGMENTS):
        raise FlowSelectorError("flow.unsafe_selector")


def resolve_flow_selector_value(
    selector: FlowSelector,
    *,
    inputs: Mapping[str, object],
    node_outputs: Mapping[str, Mapping[str, object]],
    missing: object = FLOW_SELECTOR_MISSING,
) -> object:
    assert isinstance(selector, FlowSelector)
    assert isinstance(inputs, Mapping)
    assert isinstance(node_outputs, Mapping)
    if selector.root == FlowSelectorRoot.FLOW_INPUT:
        current = inputs.get(selector.source, missing)
    else:
        outputs = node_outputs.get(selector.source)
        if outputs is None or selector.output is None:
            return missing
        assert isinstance(outputs, Mapping)
        current = outputs.get(selector.output, missing)
    for step in selector.path:
        if current is missing:
            return missing
        if step.kind == FlowSelectorStepKind.FIELD:
            if not isinstance(current, Mapping):
                return missing
            current = current.get(step.value, missing)
            continue
        if not isinstance(current, list | tuple):
            return missing
        assert isinstance(step.value, int)
        if step.value >= len(current):
            return missing
        current = current[step.value]
    return current
