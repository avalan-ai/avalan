"""Test dormant OpenAI Responses conversation provider contracts."""

from ast import (
    AST,
    Add,
    AnnAssign,
    Assert,
    Assign,
    AsyncFunctionDef,
    Attribute,
    AugAssign,
    BinOp,
    BoolOp,
    Call,
    ClassDef,
    Constant,
    Del,
    DictComp,
    ExceptHandler,
    FunctionDef,
    Global,
    Import,
    ImportFrom,
    Lambda,
    MatchAs,
    MatchMapping,
    MatchStar,
    Name,
    NamedExpr,
    Nonlocal,
    Or,
    Return,
    Store,
    Subscript,
    iter_child_nodes,
    parse,
    walk,
)
from ast import Dict as DictNode
from ast import (
    List as ListNode,
)
from asyncio import run
from dataclasses import fields
from hashlib import sha256
from importlib import import_module
from inspect import Parameter, signature
from json import dumps, loads
from pathlib import Path
from tomllib import load as load_toml
from types import UnionType
from typing import (
    Annotated,
    Callable,
    Literal,
    Required,
    Self,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import pytest
from openai import AsyncOpenAI
from openai import Timeout as OpenAITimeout
from openai import __version__ as openai_version
from openai.types.responses import (
    CompactedResponse,
    ResponseCompactionItem,
    ResponseCompactionItemParamParam,
    ResponseInputItemParam,
    ResponseOutputItem,
)
from openai.types.responses.response_create_params import ContextManagement
from openai.types.shared import Reasoning as ResponseReasoning
from openai.types.shared_params import Reasoning as RequestReasoning
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import BaseModel

from avalan.entities import GenerationSettings
from avalan.model.capability import ProviderCapabilitySupport
from avalan.model.nlp.text.vendor.openai import OpenAIClient
from avalan.server.entities import (
    DORMANT_CONVERSATION_REQUEST_FIELDS,
    ResponsesRequest,
)

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PROVIDER_CONTRACT_PATH = (
    _REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "conversation"
    / "provider_contract.json"
)
_PROVIDER_CONFORMANCE_PATH = (
    _REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "conversation"
    / "provider_conformance.json"
)
_PHASE14_PROVIDER_TRANSITION_PATH = (
    _REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "conversation"
    / "provider_transition.phase14.json"
)
_PHASE15_PROVIDER_TRANSITION_PATH = (
    _REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "conversation"
    / "provider_transition.phase15.json"
)
_ADAPTER_PATH = (
    _REPOSITORY_ROOT
    / "src"
    / "avalan"
    / "model"
    / "nlp"
    / "text"
    / "vendor"
    / "openai.py"
)
_PROVIDER_CONTRACT_CANONICAL_SHA256 = (
    "2c5e6e8fd1757bcf669ffdcb6e433b4ca5b35b64f5e26d31d6aa0900e918750f"
)
_PROVIDER_CONFORMANCE_CANONICAL_SHA256 = (
    "02d8f97d27572258cd4c8fcec7bc193500f7c2a293b025be63b75975dfaed3cc"
)
_PHASE0_PROVIDER_SOURCE_SHA256 = (
    "47d250ded5a4e0006fe3116ed51b9552f3a2b1caa313c73d77581e09e9ee5a0d"
)
_PRE_PHASE14_PROVIDER_SOURCE_SIZE = 363_543
_PRE_PHASE14_PROVIDER_SOURCE_SHA256 = (
    "ff4cc8edfc66009e72f9a511bc3035ad4f78354b0d254290db223a71a893ae1b"
)
_PRE_PHASE15_PROVIDER_SOURCE_SIZE = 392_855
_PRE_PHASE15_PROVIDER_SOURCE_SHA256 = (
    "262cefd693679d1dba6c236d631e7432e5490e623a9af135ecbe0a9f02356686"
)
_CANONICAL_JSON_ENCODING = (
    "utf-8 canonical JSON with sorted keys and compact separators"
)
_PHASE0_SOURCE_INTEGRITY_ENCODING = (
    "sha256 of exact UTF-8 provider module source bytes"
)
_PHASE0_SOURCE_INTEGRITY_COVERS = (
    "module_import_and_binding_topology",
    "_strict_replay_json_copy",
    "OpenAIClient.__call__",
    "OpenAIClient._reasoning_config",
)
_PRIMARY_OPENAI_SOURCE_PREFIXES = (
    "https://developers.openai.com/",
    "https://api.openai.com/",
)
_PROVIDER_PROFILE_SCHEMA_VERSION = "conversation-provider-profile-v1"
_PROTECTED_STATEFUL_CREATE_FIELDS = frozenset({"background", "store"})
_FORBIDDEN_PROVIDER_WIRE_ROOTS = frozenset(
    {
        "background",
        "compact_threshold",
        "context_management",
        "conversation",
        "extra_body",
        "previous_response_id",
        "store",
    }
)
_CLOSED_GATE_TRUSTED_HELPERS = frozenset({"_strict_replay_json_copy", "cast"})
_CLOSED_GATE_REFLECTION_NAMES = frozenset(
    {
        "eval",
        "exec",
        "globals",
        "locals",
        "vars",
    }
)
_FRAME_REFLECTION_ATTRIBUTES = frozenset(
    {
        "_getframe",
        "ag_frame",
        "cr_frame",
        "currentframe",
        "f_back",
        "f_globals",
        "f_locals",
        "gi_frame",
        "tb_frame",
    }
)
_REASONING_CONFIG_ALLOWED_KEYS = frozenset({"effort", "summary"})
_RESPONSE_LIFECYCLE_METHODS = frozenset(
    {
        "compact",
        "create",
        "delete",
        "retrieve",
    }
)
_TRACKED_REQUEST_MAPPINGS = frozenset(
    {
        "attempt_kwargs",
        "kwargs",
        "normalized_request_kwargs",
        "request_kwargs",
    }
)
_TRACKED_TRANSPORT_BINDINGS = _TRACKED_REQUEST_MAPPINGS | {"request_client"}


def _contains_reasoning_context(
    value: object,
    *,
    under_reasoning: bool = False,
) -> bool:
    if isinstance(value, dict):
        if under_reasoning and "context" in value:
            return True
        return any(
            _contains_reasoning_context(
                item,
                under_reasoning=under_reasoning or key == "reasoning",
            )
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(
            _contains_reasoning_context(
                item,
                under_reasoning=under_reasoning,
            )
            for item in value
        )
    return False


class _TransportProbeStop(Exception):
    """Stop a provider transport probe immediately after dispatch."""


class _ResponsesCreateSpy:
    """Record provider create kwargs without returning a response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.lifecycle_counts = {
            lifecycle_name: 0 for lifecycle_name in _RESPONSE_LIFECYCLE_METHODS
        }

    async def create(self, **kwargs: object) -> object:
        """Record one provider request and stop the transport probe."""
        provider_roots = _FORBIDDEN_PROVIDER_WIRE_ROOTS.intersection(kwargs)
        assert provider_roots == {"store"}
        assert type(kwargs["store"]) is bool and kwargs["store"] is False
        assert not _contains_reasoning_context(kwargs)
        self.lifecycle_counts["create"] += 1
        self.calls.append(dict(kwargs))
        raise _TransportProbeStop()

    async def compact(self, *_: object, **__: object) -> object:
        """Reject a Responses compact lifecycle call."""
        return await self._reject_lifecycle("compact")

    async def delete(self, *_: object, **__: object) -> object:
        """Reject a Responses delete lifecycle call."""
        return await self._reject_lifecycle("delete")

    async def retrieve(self, *_: object, **__: object) -> object:
        """Reject a Responses retrieve lifecycle call."""
        return await self._reject_lifecycle("retrieve")

    async def _reject_lifecycle(self, lifecycle_name: str) -> object:
        """Reject a lifecycle call other than the expected create."""
        self.lifecycle_counts[lifecycle_name] += 1
        raise AssertionError(
            f"unexpected Responses lifecycle call: {lifecycle_name}"
        )


class _ProviderTransportSpy:
    """Provide the narrow client surface used before a create dispatch."""

    def __init__(self) -> None:
        self.responses = _ResponsesCreateSpy()

    def with_options(self, **_: object) -> Self:
        """Return the same transport spy for request-local options."""
        return self


def _mapping(value: object) -> dict[str, object]:
    assert type(value) is dict
    assert all(type(key) is str for key in value)
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> list[object]:
    assert type(value) is list
    return value


def _strings(value: object) -> list[str]:
    values = _sequence(value)
    assert all(type(item) is str for item in values)
    return [str(item) for item in values]


def _load_json(path: Path) -> dict[str, object]:
    value: object = loads(path.read_text(encoding="utf-8"))
    return _mapping(value)


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        value: object = load_toml(stream)
    return _mapping(value)


def _canonical_digest(value: object) -> str:
    canonical = dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _phase0_provider_source_digest(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def _phase14_provider_source_sha256() -> str:
    transition = _load_json(_PHASE14_PROVIDER_TRANSITION_PATH)
    assert transition["phase"] == 14
    assert transition["kind"] == "reviewed_provider_source_transition"
    payload = dict(transition)
    digest = payload.pop("canonical_sha256")
    assert digest == _canonical_digest(payload)
    entries = [
        _mapping(value) for value in _sequence(transition["transitions"])
    ]
    source = next(
        item
        for item in entries
        if item["path"] == "src/avalan/model/nlp/text/vendor/openai.py"
    )
    assert source["from_size"] == _PRE_PHASE14_PROVIDER_SOURCE_SIZE
    assert source["from_sha256"] == _PRE_PHASE14_PROVIDER_SOURCE_SHA256
    target = source["to_sha256"]
    assert type(target) is str
    return target


def _phase15_provider_source_sha256() -> str:
    transition = _load_json(_PHASE15_PROVIDER_TRANSITION_PATH)
    assert transition["phase"] == 15
    assert transition["kind"] == "reviewed_provider_source_transition"
    payload = dict(transition)
    digest = payload.pop("canonical_sha256")
    assert digest == _canonical_digest(payload)
    entries = [
        _mapping(value) for value in _sequence(transition["transitions"])
    ]
    source = next(
        item
        for item in entries
        if item["path"] == "src/avalan/model/nlp/text/vendor/openai.py"
    )
    assert source["from_size"] == _PRE_PHASE15_PROVIDER_SOURCE_SIZE
    assert source["from_sha256"] == _PRE_PHASE15_PROVIDER_SOURCE_SHA256
    target = source["to_sha256"]
    assert type(target) is str
    return target


def _assert_phase0_provider_source_integrity(source: str) -> None:
    assert (
        _phase0_provider_source_digest(source)
        == _phase15_provider_source_sha256()
    ), (
        "Phase 0 provider source integrity drifted; rotate the immutable "
        "anchor only through a reviewed provider phase transition"
    )


def _literal_strings(annotation: object) -> list[str]:
    if get_origin(annotation) is Literal:
        values = get_args(annotation)
        assert all(type(value) is str for value in values)
        return [str(value) for value in values]
    output: list[str] = []
    for argument in get_args(annotation):
        for value in _literal_strings(argument):
            if value not in output:
                output.append(value)
    return output


def _type_symbol(value: object) -> str:
    if value is OpenAITimeout:
        return "openai.Timeout"
    if value is UnionType:
        return "typing.Union"
    module = getattr(value, "__module__", None)
    qualified_name = getattr(value, "__qualname__", None)
    assert type(module) is str
    assert type(qualified_name) is str
    return f"{module}.{qualified_name}"


def _annotation_contract(annotation: object) -> dict[str, object]:
    if isinstance(annotation, TypeVar):
        return {"kind": "type_variable", "name": annotation.__name__}
    origin = get_origin(annotation)
    if origin is Literal:
        return {
            "kind": "literal",
            "values": [
                {"type": _type_symbol(type(value)), "value": value}
                for value in get_args(annotation)
            ],
        }
    if origin is Annotated:
        base, *metadata = get_args(annotation)
        return {
            "kind": "annotated",
            "base": _annotation_contract(base),
            "metadata": [
                {"type": _type_symbol(type(value)), "value": repr(value)}
                for value in metadata
            ],
        }
    if origin is not None:
        return {
            "kind": "generic",
            "origin": _type_symbol(origin),
            "arguments": [
                _annotation_contract(value) for value in get_args(annotation)
            ],
        }
    if isinstance(annotation, type):
        return {"kind": "type", "symbol": _type_symbol(annotation)}
    return {
        "kind": "value",
        "type": _type_symbol(type(annotation)),
        "value": repr(annotation),
    }


def _annotation_sha256(annotation: object) -> str:
    return _canonical_digest(_annotation_contract(annotation))


def test_annotation_contract_normalizes_union_origins() -> None:
    """Normalize equivalent union spellings without changing arguments."""
    pep_604_contract = _annotation_contract(str | int)
    typing_annotation = Union[str, int]  # noqa: UP007
    typing_contract = _annotation_contract(typing_annotation)

    assert pep_604_contract == {
        "kind": "generic",
        "origin": "typing.Union",
        "arguments": [
            {"kind": "type", "symbol": "builtins.str"},
            {"kind": "type", "symbol": "builtins.int"},
        ],
    }
    assert pep_604_contract == typing_contract
    assert _annotation_sha256(str | int) == _annotation_sha256(
        typing_annotation
    )
    assert pep_604_contract != _annotation_contract(int | str)


def _default_contract(value: object) -> str:
    if value is Parameter.empty:
        return "required"
    if value is None:
        return "value:builtins.NoneType:null"
    if type(value) in {bool, float, int, str}:
        return f"value:{_type_symbol(type(value))}:{value!r}"
    return f"singleton:{_type_symbol(type(value))}"


def _method_contract(method: Callable[..., object]) -> dict[str, object]:
    method_signature = signature(method)
    hints = get_type_hints(method, include_extras=True)
    parameters = list(method_signature.parameters.values())
    parameter_kinds: dict[str, list[str]] = {}
    parameter_defaults: dict[str, list[str]] = {}
    for parameter in parameters:
        parameter_kinds.setdefault(parameter.kind.name, []).append(
            parameter.name
        )
        if parameter.default is not Parameter.empty:
            default = _default_contract(parameter.default)
            parameter_defaults.setdefault(default, []).append(parameter.name)
    return {
        "symbol": _type_symbol(method),
        "parameter_names": [parameter.name for parameter in parameters],
        "parameter_kinds": parameter_kinds,
        "required_parameters": [
            parameter.name
            for parameter in parameters
            if parameter.default is Parameter.empty
        ],
        "parameter_defaults": parameter_defaults,
        "resolved_parameter_annotations_sha256": {
            parameter.name: _annotation_sha256(hints[parameter.name])
            for parameter in parameters
        },
        "resolved_return_annotation_sha256": _annotation_sha256(
            hints["return"]
        ),
    }


def _parameter_contract(
    method: Callable[..., object],
    name: str,
) -> dict[str, object]:
    method_signature = signature(method)
    hints = get_type_hints(method, include_extras=True)
    parameter = method_signature.parameters[name]
    return {
        "sdk_parameter_kind": parameter.kind.name,
        "sdk_default_contract": _default_contract(parameter.default),
        "sdk_resolved_annotation_sha256": _annotation_sha256(hints[name]),
    }


def _typed_dict_contract(value: type[object]) -> dict[str, object]:
    hints = get_type_hints(value, include_extras=True)
    required_keys_value = getattr(value, "__required_keys__")
    assert isinstance(required_keys_value, frozenset)
    assert all(type(key) is str for key in required_keys_value)
    required_keys = {str(key) for key in required_keys_value}
    total = getattr(value, "__total__")
    assert type(total) is bool
    field_contracts: list[dict[str, object]] = []
    for name, annotation in hints.items():
        required = name in required_keys or get_origin(annotation) is Required
        field_contract = {
            "name": name,
            "required": required,
            "resolved_annotation_sha256": _annotation_sha256(annotation),
        }
        field_contracts.append(field_contract)
    return {
        "symbol": _type_symbol(value),
        "total": total,
        "fields": field_contracts,
    }


def _typed_dict_required_fields(value: type[object]) -> list[str]:
    contract = _typed_dict_contract(value)
    return sorted(
        str(field["name"])
        for field in (_mapping(item) for item in _sequence(contract["fields"]))
        if field["required"] is True
    )


def _model_contract(value: type[BaseModel]) -> dict[str, object]:
    return {
        "symbol": _type_symbol(value),
        "fields": [
            {
                "name": name,
                "required": field.is_required(),
                "resolved_annotation_sha256": _annotation_sha256(
                    field.annotation
                ),
            }
            for name, field in value.model_fields.items()
        ],
    }


def _resolve_symbol(path: str) -> object:
    parts = path.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        try:
            value: object = import_module(".".join(parts[:split_at]))
        except ModuleNotFoundError:
            continue
        for name in parts[split_at:]:
            value = getattr(value, name)
        return value
    raise AssertionError(f"cannot resolve SDK symbol: {path}")


def _input_item_wire_types() -> list[str]:
    wire_types: list[str] = []
    for value in get_args(ResponseInputItemParam):
        annotation = get_type_hints(value, include_extras=True).get("type")
        if annotation is None:
            continue
        for wire_type in _literal_strings(annotation):
            if wire_type not in wire_types:
                wire_types.append(wire_type)
    return wire_types


def _static_string(value: AST) -> str | None:
    if isinstance(value, Constant) and type(value.value) is str:
        return value.value
    if isinstance(value, BinOp) and isinstance(value.op, Add):
        left = _static_string(value.left)
        right = _static_string(value.right)
        if left is not None and right is not None:
            return left + right
    return None


def _assert_full_fixture_digest(
    payload: dict[str, object],
    external_anchor: str,
) -> None:
    digest = _mapping(payload["canonical_digest"])
    assert digest["algorithm"] == "sha256"
    assert digest["encoding"] == _CANONICAL_JSON_ENCODING
    expected_scope = [key for key in payload if key != "canonical_digest"]
    assert _strings(digest["scope"]) == expected_scope
    scoped = {field: payload[field] for field in expected_scope}
    assert digest["value"] == _canonical_digest(scoped)
    assert digest["value"] == external_anchor


def _attribute_chain(value: AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(value, Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, Name):
        parts.append(value.id)
    return tuple(reversed(parts))


def _openai_class_node(source: str) -> ClassDef:
    tree = parse(source, filename=str(_ADAPTER_PATH))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ClassDef) and node.name == "OpenAIClient"
    ]
    assert len(classes) == 1, "OpenAIClient must have one direct definition"
    return classes[0]


def _openai_call_node(source: str) -> AsyncFunctionDef:
    class_node = _openai_class_node(source)
    methods = [
        node
        for node in class_node.body
        if isinstance(node, AsyncFunctionDef) and node.name == "__call__"
    ]
    assert (
        len(methods) == 1
    ), "OpenAIClient must have one direct async __call__"
    return methods[0]


def _openai_reasoning_config_node(source: str) -> FunctionDef:
    class_node = _openai_class_node(source)
    methods = [
        node
        for node in class_node.body
        if isinstance(node, FunctionDef) and node.name == "_reasoning_config"
    ]
    assert (
        len(methods) == 1
    ), "OpenAIClient must have one direct _reasoning_config"
    return methods[0]


def _direct_assignments(
    function: FunctionDef | AsyncFunctionDef,
    name: str,
) -> list[tuple[Assign | AnnAssign, Name, AST]]:
    assignments: list[tuple[Assign | AnnAssign, Name, AST]] = []
    for syntax_node in walk(function):
        if isinstance(syntax_node, AnnAssign):
            if (
                isinstance(syntax_node.target, Name)
                and syntax_node.target.id == name
            ):
                assert (
                    syntax_node.value is not None
                ), f"{name} must have a value"
                assignments.append(
                    (
                        syntax_node,
                        syntax_node.target,
                        syntax_node.value,
                    )
                )
            continue
        if not isinstance(syntax_node, Assign):
            continue
        for target in syntax_node.targets:
            if isinstance(target, Name) and target.id == name:
                assignments.append((syntax_node, target, syntax_node.value))
    assignments.sort(key=lambda item: item[0].lineno)
    return assignments


def _single_assignment_value(
    function: FunctionDef | AsyncFunctionDef,
    name: str,
) -> AST:
    assignments = _direct_assignments(function, name)
    assert len(assignments) == 1, f"{name} must have exactly one assignment"
    return assignments[0][2]


def _direct_name(value: AST) -> str | None:
    return value.id if isinstance(value, Name) else None


def _strict_copy_source(value: AST) -> str | None:
    if not isinstance(value, Call):
        return None
    if _attribute_chain(value.func) != ("_strict_replay_json_copy",):
        return None
    if len(value.args) != 1 or value.keywords:
        return None
    return _direct_name(value.args[0])


def _cast_source(value: AST) -> str | None:
    if not isinstance(value, Call):
        return None
    if _attribute_chain(value.func) != ("cast",):
        return None
    if len(value.args) != 2 or value.keywords:
        return None
    return _direct_name(value.args[1])


def _subscript_root_name(value: AST) -> str | None:
    while isinstance(value, Subscript | Attribute):
        value = value.value
    return _direct_name(value)


def _assigned_names(node: Assign | AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, Name)]


def _binding_occurrences(
    function: FunctionDef | AsyncFunctionDef,
) -> list[tuple[str, AST, str]]:
    occurrences: list[tuple[str, AST, str]] = []
    for syntax_node in walk(function):
        if isinstance(syntax_node, Name) and isinstance(
            syntax_node.ctx, Store | Del
        ):
            occurrences.append(
                (
                    syntax_node.id,
                    syntax_node,
                    type(syntax_node.ctx).__name__,
                )
            )
        if isinstance(
            syntax_node,
            FunctionDef | AsyncFunctionDef | Lambda,
        ):
            arguments = syntax_node.args
            argument_nodes = [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ]
            if arguments.vararg is not None:
                argument_nodes.append(arguments.vararg)
            if arguments.kwarg is not None:
                argument_nodes.append(arguments.kwarg)
            occurrences.extend(
                (argument.arg, argument, "argument")
                for argument in argument_nodes
            )
        if (
            isinstance(
                syntax_node,
                FunctionDef | AsyncFunctionDef | ClassDef,
            )
            and syntax_node is not function
        ):
            occurrences.append((syntax_node.name, syntax_node, "definition"))
        if isinstance(syntax_node, ExceptHandler) and syntax_node.name:
            occurrences.append(
                (
                    syntax_node.name,
                    syntax_node,
                    "exception_handler",
                )
            )
        if isinstance(syntax_node, Import | ImportFrom):
            for import_name in syntax_node.names:
                binding_name = (
                    import_name.asname
                    or import_name.name.split(".", maxsplit=1)[0]
                )
                if binding_name != "*":
                    occurrences.append((binding_name, import_name, "import"))
        if isinstance(syntax_node, Global | Nonlocal):
            occurrences.extend(
                (name, syntax_node, type(syntax_node).__name__.lower())
                for name in syntax_node.names
            )
        if isinstance(syntax_node, MatchAs) and syntax_node.name is not None:
            occurrences.append((syntax_node.name, syntax_node, "match_as"))
        if isinstance(syntax_node, MatchStar) and syntax_node.name is not None:
            occurrences.append((syntax_node.name, syntax_node, "match_star"))
        if (
            isinstance(syntax_node, MatchMapping)
            and syntax_node.rest is not None
        ):
            occurrences.append((syntax_node.rest, syntax_node, "match_rest"))
    return occurrences


def _assert_closed_bindings(
    function: FunctionDef | AsyncFunctionDef,
    *,
    canonical_bindings: dict[str, set[AST]],
    forbidden_bindings: frozenset[str],
) -> None:
    for name, owner, binding_kind in _binding_occurrences(function):
        if name in canonical_bindings:
            assert (
                owner in canonical_bindings[name]
            ), f"{name} has noncanonical {binding_kind} binding"
            continue
        assert (
            name not in forbidden_bindings
        ), f"closed provider gate forbids {binding_kind} binding for {name}"


def _assert_no_closed_gate_reflection(
    function: FunctionDef | AsyncFunctionDef,
) -> None:
    for syntax_node in walk(function):
        assert _static_string(syntax_node) not in (
            _FRAME_REFLECTION_ATTRIBUTES
        ), "closed provider gate forbids static frame reflection names"
        if (
            isinstance(syntax_node, Name)
            and not isinstance(syntax_node.ctx, Store | Del)
            and syntax_node.id in _CLOSED_GATE_REFLECTION_NAMES
        ):
            raise AssertionError(
                f"closed provider gate forbids {syntax_node.id} reflection"
            )
        if (
            isinstance(syntax_node, Attribute)
            and syntax_node.attr in _FRAME_REFLECTION_ATTRIBUTES
        ):
            raise AssertionError(
                "closed provider gate forbids frame attribute reflection"
            )
        if not isinstance(syntax_node, Call):
            continue
        chain = _attribute_chain(syntax_node.func)
        if chain[-1:] not in {("getattr",), ("__getattribute__",)}:
            continue
        reflected_name = next(
            (
                value
                for value in (
                    _static_string(argument) for argument in syntax_node.args
                )
                if value in _FRAME_REFLECTION_ATTRIBUTES
            ),
            None,
        )
        assert (
            reflected_name is None
        ), "closed provider gate forbids dynamic frame reflection"


def _assert_openai_call_transport_policy(source: str) -> None:
    function = _openai_call_node(source)
    function_parameter_names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    assert _FORBIDDEN_PROVIDER_WIRE_ROOTS.isdisjoint(
        function_parameter_names
    ), "OpenAIClient.__call__ cannot expose protected provider field control"
    initial_mapping = _single_assignment_value(function, "kwargs")
    normalized_mapping = _single_assignment_value(
        function,
        "normalized_request_kwargs",
    )
    request_mapping = _single_assignment_value(function, "request_kwargs")
    attempt_mapping = _single_assignment_value(function, "attempt_kwargs")
    mapping_assignments = {
        name: _direct_assignments(function, name)
        for name in _TRACKED_REQUEST_MAPPINGS
    }
    assert all(
        len(assignments) == 1 for assignments in mapping_assignments.values()
    )
    request_client_assignments = _direct_assignments(
        function, "request_client"
    )
    assert (
        len(request_client_assignments) == 2
    ), "request_client must have exactly two canonical assignments"
    assert _attribute_chain(request_client_assignments[0][2]) == (
        "self",
        "_client",
    )
    request_client_options = request_client_assignments[1][2]
    assert isinstance(request_client_options, Call)
    assert _attribute_chain(request_client_options.func) == (
        "self",
        "_client",
        "with_options",
    )
    assert request_client_options.args == []
    assert [keyword.arg for keyword in request_client_options.keywords] == [
        "max_retries"
    ]
    canonical_bindings: dict[str, set[AST]] = {
        name: {assignments[0][1]}
        for name, assignments in mapping_assignments.items()
    }
    canonical_bindings["request_client"] = {
        assignment[1] for assignment in request_client_assignments
    }
    _assert_closed_bindings(
        function,
        canonical_bindings=canonical_bindings,
        forbidden_bindings=(
            _TRACKED_TRANSPORT_BINDINGS
            | _FORBIDDEN_PROVIDER_WIRE_ROOTS
            | _CLOSED_GATE_TRUSTED_HELPERS
            | _CLOSED_GATE_REFLECTION_NAMES
        ),
    )
    _assert_no_closed_gate_reflection(function)

    assert isinstance(
        initial_mapping, DictNode
    ), "kwargs must be the initial request mapping literal"
    assert all(
        key is not None for key in initial_mapping.keys
    ), "initial request mapping cannot unpack another mapping"
    initial_keys = [
        _static_string(key) for key in initial_mapping.keys if key is not None
    ]
    assert all(
        key is not None for key in initial_keys
    ), "initial request mapping keys must be static strings"
    store_positions = [
        index for index, key in enumerate(initial_keys) if key == "store"
    ]
    assert (
        len(store_positions) == 1
    ), "initial request mapping must write store exactly once"
    store_value = initial_mapping.values[store_positions[0]]
    assert (
        isinstance(store_value, Constant)
        and type(store_value.value) is bool
        and store_value.value is False
    ), "initial store value must be the AST literal False"
    assert (_FORBIDDEN_PROVIDER_WIRE_ROOTS - {"store"}).isdisjoint(
        initial_keys
    ), "initial request mapping contains a forbidden provider field"
    allowed_store_key = initial_mapping.keys[store_positions[0]]
    assert allowed_store_key is not None
    inline_compaction_assignments = [
        node
        for node in walk(function)
        if isinstance(node, Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], Subscript)
        and isinstance(node.targets[0].value, Name)
        and node.targets[0].value.id == "kwargs"
        and _static_string(node.targets[0].slice) == "context_management"
    ]
    assert len(inline_compaction_assignments) == 1
    inline_compaction_assignment = inline_compaction_assignments[0]
    inline_compaction_target = inline_compaction_assignment.targets[0]
    assert isinstance(inline_compaction_target, Subscript)
    inline_compaction_key = inline_compaction_target.slice
    inline_compaction_value = inline_compaction_assignment.value
    assert isinstance(inline_compaction_value, Call)
    assert isinstance(inline_compaction_value.func, Name)
    assert inline_compaction_value.func.id == "_strict_replay_json_copy"
    assert len(inline_compaction_value.args) == 1
    inline_compaction_payload = inline_compaction_value.args[0]
    assert isinstance(inline_compaction_payload, ListNode)
    assert len(inline_compaction_payload.elts) == 1
    inline_compaction_entry = inline_compaction_payload.elts[0]
    assert isinstance(inline_compaction_entry, DictNode)
    inline_compaction_keys = [
        _static_string(key) for key in inline_compaction_entry.keys
    ]
    assert inline_compaction_keys == ["type", "compact_threshold"]
    inline_compaction_type = inline_compaction_entry.values[0]
    assert isinstance(inline_compaction_type, Constant)
    assert inline_compaction_type.value == "compaction"
    assert isinstance(inline_compaction_entry.values[1], Name)
    assert (
        inline_compaction_entry.values[1].id == "inline_compaction_threshold"
    )
    allowed_compact_threshold_key = inline_compaction_entry.keys[1]
    assert allowed_compact_threshold_key is not None
    allowed_provider_wire_keys = {
        allowed_store_key: "store",
        inline_compaction_key: "context_management",
        allowed_compact_threshold_key: "compact_threshold",
    }
    for syntax_node in walk(function):
        static_value = _static_string(syntax_node)
        if static_value not in _FORBIDDEN_PROVIDER_WIRE_ROOTS:
            continue
        assert (
            allowed_provider_wire_keys.get(syntax_node) == static_value
        ), f"static provider wire root {static_value} is forbidden"

    assert (
        _strict_copy_source(normalized_mapping) == "kwargs"
    ), "normalized_request_kwargs must strictly copy kwargs"
    assert (
        _direct_name(request_mapping) == "normalized_request_kwargs"
    ), "request_kwargs must retain the validated normalized mapping"
    normalization_assertions = [
        node
        for node in walk(function)
        if isinstance(node, Assert)
        and isinstance(node.test, Call)
        and isinstance(node.test.func, Name)
        and node.test.func.id == "isinstance"
        and len(node.test.args) == 2
        and isinstance(node.test.args[0], Name)
        and node.test.args[0].id == "normalized_request_kwargs"
        and isinstance(node.test.args[1], Name)
        and node.test.args[1].id == "dict"
    ]
    assert (
        len(normalization_assertions) == 1
    ), "request_kwargs must follow one explicit normalized mapping check"
    assert (
        _strict_copy_source(attempt_mapping) == "request_kwargs"
    ), "attempt_kwargs must strictly copy request_kwargs"

    assignment_nodes = [
        node for node in walk(function) if isinstance(node, Assign | AnnAssign)
    ]
    allowed_copy_assignments = {
        ("attempt_kwargs", "request_kwargs"),
        ("normalized_request_kwargs", "kwargs"),
    }
    allowed_direct_assignments = {
        ("request_kwargs", "normalized_request_kwargs")
    }
    for assignment_node in assignment_nodes:
        value = assignment_node.value
        assert value is not None
        targets = _assigned_names(assignment_node)
        direct_source = _direct_name(value)
        if direct_source in _TRACKED_REQUEST_MAPPINGS:
            assert len(targets) == 1
            assert (
                targets[0],
                direct_source,
            ) in allowed_direct_assignments, (
                "tracked request mapping alias is prohibited"
            )
        if direct_source == "request_client":
            raise AssertionError("request client aliases are prohibited")
        assert _attribute_chain(value)[-1:] != (
            "responses",
        ), "Responses resource aliases are prohibited"
        copy_source = _strict_copy_source(value)
        if copy_source in _TRACKED_REQUEST_MAPPINGS:
            assert len(targets) == 1
            assert (
                targets[0],
                copy_source,
            ) in allowed_copy_assignments, (
                "tracked request mapping has an alternate strict-copy route"
            )
        cast_source = _cast_source(value)
        if cast_source in _TRACKED_REQUEST_MAPPINGS:
            raise AssertionError(
                "tracked request mapping cannot use a cast alias"
            )
        for target in targets:
            assert (
                target not in _FORBIDDEN_PROVIDER_WIRE_ROOTS
            ), f"local {target} control is prohibited"

    for syntax_node in walk(function):
        if isinstance(syntax_node, AugAssign):
            target_names = [
                value.id
                for value in walk(syntax_node.target)
                if isinstance(value, Name)
            ]
            assert _TRACKED_REQUEST_MAPPINGS.isdisjoint(
                target_names
            ), "tracked request mappings cannot be merged or augmented"
        if isinstance(syntax_node, NamedExpr) and isinstance(
            syntax_node.target, Name
        ):
            assert (
                syntax_node.target.id not in _TRACKED_REQUEST_MAPPINGS
            ), "tracked request mappings cannot use assignment expressions"

    assert not any(
        isinstance(node, DictComp) for node in walk(function)
    ), "closed provider gate forbids dynamic dictionary comprehensions"
    allowed_provider_wire_entries: list[tuple[DictNode, int]] = []
    for dict_node in walk(function):
        if not isinstance(dict_node, DictNode):
            continue
        for index, key in enumerate(dict_node.keys):
            assert (
                key is not None
            ), "closed provider gate forbids mapping-literal unpack routes"
            resolved_key = _static_string(key)
            assert (
                resolved_key is not None
            ), "closed provider gate requires static mapping-literal keys"
            if resolved_key not in _FORBIDDEN_PROVIDER_WIRE_ROOTS:
                continue
            value = dict_node.values[index]
            allowed_store = (
                dict_node is initial_mapping
                and index == store_positions[0]
                and resolved_key == "store"
                and isinstance(value, Constant)
                and type(value.value) is bool
                and value.value is False
            )
            allowed_inline_compaction = (
                dict_node is inline_compaction_entry
                and index == 1
                and resolved_key == "compact_threshold"
                and value is inline_compaction_entry.values[1]
            )
            assert (
                allowed_store or allowed_inline_compaction
            ), f"provider wire root {resolved_key} is forbidden"
            allowed_provider_wire_entries.append((dict_node, index))
    assert allowed_provider_wire_entries == [
        (initial_mapping, store_positions[0]),
        (inline_compaction_entry, 1),
    ], "only the exact store and inline-compaction mapping entries are allowed"

    for target_node in walk(function):
        if isinstance(target_node, Subscript):
            resolved_key = _static_string(target_node.slice)
            assert (
                resolved_key not in _FORBIDDEN_PROVIDER_WIRE_ROOTS
                or target_node is inline_compaction_target
            ), "protected provider fields cannot use subscript routes"
            root_name = _subscript_root_name(target_node.value)
            if root_name in _TRACKED_REQUEST_MAPPINGS and isinstance(
                target_node.ctx, Store | Del
            ):
                assert isinstance(target_node.ctx, Store)
                assert (
                    root_name == "kwargs"
                ), "normalized and attempt request mappings are immutable"
                assert (
                    resolved_key is not None
                ), "tracked request mappings reject dynamic subscript writes"
        if isinstance(target_node, Attribute):
            assert (
                target_node.attr not in _FORBIDDEN_PROVIDER_WIRE_ROOTS
            ), "protected provider fields cannot use attribute routes"

    calls = [node for node in walk(function) if isinstance(node, Call)]
    for call in calls:
        for keyword in call.keywords:
            assert (
                keyword.arg not in _FORBIDDEN_PROVIDER_WIRE_ROOTS
            ), "protected provider fields cannot use named keywords"
        if isinstance(call.func, Attribute):
            root_name = _subscript_root_name(call.func.value)
            if root_name in _TRACKED_REQUEST_MAPPINGS:
                raise AssertionError(
                    "tracked request mapping method calls are prohibited"
                )
        chain = _attribute_chain(call.func)
        if isinstance(call.func, Name):
            assert (
                call.func.id not in _RESPONSE_LIFECYCLE_METHODS
            ), "response lifecycle method aliases are prohibited"
        if chain == ("getattr",):
            getattr_name = (
                _static_string(call.args[1]) if len(call.args) >= 2 else None
            )
            assert len(call.args) < 2 or not (
                getattr_name is not None
                and getattr_name in _RESPONSE_LIFECYCLE_METHODS | {"responses"}
                or _attribute_chain(call.args[0])[-1:] == ("responses",)
            ), "response lifecycle getattr dispatch is prohibited"
        if chain[-1:] == ("__getattribute__",):
            assert chain[-2:-1] != (
                "responses",
            ), "response lifecycle __getattribute__ dispatch is prohibited"

    lifecycle_attributes = [
        node
        for node in walk(function)
        if isinstance(node, Attribute)
        and node.attr in _RESPONSE_LIFECYCLE_METHODS
    ]
    assert (
        len(lifecycle_attributes) == 1
    ), "exactly one direct Responses lifecycle attribute is allowed"
    create_calls = [
        node
        for node in calls
        if _attribute_chain(node.func)
        == ("request_client", "responses", "create")
    ]
    assert (
        len(create_calls) == 1
    ), "exactly one direct request_client.responses.create call is required"
    create_call = create_calls[0]
    assert lifecycle_attributes[0] is create_call.func
    assert (
        create_call.args == []
    ), "Responses create must use keyword arguments"
    assert (
        len(create_call.keywords) == 1
    ), "Responses create must have exactly one mapping unpack"
    create_unpack = create_call.keywords[0]
    assert (
        create_unpack.arg is None
    ), "Responses create must not have named provider arguments"
    assert (
        _direct_name(create_unpack.value) == "attempt_kwargs"
    ), "Responses create must unpack the validated attempt_kwargs mapping"

    parent_by_node = {
        child: parent
        for parent in walk(function)
        for child in iter_child_nodes(parent)
    }
    for request_client_node in walk(function):
        if (
            not isinstance(request_client_node, Name)
            or request_client_node.id != "request_client"
            or isinstance(request_client_node.ctx, Store | Del)
        ):
            continue
        responses_attribute = parent_by_node[request_client_node]
        assert (
            isinstance(responses_attribute, Attribute)
            and responses_attribute.attr == "responses"
        ), "request_client may only access the direct Responses resource"
        lifecycle_attribute = parent_by_node[responses_attribute]
        lifecycle_route_message = (
            "request_client Responses aliases and alternate calls "
            "are prohibited"
        )
    assert lifecycle_attribute is create_call.func, lifecycle_route_message
    allowed_tracked_uses = {
        "attempt_kwargs": {("isinstance", 0)},
        "kwargs": {("_strict_replay_json_copy", 0)},
        "normalized_request_kwargs": {("isinstance", 0)},
        "request_kwargs": {("_strict_replay_json_copy", 0)},
    }
    for use_node in walk(function):
        if (
            not isinstance(use_node, Name)
            or use_node.id not in _TRACKED_REQUEST_MAPPINGS
            or isinstance(use_node.ctx, Store | Del)
        ):
            continue
        parent = parent_by_node[use_node]
        if (
            isinstance(parent, Subscript)
            and isinstance(parent.ctx, Store)
            and use_node.id == "kwargs"
            and _subscript_root_name(parent.value) == "kwargs"
            and _static_string(parent.slice) is not None
        ):
            continue
        if (
            use_node.id == "normalized_request_kwargs"
            and parent is parent_by_node[request_mapping]
        ):
            continue
        if use_node.id == "attempt_kwargs" and parent is create_unpack:
            continue
        data_route_message = (
            f"tracked request mapping {use_node.id} has an "
            "unapproved data route"
        )
        assert isinstance(parent, Call), data_route_message
        call_name = _attribute_chain(parent.func)
        assert len(call_name) == 1
        argument_index = next(
            (
                index
                for index, argument in enumerate(parent.args)
                if argument is use_node
            ),
            -1,
        )
        call_route_message = (
            f"tracked request mapping {use_node.id} has an "
            "unapproved call route"
        )
        assert (call_name[0], argument_index) in allowed_tracked_uses[
            use_node.id
        ], call_route_message


def _assert_reasoning_config_policy(source: str) -> None:
    function = _openai_reasoning_config_node(source)
    reasoning_assignments = _direct_assignments(function, "reasoning")
    assert (
        len(reasoning_assignments) == 1
    ), "reasoning mapping must have exactly one direct assignment"
    reasoning_target = reasoning_assignments[0][1]
    reasoning_mapping = reasoning_assignments[0][2]
    assert isinstance(reasoning_mapping, DictNode)
    assert reasoning_mapping.keys == []
    assert reasoning_mapping.values == []
    _assert_closed_bindings(
        function,
        canonical_bindings={"reasoning": {reasoning_target}},
        forbidden_bindings=(
            frozenset({"context", "reasoning"}) | _CLOSED_GATE_REFLECTION_NAMES
        ),
    )
    _assert_no_closed_gate_reflection(function)
    assert not any(
        isinstance(node, DictComp) for node in walk(function)
    ), "reasoning config cannot build dynamic dictionaries"
    dict_nodes = [
        node for node in walk(function) if isinstance(node, DictNode)
    ]
    assert dict_nodes == [
        reasoning_mapping
    ], "reasoning config cannot build alternate mapping literals"

    reasoning_subscripts = [
        node
        for node in walk(function)
        if isinstance(node, Subscript)
        and _subscript_root_name(node.value) == "reasoning"
    ]
    reasoning_keys: list[str] = []
    for subscript in reasoning_subscripts:
        assert isinstance(
            subscript.ctx, Store
        ), "reasoning config only permits mapping writes"
        reasoning_key = _static_string(subscript.slice)
        assert (
            reasoning_key in _REASONING_CONFIG_ALLOWED_KEYS
        ), "reasoning config key must be static effort or summary"
        assert reasoning_key is not None
        reasoning_keys.append(reasoning_key)
    assert sorted(reasoning_keys) == ["effort", "summary"]

    for syntax_node in walk(function):
        if isinstance(syntax_node, Attribute):
            assert (
                syntax_node.attr != "context"
            ), "reasoning.context must remain dormant"
        if isinstance(syntax_node, Call):
            assert all(
                keyword.arg != "context" for keyword in syntax_node.keywords
            ), "reasoning.context named keywords remain dormant"

    return_nodes = [
        node for node in walk(function) if isinstance(node, Return)
    ]
    assert len(return_nodes) == 1
    return_value = return_nodes[0].value
    assert isinstance(return_value, BoolOp)
    assert isinstance(return_value.op, Or)
    assert len(return_value.values) == 2
    returned_reasoning = return_value.values[0]
    assert isinstance(returned_reasoning, Name)
    assert returned_reasoning.id == "reasoning"
    returned_none = return_value.values[1]
    assert isinstance(returned_none, Constant)
    assert returned_none.value is None

    parent_by_node = {
        child: parent
        for parent in walk(function)
        for child in iter_child_nodes(parent)
    }
    for use_node in walk(function):
        if (
            not isinstance(use_node, Name)
            or use_node.id != "reasoning"
            or isinstance(use_node.ctx, Store | Del)
        ):
            continue
        if use_node is returned_reasoning:
            continue
        parent = parent_by_node[use_node]
        assert (
            isinstance(parent, Subscript)
            and _subscript_root_name(parent.value) == "reasoning"
        ), "reasoning mapping aliases and mutator calls are prohibited"


def _assert_provider_adapter_transport_policy(source: str) -> None:
    _assert_openai_call_transport_policy(source)
    _assert_reasoning_config_policy(source)
    _assert_phase0_provider_source_integrity(source)


def _mutate_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, "mutation anchor must be unique"
    return source.replace(old, new, 1)


def test_provider_contract_snapshot_matches_typed_sdk_surface() -> None:
    contract = _load_json(_PROVIDER_CONTRACT_PATH)
    assert contract["schema_version"] == 1
    assert contract["contract_version"] == "conversation-provider-contract-v1"
    assert contract["feature"] == "conversation_continuity"
    assert contract["current_phase"] == 0
    assert contract["activation_state"] == "dormant"
    assert contract["retrieved_date"] == "2026-08-01"

    sources = [_mapping(value) for value in _sequence(contract["sources"])]
    assert {source["url"] for source in sources} == {
        "https://developers.openai.com/api/docs/guides/compaction",
        "https://developers.openai.com/api/docs/guides/conversation-state",
        "https://developers.openai.com/api/docs/guides/reasoning",
        "https://developers.openai.com/api/reference/resources/responses/methods/compact",
        "https://developers.openai.com/api/reference/resources/responses/methods/create",
        "https://developers.openai.com/api/reference/resources/responses/methods/delete",
        "https://developers.openai.com/api/reference/resources/responses/methods/retrieve",
        "https://api.openai.com/v1/responses",
        "https://api.openai.com/v1/responses/compact",
        "https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses",
    }
    for source in sources:
        assert source["retrieved_date"] == "2026-08-01"
        url = source["url"]
        assert type(url) is str
        if url.startswith(_PRIMARY_OPENAI_SOURCE_PREFIXES):
            assert str(source["retrieval_method"]).startswith(
                "developers_openai_mcp"
            )
    openapi_sources = [
        source for source in sources if source["authority"] == "OpenAI OpenAPI"
    ]
    assert openapi_sources
    assert all(
        source["document_version"] == "2.3.0" for source in openapi_sources
    )
    microsoft_source = next(
        source
        for source in sources
        if source["authority"] == "Microsoft Learn"
    )
    assert microsoft_source["document_last_updated"] == "2026-06-11"

    sdk_boundary = _mapping(contract["sdk_boundary"])
    constraint = str(sdk_boundary["required_constraint"])
    assert constraint == ">=2.42.0,<3.0.0"
    assert Version("2.42.0") in SpecifierSet(constraint)
    assert Version("2.41.1") not in SpecifierSet(constraint)
    assert sdk_boundary["minimum_typed_version"] == "2.42.0"
    assert sdk_boundary["last_version_missing_required_surface"] == "2.41.1"
    assert sdk_boundary["locked_version"] == "2.42.0"
    assert sdk_boundary["lock_selection"] == "minimum_typed_version"
    assert openai_version == sdk_boundary["locked_version"]

    project = _mapping(
        _load_toml(_REPOSITORY_ROOT / "pyproject.toml")["project"]
    )
    optional_dependencies = _mapping(project["optional-dependencies"])
    assert "openai>=2.42.0,<3.0.0" in _strings(
        optional_dependencies["vendors"]
    )
    lock = _load_toml(_REPOSITORY_ROOT / "poetry.lock")
    locked_packages = [_mapping(value) for value in _sequence(lock["package"])]
    locked_openai = next(
        package for package in locked_packages if package["name"] == "openai"
    )
    assert locked_openai["version"] == sdk_boundary["locked_version"]
    registry_evidence = _mapping(sdk_boundary["registry_evidence"])
    assert registry_evidence["authority"] == "Python Package Index"
    assert registry_evidence["retrieved_date"] == "2026-08-01"
    registry_artifacts = [
        _mapping(value) for value in _sequence(registry_evidence["artifacts"])
    ]
    assert all(
        artifact["verified_against_downloaded_bytes"] is True
        for artifact in registry_artifacts
    )
    registry_files = {
        str(artifact["filename"]): f"sha256:{artifact['sha256']}"
        for artifact in registry_artifacts
    }
    locked_files = {
        str(item["file"]): str(item["hash"])
        for item in (
            _mapping(value) for value in _sequence(locked_openai["files"])
        )
    }
    assert locked_files == registry_files

    client = AsyncOpenAI(api_key="phase-0-signature-inspection")
    try:
        responses = client.responses
        assert _type_symbol(type(responses)) == sdk_boundary["async_resource"]
        methods = _mapping(sdk_boundary["methods"])
        assert set(methods) == {
            "compact",
            "create",
            "delete",
            "retrieve",
        }
        for method_name, evidence_value in methods.items():
            evidence = _mapping(evidence_value)
            method = getattr(responses, method_name)
            assert evidence == _method_contract(method)
    finally:
        run(client.close())

    typed_symbols = _mapping(sdk_boundary["typed_symbols"])
    expected_typed_symbols = {
        "compacted_response": CompactedResponse,
        "compaction_input_item": ResponseCompactionItemParamParam,
        "compaction_output_item": ResponseCompactionItem,
        "context_management": ContextManagement,
        "input_item_union": ResponseInputItemParam,
        "output_item_union": ResponseOutputItem,
        "reasoning_request": RequestReasoning,
        "reasoning_response": ResponseReasoning,
    }
    assert set(typed_symbols) == set(expected_typed_symbols)
    for name, expected_symbol in expected_typed_symbols.items():
        symbol = typed_symbols[name]
        assert type(symbol) is str
        assert _resolve_symbol(symbol) == expected_symbol

    request_context = get_type_hints(
        RequestReasoning,
        include_extras=True,
    )["context"]
    response_context = ResponseReasoning.model_fields["context"].annotation
    reasoning_context = _mapping(sdk_boundary["reasoning_context"])
    assert _mapping(reasoning_context["request"]) == {
        "required": "context" in RequestReasoning.__required_keys__,
        "literal_values": _literal_strings(request_context),
        "resolved_annotation_sha256": _annotation_sha256(request_context),
    }
    assert _mapping(reasoning_context["response"]) == {
        "required": ResponseReasoning.model_fields["context"].is_required(),
        "literal_values": _literal_strings(response_context),
        "resolved_annotation_sha256": _annotation_sha256(response_context),
    }
    assert _strings(
        reasoning_context["documented_effective_response_values"]
    ) == ["current_turn", "all_turns"]

    context_management = _mapping(sdk_boundary["context_management"])
    assert _mapping(context_management["typed_contract"]) == (
        _typed_dict_contract(ContextManagement)
    )
    assert context_management["runtime_disposition"] == "deferred_fail_closed"
    assert type(context_management["typed_gap"]) is str
    assert context_management["typed_gap"]

    compaction_item = _mapping(sdk_boundary["compaction_item"])
    assert _mapping(compaction_item["input_contract"]) == (
        _typed_dict_contract(ResponseCompactionItemParamParam)
    )
    assert _mapping(compaction_item["output_contract"]) == _model_contract(
        ResponseCompactionItem
    )
    assert _mapping(compaction_item["response_contract"]) == _model_contract(
        CompactedResponse
    )
    compaction_parameter_hints = get_type_hints(
        ResponseCompactionItemParamParam,
        include_extras=True,
    )
    assert (
        compaction_item["wire_type"]
        == _literal_strings(compaction_parameter_hints["type"])[0]
    )
    assert _strings(compaction_item["input_required_fields"]) == (
        _typed_dict_required_fields(ResponseCompactionItemParamParam)
    )
    output_required_fields = [
        name
        for name, field in ResponseCompactionItem.model_fields.items()
        if field.is_required()
    ]
    output_required_fields.sort()
    assert _strings(compaction_item["output_required_fields"]) == (
        output_required_fields
    )
    assert (
        compaction_item["object_type"]
        == _literal_strings(
            CompactedResponse.model_fields["object"].annotation
        )[0]
    )
    assert compaction_item["response_output_required"] is (
        CompactedResponse.model_fields["output"].is_required()
    )
    assert compaction_item["output_union_resolved_annotation_sha256"] == (
        _annotation_sha256(ResponseOutputItem)
    )
    output_union = get_args(ResponseOutputItem)[0]
    assert ResponseCompactionItem in get_args(output_union)

    input_item_union = _mapping(sdk_boundary["locked_input_item_union"])
    assert input_item_union["runtime_disposition"] == "inventory_only_dormant"
    actual_input_symbols = [
        _type_symbol(value) for value in get_args(ResponseInputItemParam)
    ]
    assert actual_input_symbols == _strings(input_item_union["symbols"])
    assert _strings(input_item_union["wire_types"]) == (
        _input_item_wire_types()
    )
    assert input_item_union["resolved_annotation_sha256"] == (
        _annotation_sha256(ResponseInputItemParam)
    )


def test_provider_capability_profiles_are_exact_and_dormant() -> None:
    conformance = _load_json(_PROVIDER_CONFORMANCE_PATH)
    assert conformance["schema_version"] == 1
    assert (
        conformance["profile_schema_version"]
        == _PROVIDER_PROFILE_SCHEMA_VERSION
    )
    assert conformance["feature"] == "conversation_continuity"
    assert conformance["current_phase"] == 0
    assert conformance["activation_state"] == "dormant"
    assert conformance["production_dispatch_enabled"] is False
    assert conformance["production_advertisement_enabled"] is False

    capability_names = set(_strings(conformance["capability_names"]))
    identity_dimensions = set(_strings(conformance["identity_dimensions"]))
    allowed_states = set(_strings(conformance["capability_states"]))
    assert allowed_states == {"dormant", "incapable"}
    profiles = [
        _mapping(value) for value in _sequence(conformance["profiles"])
    ]
    profile_ids = [str(profile["profile_id"]) for profile in profiles]
    assert len(profile_ids) == len(set(profile_ids))
    assert {
        (
            _mapping(profile["binding"])["provider_family"],
            _mapping(profile["binding"])["transport"],
        )
        for profile in profiles
    } == {
        ("openai", "non_streaming"),
        ("openai", "streaming"),
        ("azure_openai", "non_streaming"),
        ("azure_openai", "streaming"),
        ("openai_compatible", "non_streaming"),
        ("openai_compatible", "streaming"),
    }

    for profile in profiles:
        state = profile["activation_state"]
        assert state in allowed_states
        assert state != "active"
        assert profile["identity_complete"] is False
        assert set(_mapping(profile["binding"])) == identity_dimensions
        capabilities = _mapping(profile["capabilities"])
        assert set(capabilities) == capability_names
        assert set(capabilities.values()) == {state}
        assert _sequence(profile["activation_evidence"]) == []
        evidence_nodes = _strings(profile["evidence_node_ids"])
        assert evidence_nodes
        assert all("::test_" in node for node in evidence_nodes)
        match profile["lifecycle"]:
            case "planned":
                assert state == "dormant"
                assert profile["active_from_phase"] == 12
            case "incapable":
                assert state == "incapable"
                assert profile["active_from_phase"] is None
            case _:
                pytest.fail("provider profile lifecycle must fail closed")


def test_identity_hints_never_activate_conversation_capabilities() -> None:
    conformance = _load_json(_PROVIDER_CONFORMANCE_PATH)
    policy = _mapping(conformance["inference_policy"])
    assert policy["default_state"] == "incapable"
    assert set(_strings(policy["rejected_inputs"])) == {
        "provider_name",
        "provider_family",
        "base_url",
        "endpoint_shape",
        "model_name",
        "sdk_method_presence",
        "sdk_type_presence",
        "schema_acceptance_without_round_trip",
        "one_live_success",
    }
    assert policy["profile_name_is_not_evidence"] is True
    assert policy["sdk_shape_is_not_evidence"] is True
    assert policy["url_shape_is_not_evidence"] is True

    cases = [
        _mapping(value)
        for value in _sequence(conformance["rejected_inference_cases"])
    ]
    assert cases
    rejected_case_inputs = {
        case["case_id"]: set(_strings(case["inputs"])) for case in cases
    }
    assert rejected_case_inputs == {
        "provider-name-and-native-url": {
            "provider_name",
            "base_url",
            "sdk_method_presence",
        },
        "azure-shaped-url-and-model-label": {
            "endpoint_shape",
            "model_name",
            "sdk_type_presence",
        },
        "generic-compatible-sdk-shape": {
            "provider_family",
            "base_url",
            "sdk_method_presence",
            "schema_acceptance_without_round_trip",
        },
        "single-deployment-success": {"one_live_success"},
    }
    assert all(
        case["expected_state"] in {"dormant", "incapable"} for case in cases
    )
    profiles = [
        _mapping(value) for value in _sequence(conformance["profiles"])
    ]
    assert not any(
        profile["activation_state"] == "active" for profile in profiles
    )


def test_no_production_conversation_dispatch_or_advertisement() -> None:
    contract = _load_json(_PROVIDER_CONTRACT_PATH)
    sdk_boundary = _mapping(contract["sdk_boundary"])
    policy = _mapping(sdk_boundary["conversation_state_transport_policy"])
    assert set(policy) == {
        "legacy_generic_request_kwargs_acknowledged",
        "legacy_generic_request_kwargs_description",
        "prohibited_routes",
        "provider_wire_paths",
        "public_request_fields",
        "reasoning_mapping_policy",
        "runtime_disposition",
        "scope",
        "stateful_create_field_policy",
    }
    assert policy["scope"] == "conversation_state_and_stateful_create_fields"
    assert policy["runtime_disposition"] == "dormant_fail_closed"
    assert policy["legacy_generic_request_kwargs_acknowledged"] is True
    assert type(policy["legacy_generic_request_kwargs_description"]) is str
    assert _strings(policy["prohibited_routes"]) == [
        "extra_body",
        "conversation_state_dict[str, Any]",
        "conversation_state_mapping_unpack",
        "untyped_generation_override",
        "caller_or_dynamic_store_control",
        "background_dispatch",
        "alternate_response_create_mapping_unpack",
        "responses_lifecycle_alias_or_getattr",
        "tracked_request_binding_rebind",
        "trusted_helper_shadow",
        "runtime_namespace_or_frame_reflection",
        "reasoning_mapping_alias_or_mutator",
        "phase0_provider_source_integrity_drift",
        "runtime_non_create_response_lifecycle",
    ]

    conformance = _load_json(_PROVIDER_CONFORMANCE_PATH)
    capability_names = set(_strings(conformance["capability_names"]))
    provider_capabilities = {
        value.name for value in fields(ProviderCapabilitySupport)
    }
    assert capability_names.isdisjoint(provider_capabilities)
    stateful_request_fields = set(_strings(policy["public_request_fields"]))
    assert stateful_request_fields.isdisjoint(
        GenerationSettings.__dataclass_fields__
    )
    assert stateful_request_fields & ResponsesRequest.model_fields.keys() == {
        "background",
        "context_management",
        "previous_response_id",
        "store",
    }
    assert stateful_request_fields <= DORMANT_CONVERSATION_REQUEST_FIELDS

    provider_wire_paths = _strings(policy["provider_wire_paths"])
    assert provider_wire_paths == [
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "context_management.compact_threshold",
        "reasoning.context",
        "store",
    ]
    assert _strings(policy["public_request_fields"]) == [
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "reasoning_context",
        "conversation_handle",
        "continuation_envelope",
        "store",
    ]
    reasoning_policy = _mapping(policy["reasoning_mapping_policy"])
    assert reasoning_policy == {
        "mapping_name": "reasoning",
        "allowed_static_keys": ["effort", "summary"],
        "forbidden_path": "reasoning.context",
        "dynamic_keys_allowed": False,
        "aliases_allowed": False,
        "mutator_calls_allowed": False,
    }

    create_policy = _mapping(policy["stateful_create_field_policy"])
    assert set(create_policy) == {
        "closed_ast_gate",
        "forbidden_provider_wire_roots",
        "legacy_fixed_provider_values",
        "provider_mapping_flow",
        "public_runtime_disposition",
        "typed_sdk_create_fields",
    }
    assert create_policy["public_runtime_disposition"] == "dormant_fail_closed"
    assert _mapping(create_policy["legacy_fixed_provider_values"]) == {
        "store": False
    }
    assert set(_strings(create_policy["forbidden_provider_wire_roots"])) == (
        _FORBIDDEN_PROVIDER_WIRE_ROOTS
    )
    mapping_flow = _mapping(create_policy["provider_mapping_flow"])
    assert mapping_flow == {
        "initial_request_mapping": "kwargs",
        "normalization_temporary": "normalized_request_kwargs",
        "normalized_request_mapping": "request_kwargs",
        "attempt_request_mapping": "attempt_kwargs",
        "copy_function": "_strict_replay_json_copy",
        "create_target": "request_client.responses.create",
        "create_unpack_source": "attempt_kwargs",
        "create_call_count": 1,
        "mapping_unpack_count": 1,
    }
    closed_gate = _mapping(create_policy["closed_ast_gate"])
    assert set(closed_gate) == {
        "forbidden_frame_attributes",
        "forbidden_reflection_names",
        "phase0_source_integrity",
        "tracked_bindings",
        "trusted_helpers",
    }
    assert set(_strings(closed_gate["tracked_bindings"])) == (
        _TRACKED_TRANSPORT_BINDINGS
    )
    assert set(_strings(closed_gate["trusted_helpers"])) == (
        _CLOSED_GATE_TRUSTED_HELPERS
    )
    assert set(_strings(closed_gate["forbidden_reflection_names"])) == (
        _CLOSED_GATE_REFLECTION_NAMES
    )
    assert set(_strings(closed_gate["forbidden_frame_attributes"])) == (
        _FRAME_REFLECTION_ATTRIBUTES
    )
    source_integrity = _mapping(closed_gate["phase0_source_integrity"])
    assert source_integrity == {
        "phase": 0,
        "kind": "exact_source_sha256",
        "algorithm": "sha256",
        "encoding": _PHASE0_SOURCE_INTEGRITY_ENCODING,
        "source_path": "src/avalan/model/nlp/text/vendor/openai.py",
        "covers": list(_PHASE0_SOURCE_INTEGRITY_COVERS),
        "rotation_policy": "reviewed_provider_phase_transition_only",
        "value": _PHASE0_PROVIDER_SOURCE_SHA256,
    }
    adapter_source = _ADAPTER_PATH.read_text(encoding="utf-8")
    assert sha256(_ADAPTER_PATH.read_bytes()).hexdigest() == (
        _phase15_provider_source_sha256()
    )
    assert _phase0_provider_source_digest(adapter_source) == (
        _phase15_provider_source_sha256()
    )

    typed_fields = _mapping(create_policy["typed_sdk_create_fields"])
    assert set(typed_fields) == _PROTECTED_STATEFUL_CREATE_FIELDS
    sdk_client = AsyncOpenAI(api_key="phase-0-create-field-inspection")
    try:
        create_method = sdk_client.responses.create
        for field_name in sorted(_PROTECTED_STATEFUL_CREATE_FIELDS):
            field_policy = _mapping(typed_fields[field_name])
            sdk_contract = _parameter_contract(create_method, field_name)
            assert {
                key: field_policy[key] for key in sdk_contract
            } == sdk_contract
            assert (
                field_policy["public_runtime_disposition"]
                == "dormant_fail_closed"
            )
        background_policy = _mapping(typed_fields["background"])
        assert background_policy == {
            **_parameter_contract(create_method, "background"),
            "provider_runtime_disposition": "prohibited",
            "allowed_provider_write_count": 0,
            "public_runtime_disposition": "dormant_fail_closed",
        }
        store_policy = _mapping(typed_fields["store"])
        assert store_policy == {
            **_parameter_contract(create_method, "store"),
            "provider_runtime_disposition": "legacy_fixed_false_only",
            "allowed_provider_write_count": 1,
            "allowed_provider_value": False,
            "public_runtime_disposition": "dormant_fail_closed",
        }
    finally:
        run(sdk_client.close())

    for field_name in sorted(_PROTECTED_STATEFUL_CREATE_FIELDS):
        assert field_name in DORMANT_CONVERSATION_REQUEST_FIELDS
        assert field_name not in GenerationSettings.__dataclass_fields__
        assert field_name not in signature(GenerationSettings).parameters
        assert field_name in ResponsesRequest.model_fields
        request = ResponsesRequest.model_validate(
            {
                "input": "phase-0",
                field_name: False,
            }
        )
        assert getattr(request, field_name) is False

    _assert_provider_adapter_transport_policy(adapter_source)

    for source_path in (_REPOSITORY_ROOT / "src").rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "provider_contract.json" not in source
        assert "provider_conformance.json" not in source


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            '                "store": False,\n',
            '                "store": settings.use_cache,\n',
            id="dynamic-initial-store-value",
        ),
        pytest.param(
            '                "store": False,\n',
            '                "store": False,\n'
            '                "background": False,\n',
            id="background-initial-mapping",
        ),
        pytest.param(
            '                "store": False,\n',
            '                "store": False,\n'
            '                "previous_response_id": "resp_unsafe",\n',
            id="previous-response-id-initial-mapping",
        ),
        pytest.param(
            '                "store": False,\n',
            '                "store": False,\n'
            '                "extra_body": {\n'
            '                    "previous_response_id": "resp_unsafe"\n'
            "                },\n",
            id="extra-body-carrying-previous-response-id",
        ),
        pytest.param(
            '            reasoning["summary"] = summary.value\n',
            '            reasoning["summary"] = summary.value\n'
            '            reasoning["context"] = "all_turns"\n',
            id="reasoning-context-write",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            stateful_key = "store"\n'
            "            alternate_mapping = {stateful_key: True}\n"
            "            kwargs, alternate_mapping = (\n"
            "                alternate_mapping, kwargs\n"
            "            )\n"
            "            request_client = self._client\n",
            id="dynamic-alternate-mapping-tuple-rebind",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            request_alias = locals()["kwargs"]\n'
            '            request_alias["background"] = True\n'
            "            request_client = self._client\n",
            id="locals-alias-background",
        ),
        pytest.param(
            "            normalized_request_kwargs = "
            "_strict_replay_json_copy(kwargs)\n",
            "            def _strict_replay_json_copy(\n"
            "                value: object,\n"
            "            ) -> object:\n"
            '                return {"store": True}\n'
            "\n"
            "            normalized_request_kwargs = "
            "_strict_replay_json_copy(kwargs)\n",
            id="shadowed-strict-copy-helper-store-injection",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            cast = untrusted_cast\n"
            "            request_client = self._client\n",
            id="shadowed-cast-helper",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            kwargs["store"] = False\n'
            "            request_client = self._client\n",
            id="literal-subscript-store",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            kwargs["st" + "ore"] = False\n'
            "            request_client = self._client\n",
            id="concatenated-subscript-store",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            request_alias = kwargs\n"
            '            request_alias["store"] = False\n'
            "            request_client = self._client\n",
            id="aliased-subscript-store",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            field_name = "store"\n'
            "            kwargs[field_name] = False\n"
            "            request_client = self._client\n",
            id="dynamic-subscript-store",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            kwargs.update({"store": False})\n'
            "            request_client = self._client\n",
            id="mapping-update-store",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            kwargs.setdefault("store", False)\n'
            "            request_client = self._client\n",
            id="mapping-setdefault-store",
        ),
        pytest.param(
            "            request_client = self._client\n",
            '            kwargs |= {"store": False}\n'
            "            request_client = self._client\n",
            id="mapping-union-store",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            mutate_request(kwargs)\n"
            "            request_client = self._client\n",
            id="dynamic-mapping-mutation-helper",
        ),
        pytest.param(
            "normalized_request_kwargs = _strict_replay_json_copy(kwargs)",
            "normalized_request_kwargs = _strict_replay_json_copy("
            '{**kwargs, "store": False})',
            id="normalization-mapping-merge",
        ),
        pytest.param(
            "attempt_kwargs = _strict_replay_json_copy(request_kwargs)",
            "attempt_kwargs = request_kwargs",
            id="attempt-mapping-alias",
        ),
        pytest.param(
            "                    assert isinstance(attempt_kwargs, dict)\n",
            "                    assert isinstance(attempt_kwargs, dict)\n"
            '                    attempt_kwargs.update({"store": False})\n',
            id="attempt-mapping-update",
        ),
        pytest.param(
            "                                **attempt_kwargs\n",
            "                                store=False,\n"
            "                                **attempt_kwargs\n",
            id="named-store-keyword",
        ),
        pytest.param(
            "                                **attempt_kwargs\n",
            "                                background=False,\n"
            "                                **attempt_kwargs\n",
            id="named-background-keyword",
        ),
        pytest.param(
            "                                **attempt_kwargs\n",
            "                                **attempt_kwargs,\n"
            "                                **{},\n",
            id="extra-create-mapping-unpack",
        ),
        pytest.param(
            "                                **attempt_kwargs\n",
            "                                **kwargs\n",
            id="alternate-create-unpack-source",
        ),
        pytest.param(
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                        create = request_client.responses."
            "create\n"
            "                        created_response = (\n"
            "                            await create(\n",
            id="aliased-create-method",
        ),
        pytest.param(
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                        created_response = (\n"
            "                            await getattr(\n"
            "                                request_client.responses, "
            '"create"\n'
            "                            )(\n",
            id="getattr-create-method",
        ),
        pytest.param(
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            '                        lifecycle_method = "create"\n'
            "                        created_response = (\n"
            "                            await getattr(\n"
            "                                request_client.responses, "
            "lifecycle_method\n"
            "                            )(\n",
            id="dynamic-getattr-create-method",
        ),
        pytest.param(
            "                    try:\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                    try:\n"
            "                        compact = request_client.responses."
            "compact\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            id="aliased-compact-method",
        ),
        pytest.param(
            "                    try:\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                    try:\n"
            "                        retrieve = "
            "request_client.responses.retrieve\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            id="aliased-retrieve-method",
        ),
        pytest.param(
            "                    try:\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                    try:\n"
            "                        delete = request_client.responses."
            "delete\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            id="aliased-delete-method",
        ),
        pytest.param(
            "                    try:\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                    try:\n"
            "                        responses_alias = request_client."
            "responses\n"
            '                        lifecycle_method = "create"\n'
            "                        getattr(\n"
            "                            responses_alias, lifecycle_method\n"
            "                        )\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            id="aliased-responses-resource",
        ),
        pytest.param(
            "                    try:\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                    try:\n"
            "                        request_client_alias = request_client\n"
            "                        getattr(\n"
            "                            request_client_alias.responses, "
            '"create"\n'
            "                        )\n"
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            id="aliased-request-client",
        ),
        pytest.param(
            "                        created_response = (\n"
            "                            await request_client.responses."
            "create(\n",
            "                        created_response = (\n"
            "                            await request_client.responses."
            "__getattribute__(\n"
            '                                "create"\n'
            "                            )(\n",
            id="dunder-getattribute-create-method",
        ),
    ],
)
def test_provider_transport_policy_rejects_data_flow_bypasses(
    old: str,
    new: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    mutated_source = _mutate_once(source, old, new)
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    "field_name",
    sorted(_FORBIDDEN_PROVIDER_WIRE_ROOTS),
)
@pytest.mark.parametrize(
    "route_name",
    ["attribute", "dict", "named_keyword", "subscript"],
)
def test_every_forbidden_wire_root_is_rejected_on_every_route(
    field_name: str,
    route_name: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    match route_name:
        case "attribute":
            injection = f"            forbidden_source.{field_name}\n"
        case "dict":
            injection = (
                f'            forbidden_mapping = {{"{field_name}": None}}\n'
            )
        case "named_keyword":
            injection = f"            forbidden_probe({field_name}=None)\n"
        case "subscript":
            injection = f'            kwargs["{field_name}"] = None\n'
        case _:
            raise AssertionError(f"unknown route: {route_name}")
    mutated_source = _mutate_once(
        source,
        "            request_client = self._client\n",
        injection + "            request_client = self._client\n",
    )
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    "binding_name",
    sorted(_TRACKED_TRANSPORT_BINDINGS),
)
def test_every_tracked_transport_binding_rejects_tuple_rebinding(
    binding_name: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    injection = f"            {binding_name}, alias = values\n"
    mutated_source = _mutate_once(
        source,
        "            request_client = self._client\n",
        injection + "            request_client = self._client\n",
    )
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    ("anchor", "injection"),
    [
        pytest.param(
            "            request_client = self._client\n",
            "            [kwargs, *aliases] = values\n",
            id="list-starred-target",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            for kwargs in values:\n                pass\n",
            id="for-target",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            shadow = [None for kwargs in values]\n",
            id="comprehension-target",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            (kwargs := {})\n",
            id="assignment-expression-target",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            with manager() as kwargs:\n                pass\n",
            id="with-target",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            match value:\n"
            '                case {"kwargs": kwargs}:\n'
            "                    pass\n",
            id="match-as-capture",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            match value:\n"
            "                case [head, *kwargs]:\n"
            "                    pass\n",
            id="match-star-capture",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            match value:\n"
            "                case {**kwargs}:\n"
            "                    pass\n",
            id="match-mapping-rest-capture",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            try:\n"
            "                pass\n"
            "            except Exception as kwargs:\n"
            "                pass\n",
            id="except-handler-name",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            import json as kwargs\n",
            id="import-alias",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            def kwargs() -> None:\n                pass\n",
            id="nested-function-name",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            class kwargs:\n                pass\n",
            id="nested-class-name",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            def binding_probe(kwargs: object) -> None:\n"
            "                pass\n",
            id="nested-function-argument",
        ),
        pytest.param(
            "            kwargs: dict[str, LooseJsonValue] = {\n",
            "            global kwargs\n",
            id="global-name",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            def binding_probe() -> None:\n"
            "                nonlocal kwargs\n",
            id="nonlocal-name",
        ),
        pytest.param(
            "            request_client = self._client\n",
            "            del kwargs\n",
            id="delete-target",
        ),
    ],
)
def test_tracked_request_mapping_rejects_every_binding_form(
    anchor: str,
    injection: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    mutated_source = _mutate_once(source, anchor, injection + anchor)
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    "injection",
    [
        pytest.param(
            "            reflected_namespace = locals()\n",
            id="locals",
        ),
        pytest.param(
            "            reflected_namespace = globals()\n",
            id="globals",
        ),
        pytest.param(
            "            reflected_namespace = vars()\n",
            id="vars",
        ),
        pytest.param(
            '            reflected_value = eval("None")\n',
            id="eval",
        ),
        pytest.param(
            '            exec("pass")\n',
            id="exec",
        ),
        pytest.param(
            "            frame = system._getframe()\n",
            id="getframe-attribute",
        ),
        pytest.param(
            "            frame = inspect.currentframe()\n",
            id="currentframe-attribute",
        ),
        pytest.param(
            "            reflected_namespace = frame.f_locals\n",
            id="frame-locals-attribute",
        ),
        pytest.param(
            '            reflected_namespace = getattr(frame, "f_globals")\n',
            id="getattr-frame-globals",
        ),
        pytest.param(
            '            frame_name = "f_" + "locals"\n'
            "            reflected_namespace = getattr(frame, frame_name)\n",
            id="indirect-static-frame-locals",
        ),
        pytest.param(
            "            reflected_namespace = frame.__getattribute__(\n"
            '                "f_back"\n'
            "            )\n",
            id="dunder-getattribute-frame-back",
        ),
    ],
)
def test_provider_transport_policy_rejects_namespace_and_frame_reflection(
    injection: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    mutated_source = _mutate_once(
        source,
        "            request_client = self._client\n",
        injection + "            request_client = self._client\n",
    )
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    "injection",
    [
        pytest.param(
            "            def _strict_replay_json_copy(\n"
            "                value: object,\n"
            "            ) -> object:\n"
            "                return value\n"
            "\n",
            id="strict-copy-helper",
        ),
        pytest.param(
            "            cast = untrusted_cast\n",
            id="cast-helper",
        ),
    ],
)
def test_provider_transport_policy_rejects_trusted_helper_shadowing(
    injection: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    mutated_source = _mutate_once(
        source,
        "            request_client = self._client\n",
        injection + "            request_client = self._client\n",
    )
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    "injection",
    [
        pytest.param(
            '        reasoning["context"] = "all_turns"\n',
            id="static-context-key",
        ),
        pytest.param(
            '        reasoning_key = "context"\n'
            '        reasoning[reasoning_key] = "all_turns"\n',
            id="dynamic-key",
        ),
        pytest.param(
            "        reasoning_alias = reasoning\n",
            id="mapping-alias",
        ),
        pytest.param(
            '        reasoning.update({"effort": "high"})\n',
            id="mapping-mutator",
        ),
        pytest.param(
            '        alternate_reasoning = {"effort": "high"}\n',
            id="alternate-mapping",
        ),
        pytest.param(
            "        reasoning.context\n",
            id="context-attribute",
        ),
        pytest.param(
            '        reasoning_probe(context="all_turns")\n',
            id="context-named-keyword",
        ),
    ],
)
def test_reasoning_config_rejects_nonstatic_or_context_routes(
    injection: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    mutated_source = _mutate_once(
        source,
        "        return reasoning or None\n",
        injection + "        return reasoning or None\n",
    )
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    ("replacements", "suffix"),
    [
        pytest.param(
            (
                (
                    "from asyncio import (\n",
                    (
                        "import builtins as runtime_builtins\n\n"
                        "from asyncio import (\n"
                    ),
                ),
                (
                    "            request_client = self._client\n",
                    (
                        "            reflected_namespace = "
                        "runtime_builtins.locals()\n"
                        '            runtime_builtins.eval("None")\n'
                        "            request_client = self._client\n"
                    ),
                ),
            ),
            "",
            id="qualified-builtins-locals-eval",
        ),
        pytest.param(
            (
                (
                    "            request_client = self._client\n",
                    (
                        "            reflected_namespace = "
                        '__builtins__["locals"]()\n'
                        "            request_client = self._client\n"
                    ),
                ),
            ),
            "",
            id="builtins-subscript-locals",
        ),
        pytest.param(
            (
                (
                    "            request_client = self._client\n",
                    (
                        '            frame_prefix = "f"\n'
                        '            frame_suffix = "locals"\n'
                        '            frame_name = "_".join(\n'
                        "                (frame_prefix, frame_suffix)\n"
                        "            )\n"
                        "            reflected_namespace = getattr(\n"
                        "                frame, frame_name\n"
                        "            )\n"
                        "            request_client = self._client\n"
                    ),
                ),
            ),
            "",
            id="dynamically-spelled-frame-attribute",
        ),
        pytest.param(
            (
                (
                    (
                        "            async def stream_factory() "
                        "-> AsyncIterator[object]:\n"
                    ),
                    (
                        "            recovered_alias = next(\n               "
                        " cell.cell_contents\n                for cell in"
                        " (create_response.__closure__ or ())\n               "
                        " if isinstance(cell.cell_contents, dict)\n           "
                        ' )\n            recovered_field = "".join(\n         '
                        '       ("back", "ground")\n            )\n           '
                        " recovered_alias[recovered_field] = True\n\n         "
                        "   async def stream_factory() ->"
                        " AsyncIterator[object]:\n"
                    ),
                ),
            ),
            "",
            id="closure-cell-contents-alias-recovery",
        ),
        pytest.param(
            (
                (
                    (
                        "                    try:\n"
                        "                        created_response = (\n"
                        "                            await "
                        "request_client.responses.create(\n"
                    ),
                    (
                        "                    alternate_client = self._client\n"
                        '                    responses_name = "".join(\n'
                        '                        ("res", "ponses")\n'
                        "                    )\n"
                        "                    responses_resource = getattr(\n"
                        "                        alternate_client, "
                        "responses_name\n"
                        "                    )\n"
                        '                    compact_name = "".join(\n'
                        '                        ("com", "pact")\n'
                        "                    )\n"
                        "                    await getattr(\n"
                        "                        responses_resource, "
                        "compact_name\n"
                        "                    )()\n"
                        "                    try:\n"
                        "                        created_response = (\n"
                        "                            await "
                        "request_client.responses.create(\n"
                    ),
                ),
            ),
            "",
            id="dynamic-responses-compact-alongside-create",
        ),
        pytest.param(
            (),
            "\n\ndef cast(type_: object, value: object) -> object:\n"
            "    return value\n",
            id="module-scope-cast-redefinition",
        ),
        pytest.param(
            (),
            "\n\ndef _strict_replay_json_copy(value: object) -> object:\n"
            '    return {"store": True}\n',
            id="module-scope-strict-copy-redefinition",
        ),
    ],
)
def test_phase0_source_integrity_rejects_reviewed_semantic_bypasses(
    replacements: tuple[tuple[str, str], ...],
    suffix: str,
) -> None:
    source = _ADAPTER_PATH.read_text(encoding="utf-8")
    mutated_source = source
    for old, new in replacements:
        mutated_source = _mutate_once(mutated_source, old, new)
    mutated_source += suffix
    assert mutated_source != source
    _assert_openai_call_transport_policy(mutated_source)
    _assert_reasoning_config_policy(mutated_source)
    with pytest.raises(AssertionError, match="source integrity drifted"):
        _assert_phase0_provider_source_integrity(mutated_source)
    with pytest.raises(AssertionError):
        _assert_provider_adapter_transport_policy(mutated_source)


@pytest.mark.parametrize(
    "field_name",
    sorted(_FORBIDDEN_PROVIDER_WIRE_ROOTS),
)
def test_runtime_transport_spy_rejects_every_forbidden_root(
    field_name: str,
) -> None:
    responses = _ResponsesCreateSpy()
    request: dict[str, object] = {"store": False}
    request[field_name] = True
    with pytest.raises(AssertionError):
        run(responses.create(**request))
    assert responses.calls == []
    assert set(responses.lifecycle_counts) == _RESPONSE_LIFECYCLE_METHODS
    assert not any(responses.lifecycle_counts.values())


@pytest.mark.parametrize("store_value", [None, 0, 1, True, "false"])
def test_runtime_transport_spy_rejects_missing_or_nonliteral_false_store(
    store_value: object,
) -> None:
    responses = _ResponsesCreateSpy()
    request = {} if store_value is None else {"store": store_value}
    with pytest.raises(AssertionError):
        run(responses.create(**request))
    assert responses.calls == []
    assert not any(responses.lifecycle_counts.values())


@pytest.mark.parametrize(
    "request_payload",
    [
        pytest.param(
            {"reasoning": {"context": "all_turns"}},
            id="direct",
        ),
        pytest.param(
            {
                "wrapper": [
                    {"reasoning": {"context": "all_turns"}},
                ]
            },
            id="recursive",
        ),
    ],
)
def test_runtime_transport_spy_recursively_rejects_reasoning_context(
    request_payload: dict[str, object],
) -> None:
    responses = _ResponsesCreateSpy()
    request = {"store": False, **request_payload}
    with pytest.raises(AssertionError):
        run(responses.create(**request))
    assert responses.calls == []
    assert not any(responses.lifecycle_counts.values())


@pytest.mark.parametrize(
    "lifecycle_name",
    sorted(_RESPONSE_LIFECYCLE_METHODS - {"create"}),
)
def test_runtime_transport_spy_rejects_every_non_create_lifecycle(
    lifecycle_name: str,
) -> None:
    responses = _ResponsesCreateSpy()
    lifecycle_call = getattr(responses, lifecycle_name)
    with pytest.raises(
        AssertionError,
        match=f"unexpected Responses lifecycle call: {lifecycle_name}",
    ):
        run(lifecycle_call())
    expected_counts = {
        name: int(name == lifecycle_name)
        for name in _RESPONSE_LIFECYCLE_METHODS
    }
    assert responses.lifecycle_counts == expected_counts
    assert responses.calls == []


@pytest.mark.parametrize(
    ("base_url", "use_async_generator"),
    [
        pytest.param(None, False, id="openai-non-streaming"),
        pytest.param(None, True, id="openai-streaming"),
        pytest.param(
            "https://phase0.openai.azure.com/openai/v1/",
            False,
            id="azure-non-streaming",
        ),
        pytest.param(
            "https://phase0.openai.azure.com/openai/v1/",
            True,
            id="azure-streaming",
        ),
    ],
)
def test_legacy_transport_always_sends_exact_fixed_store_false(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str | None,
    use_async_generator: bool,
) -> None:
    transport = _ProviderTransportSpy()

    def create_client(**_: object) -> _ProviderTransportSpy:
        return transport

    monkeypatch.setattr("openai.AsyncOpenAI", create_client)
    client = OpenAIClient(api_key="phase-0", base_url=base_url)

    with pytest.raises(_TransportProbeStop):
        run(
            client(
                "phase-0-model",
                [],
                use_async_generator=use_async_generator,
            )
        )

    assert len(transport.responses.calls) == 1
    request = transport.responses.calls[0]
    assert _FORBIDDEN_PROVIDER_WIRE_ROOTS.intersection(request) == {"store"}
    assert type(request["store"]) is bool
    assert request["store"] is False
    assert not _contains_reasoning_context(request)
    assert transport.responses.lifecycle_counts == {
        "compact": 0,
        "create": 1,
        "delete": 0,
        "retrieve": 0,
    }
    assert request["stream"] is use_async_generator


def test_provider_fixture_digests_are_canonical() -> None:
    contract = _load_json(_PROVIDER_CONTRACT_PATH)
    _assert_full_fixture_digest(
        contract,
        _PROVIDER_CONTRACT_CANONICAL_SHA256,
    )

    conformance = _load_json(_PROVIDER_CONFORMANCE_PATH)
    _assert_full_fixture_digest(
        conformance,
        _PROVIDER_CONFORMANCE_CANONICAL_SHA256,
    )
