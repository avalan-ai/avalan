from ast import (
    AST,
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    Attribute,
    BinOp,
    BitOr,
    Call,
    ClassDef,
    Constant,
    Dict,
    FunctionDef,
    If,
    Import,
    ImportFrom,
    Name,
    NodeVisitor,
    Raise,
    Subscript,
    Tuple,
    iter_child_nodes,
    parse,
)
from asyncio import (
    CancelledError,
    create_task,
    run,
    sleep,
    wait_for,
)
from asyncio import (
    Event as AsyncEvent,
)
from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from unittest import TestCase
from unittest.mock import patch

from avalan.entities import (
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.model.provider import ProviderFamily
from avalan.model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    LocalTextStreamEventParser,
    StreamBackpressurePolicy,
    StreamCancellationDrainPolicy,
    StreamCancellationPropagation,
    StreamCancellationPropagationTarget,
    StreamChannel,
    StreamConsumerProjection,
    StreamGoldenTrace,
    StreamItemCorrelation,
    StreamItemKind,
    StreamLegacyBoundaryCategory,
    StreamLegacyBoundaryDirection,
    StreamLegacyClassifierInventoryEntry,
    StreamLegacyInventoryScope,
    StreamLegacyRuntimeBoundaryInventoryEntry,
    StreamLegacySurface,
    StreamLegacySurfaceClassification,
    StreamLegacySurfaceInventoryEntry,
    StreamPerformanceBudget,
    StreamPerformanceBudgetReconciliation,
    StreamProducerBackend,
    StreamProjectionState,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamRetentionPolicy,
    StreamRuntimeContract,
    StreamSessionLifecycle,
    StreamTerminalOutcome,
    StreamToolLifecycleContract,
    StreamToolObservation,
    StreamValidationError,
    StreamVisibility,
    TextGenerationSingleStream,
    TextGenerationStream,
    accumulate_canonical_stream_items,
    assemble_tool_observations,
    canonical_item_from_consumer_projection,
    classify_legacy_stream_classifier,
    classify_legacy_stream_runtime_boundary,
    classify_legacy_stream_surface,
    is_stream_terminal_kind,
    is_tool_execution_terminal_kind,
    iter_stream_consumer_projections,
    legacy_stream_classifier_inventory,
    legacy_stream_runtime_boundary_inventory,
    legacy_stream_surface_inventory,
    normalize_provider_stream,
    project_canonical_stream_item,
    project_stream_consumer_item,
    stream_channel_for_kind,
    stream_observability_payload,
    stream_projection_display_token,
    stream_projection_is_reasoning,
    stream_projection_is_tool_call,
    stream_projection_text_delta,
    stream_terminal_outcome_for_kind,
    stream_token_metadata,
    validate_canonical_stream_items,
    validate_stream_runtime_contract,
    validate_tool_lifecycle_items,
)

STREAM_TEST_TIMEOUT = 1.0


def _item(
    kind: StreamItemKind,
    sequence: int,
    *,
    stream_session_id: str = "stream-1",
    run_id: str = "run-1",
    turn_id: str = "turn-1",
    channel: StreamChannel | None = None,
    correlation: StreamItemCorrelation | None = None,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    metadata: dict[str, object] | None = None,
    provider_payload: object | None = None,
    provider_family: str | None = None,
    provider_event_type: str | None = None,
    timestamp: datetime | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=kind,
        channel=channel or stream_channel_for_kind(kind),
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        terminal_outcome=terminal_outcome,
        visibility=visibility,
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
        provider_payload=provider_payload,  # type: ignore[arg-type]
        provider_family=provider_family,
        provider_event_type=provider_event_type,
        timestamp=timestamp,
    )


def _stream_completed(sequence: int) -> CanonicalStreamItem:
    return _item(
        StreamItemKind.STREAM_COMPLETED,
        sequence,
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )


def _stream_errored(sequence: int) -> CanonicalStreamItem:
    return _item(
        StreamItemKind.STREAM_ERRORED,
        sequence,
        terminal_outcome=StreamTerminalOutcome.ERRORED,
    )


def _tool_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    tool_call_id: str = "tool-1",
    text_delta: str | None = None,
    data: object | None = None,
) -> CanonicalStreamItem:
    return _item(
        kind,
        sequence,
        correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        text_delta=text_delta,
        data=data,
    )


def _first_sequence(
    items: tuple[CanonicalStreamItem, ...],
    kind: StreamItemKind,
) -> int:
    return next(item.sequence for item in items if item.kind is kind)


def _last_sequence(
    items: tuple[CanonicalStreamItem, ...],
    kind: StreamItemKind,
) -> int:
    return next(item.sequence for item in reversed(items) if item.kind is kind)


class _StreamProbe(TextGenerationStream):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> AsyncIterator[CanonicalStreamItem]:
        return TextGenerationStream.__call__(self, *args, **kwargs)

    async def __anext__(self) -> CanonicalStreamItem:
        return await TextGenerationStream.__anext__(self)


async def _single_canonical_generator() -> AsyncIterator[CanonicalStreamItem]:
    yield _item(StreamItemKind.STREAM_STARTED, 0)


async def _provider_events(
    events: tuple[StreamProviderEvent, ...],
) -> AsyncIterator[StreamProviderEvent]:
    for event in events:
        yield event


async def _collect_provider_items(
    events: AsyncIterable[StreamProviderEvent],
    *,
    provider_family: str | None = "openai",
    capabilities: StreamProviderCapabilities | None = None,
    close_after_terminal: bool = True,
) -> tuple[CanonicalStreamItem, ...]:
    return tuple(
        [
            item
            async for item in normalize_provider_stream(
                events,
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                provider_family=provider_family,
                capabilities=capabilities,
                close_after_terminal=close_after_terminal,
            )
        ]
    )


async def _local_text_chunks(
    chunks: tuple[str, ...],
) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def _local_text_events(
    chunks: AsyncIterable[str],
) -> AsyncIterator[StreamProviderEvent]:
    parser = LocalTextStreamEventParser()
    iterator = chunks.__aiter__()
    chunks_exhausted = False

    try:
        while True:
            try:
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                chunks_exhausted = True
                break
            for event in parser.push(chunk):
                yield event
        for event in parser.flush():
            yield event
    finally:
        if not chunks_exhausted:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()


def _local_capabilities(
    provider_family: str | None,
    capabilities: StreamProviderCapabilities | None,
) -> StreamProviderCapabilities:
    if capabilities is not None:
        assert capabilities.backend is StreamProducerBackend.LOCAL
        return capabilities
    return StreamProviderCapabilities(
        backend=StreamProducerBackend.LOCAL,
        provider_family=provider_family,
        supports_reasoning=True,
        supports_tool_calls=True,
        supports_cancellation=True,
        max_queue_depth=StreamPerformanceBudget().max_queue_depth,
    )


def _local_text_stream(
    chunks: AsyncIterable[str],
    *,
    stream_session_id: str = "local-stream",
    run_id: str = "local-run",
    turn_id: str = "local-turn",
    provider_family: str | None = "transformers",
    capabilities: StreamProviderCapabilities | None = None,
    close_after_terminal: bool = True,
) -> AsyncIterator[CanonicalStreamItem]:
    iterator = chunks.__aiter__()
    events_exhausted = False

    async def tracked_events() -> AsyncIterator[StreamProviderEvent]:
        nonlocal events_exhausted
        async for event in _local_text_events(iterator):
            yield event
        events_exhausted = True

    stream = normalize_provider_stream(
        tracked_events(),
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        provider_family=provider_family,
        capabilities=_local_capabilities(provider_family, capabilities),
        close_after_terminal=close_after_terminal,
    )

    async def closeable_stream() -> AsyncIterator[CanonicalStreamItem]:
        try:
            async for item in stream:
                yield item
        finally:
            stream_aclose = getattr(stream, "aclose", None)
            if stream_aclose is not None:
                await stream_aclose()
            if not events_exhausted:
                chunk_aclose = getattr(iterator, "aclose", None)
                if chunk_aclose is not None:
                    await chunk_aclose()

    return closeable_stream()


async def _collect_local_items(
    chunks: AsyncIterable[str],
    *,
    provider_family: str | None = "transformers",
    capabilities: StreamProviderCapabilities | None = None,
    close_after_terminal: bool = True,
) -> tuple[CanonicalStreamItem, ...]:
    return tuple(
        [
            item
            async for item in _local_text_stream(
                chunks,
                provider_family=provider_family,
                capabilities=capabilities,
                close_after_terminal=close_after_terminal,
            )
        ]
    )


async def _collect_stream_items(
    items: AsyncIterable[CanonicalStreamItem],
) -> tuple[CanonicalStreamItem, ...]:
    return tuple([item async for item in items])


async def _collect_projection_items(
    items: AsyncIterable[CanonicalStreamItem],
    *,
    validate_order: bool = True,
) -> tuple[StreamConsumerProjection, ...]:
    return tuple(
        [
            item
            async for item in iter_stream_consumer_projections(
                items,
                validate_order=validate_order,
            )
        ]
    )


_LEGACY_CLASSIFIER_SYMBOL_SURFACES = {
    "Token": StreamLegacySurface.TOKEN,
    "TokenDetail": StreamLegacySurface.TOKEN_DETAIL,
    "ReasoningToken": StreamLegacySurface.REASONING_TOKEN,
    "ToolCallToken": StreamLegacySurface.TOOL_CALL_TOKEN,
    "Event": StreamLegacySurface.EVENT,
}
_LEGACY_CLASSIFIER_EVENT_IMPORT_MODULES = {
    "avalan.event",
    "event",
}

_LEGACY_CLASSIFIER_STRING_SITES = {
    (
        "avalan.model.nlp.text.vendor.anthropic",
        "AnthropicClient._non_stream_response_content",
    ),
    (
        "avalan.model.nlp.text.vendor.openai",
        "OpenAIClient._non_stream_response_content",
    ),
    ("avalan.model.stream", "_LegacyTokenStreamAdapter.item_from_token"),
    ("avalan.model.stream", "_LegacyTokenStreamAdapter.events_from_token"),
}
_LEGACY_CLASSIFIER_STRING_CONTEXT_MARKERS = (
    "stream_",
    "_stream",
    "project_stream",
)

_PRODUCTION_LEGACY_BOUNDARY_CATEGORIES = {
    StreamLegacyBoundaryCategory.PRODUCER,
    StreamLegacyBoundaryCategory.SDK_RESPONSE,
    StreamLegacyBoundaryCategory.ORCHESTRATOR,
    StreamLegacyBoundaryCategory.PARSER,
    StreamLegacyBoundaryCategory.EVENTING,
    StreamLegacyBoundaryCategory.CLI_STDOUT,
    StreamLegacyBoundaryCategory.CHAT_SSE,
    StreamLegacyBoundaryCategory.RESPONSES_SSE,
    StreamLegacyBoundaryCategory.MCP,
    StreamLegacyBoundaryCategory.A2A,
    StreamLegacyBoundaryCategory.FLOW,
    StreamLegacyBoundaryCategory.HELPER_ONLY,
}

_TEMPORARY_LEGACY_SURFACE_CLASSIFICATIONS = {
    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM,
    StreamLegacySurfaceClassification.TEMPORARY_COMPATIBILITY_SHIM,
}

_PHASE_1_1_LEGACY_CLASSIFIER_DEBT_CEILING = {
    ("avalan.cli.commands.agent", "agent_run._event_listener"): frozenset(
        {StreamLegacySurface.EVENT}
    ),
    (
        "avalan.model.nlp.text.vendor.anthropic",
        "AnthropicClient._non_stream_response_content",
    ): frozenset({StreamLegacySurface.STRING}),
    (
        "avalan.model.nlp.text.vendor.openai",
        "OpenAIClient._non_stream_response_content",
    ): frozenset({StreamLegacySurface.STRING}),
    (
        "avalan.model.stream",
        "_LegacyTokenStreamAdapter.events_from_token",
    ): frozenset({StreamLegacySurface.STRING}),
    (
        "avalan.model.stream",
        "_LegacyTokenStreamAdapter.item_from_token",
    ): frozenset(
        {
            StreamLegacySurface.STRING,
            StreamLegacySurface.TOKEN,
            StreamLegacySurface.TOKEN_DETAIL,
            StreamLegacySurface.REASONING_TOKEN,
            StreamLegacySurface.TOOL_CALL_TOKEN,
        }
    ),
    ("avalan.task.event", "_raw_event_payload"): frozenset(
        {StreamLegacySurface.EVENT}
    ),
    ("avalan.task.event", "_raw_event_type"): frozenset(
        {StreamLegacySurface.EVENT}
    ),
    (
        "avalan.task.targets.flow",
        "_flow_node_event_listener.observe",
    ): frozenset({StreamLegacySurface.EVENT}),
}

_PROTOCOL_PROJECTION_STATE_NAMES = {
    "ProtocolStreamProjectionState",
    "StreamProjectionState",
}
_LEGACY_ITEM_MAPPER_KEYWORD = "legacy_item_mapper"

_STREAMING_RETURN_CONTAINER_NAMES = {
    "AsyncIterator",
    "AsyncGenerator",
    "AsyncIterable",
    "Generator",
    "Iterator",
}
_STREAMING_RETURN_ALIAS_NAMES = {"OutputGenerator"}
_LEGACY_STREAMING_RETURN_SYMBOL_NAMES = {
    "str",
    "Token",
    "TokenDetail",
    "ReasoningToken",
    "ToolCallToken",
    "Event",
    "OutputItem",
    "LegacyOutputItem",
    "OutputGenerator",
}
_NON_STRING_LEGACY_STREAMING_RETURN_SYMBOL_NAMES = (
    _LEGACY_STREAMING_RETURN_SYMBOL_NAMES - {"str"}
)
_PHASE_1_1_PUBLIC_STREAMING_RETURN_DEBT_CEILING = {
    ("avalan.event.manager", "EventManager.listen", "return"): frozenset(
        {"Event"}
    ),
}

_PHASE_1_1_INHERITED_TEXT_STREAM_CANONICALIZATION_CEILING: set[
    tuple[str, str]
] = set()
_TEXT_STREAM_LEGACY_CANONICALIZATION_BASE_NAMES = {
    "TextGenerationStream",
    "TextGenerationVendorStream",
}


class _LegacyStreamClassifierVisitor(NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.sites: dict[tuple[str, str], set[StreamLegacySurface]] = {}
        self._legacy_classifier_surface_name_scopes = [
            {
                name: frozenset({surface})
                for name, surface in _LEGACY_CLASSIFIER_SYMBOL_SURFACES.items()
            }
        ]

    def visit_ClassDef(self, node: ClassDef) -> None:
        self.stack.append(node.name)
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()
        self.stack.pop()

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self.stack.append(node.name)
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()
        self.stack.pop()

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            surface = self._legacy_classifier_import_surface(
                module,
                alias.name,
            )
            if surface is not None:
                self._record_legacy_classifier_alias(
                    alias.asname or alias.name,
                    {surface},
                )

    def visit_Assign(self, node: Assign) -> None:
        self._record_assignment_aliases(tuple(node.targets), node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if node.value is not None:
            self._record_assignment_aliases((node.target,), node.value)
        self.generic_visit(node)

    def visit_Call(self, node: Call) -> None:
        if (
            isinstance(node.func, Name)
            and node.func.id == "isinstance"
            and len(node.args) >= 2
        ):
            key = (self.module, ".".join(self.stack))
            surfaces = self._surfaces_for_node(node.args[1], key)
            if surfaces:
                self.sites.setdefault(key, set()).update(surfaces)
        self.generic_visit(node)

    def visit_If(self, node: If) -> None:
        if self._is_legacy_stream_rejection_guard(node):
            return
        self.generic_visit(node)

    def _surfaces_for_node(
        self,
        node: AST,
        key: tuple[str, str],
    ) -> set[StreamLegacySurface]:
        if isinstance(node, Name):
            if node.id == "str":
                if self._tracks_string_classifier(key):
                    return {StreamLegacySurface.STRING}
                return set()
            return self._legacy_classifier_surfaces_for_name(node.id)
        if isinstance(node, Attribute):
            surface = _LEGACY_CLASSIFIER_SYMBOL_SURFACES.get(node.attr)
            return set() if surface is None else {surface}
        if isinstance(node, Tuple):
            surfaces: set[StreamLegacySurface] = set()
            for element in node.elts:
                surfaces.update(self._surfaces_for_node(element, key))
            return surfaces
        if isinstance(node, BinOp) and isinstance(node.op, BitOr):
            return self._surfaces_for_node(
                node.left, key
            ) | self._surfaces_for_node(node.right, key)
        return set()

    def _push_scope(self) -> None:
        self._legacy_classifier_surface_name_scopes.append({})

    def _pop_scope(self) -> None:
        self._legacy_classifier_surface_name_scopes.pop()

    def _record_assignment_aliases(
        self,
        targets: tuple[AST, ...],
        value: AST,
    ) -> None:
        surfaces = self._alias_surfaces_for_node(value)
        if not surfaces:
            return
        target_names: set[str] = set()
        for target in targets:
            _add_defined_target_name(target_names, target)
        for target_name in target_names:
            self._record_legacy_classifier_alias(target_name, surfaces)

    def _alias_surfaces_for_node(
        self,
        node: AST,
    ) -> set[StreamLegacySurface]:
        if isinstance(node, Name):
            if node.id == "str":
                return set()
            return self._legacy_classifier_surfaces_for_name(node.id)
        if isinstance(node, Attribute):
            surface = _LEGACY_CLASSIFIER_SYMBOL_SURFACES.get(node.attr)
            return set() if surface is None else {surface}
        if isinstance(node, Tuple):
            surfaces: set[StreamLegacySurface] = set()
            for element in node.elts:
                surfaces.update(self._alias_surfaces_for_node(element))
            return surfaces
        if isinstance(node, BinOp) and isinstance(node.op, BitOr):
            return self._alias_surfaces_for_node(
                node.left
            ) | self._alias_surfaces_for_node(node.right)
        return set()

    def _legacy_classifier_surfaces_for_name(
        self,
        name: str,
    ) -> set[StreamLegacySurface]:
        for scope in reversed(self._legacy_classifier_surface_name_scopes):
            surfaces = scope.get(name)
            if surfaces is not None:
                return set(surfaces)
        return set()

    def _record_legacy_classifier_alias(
        self,
        name: str,
        surfaces: set[StreamLegacySurface],
    ) -> None:
        self._legacy_classifier_surface_name_scopes[-1][name] = frozenset(
            surfaces
        )

    @staticmethod
    def _legacy_classifier_import_surface(
        module: str,
        name: str,
    ) -> StreamLegacySurface | None:
        surface = _LEGACY_CLASSIFIER_SYMBOL_SURFACES.get(name)
        if surface is None:
            return None
        if name != "Event":
            return surface
        if module in _LEGACY_CLASSIFIER_EVENT_IMPORT_MODULES:
            return surface
        return None

    @staticmethod
    def _tracks_string_classifier(key: tuple[str, str]) -> bool:
        if key in _LEGACY_CLASSIFIER_STRING_SITES:
            return True
        full_qualname = key[1].lower()
        qualname = full_qualname.rsplit(".", maxsplit=1)[-1]
        if qualname == "map" and "streamadapter" in full_qualname:
            return True
        return any(
            marker in qualname
            for marker in _LEGACY_CLASSIFIER_STRING_CONTEXT_MARKERS
        )

    def _is_legacy_stream_rejection_guard(self, node: If) -> bool:
        if not isinstance(node.test, Call):
            return False
        call = node.test
        if not (
            isinstance(call.func, Name)
            and call.func.id == "isinstance"
            and len(call.args) >= 2
        ):
            return False
        key = (self.module, ".".join(self.stack))
        surfaces = self._surfaces_for_node(call.args[1], key)
        if not surfaces:
            return False
        return all(
            self._raises_stream_validation_error(statement)
            for statement in node.body
        )

    @staticmethod
    def _raises_stream_validation_error(statement: AST) -> bool:
        if not isinstance(statement, Raise):
            return False
        exception = statement.exc
        if isinstance(exception, Call):
            exception = exception.func
        return (
            isinstance(exception, Name)
            and exception.id == "StreamValidationError"
        )


def _legacy_stream_classifier_sites(
    module: str,
    tree: AST,
) -> dict[tuple[str, str], set[StreamLegacySurface]]:
    visitor = _LegacyStreamClassifierVisitor(module)
    visitor.visit(tree)
    return visitor.sites


def _source_legacy_stream_classifier_sites(
    root: Path,
) -> dict[tuple[str, str], frozenset[StreamLegacySurface]]:
    sites: dict[tuple[str, str], set[StreamLegacySurface]] = {}
    for module, path in _legacy_classifier_module_paths(root).items():
        module_sites = _legacy_stream_classifier_sites(
            module,
            parse((root / path).read_text(encoding="utf-8")),
        )
        for key, surfaces in module_sites.items():
            sites.setdefault(key, set()).update(surfaces)
    return {key: frozenset(value) for key, value in sites.items()}


def _legacy_classifier_module_paths(root: Path) -> dict[str, Path]:
    return {
        _module_name_from_source_path(root, path): path.relative_to(root)
        for path in sorted((root / "src" / "avalan").rglob("*.py"))
    }


def _inventory_legacy_stream_classifier_sites() -> (
    dict[tuple[str, str], frozenset[StreamLegacySurface]]
):
    sites: dict[tuple[str, str], set[StreamLegacySurface]] = {}
    for entry in legacy_stream_classifier_inventory():
        key = (entry.module, entry.qualname)
        class_surfaces = {
            surface
            for surface in entry.surfaces
            if (
                surface is not StreamLegacySurface.STRING
                or key in _LEGACY_CLASSIFIER_STRING_SITES
            )
        }
        if class_surfaces:
            sites[key] = class_surfaces
    return {key: frozenset(value) for key, value in sites.items()}


def _module_source_path(root: Path, module: str) -> Path:
    return root / Path("src").joinpath(*module.split(".")).with_suffix(".py")


def _add_defined_target_name(names: set[str], target: AST) -> None:
    if isinstance(target, Name):
        names.add(target.id)
        return
    if isinstance(target, Tuple):
        for element in target.elts:
            _add_defined_target_name(names, element)


def _module_defined_qualnames(tree: AST) -> set[str]:
    names: set[str] = set()
    for node in getattr(tree, "body", ()):
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            names.add(node.name)
        if isinstance(node, ClassDef):
            for child in node.body:
                if isinstance(child, (FunctionDef, AsyncFunctionDef)):
                    names.add(f"{node.name}.{child.name}")
            continue
        if isinstance(node, Assign):
            for target in node.targets:
                _add_defined_target_name(names, target)
            continue
        if isinstance(node, AnnAssign):
            _add_defined_target_name(names, node.target)
            continue
        if isinstance(node, (Import, ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
    return names


def _module_name_from_source_path(root: Path, path: Path) -> str:
    return ".".join(path.relative_to(root / "src").with_suffix("").parts)


def _runtime_source_test_import_sites(tree: AST) -> set[tuple[int, str]]:
    sites: set[tuple[int, str]] = set()

    def visit(node: AST) -> None:
        if isinstance(node, Import):
            for alias in node.names:
                if _is_tests_module(alias.name):
                    sites.add((node.lineno, alias.name))
        if isinstance(node, ImportFrom):
            module = node.module or ""
            if _is_tests_module(module):
                sites.add((node.lineno, module))
        for child in iter_child_nodes(node):
            visit(child)

    visit(tree)
    return sites


def _is_tests_module(module: str) -> bool:
    return module == "tests" or module.startswith("tests.")


def _ast_root_name(node: AST) -> str | None:
    if isinstance(node, Name):
        return node.id
    if isinstance(node, Attribute):
        return node.attr
    if isinstance(node, Subscript):
        return _ast_root_name(node.value)
    return None


def _subscript_args(node: Subscript) -> tuple[AST, ...]:
    if isinstance(node.slice, Tuple):
        return tuple(node.slice.elts)
    return (node.slice,)


def _legacy_streaming_item_symbols(
    node: AST,
    *,
    allow_string: bool,
    item_alias_symbols: dict[str, frozenset[str]] | None = None,
) -> set[str]:
    if isinstance(node, Name):
        if item_alias_symbols is not None and node.id in item_alias_symbols:
            return set(item_alias_symbols[node.id])
        if node.id == "str":
            return {"str"} if allow_string else set()
        if node.id in _NON_STRING_LEGACY_STREAMING_RETURN_SYMBOL_NAMES:
            return {node.id}
        return set()
    if isinstance(node, Attribute):
        if node.attr in _NON_STRING_LEGACY_STREAMING_RETURN_SYMBOL_NAMES:
            return {node.attr}
        return set()
    if isinstance(node, BinOp) and isinstance(node.op, BitOr):
        return _legacy_streaming_item_symbols(
            node.left,
            allow_string=True,
            item_alias_symbols=item_alias_symbols,
        ) | _legacy_streaming_item_symbols(
            node.right,
            allow_string=True,
            item_alias_symbols=item_alias_symbols,
        )
    if isinstance(node, Tuple):
        symbols: set[str] = set()
        for element in node.elts:
            symbols.update(
                _legacy_streaming_item_symbols(
                    element,
                    allow_string=allow_string,
                    item_alias_symbols=item_alias_symbols,
                )
            )
        return symbols
    if isinstance(node, Subscript):
        root_name = _ast_root_name(node.value)
        symbols = set[str]()
        for element in _subscript_args(node):
            symbols.update(
                _legacy_streaming_item_symbols(
                    element,
                    allow_string=root_name in {"tuple", "Tuple"},
                    item_alias_symbols=item_alias_symbols,
                )
            )
        return symbols
    return set()


def _streaming_return_legacy_symbols(
    node: AST,
    *,
    container_names: set[str] | None = None,
    item_alias_symbols: dict[str, frozenset[str]] | None = None,
) -> set[str]:
    streaming_container_names = (
        _STREAMING_RETURN_CONTAINER_NAMES
        if container_names is None
        else container_names
    )
    if isinstance(node, Name):
        if node.id in _STREAMING_RETURN_ALIAS_NAMES:
            return {node.id}
        return set()
    if isinstance(node, Attribute):
        if node.attr in _STREAMING_RETURN_ALIAS_NAMES:
            return {node.attr}
        return set()
    if isinstance(node, BinOp) and isinstance(node.op, BitOr):
        return _streaming_return_legacy_symbols(
            node.left,
            container_names=streaming_container_names,
            item_alias_symbols=item_alias_symbols,
        ) | _streaming_return_legacy_symbols(
            node.right,
            container_names=streaming_container_names,
            item_alias_symbols=item_alias_symbols,
        )
    if isinstance(node, Tuple):
        symbols: set[str] = set()
        for element in node.elts:
            symbols.update(
                _streaming_return_legacy_symbols(
                    element,
                    container_names=streaming_container_names,
                    item_alias_symbols=item_alias_symbols,
                )
            )
        return symbols
    if isinstance(node, Subscript):
        root_name = _ast_root_name(node.value)
        args = _subscript_args(node)
        if root_name in streaming_container_names and args:
            return _legacy_streaming_item_symbols(
                args[0],
                allow_string=True,
                item_alias_symbols=item_alias_symbols,
            )
        symbols = set[str]()
        for element in args:
            symbols.update(
                _streaming_return_legacy_symbols(
                    element,
                    container_names=streaming_container_names,
                    item_alias_symbols=item_alias_symbols,
                )
            )
        return symbols
    return set()


def _target_name(node: AST) -> str | None:
    if isinstance(node, Name):
        return node.id
    if isinstance(node, Attribute):
        return node.attr
    return None


class _PublicStreamingReturnVisitor(NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.sites: dict[tuple[str, str, str], frozenset[str]] = {}
        self._streaming_container_name_scopes = [
            set(_STREAMING_RETURN_CONTAINER_NAMES)
        ]
        self._legacy_item_alias_symbol_scopes: list[
            dict[str, frozenset[str]]
        ] = [{}]

    def visit_ClassDef(self, node: ClassDef) -> None:
        for base in node.bases:
            symbols = _streaming_return_legacy_symbols(
                base,
                container_names=self._streaming_container_names(),
                item_alias_symbols=self._legacy_item_alias_symbols(),
            )
            if symbols:
                self.sites[(self.module, node.name, "base")] = frozenset(
                    symbols
                )
        self.stack.append(node.name)
        self._push_scope()
        for child in node.body:
            if isinstance(
                child,
                (
                    FunctionDef,
                    AsyncFunctionDef,
                    ClassDef,
                    Assign,
                    AnnAssign,
                    ImportFrom,
                ),
            ):
                self.visit(child)
        self._pop_scope()
        self.stack.pop()

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self._visit_function_return(node)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        self._visit_function_return(node)

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        for alias in node.names:
            if alias.name in self._streaming_container_names():
                self._streaming_container_name_scopes[-1].add(
                    alias.asname or alias.name
                )

    def visit_Assign(self, node: Assign) -> None:
        self._record_streaming_container_aliases(
            tuple(node.targets),
            node.value,
        )
        symbols = _streaming_return_legacy_symbols(
            node.value,
            container_names=self._streaming_container_names(),
            item_alias_symbols=self._legacy_item_alias_symbols(),
        )
        self._record_legacy_item_aliases(tuple(node.targets), node.value)
        if not symbols:
            return
        for target in node.targets:
            target_name = _target_name(target)
            if target_name is not None:
                self.sites[
                    (
                        self.module,
                        ".".join(self.stack + [target_name]),
                        "alias",
                    )
                ] = frozenset(symbols)

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        value = node.value or node.annotation
        if value is None:
            return
        if node.value is not None:
            self._record_streaming_container_aliases(
                (node.target,),
                node.value,
            )
        symbols = _streaming_return_legacy_symbols(
            value,
            container_names=self._streaming_container_names(),
            item_alias_symbols=self._legacy_item_alias_symbols(),
        )
        self._record_legacy_item_aliases((node.target,), value)
        target_name = _target_name(node.target)
        if symbols and target_name is not None:
            self.sites[
                (self.module, ".".join(self.stack + [target_name]), "alias")
            ] = frozenset(symbols)

    def _visit_function_return(
        self, node: FunctionDef | AsyncFunctionDef
    ) -> None:
        if node.returns is None:
            return
        symbols = _streaming_return_legacy_symbols(
            node.returns,
            container_names=self._streaming_container_names(),
            item_alias_symbols=self._legacy_item_alias_symbols(),
        )
        if symbols:
            self.sites[
                (self.module, ".".join(self.stack + [node.name]), "return")
            ] = frozenset(symbols)

    def _push_scope(self) -> None:
        self._streaming_container_name_scopes.append(set())
        self._legacy_item_alias_symbol_scopes.append({})

    def _pop_scope(self) -> None:
        self._streaming_container_name_scopes.pop()
        self._legacy_item_alias_symbol_scopes.pop()

    def _streaming_container_names(self) -> set[str]:
        names: set[str] = set()
        for scope in self._streaming_container_name_scopes:
            names.update(scope)
        return names

    def _record_streaming_container_aliases(
        self,
        targets: tuple[AST, ...],
        value: AST,
    ) -> None:
        if not isinstance(value, (Name, Attribute)):
            return
        if not self._is_streaming_container_reference(value):
            return

        target_names: set[str] = set()
        for target in targets:
            _add_defined_target_name(target_names, target)
        self._streaming_container_name_scopes[-1].update(target_names)

    def _is_streaming_container_reference(self, node: AST) -> bool:
        root_name = _ast_root_name(node)
        return (
            root_name is not None
            and root_name in self._streaming_container_names()
        )

    def _record_legacy_item_aliases(
        self,
        targets: tuple[AST, ...],
        value: AST,
    ) -> None:
        root_name = (
            _ast_root_name(value.value)
            if isinstance(value, Subscript)
            else None
        )
        if root_name in self._streaming_container_names():
            return

        symbols = _legacy_streaming_item_symbols(
            value,
            allow_string=True,
            item_alias_symbols=self._legacy_item_alias_symbols(),
        )
        if not symbols:
            return

        target_names: set[str] = set()
        for target in targets:
            _add_defined_target_name(target_names, target)
        for target_name in target_names:
            self._legacy_item_alias_symbol_scopes[-1][target_name] = frozenset(
                symbols
            )

    def _legacy_item_alias_symbols(self) -> dict[str, frozenset[str]]:
        aliases: dict[str, frozenset[str]] = {}
        for scope in self._legacy_item_alias_symbol_scopes:
            aliases.update(scope)
        return aliases


def _source_public_streaming_return_legacy_sites(
    root: Path,
) -> dict[tuple[str, str, str], frozenset[str]]:
    sites: dict[tuple[str, str, str], frozenset[str]] = {}
    for path in (root / "src" / "avalan").rglob("*.py"):
        visitor = _PublicStreamingReturnVisitor(
            _module_name_from_source_path(root, path)
        )
        visitor.visit(parse(path.read_text(encoding="utf-8")))
        sites.update(visitor.sites)
    return sites


class _ProtocolProjectionMapperVisitor(NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.sites: set[tuple[str, str, int]] = set()
        self._projection_state_name_scopes = [
            set(_PROTOCOL_PROJECTION_STATE_NAMES)
        ]
        self._legacy_mapper_kwargs_name_scopes = [set[str]()]
        self._legacy_mapper_key_name_scopes = [set[str]()]

    def visit_Module(self, node: AST) -> None:
        self._prime_scope_references(list(getattr(node, "body", ())))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ClassDef) -> None:
        self.stack.append(node.name)
        self._push_scope()
        self._prime_scope_references(node.body)
        self.generic_visit(node)
        self._pop_scope()
        self.stack.pop()

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self._record_projection_state_wrapper(node)
        self.stack.append(node.name)
        self._push_scope()
        self._prime_scope_references(node.body)
        self.generic_visit(node)
        self._pop_scope()
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        self._record_projection_state_wrapper(node)
        self.stack.append(node.name)
        self._push_scope()
        self._prime_scope_references(node.body)
        self.generic_visit(node)
        self._pop_scope()
        self.stack.pop()

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        self._record_import_aliases(node)
        self.generic_visit(node)

    def visit_Assign(self, node: Assign) -> None:
        self._record_assignment_aliases(tuple(node.targets), node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if node.value is not None:
            self._record_assignment_aliases((node.target,), node.value)
        self.generic_visit(node)

    def visit_Call(self, node: Call) -> None:
        if self._is_projection_state_reference(
            node.func
        ) and self._call_has_legacy_item_mapper(node):
            self.sites.add(
                (
                    self.module,
                    ".".join(self.stack) or "<module>",
                    node.lineno,
                )
            )
        self.generic_visit(node)

    def _push_scope(self) -> None:
        self._projection_state_name_scopes.append(set())
        self._legacy_mapper_kwargs_name_scopes.append(set())
        self._legacy_mapper_key_name_scopes.append(set())

    def _pop_scope(self) -> None:
        self._projection_state_name_scopes.pop()
        self._legacy_mapper_kwargs_name_scopes.pop()
        self._legacy_mapper_key_name_scopes.pop()

    def _prime_scope_references(self, body: list[AST]) -> None:
        for child in body:
            if isinstance(child, ImportFrom):
                self._record_import_aliases(child)
        for child in body:
            if isinstance(child, Assign):
                self._record_assignment_aliases(
                    tuple(child.targets),
                    child.value,
                )
            if isinstance(child, AnnAssign) and child.value is not None:
                self._record_assignment_aliases((child.target,), child.value)
        for child in body:
            if isinstance(child, (FunctionDef, AsyncFunctionDef)):
                self._record_projection_state_wrapper(child)

    def _record_import_aliases(self, node: ImportFrom) -> None:
        for alias in node.names:
            if alias.name in _PROTOCOL_PROJECTION_STATE_NAMES:
                self._projection_state_name_scopes[-1].add(
                    alias.asname or alias.name
                )

    def _record_assignment_aliases(
        self,
        targets: tuple[AST, ...],
        value: AST,
    ) -> None:
        target_names: set[str] = set()
        for target in targets:
            _add_defined_target_name(target_names, target)
        if not target_names:
            return
        if self._is_projection_state_reference(value):
            self._projection_state_name_scopes[-1].update(target_names)
        if self._is_legacy_item_mapper_key(value):
            self._legacy_mapper_key_name_scopes[-1].update(target_names)
        if self._kwargs_value_has_legacy_item_mapper(value):
            self._legacy_mapper_kwargs_name_scopes[-1].update(target_names)

    def _record_projection_state_wrapper(
        self,
        node: FunctionDef | AsyncFunctionDef,
    ) -> None:
        if self._function_forwards_kwargs_to_projection_state(node):
            self._projection_state_name_scopes[-1].add(node.name)

    def _function_forwards_kwargs_to_projection_state(
        self,
        node: FunctionDef | AsyncFunctionDef,
    ) -> bool:
        if node.args.kwarg is None:
            return False
        forwarded_names = {node.args.kwarg.arg}
        projection_state_names = self._projection_state_names()
        return any(
            self._node_forwards_kwargs_to_projection_state(
                child,
                forwarded_names,
                projection_state_names,
            )
            for child in node.body
        )

    def _node_forwards_kwargs_to_projection_state(
        self,
        node: AST,
        forwarded_names: set[str],
        projection_state_names: set[str],
    ) -> bool:
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            return False
        if isinstance(node, Assign):
            self._record_projection_state_aliases(
                tuple(node.targets),
                node.value,
                projection_state_names,
            )
        if isinstance(node, AnnAssign) and node.value is not None:
            self._record_projection_state_aliases(
                (node.target,),
                node.value,
                projection_state_names,
            )
        if isinstance(node, Call) and self._node_root_name_is_in(
            node.func,
            projection_state_names,
        ):
            return any(
                keyword.arg is None
                and isinstance(keyword.value, Name)
                and keyword.value.id in forwarded_names
                for keyword in node.keywords
            )
        return any(
            self._node_forwards_kwargs_to_projection_state(
                child,
                forwarded_names,
                projection_state_names,
            )
            for child in iter_child_nodes(node)
        )

    def _record_projection_state_aliases(
        self,
        targets: tuple[AST, ...],
        value: AST,
        projection_state_names: set[str],
    ) -> None:
        if not self._node_root_name_is_in(value, projection_state_names):
            return
        for target in targets:
            _add_defined_target_name(projection_state_names, target)

    @staticmethod
    def _node_root_name_is_in(node: AST, names: set[str]) -> bool:
        root_name = _ast_root_name(node)
        return root_name is not None and root_name in names

    def _is_projection_state_reference(self, node: AST) -> bool:
        root_name = _ast_root_name(node)
        return (
            root_name is not None
            and root_name in self._projection_state_names()
        )

    def _projection_state_names(self) -> set[str]:
        names: set[str] = set()
        for scope in self._projection_state_name_scopes:
            names.update(scope)
        return names

    def _legacy_mapper_kwargs_names(self) -> set[str]:
        names: set[str] = set()
        for scope in self._legacy_mapper_kwargs_name_scopes:
            names.update(scope)
        return names

    def _legacy_mapper_key_names(self) -> set[str]:
        names: set[str] = set()
        for scope in self._legacy_mapper_key_name_scopes:
            names.update(scope)
        return names

    def _call_has_legacy_item_mapper(self, node: Call) -> bool:
        for keyword in node.keywords:
            if keyword.arg == _LEGACY_ITEM_MAPPER_KEYWORD:
                return True
            if (
                keyword.arg is None
                and self._kwargs_value_has_legacy_item_mapper(keyword.value)
            ):
                return True
        return False

    def _kwargs_value_has_legacy_item_mapper(self, node: AST) -> bool:
        if isinstance(node, Name):
            return node.id in self._legacy_mapper_kwargs_names()
        if isinstance(node, Dict):
            for key, value in zip(node.keys, node.values):
                if self._is_legacy_item_mapper_key(key):
                    return True
                if key is None and self._kwargs_value_has_legacy_item_mapper(
                    value
                ):
                    return True
            return False
        if isinstance(node, Call) and _ast_root_name(node.func) == "dict":
            for arg in node.args:
                if self._kwargs_value_has_legacy_item_mapper(arg):
                    return True
            return any(
                keyword.arg == _LEGACY_ITEM_MAPPER_KEYWORD
                or (
                    keyword.arg is None
                    and self._kwargs_value_has_legacy_item_mapper(
                        keyword.value
                    )
                )
                for keyword in node.keywords
            )
        return False

    def _is_legacy_item_mapper_key(self, node: AST | None) -> bool:
        if (
            isinstance(node, Constant)
            and node.value == _LEGACY_ITEM_MAPPER_KEYWORD
        ):
            return True
        if (
            isinstance(node, Name)
            and node.id in self._legacy_mapper_key_names()
        ):
            return True
        return False


def _source_protocol_projection_legacy_mapper_sites(
    root: Path,
) -> set[tuple[str, str, int]]:
    sites: set[tuple[str, str, int]] = set()
    for path in (root / "src" / "avalan" / "server").rglob("*.py"):
        visitor = _ProtocolProjectionMapperVisitor(
            _module_name_from_source_path(root, path)
        )
        visitor.visit(parse(path.read_text(encoding="utf-8")))
        sites.update(visitor.sites)
    return sites


def _source_inherited_text_stream_canonicalization_sites(
    root: Path,
) -> set[tuple[str, str]]:
    module_trees = {
        _module_name_from_source_path(root, path): parse(
            path.read_text(encoding="utf-8")
        )
        for path in sorted((root / "src" / "avalan").rglob("*.py"))
    }
    return _inherited_text_stream_canonicalization_sites(module_trees)


def _inherited_text_stream_canonicalization_sites(
    module_trees: dict[str, AST],
) -> set[tuple[str, str]]:
    class_defs: dict[tuple[str, str], tuple[set[str], bool]] = {}
    for module, tree in module_trees.items():
        for node in getattr(tree, "body", ()):
            if not isinstance(node, ClassDef):
                continue
            base_names = {
                root_name
                for base in node.bases
                if (root_name := _ast_root_name(base)) is not None
            }
            class_defs[(module, node.name)] = (
                base_names,
                _class_defines_canonical_stream(node),
            )

    legacy_stream_names = set(_TEXT_STREAM_LEGACY_CANONICALIZATION_BASE_NAMES)
    sites: set[tuple[str, str]] = set()
    changed = True
    while changed:
        changed = False
        for key, (base_names, defines_canonical_stream) in class_defs.items():
            if key in sites or defines_canonical_stream:
                continue
            if base_names.isdisjoint(legacy_stream_names):
                continue
            sites.add(key)
            legacy_stream_names.add(key[1])
            changed = True
    return sites


def _class_defines_canonical_stream(node: ClassDef) -> bool:
    return any(
        isinstance(child, (FunctionDef, AsyncFunctionDef))
        and child.name == "canonical_stream"
        for child in node.body
    )


class StreamContractTestCase(TestCase):
    def test_taxonomy_maps_every_kind_to_channel_and_terminal_outcome(
        self,
    ) -> None:
        self.assertEqual(
            {channel.value for channel in StreamChannel},
            {
                "answer",
                "reasoning",
                "tool_call",
                "tool_execution",
                "flow",
                "usage",
                "control",
            },
        )
        self.assertEqual(
            {outcome.value for outcome in StreamTerminalOutcome},
            {"completed", "errored", "cancelled"},
        )
        self.assertEqual(
            {visibility.value for visibility in StreamVisibility},
            {"public", "private", "redacted", "diagnostic"},
        )

        for kind in StreamItemKind:
            self.assertIsInstance(stream_channel_for_kind(kind), StreamChannel)

        self.assertIs(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_COMPLETED),
            StreamTerminalOutcome.COMPLETED,
        )
        self.assertIs(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_ERRORED),
            StreamTerminalOutcome.ERRORED,
        )
        self.assertIs(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_CANCELLED),
            StreamTerminalOutcome.CANCELLED,
        )
        self.assertIsNone(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_CLOSED)
        )
        self.assertTrue(is_stream_terminal_kind(StreamItemKind.STREAM_ERRORED))
        self.assertFalse(
            is_stream_terminal_kind(StreamItemKind.TOOL_EXECUTION_ERROR)
        )
        self.assertTrue(
            is_tool_execution_terminal_kind(
                StreamItemKind.TOOL_EXECUTION_CANCELLED
            )
        )
        self.assertFalse(
            is_tool_execution_terminal_kind(StreamItemKind.STREAM_CANCELLED)
        )

        with self.assertRaises(AssertionError):
            stream_channel_for_kind("answer.delta")  # type: ignore[arg-type]

    def test_observability_payload_uses_lightweight_stream_summary(
        self,
    ) -> None:
        item = _item(
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            3,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="large output",
            data={"z": "body", "a": 1},
            metadata={"debug": "full"},
            provider_payload={"raw": "payload"},
            provider_family="openai",
            provider_event_type="response.output_text.delta",
        )

        payload = stream_observability_payload(item)

        self.assertEqual(payload["stream_session_id"], "stream-1")
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["turn_id"], "turn-1")
        self.assertEqual(payload["sequence"], 3)
        self.assertEqual(
            payload["kind"], StreamItemKind.TOOL_EXECUTION_OUTPUT.value
        )
        self.assertEqual(
            payload["channel"], StreamChannel.TOOL_EXECUTION.value
        )
        self.assertEqual(payload["correlation"], {"tool_call_id": "call-1"})
        self.assertEqual(payload["provider_family"], "openai")
        self.assertEqual(
            payload["provider_event_type"], "response.output_text.delta"
        )
        self.assertEqual(
            payload["summary"],
            {
                "text_delta_length": 12,
                "data_keys": ["a", "z"],
                "metadata_keys": ["debug"],
                "has_provider_payload": True,
            },
        )
        self.assertNotIn("text_delta", payload)
        self.assertNotIn("data", payload)
        self.assertNotIn("metadata", payload)
        self.assertNotIn("provider_payload", payload)

    def test_observability_payload_bounds_mapping_key_summaries(
        self,
    ) -> None:
        item = _item(
            StreamItemKind.STREAM_DIAGNOSTIC,
            4,
            data={f"d{i:02d}": i for i in range(20)},
            metadata={f"m{i:02d}": i for i in range(18)},
        )

        payload = stream_observability_payload(item)
        summary = cast(dict[str, object], payload["summary"])

        self.assertEqual(
            summary["data_keys"],
            [f"d{i:02d}" for i in range(16)],
        )
        self.assertEqual(summary["data_key_count"], 20)
        self.assertIs(summary["data_keys_truncated"], True)
        self.assertEqual(
            summary["metadata_keys"],
            [f"m{i:02d}" for i in range(16)],
        )
        self.assertEqual(summary["metadata_key_count"], 18)
        self.assertIs(summary["metadata_keys_truncated"], True)
        self.assertNotIn("d19", repr(summary["data_keys"]))
        self.assertNotIn("m17", repr(summary["metadata_keys"]))

    def test_observability_payload_bounds_mapping_key_lengths(
        self,
    ) -> None:
        data_key = "d" * 200
        metadata_key = "m" * 200
        item = _item(
            StreamItemKind.STREAM_DIAGNOSTIC,
            4,
            data={data_key: 1},
            metadata={metadata_key: 2},
        )

        payload = stream_observability_payload(item)
        summary = cast(dict[str, object], payload["summary"])

        self.assertEqual(summary["data_keys"], ["d" * 125 + "..."])
        self.assertEqual(summary["data_key_count"], 1)
        self.assertIs(summary["data_keys_truncated"], True)
        self.assertEqual(summary["metadata_keys"], ["m" * 125 + "..."])
        self.assertEqual(summary["metadata_key_count"], 1)
        self.assertIs(summary["metadata_keys_truncated"], True)
        self.assertNotIn(data_key, repr(summary["data_keys"]))
        self.assertNotIn(metadata_key, repr(summary["metadata_keys"]))

    def test_observability_payload_includes_usage_and_terminal_outcome(
        self,
    ) -> None:
        usage_item = _item(
            StreamItemKind.STREAM_COMPLETED,
            2,
            usage={"output_tokens": 4},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

        payload = stream_observability_payload(usage_item)

        self.assertEqual(payload["usage"], {"output_tokens": 4})
        self.assertEqual(payload["terminal_outcome"], "completed")
        self.assertNotIn("summary", payload)

    def test_observability_payload_summarizes_non_mapping_data(self) -> None:
        item = _item(
            StreamItemKind.STREAM_DIAGNOSTIC,
            4,
            data=["diagnostic"],
        )

        payload = stream_observability_payload(item)

        self.assertEqual(payload["summary"], {"data_type": "list"})

    def test_observability_payload_rejects_non_stream_items(self) -> None:
        with self.assertRaises(AssertionError):
            stream_observability_payload(cast(Any, object()))

    def test_project_canonical_stream_item_preserves_fields(
        self,
    ) -> None:
        item = _item(
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            3,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="stdout",
            data={"kind": "stdout"},
            usage=None,
            provider_family="openai",
            provider_event_type="tool.output",
        )

        projection = project_canonical_stream_item(item)

        self.assertEqual(projection.stream_session_id, item.stream_session_id)
        self.assertEqual(projection.sequence, item.sequence)
        self.assertIs(projection.kind, StreamItemKind.TOOL_EXECUTION_OUTPUT)
        self.assertEqual(projection.text_delta, "stdout")
        self.assertEqual(projection.data, {"kind": "stdout"})
        self.assertEqual(projection.tool_call_id, "call-1")
        self.assertEqual(projection.provider_family, "openai")
        self.assertEqual(projection.provider_event_type, "tool.output")

    def test_stream_projection_state_projects_canonical_and_projection_items(
        self,
    ) -> None:
        started = _item(StreamItemKind.STREAM_STARTED, 0)
        answer = _item(
            StreamItemKind.ANSWER_DELTA,
            1,
            text_delta="answer",
        )
        answer_done = _item(StreamItemKind.ANSWER_DONE, 2)
        completed = _item(
            StreamItemKind.STREAM_COMPLETED,
            3,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        projections = (
            state.project(
                started,
                99,
                unsupported_message="unsupported stream item",
            ),
            state.project(
                project_canonical_stream_item(answer),
                100,
                unsupported_message="unsupported stream item",
            ),
            state.project(
                answer_done,
                101,
                unsupported_message="unsupported stream item",
            ),
            state.project(
                completed,
                102,
                unsupported_message="unsupported stream item",
            ),
        )

        state.validate_complete()
        self.assertTrue(state.has_canonical_items)
        self.assertEqual([item.sequence for item in projections], [0, 1, 2, 3])
        self.assertEqual(state.accumulator.answer_text, "answer")
        terminal = state.terminal_projection()
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal.sequence, 3)

    def test_stream_projection_state_single_item_path_bypasses_project_many(
        self,
    ) -> None:
        state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        with patch.object(
            StreamProjectionState,
            "project_many",
            side_effect=AssertionError("project_many called"),
        ):
            projection = state.project(
                _item(StreamItemKind.STREAM_STARTED, 0),
                99,
                unsupported_message="unsupported stream item",
            )

        self.assertEqual(projection.sequence, 0)
        self.assertTrue(state.has_canonical_items)
        self.assertEqual(
            [item.sequence for item in state.accumulator.items],
            [0],
        )

    def test_stream_projection_state_project_many_keeps_single_item_contract(
        self,
    ) -> None:
        state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )
        answer_projection = project_canonical_stream_item(
            _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="answer")
        )

        started = state.project_many(
            _item(StreamItemKind.STREAM_STARTED, 0),
            99,
            unsupported_message="unsupported stream item",
        )
        answer = state.project_many(
            answer_projection,
            100,
            unsupported_message="unsupported stream item",
        )

        self.assertEqual(len(started), 1)
        self.assertEqual(started[0].sequence, 0)
        self.assertEqual(answer, (answer_projection,))
        self.assertTrue(state.has_canonical_items)
        self.assertEqual(
            [item.sequence for item in state.accumulator.items],
            [0, 1],
        )
        self.assertEqual(state.accumulator.answer_text, "answer")

    def test_stream_projection_state_rejects_terminal_with_open_channel(
        self,
    ) -> None:
        state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )
        state.project(
            project_canonical_stream_item(
                _item(StreamItemKind.STREAM_STARTED, 0)
            ),
            99,
            unsupported_message="unsupported stream item",
        )
        state.project(
            project_canonical_stream_item(
                _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="answer")
            ),
            100,
            unsupported_message="unsupported stream item",
        )

        with self.assertRaisesRegex(
            StreamValidationError, "answer channel missing done"
        ):
            state.project(
                project_canonical_stream_item(_stream_errored(2)),
                101,
                unsupported_message="unsupported stream item",
            )
        self.assertIsNone(state.terminal_projection())

    def test_stream_projection_state_keeps_terminal_after_history_eviction(
        self,
    ) -> None:
        state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
            accumulator=CanonicalStreamAccumulator(
                retention_policy=StreamRetentionPolicy(
                    accumulator_item_limit=1,
                )
            ),
        )
        for item in (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(StreamItemKind.USAGE_COMPLETED, 1, usage={}),
            _stream_completed(2),
            _item(StreamItemKind.STREAM_CLOSED, 3),
        ):
            state.project(
                item,
                item.sequence,
                unsupported_message="unsupported stream item",
            )

        state.validate_complete()
        self.assertEqual(
            [item.kind for item in state.accumulator.items],
            [StreamItemKind.STREAM_CLOSED],
        )
        terminal = state.terminal_projection()
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal.sequence, 2)
        self.assertIs(
            terminal.terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    def test_stream_projection_state_returns_no_terminal_for_open_stream(
        self,
    ) -> None:
        state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )
        state.project(
            _item(StreamItemKind.STREAM_STARTED, 0),
            0,
            unsupported_message="unsupported stream item",
        )

        self.assertIsNone(state.terminal_projection())

    def test_stream_projection_state_legacy_rejection_first_item(
        self,
    ) -> None:
        state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )
        legacy_rejection_token = Token(token="legacy")

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported stream item",
        ):
            state.project(
                legacy_rejection_token,
                7,
                unsupported_message="unsupported stream item",
            )

        self.assertFalse(state.has_canonical_items)

    def test_stream_projection_state_rejects_legacy_per_state(
        self,
    ) -> None:
        states = (
            StreamProjectionState(
                stream_session_id="first-stream",
                run_id="first-run",
                turn_id="first-turn",
            ),
            StreamProjectionState(
                stream_session_id="second-stream",
                run_id="second-run",
                turn_id="second-turn",
            ),
        )

        for state in states:
            with self.subTest(stream_session_id=state.stream_session_id):
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "unsupported stream item",
                ):
                    state.project(
                        "legacy",
                        0,
                        unsupported_message="unsupported stream item",
                    )
                self.assertFalse(state.has_canonical_items)

    def test_project_stream_consumer_item_uses_shared_projection_state(
        self,
    ) -> None:
        canonical = _item(
            StreamItemKind.ANSWER_DELTA,
            3,
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            text_delta="canonical",
        )
        legacy_rejection_items = (
            "legacy",
            Token(token="legacy"),
            TokenDetail(token="legacy", id=1),
            ReasoningToken(token="legacy"),
            ToolCallToken(
                token="{}",
                call=ToolCall(
                    id="legacy-call",
                    name="legacy",
                    arguments={},
                ),
            ),
            Event(type=EventType.START),
            object(),
        )

        canonical_projection = project_stream_consumer_item(
            canonical,
            99,
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
            unsupported_message="unsupported helper stream item",
        )
        for legacy_rejection_item in legacy_rejection_items:
            with self.subTest(item_type=type(legacy_rejection_item).__name__):
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "unsupported helper stream item",
                ):
                    project_stream_consumer_item(
                        legacy_rejection_item,
                        5,
                        stream_session_id="fallback-stream",
                        run_id="fallback-run",
                        turn_id="fallback-turn",
                        unsupported_message="unsupported helper stream item",
                    )

        self.assertEqual(canonical_projection.stream_session_id, "stream")
        self.assertEqual(canonical_projection.sequence, 3)
        self.assertEqual(canonical_projection.text_delta, "canonical")

    def test_project_stream_consumer_item_rejects_invalid_values(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported helper stream item",
        ):
            project_stream_consumer_item(
                object(),
                0,
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
                unsupported_message="unsupported helper stream item",
            )
        with self.assertRaises(AssertionError):
            project_stream_consumer_item(
                "legacy",
                -1,
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
                unsupported_message="unsupported helper stream item",
            )

    def test_stream_projection_state_rejects_invalid_and_mixed_items(
        self,
    ) -> None:
        invalid_state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported stream item",
        ):
            invalid_state.project(
                object(),
                0,
                unsupported_message="unsupported stream item",
            )

        mixed_state = StreamProjectionState(
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )
        mixed_state.project(
            _item(StreamItemKind.STREAM_STARTED, 0),
            0,
            unsupported_message="unsupported stream item",
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported stream item",
        ):
            mixed_state.project(
                "legacy",
                1,
                unsupported_message="unsupported stream item",
            )

    def test_valid_trace_fixture_serializes_contract_fields(self) -> None:
        timestamp = datetime(2026, 1, 2, 3, 4, 5)
        tool = StreamItemCorrelation(tool_call_id="tool-1")
        full_correlation = StreamItemCorrelation(
            provider_request_id="request-1",
            model_continuation_id="continuation-1",
            tool_call_id="tool-1",
            flow_run_id="flow-1",
            node_id="node-1",
            parent_sequence=1,
            protocol_item_id="protocol-1",
            task_id="task-1",
            artifact_id="artifact-1",
        )
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                1,
                correlation=StreamItemCorrelation(
                    model_continuation_id="continuation-1",
                    parent_sequence=0,
                ),
            ),
            _item(
                StreamItemKind.REASONING_DELTA,
                2,
                text_delta="plan",
                visibility=StreamVisibility.PRIVATE,
            ),
            _item(StreamItemKind.REASONING_DONE, 3),
            _item(
                StreamItemKind.ANSWER_DELTA,
                4,
                correlation=full_correlation,
                text_delta="answer",
                data={"chunk": 1},
                metadata={"provider": "fixture"},
                provider_payload={"native": True},
                provider_family="openai",
                provider_event_type="response.output_text.delta",
                timestamp=timestamp,
            ),
            _item(StreamItemKind.ANSWER_DONE, 5),
            _item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                6,
                correlation=tool,
                text_delta='{"expression"',
            ),
            _item(
                StreamItemKind.TOOL_CALL_READY,
                7,
                correlation=tool,
                data={"name": "math.calculator"},
            ),
            _item(StreamItemKind.TOOL_CALL_DONE, 8, correlation=tool),
            _item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                9,
                correlation=tool,
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                10,
                correlation=tool,
                text_delta="4",
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                11,
                correlation=tool,
                data={"step": 1},
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                12,
                correlation=tool,
                data={"result": 4},
            ),
            _item(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                13,
                correlation=StreamItemCorrelation(
                    model_continuation_id="continuation-1"
                ),
            ),
            _item(
                StreamItemKind.FLOW_EVENT,
                14,
                correlation=StreamItemCorrelation(
                    flow_run_id="flow-1", node_id="node-1"
                ),
                data={"state": "completed"},
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                15,
                text_delta="redacted detail",
                data={"code": "stream.note"},
                visibility=StreamVisibility.DIAGNOSTIC,
            ),
            _item(
                StreamItemKind.USAGE_UPDATE,
                16,
                usage={"input_tokens": 2},
            ),
            _item(
                StreamItemKind.USAGE_COMPLETED,
                17,
                usage={"input_tokens": 2, "output_tokens": 1},
            ),
            _stream_completed(18),
            _item(StreamItemKind.STREAM_CLOSED, 19),
        )

        validated = validate_canonical_stream_items(items)
        self.assertEqual(validated, items)
        self.assertTrue(items[18].is_stream_terminal)
        self.assertTrue(items[12].is_tool_execution_terminal)

        trace = StreamGoldenTrace(
            name="contract-fixture",
            description="Canonical stream contract fixture",
            items=items,
        )
        fixture = trace.to_fixture()

        self.assertEqual(fixture["format_version"], 1)
        self.assertEqual(fixture["name"], "contract-fixture")
        self.assertEqual(fixture["description"], trace.description)
        fixture_items = fixture["items"]
        self.assertIsInstance(fixture_items, list)
        answer = fixture_items[4]  # type: ignore[index]
        self.assertEqual(answer["kind"], "answer.delta")
        self.assertEqual(answer["text_delta"], "answer")
        self.assertEqual(answer["timestamp"], "2026-01-02T03:04:05")
        self.assertEqual(
            answer["correlation"],
            full_correlation.to_trace_dict(),
        )

    def test_completed_stream_allows_atomic_final_usage(self) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.STREAM_COMPLETED,
                1,
                usage={"input_tokens": 1, "output_tokens": 1},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        self.assertEqual(validate_canonical_stream_items(items), items)

    def test_completed_stream_allows_diagnostic_after_final_usage(
        self,
    ) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.USAGE_COMPLETED,
                1,
                usage={"input_tokens": 1},
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                2,
                data={"code": "stream.note"},
            ),
            _item(
                StreamItemKind.STREAM_COMPLETED,
                3,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            _item(StreamItemKind.STREAM_CLOSED, 4),
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.items, items)
        self.assertEqual(accumulator.final_usage, {"input_tokens": 1})
        self.assertEqual(accumulator.diagnostics, (items[2],))

    def test_accumulator_bounds_retained_history_losslessly(self) -> None:
        final_usage: object = {"output_tokens": 2}
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="hel"),
            _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="lo"),
            _item(StreamItemKind.ANSWER_DONE, 3),
            _item(
                StreamItemKind.FLOW_EVENT,
                4,
                data={"node": "old"},
            ),
            _item(
                StreamItemKind.FLOW_EVENT,
                5,
                data={"node": "new"},
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                6,
                text_delta="old diagnostic",
                visibility=StreamVisibility.DIAGNOSTIC,
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                7,
                text_delta="new diagnostic",
                visibility=StreamVisibility.DIAGNOSTIC,
            ),
            _item(
                StreamItemKind.USAGE_UPDATE,
                8,
                usage={"output_tokens": 1},
            ),
            _item(
                StreamItemKind.USAGE_COMPLETED,
                9,
                usage=final_usage,
            ),
            _stream_completed(10),
            _item(StreamItemKind.STREAM_CLOSED, 11),
        )
        retention_policy = StreamRetentionPolicy(
            accumulator_item_limit=3,
            replay_history_item_limit=1,
            flow_history_item_limit=1,
            metrics_history_item_limit=1,
        )
        accumulator = CanonicalStreamAccumulator(
            retention_policy=retention_policy
        )

        accumulator.add_many(items)

        self.assertIs(accumulator.retention_policy, retention_policy)
        self.assertEqual(accumulator.items, items[-3:])
        self.assertEqual(accumulator.answer_text, "hello")
        self.assertEqual(accumulator.flow_items, (items[5],))
        self.assertEqual(accumulator.diagnostics, (items[7],))
        self.assertEqual(accumulator.usage_items, (items[9],))
        self.assertEqual(accumulator.control_items, (items[11],))
        self.assertEqual(accumulator.final_usage, final_usage)
        self.assertIs(accumulator.terminal_item, items[10])
        self.assertEqual(accumulator.validate_complete(), items[-3:])

    def test_accumulator_allows_disabled_retained_views(self) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.USAGE_COMPLETED,
                1,
                usage={"output_tokens": 0},
            ),
            _stream_completed(2),
        )
        accumulator = CanonicalStreamAccumulator(
            retention_policy=StreamRetentionPolicy(
                replay_history_item_limit=0,
                flow_history_item_limit=0,
                metrics_history_item_limit=0,
            )
        )

        accumulator.add_many(items)

        self.assertEqual(accumulator.items, items)
        self.assertEqual(accumulator.usage_items, ())
        self.assertEqual(accumulator.control_items, ())
        self.assertEqual(accumulator.final_usage, {"output_tokens": 0})

    def test_accumulator_rejects_invalid_retention_policy(self) -> None:
        with self.assertRaises(AssertionError):
            CanonicalStreamAccumulator(retention_policy=cast(Any, object()))

    def test_error_and_cancel_are_terminal_without_final_usage(self) -> None:
        for kind, outcome in (
            (StreamItemKind.STREAM_ERRORED, StreamTerminalOutcome.ERRORED),
            (
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
            ),
        ):
            with self.subTest(kind=kind):
                items = (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _item(kind, 1, terminal_outcome=outcome),
                )
                self.assertEqual(validate_canonical_stream_items(items), items)

    def test_session_lifecycle_defaults_and_validation(self) -> None:
        lifecycle = StreamSessionLifecycle()

        self.assertTrue(lifecycle.single_use)
        self.assertTrue(lifecycle.cancellable)
        self.assertTrue(lifecycle.closeable)
        self.assertTrue(lifecycle.cleanup_owned)
        self.assertFalse(StreamSessionLifecycle(single_use=False).single_use)
        with self.assertRaises(AssertionError):
            StreamSessionLifecycle(cancellable="yes")  # type: ignore[arg-type]

    def test_runtime_contract_defaults_and_validation(self) -> None:
        contract = StreamRuntimeContract()

        self.assertIs(
            contract.backpressure_policy, StreamBackpressurePolicy.BLOCK
        )
        self.assertIs(
            contract.cancellation_drain_policy,
            StreamCancellationDrainPolicy.DRAIN_BUFFERED,
        )
        self.assertTrue(contract.close_after_terminal)
        self.assertTrue(contract.cancellation_as_terminal)
        self.assertTrue(contract.buffered_items_may_drain_after_cancellation)
        self.assertIs(validate_stream_runtime_contract(contract), contract)
        self.assertEqual(
            contract.retention_policy.accumulator_item_limit, 4096
        )
        self.assertEqual(contract.performance_budget.max_queue_depth, 64)
        self.assertEqual(
            contract.cancellation_propagation.targets,
            tuple(StreamCancellationPropagationTarget),
        )

        discard = StreamRuntimeContract(
            cancellation_drain_policy=(
                StreamCancellationDrainPolicy.DISCARD_BUFFERED
            ),
            retention_policy=StreamRetentionPolicy(
                replay_history_item_limit=0,
                ui_buffer_item_limit=0,
                metrics_history_item_limit=0,
                event_history_item_limit=0,
                mcp_resource_item_limit=0,
                a2a_task_record_item_limit=0,
                flow_history_item_limit=0,
            ),
            performance_budget=StreamPerformanceBudget(
                time_to_first_item_ms=1,
                cancellation_latency_ms=1,
                close_latency_ms=1,
                max_queue_depth=1,
                max_memory_bytes=1,
                per_item_overhead_us=1,
            ),
            cancellation_propagation=StreamCancellationPropagation(
                targets=(
                    StreamCancellationPropagationTarget.CONSUMER,
                    StreamCancellationPropagationTarget.STREAM_SESSION,
                )
            ),
        )
        self.assertFalse(discard.buffered_items_may_drain_after_cancellation)

        invalid_values = (
            lambda: StreamRetentionPolicy(accumulator_item_limit=0),
            lambda: StreamRetentionPolicy(accumulator_item_limit=True),  # type: ignore[arg-type]
            lambda: StreamRetentionPolicy(replay_history_item_limit=-1),
            lambda: StreamRetentionPolicy(active_session_lossless=False),
            lambda: StreamPerformanceBudget(max_queue_depth=0),
            lambda: StreamRuntimeContract(backpressure_policy="block"),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(
                cancellation_drain_policy="drain_buffered",  # type: ignore[arg-type]
            ),
            lambda: StreamRuntimeContract(retention_policy=object()),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(performance_budget=object()),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(
                cancellation_propagation=object(),  # type: ignore[arg-type]
            ),
            lambda: StreamCancellationPropagation(targets=[]),  # type: ignore[arg-type]
            lambda: StreamCancellationPropagation(targets=()),
            lambda: StreamCancellationPropagation(
                targets=(StreamCancellationPropagationTarget.CONSUMER,) * 2,
            ),
            lambda: StreamCancellationPropagation(
                targets=(cast(Any, "consumer"),)
            ),
            lambda: StreamCancellationPropagation(idempotent=False),
            lambda: StreamCancellationPropagation(
                starts_no_new_work_after_terminal=False
            ),
            lambda: StreamRuntimeContract(close_after_terminal="yes"),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(close_after_terminal=False),
            lambda: StreamRuntimeContract(cancellation_as_terminal=False),
            lambda: validate_stream_runtime_contract(cast(Any, None)),
        )

        for build_value in invalid_values:
            with self.subTest(build_value=build_value):
                with self.assertRaises(AssertionError):
                    build_value()

    def test_performance_budget_reconciliation_validation(self) -> None:
        reconciliation = StreamPerformanceBudgetReconciliation()

        self.assertEqual(reconciliation.tightened_metrics, ())
        self.assertEqual(reconciliation.loosened_metrics, ())
        self.assertEqual(
            reconciliation.benchmark_source,
            "specs/streaming/BENCHMARKS.md",
        )

        tightened_budget = StreamPerformanceBudget(
            time_to_first_item_ms=4000,
            cancellation_latency_ms=750,
            close_latency_ms=750,
            max_queue_depth=32,
            max_memory_bytes=8 * 1024 * 1024,
            per_item_overhead_us=125,
        )
        tightened = StreamPerformanceBudgetReconciliation(
            enforced_budget=tightened_budget,
            benchmark_source="benchmarks.md",
        )

        self.assertEqual(
            tightened.tightened_metrics,
            (
                "time_to_first_item_ms",
                "cancellation_latency_ms",
                "close_latency_ms",
                "max_queue_depth",
                "max_memory_bytes",
                "per_item_overhead_us",
            ),
        )
        self.assertEqual(tightened.loosened_metrics, ())

        invalid_values = (
            lambda: StreamPerformanceBudgetReconciliation(
                baseline_budget=object(),  # type: ignore[arg-type]
            ),
            lambda: StreamPerformanceBudgetReconciliation(
                enforced_budget=object(),  # type: ignore[arg-type]
            ),
            lambda: StreamPerformanceBudgetReconciliation(
                equivalence_harness_passed="yes",  # type: ignore[arg-type]
            ),
            lambda: StreamPerformanceBudgetReconciliation(
                benchmark_source="",
            ),
            lambda: StreamPerformanceBudgetReconciliation(
                enforced_budget=StreamPerformanceBudget(
                    time_to_first_item_ms=5001,
                ),
            ),
            lambda: StreamPerformanceBudgetReconciliation(
                enforced_budget=tightened_budget,
                equivalence_harness_passed=False,
            ),
        )

        for build_value in invalid_values:
            with self.subTest(build_value=build_value):
                with self.assertRaises(AssertionError):
                    build_value()

    def test_tool_lifecycle_contract_defaults_and_validation(self) -> None:
        contract = StreamToolLifecycleContract()

        self.assertTrue(contract.stable_tool_call_id_required)
        self.assertTrue(contract.confirmation_required_before_parallel_fanout)
        self.assertTrue(contract.side_effecting_tools_serial_by_default)
        self.assertTrue(contract.terminal_exactly_once)
        self.assertTrue(contract.terminal_idempotent_for_accumulation)

        invalid_values = (
            lambda: StreamToolLifecycleContract(observation_order="emission"),  # type: ignore[arg-type]
            lambda: StreamToolLifecycleContract(
                stable_tool_call_id_required=False
            ),
            lambda: StreamToolLifecycleContract(
                confirmation_required_before_parallel_fanout=False
            ),
            lambda: StreamToolLifecycleContract(
                side_effecting_tools_serial_by_default=False
            ),
            lambda: StreamToolLifecycleContract(terminal_exactly_once=False),
            lambda: StreamToolLifecycleContract(
                terminal_idempotent_for_accumulation=False
            ),
            lambda: StreamToolObservation(
                tool_call_id="",
                arguments="",
                output="",
                terminal_kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ),
            lambda: StreamToolObservation(
                tool_call_id="tool-1",
                arguments="",
                output="",
                terminal_kind=StreamItemKind.ANSWER_DELTA,
            ),
        )

        for build_value in invalid_values:
            with self.subTest(build_value=build_value):
                with self.assertRaises(AssertionError):
                    build_value()

    def test_item_validation_rejects_malformed_fields(self) -> None:
        bad_items = (
            lambda: _item(
                StreamItemKind.ANSWER_DELTA,
                0,
                channel=StreamChannel.REASONING,
                text_delta="x",
            ),
            lambda: _item(StreamItemKind.ANSWER_DELTA, 0),
            lambda: _item(StreamItemKind.ANSWER_DONE, 0, text_delta="x"),
            lambda: _item(
                StreamItemKind.TOOL_CALL_READY,
                0,
                correlation=StreamItemCorrelation(),
            ),
            lambda: _item(StreamItemKind.USAGE_COMPLETED, 0),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                usage={"input_tokens": 1},
            ),
            lambda: _item(StreamItemKind.STREAM_ERRORED, 0),
            lambda: _item(
                StreamItemKind.STREAM_ERRORED,
                0,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
            lambda: _item(
                StreamItemKind.STREAM_CLOSED,
                0,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                stream_session_id="",
            ),
            lambda: _item(StreamItemKind.STREAM_STARTED, -1),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                metadata=[],  # type: ignore[arg-type]
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                provider_family="",
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                provider_event_type="",
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                timestamp="now",  # type: ignore[arg-type]
            ),
        )

        for build_item in bad_items:
            with self.subTest(build_item=build_item):
                with self.assertRaises(AssertionError):
                    build_item()

    def test_correlation_rejects_malformed_fields(self) -> None:
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(provider_request_id="")
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(node_id=1)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(parent_sequence="0")  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(parent_sequence=-1)

    def test_golden_trace_rejects_malformed_metadata(self) -> None:
        valid_items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _stream_errored(1),
        )

        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="", items=valid_items)
        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="x", format_version=0, items=valid_items)
        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="x", items=list(valid_items))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="x", description="", items=valid_items)
        with self.assertRaises(StreamValidationError):
            StreamGoldenTrace(name="x", items=valid_items[:1])

    def test_sequence_validator_rejects_terminal_and_order_errors(
        self,
    ) -> None:
        cases = (
            (),
            (_item(StreamItemKind.STREAM_STARTED, 0),),
            (
                _item(StreamItemKind.ANSWER_DELTA, 0, text_delta="early"),
                _stream_errored(1),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.STREAM_STARTED, 1),
                _stream_errored(2),
            ),
            (_item(StreamItemKind.STREAM_STARTED, 0), _stream_errored(0)),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_ERRORED,
                    1,
                    run_id="run-2",
                    terminal_outcome=StreamTerminalOutcome.ERRORED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_DIAGNOSTIC,
                    1,
                    correlation=StreamItemCorrelation(parent_sequence=1),
                    text_delta="diagnostic",
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(
                    StreamItemKind.STREAM_CANCELLED,
                    2,
                    terminal_outcome=StreamTerminalOutcome.CANCELLED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.STREAM_DIAGNOSTIC, 2, text_delta="late"),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.STREAM_CLOSED, 1),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.STREAM_CLOSED, 2),
                _item(StreamItemKind.STREAM_CLOSED, 3),
            ),
            (_item(StreamItemKind.STREAM_STARTED, 0), _stream_completed(1)),
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(StreamValidationError):
                    validate_canonical_stream_items(items)

    def test_sequence_validator_rejects_channel_boundary_errors(self) -> None:
        tool = StreamItemCorrelation(tool_call_id="tool-1")
        continuation = StreamItemCorrelation(
            model_continuation_id="continuation-1"
        )
        cases = (
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="open"),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.REASONING_DELTA, 1, text_delta="open"),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="open"),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    2,
                    usage={"input_tokens": 1},
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="done"),
                _item(StreamItemKind.ANSWER_DONE, 2),
                _item(StreamItemKind.ANSWER_DONE, 3),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.REASONING_DELTA, 1, text_delta="done"),
                _item(StreamItemKind.REASONING_DONE, 2),
                _item(StreamItemKind.REASONING_DONE, 3),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DONE, 1),
                _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late"),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.REASONING_DONE, 1),
                _item(StreamItemKind.REASONING_DELTA, 2, text_delta="late"),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_UPDATE,
                    2,
                    usage={"input_tokens": 2},
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    2,
                    usage={"input_tokens": 1},
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late"),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    2,
                    correlation=tool,
                    text_delta='{"late":true}',
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.FLOW_EVENT,
                    2,
                    correlation=StreamItemCorrelation(flow_run_id="flow-1"),
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(StreamItemKind.MODEL_CONTINUATION_STARTED, 2),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    1,
                    correlation=tool,
                    text_delta='{"x":1}',
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    1,
                    correlation=tool,
                    text_delta='{"x":1}',
                ),
                _item(StreamItemKind.TOOL_CALL_READY, 2, correlation=tool),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    1,
                    correlation=tool,
                    text_delta='{"x":1}',
                ),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    2,
                    usage={"input_tokens": 1},
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    1,
                    correlation=tool,
                    text_delta='{"x":1}',
                ),
                _item(StreamItemKind.TOOL_CALL_READY, 2, correlation=tool),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    3,
                    usage={"input_tokens": 1},
                ),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    1,
                    correlation=tool,
                    text_delta='{"x":1}',
                ),
                _item(StreamItemKind.TOOL_CALL_READY, 2, correlation=tool),
                _item(StreamItemKind.TOOL_CALL_READY, 3, correlation=tool),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    1,
                    correlation=tool,
                    text_delta='{"x":1}',
                ),
                _item(StreamItemKind.TOOL_CALL_READY, 2, correlation=tool),
                _item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    3,
                    correlation=tool,
                    text_delta='{"late":true}',
                ),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.TOOL_CALL_DONE, 1, correlation=tool),
                _item(StreamItemKind.TOOL_CALL_READY, 2, correlation=tool),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.TOOL_CALL_READY, 1, correlation=tool),
                _item(StreamItemKind.TOOL_CALL_DONE, 2, correlation=tool),
                _item(StreamItemKind.TOOL_CALL_DONE, 3, correlation=tool),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    1,
                    correlation=tool,
                    text_delta="early",
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    1,
                    correlation=tool,
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    1,
                    correlation=tool,
                ),
                _item(
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    2,
                    correlation=tool,
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    1,
                    correlation=tool,
                ),
                _item(
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    2,
                    correlation=tool,
                ),
                _item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    3,
                    correlation=tool,
                    text_delta="late",
                ),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    1,
                    correlation=tool,
                ),
                _item(
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    2,
                    correlation=tool,
                ),
                _item(
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                    3,
                    correlation=tool,
                ),
                _stream_errored(4),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.MODEL_CONTINUATION_STARTED,
                    1,
                    correlation=continuation,
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.MODEL_CONTINUATION_STARTED,
                    1,
                    correlation=continuation,
                ),
                _item(
                    StreamItemKind.MODEL_CONTINUATION_STARTED,
                    2,
                    correlation=continuation,
                ),
                _stream_errored(3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                    1,
                    correlation=continuation,
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.MODEL_CONTINUATION_STARTED, 1),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.MODEL_CONTINUATION_STARTED,
                    1,
                    correlation=continuation,
                ),
                _item(
                    StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                    2,
                    correlation=continuation,
                ),
                _item(
                    StreamItemKind.MODEL_CONTINUATION_ERROR,
                    3,
                    correlation=continuation,
                ),
                _stream_errored(4),
            ),
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(StreamValidationError):
                    validate_canonical_stream_items(items)

    def test_sequence_validator_rejects_done_before_content(self) -> None:
        cases = (
            (
                StreamItemKind.ANSWER_DONE,
                "answer done before content",
            ),
            (
                StreamItemKind.REASONING_DONE,
                "reasoning done before content",
            ),
        )

        for done_kind, message in cases:
            with self.subTest(done_kind=done_kind):
                with self.assertRaisesRegex(StreamValidationError, message):
                    validate_canonical_stream_items(
                        (
                            _item(StreamItemKind.STREAM_STARTED, 0),
                            _item(done_kind, 1),
                            _stream_errored(2),
                        )
                    )

    def test_sequence_validator_rejects_items_after_channel_terminal(
        self,
    ) -> None:
        cases = (
            (
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _item(
                        StreamItemKind.ANSWER_DELTA,
                        1,
                        text_delta="answer",
                    ),
                    _item(StreamItemKind.ANSWER_DONE, 2),
                    _item(
                        StreamItemKind.ANSWER_DELTA,
                        3,
                        text_delta="late",
                    ),
                    _stream_errored(4),
                ),
                "answer item emitted after answer done",
            ),
            (
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 1),
                    _stream_errored(2),
                ),
                "tool execution terminal before start",
            ),
            (
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _item(
                        StreamItemKind.MODEL_CONTINUATION_STARTED,
                        1,
                        correlation=StreamItemCorrelation(
                            model_continuation_id="continuation-1"
                        ),
                    ),
                    _item(
                        StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                        2,
                        correlation=StreamItemCorrelation(
                            model_continuation_id="continuation-1"
                        ),
                    ),
                    _item(
                        StreamItemKind.MODEL_CONTINUATION_STARTED,
                        3,
                        correlation=StreamItemCorrelation(
                            model_continuation_id="continuation-1"
                        ),
                    ),
                    _stream_errored(4),
                ),
                "model continuation item emitted after terminal item",
            ),
        )

        for items, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(StreamValidationError, message):
                    validate_canonical_stream_items(items)

    def test_tool_lifecycle_assembles_planned_order_observations(
        self,
    ) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                tool_call_id="tool-2",
                text_delta='{"city"',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                2,
                tool_call_id="tool-2",
                text_delta=':"Paris"}',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                3,
                tool_call_id="tool-2",
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                4,
                tool_call_id="tool-2",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                5,
                tool_call_id="tool-2",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                6,
                tool_call_id="tool-2",
                text_delta="city ",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                7,
                tool_call_id="tool-2",
                data={"percent": 50},
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                8,
                tool_call_id="tool-2",
                data={"message": "failed"},
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                9,
                tool_call_id="tool-1",
                text_delta='{"expression"',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                10,
                tool_call_id="tool-1",
                text_delta=':"2+2"}',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                11,
                tool_call_id="tool-1",
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                12,
                tool_call_id="tool-1",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                13,
                tool_call_id="tool-1",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                14,
                tool_call_id="tool-1",
                text_delta="4",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                15,
                tool_call_id="tool-1",
                data={"result": 4},
            ),
        )

        validated = validate_tool_lifecycle_items(
            items, planned_tool_call_ids=("tool-1", "tool-2")
        )
        emission_order_observations = assemble_tool_observations(validated)
        observations = assemble_tool_observations(
            validated, planned_tool_call_ids=("tool-1", "tool-2")
        )

        self.assertEqual(validated, items)
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in validated
                if item.kind
                in {
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                    StreamItemKind.TOOL_EXECUTION_CANCELLED,
                }
            ],
            ["tool-2", "tool-1"],
        )
        self.assertEqual(
            [item.tool_call_id for item in emission_order_observations],
            ["tool-2", "tool-1"],
        )
        self.assertEqual(
            [item.tool_call_id for item in observations], ["tool-1", "tool-2"]
        )
        self.assertEqual(observations[0].arguments, '{"expression":"2+2"}')
        self.assertEqual(observations[0].output, "4")
        self.assertIs(
            observations[0].terminal_kind,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
        )
        self.assertEqual(observations[0].terminal_data, {"result": 4})
        self.assertEqual(observations[1].arguments, '{"city":"Paris"}')
        self.assertEqual(observations[1].output, "city ")
        self.assertIs(
            observations[1].terminal_kind,
            StreamItemKind.TOOL_EXECUTION_ERROR,
        )
        self.assertEqual(observations[1].terminal_data, {"message": "failed"})

    def test_tool_lifecycle_allows_empty_tool_stream(self) -> None:
        self.assertEqual(validate_tool_lifecycle_items(()), ())
        self.assertEqual(assemble_tool_observations(()), ())

    def test_tool_lifecycle_rejects_malformed_planned_order(self) -> None:
        invalid_values = (
            lambda: validate_tool_lifecycle_items(
                (), planned_tool_call_ids=("",)
            ),
            lambda: validate_tool_lifecycle_items(
                (), planned_tool_call_ids=("tool-1", "tool-1")
            ),
            lambda: validate_tool_lifecycle_items(
                (), planned_tool_call_ids=(cast(Any, 1),)
            ),
        )

        for validate in invalid_values:
            with self.subTest(validate=validate):
                with self.assertRaises(AssertionError):
                    validate()

    def test_tool_lifecycle_observation_assembly_rejects_planned_mismatches(
        self,
    ) -> None:
        items = (
            _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
            _tool_item(StreamItemKind.TOOL_CALL_DONE, 1),
            _tool_item(StreamItemKind.TOOL_EXECUTION_STARTED, 2),
            _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),
        )
        cases = (
            (("tool-2",), "unexpected tool call id"),
            (("tool-1", "tool-2"), "planned tool call missing"),
        )

        for planned_tool_call_ids, message in cases:
            with self.subTest(planned_tool_call_ids=planned_tool_call_ids):
                with self.assertRaisesRegex(StreamValidationError, message):
                    assemble_tool_observations(
                        items,
                        planned_tool_call_ids=planned_tool_call_ids,
                    )

    def test_tool_lifecycle_rejects_invalid_transitions(self) -> None:
        complete_prefix = (
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                0,
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                1,
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                2,
            ),
        )
        cases = (
            (
                _tool_item(
                    StreamItemKind.TOOL_CALL_READY,
                    0,
                    tool_call_id="tool-2",
                ),
            ),
            (
                _tool_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    0,
                    text_delta="late",
                ),
                _tool_item(StreamItemKind.TOOL_CALL_READY, 1),
                _tool_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    2,
                    text_delta="later",
                ),
            ),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_CALL_READY, 1),
            ),
            (_tool_item(StreamItemKind.TOOL_CALL_DONE, 0),),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_CALL_DONE, 1),
                _tool_item(StreamItemKind.TOOL_CALL_DONE, 2),
            ),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_EXECUTION_STARTED, 1),
            ),
            complete_prefix
            + (_tool_item(StreamItemKind.TOOL_EXECUTION_STARTED, 3),),
            (
                _tool_item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    0,
                    text_delta="early",
                ),
            ),
            (_tool_item(StreamItemKind.TOOL_EXECUTION_PROGRESS, 0),),
            (_tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 0),),
            complete_prefix
            + (
                _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),
                _tool_item(StreamItemKind.TOOL_EXECUTION_ERROR, 4),
            ),
            complete_prefix
            + (
                _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),
                _tool_item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    4,
                    text_delta="late",
                ),
            ),
            complete_prefix
            + (
                _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),
                _tool_item(
                    StreamItemKind.TOOL_EXECUTION_PROGRESS,
                    4,
                    data={"category": "progress", "progress": 0.9},
                ),
            ),
            (
                _tool_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    0,
                    text_delta="partial",
                ),
            ),
            (_tool_item(StreamItemKind.TOOL_CALL_READY, 0),),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_CALL_DONE, 1),
            ),
            complete_prefix,
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(StreamValidationError):
                    validate_tool_lifecycle_items(items)

        with self.assertRaises(StreamValidationError):
            validate_tool_lifecycle_items(
                complete_prefix
                + (_tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),),
                planned_tool_call_ids=("tool-2",),
            )
        with self.assertRaises(StreamValidationError):
            validate_tool_lifecycle_items(
                complete_prefix
                + (_tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),),
                planned_tool_call_ids=("tool-1", "tool-2"),
            )

    def test_tool_lifecycle_rejects_live_items_after_completion(self) -> None:
        completed = (
            _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
            _tool_item(StreamItemKind.TOOL_CALL_DONE, 1),
            _tool_item(StreamItemKind.TOOL_EXECUTION_STARTED, 2),
            _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),
        )
        late_items = (
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                4,
                text_delta="late",
                data={"category": "stdout", "content": "late"},
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                4,
                data={"category": "progress", "progress": 0.9},
            ),
        )

        for item in late_items:
            with self.subTest(kind=item.kind):
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "tool execution item emitted after terminal item",
                ):
                    validate_tool_lifecycle_items(completed + (item,))

    def test_accumulator_separates_answer_from_other_channels(self) -> None:
        tool = StreamItemCorrelation(tool_call_id="tool-1")
        usage = {"input_tokens": 3, "output_tokens": 2}
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="Hello"),
            _item(StreamItemKind.REASONING_DELTA, 2, text_delta="think"),
            _item(StreamItemKind.REASONING_DONE, 3),
            _item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                4,
                correlation=tool,
                text_delta='{"city"',
            ),
            _item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                5,
                correlation=tool,
                text_delta=':"Paris"}',
            ),
            _item(StreamItemKind.TOOL_CALL_READY, 6, correlation=tool),
            _item(StreamItemKind.TOOL_CALL_DONE, 7, correlation=tool),
            _item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                8,
                correlation=tool,
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                9,
                correlation=tool,
                text_delta="tool ",
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                10,
                correlation=tool,
                text_delta="output",
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                11,
                correlation=tool,
            ),
            _item(StreamItemKind.ANSWER_DELTA, 12, text_delta=" world"),
            _item(StreamItemKind.ANSWER_DONE, 13),
            _item(
                StreamItemKind.FLOW_EVENT,
                14,
                data={"node": "done"},
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                15,
                text_delta="internal",
                visibility=StreamVisibility.DIAGNOSTIC,
            ),
            _item(
                StreamItemKind.USAGE_UPDATE,
                16,
                usage={"input_tokens": 3},
            ),
            _item(StreamItemKind.USAGE_COMPLETED, 17, usage=usage),
            _stream_completed(18),
            _item(StreamItemKind.STREAM_CLOSED, 19),
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.items, items)
        self.assertEqual(accumulator.answer_text, "Hello world")
        self.assertEqual(accumulator.reasoning_text, "think")
        self.assertEqual(
            accumulator.tool_call_arguments, {"tool-1": '{"city":"Paris"}'}
        )
        self.assertEqual(
            accumulator.tool_execution_outputs, {"tool-1": "tool output"}
        )
        self.assertEqual(accumulator.diagnostics, (items[15],))
        self.assertEqual(accumulator.flow_items, (items[14],))
        self.assertEqual(accumulator.usage_items, (items[16], items[17]))
        self.assertEqual(
            accumulator.control_items,
            (items[0], items[15], items[18], items[19]),
        )
        self.assertEqual(accumulator.final_usage, usage)
        self.assertIs(
            accumulator.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertTrue(accumulator.closed)

        copy = accumulator.tool_call_arguments
        copy["tool-1"] = "changed"
        self.assertEqual(
            accumulator.tool_call_arguments, {"tool-1": '{"city":"Paris"}'}
        )

    def test_accumulator_allows_atomic_completion_usage(self) -> None:
        usage: object = {}
        accumulator = CanonicalStreamAccumulator()
        returned = accumulator.add_many(
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_COMPLETED,
                    1,
                    usage=usage,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            )
        )

        self.assertIs(returned, accumulator)
        self.assertEqual(accumulator.final_usage, usage)
        self.assertEqual(accumulator.validate_complete(), accumulator.items)

    def test_accumulator_rejects_incremental_invalid_sequences(self) -> None:
        cases = (
            (_item(StreamItemKind.ANSWER_DELTA, 0, text_delta="early"),),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.STREAM_STARTED, 1),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DELTA, 0, text_delta="again"),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_ERRORED,
                    1,
                    turn_id="turn-2",
                    terminal_outcome=StreamTerminalOutcome.ERRORED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_COMPLETED,
                    1,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late"),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(
                    StreamItemKind.STREAM_CANCELLED,
                    2,
                    terminal_outcome=StreamTerminalOutcome.CANCELLED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.STREAM_CLOSED, 1),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.STREAM_CLOSED, 2),
                _item(StreamItemKind.STREAM_CLOSED, 3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    2,
                    usage={"input_tokens": 1},
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_UPDATE,
                    2,
                    usage={"input_tokens": 2},
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late"),
            ),
        )

        for items in cases:
            with self.subTest(items=items):
                accumulator = CanonicalStreamAccumulator()
                with self.assertRaises(StreamValidationError):
                    accumulator.add_many(items)

    def test_accumulator_rejects_stream_terminal_with_open_channels(
        self,
    ) -> None:
        tool = StreamItemCorrelation(tool_call_id="tool-1")
        continuation = StreamItemCorrelation(
            model_continuation_id="continuation-1"
        )
        open_channel_cases = (
            (
                "answer",
                (
                    _item(
                        StreamItemKind.ANSWER_DELTA,
                        1,
                        text_delta="answer",
                    ),
                ),
                "answer channel missing done",
            ),
            (
                "reasoning",
                (
                    _item(
                        StreamItemKind.REASONING_DELTA,
                        1,
                        text_delta="reasoning",
                    ),
                ),
                "reasoning channel missing done",
            ),
            (
                "tool-call missing ready",
                (
                    _item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        1,
                        correlation=tool,
                        text_delta='{"city":"Paris"}',
                    ),
                ),
                "tool call tool-1 missing ready",
            ),
            (
                "tool-call missing done",
                (
                    _item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        1,
                        correlation=tool,
                        text_delta='{"city":"Paris"}',
                    ),
                    _item(
                        StreamItemKind.TOOL_CALL_READY,
                        2,
                        correlation=tool,
                    ),
                ),
                "tool call tool-1 missing done",
            ),
            (
                "tool execution",
                (
                    _item(
                        StreamItemKind.TOOL_EXECUTION_STARTED,
                        1,
                        correlation=tool,
                    ),
                ),
                "tool execution tool-1 missing terminal",
            ),
            (
                "model continuation",
                (
                    _item(
                        StreamItemKind.MODEL_CONTINUATION_STARTED,
                        1,
                        correlation=continuation,
                    ),
                ),
                "model continuation continuation-1 missing terminal",
            ),
        )
        terminal_cases = (
            (
                "completed",
                lambda sequence: _item(
                    StreamItemKind.STREAM_COMPLETED,
                    sequence,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ),
            ("errored", _stream_errored),
            (
                "cancelled",
                lambda sequence: _item(
                    StreamItemKind.STREAM_CANCELLED,
                    sequence,
                    terminal_outcome=StreamTerminalOutcome.CANCELLED,
                ),
            ),
        )

        for channel_label, open_items, message in open_channel_cases:
            for terminal_label, terminal_factory in terminal_cases:
                with self.subTest(
                    channel=channel_label,
                    terminal=terminal_label,
                ):
                    accumulator = CanonicalStreamAccumulator()
                    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
                    for open_item in open_items:
                        accumulator.add(open_item)

                    with self.assertRaisesRegex(
                        StreamValidationError, message
                    ):
                        accumulator.add(terminal_factory(len(open_items) + 1))
                    self.assertIsNone(accumulator.terminal_outcome)
                    self.assertIsNone(accumulator.terminal_item)

    def test_accumulator_rejects_bad_item_and_incomplete_stream(self) -> None:
        accumulator = CanonicalStreamAccumulator()

        with self.assertRaises(AssertionError):
            accumulator.add("bad")  # type: ignore[arg-type]

        accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
        with self.assertRaises(StreamValidationError):
            accumulator.validate_complete()

    def test_stream_negative_acceptance_edges(self) -> None:
        with self.assertRaisesRegex(
            StreamValidationError, "duplicate stream terminal item"
        ):
            validate_canonical_stream_items(
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _stream_errored(1),
                    _item(
                        StreamItemKind.STREAM_CANCELLED,
                        2,
                        terminal_outcome=StreamTerminalOutcome.CANCELLED,
                    ),
                )
            )
        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            validate_canonical_stream_items(
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _stream_errored(1),
                    _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late"),
                )
            )
        with self.assertRaisesRegex(
            StreamValidationError,
            "stream missing terminal outcome",
        ):
            validate_canonical_stream_items(
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="open"),
                )
            )
        with self.assertRaises(AssertionError):
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                1,
                text_delta="missing id",
            )

        malformed_items = run(
            _collect_local_items(
                _local_text_chunks(("<tool_call>not-json</tool_call>",))
            )
        )
        diagnostic = next(
            item
            for item in malformed_items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        self.assertEqual(diagnostic.data["code"], "tool_call.malformed")
        self.assertFalse(
            any(
                item.kind is StreamItemKind.TOOL_CALL_READY
                for item in malformed_items
            )
        )

        retention_policy = StreamRetentionPolicy(accumulator_item_limit=1)
        accumulator = CanonicalStreamAccumulator(
            retention_policy=retention_policy
        )
        accumulator.add_many(
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="kept"),
                _item(StreamItemKind.ANSWER_DONE, 2),
                _item(
                    StreamItemKind.STREAM_COMPLETED,
                    3,
                    usage={"output_tokens": 1},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            )
        )
        self.assertEqual(
            [item.kind for item in accumulator.items],
            [StreamItemKind.STREAM_COMPLETED],
        )
        self.assertEqual(accumulator.answer_text, "kept")
        self.assertEqual(accumulator.final_usage, {"output_tokens": 1})
        with self.assertRaises(AssertionError):
            StreamRetentionPolicy(accumulator_item_limit=0)

    def test_consumer_projection_preserves_canonical_fields(self) -> None:
        correlation = StreamItemCorrelation(
            provider_request_id="provider-1",
            tool_call_id="tool-1",
            parent_sequence=1,
            protocol_item_id="protocol-1",
        )
        item = _item(
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            2,
            stream_session_id="session",
            run_id="run",
            turn_id="turn",
            correlation=correlation,
            text_delta="chunk",
            data={"structured": True},
            metadata={"source": "tool"},
            provider_family="openai",
            provider_event_type="tool.output",
        )

        projection = project_canonical_stream_item(item)

        self.assertEqual(projection.stream_session_id, "session")
        self.assertEqual(projection.run_id, "run")
        self.assertEqual(projection.turn_id, "turn")
        self.assertEqual(projection.sequence, 2)
        self.assertIs(projection.kind, StreamItemKind.TOOL_EXECUTION_OUTPUT)
        self.assertIs(projection.channel, StreamChannel.TOOL_EXECUTION)
        self.assertIs(projection.correlation, correlation)
        self.assertEqual(projection.tool_call_id, "tool-1")
        self.assertEqual(projection.text_delta, "chunk")
        self.assertEqual(projection.data, {"structured": True})
        self.assertEqual(projection.metadata, {"source": "tool"})
        self.assertEqual(projection.provider_family, "openai")
        self.assertEqual(projection.provider_event_type, "tool.output")
        self.assertFalse(projection.is_stream_terminal)

        item.metadata["source"] = "mutated"
        self.assertEqual(projection.metadata, {"source": "tool"})

    def test_canonical_item_from_projection_preserves_projected_fields(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(tool_call_id="tool-1")
        projection = project_canonical_stream_item(
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                2,
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                correlation=correlation,
                text_delta="chunk",
                data={"structured": True},
                metadata={"source": "tool"},
                provider_family="openai",
                provider_event_type="tool.output",
            )
        )

        item = canonical_item_from_consumer_projection(projection)

        self.assertEqual(item.stream_session_id, "session")
        self.assertEqual(item.run_id, "run")
        self.assertEqual(item.turn_id, "turn")
        self.assertEqual(item.sequence, 2)
        self.assertIs(item.kind, StreamItemKind.TOOL_EXECUTION_OUTPUT)
        self.assertIs(item.channel, StreamChannel.TOOL_EXECUTION)
        self.assertIs(item.correlation, correlation)
        self.assertEqual(item.text_delta, "chunk")
        self.assertEqual(item.data, {"structured": True})
        self.assertEqual(item.metadata, {"source": "tool"})
        self.assertEqual(item.provider_family, "openai")
        self.assertEqual(item.provider_event_type, "tool.output")

        projection.metadata["source"] = "mutated"
        self.assertEqual(item.metadata, {"source": "tool"})

    def test_consumer_projection_preserves_usage_and_terminal_state(
        self,
    ) -> None:
        item = _item(
            StreamItemKind.STREAM_COMPLETED,
            1,
            usage={"output_tokens": 3},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

        projection = project_canonical_stream_item(item)

        self.assertEqual(projection.usage, {"output_tokens": 3})
        self.assertIs(
            projection.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertTrue(projection.is_stream_terminal)

    def test_consumer_projection_helpers_classify_display_items(self) -> None:
        answer = _item(
            StreamItemKind.ANSWER_DELTA,
            1,
            text_delta="answer",
        )
        reasoning = project_canonical_stream_item(
            _item(
                StreamItemKind.REASONING_DELTA,
                2,
                text_delta="reason",
            )
        )
        tool_call = project_canonical_stream_item(
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                3,
                text_delta='{"city"',
            )
        )
        done = _item(StreamItemKind.ANSWER_DONE, 4)

        self.assertEqual(stream_projection_text_delta(answer), "answer")
        self.assertEqual(stream_projection_text_delta(reasoning), "reason")
        self.assertEqual(stream_projection_text_delta(tool_call), '{"city"')
        self.assertIsNone(stream_projection_text_delta(done))
        self.assertFalse(stream_projection_is_reasoning(answer))
        self.assertTrue(stream_projection_is_reasoning(reasoning))
        self.assertFalse(stream_projection_is_tool_call(reasoning))
        self.assertTrue(stream_projection_is_tool_call(tool_call))

    def test_consumer_projection_helpers_reject_invalid_items(self) -> None:
        for helper in (
            stream_projection_text_delta,
            stream_projection_is_reasoning,
            stream_projection_is_tool_call,
        ):
            with self.assertRaises(AssertionError):
                helper("bad")  # type: ignore[arg-type]

    def test_consumer_projection_validates_invalid_arguments(self) -> None:
        item = _item(StreamItemKind.STREAM_STARTED, 0)

        with self.assertRaises(AssertionError):
            project_canonical_stream_item("bad")  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.REASONING,
                correlation=StreamItemCorrelation(),
            )
        with self.assertRaises(AssertionError):
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(),
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
        invalid_payloads = (
            lambda: StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
            ),
            lambda: StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
                text_delta="late",
            ),
            lambda: StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(),
                text_delta="{}",
            ),
            lambda: StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                correlation=StreamItemCorrelation(),
            ),
            lambda: StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(),
            ),
        )
        for build_projection in invalid_payloads:
            with self.subTest(build_projection=build_projection):
                with self.assertRaises(AssertionError):
                    build_projection()
        self.assertEqual(
            project_canonical_stream_item(item).stream_session_id,
            "stream-1",
        )

    def test_consumer_projection_iterator_preserves_order_losslessly(
        self,
    ) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="a"),
            _item(StreamItemKind.REASONING_DELTA, 2, text_delta="r"),
            _item(StreamItemKind.REASONING_DONE, 3),
            _item(StreamItemKind.ANSWER_DONE, 4),
            _item(
                StreamItemKind.USAGE_COMPLETED,
                5,
                usage={"output_tokens": 1},
            ),
            _stream_completed(6),
            _item(StreamItemKind.STREAM_CLOSED, 7),
        )

        async def stream() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

        projections = run(_collect_projection_items(stream()))

        self.assertEqual(
            [item.sequence for item in projections], list(range(8))
        )
        self.assertEqual(
            [item.kind for item in projections],
            [item.kind for item in items],
        )
        self.assertEqual(
            [item.text_delta for item in projections],
            [None, "a", "r", None, None, None, None, None],
        )
        self.assertEqual(projections[5].usage, {"output_tokens": 1})
        self.assertIs(
            projections[6].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    def test_consumer_projection_iterator_rejects_sequence_gap(
        self,
    ) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.ANSWER_DELTA,
                2,
                text_delta="dropped predecessor",
            ),
        )

        async def stream() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

        with self.assertRaisesRegex(
            StreamValidationError,
            "lossless consumer stream sequence gap",
        ):
            run(_collect_projection_items(stream()))

    def test_consumer_projection_per_item_overhead_within_budget(self) -> None:
        count = 1000
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            *(
                _item(
                    StreamItemKind.ANSWER_DELTA,
                    sequence,
                    text_delta="x",
                )
                for sequence in range(1, count + 1)
            ),
            _item(StreamItemKind.ANSWER_DONE, count + 1),
            _item(
                StreamItemKind.STREAM_COMPLETED,
                count + 2,
                usage={"output_tokens": count},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )
        budget = StreamPerformanceBudget()

        async def stream() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

        started = perf_counter()
        projections = run(_collect_projection_items(stream()))
        elapsed_us = (perf_counter() - started) * 1_000_000

        self.assertEqual(len(projections), count + 3)
        self.assertLessEqual(
            elapsed_us / len(projections),
            budget.per_item_overhead_us,
        )

    def test_consumer_projection_iterator_rejects_invalid_streams(
        self,
    ) -> None:
        cases = (
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_completed(1),
                _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late"),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_completed(1),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DELTA, 0, text_delta="repeat"),
            ),
            (_item(StreamItemKind.STREAM_STARTED, 0),),
            (),
        )

        for items in cases:
            with self.subTest(items=items):

                async def stream() -> AsyncIterator[CanonicalStreamItem]:
                    for item in items:
                        yield item

                with self.assertRaises(StreamValidationError):
                    run(_collect_projection_items(stream()))

    def test_consumer_projection_iterator_can_skip_order_validation(
        self,
    ) -> None:
        async def stream() -> AsyncIterator[CanonicalStreamItem]:
            yield _item(StreamItemKind.ANSWER_DELTA, 0, text_delta="early")

        projections = run(
            _collect_projection_items(stream(), validate_order=False)
        )

        self.assertEqual(projections[0].text_delta, "early")

    def test_consumer_projection_iterator_closes_wrapped_stream(self) -> None:
        class CloseableStream:
            def __init__(self) -> None:
                self.closed = False
                self._sequence = 0

            def __aiter__(self) -> "CloseableStream":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 1:
                    raise StopAsyncIteration
                sequence = self._sequence
                self._sequence += 1
                if sequence == 0:
                    return _item(StreamItemKind.STREAM_STARTED, sequence)
                return _item(
                    StreamItemKind.ANSWER_DELTA,
                    sequence,
                    text_delta="late",
                )

            async def aclose(self) -> None:
                self.closed = True

        stream = CloseableStream()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        run(projections.aclose())

        self.assertTrue(stream.closed)

    def test_consumer_projection_iterator_closes_active_iterator(
        self,
    ) -> None:
        class StreamIterator:
            def __init__(self) -> None:
                self.closed = False
                self._sequence = 0

            def __aiter__(self) -> "StreamIterator":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 1:
                    raise StopAsyncIteration
                sequence = self._sequence
                self._sequence += 1
                if sequence == 0:
                    return _item(StreamItemKind.STREAM_STARTED, sequence)
                return _item(
                    StreamItemKind.ANSWER_DELTA,
                    sequence,
                    text_delta="late",
                )

            async def aclose(self) -> None:
                self.closed = True

        class StreamIterable:
            def __init__(self) -> None:
                self.iterator = StreamIterator()

            def __aiter__(self) -> StreamIterator:
                return self.iterator

        stream = StreamIterable()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        run(projections.aclose())

        self.assertTrue(stream.iterator.closed)

    def test_consumer_projection_iterator_closes_outer_after_iterator_error(
        self,
    ) -> None:
        class StreamIterator:
            def __init__(self) -> None:
                self._sequence = 0

            def __aiter__(self) -> "StreamIterator":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 0:
                    raise StopAsyncIteration
                self._sequence += 1
                return _item(StreamItemKind.STREAM_STARTED, 0)

            async def aclose(self) -> None:
                raise RuntimeError("iterator close failed")

        class StreamIterable:
            def __init__(self) -> None:
                self.close_count = 0
                self.iterator = StreamIterator()

            def __aiter__(self) -> StreamIterator:
                return self.iterator

            async def aclose(self) -> None:
                self.close_count += 1

        stream = StreamIterable()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        with self.assertRaisesRegex(RuntimeError, "iterator close failed"):
            run(projections.aclose())

        self.assertEqual(stream.close_count, 1)

    def test_consumer_projection_iterator_preserves_close_cancellation(
        self,
    ) -> None:
        class StreamIterator:
            def __init__(self) -> None:
                self._sequence = 0

            def __aiter__(self) -> "StreamIterator":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 0:
                    raise StopAsyncIteration
                self._sequence += 1
                return _item(StreamItemKind.STREAM_STARTED, 0)

            async def aclose(self) -> None:
                raise CancelledError()

        class StreamIterable:
            def __init__(self) -> None:
                self.close_count = 0
                self.iterator = StreamIterator()

            def __aiter__(self) -> StreamIterator:
                return self.iterator

            async def aclose(self) -> None:
                self.close_count += 1

        stream = StreamIterable()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        with self.assertRaises(CancelledError):
            run(projections.aclose())

        self.assertEqual(stream.close_count, 1)

    def test_consumer_projection_iterator_reports_multiple_close_errors(
        self,
    ) -> None:
        class StreamIterator:
            def __init__(self) -> None:
                self._sequence = 0

            def __aiter__(self) -> "StreamIterator":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 0:
                    raise StopAsyncIteration
                self._sequence += 1
                return _item(StreamItemKind.STREAM_STARTED, 0)

            async def aclose(self) -> None:
                raise RuntimeError("iterator close failed")

        class StreamIterable:
            def __init__(self) -> None:
                self.iterator = StreamIterator()

            def __aiter__(self) -> StreamIterator:
                return self.iterator

            async def aclose(self) -> None:
                raise RuntimeError("iterable close failed")

        stream = StreamIterable()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        with self.assertRaises(BaseExceptionGroup) as context:
            run(projections.aclose())

        self.assertEqual(
            [str(error) for error in context.exception.exceptions],
            ["iterator close failed", "iterable close failed"],
        )

    def test_consumer_projection_iterator_accepts_sync_close(self) -> None:
        class CloseableStream:
            def __init__(self) -> None:
                self.close_count = 0
                self._sequence = 0

            def __aiter__(self) -> "CloseableStream":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 0:
                    raise StopAsyncIteration
                self._sequence += 1
                return _item(StreamItemKind.STREAM_STARTED, 0)

            def aclose(self) -> None:
                self.close_count += 1

        stream = CloseableStream()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        run(projections.aclose())

        self.assertEqual(stream.close_count, 1)

    def test_consumer_projection_iterator_rejects_bad_sync_close_result(
        self,
    ) -> None:
        class CloseableStream:
            def __init__(self) -> None:
                self._sequence = 0

            def __aiter__(self) -> "CloseableStream":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 0:
                    raise StopAsyncIteration
                self._sequence += 1
                return _item(StreamItemKind.STREAM_STARTED, 0)

            def aclose(self) -> object:
                return object()

        stream = CloseableStream()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        with self.assertRaises(AssertionError):
            run(projections.aclose())

    def test_consumer_projection_iterator_rejects_bad_async_close_result(
        self,
    ) -> None:
        class CloseableStream:
            def __init__(self) -> None:
                self._sequence = 0

            def __aiter__(self) -> "CloseableStream":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                if self._sequence > 0:
                    raise StopAsyncIteration
                self._sequence += 1
                return _item(StreamItemKind.STREAM_STARTED, 0)

            async def aclose(self) -> object:
                return object()

        stream = CloseableStream()
        projections = iter_stream_consumer_projections(stream)

        self.assertIs(
            run(projections.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        with self.assertRaises(AssertionError):
            run(projections.aclose())

    def test_text_generation_stream_base_contract(self) -> None:
        probe = _StreamProbe()

        with self.assertRaises(AssertionError):
            probe.__aiter__()
        with self.assertRaises(NotImplementedError):
            probe()
        with self.assertRaises(NotImplementedError):
            run(probe.__anext__())

        probe._generator = _single_canonical_generator()
        self.assertIs(probe.__aiter__(), probe)
        self.assertIs(
            probe.canonical_stream(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                provider_family="openai",
                capabilities=StreamProviderCapabilities(
                    backend=StreamProducerBackend.HOSTED
                ),
                close_after_terminal=False,
            ),
            probe,
        )

    def test_single_stream_iterates_content_and_resets(self) -> None:
        stream = TextGenerationSingleStream(
            "one",
            provider_family="openai",
            usage={"output_tokens": 1},
        )

        self.assertEqual(stream.content, "one")
        self.assertEqual(stream.provider_family, "openai")
        self.assertEqual(stream.usage, {"output_tokens": 1})
        self.assertIs(stream(), stream)
        self.assertIs(
            run(stream.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        self.assertEqual(
            run(stream.__anext__()).text_delta,
            "one",
        )
        self.assertIs(
            run(stream.__anext__()).kind,
            StreamItemKind.ANSWER_DONE,
        )
        terminal = run(stream.__anext__())
        self.assertIs(terminal.kind, StreamItemKind.STREAM_COMPLETED)
        self.assertEqual(terminal.usage, {"output_tokens": 1})
        self.assertIs(
            run(stream.__anext__()).kind,
            StreamItemKind.STREAM_CLOSED,
        )
        with self.assertRaises(StopAsyncIteration):
            run(stream.__anext__())

        self.assertIs(stream.__aiter__(), stream)
        self.assertIs(
            run(stream.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        canonical_items = run(
            _collect_stream_items(
                stream.canonical_stream(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                )
            )
        )
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            {item.stream_session_id for item in canonical_items},
            {"stream"},
        )
        self.assertEqual({item.run_id for item in canonical_items}, {"run"})
        self.assertEqual({item.turn_id for item in canonical_items}, {"turn"})

    def test_single_stream_final_text_uses_canonical_accumulator(
        self,
    ) -> None:
        stream = TextGenerationSingleStream(
            "answer",
            usage={"output_tokens": 1},
        )

        self.assertEqual(stream.final_text, "answer")
        self.assertEqual(run(stream.to_str()), "answer")
        self.assertEqual(stream.accumulator.answer_text, "answer")
        self.assertEqual(
            [item.kind for item in stream.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

        with self.assertRaises(AssertionError):
            TextGenerationSingleStream(cast(Any, Token(token="legacy")))

    def test_provider_capabilities_validate_and_serialize_metadata(
        self,
    ) -> None:
        capabilities = StreamProviderCapabilities(
            backend=StreamProducerBackend.HOSTED,
            provider_family=ProviderFamily.OPENAI,
            supports_reasoning=True,
            supports_tool_calls=True,
            supports_usage=True,
            supports_terminal_events=True,
            supports_cancellation=True,
            max_queue_depth=64,
            max_item_bytes=1024,
        )

        self.assertEqual(capabilities.normalized_provider_family, "openai")
        self.assertEqual(
            capabilities.to_metadata(),
            {
                "backend": "hosted",
                "provider_family": "openai",
                "supports_reasoning": True,
                "supports_tool_calls": True,
                "supports_usage": True,
                "supports_terminal_events": True,
                "supports_cancellation": True,
                "max_queue_depth": 64,
                "max_item_bytes": 1024,
            },
        )

        invalid_capabilities = (
            lambda: StreamProviderCapabilities(
                backend="hosted",  # type: ignore[arg-type]
            ),
            lambda: StreamProviderCapabilities(
                backend=StreamProducerBackend.LOCAL,
                supports_usage="yes",  # type: ignore[arg-type]
            ),
            lambda: StreamProviderCapabilities(
                backend=StreamProducerBackend.LOCAL,
                max_queue_depth=0,
            ),
            lambda: StreamProviderCapabilities(
                backend=StreamProducerBackend.LOCAL,
                max_item_bytes=-1,
            ),
        )
        for build_capabilities in invalid_capabilities:
            with self.subTest(build_capabilities=build_capabilities):
                with self.assertRaises(AssertionError):
                    build_capabilities()

    def test_provider_event_rejects_invalid_payload_boundaries(self) -> None:
        invalid_events = (
            lambda: StreamProviderEvent(
                kind="answer.delta",  # type: ignore[arg-type]
            ),
            lambda: StreamProviderEvent(kind=StreamItemKind.STREAM_STARTED),
            lambda: StreamProviderEvent(kind=StreamItemKind.STREAM_CLOSED),
            lambda: StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta=object(),  # type: ignore[arg-type]
            ),
            lambda: StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                metadata=[],  # type: ignore[arg-type]
            ),
            lambda: StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                provider_event_type="",
            ),
        )

        for build_event in invalid_events:
            with self.subTest(build_event=build_event):
                with self.assertRaises(AssertionError):
                    build_event()

    def test_provider_stream_normalizer_assigns_identity_and_metadata(
        self,
    ) -> None:
        capabilities = StreamProviderCapabilities(
            backend=StreamProducerBackend.HOSTED,
            provider_family=ProviderFamily.OPENAI,
            supports_reasoning=True,
            supports_tool_calls=True,
            supports_usage=True,
            supports_terminal_events=True,
            supports_cancellation=True,
            max_queue_depth=32,
        )
        provider_payload = {"native": {"id": "chunk-1"}}
        events = (
            StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta="hello ",
                provider_payload=provider_payload,
                provider_event_type="response.output_text.delta",
            ),
            StreamProviderEvent(
                kind=StreamItemKind.REASONING_DELTA,
                text_delta="private",
                visibility=StreamVisibility.PRIVATE,
                provider_event_type="response.reasoning_text.delta",
            ),
            StreamProviderEvent(
                kind=StreamItemKind.USAGE_COMPLETED,
                usage={"input_tokens": 1, "output_tokens": 2},
                provider_event_type="response.completed",
            ),
            StreamProviderEvent(
                kind=StreamItemKind.STREAM_COMPLETED,
                provider_event_type="response.completed",
            ),
        )
        items = run(
            _collect_provider_items(
                _provider_events(events),
                capabilities=capabilities,
                provider_family=None,
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(8)))
        self.assertEqual(
            {item.stream_session_id for item in items},
            {"provider-stream"},
        )
        self.assertEqual({item.run_id for item in items}, {"provider-run"})
        self.assertEqual({item.turn_id for item in items}, {"provider-turn"})
        self.assertEqual({item.provider_family for item in items}, {"openai"})
        self.assertEqual(
            items[0].metadata["capabilities"]["backend"], "hosted"
        )
        self.assertEqual(items[1].provider_payload, provider_payload)
        self.assertEqual(
            items[1].provider_event_type, "response.output_text.delta"
        )
        self.assertIs(items[2].visibility, StreamVisibility.PRIVATE)
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "hello ")
        self.assertEqual(accumulator.reasoning_text, "private")
        self.assertEqual(
            accumulator.final_usage,
            {"input_tokens": 1, "output_tokens": 2},
        )

    def test_provider_stream_normalizer_reuses_empty_internal_correlation(
        self,
    ) -> None:
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="answer",
                        ),
                    )
                ),
                provider_family=None,
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[0].correlation, StreamItemCorrelation())
        self.assertIs(items[2].correlation, items[0].correlation)
        self.assertIs(items[3].correlation, items[0].correlation)
        self.assertIs(items[4].correlation, items[0].correlation)
        self.assertIsNot(items[1].correlation, items[0].correlation)

    def test_normalizers_emit_done_items_at_deterministic_boundaries(
        self,
    ) -> None:
        native_items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="native",
                        ),
                        StreamProviderEvent(kind=StreamItemKind.ANSWER_DONE),
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_COMPLETED,
                            usage={"output_tokens": 1},
                        ),
                    )
                )
            )
        )
        self.assertEqual(
            [item.kind for item in native_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

        final_usage_items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="answer",
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.REASONING_DELTA,
                            text_delta="reason",
                            visibility=StreamVisibility.PRIVATE,
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.USAGE_COMPLETED,
                            usage={"output_tokens": 2},
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_COMPLETED,
                        ),
                    )
                )
            )
        )
        self.assertEqual(
            [item.kind for item in final_usage_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

        exhausted_items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="exhausted",
                        ),
                    )
                )
            )
        )
        self.assertEqual(
            [item.kind for item in exhausted_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

        parsed_reasoning_items = run(
            _collect_local_items(_local_text_chunks(("<think>r</think>a",)))
        )
        self.assertLess(
            _first_sequence(
                parsed_reasoning_items, StreamItemKind.REASONING_DONE
            ),
            _first_sequence(
                parsed_reasoning_items, StreamItemKind.ANSWER_DELTA
            ),
        )

        parsed_tool_items = run(
            _collect_local_items(
                _local_text_chunks(("<tool_call>{}</tool_call> after",))
            )
        )
        self.assertLess(
            _first_sequence(parsed_tool_items, StreamItemKind.TOOL_CALL_DONE),
            _last_sequence(parsed_tool_items, StreamItemKind.ANSWER_DELTA),
        )

    def test_provider_stream_normalizer_preserves_terminal_event_context(
        self,
    ) -> None:
        error_correlation = StreamItemCorrelation(
            provider_request_id="request-1",
            tool_call_id="tool-1",
        )
        completed_correlation = StreamItemCorrelation(
            provider_request_id="request-2",
        )
        cancelled_correlation = StreamItemCorrelation(
            provider_request_id="request-3",
        )

        error_items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_ERRORED,
                            data={"message": "failed"},
                            correlation=error_correlation,
                            visibility=StreamVisibility.DIAGNOSTIC,
                            metadata={"trace_id": "trace-1"},
                            provider_payload={"native": {"id": "event-1"}},
                            provider_event_type="response.failed",
                        ),
                    )
                )
            )
        )
        completed_items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_COMPLETED,
                            usage={"output_tokens": 1},
                            correlation=completed_correlation,
                            visibility=StreamVisibility.REDACTED,
                            metadata={"trace_id": "trace-2"},
                            provider_payload={"native": {"id": "event-2"}},
                            provider_event_type="response.completed",
                        ),
                    )
                )
            )
        )
        cancelled_items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_CANCELLED,
                            data={"reason": "disconnect"},
                            correlation=cancelled_correlation,
                            visibility=StreamVisibility.DIAGNOSTIC,
                            metadata={"trace_id": "trace-3"},
                            provider_payload={"native": {"id": "event-3"}},
                            provider_event_type="response.cancelled",
                        ),
                    )
                )
            )
        )

        error_terminal = error_items[-2]
        completed_terminal = completed_items[-2]
        cancelled_terminal = cancelled_items[-2]

        self.assertIs(error_terminal.kind, StreamItemKind.STREAM_ERRORED)
        self.assertIs(error_terminal.correlation, error_correlation)
        self.assertIs(error_terminal.visibility, StreamVisibility.DIAGNOSTIC)
        self.assertEqual(error_terminal.metadata, {"trace_id": "trace-1"})
        self.assertEqual(
            error_terminal.provider_payload, {"native": {"id": "event-1"}}
        )
        self.assertEqual(error_terminal.provider_event_type, "response.failed")

        self.assertIs(completed_terminal.kind, StreamItemKind.STREAM_COMPLETED)
        self.assertIs(completed_terminal.correlation, completed_correlation)
        self.assertIs(completed_terminal.visibility, StreamVisibility.REDACTED)
        self.assertEqual(completed_terminal.usage, {"output_tokens": 1})
        self.assertEqual(completed_terminal.metadata, {"trace_id": "trace-2"})
        self.assertEqual(
            completed_terminal.provider_payload,
            {"native": {"id": "event-2"}},
        )
        self.assertEqual(
            completed_terminal.provider_event_type, "response.completed"
        )

        self.assertIs(cancelled_terminal.kind, StreamItemKind.STREAM_CANCELLED)
        self.assertIs(cancelled_terminal.correlation, cancelled_correlation)
        self.assertIs(
            cancelled_terminal.visibility, StreamVisibility.DIAGNOSTIC
        )
        self.assertEqual(
            cancelled_terminal.provider_payload,
            {"native": {"id": "event-3"}},
        )
        self.assertEqual(
            cancelled_terminal.provider_event_type, "response.cancelled"
        )

    def test_provider_stream_normalizer_preserves_tool_call_correlation(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(tool_call_id="call-1")
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                            correlation=correlation,
                            text_delta='{"expression"',
                            provider_event_type=(
                                "response.function_call_arguments.delta"
                            ),
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.TOOL_CALL_READY,
                            correlation=correlation,
                            data={"name": "math.calculator"},
                            provider_event_type="response.output_item.done",
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.TOOL_CALL_DONE,
                            correlation=correlation,
                            provider_event_type="response.output_item.done",
                        ),
                    )
                ),
                capabilities=StreamProviderCapabilities(
                    backend=StreamProducerBackend.LOCAL,
                    supports_tool_calls=True,
                ),
                provider_family="transformers",
                close_after_terminal=False,
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertEqual(
            {
                item.correlation.tool_call_id
                for item in items
                if item.channel is StreamChannel.TOOL_CALL
            },
            {"call-1"},
        )
        self.assertEqual(items[-1].usage, {})
        self.assertEqual(
            {item.provider_family for item in items}, {"transformers"}
        )
        self.assertIsNone(items[-1].provider_event_type)
        validate_canonical_stream_items(items)

    def test_local_text_parser_stream_normalizer_maps_channels(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "answer ",
                        "<think>private</think>",
                        '<tool_call name="math">{"x":1}</tool_call>',
                    )
                ),
                capabilities=StreamProviderCapabilities(
                    backend=StreamProducerBackend.LOCAL,
                    provider_family="transformers",
                    supports_reasoning=True,
                    supports_tool_calls=True,
                    supports_cancellation=True,
                    max_queue_depth=8,
                ),
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(10)))
        self.assertEqual(
            {item.provider_family for item in items}, {"transformers"}
        )
        self.assertEqual(items[0].metadata["capabilities"]["backend"], "local")
        self.assertEqual(
            items[0].metadata["capabilities"]["max_queue_depth"], 8
        )
        self.assertEqual(items[1].text_delta, "answer ")
        self.assertIs(items[2].visibility, StreamVisibility.PRIVATE)
        self.assertEqual(
            items[4].correlation.tool_call_id, "local-tool-call-1"
        )
        self.assertEqual(
            items[5].data, {"name": "math", "arguments": {"x": 1}}
        )
        self.assertEqual(
            items[6].correlation.tool_call_id, "local-tool-call-1"
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "answer ")
        self.assertEqual(accumulator.reasoning_text, "private")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"local-tool-call-1": '{"x":1}'},
        )

    def test_local_text_parser_stream_normalizer_marks_complete_tool_calls(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        '<tool_call name="math">{"expression"',
                        ':"2+2"}</tool_call>',
                    )
                )
            )
        )
        tool_items = [
            item for item in items if item.channel is StreamChannel.TOOL_CALL
        ]

        self.assertEqual(
            [item.kind for item in tool_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        self.assertEqual(
            [item.correlation.tool_call_id for item in tool_items],
            [
                "local-tool-call-1",
                "local-tool-call-1",
                "local-tool-call-1",
                "local-tool-call-1",
            ],
        )
        self.assertEqual(
            tool_items[2].data,
            {"name": "math", "arguments": {"expression": "2+2"}},
        )
        self.assertEqual([item.sequence for item in items], list(range(7)))
        self.assertEqual(
            accumulate_canonical_stream_items(items).tool_call_arguments,
            {"local-tool-call-1": '{"expression":"2+2"}'},
        )

    def test_local_text_parser_closes_tool_calls_before_next_channel(
        self,
    ) -> None:
        different_call_items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        '<tool_call name="math">{"x":1}</tool_call>',
                        '<tool_call name="lookup">{"q":2}</tool_call>',
                    )
                )
            )
        )
        different_call_tool_items = [
            item
            for item in different_call_items
            if item.channel is StreamChannel.TOOL_CALL
        ]

        self.assertEqual(
            [item.kind for item in different_call_tool_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in different_call_tool_items
            ],
            [
                "local-tool-call-1",
                "local-tool-call-1",
                "local-tool-call-1",
                "local-tool-call-2",
                "local-tool-call-2",
                "local-tool-call-2",
            ],
        )
        self.assertEqual(
            different_call_tool_items[1].data,
            {"name": "math", "arguments": {"x": 1}},
        )
        self.assertEqual(
            different_call_tool_items[4].data,
            {"name": "lookup", "arguments": {"q": 2}},
        )

        text_items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        '<tool_call name="math">{"x":1}</tool_call>',
                        "answer",
                    )
                )
            )
        )
        self.assertEqual(
            [item.kind for item in text_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

        reasoning_items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        '<tool_call name="math">{"x":1}</tool_call>',
                        "<think>private</think>",
                    )
                )
            )
        )
        self.assertEqual(
            [item.kind for item in reasoning_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    def test_local_text_parser_stream_reports_invalid_chunks(
        self,
    ) -> None:
        async def invalid_tokens() -> AsyncIterator[Any]:
            yield object()

        items = run(
            _collect_local_items(cast(AsyncIterable[str], invalid_tokens()))
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertIsInstance(items[1].data, dict)
        assert isinstance(items[1].data, dict)
        self.assertEqual(items[1].data["error_type"], "AssertionError")

    def test_local_stream_normalizer_parses_split_reasoning_tags(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "pre ",
                        "<thi",
                        "nk>",
                        " private ",
                        "</thi",
                        "nk>",
                        " post",
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].text_delta, "pre ")
        self.assertEqual(items[2].text_delta, " private ")
        self.assertIs(items[2].visibility, StreamVisibility.PRIVATE)
        self.assertEqual(items[4].text_delta, " post")
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "pre  post")
        self.assertEqual(accumulator.reasoning_text, " private ")

    def test_local_stream_normalizer_preserves_split_marker_whitespace(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "  before \n<thi",
                        "nk>\n  first",
                        "\nsecond  ",
                        "</thi",
                        "nk>\n after  ",
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].text_delta, "  before \n")
        self.assertEqual(items[2].text_delta, "\n  first")
        self.assertEqual(items[3].text_delta, "\nsecond  ")
        self.assertEqual(items[5].text_delta, "\n after  ")
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "  before \n\n after  ")
        self.assertEqual(accumulator.reasoning_text, "\n  first\nsecond  ")

    def test_local_stream_normalizer_detects_character_split_reasoning_tags(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    tuple("pre <think>\n  private\t</think> post")
                )
            )
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.answer_text, "pre  post")
        self.assertEqual(accumulator.reasoning_text, "\n  private\t")
        self.assertNotIn("<think>", accumulator.answer_text)
        self.assertNotIn("</think>", accumulator.reasoning_text)
        self.assertLess(
            _first_sequence(items, StreamItemKind.REASONING_DONE),
            _last_sequence(items, StreamItemKind.ANSWER_DELTA),
        )

    def test_local_stream_normalizer_handles_adjacent_reasoning_sections(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(tuple("x<think>a</think><think>b</think>y"))
            )
        )

        accumulator = accumulate_canonical_stream_items(items)
        reasoning_done_items = [
            item
            for item in items
            if item.kind is StreamItemKind.REASONING_DONE
        ]

        self.assertEqual(accumulator.answer_text, "xy")
        self.assertEqual(accumulator.reasoning_text, "ab")
        self.assertEqual(len(reasoning_done_items), 1)
        self.assertGreater(
            reasoning_done_items[0].sequence,
            _last_sequence(items, StreamItemKind.REASONING_DELTA),
        )
        self.assertLess(
            reasoning_done_items[0].sequence,
            _last_sequence(items, StreamItemKind.ANSWER_DELTA),
        )

    def test_local_stream_normalizer_handles_split_adjacent_sections(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "x<thi",
                        "nk>a</thi",
                        "nk><thi",
                        "nk>b</thi",
                        "nk>y",
                    )
                )
            )
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.answer_text, "xy")
        self.assertEqual(accumulator.reasoning_text, "ab")
        self.assertEqual(
            [
                item.kind
                for item in items
                if item.kind
                in {
                    StreamItemKind.REASONING_DELTA,
                    StreamItemKind.REASONING_DONE,
                }
            ],
            [
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
            ],
        )

    def test_local_stream_normalizer_keeps_adjacent_gap_private(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(("x<think>a</think> \n <think>b</think>y",))
            )
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.answer_text, "xy")
        self.assertEqual(accumulator.reasoning_text, "a \n b")

    def test_local_stream_normalizer_closes_before_false_repeated_marker(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(("x<think>a</think><thinking>b",))
            )
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.answer_text, "x<thinking>b")
        self.assertEqual(accumulator.reasoning_text, "a")
        self.assertEqual(
            len(
                [
                    item
                    for item in items
                    if item.kind is StreamItemKind.REASONING_DONE
                ]
            ),
            1,
        )

    def test_local_stream_normalizer_handles_empty_reasoning_markers(
        self,
    ) -> None:
        cases = (
            ("empty", tuple("a<think></think>b"), "ab", ""),
            (
                "whitespace",
                tuple("a<think>   \n</think>b"),
                "ab",
                "   \n",
            ),
        )

        for label, tokens, answer, reasoning_text in cases:
            with self.subTest(label=label):
                items = run(_collect_local_items(_local_text_chunks(tokens)))
                accumulator = accumulate_canonical_stream_items(items)
                reasoning_deltas = [
                    cast(str, item.text_delta)
                    for item in items
                    if item.kind is StreamItemKind.REASONING_DELTA
                ]

                self.assertEqual(accumulator.answer_text, answer)
                self.assertEqual("".join(reasoning_deltas), reasoning_text)
                self.assertEqual(accumulator.reasoning_text, reasoning_text)
                if label == "empty":
                    self.assertEqual(reasoning_deltas, [""])
                self.assertIn(
                    StreamItemKind.REASONING_DONE,
                    [item.kind for item in items],
                )

    def test_local_stream_normalizer_closes_unterminated_reasoning(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(("answer ", "<think>", "private"))
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "answer ")
        self.assertEqual(accumulator.reasoning_text, "private")

        partial_marker_items = run(
            _collect_local_items(
                _local_text_chunks(("answer ", "<think>", "private</thi"))
            )
        )
        partial_marker_accumulator = accumulate_canonical_stream_items(
            partial_marker_items
        )
        self.assertEqual(partial_marker_accumulator.answer_text, "answer ")
        self.assertEqual(
            partial_marker_accumulator.reasoning_text, "private</thi"
        )

    def test_local_stream_normalizer_keeps_partial_end_marker_reasoning(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "answer ",
                        "<think>",
                        " private </think",
                        " not closed",
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "answer ")
        self.assertEqual(
            accumulator.reasoning_text, " private </think not closed"
        )

    def test_local_stream_normalizer_flushes_partial_reasoning_marker(
        self,
    ) -> None:
        items = run(_collect_local_items(_local_text_chunks(("answer <thi",))))

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.answer_text, "answer <thi")
        self.assertEqual(accumulator.reasoning_text, "")

    def test_local_stream_normalizer_keeps_malformed_markers_as_answer(
        self,
    ) -> None:
        cases = (
            (("lead <thinking> tail",), "lead <thinking> tail"),
            (("lead </think> tail",), "lead </think> tail"),
            (("<thin", "g> visible"), "<thing> visible"),
            (("<", " think", "> visible"), "< think> visible"),
            (("lead <think",), "lead <think"),
            (("lead <think", " unfinished"), "lead <think unfinished"),
        )

        for tokens, answer in cases:
            with self.subTest(tokens=tokens):
                items = run(_collect_local_items(_local_text_chunks(tokens)))
                accumulator = accumulate_canonical_stream_items(items)

                self.assertEqual(accumulator.answer_text, answer)
                self.assertEqual(accumulator.reasoning_text, "")
                self.assertNotIn(
                    StreamItemKind.REASONING_DELTA,
                    [item.kind for item in items],
                )
                self.assertNotIn(
                    StreamItemKind.REASONING_DONE,
                    [item.kind for item in items],
                )

    def test_local_stream_normalizer_preserves_tool_call_prefix_text(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "before <tool_callout> ",
                        "<tool_call",
                        "backs are text",
                    )
                )
            )
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(
            accumulator.answer_text,
            "before <tool_callout> <tool_callbacks are text",
        )
        self.assertFalse(
            any(item.channel is StreamChannel.TOOL_CALL for item in items)
        )
        self.assertFalse(
            any(
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC for item in items
            )
        )

    def test_local_stream_normalizer_parses_streamed_tool_call_text(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "before ",
                        "<tool_",
                        'call name="math">',
                        '{"x":',
                        "1}",
                        "</tool_call>",
                        " after",
                    )
                )
            )
        )

        tool_items = [
            item for item in items if item.channel is StreamChannel.TOOL_CALL
        ]
        self.assertEqual(
            [item.kind for item in tool_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        self.assertEqual(
            {item.correlation.tool_call_id for item in tool_items},
            {"local-tool-call-1"},
        )
        self.assertEqual(tool_items[0].text_delta, '{"x":')
        self.assertEqual(tool_items[1].text_delta, "1}")
        self.assertEqual(
            tool_items[2].data, {"name": "math", "arguments": {"x": 1}}
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "before  after")
        self.assertEqual(
            accumulator.tool_call_arguments, {"local-tool-call-1": '{"x":1}'}
        )

        split_boundary_items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "before ",
                        "<tool_call",
                        ' name="math">',
                        "{}",
                        "</tool_call>",
                    )
                )
            )
        )
        split_boundary_accumulator = accumulate_canonical_stream_items(
            split_boundary_items
        )
        self.assertEqual(split_boundary_accumulator.answer_text, "before ")
        self.assertEqual(
            split_boundary_accumulator.tool_call_arguments,
            {"local-tool-call-1": "{}"},
        )
        split_boundary_ready = next(
            item
            for item in split_boundary_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(
            split_boundary_ready.data, {"name": "math", "arguments": {}}
        )

    def test_local_stream_normalizer_parses_tool_call_edge_metadata(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "<tool_call></tool_call>",
                        "<tool_call>{}</tool_call>",
                    )
                )
            )
        )

        ready_items = [
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]

        self.assertEqual(ready_items[0].data, {"name": None, "arguments": {}})
        self.assertEqual(ready_items[1].data, {"name": None, "arguments": {}})
        self.assertEqual(
            {item.correlation.tool_call_id for item in ready_items},
            {"local-tool-call-1", "local-tool-call-2"},
        )

    def test_local_stream_normalizer_reports_malformed_tool_call_text(
        self,
    ) -> None:
        items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "before ",
                        '<tool_call name="math">',
                        '{"x":',
                    )
                )
            )
        )

        diagnostic = next(
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        self.assertEqual(diagnostic.data["code"], "tool_call.malformed")
        self.assertEqual(
            diagnostic.correlation.tool_call_id, "local-tool-call-1"
        )
        self.assertIs(diagnostic.visibility, StreamVisibility.DIAGNOSTIC)
        self.assertFalse(
            any(item.kind is StreamItemKind.TOOL_CALL_READY for item in items)
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "before ")
        self.assertEqual(
            accumulator.tool_call_arguments, {"local-tool-call-1": '{"x":'}
        )

        invalid_json_items = run(
            _collect_local_items(
                _local_text_chunks(
                    (
                        "before ",
                        "<tool_call name='math'>not-json</tool_call>",
                        " after",
                    )
                )
            )
        )
        invalid_json_diagnostic = next(
            item
            for item in invalid_json_items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        self.assertEqual(
            invalid_json_diagnostic.data["message"],
            "malformed tool call arguments",
        )
        self.assertEqual(
            invalid_json_diagnostic.correlation.tool_call_id,
            "local-tool-call-1",
        )
        self.assertFalse(
            any(
                item.kind is StreamItemKind.TOOL_CALL_READY
                for item in invalid_json_items
            )
        )
        invalid_json_accumulator = accumulate_canonical_stream_items(
            invalid_json_items
        )
        self.assertEqual(invalid_json_accumulator.answer_text, "before  after")
        self.assertEqual(
            invalid_json_accumulator.tool_call_arguments,
            {"local-tool-call-1": "not-json"},
        )

        non_object_items = run(
            _collect_local_items(
                _local_text_chunks(("<tool_call>[]</tool_call>",))
            )
        )
        non_object_diagnostic = next(
            item
            for item in non_object_items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        self.assertEqual(
            non_object_diagnostic.correlation.tool_call_id,
            "local-tool-call-1",
        )
        self.assertFalse(
            any(
                item.kind is StreamItemKind.TOOL_CALL_READY
                for item in non_object_items
            )
        )

        partial_open_items = run(
            _collect_local_items(_local_text_chunks(("<tool_call name",)))
        )
        self.assertTrue(
            any(
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                for item in partial_open_items
            )
        )

        partial_close_items = run(
            _collect_local_items(_local_text_chunks(("<tool_call>{}</tool_",)))
        )
        self.assertEqual(
            accumulate_canonical_stream_items(
                partial_close_items
            ).tool_call_arguments,
            {"local-tool-call-1": "{}</tool_"},
        )

    def test_local_stream_normalizer_maps_bad_chunk_to_error_terminal(
        self,
    ) -> None:
        async def bad_tokens() -> AsyncIterator[Any]:
            yield object()

        items = run(_collect_local_items(bad_tokens()))

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].data["error_type"], "AssertionError")
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    def test_local_stream_normalizer_closes_on_bad_chunk(
        self,
    ) -> None:
        class BadTokens:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "BadTokens":
                return self

            async def __anext__(self) -> Any:
                self.read_count += 1
                if self.read_count == 1:
                    return "good"
                return object()

            async def aclose(self) -> None:
                self.closed = True

        tokens = BadTokens()
        items = run(_collect_local_items(tokens))

        self.assertEqual(tokens.read_count, 2)
        self.assertTrue(tokens.closed)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[3].data["error_type"], "AssertionError")
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    def test_local_stream_normalizer_rejects_hosted_capabilities(self) -> None:
        with self.assertRaises(AssertionError):
            run(
                _collect_local_items(
                    _local_text_chunks(("x",)),
                    capabilities=StreamProviderCapabilities(
                        backend=StreamProducerBackend.HOSTED,
                    ),
                )
            )

    def test_provider_stream_normalizer_maps_exhaustion_to_completion(
        self,
    ) -> None:
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="done",
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[-2].usage, {})
        self.assertIs(
            items[-2].terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "done",
        )

    def test_provider_stream_normalizer_maps_provider_error_to_terminal(
        self,
    ) -> None:
        class FailingEvents:
            def __aiter__(self) -> "FailingEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                raise RuntimeError("provider failed")

        items = run(_collect_provider_items(FailingEvents()))

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            items[1].data,
            {"error_type": "RuntimeError", "message": "provider failed"},
        )
        self.assertEqual(
            items[1].correlation,
            StreamItemCorrelation(),
        )
        self.assertIs(items[1].visibility, StreamVisibility.PUBLIC)
        self.assertIs(items[1].terminal_outcome, StreamTerminalOutcome.ERRORED)
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

        provider_error_items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_ERRORED,
                            data={"message": "provider error"},
                            provider_event_type="response.failed",
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in provider_error_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            provider_error_items[1].data, {"message": "provider error"}
        )
        self.assertEqual(
            provider_error_items[1].provider_event_type, "response.failed"
        )

    def test_provider_stream_normalizer_maps_validation_error_to_terminal(
        self,
    ) -> None:
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(kind=StreamItemKind.ANSWER_DONE),
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="late",
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(3)))
        self.assertEqual(items[1].data["error_type"], "StreamValidationError")
        self.assertIn("answer done before content", items[1].data["message"])
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    def test_provider_stream_normalizer_rejects_tool_call_boundary_errors(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(tool_call_id="call-1")
        cases = (
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_READY,
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "missing tool_call_id",
            ),
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_DONE,
                        correlation=correlation,
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "done before ready",
            ),
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_READY,
                        correlation=correlation,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_READY,
                        correlation=correlation,
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "duplicate tool-call ready",
            ),
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=correlation,
                        text_delta='{"x":1}',
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_READY,
                        correlation=correlation,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=correlation,
                        text_delta='{"late":true}',
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "argument emitted after ready",
            ),
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=correlation,
                        text_delta='{"x":1}',
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_READY,
                        correlation=correlation,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_DONE,
                        correlation=correlation,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=correlation,
                        text_delta='{"late":true}',
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "after tool-call done",
            ),
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=correlation,
                        text_delta='{"x":1}',
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "missing ready",
            ),
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=correlation,
                        text_delta='{"x":1}',
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_READY,
                        correlation=correlation,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.STREAM_COMPLETED,
                        usage={"output_tokens": 1},
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "missing done",
            ),
            (
                (
                    StreamProviderEvent(
                        kind=StreamItemKind.STREAM_DIAGNOSTIC,
                        correlation=correlation,
                        data={"code": "tool_call.malformed"},
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=correlation,
                        text_delta='{"late":true}',
                    ),
                ),
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.STREAM_DIAGNOSTIC,
                    StreamItemKind.STREAM_ERRORED,
                    StreamItemKind.STREAM_CLOSED,
                ],
                "after malformed diagnostic",
            ),
        )

        for events, expected_kinds, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                items = run(_collect_provider_items(_provider_events(events)))

                self.assertEqual([item.kind for item in items], expected_kinds)
                self.assertEqual(
                    [item.sequence for item in items],
                    list(range(len(items))),
                )
                self.assertEqual(
                    items[-2].data["error_type"], "StreamValidationError"
                )
                self.assertIn(expected_message, items[-2].data["message"])
                if expected_message == "missing ready":
                    done = next(
                        item
                        for item in items
                        if item.kind is StreamItemKind.TOOL_CALL_DONE
                    )
                    self.assertEqual(
                        done.metadata["tool_call.close_reason"], "error"
                    )
                self.assertIs(
                    accumulate_canonical_stream_items(items).terminal_outcome,
                    StreamTerminalOutcome.ERRORED,
                )

    def test_provider_stream_normalizer_allows_diagnosed_malformed_tool_call(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(tool_call_id="call-1")
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                            correlation=correlation,
                            text_delta="{",
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_DIAGNOSTIC,
                            correlation=correlation,
                            data={"code": "tool_call.malformed"},
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        done = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_DONE
        )
        self.assertEqual(done.metadata["tool_call.close_reason"], "malformed")
        self.assertIs(
            items[-2].terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).tool_call_arguments,
            {"call-1": "{"},
        )

    def test_provider_stream_normalizer_rejects_content_after_final_usage(
        self,
    ) -> None:
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="done",
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.USAGE_COMPLETED,
                            usage={"output_tokens": 1},
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta="late",
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(6)))
        self.assertEqual(items[3].usage, {"output_tokens": 1})
        self.assertEqual(items[4].data["error_type"], "StreamValidationError")
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    def test_provider_stream_normalizer_rejects_new_content_after_final_usage(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(tool_call_id="call-1")
        cases = (
            StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta="late",
            ),
            StreamProviderEvent(
                kind=StreamItemKind.REASONING_DELTA,
                text_delta="late",
                visibility=StreamVisibility.PRIVATE,
            ),
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=correlation,
                text_delta='{"late":true}',
            ),
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                correlation=correlation,
                text_delta="late",
            ),
            StreamProviderEvent(
                kind=StreamItemKind.FLOW_EVENT,
                correlation=StreamItemCorrelation(flow_run_id="flow-1"),
            ),
            StreamProviderEvent(
                kind=StreamItemKind.MODEL_CONTINUATION_STARTED,
            ),
        )

        for late_event in cases:
            with self.subTest(kind=late_event.kind):
                items = run(
                    _collect_provider_items(
                        _provider_events(
                            (
                                StreamProviderEvent(
                                    kind=StreamItemKind.USAGE_COMPLETED,
                                    usage={"output_tokens": 1},
                                ),
                                late_event,
                            )
                        )
                    )
                )

                self.assertEqual(
                    [item.kind for item in items],
                    [
                        StreamItemKind.STREAM_STARTED,
                        StreamItemKind.USAGE_COMPLETED,
                        StreamItemKind.STREAM_ERRORED,
                        StreamItemKind.STREAM_CLOSED,
                    ],
                )
                self.assertEqual(
                    items[2].data["error_type"], "StreamValidationError"
                )
                self.assertIn("final usage", items[2].data["message"])
                self.assertIs(
                    accumulate_canonical_stream_items(items).terminal_outcome,
                    StreamTerminalOutcome.ERRORED,
                )

    def test_provider_stream_normalizer_allows_diagnostic_after_final_usage(
        self,
    ) -> None:
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.USAGE_COMPLETED,
                            usage={"output_tokens": 1},
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_DIAGNOSTIC,
                            data={"code": "stream.note"},
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.final_usage, {"output_tokens": 1})
        self.assertEqual(accumulator.diagnostics, (items[2],))

    def test_provider_stream_normalizer_allows_terminal_after_final_usage(
        self,
    ) -> None:
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.USAGE_COMPLETED,
                            usage={"output_tokens": 1},
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_CANCELLED,
                            data={"reason": "disconnect"},
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.final_usage, {"output_tokens": 1})
        self.assertIs(
            accumulator.terminal_outcome, StreamTerminalOutcome.CANCELLED
        )

    def test_provider_stream_normalizer_rejects_incomplete_tool_final_usage(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(tool_call_id="call-1")
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                            correlation=correlation,
                            text_delta='{"x":1}',
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.USAGE_COMPLETED,
                            usage={"output_tokens": 1},
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[2].metadata["tool_call.close_reason"], "error")
        self.assertIn("missing ready", items[3].data["message"])
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    def test_provider_stream_normalizer_maps_cancellation_to_terminal(
        self,
    ) -> None:
        class CancelledEvents:
            def __aiter__(self) -> "CancelledEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                raise CancelledError()

        items = run(_collect_provider_items(CancelledEvents()))

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertIs(
            items[1].terminal_outcome, StreamTerminalOutcome.CANCELLED
        )
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )

    def test_provider_stream_normalizer_maps_provider_cancel_event(
        self,
    ) -> None:
        items = run(
            _collect_provider_items(
                _provider_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.STREAM_CANCELLED,
                            data={"reason": "disconnect"},
                            provider_event_type="response.cancelled",
                        ),
                    )
                )
            )
        )

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].data, {"reason": "disconnect"})
        self.assertEqual(items[1].provider_event_type, "response.cancelled")
        self.assertIs(
            items[1].terminal_outcome, StreamTerminalOutcome.CANCELLED
        )

    def test_provider_stream_normalizer_closes_after_terminal_event(
        self,
    ) -> None:
        class TerminalThenLateEvents:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "TerminalThenLateEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.read_count += 1
                if self.read_count == 1:
                    return StreamProviderEvent(
                        kind=StreamItemKind.STREAM_CANCELLED,
                        data={"reason": "disconnect"},
                    )
                raise AssertionError("provider was read after terminal")

            async def aclose(self) -> None:
                self.closed = True

        events = TerminalThenLateEvents()
        items = run(_collect_provider_items(events))

        self.assertEqual(events.read_count, 1)
        self.assertTrue(events.closed)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    def test_provider_stream_normalizer_closes_after_provider_cancellation(
        self,
    ) -> None:
        class CancelledEvents:
            def __init__(self) -> None:
                self.closed = False

            def __aiter__(self) -> "CancelledEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                raise CancelledError()

            async def aclose(self) -> None:
                self.closed = True

        events = CancelledEvents()
        items = run(_collect_provider_items(events))

        self.assertTrue(events.closed)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    def test_provider_stream_cancellation_race_closes_source(self) -> None:
        class PendingEvents:
            def __init__(self) -> None:
                self.started = AsyncEvent()
                self.closed = False
                self.cancelled = False

            def __aiter__(self) -> "PendingEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.started.set()
                try:
                    await AsyncEvent().wait()
                except CancelledError:
                    self.cancelled = True
                    raise
                return StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta="late",
                )

            async def aclose(self) -> None:
                self.closed = True

        async def cancel_pending_provider_read() -> (
            tuple[PendingEvents, CanonicalStreamItem | None]
        ):
            events = PendingEvents()
            stream = normalize_provider_stream(
                events,
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
            )
            started = await stream.__anext__()
            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            pull = create_task(stream.__anext__())
            item: CanonicalStreamItem | None = None
            try:
                await wait_for(events.started.wait(), STREAM_TEST_TIMEOUT)
                pull.cancel()
                try:
                    item = await wait_for(pull, STREAM_TEST_TIMEOUT)
                except CancelledError:
                    item = None
            finally:
                if not pull.done():
                    pull.cancel()
                    try:
                        await wait_for(pull, STREAM_TEST_TIMEOUT)
                    except CancelledError:
                        pass
                await cast(Any, stream).aclose()
            return events, item

        events, item = run(cancel_pending_provider_read())

        self.assertTrue(events.cancelled)
        self.assertTrue(events.closed)
        if item is not None:
            self.assertIs(item.kind, StreamItemKind.STREAM_CANCELLED)
            self.assertIs(
                item.terminal_outcome, StreamTerminalOutcome.CANCELLED
            )

    def test_provider_stream_normalizer_closes_on_consumer_disconnect(
        self,
    ) -> None:
        class PendingEvents:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "PendingEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.read_count += 1
                return StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta="late",
                )

            async def aclose(self) -> None:
                self.closed = True

        async def close_after_start() -> PendingEvents:
            events = PendingEvents()
            stream = normalize_provider_stream(
                events,
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
            )
            item = await stream.__anext__()
            self.assertIs(item.kind, StreamItemKind.STREAM_STARTED)
            await cast(Any, stream).aclose()
            return events

        events = run(close_after_start())

        self.assertEqual(events.read_count, 0)
        self.assertTrue(events.closed)

    def test_provider_stream_normalizer_does_not_read_ahead(
        self,
    ) -> None:
        class PullTrackedEvents:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "PullTrackedEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.read_count += 1
                if self.read_count == 1:
                    return StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DELTA,
                        text_delta="a",
                    )
                if self.read_count == 2:
                    return StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DELTA,
                        text_delta="b",
                    )
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.closed = True

        async def consume_slowly() -> tuple[PullTrackedEvents, list[int]]:
            events = PullTrackedEvents()
            stream = normalize_provider_stream(
                events,
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
            )
            read_counts: list[int] = []

            started = await stream.__anext__()
            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            read_counts.append(events.read_count)

            first = await stream.__anext__()
            self.assertIs(first.kind, StreamItemKind.ANSWER_DELTA)
            await sleep(0)
            read_counts.append(events.read_count)

            second = await stream.__anext__()
            self.assertIs(second.kind, StreamItemKind.ANSWER_DELTA)
            await sleep(0)
            read_counts.append(events.read_count)

            rest = [item async for item in stream]
            self.assertEqual(
                [item.kind for item in rest],
                [
                    StreamItemKind.ANSWER_DONE,
                    StreamItemKind.STREAM_COMPLETED,
                    StreamItemKind.STREAM_CLOSED,
                ],
            )
            return events, read_counts

        events, read_counts = run(consume_slowly())

        self.assertEqual(read_counts, [0, 1, 2])
        self.assertEqual(events.read_count, 3)
        self.assertTrue(events.closed)

    def test_local_text_stream_does_not_read_ahead(self) -> None:
        class PullTrackedTokens:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "PullTrackedTokens":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                if self.read_count == 1:
                    return "a"
                if self.read_count == 2:
                    return "b"
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.closed = True

        async def consume_slowly() -> tuple[PullTrackedTokens, list[int]]:
            tokens = PullTrackedTokens()
            stream = _local_text_stream(
                tokens,
                stream_session_id="local-stream",
                run_id="local-run",
                turn_id="local-turn",
            )
            read_counts: list[int] = []

            started = await stream.__anext__()
            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            read_counts.append(tokens.read_count)

            first = await stream.__anext__()
            self.assertIs(first.kind, StreamItemKind.ANSWER_DELTA)
            await sleep(0)
            read_counts.append(tokens.read_count)

            second = await stream.__anext__()
            self.assertIs(second.kind, StreamItemKind.ANSWER_DELTA)
            await sleep(0)
            read_counts.append(tokens.read_count)

            rest = [item async for item in stream]
            self.assertEqual(
                [item.kind for item in rest],
                [
                    StreamItemKind.ANSWER_DONE,
                    StreamItemKind.STREAM_COMPLETED,
                    StreamItemKind.STREAM_CLOSED,
                ],
            )
            return tokens, read_counts

        tokens, read_counts = run(consume_slowly())

        self.assertEqual(read_counts, [0, 1, 2])
        self.assertEqual(tokens.read_count, 3)
        self.assertFalse(tokens.closed)

    def test_local_text_event_parser_preserves_split_semantics(self) -> None:
        parser = LocalTextStreamEventParser()

        events = (
            *parser.push("a<think>r"),
            *parser.push('</think><tool_call name="lookup">{"q"'),
            *parser.push(':"v"}</tool_call>b'),
            *parser.flush(),
        )

        self.assertEqual(
            [(event.kind, event.text_delta) for event in events],
            [
                (StreamItemKind.ANSWER_DELTA, "a"),
                (StreamItemKind.REASONING_DELTA, "r"),
                (StreamItemKind.REASONING_DONE, None),
                (StreamItemKind.TOOL_CALL_ARGUMENT_DELTA, '{"q"'),
                (StreamItemKind.TOOL_CALL_ARGUMENT_DELTA, ':"v"}'),
                (StreamItemKind.TOOL_CALL_READY, None),
                (StreamItemKind.TOOL_CALL_DONE, None),
                (StreamItemKind.ANSWER_DELTA, "b"),
            ],
        )
        tool_call_ids = {
            event.correlation.tool_call_id
            for event in events
            if event.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
        }
        self.assertEqual(tool_call_ids, {"local-tool-call-1"})
        ready = next(
            event
            for event in events
            if event.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(
            ready.data,
            {"name": "lookup", "arguments": {"q": "v"}},
        )

    def test_stream_token_metadata_is_metadata_only_enrichment(self) -> None:
        metadata = stream_token_metadata(
            token_id=7,
            probability=0.25,
            step=3,
            probability_distribution="softmax",
            candidates=(("A", 7, 0.25), ("B", None, None)),
            provider_name="ds4",
        )

        self.assertEqual(
            metadata,
            {
                "token_id": 7,
                "probability": 0.25,
                "step": 3,
                "probability_distribution": "softmax",
                "tokens": [
                    {"token": "A", "token_id": 7, "probability": 0.25},
                    {"token": "B"},
                ],
                "provider_name": "ds4",
            },
        )

    def test_provider_and_local_streams_wait_for_slow_consumers(
        self,
    ) -> None:
        class GatedProviderEvents:
            def __init__(self) -> None:
                self.read_count = 0
                self.second_pull_started = AsyncEvent()
                self.release_second = AsyncEvent()
                self.closed = False

            def __aiter__(self) -> "GatedProviderEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.read_count += 1
                if self.read_count == 1:
                    return StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DELTA,
                        text_delta="a",
                    )
                if self.read_count == 2:
                    self.second_pull_started.set()
                    await self.release_second.wait()
                    return StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DELTA,
                        text_delta="b",
                    )
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.closed = True

        class GatedLocalTokens:
            def __init__(self) -> None:
                self.read_count = 0
                self.second_pull_started = AsyncEvent()
                self.release_second = AsyncEvent()
                self.closed = False

            def __aiter__(self) -> "GatedLocalTokens":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                if self.read_count == 1:
                    return "a"
                if self.read_count == 2:
                    self.second_pull_started.set()
                    await self.release_second.wait()
                    return "b"
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.closed = True

        async def assert_provider_backpressure() -> None:
            events = GatedProviderEvents()
            stream = normalize_provider_stream(
                events,
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
            )

            started = await stream.__anext__()
            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            self.assertEqual(events.read_count, 0)

            first = await stream.__anext__()
            self.assertIs(first.kind, StreamItemKind.ANSWER_DELTA)
            self.assertEqual(events.read_count, 1)
            await sleep(0)
            self.assertEqual(events.read_count, 1)

            second_pull = create_task(stream.__anext__())
            try:
                await wait_for(
                    events.second_pull_started.wait(), STREAM_TEST_TIMEOUT
                )
                self.assertEqual(events.read_count, 2)
                self.assertFalse(second_pull.done())
                events.release_second.set()
                second = await wait_for(second_pull, STREAM_TEST_TIMEOUT)
                self.assertIs(second.kind, StreamItemKind.ANSWER_DELTA)
            finally:
                if not second_pull.done():
                    second_pull.cancel()
                    try:
                        await wait_for(second_pull, STREAM_TEST_TIMEOUT)
                    except CancelledError:
                        pass
                await cast(Any, stream).aclose()
            self.assertTrue(events.closed)

        async def assert_local_backpressure() -> None:
            tokens = GatedLocalTokens()
            stream = _local_text_stream(
                tokens,
                stream_session_id="local-stream",
                run_id="local-run",
                turn_id="local-turn",
            )

            started = await stream.__anext__()
            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            self.assertEqual(tokens.read_count, 0)

            first = await stream.__anext__()
            self.assertIs(first.kind, StreamItemKind.ANSWER_DELTA)
            self.assertEqual(tokens.read_count, 1)
            await sleep(0)
            self.assertEqual(tokens.read_count, 1)

            second_pull = create_task(stream.__anext__())
            try:
                await wait_for(
                    tokens.second_pull_started.wait(), STREAM_TEST_TIMEOUT
                )
                self.assertEqual(tokens.read_count, 2)
                self.assertFalse(second_pull.done())
                tokens.release_second.set()
                second = await wait_for(second_pull, STREAM_TEST_TIMEOUT)
                self.assertIs(second.kind, StreamItemKind.ANSWER_DELTA)
            finally:
                if not second_pull.done():
                    second_pull.cancel()
                    try:
                        await wait_for(second_pull, STREAM_TEST_TIMEOUT)
                    except CancelledError:
                        pass
                await cast(Any, stream).aclose()
            self.assertTrue(tokens.closed)

        run(assert_provider_backpressure())
        run(assert_local_backpressure())

    def test_local_text_stream_closes_on_consumer_disconnect(
        self,
    ) -> None:
        class PendingTokens:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "PendingTokens":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                return "late"

            async def aclose(self) -> None:
                self.closed = True

        async def close_after_start() -> PendingTokens:
            tokens = PendingTokens()
            stream = _local_text_stream(
                tokens,
                stream_session_id="local-stream",
                run_id="local-run",
                turn_id="local-turn",
            )
            item = await stream.__anext__()
            self.assertIs(item.kind, StreamItemKind.STREAM_STARTED)
            await cast(Any, stream).aclose()
            return tokens

        tokens = run(close_after_start())

        self.assertEqual(tokens.read_count, 0)
        self.assertTrue(tokens.closed)

    def test_provider_stream_normalizer_closes_on_validation_error(
        self,
    ) -> None:
        class InvalidSequenceEvents:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "InvalidSequenceEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.read_count += 1
                if self.read_count == 1:
                    return StreamProviderEvent(kind=StreamItemKind.ANSWER_DONE)
                if self.read_count == 2:
                    return StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DELTA,
                        text_delta="late",
                    )
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.closed = True

        async def collect_invalid() -> (
            tuple[InvalidSequenceEvents, tuple[CanonicalStreamItem, ...]]
        ):
            events = InvalidSequenceEvents()
            stream = normalize_provider_stream(
                events,
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
            )
            return events, tuple([item async for item in stream])

        events, items = run(collect_invalid())

        self.assertEqual(events.read_count, 1)
        self.assertTrue(events.closed)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(3)))
        self.assertEqual(items[1].data["error_type"], "StreamValidationError")
        self.assertIn("answer done before content", items[1].data["message"])

    def test_provider_stream_normalizer_ignores_late_provider_failure(
        self,
    ) -> None:
        class FailingAfterTerminalEvents:
            def __init__(self) -> None:
                self._count = 0

            def __aiter__(self) -> "FailingAfterTerminalEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self._count += 1
                if self._count == 1:
                    return StreamProviderEvent(
                        kind=StreamItemKind.STREAM_COMPLETED
                    )
                raise RuntimeError("late provider failure")

        items = run(_collect_provider_items(FailingAfterTerminalEvents()))

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    def test_provider_stream_normalizer_ignores_late_cancellation(
        self,
    ) -> None:
        class CancelledAfterTerminalEvents:
            def __init__(self) -> None:
                self._count = 0

            def __aiter__(self) -> "CancelledAfterTerminalEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self._count += 1
                if self._count == 1:
                    return StreamProviderEvent(
                        kind=StreamItemKind.STREAM_COMPLETED
                    )
                raise CancelledError()

        items = run(_collect_provider_items(CancelledAfterTerminalEvents()))

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertIs(
            accumulate_canonical_stream_items(items).terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    def test_provider_stream_normalizer_stops_before_double_terminal(
        self,
    ) -> None:
        class DoubleTerminalEvents:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "DoubleTerminalEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.read_count += 1
                return StreamProviderEvent(
                    kind=(
                        StreamItemKind.STREAM_COMPLETED
                        if self.read_count == 1
                        else StreamItemKind.STREAM_ERRORED
                    )
                )

            async def aclose(self) -> None:
                self.closed = True

        events = DoubleTerminalEvents()
        items = run(_collect_provider_items(events))

        self.assertEqual(events.read_count, 1)
        self.assertTrue(events.closed)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    def test_provider_stream_normalizer_stops_before_post_terminal_content(
        self,
    ) -> None:
        class PostTerminalContentEvents:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "PostTerminalContentEvents":
                return self

            async def __anext__(self) -> StreamProviderEvent:
                self.read_count += 1
                if self.read_count == 1:
                    return StreamProviderEvent(
                        kind=StreamItemKind.STREAM_COMPLETED
                    )
                return StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta="late",
                )

            async def aclose(self) -> None:
                self.closed = True

        events = PostTerminalContentEvents()
        items = run(_collect_provider_items(events))

        self.assertEqual(events.read_count, 1)
        self.assertTrue(events.closed)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    def test_provider_stream_validator_rejects_double_terminal(self) -> None:
        with self.assertRaises(StreamValidationError):
            validate_canonical_stream_items(
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _stream_completed(1),
                    _stream_errored(2),
                )
            )

    def test_provider_stream_validator_rejects_post_terminal_content(
        self,
    ) -> None:
        with self.assertRaises(StreamValidationError):
            validate_canonical_stream_items(
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _stream_completed(1),
                    _item(
                        StreamItemKind.ANSWER_DELTA,
                        2,
                        text_delta="late",
                    ),
                )
            )

    def test_legacy_surface_inventory_classifies_current_shapes(
        self,
    ) -> None:
        inventory = legacy_stream_surface_inventory()

        self.assertEqual(
            {category.value for category in StreamLegacyBoundaryCategory},
            {
                "producer",
                "sdk_response",
                "orchestrator",
                "parser",
                "eventing",
                "cli_stdout",
                "chat_sse",
                "responses_sse",
                "mcp",
                "a2a",
                "flow",
                "test_fixture",
                "helper_only",
            },
        )
        self.assertEqual(
            {scope.value for scope in StreamLegacyInventoryScope},
            {"production_runtime", "test_fixture", "helper_only"},
        )
        self.assertEqual(
            {direction.value for direction in StreamLegacyBoundaryDirection},
            {
                "accepts",
                "emits",
                "projects",
                "public_return_type",
                "control",
            },
        )
        self.assertEqual(inventory, ())

        with self.assertRaises(AssertionError):
            classify_legacy_stream_surface("Token")  # type: ignore[arg-type]
        for surface in StreamLegacySurface:
            with self.subTest(surface=surface):
                with self.assertRaises(StreamValidationError):
                    classify_legacy_stream_surface(surface)
        with patch("avalan.model.stream._LEGACY_STREAM_SURFACE_INVENTORY", ()):
            with self.assertRaises(StreamValidationError):
                classify_legacy_stream_surface(StreamLegacySurface.TOKEN)
        fixture_entry = StreamLegacySurfaceInventoryEntry(
            surface=StreamLegacySurface.TOKEN,
            classification=(
                StreamLegacySurfaceClassification.TEMPORARY_COMPATIBILITY_SHIM
            ),
            categories=(StreamLegacyBoundaryCategory.TEST_FIXTURE,),
            scope=StreamLegacyInventoryScope.TEST_FIXTURE,
            owner="tests.model.model_stream_contract_test",
            removal_condition="synthetic classification coverage",
        )
        with patch(
            "avalan.model.stream._LEGACY_STREAM_SURFACE_INVENTORY",
            (fixture_entry,),
        ):
            self.assertIs(
                classify_legacy_stream_surface(StreamLegacySurface.TOKEN),
                fixture_entry,
            )

    def test_legacy_surface_inventory_rejects_malformed_entries(self) -> None:
        def make_entry(
            **overrides: object,
        ) -> StreamLegacySurfaceInventoryEntry:
            values: dict[str, object] = {
                "surface": StreamLegacySurface.TOKEN,
                "classification": StreamLegacySurfaceClassification.REMOVE_NOW,
                "categories": (StreamLegacyBoundaryCategory.PRODUCER,),
                "scope": StreamLegacyInventoryScope.PRODUCTION_RUNTIME,
                "owner": "model.stream",
                "removal_condition": "done",
                "canonical_kind": StreamItemKind.ANSWER_DELTA,
                "canonical_channel": StreamChannel.ANSWER,
            }
            values.update(overrides)
            return StreamLegacySurfaceInventoryEntry(**cast(Any, values))

        invalid_entries = (
            lambda: make_entry(
                surface="Token",
            ),
            lambda: make_entry(
                classification="temporary",
            ),
            lambda: make_entry(
                categories=[StreamLegacyBoundaryCategory.PRODUCER],
            ),
            lambda: make_entry(categories=()),
            lambda: make_entry(
                categories=(
                    StreamLegacyBoundaryCategory.PRODUCER,
                    StreamLegacyBoundaryCategory.PRODUCER,
                ),
            ),
            lambda: make_entry(categories=("producer",)),
            lambda: make_entry(scope="production_runtime"),
            lambda: make_entry(
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(
                categories=(StreamLegacyBoundaryCategory.TEST_FIXTURE,),
            ),
            lambda: make_entry(
                categories=(StreamLegacyBoundaryCategory.PRODUCER,),
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                ingestion_shim="canonical_item_from_token",
            ),
            lambda: make_entry(
                categories=(StreamLegacyBoundaryCategory.PRODUCER,),
                scope=StreamLegacyInventoryScope.HELPER_ONLY,
                classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
            ),
            lambda: make_entry(owner=""),
            lambda: make_entry(removal_condition=""),
            lambda: make_entry(
                categories=(StreamLegacyBoundaryCategory.TEST_FIXTURE,),
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                ingestion_shim="",
            ),
            lambda: make_entry(
                canonical_kind="answer.delta",
            ),
            lambda: make_entry(
                canonical_channel="answer",
            ),
            lambda: make_entry(
                canonical_channel=StreamChannel.REASONING,
            ),
            lambda: make_entry(canonical_kind=None),
            lambda: make_entry(canonical_channel=None),
            lambda: make_entry(
                ingestion_shim="canonical_item_from_token",
            ),
            lambda: make_entry(
                categories=(StreamLegacyBoundaryCategory.TEST_FIXTURE,),
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                ingestion_shim=None,
            ),
            lambda: make_entry(
                categories=(StreamLegacyBoundaryCategory.TEST_FIXTURE,),
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_COMPATIBILITY_SHIM
                ),
                ingestion_shim="canonical_item_from_token",
            ),
        )

        for build_invalid_entry in invalid_entries:
            with self.subTest(build_entry=build_invalid_entry):
                with self.assertRaises(AssertionError):
                    build_invalid_entry()

        fixture_entry = StreamLegacySurfaceInventoryEntry(
            surface=StreamLegacySurface.TOKEN,
            classification=(
                StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
            ),
            categories=(StreamLegacyBoundaryCategory.TEST_FIXTURE,),
            scope=StreamLegacyInventoryScope.TEST_FIXTURE,
            owner="tests",
            removal_condition="legacy rejection fixture",
            ingestion_shim="legacy_rejection_fixture",
            canonical_kind=StreamItemKind.ANSWER_DELTA,
            canonical_channel=StreamChannel.ANSWER,
        )
        self.assertIs(
            fixture_entry.scope, StreamLegacyInventoryScope.TEST_FIXTURE
        )

        helper_entry = StreamLegacySurfaceInventoryEntry(
            surface=StreamLegacySurface.STRING,
            classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
            categories=(StreamLegacyBoundaryCategory.HELPER_ONLY,),
            scope=StreamLegacyInventoryScope.HELPER_ONLY,
            owner="tests",
            removal_condition="private migration helper",
        )
        self.assertIs(
            helper_entry.scope, StreamLegacyInventoryScope.HELPER_ONLY
        )

    def test_legacy_runtime_boundary_inventory_classifies_live_paths(
        self,
    ) -> None:
        inventory = legacy_stream_runtime_boundary_inventory()

        self.assertEqual(inventory, ())

        with self.assertRaises(AssertionError):
            classify_legacy_stream_runtime_boundary("", "qualname")
        with self.assertRaises(AssertionError):
            classify_legacy_stream_runtime_boundary("module", "")
        with self.assertRaises(StreamValidationError):
            classify_legacy_stream_runtime_boundary(
                "avalan.model.stream",
                "StreamProjectionState.project",
            )
        fixture_entry = StreamLegacyRuntimeBoundaryInventoryEntry(
            module="tests.model.model_stream_contract_test",
            qualname="legacy_fixture_runtime_boundary",
            surfaces=(StreamLegacySurface.TOKEN,),
            classification=(
                StreamLegacySurfaceClassification.TEMPORARY_COMPATIBILITY_SHIM
            ),
            category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
            scope=StreamLegacyInventoryScope.TEST_FIXTURE,
            directions=(StreamLegacyBoundaryDirection.ACCEPTS,),
            owner="tests.model.model_stream_contract_test",
            removal_condition="synthetic classification coverage",
        )
        with patch(
            "avalan.model.stream._LEGACY_STREAM_RUNTIME_BOUNDARY_INVENTORY",
            (fixture_entry,),
        ):
            self.assertIs(
                classify_legacy_stream_runtime_boundary(
                    "tests.model.model_stream_contract_test",
                    "legacy_fixture_runtime_boundary",
                ),
                fixture_entry,
            )

    def test_legacy_runtime_boundary_inventory_has_no_flow_listeners(
        self,
    ) -> None:
        inventory = legacy_stream_runtime_boundary_inventory()
        inventory_keys = {
            (entry.module, entry.qualname) for entry in inventory
        }
        removed_flow_listener_boundaries = {
            ("avalan.flow.executor", "FlowExecutor.run"),
            ("avalan.flow.runtime", "execute_flow_plan"),
            ("avalan.task.targets.flow", "_emit_flow_event"),
            ("avalan.task.targets.flow", "_flow_node_event_listener"),
            ("avalan.task.targets.flow", "_task_flow_stream_listener"),
        }

        self.assertTrue(
            removed_flow_listener_boundaries.isdisjoint(inventory_keys)
        )
        self.assertEqual(
            [
                (entry.module, entry.qualname)
                for entry in inventory
                if entry.category is StreamLegacyBoundaryCategory.FLOW
                and StreamLegacySurface.EVENT in entry.surfaces
            ],
            [],
        )

    def test_legacy_runtime_boundary_inventory_entries_resolve_to_source(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        module_names: dict[str, set[str]] = {}
        for entry in legacy_stream_runtime_boundary_inventory():
            if entry.module not in module_names:
                module_path = _module_source_path(
                    repository_root, entry.module
                )
                self.assertTrue(
                    module_path.exists(),
                    f"{entry.module} resolves to {module_path}",
                )
                module_names[entry.module] = _module_defined_qualnames(
                    parse(module_path.read_text(encoding="utf-8"))
                )

            with self.subTest(module=entry.module, qualname=entry.qualname):
                self.assertIn(entry.qualname, module_names[entry.module])

    def test_runtime_source_does_not_import_legacy_test_fixtures(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        for path in (repository_root / "src" / "avalan").rglob("*.py"):
            relative_path = path.relative_to(repository_root)
            source = path.read_text(encoding="utf-8")
            tree = parse(source)
            with self.subTest(path=relative_path):
                self.assertEqual(
                    _runtime_source_test_import_sites(tree),
                    set(),
                )

    def test_runtime_source_import_guard_detects_local_test_imports(
        self,
    ) -> None:
        tree = parse("""def build_fixture():
    from tests.model import model_stream_contract_test
    import tests.model.model_stream_contract_test
    return model_stream_contract_test
""")

        self.assertEqual(
            {module for _, module in _runtime_source_test_import_sites(tree)},
            {
                "tests.model",
                "tests.model.model_stream_contract_test",
            },
        )

    def test_legacy_classifier_source_debt_snapshot_does_not_grow(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        current = _source_legacy_stream_classifier_sites(repository_root)
        ceiling = _PHASE_1_1_LEGACY_CLASSIFIER_DEBT_CEILING

        new_sites = set(current) - set(ceiling)
        grown_surfaces = {
            key: current[key] - ceiling[key]
            for key in set(current).intersection(ceiling)
            if current[key] - ceiling[key]
        }

        self.assertEqual(new_sites, set())
        self.assertEqual(grown_surfaces, {})

    def test_production_legacy_classifier_inventory_has_no_temporary_entries(
        self,
    ) -> None:
        invalid_entries = tuple(
            entry
            for entry in legacy_stream_classifier_inventory()
            if (
                entry.scope is StreamLegacyInventoryScope.PRODUCTION_RUNTIME
                and entry.classification
                is not StreamLegacySurfaceClassification.REMOVE_NOW
            )
        )

        self.assertEqual(invalid_entries, ())

    def test_public_streaming_return_type_debt_snapshot_does_not_grow(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        current = _source_public_streaming_return_legacy_sites(repository_root)
        ceiling = _PHASE_1_1_PUBLIC_STREAMING_RETURN_DEBT_CEILING

        self.assertEqual(current, ceiling)

    def test_public_streaming_return_guard_detects_aliased_containers(
        self,
    ) -> None:
        tree = parse("""from collections.abc import AsyncIterator as Stream
AssignedStream = AsyncIterator
ChainedStream = AssignedStream

def imported_alias() -> Stream[Token]:
    ...

def assignment_alias() -> AssignedStream[TokenDetail]:
    ...

class AliasedBase(ChainedStream[ReasoningToken]):
    pass
""")

        visitor = _PublicStreamingReturnVisitor("avalan.synthetic")
        visitor.visit(tree)

        self.assertEqual(
            visitor.sites,
            {
                (
                    "avalan.synthetic",
                    "imported_alias",
                    "return",
                ): frozenset({"Token"}),
                (
                    "avalan.synthetic",
                    "assignment_alias",
                    "return",
                ): frozenset({"TokenDetail"}),
                (
                    "avalan.synthetic",
                    "AliasedBase",
                    "base",
                ): frozenset({"ReasoningToken"}),
            },
        )

    def test_public_streaming_return_guard_detects_legacy_item_aliases(
        self,
    ) -> None:
        tree = parse("""from collections.abc import AsyncIterator as Stream
LegacyAnswer = Token | str
LegacyTool = ToolCallToken
LegacyPair = tuple[ReasoningToken, ToolCallToken]

def alias_stream() -> Stream[LegacyAnswer]:
    ...

def chained_alias_stream() -> AsyncIterator[LegacyPair]:
    ...

class AliasedBase(Stream[LegacyTool]):
    pass

class Holder:
    InnerLegacy = TokenDetail

    def stream(self) -> Stream[InnerLegacy | LegacyAnswer]:
        ...
""")

        visitor = _PublicStreamingReturnVisitor("avalan.synthetic")
        visitor.visit(tree)

        self.assertEqual(
            visitor.sites,
            {
                (
                    "avalan.synthetic",
                    "alias_stream",
                    "return",
                ): frozenset({"Token", "str"}),
                (
                    "avalan.synthetic",
                    "chained_alias_stream",
                    "return",
                ): frozenset({"ReasoningToken", "ToolCallToken"}),
                (
                    "avalan.synthetic",
                    "AliasedBase",
                    "base",
                ): frozenset({"ToolCallToken"}),
                (
                    "avalan.synthetic",
                    "Holder.stream",
                    "return",
                ): frozenset({"Token", "TokenDetail", "str"}),
            },
        )

    def test_protocol_projection_state_construction_is_canonical_only(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]

        self.assertEqual(
            _source_protocol_projection_legacy_mapper_sites(repository_root),
            set(),
        )

    def test_protocol_projection_mapper_guard_detects_alias_and_kwargs(
        self,
    ) -> None:
        tree = parse("""def direct(mapper):
    return StreamProjectionState(legacy_item_mapper=mapper)

def alias(mapper):
    State = StreamProjectionState
    return State(legacy_item_mapper=mapper)

def kwargs(mapper):
    options = {"legacy_item_mapper": mapper}
    ProtocolState = ProtocolStreamProjectionState
    return ProtocolState(**options)

def dict_kwargs(mapper):
    return StreamProjectionState(**dict(legacy_item_mapper=mapper))

def make_state(**kwargs):
    return ProtocolStreamProjectionState(**kwargs)

def wrapper_direct_keyword(mapper):
    return make_state(legacy_item_mapper=mapper)

def wrapper_kwargs(mapper):
    options = {"legacy_item_mapper": mapper}
    return make_state(**options)
""")

        visitor = _ProtocolProjectionMapperVisitor("avalan.server.synthetic")
        visitor.visit(tree)

        self.assertEqual(
            {(module, qualname) for module, qualname, _ in visitor.sites},
            {
                ("avalan.server.synthetic", "direct"),
                ("avalan.server.synthetic", "alias"),
                ("avalan.server.synthetic", "kwargs"),
                ("avalan.server.synthetic", "dict_kwargs"),
                ("avalan.server.synthetic", "wrapper_direct_keyword"),
                ("avalan.server.synthetic", "wrapper_kwargs"),
            },
        )
        self.assertEqual(len(visitor.sites), 6)

    def test_protocol_projection_mapper_guard_detects_merged_constant_keys(
        self,
    ) -> None:
        tree = parse("""LEGACY_MAPPER_KEY = "legacy_item_mapper"

def constant_key(mapper):
    options = {LEGACY_MAPPER_KEY: mapper}
    return StreamProjectionState(**options)

def merged_kwargs(mapper):
    legacy_options = {LEGACY_MAPPER_KEY: mapper}
    return ProtocolStreamProjectionState(
        **{"accumulate": False, **legacy_options}
    )

def dict_merge(mapper):
    legacy_options = dict(**{LEGACY_MAPPER_KEY: mapper})
    State = StreamProjectionState
    return State(**dict({"accumulate": False}, **legacy_options))
""")

        visitor = _ProtocolProjectionMapperVisitor("avalan.server.synthetic")
        visitor.visit(tree)

        self.assertEqual(
            {(module, qualname) for module, qualname, _ in visitor.sites},
            {
                ("avalan.server.synthetic", "constant_key"),
                ("avalan.server.synthetic", "merged_kwargs"),
                ("avalan.server.synthetic", "dict_merge"),
            },
        )
        self.assertEqual(len(visitor.sites), 3)

    def test_protocol_projection_mapper_guard_detects_forward_wrapper(
        self,
    ) -> None:
        tree = parse("""def forward_direct(mapper):
    return make_state(legacy_item_mapper=mapper)

def forward_kwargs(mapper):
    options = {"legacy_item_mapper": mapper}
    return make_state(**options)

def make_state(**kwargs):
    return StreamProjectionState(**kwargs)
""")

        visitor = _ProtocolProjectionMapperVisitor("avalan.server.synthetic")
        visitor.visit(tree)

        self.assertEqual(
            {(module, qualname) for module, qualname, _ in visitor.sites},
            {
                ("avalan.server.synthetic", "forward_direct"),
                ("avalan.server.synthetic", "forward_kwargs"),
            },
        )
        self.assertEqual(len(visitor.sites), 2)

    def test_text_stream_canonicalization_guard_detects_indirect_subclasses(
        self,
    ) -> None:
        tree = parse("""class LegacyBase(TextGenerationVendorStream):
    pass

class IndirectLegacy(LegacyBase):
    pass

class CanonicalBase(TextGenerationVendorStream):
    def canonical_stream(self):
        pass

class InheritsCanonical(CanonicalBase):
    pass
""")

        self.assertEqual(
            _inherited_text_stream_canonicalization_sites(
                {"avalan.synthetic": tree}
            ),
            {
                ("avalan.synthetic", "LegacyBase"),
                ("avalan.synthetic", "IndirectLegacy"),
            },
        )

    def test_text_provider_inherited_canonicalization_debt_does_not_grow(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        current = _source_inherited_text_stream_canonicalization_sites(
            repository_root
        )

        self.assertEqual(
            current,
            _PHASE_1_1_INHERITED_TEXT_STREAM_CANONICALIZATION_CEILING,
        )

    def test_legacy_runtime_boundary_inventory_rejects_malformed_entries(
        self,
    ) -> None:
        def make_entry(
            **overrides: object,
        ) -> StreamLegacyRuntimeBoundaryInventoryEntry:
            values: dict[str, object] = {
                "module": "avalan.model.stream",
                "qualname": "TextGenerationStream.__aiter__",
                "surfaces": (StreamLegacySurface.STRING,),
                "classification": StreamLegacySurfaceClassification.REMOVE_NOW,
                "category": StreamLegacyBoundaryCategory.PRODUCER,
                "scope": StreamLegacyInventoryScope.PRODUCTION_RUNTIME,
                "directions": (StreamLegacyBoundaryDirection.EMITS,),
                "owner": "model.stream",
                "removal_condition": "done",
            }
            values.update(overrides)
            return StreamLegacyRuntimeBoundaryInventoryEntry(
                **cast(Any, values)
            )

        invalid_entries = (
            lambda: make_entry(module=""),
            lambda: make_entry(qualname=""),
            lambda: make_entry(surfaces=[StreamLegacySurface.STRING]),
            lambda: make_entry(surfaces=()),
            lambda: make_entry(
                surfaces=(
                    StreamLegacySurface.STRING,
                    StreamLegacySurface.STRING,
                ),
            ),
            lambda: make_entry(surfaces=("str",)),
            lambda: make_entry(classification="remove_now"),
            lambda: make_entry(
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(category="producer"),
            lambda: make_entry(scope="production_runtime"),
            lambda: make_entry(
                module="tests.model.model_stream_contract_test",
                qualname="legacy_rejection_fixture",
                category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=StreamLegacySurfaceClassification.REMOVE_NOW,
            ),
            lambda: make_entry(
                module="avalan.model.stream",
                qualname="legacy_helper_projection",
                category=StreamLegacyBoundaryCategory.HELPER_ONLY,
                scope=StreamLegacyInventoryScope.HELPER_ONLY,
                classification=StreamLegacySurfaceClassification.REMOVE_NOW,
            ),
            lambda: make_entry(
                directions=[StreamLegacyBoundaryDirection.EMITS]
            ),
            lambda: make_entry(directions=()),
            lambda: make_entry(
                directions=(
                    StreamLegacyBoundaryDirection.EMITS,
                    StreamLegacyBoundaryDirection.EMITS,
                ),
            ),
            lambda: make_entry(directions=("emits",)),
            lambda: make_entry(owner=""),
            lambda: make_entry(removal_condition=""),
            lambda: make_entry(
                module="avalan.model.stream",
                qualname="legacy_rejection_fixture",
                category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(
                module="tests.model.model_stream_contract_test",
                qualname="fixture",
                category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(
                module="tests.model.model_stream_contract_test",
                qualname="helper",
                category=StreamLegacyBoundaryCategory.HELPER_ONLY,
                scope=StreamLegacyInventoryScope.HELPER_ONLY,
                classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
            ),
        )

        for build_invalid_entry in invalid_entries:
            with self.subTest(build_entry=build_invalid_entry):
                with self.assertRaises(AssertionError):
                    build_invalid_entry()

        fixture_entry = StreamLegacyRuntimeBoundaryInventoryEntry(
            module="tests.model.model_stream_contract_test",
            qualname="legacy_rejection_fixture",
            surfaces=(StreamLegacySurface.TOKEN,),
            classification=(
                StreamLegacySurfaceClassification.TEMPORARY_COMPATIBILITY_SHIM
            ),
            category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
            scope=StreamLegacyInventoryScope.TEST_FIXTURE,
            directions=(StreamLegacyBoundaryDirection.ACCEPTS,),
            owner="tests",
            removal_condition="negative legacy fixture",
        )
        self.assertIs(
            fixture_entry.scope, StreamLegacyInventoryScope.TEST_FIXTURE
        )

        helper_entry = StreamLegacyRuntimeBoundaryInventoryEntry(
            module="avalan.model.stream",
            qualname="legacy_helper_projection",
            surfaces=(StreamLegacySurface.STRING,),
            classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
            category=StreamLegacyBoundaryCategory.HELPER_ONLY,
            scope=StreamLegacyInventoryScope.HELPER_ONLY,
            directions=(StreamLegacyBoundaryDirection.PROJECTS,),
            owner="tests",
            removal_condition="private migration helper",
        )
        self.assertIs(
            helper_entry.scope, StreamLegacyInventoryScope.HELPER_ONLY
        )

    def test_legacy_classifier_inventory_entries_resolve_to_source(
        self,
    ) -> None:
        inventory_sites = _inventory_legacy_stream_classifier_sites()
        inventory = legacy_stream_classifier_inventory()

        self.assertEqual(inventory_sites, {})
        self.assertEqual(inventory, ())

        with self.assertRaises(AssertionError):
            classify_legacy_stream_classifier("", "qualname")
        with self.assertRaises(AssertionError):
            classify_legacy_stream_classifier("module", "")
        with self.assertRaises(StreamValidationError):
            classify_legacy_stream_classifier(
                "avalan.server.a2a.router",
                "_A2ALegacyStreamAdapter.map",
            )
        with self.assertRaises(StreamValidationError):
            classify_legacy_stream_classifier(
                "avalan.model.stream",
                "_LegacyTokenStreamAdapter.item_from_token",
            )
        fixture_entry = StreamLegacyClassifierInventoryEntry(
            module="tests.model.model_stream_contract_test",
            qualname="legacy_fixture_classifier",
            surfaces=(StreamLegacySurface.TOKEN,),
            classification=(
                StreamLegacySurfaceClassification.TEMPORARY_COMPATIBILITY_SHIM
            ),
            category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
            scope=StreamLegacyInventoryScope.TEST_FIXTURE,
            owner="tests.model.model_stream_contract_test",
            removal_condition="synthetic classification coverage",
        )
        with patch(
            "avalan.model.stream._LEGACY_STREAM_CLASSIFIER_INVENTORY",
            (fixture_entry,),
        ):
            self.assertIs(
                classify_legacy_stream_classifier(
                    "tests.model.model_stream_contract_test",
                    "legacy_fixture_classifier",
                ),
                fixture_entry,
            )

    def test_legacy_classifier_guard_detects_new_direct_classifiers(
        self,
    ) -> None:
        tree = parse("""
def consume(item):
    if isinstance(item, ToolCallToken):
        return item.token
    return None
""")

        sites = _legacy_stream_classifier_sites("avalan.new_consumer", tree)

        self.assertEqual(
            sites,
            {
                ("avalan.new_consumer", "consume"): {
                    StreamLegacySurface.TOOL_CALL_TOKEN
                }
            },
        )
        self.assertNotIn(
            ("avalan.new_consumer", "consume"),
            _inventory_legacy_stream_classifier_sites(),
        )

    def test_legacy_classifier_guard_detects_import_and_assignment_aliases(
        self,
    ) -> None:
        tree = parse(
            """from avalan.entities import ReasoningToken as ThoughtToken
from avalan.entities import Token
from avalan.entities import Token as LegacyToken
from avalan.entities import TokenDetail as LegacyTokenDetail
from avalan.entities import ToolCallToken as LegacyToolCallToken
from avalan.event import Event as LegacyEvent
from asyncio import Event as AsyncEvent

AliasToken = LegacyToken
DirectAliasToken = Token
AliasTokenDetail = LegacyTokenDetail
AliasReasoningToken = ThoughtToken
AliasToolCallToken = LegacyToolCallToken
AliasEvent = LegacyEvent
AsyncAlias = AsyncEvent

def consume(item):
    if isinstance(item, LegacyToken):
        return item
    if isinstance(item, AliasTokenDetail):
        return item
    if isinstance(item, DirectAliasToken):
        return item
    if isinstance(item, AliasReasoningToken):
        return item
    if isinstance(item, AliasToolCallToken):
        return item
    if isinstance(item, AliasEvent):
        return item
    if isinstance(item, AsyncAlias):
        return item
    return None
"""
        )

        sites = _legacy_stream_classifier_sites("avalan.new_consumer", tree)

        self.assertEqual(
            sites,
            {
                ("avalan.new_consumer", "consume"): {
                    StreamLegacySurface.TOKEN,
                    StreamLegacySurface.TOKEN_DETAIL,
                    StreamLegacySurface.REASONING_TOKEN,
                    StreamLegacySurface.TOOL_CALL_TOKEN,
                    StreamLegacySurface.EVENT,
                }
            },
        )

    def test_legacy_classifier_guard_detects_grouped_aliases(
        self,
    ) -> None:
        tree = parse("""
LegacyTextSurfaces = Token | ToolCallToken
LegacyEventSurfaces: object = (ReasoningToken, Event)

def consume(item):
    if isinstance(item, LegacyTextSurfaces):
        return item
    if isinstance(item, LegacyEventSurfaces):
        return item
    return None
""")

        sites = _legacy_stream_classifier_sites("avalan.new_consumer", tree)

        self.assertEqual(
            sites,
            {
                ("avalan.new_consumer", "consume"): {
                    StreamLegacySurface.TOKEN,
                    StreamLegacySurface.TOOL_CALL_TOKEN,
                    StreamLegacySurface.REASONING_TOKEN,
                    StreamLegacySurface.EVENT,
                }
            },
        )

    def test_legacy_classifier_guard_detects_tracked_string_classifiers(
        self,
    ) -> None:
        tree = parse("""
class _LegacyTokenStreamAdapter:
    def events_from_token(self, item):
        if isinstance(item, str):
            return item
        return None

def validate_name(value):
    assert isinstance(value, str)
""")

        sites = _legacy_stream_classifier_sites("avalan.model.stream", tree)

        self.assertEqual(
            sites,
            {
                (
                    "avalan.model.stream",
                    "_LegacyTokenStreamAdapter.events_from_token",
                ): {StreamLegacySurface.STRING}
            },
        )

    def test_legacy_classifier_guard_detects_new_streaming_string_classifiers(
        self,
    ) -> None:
        tree = parse("""
def stream_mapper(item):
    if isinstance(item, str):
        return item
    return None

def project_stream_mapper(item):
    if isinstance(item, str):
        return item
    return None

class _NewStreamAdapter:
    def map(self, item):
        if isinstance(item, str):
            return item
        return None

def validate_name(value):
    assert isinstance(value, str)
""")

        sites = _legacy_stream_classifier_sites("avalan.new_consumer", tree)

        self.assertEqual(
            sites,
            {
                ("avalan.new_consumer", "stream_mapper"): {
                    StreamLegacySurface.STRING
                },
                ("avalan.new_consumer", "project_stream_mapper"): {
                    StreamLegacySurface.STRING
                },
                ("avalan.new_consumer", "_NewStreamAdapter.map"): {
                    StreamLegacySurface.STRING
                },
            },
        )
        self.assertNotIn(
            ("avalan.new_consumer", "stream_mapper"),
            _inventory_legacy_stream_classifier_sites(),
        )
        self.assertNotIn(
            ("avalan.new_consumer", "project_stream_mapper"),
            _inventory_legacy_stream_classifier_sites(),
        )
        self.assertNotIn(
            ("avalan.new_consumer", "_NewStreamAdapter.map"),
            _inventory_legacy_stream_classifier_sites(),
        )

    def test_legacy_classifier_inventory_rejects_malformed_entries(
        self,
    ) -> None:
        def make_entry(
            **overrides: object,
        ) -> StreamLegacyClassifierInventoryEntry:
            values: dict[str, object] = {
                "module": "avalan.model.stream",
                "qualname": "function",
                "surfaces": (StreamLegacySurface.TOKEN,),
                "classification": StreamLegacySurfaceClassification.REMOVE_NOW,
                "category": StreamLegacyBoundaryCategory.PRODUCER,
                "scope": StreamLegacyInventoryScope.PRODUCTION_RUNTIME,
                "owner": "model.stream",
                "removal_condition": "done",
            }
            values.update(overrides)
            return StreamLegacyClassifierInventoryEntry(**cast(Any, values))

        invalid_entries = (
            lambda: make_entry(module=""),
            lambda: make_entry(qualname=""),
            lambda: make_entry(surfaces=[StreamLegacySurface.TOKEN]),
            lambda: make_entry(surfaces=()),
            lambda: make_entry(
                surfaces=(
                    StreamLegacySurface.TOKEN,
                    StreamLegacySurface.TOKEN,
                ),
            ),
            lambda: make_entry(surfaces=("Token",)),
            lambda: make_entry(classification="temporary"),
            lambda: make_entry(category="producer"),
            lambda: make_entry(scope="production_runtime"),
            lambda: make_entry(
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(
                category=StreamLegacyBoundaryCategory.TEST_FIXTURE
            ),
            lambda: make_entry(
                category=StreamLegacyBoundaryCategory.PRODUCER,
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(
                category=StreamLegacyBoundaryCategory.PRODUCER,
                scope=StreamLegacyInventoryScope.HELPER_ONLY,
                classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
            ),
            lambda: make_entry(
                category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=StreamLegacySurfaceClassification.REMOVE_NOW,
            ),
            lambda: make_entry(
                module="avalan.model.stream",
                qualname="legacy_rejection_fixture",
                category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(
                module="tests.model.model_stream_contract_test",
                qualname="fixture",
                category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
                scope=StreamLegacyInventoryScope.TEST_FIXTURE,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
            ),
            lambda: make_entry(
                module="tests.model.model_stream_contract_test",
                qualname="helper",
                category=StreamLegacyBoundaryCategory.HELPER_ONLY,
                scope=StreamLegacyInventoryScope.HELPER_ONLY,
                classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
            ),
            lambda: make_entry(owner=""),
            lambda: make_entry(removal_condition=""),
        )

        for build_invalid_entry in invalid_entries:
            with self.subTest(build_entry=build_invalid_entry):
                with self.assertRaises(AssertionError):
                    build_invalid_entry()

        fixture_entry = StreamLegacyClassifierInventoryEntry(
            module="tests.model.model_stream_contract_test",
            qualname="legacy_rejection_fixture",
            surfaces=(StreamLegacySurface.TOKEN,),
            classification=(
                StreamLegacySurfaceClassification.TEMPORARY_COMPATIBILITY_SHIM
            ),
            category=StreamLegacyBoundaryCategory.TEST_FIXTURE,
            scope=StreamLegacyInventoryScope.TEST_FIXTURE,
            owner="tests",
            removal_condition="negative legacy fixture",
        )
        self.assertIs(
            fixture_entry.scope, StreamLegacyInventoryScope.TEST_FIXTURE
        )

        helper_entry = StreamLegacyClassifierInventoryEntry(
            module="avalan.model.stream",
            qualname="legacy_helper_projection",
            surfaces=(StreamLegacySurface.STRING,),
            classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
            category=StreamLegacyBoundaryCategory.HELPER_ONLY,
            scope=StreamLegacyInventoryScope.HELPER_ONLY,
            owner="tests",
            removal_condition="private migration helper",
        )
        self.assertIs(
            helper_entry.scope, StreamLegacyInventoryScope.HELPER_ONLY
        )

    def test_runtime_legacy_stream_helpers_are_removed(
        self,
    ) -> None:
        from avalan.model import stream as stream_module

        for name in (
            "_LegacyTokenStreamAdapter",
            "canonical_item_from_token",
            "stream_consumer_projection_from_token",
            "_normalize_local_stream",
            "token_text",
            "_token_metadata",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(stream_module, name))

    def test_stream_projection_display_token_uses_projection_metadata(
        self,
    ) -> None:
        detail = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="detail",
                metadata=stream_token_metadata(
                    token_id=7,
                    probability=0.75,
                    step=2,
                    probability_distribution="softmax",
                    candidates=(("candidate", 8, 0.25),),
                ),
            )
        )
        display_token = stream_projection_display_token(detail)

        self.assertIsInstance(display_token, TokenDetail)
        assert isinstance(display_token, TokenDetail)
        self.assertEqual(display_token.id, 7)
        self.assertEqual(display_token.token, "detail")
        self.assertEqual(display_token.probability, 0.75)
        self.assertEqual(display_token.step, 2)
        self.assertEqual(display_token.probability_distribution, "softmax")
        self.assertEqual(
            display_token.tokens,
            [Token(id=8, token="candidate", probability=0.25)],
        )

        answer = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="a",
                metadata=stream_token_metadata(token_id=9),
            )
        )
        self.assertEqual(
            stream_projection_display_token(answer),
            Token(id=9, token="a"),
        )

    def test_stream_projection_display_token_rejects_semantic_only_items(
        self,
    ) -> None:
        answer = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            )
        )
        tool = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
                text_delta="tool",
                metadata=stream_token_metadata(token_id=2),
            )
        )
        detail = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
                metadata={
                    "tokens": [
                        {"token": "candidate", "token_id": 4},
                        {"token_id": 5},
                        "bad",
                    ],
                },
            )
        )
        invalid_detail = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=4,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
                metadata={"tokens": "bad"},
            )
        )

        self.assertIsNone(stream_projection_display_token(answer))
        self.assertIsNone(stream_projection_display_token(tool))
        display_token = stream_projection_display_token(detail)
        self.assertIsInstance(display_token, TokenDetail)
        assert isinstance(display_token, TokenDetail)
        self.assertEqual(
            display_token.tokens, [Token(id=4, token="candidate")]
        )
        invalid_display_token = stream_projection_display_token(invalid_detail)
        self.assertIsInstance(invalid_display_token, TokenDetail)
        assert isinstance(invalid_display_token, TokenDetail)
        self.assertIsNone(invalid_display_token.tokens)
        with self.assertRaises(AssertionError):
            stream_projection_display_token(object())  # type: ignore[arg-type]
