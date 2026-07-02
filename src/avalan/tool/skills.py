from ..compat import override
from ..entities import ToolCallContext, ToolCapabilities
from ..event import EventType
from ..skill import (
    SkillBootstrapPromptSettings,
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillMatchLimits,
    SkillMatchResult,
    SkillMetadata,
    SkillModelValue,
    SkillResourceReader,
    SkillResponseEnvelope,
    SkillStatus,
    check_skill_registry_read,
    match_skill_registry,
)
from ..skill.entities import model_dict
from ..skill.observability import (
    SkillEventPublisher,
    assert_skill_event_publisher,
    emit_skill_audit_event,
    skill_audit_content_fields,
    skill_audit_context_fields,
    skill_audit_diagnostics_fields,
    skill_audit_registry_fields,
)
from ..skill.registry import SkillRegistry
from . import Tool, ToolSet

from collections.abc import Mapping
from contextlib import AsyncExitStack
from re import fullmatch

_MAIN_RESOURCE_ID = "main"
_LOGICAL_ID_PATTERN = r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*"
_BYTE_COUNT_KEYS = frozenset(
    {
        "end_byte",
        "limit_bytes",
        "line_count",
        "max_bytes_per_read",
        "max_lines_per_read",
        "offset_bytes",
        "size_bytes",
        "start_byte",
    }
)
_SOURCE_LABEL_KEYS = frozenset({"candidates", "source_label", "source_id"})
_READ_ONLY_PARALLEL_SAFE = ToolCapabilities(
    side_effecting=False,
    parallel_safe=True,
)
_READ_ONLY_CURSORING = ToolCapabilities(
    side_effecting=False,
    parallel_safe=False,
)


class ListSkillsTool(Tool):
    """Return compact metadata for available skills.

    Args:
        query: Optional text that skill IDs, tags, or descriptions must match.
        tags: Optional tags every returned skill must include.
        source_label: Optional source label to filter by.
        status: Optional status value to filter by.
        usable_only: Whether to return only usable skills.

    Returns:
        Structured skill response envelope without skill bodies.
    """

    tool_capabilities = _READ_ONLY_PARALLEL_SAFE

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        event_manager: SkillEventPublisher | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        assert_skill_event_publisher(event_manager)
        self.__name__ = "list"
        self._registry = registry
        self._event_manager = event_manager

    async def __call__(
        self,
        context: ToolCallContext,
        query: str = "",
        tags: list[str] | None = None,
        source_label: str | None = None,
        status: str | None = None,
        usable_only: bool = True,
    ) -> dict[str, SkillModelValue]:
        registry = _context_registry(context, self._registry)
        disabled = _disabled_envelope(registry, path="skills.list")
        if disabled is not None:
            return _skills_model_dict(registry, disabled)
        parsed_status = _parse_status(status)
        if parsed_status is None and status is not None:
            return _skills_model_dict(
                registry,
                _invalid_status_envelope(
                    registry,
                    status,
                    path="skills.list.status",
                ),
            )
        tag_filter = _tag_filter(tags)
        if tag_filter is None and tags is not None:
            return _skills_model_dict(
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.list.tags",
                    reason="invalid_tags",
                ),
            )
        if not _is_optional_logical_id(source_label):
            return _skills_model_dict(
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.list.source_label",
                    reason="invalid_source_label",
                ),
            )
        if source_label is not None and not _include_source_labels(registry):
            return _skills_model_dict(
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.list.source_label",
                    reason="source_labels_hidden",
                ),
            )
        envelope = _list_skills(
            registry,
            query=query,
            tags=tag_filter or (),
            source_label=source_label,
            status=parsed_status,
            usable_only=usable_only,
        )
        return _skills_model_dict(registry, envelope)


class MatchSkillsTool(Tool):
    """Return ranked skill metadata candidates for a task.

    Args:
        query: Task text or compact task description to match.
        tags: Optional tags every returned skill must include.
        source_label: Optional source label to filter by.
        status: Optional status value to filter by.
        usable_only: Whether to match only usable skills.
        max_results: Optional maximum number of candidates to return.

    Returns:
        Structured skill response envelope without skill bodies.
    """

    tool_capabilities = _READ_ONLY_PARALLEL_SAFE

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        event_manager: SkillEventPublisher | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        assert_skill_event_publisher(event_manager)
        self.__name__ = "match"
        self._registry = registry
        self._event_manager = event_manager

    async def __call__(
        self,
        context: ToolCallContext,
        query: str = "",
        tags: list[str] | None = None,
        source_label: str | None = None,
        status: str | None = None,
        usable_only: bool = True,
        max_results: int | None = None,
    ) -> dict[str, SkillModelValue]:
        registry = _context_registry(context, self._registry)
        disabled = _disabled_envelope(registry, path="skills.match")
        if disabled is not None:
            return _skills_model_dict(registry, disabled)
        parsed_status = _parse_status(status)
        if parsed_status is None and status is not None:
            return _skills_model_dict(
                registry,
                _invalid_status_envelope(
                    registry,
                    status,
                    path="skills.match.status",
                ),
            )
        tag_filter = _tag_filter(tags)
        if tag_filter is None and tags is not None:
            return _skills_model_dict(
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.match.tags",
                    reason="invalid_tags",
                ),
            )
        if not _is_optional_logical_id(source_label):
            return _skills_model_dict(
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.match.source_label",
                    reason="invalid_source_label",
                ),
            )
        include_source_labels = _include_source_labels(registry)
        if source_label is not None and not include_source_labels:
            return _skills_model_dict(
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.match.source_label",
                    reason="source_labels_hidden",
                ),
            )
        if not _is_optional_positive_int(max_results):
            return _skills_model_dict(
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.match.max_results",
                    reason="invalid_max_results",
                ),
            )
        await _emit_match_query_event(
            self._event_manager,
            context,
            registry,
            query=query,
            tag_count=len(tag_filter or ()),
            source_label=source_label,
            status=parsed_status,
            usable_only=usable_only,
            max_results=max_results,
        )
        envelope = await match_skill_registry(
            registry,
            query=query,
            tags=tag_filter or (),
            source_label=source_label,
            status=parsed_status,
            usable_only=usable_only,
            max_results=max_results,
            match_limits=SkillMatchLimits(),
            include_source_labels=include_source_labels,
        )
        await _emit_match_result_event(
            self._event_manager,
            context,
            registry,
            envelope,
        )
        return _skills_model_dict(registry, envelope)


class ReadSkillTool(Tool):
    """Return bounded content for an authorized skill resource.

    Args:
        skill: Skill ID or unambiguous skill name to read.
        resource_id: Logical resource ID to read.
        source_label: Optional source label to disambiguate a skill.
        cursor_id: Optional read cursor returned by a previous read.

    Returns:
        Structured skill response envelope with bounded resource content.
    """

    tool_capabilities = _READ_ONLY_CURSORING

    def __init__(
        self,
        registry: SkillRegistry,
        reader: SkillResourceReader,
        *,
        event_manager: SkillEventPublisher | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        assert isinstance(reader, SkillResourceReader)
        assert_skill_event_publisher(event_manager)
        self.__name__ = "read"
        self._registry = registry
        self._reader = reader
        self._event_manager = event_manager

    async def __call__(
        self,
        context: ToolCallContext,
        skill: str | None = None,
        resource_id: str = _MAIN_RESOURCE_ID,
        source_label: str | None = None,
        cursor_id: str | None = None,
    ) -> dict[str, SkillModelValue]:
        registry = _context_registry(context, self._registry)
        disabled = _disabled_envelope(registry, path="skills.read")
        if disabled is not None:
            return await _skills_read_response(
                self._event_manager,
                context,
                registry,
                disabled,
                skill=skill,
                resource_id=resource_id,
                source_label=source_label,
            )
        if source_label is not None and not _include_source_labels(registry):
            return await _skills_read_response(
                self._event_manager,
                context,
                registry,
                _invalid_argument_envelope(
                    registry,
                    path="skills.read.source_label",
                    reason="source_labels_hidden",
                ),
                skill=skill,
                resource_id=resource_id,
                source_label=source_label,
            )
        envelope = await self._reader.read(
            registry,
            skill,
            resource_id=resource_id,
            source_label=source_label,
            cursor_id=cursor_id,
        )
        return await _skills_read_response(
            self._event_manager,
            context,
            registry,
            envelope,
            skill=skill,
            resource_id=resource_id,
            source_label=source_label,
        )


class CheckSkillTool(Tool):
    """Return skill diagnostics without reading resource bodies.

    Args:
        skill: Skill ID or unambiguous skill name to check.
        resource_id: Logical resource ID to check.
        source_label: Optional source label to disambiguate a skill.

    Returns:
        Structured skill response envelope without resource content.
    """

    tool_capabilities = _READ_ONLY_PARALLEL_SAFE

    def __init__(
        self,
        registry: SkillRegistry,
        reader: SkillResourceReader,
        *,
        event_manager: SkillEventPublisher | None = None,
    ) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        assert isinstance(reader, SkillResourceReader)
        assert_skill_event_publisher(event_manager)
        self.__name__ = "check"
        self._registry = registry
        self._reader = reader
        self._event_manager = event_manager

    async def __call__(
        self,
        context: ToolCallContext,
        skill: str,
        resource_id: str = _MAIN_RESOURCE_ID,
        source_label: str | None = None,
    ) -> dict[str, SkillModelValue]:
        registry = _context_registry(context, self._registry)
        disabled = _disabled_envelope(registry, path="skills.check")
        if disabled is not None:
            await _emit_check_diagnostics_event(
                self._event_manager,
                context,
                registry,
                disabled,
                skill=skill,
                resource_id=resource_id,
                source_label=source_label,
            )
            return _skills_model_dict(registry, disabled)
        if source_label is not None and not _include_source_labels(registry):
            envelope = _invalid_argument_envelope(
                registry,
                path="skills.check.source_label",
                reason="source_labels_hidden",
            )
            await _emit_check_diagnostics_event(
                self._event_manager,
                context,
                registry,
                envelope,
                skill=skill,
                resource_id=resource_id,
                source_label=source_label,
            )
            return _skills_model_dict(registry, envelope)
        envelope = await check_skill_registry_read(
            registry,
            skill,
            resource_id=resource_id,
            source_label=source_label,
            reader=self._reader,
        )
        await _emit_check_diagnostics_event(
            self._event_manager,
            context,
            registry,
            envelope,
            skill=skill,
            resource_id=resource_id,
            source_label=source_label,
        )
        return _skills_model_dict(registry, envelope)


class SkillsToolSet(ToolSet):
    @override
    def __init__(
        self,
        registry: SkillRegistry,
        *,
        bootstrap_enabled: bool = True,
        bootstrap_prompt_settings: SkillBootstrapPromptSettings | None = None,
        exit_stack: AsyncExitStack | None = None,
        event_manager: SkillEventPublisher | None = None,
        namespace: str | None = "skills",
    ) -> None:
        assert isinstance(bootstrap_enabled, bool)
        if bootstrap_prompt_settings is None and registry.settings is not None:
            bootstrap_prompt_settings = registry.settings.bootstrap_prompt
        if bootstrap_prompt_settings is not None:
            assert isinstance(
                bootstrap_prompt_settings,
                SkillBootstrapPromptSettings,
            )
        assert_skill_event_publisher(event_manager)
        self.registry = registry
        self.bootstrap_enabled = bootstrap_enabled
        self.bootstrap_prompt_settings = bootstrap_prompt_settings
        reader = SkillResourceReader()
        tools = [
            ListSkillsTool(registry, event_manager=event_manager),
            MatchSkillsTool(registry, event_manager=event_manager),
            ReadSkillTool(registry, reader, event_manager=event_manager),
            CheckSkillTool(registry, reader, event_manager=event_manager),
        ]
        super().__init__(
            exit_stack=exit_stack,
            namespace=namespace,
            tools=tools,
        )


def _context_registry(
    context: ToolCallContext,
    default: SkillRegistry,
) -> SkillRegistry:
    registry = context.skills_registry
    if isinstance(registry, SkillRegistry):
        return registry
    return default


def _disabled_envelope(
    registry: SkillRegistry,
    *,
    path: str,
) -> SkillResponseEnvelope | None:
    settings = registry.settings
    if settings is None or settings.enabled:
        return None
    diagnostic = SkillDiagnosticInfo(
        code=SkillDiagnosticCode.DISABLED,
        status=SkillStatus.DISABLED,
        message="Trusted skills settings are disabled.",
        path=path,
        hint="Do not use skills tools when trusted skills are off.",
        details={"reason": "settings_disabled"},
    )
    return SkillResponseEnvelope(
        status=diagnostic.status,
        registry_version=registry.registry_version,
        diagnostics=(diagnostic,),
    )


def _list_skills(
    registry: SkillRegistry,
    *,
    query: str,
    tags: tuple[str, ...],
    source_label: str | None,
    status: SkillStatus | None,
    usable_only: bool,
) -> SkillResponseEnvelope:
    assert isinstance(query, str)
    assert isinstance(source_label, str | None)
    assert isinstance(usable_only, bool)
    include_source_labels = _include_source_labels(registry)
    candidates = registry.usable_metadata if usable_only else registry.metadata
    normalized_query = query.casefold().strip()
    items = tuple(
        metadata
        for metadata in candidates
        if _metadata_matches(
            _metadata_model_dict(
                metadata.as_model_dict(),
                include_source_labels=include_source_labels,
            ),
            query=normalized_query,
            tags=tags,
            source_label=source_label,
            status=status,
        )
    )
    if items:
        return SkillResponseEnvelope(
            status=SkillStatus.OK,
            registry_version=registry.registry_version,
            items=items,
            diagnostics=registry.diagnostics,
        )
    diagnostics = registry.diagnostics or (
        SkillDiagnosticInfo(
            code=SkillDiagnosticCode.NO_MATCH,
            status=SkillStatus.EMPTY,
            message="No skills match the list filters.",
            path="skills.list",
            hint=(
                "Call skills.match with a broader query or continue without "
                "a skill."
            ),
        ),
    )
    return SkillResponseEnvelope(
        status=diagnostics[0].status,
        registry_version=registry.registry_version,
        diagnostics=diagnostics,
    )


def _include_source_labels(registry: SkillRegistry) -> bool:
    assert isinstance(registry, SkillRegistry)
    return (
        registry.settings is None
        or registry.settings.privacy.include_source_labels
    )


async def _emit_match_query_event(
    event_manager: SkillEventPublisher | None,
    context: ToolCallContext,
    registry: SkillRegistry,
    *,
    query: str,
    tag_count: int,
    source_label: str | None,
    status: SkillStatus | None,
    usable_only: bool,
    max_results: int | None,
) -> None:
    fields: dict[str, object] = {
        **skill_audit_context_fields(context, tool_name="skills.match"),
        **skill_audit_registry_fields(registry.registry_version),
        "status": "evaluated",
        "query_character_count": len(query),
        "query_byte_count": len(query.encode("utf-8")),
        "tag_count": tag_count,
        "source_label": source_label,
        "filter_status": status.value if status is not None else None,
        "usable_only": usable_only,
        "max_results": max_results,
    }
    await emit_skill_audit_event(
        event_manager,
        registry.settings,
        EventType.SKILL_MATCH_QUERY_EVALUATED,
        fields,
    )


async def _emit_match_result_event(
    event_manager: SkillEventPublisher | None,
    context: ToolCallContext,
    registry: SkillRegistry,
    envelope: SkillResponseEnvelope,
) -> None:
    event_type = _match_result_event_type(envelope)
    fields: dict[str, object] = {
        **skill_audit_context_fields(context, tool_name="skills.match"),
        **skill_audit_registry_fields(
            envelope.registry_version,
            status=envelope.status,
        ),
        "candidate_count": len(envelope.items),
        "skill_ids": _envelope_skill_ids(envelope),
    }
    fields.update(skill_audit_diagnostics_fields(envelope.diagnostics))
    await emit_skill_audit_event(
        event_manager,
        registry.settings,
        event_type,
        fields,
    )


def _match_result_event_type(
    envelope: SkillResponseEnvelope,
) -> EventType:
    if envelope.status is SkillStatus.AMBIGUOUS:
        return EventType.SKILL_MATCH_AMBIGUOUS
    if envelope.items:
        return EventType.SKILL_MATCH_CANDIDATES_RETURNED
    return EventType.SKILL_MATCH_EMPTY


async def _skills_read_response(
    event_manager: SkillEventPublisher | None,
    context: ToolCallContext,
    registry: SkillRegistry,
    envelope: SkillResponseEnvelope,
    *,
    skill: str | None,
    resource_id: str,
    source_label: str | None,
) -> dict[str, SkillModelValue]:
    await _emit_read_event(
        event_manager,
        context,
        registry,
        envelope,
        skill=skill,
        resource_id=resource_id,
        source_label=source_label,
    )
    return _skills_model_dict(registry, envelope)


async def _emit_read_event(
    event_manager: SkillEventPublisher | None,
    context: ToolCallContext,
    registry: SkillRegistry,
    envelope: SkillResponseEnvelope,
    *,
    skill: str | None,
    resource_id: str,
    source_label: str | None,
) -> None:
    fields: dict[str, object] = {
        **skill_audit_context_fields(context, tool_name="skills.read"),
        **skill_audit_registry_fields(
            envelope.registry_version,
            status=envelope.status,
        ),
        "source_label": source_label,
        "skill_id": skill if skill and _is_logical_id(skill) else None,
        "resource_id": resource_id,
    }
    if envelope.content is not None:
        fields.update(
            skill_audit_content_fields(
                envelope.content,
                envelope.provenance[0] if envelope.provenance else None,
            )
        )
    fields.update(skill_audit_diagnostics_fields(envelope.diagnostics))
    await emit_skill_audit_event(
        event_manager,
        registry.settings,
        _read_event_type(envelope),
        fields,
    )


def _read_event_type(envelope: SkillResponseEnvelope) -> EventType:
    if envelope.content is not None:
        if envelope.content.truncated:
            return EventType.SKILL_READ_TRUNCATED
        return EventType.SKILL_READ_ALLOWED
    if envelope.status is SkillStatus.STALE:
        return EventType.SKILL_READ_STALE
    if envelope.diagnostics and (
        envelope.diagnostics[0].code is SkillDiagnosticCode.RESOURCE_MISSING
    ):
        return EventType.SKILL_READ_DELETED
    if envelope.status in {
        SkillStatus.DISABLED,
        SkillStatus.POLICY_DENIED,
    }:
        return EventType.SKILL_READ_DENIED
    return EventType.SKILL_READ_BLOCKED


async def _emit_check_diagnostics_event(
    event_manager: SkillEventPublisher | None,
    context: ToolCallContext,
    registry: SkillRegistry,
    envelope: SkillResponseEnvelope,
    *,
    skill: str,
    resource_id: str,
    source_label: str | None,
) -> None:
    if not envelope.diagnostics:
        return
    fields: dict[str, object] = {
        **skill_audit_context_fields(context, tool_name="skills.check"),
        **skill_audit_registry_fields(
            envelope.registry_version,
            status=envelope.status,
        ),
        "source_label": source_label,
        "skill_id": skill if _is_logical_id(skill) else None,
        "resource_id": resource_id,
    }
    fields.update(skill_audit_diagnostics_fields(envelope.diagnostics))
    await emit_skill_audit_event(
        event_manager,
        registry.settings,
        EventType.SKILL_CHECK_DIAGNOSTICS_PRODUCED,
        fields,
    )


def _envelope_skill_ids(
    envelope: SkillResponseEnvelope,
) -> list[str]:
    skill_ids: list[str] = []
    for item in envelope.items[:16]:
        metadata = (
            item.metadata if isinstance(item, SkillMatchResult) else item
        )
        assert isinstance(metadata, SkillMetadata)
        skill_ids.append(metadata.skill_id)
    return skill_ids


def _metadata_model_dict(
    metadata: dict[str, SkillModelValue],
    *,
    include_source_labels: bool,
) -> dict[str, SkillModelValue]:
    assert isinstance(metadata, dict)
    if include_source_labels:
        return metadata
    return {
        key: value
        for key, value in metadata.items()
        if key not in _SOURCE_LABEL_KEYS
    }


def _skills_model_dict(
    registry: SkillRegistry,
    envelope: SkillResponseEnvelope,
) -> dict[str, SkillModelValue]:
    assert isinstance(registry, SkillRegistry)
    assert isinstance(envelope, SkillResponseEnvelope)
    value = envelope.as_model_dict()
    settings = registry.settings
    if settings is None:
        return value
    filtered = _filter_skill_model_value(
        value,
        include_source_labels=settings.privacy.include_source_labels,
        include_authority=settings.privacy.include_authority,
        include_diagnostic_paths=settings.privacy.include_diagnostic_paths,
        include_diagnostics=settings.observability.include_diagnostics,
        include_byte_counts=settings.observability.include_byte_counts,
    )
    assert isinstance(filtered, Mapping)
    return model_dict(filtered)


def _filter_skill_model_value(
    value: SkillModelValue,
    *,
    include_source_labels: bool,
    include_authority: bool,
    include_diagnostic_paths: bool,
    include_diagnostics: bool,
    include_byte_counts: bool,
) -> SkillModelValue:
    if isinstance(value, Mapping):
        result: dict[str, SkillModelValue] = {}
        for key, item in value.items():
            if _drop_skill_model_key(
                key,
                include_source_labels=include_source_labels,
                include_authority=include_authority,
                include_diagnostic_paths=include_diagnostic_paths,
                include_diagnostics=include_diagnostics,
                include_byte_counts=include_byte_counts,
            ):
                continue
            result[key] = _filter_skill_model_value(
                item,
                include_source_labels=include_source_labels,
                include_authority=include_authority,
                include_diagnostic_paths=include_diagnostic_paths,
                include_diagnostics=include_diagnostics,
                include_byte_counts=include_byte_counts,
            )
        return model_dict(result)
    if isinstance(value, tuple):
        return tuple(
            _filter_skill_model_value(
                item,
                include_source_labels=include_source_labels,
                include_authority=include_authority,
                include_diagnostic_paths=include_diagnostic_paths,
                include_diagnostics=include_diagnostics,
                include_byte_counts=include_byte_counts,
            )
            for item in value
        )
    return value


def _drop_skill_model_key(
    key: str,
    *,
    include_source_labels: bool,
    include_authority: bool,
    include_diagnostic_paths: bool,
    include_diagnostics: bool,
    include_byte_counts: bool,
) -> bool:
    if not include_source_labels and key in _SOURCE_LABEL_KEYS:
        return True
    if not include_authority and key == "authority":
        return True
    if not include_diagnostics and key == "diagnostics":
        return True
    if not include_diagnostic_paths and key == "path":
        return True
    if not include_byte_counts and key in _BYTE_COUNT_KEYS:
        return True
    return False


def _metadata_matches(
    metadata: dict[str, SkillModelValue],
    *,
    query: str,
    tags: tuple[str, ...],
    source_label: str | None,
    status: SkillStatus | None,
) -> bool:
    if (
        source_label is not None
        and metadata.get("source_label") != source_label
    ):
        return False
    if status is not None and metadata["status"] != status.value:
        return False
    metadata_tags_value = metadata.get("tags", ())
    metadata_tags = (
        metadata_tags_value if isinstance(metadata_tags_value, tuple) else ()
    )
    if tags and not all(tag in metadata_tags for tag in tags):
        return False
    if not query:
        return True
    searchable = " ".join(
        str(value) for key, value in metadata.items() if key != "resources"
    ).casefold()
    return query in searchable


def _parse_status(status: str | None) -> SkillStatus | None:
    if status is None:
        return None
    try:
        return SkillStatus(status)
    except ValueError:
        return None


def _tag_filter(tags: list[str] | None) -> tuple[str, ...] | None:
    if tags is None:
        return ()
    if not all(isinstance(tag, str) and _is_logical_id(tag) for tag in tags):
        return None
    return tuple(tags)


def _is_optional_logical_id(value: str | None) -> bool:
    return value is None or _is_logical_id(value)


def _is_optional_positive_int(value: int | None) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


def _is_logical_id(value: str) -> bool:
    return fullmatch(_LOGICAL_ID_PATTERN, value) is not None


def _invalid_status_envelope(
    registry: SkillRegistry,
    status: str,
    *,
    path: str,
) -> SkillResponseEnvelope:
    return SkillResponseEnvelope(
        status=SkillStatus.POLICY_DENIED,
        registry_version=registry.registry_version,
        diagnostics=(
            SkillDiagnosticInfo(
                code=SkillDiagnosticCode.POLICY_DENIED,
                status=SkillStatus.POLICY_DENIED,
                message="The requested status filter is not supported.",
                path=path,
                hint="Use one of the documented skills status values.",
                details=model_dict({"reason": "invalid_status"}),
            ),
        ),
    )


def _invalid_argument_envelope(
    registry: SkillRegistry,
    *,
    path: str,
    reason: str,
) -> SkillResponseEnvelope:
    return SkillResponseEnvelope(
        status=SkillStatus.POLICY_DENIED,
        registry_version=registry.registry_version,
        diagnostics=(
            SkillDiagnosticInfo(
                code=SkillDiagnosticCode.POLICY_DENIED,
                status=SkillStatus.POLICY_DENIED,
                message="The requested skills tool argument is not supported.",
                path=path,
                hint=(
                    "Use logical IDs returned by skills.list or skills.match."
                ),
                details=model_dict({"reason": reason}),
            ),
        ),
    )
