from ..compat import override
from ..entities import ToolCallContext, ToolCapabilities
from ..skill import (
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillMatchLimits,
    SkillModelValue,
    SkillResourceReader,
    SkillResponseEnvelope,
    SkillStatus,
    check_skill_registry_read,
    match_skill_registry,
)
from ..skill.entities import model_dict
from ..skill.registry import SkillRegistry
from . import Tool, ToolSet

from contextlib import AsyncExitStack
from re import fullmatch

_MAIN_RESOURCE_ID = "main"
_LOGICAL_ID_PATTERN = r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*"
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

    def __init__(self, registry: SkillRegistry) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        self.__name__ = "list"
        self._registry = registry

    async def __call__(
        self,
        context: ToolCallContext,
        query: str = "",
        tags: list[str] | None = None,
        source_label: str | None = None,
        status: str | None = None,
        usable_only: bool = True,
    ) -> dict[str, SkillModelValue]:
        del context
        parsed_status = _parse_status(status)
        if parsed_status is None and status is not None:
            return _invalid_status_envelope(
                self._registry,
                status,
                path="skills.list.status",
            ).as_model_dict()
        tag_filter = _tag_filter(tags)
        if tag_filter is None and tags is not None:
            return _invalid_argument_envelope(
                self._registry,
                path="skills.list.tags",
                reason="invalid_tags",
            ).as_model_dict()
        if not _is_optional_logical_id(source_label):
            return _invalid_argument_envelope(
                self._registry,
                path="skills.list.source_label",
                reason="invalid_source_label",
            ).as_model_dict()
        envelope = _list_skills(
            self._registry,
            query=query,
            tags=tag_filter or (),
            source_label=source_label,
            status=parsed_status,
            usable_only=usable_only,
        )
        return envelope.as_model_dict()


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

    def __init__(self, registry: SkillRegistry) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        self.__name__ = "match"
        self._registry = registry

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
        del context
        parsed_status = _parse_status(status)
        if parsed_status is None and status is not None:
            return _invalid_status_envelope(
                self._registry,
                status,
                path="skills.match.status",
            ).as_model_dict()
        tag_filter = _tag_filter(tags)
        if tag_filter is None and tags is not None:
            return _invalid_argument_envelope(
                self._registry,
                path="skills.match.tags",
                reason="invalid_tags",
            ).as_model_dict()
        if not _is_optional_logical_id(source_label):
            return _invalid_argument_envelope(
                self._registry,
                path="skills.match.source_label",
                reason="invalid_source_label",
            ).as_model_dict()
        if not _is_optional_positive_int(max_results):
            return _invalid_argument_envelope(
                self._registry,
                path="skills.match.max_results",
                reason="invalid_max_results",
            ).as_model_dict()
        envelope = await match_skill_registry(
            self._registry,
            query=query,
            tags=tag_filter or (),
            source_label=source_label,
            status=parsed_status,
            usable_only=usable_only,
            max_results=max_results,
            match_limits=SkillMatchLimits(),
        )
        return envelope.as_model_dict()


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
    ) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        assert isinstance(reader, SkillResourceReader)
        self.__name__ = "read"
        self._registry = registry
        self._reader = reader

    async def __call__(
        self,
        context: ToolCallContext,
        skill: str | None = None,
        resource_id: str = _MAIN_RESOURCE_ID,
        source_label: str | None = None,
        cursor_id: str | None = None,
    ) -> dict[str, SkillModelValue]:
        del context
        envelope = await self._reader.read(
            self._registry,
            skill,
            resource_id=resource_id,
            source_label=source_label,
            cursor_id=cursor_id,
        )
        return envelope.as_model_dict()


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
    ) -> None:
        super().__init__()
        assert isinstance(registry, SkillRegistry)
        assert isinstance(reader, SkillResourceReader)
        self.__name__ = "check"
        self._registry = registry
        self._reader = reader

    async def __call__(
        self,
        context: ToolCallContext,
        skill: str,
        resource_id: str = _MAIN_RESOURCE_ID,
        source_label: str | None = None,
    ) -> dict[str, SkillModelValue]:
        del context
        envelope = await check_skill_registry_read(
            self._registry,
            skill,
            resource_id=resource_id,
            source_label=source_label,
            reader=self._reader,
        )
        return envelope.as_model_dict()


class SkillsToolSet(ToolSet):
    @override
    def __init__(
        self,
        registry: SkillRegistry,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = "skills",
    ) -> None:
        reader = SkillResourceReader()
        tools = [
            ListSkillsTool(registry),
            MatchSkillsTool(registry),
            ReadSkillTool(registry, reader),
            CheckSkillTool(registry, reader),
        ]
        super().__init__(
            exit_stack=exit_stack,
            namespace=namespace,
            tools=tools,
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
    candidates = registry.usable_metadata if usable_only else registry.metadata
    normalized_query = query.casefold().strip()
    items = tuple(
        metadata
        for metadata in candidates
        if _metadata_matches(
            metadata.as_model_dict(),
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


def _metadata_matches(
    metadata: dict[str, SkillModelValue],
    *,
    query: str,
    tags: tuple[str, ...],
    source_label: str | None,
    status: SkillStatus | None,
) -> bool:
    if source_label is not None and metadata["source_label"] != source_label:
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
