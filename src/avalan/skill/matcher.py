from .contract import SkillDiagnosticCode, SkillFailureMode, SkillStatus
from .entities import (
    SkillDiagnosticInfo,
    SkillMatchResult,
    SkillMetadata,
    SkillModelValue,
    SkillRegistryVersion,
    diagnostic_from_failure,
    model_dict,
)
from .envelope import SkillResponseEnvelope
from .registry import SkillRegistry, SkillRegistrySkill
from .resolver import SkillAsyncFileSystem, SkillSourceFileSystem
from .settings import SkillIndexLimits

from asyncio import sleep
from dataclasses import dataclass, field, replace
from re import findall, fullmatch

_RANK_EXACT = 0
_RANK_NAMED = 1
_RANK_STRONG_METADATA = 2
_RANK_POSSIBLE = 3
_RANK_AVAILABLE = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMatchLimits:
    max_results: int = 16
    max_query_tokens: int = 16
    max_query_characters: int = 4096
    max_index_tokens_per_skill: int = 256
    max_excerpt_bytes_per_skill: int = 2048

    def __post_init__(self) -> None:
        _assert_positive_int(self.max_results, "max_results")
        _assert_positive_int(self.max_query_tokens, "max_query_tokens")
        _assert_positive_int(
            self.max_query_characters,
            "max_query_characters",
        )
        _assert_positive_int(
            self.max_index_tokens_per_skill,
            "max_index_tokens_per_skill",
        )
        _assert_positive_int(
            self.max_excerpt_bytes_per_skill,
            "max_excerpt_bytes_per_skill",
        )

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "max_results": self.max_results,
                "max_query_tokens": self.max_query_tokens,
                "max_query_characters": self.max_query_characters,
                "max_index_tokens_per_skill": self.max_index_tokens_per_skill,
                "max_excerpt_bytes_per_skill": (
                    self.max_excerpt_bytes_per_skill
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMatchFilters:
    query: str = ""
    tags: tuple[str, ...] = ()
    source_label: str | None = None
    status: SkillStatus | None = None
    usable_only: bool = True

    def __post_init__(self) -> None:
        _assert_query(self.query)
        _assert_logical_tuple(self.tags, "tags")
        if self.source_label is not None:
            _assert_logical_id(self.source_label, "source_label")
        if self.status is not None:
            assert isinstance(self.status, SkillStatus)
        assert isinstance(self.usable_only, bool)

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        value: dict[str, object] = {"usable_only": self.usable_only}
        if self.query:
            value["query_tokens"] = _query_tokens(
                self.query,
                SkillMatchLimits(),
            )
        if self.tags:
            value["tags"] = self.tags
        if self.source_label is not None:
            value["source_label"] = self.source_label
        if self.status is not None:
            value["status"] = self.status.value
        return model_dict(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMatchIndexEntry:
    metadata: SkillMetadata
    status: SkillStatus
    usable: bool
    id_tokens: tuple[str, ...]
    name_tokens: tuple[str, ...]
    tag_tokens: tuple[str, ...]
    source_tokens: tuple[str, ...]
    description_tokens: tuple[str, ...]
    excerpt_tokens: tuple[str, ...] = ()
    indexed_excerpt_bytes: int = 0

    def __post_init__(self) -> None:
        assert isinstance(self.metadata, SkillMetadata)
        assert isinstance(self.status, SkillStatus)
        assert isinstance(self.usable, bool)
        _assert_token_tuple(self.id_tokens, "id_tokens")
        _assert_token_tuple(self.name_tokens, "name_tokens")
        _assert_token_tuple(self.tag_tokens, "tag_tokens")
        _assert_token_tuple(self.source_tokens, "source_tokens")
        _assert_token_tuple(
            self.description_tokens,
            "description_tokens",
        )
        _assert_token_tuple(self.excerpt_tokens, "excerpt_tokens")
        _assert_non_negative_int(
            self.indexed_excerpt_bytes,
            "indexed_excerpt_bytes",
        )
        if self.usable:
            assert self.status is SkillStatus.OK

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "metadata": self.metadata.as_model_dict(),
                "status": self.status.value,
                "usable": self.usable,
                "indexed_excerpt_bytes": self.indexed_excerpt_bytes,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMatchIndex:
    registry_version: SkillRegistryVersion
    entries: tuple[SkillMatchIndexEntry, ...] = ()
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()
    indexed_bytes: int = 0
    limits: SkillMatchLimits = field(default_factory=SkillMatchLimits)
    _trusted_registry_version: SkillRegistryVersion | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        assert isinstance(self.registry_version, SkillRegistryVersion)
        _assert_entry_tuple(self.entries)
        _assert_diagnostic_tuple(self.diagnostics)
        _assert_non_negative_int(self.indexed_bytes, "indexed_bytes")
        assert isinstance(self.limits, SkillMatchLimits)
        assert self.indexed_bytes == sum(
            entry.indexed_excerpt_bytes for entry in self.entries
        ), "indexed_bytes must equal entry excerpt bytes"

    @property
    def status(self) -> SkillStatus:
        if self.diagnostics:
            return self.diagnostics[0].status
        if self.entries:
            return SkillStatus.OK
        return SkillStatus.EMPTY

    def as_model_dict(self) -> dict[str, SkillModelValue]:
        return model_dict(
            {
                "status": self.status.value,
                "registry_version": self.registry_version.as_model_value(),
                "entry_count": len(self.entries),
                "indexed_bytes": self.indexed_bytes,
                "limits": self.limits.as_model_dict(),
                "entries": tuple(
                    entry.as_model_dict() for entry in self.entries
                ),
                "diagnostics": tuple(
                    diagnostic.as_model_dict()
                    for diagnostic in self.diagnostics
                ),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ScoredEntry:
    entry: SkillMatchIndexEntry
    score: float
    rank: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.entry, SkillMatchIndexEntry)
        assert isinstance(self.score, float)
        assert 0 <= self.score <= 1
        assert isinstance(self.rank, int)
        _assert_reason_tuple(self.reasons)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreparedQuery:
    tokens: tuple[str, ...]
    normalized: str
    diagnostics: tuple[SkillDiagnosticInfo, ...] = ()

    def __post_init__(self) -> None:
        _assert_token_tuple(self.tokens, "tokens")
        assert isinstance(self.normalized, str)
        if self.normalized:
            _assert_token_tuple((self.normalized,), "normalized")
        _assert_diagnostic_tuple(self.diagnostics)


async def build_skill_match_index(
    registry: SkillRegistry,
    *,
    include_resource_excerpts: bool = False,
    file_system: SkillSourceFileSystem | None = None,
    index_limits: SkillIndexLimits | None = None,
    match_limits: SkillMatchLimits | None = None,
) -> SkillMatchIndex:
    assert isinstance(registry, SkillRegistry)
    assert isinstance(include_resource_excerpts, bool)
    if index_limits is None:
        index_limits = registry.index_limits
    if match_limits is None:
        match_limits = SkillMatchLimits()
    if file_system is None:
        file_system = SkillAsyncFileSystem()
    assert isinstance(index_limits, SkillIndexLimits)
    assert isinstance(match_limits, SkillMatchLimits)

    if len(registry.skills) > index_limits.max_skills:
        return _trusted_match_index(
            SkillMatchIndex(
                registry_version=registry.registry_version,
                diagnostics=(
                    _index_oversized_diagnostic(
                        reason="max_skills_exceeded",
                    ),
                ),
                limits=match_limits,
            )
        )

    diagnostics: list[SkillDiagnosticInfo] = list(registry.diagnostics)
    entries: list[SkillMatchIndexEntry] = []
    indexed_bytes = 0
    for skill in registry.skills:
        if len(skill.resources) > index_limits.max_resources_per_skill:
            return _trusted_match_index(
                SkillMatchIndex(
                    registry_version=registry.registry_version,
                    diagnostics=(
                        _index_oversized_diagnostic(
                            reason="max_resources_per_skill_exceeded",
                            candidates=_skill_candidates((skill,)),
                        ),
                    ),
                    limits=match_limits,
                )
            )
        if skill.metadata is not None:
            excerpt_tokens: tuple[str, ...] = ()
            excerpt_bytes = 0
            if include_resource_excerpts and skill.usable:
                (
                    excerpt_tokens,
                    excerpt_bytes,
                    excerpt_diagnostics,
                ) = await _indexed_excerpt_tokens(
                    skill,
                    file_system=file_system,
                    index_limits=index_limits,
                    match_limits=match_limits,
                    indexed_bytes=indexed_bytes,
                )
                diagnostics.extend(excerpt_diagnostics)
                indexed_bytes += excerpt_bytes
            entries.append(
                _index_entry(
                    skill.metadata,
                    status=skill.status,
                    usable=skill.usable,
                    excerpt_tokens=excerpt_tokens,
                    indexed_excerpt_bytes=excerpt_bytes,
                    match_limits=match_limits,
                )
            )
        await sleep(0)

    if not entries and not diagnostics:
        diagnostics.append(
            diagnostic_from_failure(
                SkillFailureMode.EMPTY_REGISTRY,
                path="skills",
            )
        )
    return _trusted_match_index(
        SkillMatchIndex(
            registry_version=registry.registry_version,
            entries=tuple(entries),
            diagnostics=tuple(diagnostics),
            indexed_bytes=indexed_bytes,
            limits=match_limits,
        )
    )


async def match_skill_registry(
    registry: SkillRegistry,
    *,
    query: str = "",
    tags: tuple[str, ...] = (),
    source_label: str | None = None,
    status: SkillStatus | None = None,
    usable_only: bool = True,
    max_results: int | None = None,
    index: SkillMatchIndex | None = None,
    include_resource_excerpts: bool = False,
    file_system: SkillSourceFileSystem | None = None,
    match_limits: SkillMatchLimits | None = None,
    include_source_labels: bool = True,
) -> SkillResponseEnvelope:
    assert isinstance(registry, SkillRegistry)
    assert isinstance(include_source_labels, bool)
    if index is not None:
        assert isinstance(index, SkillMatchIndex)
        assert index.registry_version == registry.registry_version
    if match_limits is None:
        match_limits = (
            index.limits
            if index is not None and _index_is_trusted(index, registry)
            else SkillMatchLimits()
        )
    if max_results is not None:
        _assert_positive_int(max_results, "max_results")
        match_limits = replace(match_limits, max_results=max_results)
    filters = SkillMatchFilters(
        query=query,
        tags=tags,
        source_label=source_label,
        status=status,
        usable_only=usable_only,
    )
    if filters.source_label is not None and not include_source_labels:
        diagnostic = _hidden_source_label_diagnostic(
            "skills.match.source_label"
        )
        return SkillResponseEnvelope(
            status=diagnostic.status,
            registry_version=registry.registry_version,
            diagnostics=(diagnostic,),
        )
    prepared_query = _prepare_query(filters.query, match_limits)
    if index is None:
        index = await build_skill_match_index(
            registry,
            include_resource_excerpts=include_resource_excerpts,
            file_system=file_system,
            match_limits=match_limits,
        )
    index = _validated_match_index(
        registry,
        index,
        match_limits=match_limits,
    )

    if not index.entries:
        diagnostics = _with_query_diagnostics(
            index.diagnostics
            or (
                diagnostic_from_failure(
                    SkillFailureMode.EMPTY_REGISTRY,
                    path="skills",
                ),
            ),
            prepared_query,
        )
        return SkillResponseEnvelope(
            status=diagnostics[0].status,
            registry_version=registry.registry_version,
            diagnostics=diagnostics,
        )

    scored = _scored_entries(
        index.entries,
        filters,
        prepared_query,
        include_source_labels=include_source_labels,
    )
    excluded: tuple[_ScoredEntry, ...] = ()
    if filters.usable_only:
        excluded = tuple(
            score
            for score in scored
            if not score.entry.usable
            or score.entry.status is not SkillStatus.OK
        )
        disabled = tuple(
            score
            for score in excluded
            if score.entry.status is SkillStatus.DISABLED
        )
        scored = tuple(
            score
            for score in scored
            if score.entry.usable and score.entry.status is SkillStatus.OK
        )
        if not scored and disabled:
            diagnostics = _with_query_diagnostics(
                (
                    diagnostic_from_failure(
                        SkillFailureMode.DISABLED_SKILL,
                        path="skills.match",
                        candidates=_scored_candidates(disabled),
                        details={"usable_only": True},
                    ),
                ),
                prepared_query,
            )
            return SkillResponseEnvelope(
                status=SkillStatus.DISABLED,
                registry_version=registry.registry_version,
                diagnostics=diagnostics,
            )

    if not scored:
        unusable = _excluded_match_diagnostics(registry, excluded)
        unavailable = _unavailable_diagnostics(index, filters)
        diagnostics = _with_query_diagnostics(
            unusable
            or unavailable
            or _supplied_index_diagnostics(index)
            or (
                diagnostic_from_failure(
                    SkillFailureMode.NO_MATCH,
                    path="skills.match",
                ),
            ),
            prepared_query,
        )
        return SkillResponseEnvelope(
            status=diagnostics[0].status,
            registry_version=registry.registry_version,
            diagnostics=diagnostics,
        )

    limited = scored[: match_limits.max_results]
    items = tuple(_match_result(score) for score in limited)
    ambiguous = _ambiguous_named_diagnostic(scored)
    if ambiguous is not None:
        diagnostics = _with_query_diagnostics(
            (ambiguous,),
            prepared_query,
        )
        return SkillResponseEnvelope(
            status=SkillStatus.AMBIGUOUS,
            registry_version=registry.registry_version,
            items=items,
            diagnostics=diagnostics,
        )
    return SkillResponseEnvelope(
        status=SkillStatus.OK,
        registry_version=registry.registry_version,
        items=items,
        diagnostics=_with_query_diagnostics(
            index.diagnostics,
            prepared_query,
        ),
    )


async def _indexed_excerpt_tokens(
    skill: SkillRegistrySkill,
    *,
    file_system: SkillSourceFileSystem,
    index_limits: SkillIndexLimits,
    match_limits: SkillMatchLimits,
    indexed_bytes: int,
) -> tuple[tuple[str, ...], int, tuple[SkillDiagnosticInfo, ...]]:
    tokens: list[str] = []
    excerpt_bytes = 0
    diagnostics: list[SkillDiagnosticInfo] = []
    remaining_global = index_limits.max_indexed_bytes - indexed_bytes
    if remaining_global <= 0:
        return (
            (),
            0,
            (
                _index_oversized_diagnostic(
                    reason="max_indexed_bytes_exceeded",
                    candidates=_skill_candidates((skill,)),
                ),
            ),
        )
    remaining_skill = match_limits.max_excerpt_bytes_per_skill
    for resource in skill.resources:
        if remaining_skill <= 0 or remaining_global <= 0:
            break
        limit = min(remaining_skill, remaining_global)
        try:
            content = await file_system.read_bytes(resource.path, limit)
        except OSError:
            diagnostics.append(
                diagnostic_from_failure(
                    SkillFailureMode.SOURCE_UNAVAILABLE,
                    path="source.availability",
                    details={
                        "resource_id": resource.handle.resource_id,
                        "source_label": resource.handle.source_label,
                    },
                )
            )
            continue
        excerpt_bytes += len(content)
        remaining_skill -= len(content)
        remaining_global -= len(content)
        tokens.extend(_tokens(content.decode("utf-8", "replace")))
        if len(tokens) >= match_limits.max_index_tokens_per_skill:
            tokens = tokens[: match_limits.max_index_tokens_per_skill]
            break
    return tuple(dict.fromkeys(tokens)), excerpt_bytes, tuple(diagnostics)


def _index_entry(
    metadata: SkillMetadata,
    *,
    status: SkillStatus,
    usable: bool,
    excerpt_tokens: tuple[str, ...],
    indexed_excerpt_bytes: int,
    match_limits: SkillMatchLimits,
) -> SkillMatchIndexEntry:
    remaining = match_limits.max_index_tokens_per_skill
    id_tokens, remaining = _limited_tokens(metadata.skill_id, remaining)
    name_tokens, remaining = _limited_tokens(metadata.name, remaining)
    tag_tokens, remaining = _limited_tokens(" ".join(metadata.tags), remaining)
    source_tokens, remaining = _limited_tokens(
        metadata.source_label,
        remaining,
    )
    description_tokens, remaining = _limited_tokens(
        metadata.description,
        remaining,
    )
    excerpt_tokens = tuple(excerpt_tokens[:remaining])
    return SkillMatchIndexEntry(
        metadata=metadata,
        status=status,
        usable=usable,
        id_tokens=id_tokens,
        name_tokens=name_tokens,
        tag_tokens=tag_tokens,
        source_tokens=source_tokens,
        description_tokens=description_tokens,
        excerpt_tokens=excerpt_tokens,
        indexed_excerpt_bytes=indexed_excerpt_bytes,
    )


def _validated_match_index(
    registry: SkillRegistry,
    index: SkillMatchIndex,
    *,
    match_limits: SkillMatchLimits,
) -> SkillMatchIndex:
    valid_entries: list[SkillMatchIndexEntry] = []
    invalid_candidates: list[str] = []
    duplicate_candidates: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    trust_index = _index_is_trusted(index, registry)
    validation_limits = index.limits if trust_index else match_limits
    if (
        not trust_index
        and len(index.entries) > registry.index_limits.max_skills
    ):
        valid_entries.extend(
            _missing_registry_entries(
                registry,
                seen_keys=seen_keys,
                match_limits=match_limits,
            )
        )
        return SkillMatchIndex(
            registry_version=index.registry_version,
            entries=tuple(valid_entries),
            diagnostics=(
                _index_oversized_diagnostic(
                    reason="max_supplied_index_entries_exceeded",
                ),
            ),
            indexed_bytes=0,
            limits=match_limits,
        )

    for entry in index.entries:
        skill = _registry_skill_for_entry(registry, entry)
        if skill is None:
            invalid_candidates.append(entry.metadata.skill_id)
            continue
        canonical = _validated_registry_entry(
            entry,
            skill,
            match_limits=validation_limits,
            trust_excerpt_tokens=trust_index,
        )
        if canonical is None:
            invalid_candidates.append(entry.metadata.skill_id)
            continue
        key = (
            canonical.metadata.source_label,
            canonical.metadata.skill_id,
        )
        if key in seen_keys:
            duplicate_candidates.append(canonical.metadata.skill_id)
            continue
        seen_keys.add(key)
        valid_entries.append(canonical)

    if not trust_index:
        for entry in _missing_registry_entries(
            registry,
            seen_keys=seen_keys,
            match_limits=match_limits,
        ):
            seen_keys.add(_entry_key(entry))
            valid_entries.append(entry)

    diagnostics = index.diagnostics if trust_index else ()
    if invalid_candidates:
        diagnostics = (
            *diagnostics,
            _invalid_supplied_index_diagnostic(
                candidates=tuple(sorted(set(invalid_candidates))),
            ),
        )
    if duplicate_candidates:
        diagnostics = (
            *diagnostics,
            _duplicate_supplied_index_diagnostic(
                candidates=tuple(sorted(set(duplicate_candidates))),
            ),
        )
    return SkillMatchIndex(
        registry_version=index.registry_version,
        entries=tuple(valid_entries),
        diagnostics=diagnostics,
        indexed_bytes=sum(
            entry.indexed_excerpt_bytes for entry in valid_entries
        ),
        limits=index.limits if trust_index else match_limits,
    )


def _validated_registry_entry(
    entry: SkillMatchIndexEntry,
    skill: SkillRegistrySkill,
    *,
    match_limits: SkillMatchLimits,
    trust_excerpt_tokens: bool,
) -> SkillMatchIndexEntry | None:
    assert skill.metadata is not None
    assert entry.metadata == skill.metadata
    if entry.status is not skill.status or entry.usable != skill.usable:
        return None
    canonical = _index_entry(
        skill.metadata,
        status=skill.status,
        usable=skill.usable,
        excerpt_tokens=entry.excerpt_tokens if trust_excerpt_tokens else (),
        indexed_excerpt_bytes=(
            entry.indexed_excerpt_bytes if trust_excerpt_tokens else 0
        ),
        match_limits=match_limits,
    )
    if (
        entry.id_tokens != canonical.id_tokens
        or entry.name_tokens != canonical.name_tokens
        or entry.tag_tokens != canonical.tag_tokens
        or entry.source_tokens != canonical.source_tokens
        or entry.description_tokens != canonical.description_tokens
    ):
        return None
    return canonical


def _trusted_match_index(index: SkillMatchIndex) -> SkillMatchIndex:
    object.__setattr__(
        index,
        "_trusted_registry_version",
        index.registry_version,
    )
    return index


def _index_is_trusted(
    index: SkillMatchIndex,
    registry: SkillRegistry,
) -> bool:
    return index._trusted_registry_version == registry.registry_version


def _missing_registry_entries(
    registry: SkillRegistry,
    *,
    seen_keys: set[tuple[str, str]],
    match_limits: SkillMatchLimits,
) -> tuple[SkillMatchIndexEntry, ...]:
    entries: list[SkillMatchIndexEntry] = []
    for skill in registry.skills:
        if skill.metadata is None:
            continue
        if _skill_key(skill) in seen_keys:
            continue
        entries.append(
            _index_entry(
                skill.metadata,
                status=skill.status,
                usable=skill.usable,
                excerpt_tokens=(),
                indexed_excerpt_bytes=0,
                match_limits=match_limits,
            )
        )
    return tuple(entries)


def _entry_key(entry: SkillMatchIndexEntry) -> tuple[str, str]:
    return (entry.metadata.source_label, entry.metadata.skill_id)


def _skill_key(skill: SkillRegistrySkill) -> tuple[str, str]:
    assert skill.metadata is not None
    return (skill.metadata.source_label, skill.metadata.skill_id)


def _registry_skill_for_entry(
    registry: SkillRegistry,
    entry: SkillMatchIndexEntry,
) -> SkillRegistrySkill | None:
    matches = tuple(
        skill
        for skill in registry.skills
        if skill.metadata is not None
        and skill.source_label == entry.metadata.source_label
        and skill.skill_id == entry.metadata.skill_id
        and skill.metadata == entry.metadata
    )
    if not matches:
        return None
    return matches[0]


def _excluded_match_diagnostics(
    registry: SkillRegistry,
    excluded: tuple[_ScoredEntry, ...],
) -> tuple[SkillDiagnosticInfo, ...]:
    for score in excluded:
        skill = _registry_skill_for_entry(registry, score.entry)
        if skill is not None and skill.diagnostics:
            return skill.diagnostics
    return ()


def _prepare_query(
    query: str,
    limits: SkillMatchLimits,
) -> _PreparedQuery:
    text, truncated = _bounded_query_text(query, limits)
    diagnostics = (_query_truncated_diagnostic(limits),) if truncated else ()
    return _PreparedQuery(
        tokens=_query_tokens(query, limits),
        normalized=_normalized_query_from_text(text),
        diagnostics=diagnostics,
    )


def _bounded_query_text(
    query: str,
    limits: SkillMatchLimits,
) -> tuple[str, bool]:
    assert isinstance(query, str)
    assert isinstance(limits, SkillMatchLimits)
    truncated = len(query) > limits.max_query_characters
    return query[: limits.max_query_characters], truncated


def _with_query_diagnostics(
    diagnostics: tuple[SkillDiagnosticInfo, ...],
    prepared_query: _PreparedQuery,
) -> tuple[SkillDiagnosticInfo, ...]:
    return (*diagnostics, *prepared_query.diagnostics)


def _scored_entries(
    entries: tuple[SkillMatchIndexEntry, ...],
    filters: SkillMatchFilters,
    prepared_query: _PreparedQuery,
    *,
    include_source_labels: bool,
) -> tuple[_ScoredEntry, ...]:
    scored: list[_ScoredEntry] = []
    for entry in entries:
        if not _passes_filters(entry, filters):
            continue
        score = _score_entry(
            entry,
            filters,
            prepared_query,
            include_source_labels=include_source_labels,
        )
        if score is not None:
            scored.append(score)
    return tuple(
        sorted(
            scored,
            key=lambda value: (
                value.rank,
                -value.score,
                value.entry.metadata.source_label,
                value.entry.metadata.skill_id,
            ),
        )
    )


def _passes_filters(
    entry: SkillMatchIndexEntry,
    filters: SkillMatchFilters,
) -> bool:
    if filters.source_label is not None:
        if entry.metadata.source_label != filters.source_label:
            return False
    if filters.status is not None and entry.status is not filters.status:
        return False
    if filters.tags and not set(filters.tags).issubset(entry.metadata.tags):
        return False
    return True


def _score_entry(
    entry: SkillMatchIndexEntry,
    filters: SkillMatchFilters,
    prepared_query: _PreparedQuery,
    *,
    include_source_labels: bool,
) -> _ScoredEntry | None:
    query_tokens = prepared_query.tokens
    if not query_tokens:
        filter_reasons = _filter_reasons(
            filters,
            include_source_labels=include_source_labels,
        )
        reasons = filter_reasons or ("available skill matched",)
        return _ScoredEntry(
            entry=entry,
            score=0.5 if filter_reasons else 0.2,
            rank=(
                _RANK_STRONG_METADATA if filter_reasons else _RANK_AVAILABLE
            ),
            reasons=reasons,
        )

    normalized_query = prepared_query.normalized
    if normalized_query in {entry.metadata.skill_id, entry.metadata.name}:
        return _ScoredEntry(
            entry=entry,
            score=1.0,
            rank=_RANK_EXACT,
            reasons=(
                "exact skill_id matched query",
                "exact name matched query",
            ),
        )

    id_name_tokens = set(entry.id_tokens) | set(entry.name_tokens)
    if set(query_tokens).issubset(id_name_tokens):
        return _ScoredEntry(
            entry=entry,
            score=0.9,
            rank=_RANK_NAMED,
            reasons=("name tokens matched query",),
        )

    strong_reasons: list[str] = []
    if set(query_tokens).issubset(entry.tag_tokens):
        strong_reasons.append("tag metadata matched query")
    if include_source_labels and set(query_tokens).issubset(
        entry.source_tokens
    ):
        strong_reasons.append("source label matched query")
    if set(query_tokens).issubset(entry.description_tokens):
        strong_reasons.append("description matched query")
    if strong_reasons:
        return _ScoredEntry(
            entry=entry,
            score=0.78,
            rank=_RANK_STRONG_METADATA,
            reasons=tuple(strong_reasons),
        )

    possible_reasons: list[str] = []
    metadata_tokens = set(entry.tag_tokens) | set(entry.description_tokens)
    if include_source_labels:
        metadata_tokens |= set(entry.source_tokens)
    if metadata_tokens.intersection(query_tokens):
        possible_reasons.append("metadata partially matched query")
    if set(query_tokens).issubset(entry.excerpt_tokens):
        possible_reasons.append("bounded indexed excerpt matched query")
    elif set(entry.excerpt_tokens).intersection(query_tokens):
        possible_reasons.append(
            "bounded indexed excerpt partially matched query"
        )
    if possible_reasons:
        return _ScoredEntry(
            entry=entry,
            score=0.45,
            rank=_RANK_POSSIBLE,
            reasons=tuple(possible_reasons),
        )
    return None


def _filter_reasons(
    filters: SkillMatchFilters,
    *,
    include_source_labels: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if filters.tags:
        reasons.append("tag filter matched")
    if filters.source_label is not None and include_source_labels:
        reasons.append("source filter matched")
    if filters.status is not None:
        reasons.append("status filter matched")
    return tuple(reasons)


def _hidden_source_label_diagnostic(path: str) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.POLICY_DENIED,
        status=SkillStatus.POLICY_DENIED,
        message="The requested source label filter is not exposed.",
        path=path,
        hint="Use skills.list or skills.match without a source label filter.",
        details=model_dict({"reason": "source_labels_hidden"}),
    )


def _match_result(score: _ScoredEntry) -> SkillMatchResult:
    return SkillMatchResult(
        metadata=score.entry.metadata,
        score=score.score,
        reasons=score.reasons,
    )


def _ambiguous_named_diagnostic(
    scored: tuple[_ScoredEntry, ...],
) -> SkillDiagnosticInfo | None:
    if not scored or scored[0].rank <= _RANK_EXACT:
        return None
    named = tuple(
        score
        for score in scored
        if score.rank == _RANK_NAMED and score.score == scored[0].score
    )
    if len(named) < 2:
        return None
    return diagnostic_from_failure(
        SkillFailureMode.AMBIGUOUS_SKILL_NAME,
        path="skills.match",
        candidates=_scored_candidates(named),
    )


def _unavailable_diagnostics(
    index: SkillMatchIndex,
    filters: SkillMatchFilters,
) -> tuple[SkillDiagnosticInfo, ...]:
    diagnostics = tuple(
        diagnostic
        for diagnostic in index.diagnostics
        if diagnostic.status is SkillStatus.UNAVAILABLE
    )
    if filters.source_label is None:
        return diagnostics
    return tuple(
        diagnostic
        for diagnostic in diagnostics
        if filters.source_label in diagnostic.candidates
        or diagnostic.details.get("source_label") == filters.source_label
    )


def _supplied_index_diagnostics(
    index: SkillMatchIndex,
) -> tuple[SkillDiagnosticInfo, ...]:
    return tuple(
        diagnostic
        for diagnostic in index.diagnostics
        if diagnostic.path == "registry.index"
        and diagnostic.details.get("reason")
        in {
            "duplicate_index_entry",
            "index_entry_not_in_registry",
            "max_supplied_index_entries_exceeded",
        }
    )


def _scored_candidates(values: tuple[_ScoredEntry, ...]) -> tuple[str, ...]:
    return tuple(score.entry.metadata.skill_id for score in values)


def _skill_candidates(
    values: tuple[SkillRegistrySkill, ...],
) -> tuple[str, ...]:
    return tuple(
        skill.skill_id for skill in values if skill.skill_id is not None
    )


def _limited_tokens(text: str, remaining: int) -> tuple[tuple[str, ...], int]:
    if remaining <= 0:
        return (), 0
    tokens = _tokens(text)[:remaining]
    return tuple(dict.fromkeys(tokens)), remaining - len(tokens)


def _query_tokens(
    query: str,
    limits: SkillMatchLimits,
) -> tuple[str, ...]:
    text, _ = _bounded_query_text(query, limits)
    return tuple(dict.fromkeys(_tokens(text)[: limits.max_query_tokens]))


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = findall(r"[a-z0-9]+", lowered)
    if _is_logical_id(lowered):
        tokens.insert(0, lowered)
    return tokens


def _normalized_query_from_text(query: str) -> str:
    lowered = query.lower().strip()
    if _is_logical_id(lowered):
        return lowered
    return "-".join(findall(r"[a-z0-9]+", lowered))


def _invalid_supplied_index_diagnostic(
    *,
    candidates: tuple[str, ...],
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.NOT_FOUND,
        status=SkillStatus.NOT_FOUND,
        message="The supplied skill match index is not in the registry.",
        path="registry.index",
        hint="Rebuild the match index from the current registry.",
        candidates=candidates,
        details={"reason": "index_entry_not_in_registry"},
    )


def _duplicate_supplied_index_diagnostic(
    *,
    candidates: tuple[str, ...],
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.DUPLICATE_ID,
        status=SkillStatus.BLOCKED,
        message="The supplied skill match index contains duplicate entries.",
        path="registry.index",
        hint="Rebuild the match index from the current registry.",
        candidates=candidates,
        details={"reason": "duplicate_index_entry"},
    )


def _query_truncated_diagnostic(
    limits: SkillMatchLimits,
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_OVERSIZED,
        status=SkillStatus.TRUNCATED,
        message="The skill match query exceeds configured limits.",
        path="skills.match",
        hint="Shorten the query or use metadata filters.",
        details={
            "reason": "max_query_characters_exceeded",
            "max_query_characters": limits.max_query_characters,
        },
    )


def _index_oversized_diagnostic(
    *,
    reason: str,
    candidates: tuple[str, ...] = (),
) -> SkillDiagnosticInfo:
    return SkillDiagnosticInfo(
        code=SkillDiagnosticCode.RESOURCE_OVERSIZED,
        status=SkillStatus.TRUNCATED,
        message="The skill match index input exceeds configured limits.",
        path="registry.index",
        hint="Reduce configured skills or use tighter trusted filters.",
        candidates=candidates,
        details={"reason": reason},
    )


def _assert_positive_int(value: int, field_name: str) -> None:
    assert isinstance(value, int) and not isinstance(
        value,
        bool,
    ), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_non_negative_int(value: int, field_name: str) -> None:
    assert isinstance(value, int) and not isinstance(
        value,
        bool,
    ), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must be non-negative"


def _assert_query(value: str) -> None:
    assert isinstance(value, str), "query must be a string"
    assert "\x00" not in value, "query must not contain NUL bytes"


def _assert_logical_id(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert _is_logical_id(value), f"{field_name} must be a logical ID"


def _assert_logical_tuple(
    values: tuple[str, ...],
    field_name: str,
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_logical_id(value, field_name)
    assert len(values) == len(set(values)), f"{field_name} must be unique"


def _assert_token_tuple(
    values: tuple[str, ...],
    field_name: str,
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        assert isinstance(value, str), f"{field_name} values must be strings"
        assert fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is not None


def _assert_reason_tuple(values: tuple[str, ...]) -> None:
    assert isinstance(values, tuple), "reasons must be a tuple"
    assert values, "reasons must be non-empty"
    for value in values:
        assert isinstance(value, str), "reasons must be strings"
        assert value.strip(), "reasons must be non-empty"


def _assert_entry_tuple(values: tuple[SkillMatchIndexEntry, ...]) -> None:
    assert isinstance(values, tuple), "entries must be a tuple"
    for value in values:
        assert isinstance(value, SkillMatchIndexEntry)


def _assert_diagnostic_tuple(
    values: tuple[SkillDiagnosticInfo, ...],
) -> None:
    assert isinstance(values, tuple), "diagnostics must be a tuple"
    for value in values:
        assert isinstance(value, SkillDiagnosticInfo)


def _is_logical_id(value: str) -> bool:
    return fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", value) is not None
