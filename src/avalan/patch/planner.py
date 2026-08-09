"""Plan dormant text mutations over immutable abstract snapshots."""

from asyncio import Semaphore, to_thread
from dataclasses import dataclass
from difflib import unified_diff
from enum import Enum
from hashlib import sha256
from typing import Protocol

from avalan.patch.domain import (
    AlgorithmDigest,
    ByteSize,
    Capability,
    FileMode,
    LogicalPath,
    MetadataProfile,
    PatchLineageId,
    PatchValidationError,
    ProposedBytes,
    SourceBytes,
)
from avalan.patch.parser import (
    AddDeclarationSyntax,
    CanonicalPatchRequest,
    DeleteDeclarationSyntax,
    PatchDocumentSyntax,
    PatchHunkSyntax,
    PatchLineSyntax,
    StructuredEditSyntax,
    UpdateDeclarationSyntax,
)

_UTF8_BOM = b"\xef\xbb\xbf"
_LOGICAL_VIEW_FIXED_BYTES = 1_024
_LOGICAL_VIEW_BYTE_MULTIPLIER = 256
_CONTAINER_ENTRY_BYTES = 256
_CONTAINER_FIXED_BYTES = 1_024


class PlannerErrorCode(str, Enum):
    """Name stable pure-planning diagnostics."""

    CONTENT = "patch.unsupported_content"
    REPRESENTATION = "patch.representation_unsupported"
    SOURCE_MISSING = "patch.source_missing"
    DESTINATION_EXISTS = "patch.destination_exists"
    PARENT_MISSING = "patch.parent_missing"
    MATCH_NOT_FOUND = "patch.match_not_found"
    AMBIGUOUS_MATCH = "patch.ambiguous_match"
    OVERLAPPING_EDITS = "patch.overlapping_edits"
    CONFLICT = "patch.conflicting_operations"
    NO_EFFECT = "patch.no_effect"
    LIMIT = "patch.limit_exceeded"
    MOUNT = "patch.mount_denied"


class PlannerError(PatchValidationError):
    """Report one closed planner rejection without retaining content."""

    def __init__(self, code: PlannerErrorCode) -> None:
        """Initialize the stable error code."""
        super().__init__(code.value)
        self.code = code


class TextRepresentation(str, Enum):
    """Name the supported physical text representations."""

    NONE = "none"
    LF = "lf"
    CRLF = "crlf"


class MatchKind(str, Enum):
    """Name the sole matching strategies used by the planner."""

    EXACT_TEXT = "exact_text"
    NEWLINE_COMPATIBLE = "newline_compatible"


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Map one logical character range to exact source-byte offsets."""

    logical_start: int
    logical_end: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class LogicalText:
    """Store a lossless normalized-text view over supported UTF-8 bytes."""

    physical: str
    logical: str
    representation: TextRepresentation
    has_bom: bool
    byte_offsets: tuple[int, ...]

    @classmethod
    def from_bytes(cls, value: bytes) -> "LogicalText":
        """Decode one supported existing or proposed byte value."""
        has_bom = value.startswith(_UTF8_BOM)
        raw = value[len(_UTF8_BOM) :] if has_bom else value
        if raw.startswith(_UTF8_BOM):
            raise PlannerError(PlannerErrorCode.CONTENT)
        try:
            physical = raw.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise PlannerError(PlannerErrorCode.CONTENT) from exc
        return cls._from_text(physical, has_bom)

    @classmethod
    def from_request(cls, value: str) -> "LogicalText":
        """Validate one request or syntax text value without a BOM."""
        if type(value) is not str or value.startswith("\ufeff"):
            raise PlannerError(PlannerErrorCode.CONTENT)
        return cls._from_text(value, False)

    @classmethod
    def _from_text(cls, physical: str, has_bom: bool) -> "LogicalText":
        """Build logical tokens and exact byte-boundary offsets."""
        if any(
            character == "\x00" or 0xD800 <= ord(character) <= 0xDFFF
            for character in physical
        ):
            raise PlannerError(PlannerErrorCode.CONTENT)
        if "\r" in physical.replace("\r\n", ""):
            raise PlannerError(PlannerErrorCode.REPRESENTATION)
        has_lf = "\n" in physical.replace("\r\n", "")
        has_crlf = "\r\n" in physical
        if has_lf and has_crlf:
            raise PlannerError(PlannerErrorCode.REPRESENTATION)
        representation = (
            TextRepresentation.CRLF
            if has_crlf
            else TextRepresentation.LF if has_lf else TextRepresentation.NONE
        )
        logical_parts: list[str] = []
        offsets: list[int] = [len(_UTF8_BOM) if has_bom else 0]
        physical_offset = offsets[0]
        index = 0
        while index < len(physical):
            character = physical[index]
            if character == "\r":
                logical_parts.append("\n")
                physical_offset += 2
                offsets.append(physical_offset)
                index += 2
            else:
                logical_parts.append(character)
                physical_offset += len(character.encode("utf-8"))
                offsets.append(physical_offset)
                index += 1
        return cls(
            physical=physical,
            logical="".join(logical_parts),
            representation=representation,
            has_bom=has_bom,
            byte_offsets=tuple(offsets),
        )

    def span(self, logical_start: int, logical_end: int) -> TextSpan:
        """Return the exact byte boundaries for one logical range."""
        if not 0 <= logical_start <= logical_end <= len(self.logical):
            raise PlannerError(PlannerErrorCode.CONFLICT)
        return TextSpan(
            logical_start,
            logical_end,
            self.byte_offsets[logical_start],
            self.byte_offsets[logical_end],
        )


@dataclass(frozen=True, slots=True)
class Match:
    """Store one resolved text match and its concrete source span."""

    kind: MatchKind
    span: TextSpan


@dataclass(frozen=True, slots=True)
class PlannerLimits:
    """Store independent finite pure-planning resource ceilings."""

    max_file_snapshot_bytes: int = 1_048_576
    max_snapshot_bytes: int = 4_194_304
    max_file_proposed_bytes: int = 1_048_576
    max_proposed_bytes: int = 4_194_304
    max_changed_bytes: int = 2_097_152
    max_match_candidates: int = 16_384
    max_diff_work_bytes: int = 4_194_304
    max_diff_bytes: int = 1_048_576
    max_memory_bytes: int = 8_388_608
    creation_newline: TextRepresentation = TextRepresentation.LF

    def __post_init__(self) -> None:
        """Reject unbounded or unsupported planner limit configurations."""
        values = (
            self.max_file_snapshot_bytes,
            self.max_snapshot_bytes,
            self.max_file_proposed_bytes,
            self.max_proposed_bytes,
            self.max_changed_bytes,
            self.max_match_candidates,
            self.max_diff_work_bytes,
            self.max_diff_bytes,
            self.max_memory_bytes,
        )
        if any(
            type(value) is not int or value < 1 for value in values
        ) or self.creation_newline not in {
            TextRepresentation.LF,
            TextRepresentation.CRLF,
        }:
            raise PlannerError(PlannerErrorCode.LIMIT)


@dataclass(frozen=True, slots=True)
class PlannerFile:
    """Store one target-provided regular-file snapshot.

    Keep target authority outside the value.
    """

    path: LogicalPath
    bytes_value: SourceBytes
    metadata: MetadataProfile
    parent: LogicalPath | None
    mount_id: str
    identity: str
    source_identity: tuple[int, int] | None = None
    parent_identity: tuple[int, int] | None = None
    protected_metadata: AlgorithmDigest | None = None

    def __post_init__(self) -> None:
        """Require stable opaque mount and identity observations."""
        if (
            not self.mount_id
            or not self.identity
            or self.source_identity is not None
            and (
                len(self.source_identity) != 2
                or any(
                    type(item) is not int or item < 0
                    for item in self.source_identity
                )
            )
            or self.parent_identity is not None
            and (
                len(self.parent_identity) != 2
                or any(
                    type(item) is not int or item < 0
                    for item in self.parent_identity
                )
            )
            or self.protected_metadata is not None
            and type(self.protected_metadata) is not AlgorithmDigest
        ):
            raise PlannerError(PlannerErrorCode.CONFLICT)


@dataclass(frozen=True, slots=True)
class PlannerParentMount:
    """Store one authoritative destination-parent mount observation."""

    parent: LogicalPath | None
    mount_id: str
    identity: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Reject an untyped or absent trusted mount observation."""
        if (
            (self.parent is not None and type(self.parent) is not LogicalPath)
            or not self.mount_id
            or (
                self.identity is not None
                and (
                    len(self.identity) != 2
                    or any(
                        type(item) is not int or item < 0
                        for item in self.identity
                    )
                )
            )
        ):
            raise PlannerError(PlannerErrorCode.MOUNT)


@dataclass(frozen=True, slots=True)
class PlannerWorkspace:
    """Store typed abstract target snapshots and existing parent facts."""

    files: tuple[PlannerFile, ...]
    parents: frozenset[LogicalPath]
    parent_mounts: tuple[PlannerParentMount, ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate path and identity facts before virtual planning."""
        if (
            type(self.files) is not tuple
            or type(self.parents) is not frozenset
            or type(self.parent_mounts) is not tuple
            or any(type(item) is not PlannerFile for item in self.files)
            or any(type(item) is not LogicalPath for item in self.parents)
            or any(
                type(item) is not PlannerParentMount
                for item in self.parent_mounts
            )
            or len({item.path for item in self.files}) != len(self.files)
            or len({item.identity for item in self.files}) != len(self.files)
            or len({item.parent for item in self.parent_mounts})
            != len(self.parent_mounts)
            or any(
                item.parent is not None and item.parent not in self.parents
                for item in self.parent_mounts
            )
        ):
            raise PlannerError(PlannerErrorCode.CONFLICT)


@dataclass(frozen=True, slots=True)
class PlannedFile:
    """Store one initial or proposed terminal file fact."""

    path: LogicalPath
    present: bool
    bytes_value: SourceBytes | ProposedBytes | None
    metadata: MetadataProfile | None
    digest: AlgorithmDigest | None
    size: ByteSize
    identity: tuple[int, int] | None = None
    protected_metadata: AlgorithmDigest | None = None

    def __post_init__(self) -> None:
        """Keep absent and present facts structurally consistent."""
        has_content = self.bytes_value is not None
        if self.present != has_content or self.present != (
            self.metadata is not None
        ):
            raise PlannerError(PlannerErrorCode.CONFLICT)
        if (
            self.present != (self.digest is not None)
            or (not self.present and self.size.value != 0)
            or (
                self.identity is not None
                and (
                    len(self.identity) != 2
                    or any(
                        type(item) is not int or item < 0
                        for item in self.identity
                    )
                )
            )
            or (
                self.protected_metadata is not None
                and type(self.protected_metadata) is not AlgorithmDigest
            )
        ):
            raise PlannerError(PlannerErrorCode.CONFLICT)


@dataclass(frozen=True, slots=True)
class PlannedLineage:
    """Store immutable canonical initial-to-final planning inputs."""

    lineage_id: PatchLineageId
    initial: PlannedFile
    final: PlannedFile
    source_path: LogicalPath | None
    destination_path: LogicalPath | None
    capabilities: frozenset[Capability]
    matches: tuple[Match, ...]
    parent_paths: tuple[LogicalPath, ...]
    mount_ids: tuple[str, ...]
    lock_footprint: tuple[LogicalPath, ...]
    atomicity_class: str
    step_graph: tuple[str, ...]
    staging_class: str
    diff_contribution: bytes
    parent_identities: tuple[
        tuple[LogicalPath | None, tuple[int, int]], ...
    ] = ()


@dataclass(frozen=True, slots=True)
class StructuredDiff:
    """Store the complete deterministic review-only diff representation."""

    entries: tuple[bytes, ...]
    rendered: bytes
    digest: AlgorithmDigest


@dataclass(frozen=True, slots=True)
class PlannerCandidate:
    """Store an immutable unsealed candidate.

    Keep approval and commit data outside the candidate.
    """

    request_digest: AlgorithmDigest
    lineages: tuple[PlannedLineage, ...]
    final_files: tuple[PlannedFile, ...]
    diff: StructuredDiff


@dataclass(slots=True)
class _MutableLineage:
    """Track one private virtual lineage.

    Preserve terminal restrictions while the workspace is virtual.
    """

    initial: PlannerFile | None
    current_path: LogicalPath | None
    current_bytes: bytes | None
    metadata: MetadataProfile | None
    moved: bool = False
    updated: bool = False
    deleted: bool = False
    matches: list[Match] | None = None


@dataclass(slots=True)
class _ResourceUsage:
    """Accumulate exact and conservative private planning allocations."""

    changed_bytes: int = 0
    candidate_count: int = 0
    logical_bytes: int = 0
    offset_bytes: int = 0
    candidate_bytes: int = 0
    diff_work_bytes: int = 0
    reserved_memory_bytes: int = 0


def supported_text(value: bytes | str) -> LogicalText:
    """Validate bytes or request text with the sole text predicate."""
    return (
        LogicalText.from_bytes(value)
        if isinstance(value, bytes)
        else LogicalText.from_request(value)
    )


def find_match(
    source: LogicalText,
    old_text: str,
    limit: int,
    usage: _ResourceUsage | None = None,
    limits: PlannerLimits | None = None,
) -> Match:
    """Resolve one exact-first match without fuzzy or broader fallback."""
    old = _logical_from_request(old_text, usage, limits)
    exact = _find_all(source.physical, old.physical, limit, usage, limits)
    if len(exact) > 1:
        raise PlannerError(PlannerErrorCode.AMBIGUOUS_MATCH)
    if exact:
        start = exact[0]
        return Match(
            MatchKind.EXACT_TEXT,
            _physical_span(source, start, start + len(old.physical)),
        )
    candidates = _find_all(source.logical, old.logical, limit, usage, limits)
    if len(candidates) != 1:
        raise PlannerError(
            PlannerErrorCode.MATCH_NOT_FOUND
            if not candidates
            else PlannerErrorCode.AMBIGUOUS_MATCH
        )
    start = candidates[0]
    return Match(
        MatchKind.NEWLINE_COMPATIBLE,
        source.span(start, start + len(old.logical)),
    )


def apply_replacements(
    source: bytes,
    replacements: tuple[tuple[Match, str], ...],
    newline: TextRepresentation,
    limits: PlannerLimits,
    *,
    preserve_final_newline: bool,
) -> bytes:
    """Apply validated replacements simultaneously in source order."""
    value, _ = _apply_replacements(
        source,
        replacements,
        newline,
        limits,
        preserve_final_newline=preserve_final_newline,
        usage=None,
    )
    return value


def _apply_replacements(
    source: bytes,
    replacements: tuple[tuple[Match, str], ...],
    newline: TextRepresentation,
    limits: PlannerLimits,
    *,
    preserve_final_newline: bool,
    usage: _ResourceUsage | None,
) -> tuple[bytes, int]:
    """Apply replacements and return the exact changed-content byte total."""
    view = _logical_from_bytes(source, usage, limits)
    _record_text(usage, view)
    _limit(len(source), limits.max_file_snapshot_bytes)
    _reserve_container(usage, len(replacements), limits)
    ordered = tuple(
        sorted(replacements, key=lambda item: item[0].span.byte_start)
    )
    previous_end = -1
    parts: list[bytes] = []
    offset = 0
    changed = 0
    for match, replacement in ordered:
        span = match.span
        if span.byte_start <= previous_end:
            raise PlannerError(PlannerErrorCode.OVERLAPPING_EDITS)
        replacement_view = _logical_from_request(replacement, usage, limits)
        _record_text(usage, replacement_view)
        physical = _physical(replacement_view.logical, newline)
        replacement_bytes = physical.encode("utf-8")
        removed = source[span.byte_start : span.byte_end]
        if removed != replacement_bytes:
            changed += max(len(removed), len(replacement_bytes))
        parts.extend(
            (
                source[offset : span.byte_start],
                replacement_bytes,
            )
        )
        offset = span.byte_end
        previous_end = span.byte_end
    parts.append(source[offset:])
    _reserve_memory(usage, sum(len(item) for item in parts), limits)
    value = b"".join(parts)
    proposed = _logical_from_bytes(value, usage, limits)
    _record_text(usage, proposed)
    if preserve_final_newline and _final_newline(view) != _final_newline(
        proposed
    ):
        raise PlannerError(PlannerErrorCode.CONFLICT)
    _limit(len(value), limits.max_file_proposed_bytes)
    _limit(changed, limits.max_changed_bytes)
    return value, changed


def plan(
    request: CanonicalPatchRequest,
    workspace: PlannerWorkspace,
    limits: PlannerLimits = PlannerLimits(),
) -> PlannerCandidate:
    """Build one pure unsealed candidate over abstract target snapshots."""
    usage = _ResourceUsage()
    _reserve_memory(
        usage,
        len(request.canonical_bytes)
        + _workspace_fact_bytes(workspace)
        + sum(item.bytes_value.size().value for item in workspace.files),
        limits,
    )
    _validate_workspace(workspace, limits, usage)
    _reserve_container(usage, len(workspace.files) * 2, limits)
    virtual = {item.path: item for item in workspace.files}
    lineages: dict[LogicalPath, _MutableLineage] = {
        item.path: _MutableLineage(
            item, item.path, item.bytes_value._value, item.metadata
        )
        for item in workspace.files
    }
    tombstones: set[LogicalPath] = set()
    produced: set[LogicalPath] = set()
    if isinstance(request.syntax, StructuredEditSyntax):
        _plan_edit(request.syntax, virtual, lineages, limits, usage)
    else:
        _plan_document(
            request.syntax,
            virtual,
            lineages,
            tombstones,
            produced,
            workspace,
            limits,
            usage,
        )
    terminal = _terminal_lineages(lineages, workspace, limits, usage)
    if not terminal:
        raise PlannerError(PlannerErrorCode.NO_EFFECT)
    _reserve_container(usage, len(terminal), limits)
    diff_entries = tuple(_render_lineage(item) for item in terminal)
    _reserve_memory(usage, sum(len(item) for item in diff_entries), limits)
    rendered = b"".join(diff_entries)
    _limit(
        usage.diff_work_bytes + sum(len(item) for item in diff_entries),
        limits.max_diff_work_bytes,
    )
    _limit(len(rendered), limits.max_diff_bytes)
    _limit(
        _memory_cost(request, workspace, terminal, rendered, usage),
        limits.max_memory_bytes,
    )
    return PlannerCandidate(
        request.digest,
        terminal,
        tuple(item.final for item in terminal if item.final.present),
        StructuredDiff(
            diff_entries, rendered, AlgorithmDigest.from_bytes(rendered)
        ),
    )


def render_review_diff(candidate: PlannerCandidate, maximum: int) -> bytes:
    """Render bounded display bytes without altering the complete candidate."""
    if type(maximum) is not int or maximum < 0:
        raise PlannerError(PlannerErrorCode.LIMIT)
    return candidate.diff.rendered[:maximum]


class PlannerWorker(Protocol):
    """Describe the bounded asynchronous worker selected by trusted runtime."""

    async def plan(
        self,
        request: CanonicalPatchRequest,
        workspace: PlannerWorkspace,
        limits: PlannerLimits,
    ) -> PlannerCandidate:
        """Return one pure candidate without target or approval authority."""


class BoundedPlannerWorker:
    """Run deterministic planner work under a finite private bound."""

    def __init__(self, maximum_jobs: int) -> None:
        """Initialize the finite worker admission bound."""
        if type(maximum_jobs) is not int or maximum_jobs < 1:
            raise PlannerError(PlannerErrorCode.LIMIT)
        self._semaphore = Semaphore(maximum_jobs)

    async def plan(
        self,
        request: CanonicalPatchRequest,
        workspace: PlannerWorkspace,
        limits: PlannerLimits,
    ) -> PlannerCandidate:
        """Run the synchronous pure transform outside the caller task."""
        async with self._semaphore:
            return await to_thread(plan, request, workspace, limits)


class PlannerFacade:
    """Expose the sole strictly typed asynchronous planning facade."""

    def __init__(self, worker: PlannerWorker, limits: PlannerLimits) -> None:
        """Bind one trusted worker and immutable planning limits."""
        self._worker = worker
        self._limits = limits

    async def plan(
        self, request: CanonicalPatchRequest, workspace: PlannerWorkspace
    ) -> PlannerCandidate:
        """Delegate bounded planning without approval or commit authority."""
        return await self._worker.plan(request, workspace, self._limits)


def _plan_edit(
    syntax: StructuredEditSyntax,
    virtual: dict[LogicalPath, PlannerFile],
    lineages: dict[LogicalPath, _MutableLineage],
    limits: PlannerLimits,
    usage: _ResourceUsage,
) -> None:
    """Plan simultaneous structured replacements against one snapshot."""
    item = virtual.get(syntax.path)
    if item is None:
        raise PlannerError(PlannerErrorCode.SOURCE_MISSING)
    lineage = lineages[syntax.path]
    assert lineage.current_bytes is not None
    view = _logical_from_bytes(lineage.current_bytes, usage, limits)
    _record_text(usage, view)
    _reserve_container(usage, len(syntax.edits) * 2, limits)
    matches = tuple(
        find_match(
            view, edit.old_text, limits.max_match_candidates, usage, limits
        )
        for edit in syntax.edits
    )
    replacement = tuple(
        (match, edit.new_text)
        for match, edit in zip(matches, syntax.edits, strict=True)
    )
    _record_candidates(usage, len(matches), limits)
    newline = _newline_for(view, limits)
    value, changed = _apply_replacements(
        lineage.current_bytes,
        replacement,
        newline,
        limits,
        preserve_final_newline=True,
        usage=usage,
    )
    if value == lineage.current_bytes:
        raise PlannerError(PlannerErrorCode.NO_EFFECT)
    lineage.current_bytes = value
    lineage.updated = True
    lineage.matches = list(matches)
    _record_changed(usage, changed, limits)


def _plan_document(
    document: PatchDocumentSyntax,
    virtual: dict[LogicalPath, PlannerFile],
    lineages: dict[LogicalPath, _MutableLineage],
    tombstones: set[LogicalPath],
    produced: set[LogicalPath],
    workspace: PlannerWorkspace,
    limits: PlannerLimits,
    usage: _ResourceUsage,
) -> None:
    """Evaluate Version 1 declarations in request order over virtual state."""
    for declaration in document.declarations:
        if isinstance(declaration, AddDeclarationSyntax):
            _add(
                declaration,
                virtual,
                lineages,
                tombstones,
                produced,
                workspace,
                limits,
                usage,
            )
        elif isinstance(declaration, DeleteDeclarationSyntax):
            _delete(declaration, virtual, lineages, tombstones, limits, usage)
        else:
            _update(
                declaration,
                virtual,
                lineages,
                tombstones,
                produced,
                workspace,
                limits,
                usage,
            )


def _add(
    declaration: AddDeclarationSyntax,
    virtual: dict[LogicalPath, PlannerFile],
    lineages: dict[LogicalPath, _MutableLineage],
    tombstones: set[LogicalPath],
    produced: set[LogicalPath],
    workspace: PlannerWorkspace,
    limits: PlannerLimits,
    usage: _ResourceUsage,
) -> None:
    """Add one terminal-only created lineage to virtual state."""
    if declaration.path in virtual:
        raise PlannerError(PlannerErrorCode.DESTINATION_EXISTS)
    if declaration.path in tombstones or declaration.path in produced:
        raise PlannerError(PlannerErrorCode.CONFLICT)
    parent = _parent(declaration.path)
    if parent is not None and parent not in workspace.parents:
        raise PlannerError(PlannerErrorCode.PARENT_MISSING)
    value = _lines_bytes(declaration.lines, limits.creation_newline)
    _logical_from_bytes(value, usage, limits)
    _limit(len(value), limits.max_file_proposed_bytes)
    _record_changed(usage, len(value), limits)
    metadata = MetadataProfile(
        FileMode(0o644), False, limits.creation_newline.value
    )
    item = PlannerFile(
        declaration.path,
        SourceBytes(value),
        metadata,
        parent,
        "created",
        declaration.path.value,
    )
    virtual[declaration.path] = item
    lineages[declaration.path] = _MutableLineage(
        None, declaration.path, value, metadata, updated=True, matches=[]
    )
    produced.add(declaration.path)


def _delete(
    declaration: DeleteDeclarationSyntax,
    virtual: dict[LogicalPath, PlannerFile],
    lineages: dict[LogicalPath, _MutableLineage],
    tombstones: set[LogicalPath],
    limits: PlannerLimits,
    usage: _ResourceUsage,
) -> None:
    """Delete an untouched initial lineage and tombstone its path."""
    lineage = lineages.get(declaration.path)
    if declaration.path not in virtual or lineage is None:
        raise PlannerError(PlannerErrorCode.SOURCE_MISSING)
    if lineage.initial is None or lineage.updated or lineage.moved:
        raise PlannerError(PlannerErrorCode.CONFLICT)
    assert lineage.current_bytes is not None
    _record_changed(usage, len(lineage.current_bytes), limits)
    del virtual[declaration.path]
    lineage.current_path = None
    lineage.current_bytes = None
    lineage.metadata = None
    lineage.deleted = True
    tombstones.add(declaration.path)


def _update(
    declaration: UpdateDeclarationSyntax,
    virtual: dict[LogicalPath, PlannerFile],
    lineages: dict[LogicalPath, _MutableLineage],
    tombstones: set[LogicalPath],
    produced: set[LogicalPath],
    workspace: PlannerWorkspace,
    limits: PlannerLimits,
    usage: _ResourceUsage | None = None,
) -> None:
    """Update one virtual lineage and optionally perform its sole move."""
    lineage = lineages.get(declaration.path)
    item = virtual.get(declaration.path)
    if lineage is None or item is None or lineage.current_bytes is None:
        raise PlannerError(PlannerErrorCode.SOURCE_MISSING)
    value = lineage.current_bytes
    active_usage = usage if usage is not None else _ResourceUsage()
    matches: tuple[Match, ...] = ()
    if declaration.hunks:
        view = _logical_from_bytes(value, active_usage, limits)
        _record_text(active_usage, view)
        _reserve_container(active_usage, len(declaration.hunks) * 2, limits)
        pairs = tuple(
            _hunk_match(view, hunk, limits, active_usage)
            for hunk in declaration.hunks
        )
        newline = _newline_for(view, limits)
        matches = tuple(item[0] for item in pairs)
        _record_candidates(active_usage, len(matches), limits)
        value, changed = _apply_replacements(
            value,
            tuple((match, replacement) for match, replacement in pairs),
            newline,
            limits,
            preserve_final_newline=False,
            usage=active_usage,
        )
        if value == lineage.current_bytes and declaration.move_to is None:
            raise PlannerError(PlannerErrorCode.NO_EFFECT)
        lineage.current_bytes = value
        lineage.updated = lineage.updated or value != item.bytes_value._value
        lineage.matches = list(matches)
        _record_changed(active_usage, changed, limits)
    if declaration.move_to is None:
        return
    destination = declaration.move_to
    if lineage.initial is None or lineage.moved or destination in virtual:
        raise PlannerError(
            PlannerErrorCode.CONFLICT
            if lineage.moved
            else PlannerErrorCode.DESTINATION_EXISTS
        )
    if destination in tombstones or destination in produced:
        raise PlannerError(PlannerErrorCode.CONFLICT)
    parent = _parent(destination)
    if parent is not None and parent not in workspace.parents:
        raise PlannerError(PlannerErrorCode.PARENT_MISSING)
    destination_mount = _mount_for(parent, workspace)
    if destination_mount != item.mount_id:
        raise PlannerError(PlannerErrorCode.MOUNT)
    del virtual[declaration.path]
    virtual[destination] = PlannerFile(
        destination,
        SourceBytes(value),
        item.metadata,
        parent,
        item.mount_id,
        item.identity,
    )
    del lineages[declaration.path]
    lineages[destination] = lineage
    lineage.current_path = destination
    lineage.moved = True
    tombstones.add(declaration.path)
    produced.add(destination)


def _hunk_match(
    source: LogicalText,
    hunk: PatchHunkSyntax,
    limits: PlannerLimits,
    usage: _ResourceUsage,
) -> tuple[Match, str]:
    """Resolve one hunk old side and build a representation-aware new side."""
    newline = _newline_for(source, limits)
    _reserve_memory(
        usage,
        _line_text_reservation(hunk.old_lines)
        + _line_text_reservation(hunk.new_lines),
        limits,
    )
    old = _lines_text(hunk.old_lines)
    new = _lines_text(hunk.new_lines)
    if hunk.eof_anchor and not old:
        if source.logical:
            raise PlannerError(PlannerErrorCode.MATCH_NOT_FOUND)
        return Match(MatchKind.EXACT_TEXT, source.span(0, 0)), new
    match = find_match(
        source,
        _physical(old, newline),
        limits.max_match_candidates,
        usage,
        limits,
    )
    if hunk.eof_anchor and match.span.logical_end != len(source.logical):
        raise PlannerError(PlannerErrorCode.MATCH_NOT_FOUND)
    return match, new


def _terminal_lineages(
    values: dict[LogicalPath, _MutableLineage],
    workspace: PlannerWorkspace,
    limits: PlannerLimits,
    usage: _ResourceUsage | None = None,
) -> tuple[PlannedLineage, ...]:
    """Collapse virtual chains into immutable terminal planning entries."""
    active_usage = usage if usage is not None else _ResourceUsage()
    result: list[PlannedLineage] = []
    for lineage in values.values():
        initial = lineage.initial
        if initial is None and lineage.current_path is None:
            continue
        initial_file = _planned_initial(initial, lineage.current_path)
        final = _planned_final(lineage)
        if (
            initial is not None
            and final.present
            and not lineage.moved
            and initial.bytes_value._value == lineage.current_bytes
        ):
            continue
        capabilities = _capabilities(initial, lineage, final)
        source = initial.path if initial is not None else None
        destination = lineage.current_path
        reservation = _lineage_bytes(
            initial_file, final, source, destination, lineage
        )
        _reserve_memory(active_usage, reservation, limits)
        identity = _lineage_identifier(
            source, destination, initial_file.digest, final.digest
        )
        _limit(
            active_usage.diff_work_bytes
            + _diff_reservation(initial_file, final, source, destination),
            limits.max_diff_work_bytes,
        )
        contribution = _diff_entry(
            initial_file,
            final,
            source,
            destination,
            active_usage,
            limits,
        )
        active_usage.diff_work_bytes += len(contribution)
        _limit(active_usage.diff_work_bytes, limits.max_diff_work_bytes)
        active_usage.candidate_bytes += reservation
        move_with_update = (
            lineage.moved
            and initial is not None
            and initial.bytes_value._value != lineage.current_bytes
        )
        move_only = lineage.moved and not move_with_update
        step_graph = (
            ("destination_publish", "source_remove")
            if move_only or move_with_update
            else ("terminal_effect",)
        )
        result.append(
            PlannedLineage(
                identity,
                initial_file,
                final,
                source,
                destination,
                capabilities,
                tuple(lineage.matches or ()),
                tuple(
                    path
                    for path in (
                        _parent(source) if source else None,
                        _parent(destination) if destination else None,
                    )
                    if path is not None
                ),
                tuple(
                    item
                    for item in ((initial.mount_id if initial else "created"),)
                    if item
                ),
                tuple(
                    sorted(
                        (
                            path
                            for path in (source, destination)
                            if path is not None
                        ),
                        key=lambda path: path.value,
                    )
                ),
                "dependency_ordered" if len(step_graph) > 1 else "single_step",
                step_graph,
                "target_private",
                contribution,
                _lineage_parent_identities(source, destination, workspace),
            )
        )
    _limit(
        sum(item.final.size.value for item in result if item.final.present),
        limits.max_proposed_bytes,
    )
    _limit(
        active_usage.diff_work_bytes,
        limits.max_diff_work_bytes,
    )
    return tuple(sorted(result, key=lambda item: item.lineage_id.value))


def _planned_initial(
    item: PlannerFile | None, path: LogicalPath | None
) -> PlannedFile:
    """Project one initial present or absent file fact."""
    if item is None:
        assert path is not None
        return PlannedFile(path, False, None, None, None, ByteSize(0))
    return PlannedFile(
        item.path,
        True,
        item.bytes_value,
        item.metadata,
        item.bytes_value.digest(),
        item.bytes_value.size(),
        item.source_identity,
        item.protected_metadata,
    )


def _planned_final(lineage: _MutableLineage) -> PlannedFile:
    """Project one final present or absent file fact."""
    if lineage.current_path is None:
        path = lineage.initial.path if lineage.initial is not None else None
        assert path is not None
        return PlannedFile(path, False, None, None, None, ByteSize(0))
    assert lineage.current_bytes is not None and lineage.metadata is not None
    value = ProposedBytes(lineage.current_bytes)
    return PlannedFile(
        lineage.current_path,
        True,
        value,
        lineage.metadata,
        value.digest(),
        value.size(),
        protected_metadata=(
            lineage.initial.protected_metadata
            if lineage.initial is not None
            else None
        ),
    )


def _lineage_parent_identities(
    source: LogicalPath | None,
    destination: LogicalPath | None,
    workspace: PlannerWorkspace,
) -> tuple[tuple[LogicalPath | None, tuple[int, int]], ...]:
    """Retain each source or destination parent identity for final checks."""
    witnesses = {
        item.parent: item.identity for item in workspace.parent_mounts
    }
    parents = tuple(
        sorted(
            {
                _parent(path)
                for path in (source, destination)
                if path is not None
            },
            key=lambda item: "" if item is None else item.value,
        )
    )
    result: list[tuple[LogicalPath | None, tuple[int, int]]] = []
    for parent in parents:
        identity = witnesses.get(parent)
        if identity is not None:
            result.append((parent, identity))
    return tuple(result)


def _capabilities(
    initial: PlannerFile | None, lineage: _MutableLineage, final: PlannedFile
) -> frozenset[Capability]:
    """Derive final-effect capability inputs without policy evaluation."""
    if initial is None:
        return frozenset((Capability.CREATE,))
    if not final.present:
        return frozenset((Capability.DELETE,))
    values = {Capability.MOVE} if lineage.moved else set()
    if lineage.updated:
        values.add(Capability.UPDATE)
    return frozenset(values)


def _render_lineage(lineage: PlannedLineage) -> bytes:
    """Return one deterministic structured diff contribution."""
    return lineage.diff_contribution


def _diff_entry(
    initial: PlannedFile,
    final: PlannedFile,
    source: LogicalPath | None,
    destination: LogicalPath | None,
    usage: _ResourceUsage | None = None,
    limits: PlannerLimits | None = None,
) -> bytes:
    """Render a complete deterministic unified content diff."""
    before = source.value if source is not None else "<absent>"
    after = destination.value if destination is not None else "<absent>"
    lines = [f"--- {before}\n", f"+++ {after}\n"]
    if source is None:
        lines.append("\\ initial: <absent>\n")
    if destination is None:
        lines.append("\\ final: <absent>\n")
    if (
        source is not None
        and destination is not None
        and source != destination
    ):
        lines.extend(
            (
                f"rename from {source.value}\n",
                f"rename to {destination.value}\n",
            )
        )
    if _file_has_bom(initial) != _file_has_bom(final):
        lines.append("\\ UTF-8 BOM changed\n")
    before_lines = _diff_lines(initial, usage, limits)
    after_lines = _diff_lines(final, usage, limits)
    _reserve_container(usage, len(before_lines) + len(after_lines), limits)
    content = tuple(
        unified_diff(before_lines, after_lines, n=3, lineterm="\n")
    )
    if content:
        lines.extend(
            item if item.endswith("\n") else item + "\n"
            for item in content[2:]
        )
    if initial.present and not _file_final_newline(initial):
        lines.append("\\ No newline at end of file (before)\n")
    if final.present and not _file_final_newline(final):
        lines.append("\\ No newline at end of file (after)\n")
    return "".join(lines).encode("utf-8")


def _diff_lines(
    value: PlannedFile,
    usage: _ResourceUsage | None = None,
    limits: PlannerLimits | None = None,
) -> tuple[str, ...]:
    """Return complete logical content lines for one unified diff side."""
    if value.bytes_value is None:
        return ()
    view = _logical_from_bytes(value.bytes_value._value, usage, limits)
    _reserve_container(usage, len(view.logical), limits)
    return tuple(view.logical.splitlines())


def _file_has_bom(value: PlannedFile) -> bool:
    """Return whether one present diff side has its UTF-8 BOM metadata."""
    return (
        value.bytes_value is not None
        and value.bytes_value._value.startswith(_UTF8_BOM)
    )


def _diff_reservation(
    initial: PlannedFile,
    final: PlannedFile,
    source: LogicalPath | None,
    destination: LogicalPath | None,
) -> int:
    """Bound unified-diff work before creating the full review buffer."""
    paths = sum(
        len(path.value.encode("utf-8"))
        for path in (source, destination)
        if path is not None
    )
    return initial.size.value + final.size.value + (paths * 2) + 64


def _lineage_bytes(
    initial: PlannedFile,
    final: PlannedFile,
    source: LogicalPath | None,
    destination: LogicalPath | None,
    lineage: _MutableLineage,
) -> int:
    """Conservatively count immutable lineage strings, paths, and spans."""
    paths = sum(
        len(path.value.encode("utf-8"))
        for path in (source, destination)
        if path is not None
    )
    matches = len(lineage.matches or ())
    return (
        initial.size.value
        + final.size.value
        + paths
        + (matches * 32)
        + len("single_step")
        + len("terminal_effect")
        + len("target_private")
    )


def _file_final_newline(value: PlannedFile) -> bool:
    """Derive final-newline truth from complete protected bytes."""
    if value.bytes_value is None:
        return False
    return value.bytes_value._value.endswith(b"\n")


def _final_newline(value: LogicalText) -> bool:
    """Return whether the physical text ends in its complete newline token."""
    return value.physical.endswith("\n")


def _physical_span(
    source: LogicalText, physical_start: int, physical_end: int
) -> TextSpan:
    """Map a physical-string range back through the lossless logical view."""
    before = source.physical[:physical_start]
    selected = source.physical[:physical_end]
    logical_start = len(before.replace("\r\n", "\n"))
    logical_end = len(selected.replace("\r\n", "\n"))
    bom = len(_UTF8_BOM) if source.has_bom else 0
    return TextSpan(
        logical_start,
        logical_end,
        bom + len(before.encode("utf-8")),
        bom + len(selected.encode("utf-8")),
    )


def _lines_text(lines: tuple[PatchLineSyntax, ...]) -> str:
    """Join logical grammar lines while retaining explicit EOF state."""
    return "".join(
        line.value + ("\n" if line.has_newline else "") for line in lines
    )


def _lines_bytes(
    lines: tuple[PatchLineSyntax, ...], newline: TextRepresentation
) -> bytes:
    """Encode grammar lines with trusted new-file representation defaults."""
    return _physical(_lines_text(lines), newline).encode("utf-8")


def _physical(logical: str, newline: TextRepresentation) -> str:
    """Render logical newline tokens using one established representation."""
    return logical.replace(
        "\n", "\r\n" if newline is TextRepresentation.CRLF else "\n"
    )


def _newline_for(
    value: LogicalText, limits: PlannerLimits
) -> TextRepresentation:
    """Choose existing or trusted creation newline spelling."""
    return (
        limits.creation_newline
        if value.representation is TextRepresentation.NONE
        else value.representation
    )


def _find_all(
    value: str,
    needle: str,
    limit: int,
    usage: _ResourceUsage | None = None,
    limits: PlannerLimits | None = None,
) -> tuple[int, ...]:
    """Find bounded non-overlapping literal candidates without regexes."""
    if not needle:
        raise PlannerError(PlannerErrorCode.MATCH_NOT_FOUND)
    positions: list[int] = []
    start = 0
    while True:
        found = value.find(needle, start)
        if found < 0:
            return tuple(positions)
        _reserve_container(usage, 1, limits)
        positions.append(found)
        if len(positions) > limit:
            raise PlannerError(PlannerErrorCode.LIMIT)
        start = found + 1


def _validate_workspace(
    workspace: PlannerWorkspace,
    limits: PlannerLimits,
    usage: _ResourceUsage | None = None,
) -> None:
    """Validate all abstract source snapshots before any virtual transition."""
    total = 0
    for item in workspace.files:
        value = item.bytes_value._value
        _limit(len(value), limits.max_file_snapshot_bytes)
        _logical_from_bytes(value, usage, limits)
        total += len(value)
    _limit(total, limits.max_snapshot_bytes)


def _limit(value: int, maximum: int) -> None:
    """Reject one independent N-plus-one planning resource use."""
    if value > maximum:
        raise PlannerError(PlannerErrorCode.LIMIT)


def _parent(path: LogicalPath | None) -> LogicalPath | None:
    """Return the non-root logical parent used by target parent facts."""
    if path is None or "/" not in path.value:
        return None
    return LogicalPath(path.value.rsplit("/", 1)[0])


def _mount_for(parent: LogicalPath | None, workspace: PlannerWorkspace) -> str:
    """Resolve a move destination only through an exact parent mount fact."""
    for fact in workspace.parent_mounts:
        if fact.parent == parent:
            return fact.mount_id
    raise PlannerError(PlannerErrorCode.MOUNT)


def _lineage_identifier(
    source: LogicalPath | None,
    destination: LogicalPath | None,
    initial: AlgorithmDigest | None,
    final: AlgorithmDigest | None,
) -> PatchLineageId:
    """Derive a deterministic opaque lineage identifier from terminal facts."""
    material = "\x00".join(
        (
            source.value if source is not None else "",
            destination.value if destination is not None else "",
            initial.value if initial is not None else "",
            final.value if final is not None else "",
        )
    ).encode("ascii")
    return PatchLineageId("lineage_" + sha256(material).hexdigest()[:32])


def _record_text(usage: _ResourceUsage | None, value: LogicalText) -> None:
    """Count one logical view and its offset table when a plan owns it."""
    if usage is None:
        return
    usage.logical_bytes += len(value.physical.encode("utf-8")) + len(
        value.logical.encode("utf-8")
    )
    usage.offset_bytes += len(value.byte_offsets) * _CONTAINER_ENTRY_BYTES


def _reserve_memory(
    usage: _ResourceUsage | None,
    amount: int,
    limits: PlannerLimits | None,
) -> None:
    """Reserve a conservative private allocation before creating it."""
    if usage is None or limits is None:
        return
    if amount < 0:
        raise PlannerError(PlannerErrorCode.LIMIT)
    usage.reserved_memory_bytes += amount
    _limit(usage.reserved_memory_bytes, limits.max_memory_bytes)


def _reserve_container(
    usage: _ResourceUsage | None,
    entries: int,
    limits: PlannerLimits | None,
) -> None:
    """Reserve list, tuple, dictionary, or set entries before construction."""
    if entries < 0:
        raise PlannerError(PlannerErrorCode.LIMIT)
    _reserve_memory(
        usage,
        _CONTAINER_FIXED_BYTES + (entries * _CONTAINER_ENTRY_BYTES),
        limits,
    )


def _logical_from_bytes(
    value: bytes,
    usage: _ResourceUsage | None,
    limits: PlannerLimits | None,
) -> LogicalText:
    """Reserve a UTF-8 view and offset table before decoding bytes."""
    _reserve_memory(
        usage,
        _LOGICAL_VIEW_FIXED_BYTES
        + (len(value) * _LOGICAL_VIEW_BYTE_MULTIPLIER),
        limits,
    )
    return LogicalText.from_bytes(value)


def _logical_from_request(
    value: str,
    usage: _ResourceUsage | None,
    limits: PlannerLimits | None,
) -> LogicalText:
    """Reserve a request-text view before validating or mapping it."""
    _reserve_memory(
        usage,
        _LOGICAL_VIEW_FIXED_BYTES
        + (len(value) * _LOGICAL_VIEW_BYTE_MULTIPLIER * 4),
        limits,
    )
    return LogicalText.from_request(value)


def _line_text_reservation(lines: tuple[PatchLineSyntax, ...]) -> int:
    """Return a conservative pre-allocation bound for joined hunk text."""
    return _CONTAINER_FIXED_BYTES + sum(
        _LOGICAL_VIEW_FIXED_BYTES
        + (len(line.value) * _LOGICAL_VIEW_BYTE_MULTIPLIER * 4)
        for line in lines
    )


def _record_changed(
    usage: _ResourceUsage, changed: int, limits: PlannerLimits
) -> None:
    """Accumulate exact changed content across the complete request."""
    usage.changed_bytes += changed
    _limit(usage.changed_bytes, limits.max_changed_bytes)


def _record_candidates(
    usage: _ResourceUsage, count: int, limits: PlannerLimits
) -> None:
    """Accumulate retained match candidates across the complete request."""
    usage.candidate_count += count
    _limit(usage.candidate_count, limits.max_match_candidates)


def _workspace_fact_bytes(workspace: PlannerWorkspace) -> int:
    """Count immutable workspace path, mount, identity, and parent facts."""
    file_facts = sum(
        len(item.path.value.encode("utf-8"))
        + len(item.mount_id.encode("utf-8"))
        + len(item.identity.encode("utf-8"))
        for item in workspace.files
    )
    parents = sum(
        len(item.value.encode("utf-8")) for item in workspace.parents
    )
    mounts = sum(
        len(item.mount_id.encode("utf-8"))
        + (len(item.parent.value.encode("utf-8")) if item.parent else 0)
        for item in workspace.parent_mounts
    )
    return file_facts + parents + mounts


def _memory_cost(
    request: CanonicalPatchRequest,
    workspace: PlannerWorkspace,
    lineages: tuple[PlannedLineage, ...],
    rendered: bytes,
    usage: _ResourceUsage,
) -> int:
    """Count private planner buffers conservatively and deterministically."""
    snapshots = sum(item.bytes_value.size().value for item in workspace.files)
    proposed = sum(
        item.final.size.value for item in lineages if item.final.present
    )
    current = sum(
        item.final.size.value for item in lineages if item.final.present
    )
    entries = sum(len(item.diff_contribution) for item in lineages)
    return (
        len(request.canonical_bytes)
        + _workspace_fact_bytes(workspace)
        + snapshots
        + proposed
        + current
        + entries
        + len(rendered)
        + usage.logical_bytes
        + usage.offset_bytes
        + usage.candidate_bytes
        + (usage.candidate_count * 32)
        + usage.reserved_memory_bytes
    )
