"""Parse closed, dormant Version 1 patch inputs without target effects."""

from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from enum import Enum
from re import fullmatch
from typing import NoReturn
from unicodedata import category

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    TextGenerationNonStreamToolCall,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    LogicalPath,
    OperationType,
    PatchObserverCorrelationId,
    PatchValidationError,
)

_BIDI_CONTROLS = frozenset(
    (
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    )
)
_RESERVED_DEVICES = frozenset(
    (
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
    )
)
_BEGIN = "*** Begin Patch v1"
_END = "*** End Patch"
_ADD = "*** Add File: "
_UPDATE = "*** Update File: "
_MOVE = "*** Move to: "
_DELETE = "*** Delete File: "
_EOF = "*** End of File"
_NO_NEWLINE = "\\ No newline at end of file"
_VERSION_DOMAIN = b"avalan.patch.request/v1\x00"


class RawPatchInputKind(str, Enum):
    """Name a provider projection accepted by the dormant codec."""

    EDIT_JSON = "edit_json"
    APPLY_JSON = "apply_json"
    VERIFIED_FREEFORM = "verified_freeform"


class RawPatchInputState(str, Enum):
    """Name one bounded raw ingress state."""

    COMPLETE = "complete"
    MALFORMED = "malformed"
    OVERSIZED = "oversized"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class PatchInputErrorCode(str, Enum):
    """Name stable parse-boundary diagnostics without content disclosure."""

    INCOMPLETE = "patch.input_incomplete"
    CANCELLED = "patch.input_cancelled"
    OVERSIZED = "patch.input_oversized"
    MALFORMED = "patch.input_malformed"
    SCHEMA = "patch.input_schema"
    PATH = "patch.input_path"
    GRAMMAR = "patch.input_grammar"


class PatchInputError(PatchValidationError):
    """Report one stable ingress failure without retaining raw patch text."""

    def __init__(self, code: PatchInputErrorCode) -> None:
        """Initialize one bounded error code and random correlation."""
        super().__init__(code.value)
        self.code = code
        self.correlation_id = PatchObserverCorrelationId.new()


@dataclass(frozen=True, slots=True)
class RawProviderProfile:
    """Identify one provider profile that can preserve raw patch payloads."""

    value: str

    def __post_init__(self) -> None:
        """Reject ambiguous provider-profile spellings."""
        if fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.value) is None:
            raise PatchInputError(PatchInputErrorCode.MALFORMED)


@dataclass(frozen=True, slots=True)
class RawToolCallId:
    """Bind raw input to one opaque provider tool-call identity."""

    value: str

    def __post_init__(self) -> None:
        """Require one bounded opaque call identity."""
        if fullmatch(r"[A-Za-z0-9_-]{1,128}", self.value) is None:
            raise PatchInputError(PatchInputErrorCode.MALFORMED)


@dataclass(frozen=True, slots=True)
class PatchInputLimits:
    """Store finite lexical and raw-input ceilings for one codec instance."""

    max_raw_bytes: int = 1_048_576
    max_scalars: int = 1_048_576
    max_records: int = 65_536
    max_paths: int = 4_096
    max_declarations: int = 4_096
    max_hunks: int = 16_384
    max_edits: int = 16_384
    max_json_depth: int = 256
    max_path_characters: int = 1_024
    max_path_bytes: int = 4_096
    max_path_components: int = 64
    max_component_characters: int = 255
    max_component_bytes: int = 1_024
    max_content_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        """Reject zero, negative, or non-integer input limits."""
        if any(
            type(value) is not int or value < 1
            for value in (
                self.max_raw_bytes,
                self.max_scalars,
                self.max_records,
                self.max_paths,
                self.max_declarations,
                self.max_hunks,
                self.max_edits,
                self.max_json_depth,
                self.max_path_characters,
                self.max_path_bytes,
                self.max_path_components,
                self.max_component_characters,
                self.max_component_bytes,
                self.max_content_bytes,
            )
        ):
            raise PatchInputError(PatchInputErrorCode.MALFORMED)


@dataclass(frozen=True, slots=True, repr=False)
class RawPatchIngress:
    """Store one complete raw provider input before generic mapping decode."""

    provider_profile: RawProviderProfile
    tool_call_id: RawToolCallId
    kind: RawPatchInputKind
    state: RawPatchInputState
    raw_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """Reject incomplete values from the parser entry boundary."""
        if type(self.raw_bytes) is not bytes:
            raise PatchInputError(PatchInputErrorCode.MALFORMED)
        if self.state is not RawPatchInputState.COMPLETE:
            raise PatchInputError(_state_error(self.state))


class PatchInputAccumulator:
    """Accumulate bounded provider chunks until one closed ingress exists."""

    def __init__(self, limits: PatchInputLimits) -> None:
        """Initialize an empty bounded raw-input buffer."""
        self._limits = limits
        self._buffer = b""
        self._state = RawPatchInputState.INCOMPLETE

    def append(self, chunk: bytes) -> None:
        """Append one provider byte chunk before completion."""
        if (
            self._state is not RawPatchInputState.INCOMPLETE
            or type(chunk) is not bytes
        ):
            self._clear(RawPatchInputState.MALFORMED)
        candidate = self._buffer + chunk
        if len(candidate) > self._limits.max_raw_bytes:
            self._clear(RawPatchInputState.OVERSIZED)
        self._buffer = candidate

    def cancel(self) -> None:
        """Cancel the stream and erase all accumulated provider bytes."""
        self._buffer = b""
        self._state = RawPatchInputState.CANCELLED

    def finish(
        self,
        provider_profile: RawProviderProfile,
        tool_call_id: RawToolCallId,
        kind: RawPatchInputKind,
    ) -> RawPatchIngress:
        """Return the sole complete raw ingress and clear the buffer."""
        if self._state is not RawPatchInputState.INCOMPLETE:
            self._clear(self._state)
        payload = self._buffer
        self._buffer = b""
        self._state = RawPatchInputState.COMPLETE
        return RawPatchIngress(
            provider_profile=provider_profile,
            tool_call_id=tool_call_id,
            kind=kind,
            state=RawPatchInputState.COMPLETE,
            raw_bytes=payload,
        )

    def _clear(self, state: RawPatchInputState) -> NoReturn:
        """Erase buffered bytes before exposing a stable failure."""
        self._buffer = b""
        self._state = state
        raise PatchInputError(_state_error(state))


class RawProviderIngressAdapter:
    """Project real provider tool-call lifecycles into closed raw ingress."""

    def __init__(self, limits: PatchInputLimits) -> None:
        """Initialize one dormant raw provider adapter with finite bounds."""
        self._limits = limits

    def non_streaming(
        self,
        provider_call: object,
        provider_profile: RawProviderProfile,
        kind: RawPatchInputKind,
    ) -> RawPatchIngress:
        """Reject decoded calls and accept only real raw arguments."""
        if not isinstance(provider_call, TextGenerationNonStreamToolCall):
            raise PatchInputError(PatchInputErrorCode.MALFORMED)
        if type(provider_call.arguments) is not str:
            raise PatchInputError(PatchInputErrorCode.MALFORMED)
        return RawPatchIngress(
            provider_profile=provider_profile,
            tool_call_id=RawToolCallId(provider_call.call_id),
            kind=kind,
            state=RawPatchInputState.COMPLETE,
            raw_bytes=_encode_raw_text(provider_call.arguments),
        )

    async def streaming(
        self,
        items: AsyncIterable[object],
        provider_profile: RawProviderProfile,
        tool_call_id: RawToolCallId,
        kind: RawPatchInputKind,
    ) -> RawPatchIngress:
        """Buffer a raw provider call before mapping or dispatch."""
        accumulator = PatchInputAccumulator(self._limits)
        ready = False
        async for item in items:
            if not isinstance(item, CanonicalStreamItem):
                accumulator._clear(RawPatchInputState.MALFORMED)
            if item.kind is StreamItemKind.STREAM_CANCELLED:
                accumulator.cancel()
                accumulator.finish(provider_profile, tool_call_id, kind)
            if item.kind is StreamItemKind.STREAM_ERRORED:
                accumulator._clear(RawPatchInputState.MALFORMED)
            if item.correlation.tool_call_id != tool_call_id.value:
                continue
            if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
                if type(item.text_delta) is not str:
                    accumulator._clear(RawPatchInputState.MALFORMED)
                accumulator.append(_encode_raw_text(item.text_delta))
            elif item.kind is StreamItemKind.TOOL_CALL_READY:
                if ready:
                    accumulator._clear(RawPatchInputState.MALFORMED)
                ready = True
            elif item.kind is StreamItemKind.TOOL_CALL_DONE:
                if not ready:
                    accumulator._clear(RawPatchInputState.MALFORMED)
                return accumulator.finish(provider_profile, tool_call_id, kind)
        accumulator._clear(RawPatchInputState.INCOMPLETE)


@dataclass(frozen=True, slots=True)
class TextEditSyntax:
    """Store one closed structured replacement without execution authority."""

    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class PatchLineSyntax:
    """Store one immutable logical patch line and newline contribution."""

    value: str
    has_newline: bool


@dataclass(frozen=True, slots=True)
class PatchHunkSyntax:
    """Store one immutable hunk with explicit old and new logical sides."""

    label: str | None
    old_lines: tuple[PatchLineSyntax, ...]
    new_lines: tuple[PatchLineSyntax, ...]
    eof_anchor: bool


@dataclass(frozen=True, slots=True)
class AddDeclarationSyntax:
    """Store one immutable add declaration."""

    path: LogicalPath
    lines: tuple[PatchLineSyntax, ...]


@dataclass(frozen=True, slots=True)
class UpdateDeclarationSyntax:
    """Store one immutable update declaration and optional destination."""

    path: LogicalPath
    move_to: LogicalPath | None
    hunks: tuple[PatchHunkSyntax, ...]


@dataclass(frozen=True, slots=True)
class DeleteDeclarationSyntax:
    """Store one immutable delete declaration."""

    path: LogicalPath


PatchDeclarationSyntax = (
    AddDeclarationSyntax | UpdateDeclarationSyntax | DeleteDeclarationSyntax
)


@dataclass(frozen=True, slots=True)
class PatchDocumentSyntax:
    """Store immutable parsed declarations and canonical grammar bytes."""

    declarations: tuple[PatchDeclarationSyntax, ...]
    canonical_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class StructuredEditSyntax:
    """Store a validated closed structured edit request."""

    path: LogicalPath
    edits: tuple[TextEditSyntax, ...]


@dataclass(frozen=True, slots=True)
class CanonicalPatchRequest:
    """Store an immutable parsed request with exact digest input bytes."""

    operation: OperationType
    syntax: StructuredEditSyntax | PatchDocumentSyntax
    canonical_bytes: bytes = field(repr=False)
    digest: AlgorithmDigest

    def __post_init__(self) -> None:
        """Bind the digest to exact canonical bytes at construction."""
        if self.digest != AlgorithmDigest.from_bytes(self.canonical_bytes):
            raise PatchInputError(PatchInputErrorCode.MALFORMED)


@dataclass(frozen=True, slots=True)
class DormantParameterDescriptor:
    """Store one unregistered portable parameter-schema fixture."""

    operation: OperationType
    schema_bytes: bytes


DORMANT_PARAMETER_DESCRIPTORS = (
    DormantParameterDescriptor(
        operation=OperationType.EDIT,
        schema_bytes=(
            b'{"additionalProperties":false,"properties":{"edits":{"items":{"additionalProperties":false,"properties":{"new_text":{"type":"string"},"old_text":{"minLength":1,"type":"string"}},"required":["old_text","new_text"],"type":"object"},"minItems":1,"type":"array"},"path":{"type":"string"}},"required":["path","edits"],"type":"object"}'
        ),
    ),
    DormantParameterDescriptor(
        operation=OperationType.APPLY,
        schema_bytes=b'{"additionalProperties":false,"properties":{"patch":{"type":"string"}},"required":["patch"],"type":"object"}',
    ),
)


@dataclass(frozen=True, slots=True)
class _JsonString:
    """Store a parsed JSON string before closed schema projection."""

    value: str


@dataclass(frozen=True, slots=True)
class _JsonArray:
    """Store an ordered parsed JSON array before schema projection."""

    values: tuple["_JsonValue", ...]


@dataclass(frozen=True, slots=True)
class _JsonObject:
    """Store duplicate-preserving ordered JSON object members."""

    members: tuple[tuple[str, "_JsonValue"], ...]


@dataclass(frozen=True, slots=True)
class _JsonOther:
    """Store a syntactically valid but schema-incompatible JSON primitive."""

    spelling: str


_JsonValue = _JsonString | _JsonArray | _JsonObject | _JsonOther


class PatchRequestParser:
    """Parse closed provider ingress into effect-free canonical syntax."""

    def __init__(self, limits: PatchInputLimits = PatchInputLimits()) -> None:
        """Initialize one parser with immutable lexical ceilings."""
        self._limits = limits

    def parse(self, ingress: RawPatchIngress) -> CanonicalPatchRequest:
        """Parse complete raw input before ordinary mapping construction."""
        if len(ingress.raw_bytes) > self._limits.max_raw_bytes:
            raise PatchInputError(PatchInputErrorCode.OVERSIZED)
        if ingress.kind is RawPatchInputKind.VERIFIED_FREEFORM:
            document = self._parse_freeform(ingress.raw_bytes)
            return self._canonical(OperationType.APPLY, document)
        root = self._json(ingress.raw_bytes)
        if ingress.kind is RawPatchInputKind.EDIT_JSON:
            syntax = self._edit(root)
            return self._canonical(OperationType.EDIT, syntax)
        if ingress.kind is RawPatchInputKind.APPLY_JSON:
            document = self._apply(root)
            return self._canonical(OperationType.APPLY, document)
        raise PatchInputError(PatchInputErrorCode.MALFORMED)

    def _json(self, payload: bytes) -> _JsonValue:
        """Decode strict UTF-8 and retain duplicate JSON object members."""
        text = _decode_utf8(payload)
        _validate_scalar_text(text, self._limits.max_scalars)
        return _JsonReader(text, self._limits.max_json_depth).parse()

    def _edit(self, value: _JsonValue) -> StructuredEditSyntax:
        """Project one exact structured edit schema without map collapse."""
        root = _members(value, ("path", "edits"))
        path = self._path(_string_member(root, "path"))
        edits = _array_member(root, "edits")
        if not edits.values or len(edits.values) > self._limits.max_edits:
            raise PatchInputError(PatchInputErrorCode.SCHEMA)
        projected: list[TextEditSyntax] = []
        for item in edits.values:
            fields = _members(item, ("old_text", "new_text"))
            old_text = _string_member(fields, "old_text")
            new_text = _string_member(fields, "new_text")
            if not old_text:
                raise PatchInputError(PatchInputErrorCode.SCHEMA)
            _validate_content(old_text, self._limits)
            _validate_content(new_text, self._limits)
            projected.append(
                TextEditSyntax(old_text=old_text, new_text=new_text)
            )
        return StructuredEditSyntax(path=path, edits=tuple(projected))

    def _apply(self, value: _JsonValue) -> PatchDocumentSyntax:
        """Project one exact structured apply schema into grammar syntax."""
        root = _members(value, ("patch",))
        patch = _string_member(root, "patch")
        _validate_scalar_text(patch, self._limits.max_scalars)
        return self._document(patch)

    def _parse_freeform(self, payload: bytes) -> PatchDocumentSyntax:
        """Decode only a verified freeform apply string under strict UTF-8."""
        return self._document(_decode_utf8(payload))

    def _path(self, value: str) -> LogicalPath:
        """Validate one portable lexical path without normalizing spelling."""
        _validate_path(value, self._limits)
        return LogicalPath(value)

    def _document(self, text: str) -> PatchDocumentSyntax:
        """Parse the complete Version 1 declaration state machine."""
        _validate_scalar_text(text, self._limits.max_scalars)
        records = _records(text, self._limits.max_records)
        if len(records) < 3 or records[0] != _BEGIN or records[-1] != _END:
            raise PatchInputError(PatchInputErrorCode.GRAMMAR)
        declarations: list[PatchDeclarationSyntax] = []
        hunk_count = 0
        index = 1
        while index < len(records) - 1:
            record = records[index]
            declaration: PatchDeclarationSyntax
            if record.startswith(_ADD):
                declaration, index = self._add(records, index)
            elif record.startswith(_UPDATE):
                declaration, index = self._update(
                    records, index, self._limits.max_hunks - hunk_count
                )
                hunk_count += len(declaration.hunks)
            elif record.startswith(_DELETE):
                declaration, index = self._delete(records, index)
            else:
                raise PatchInputError(PatchInputErrorCode.GRAMMAR)
            declarations.append(declaration)
            if len(declarations) > self._limits.max_declarations:
                raise PatchInputError(PatchInputErrorCode.OVERSIZED)
        path_count = sum(
            (
                2
                if isinstance(item, UpdateDeclarationSyntax) and item.move_to
                else 1
            )
            for item in declarations
        )
        if path_count > self._limits.max_paths:
            raise PatchInputError(PatchInputErrorCode.OVERSIZED)
        canonical = "\n".join(records).encode("utf-8") + b"\n"
        return PatchDocumentSyntax(tuple(declarations), canonical)

    def _add(
        self, records: tuple[str, ...], index: int
    ) -> tuple[AddDeclarationSyntax, int]:
        """Parse one add declaration and its consecutive content records."""
        path = self._path(records[index][len(_ADD) :])
        index += 1
        lines: list[PatchLineSyntax] = []
        marker = False
        while index < len(records) - 1 and not _is_header(records[index]):
            record = records[index]
            if record == _NO_NEWLINE:
                if marker or not lines:
                    raise PatchInputError(PatchInputErrorCode.GRAMMAR)
                marker = True
                lines[-1] = PatchLineSyntax(lines[-1].value, False)
            elif record.startswith("+") and not marker:
                value = record[1:]
                _validate_content(value, self._limits)
                lines.append(PatchLineSyntax(value, True))
            else:
                raise PatchInputError(PatchInputErrorCode.GRAMMAR)
            index += 1
        return AddDeclarationSyntax(path, tuple(lines)), index

    def _delete(
        self, records: tuple[str, ...], index: int
    ) -> tuple[DeleteDeclarationSyntax, int]:
        """Parse one body-free delete declaration."""
        return (
            DeleteDeclarationSyntax(
                self._path(records[index][len(_DELETE) :])
            ),
            index + 1,
        )

    def _update(
        self, records: tuple[str, ...], index: int, hunk_budget: int
    ) -> tuple[UpdateDeclarationSyntax, int]:
        """Parse one update declaration with immediate move and hunks."""
        path = self._path(records[index][len(_UPDATE) :])
        index += 1
        move_to: LogicalPath | None = None
        if index < len(records) - 1 and records[index].startswith(_MOVE):
            move_to = self._path(records[index][len(_MOVE) :])
            index += 1
        hunks: list[PatchHunkSyntax] = []
        while index < len(records) - 1 and records[index].startswith("@@"):
            if len(hunks) >= hunk_budget:
                raise PatchInputError(PatchInputErrorCode.OVERSIZED)
            hunk, index = self._hunk(records, index)
            hunks.append(hunk)
        if not hunks and move_to is None:
            raise PatchInputError(PatchInputErrorCode.GRAMMAR)
        return UpdateDeclarationSyntax(path, move_to, tuple(hunks)), index

    def _hunk(
        self, records: tuple[str, ...], index: int
    ) -> tuple[PatchHunkSyntax, int]:
        """Parse one hunk and enforce its per-side EOF state machine."""
        label = _label(records[index])
        index += 1
        old_lines: list[PatchLineSyntax] = []
        new_lines: list[PatchLineSyntax] = []
        old_closed = False
        new_closed = False
        changed = False
        eof_anchor = False
        previous = ""
        while index < len(records) - 1 and not _is_boundary(records[index]):
            record = records[index]
            if record == _EOF:
                eof_anchor = True
                index += 1
                if index < len(records) - 1 and not _is_boundary(
                    records[index]
                ):
                    raise PatchInputError(PatchInputErrorCode.GRAMMAR)
                break
            if record == _NO_NEWLINE:
                if previous == "-" and not old_closed:
                    old_lines[-1] = PatchLineSyntax(old_lines[-1].value, False)
                    old_closed = True
                elif previous == "+" and not new_closed:
                    new_lines[-1] = PatchLineSyntax(new_lines[-1].value, False)
                    new_closed = True
                elif previous == " " and not old_closed and not new_closed:
                    old_lines[-1] = PatchLineSyntax(old_lines[-1].value, False)
                    new_lines[-1] = PatchLineSyntax(new_lines[-1].value, False)
                    old_closed = True
                    new_closed = True
                else:
                    raise PatchInputError(PatchInputErrorCode.GRAMMAR)
                previous = "marker"
            elif record.startswith(" "):
                if old_closed or new_closed:
                    raise PatchInputError(PatchInputErrorCode.GRAMMAR)
                value = record[1:]
                _validate_content(value, self._limits)
                line = PatchLineSyntax(value, True)
                old_lines.append(line)
                new_lines.append(line)
                previous = " "
            elif record.startswith("-"):
                if old_closed:
                    raise PatchInputError(PatchInputErrorCode.GRAMMAR)
                value = record[1:]
                _validate_content(value, self._limits)
                old_lines.append(PatchLineSyntax(value, True))
                changed = True
                previous = "-"
            elif record.startswith("+"):
                if new_closed:
                    raise PatchInputError(PatchInputErrorCode.GRAMMAR)
                value = record[1:]
                _validate_content(value, self._limits)
                new_lines.append(PatchLineSyntax(value, True))
                changed = True
                previous = "+"
            else:
                raise PatchInputError(PatchInputErrorCode.GRAMMAR)
            index += 1
        if not changed or (not old_lines and not eof_anchor):
            raise PatchInputError(PatchInputErrorCode.GRAMMAR)
        if (old_closed or new_closed) and not eof_anchor:
            raise PatchInputError(PatchInputErrorCode.GRAMMAR)
        return (
            PatchHunkSyntax(
                label, tuple(old_lines), tuple(new_lines), eof_anchor
            ),
            index,
        )

    def _canonical(
        self,
        operation: OperationType,
        syntax: StructuredEditSyntax | PatchDocumentSyntax,
    ) -> CanonicalPatchRequest:
        """Domain-separate exact canonical request bytes and their digest."""
        if operation is OperationType.EDIT:
            assert isinstance(syntax, StructuredEditSyntax)
            body = _canonical_edit(syntax)
        else:
            assert isinstance(syntax, PatchDocumentSyntax)
            body = syntax.canonical_bytes
        payload = (
            _VERSION_DOMAIN
            + operation.value.encode("ascii")
            + b"\x00grammar-v1\x00"
            + body
        )
        return CanonicalPatchRequest(
            operation, syntax, payload, AlgorithmDigest.from_bytes(payload)
        )


def _state_error(state: RawPatchInputState) -> PatchInputErrorCode:
    """Map one terminal raw state to its public stable diagnostic code."""
    return {
        RawPatchInputState.INCOMPLETE: PatchInputErrorCode.INCOMPLETE,
        RawPatchInputState.CANCELLED: PatchInputErrorCode.CANCELLED,
        RawPatchInputState.OVERSIZED: PatchInputErrorCode.OVERSIZED,
        RawPatchInputState.MALFORMED: PatchInputErrorCode.MALFORMED,
        RawPatchInputState.COMPLETE: PatchInputErrorCode.MALFORMED,
    }[state]


def _decode_utf8(payload: bytes) -> str:
    """Decode strict BOM-free UTF-8 without preserving raw error details."""
    if payload.startswith(b"\xef\xbb\xbf"):
        raise PatchInputError(PatchInputErrorCode.MALFORMED)
    try:
        value = payload.decode("utf-8", "strict")
    except UnicodeDecodeError:
        value = None
    if value is None:
        raise PatchInputError(PatchInputErrorCode.MALFORMED)
    return value


def _encode_raw_text(value: str) -> bytes:
    """Encode raw provider text without chaining its content-bearing error."""
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        encoded = None
    if encoded is None:
        raise PatchInputError(PatchInputErrorCode.MALFORMED)
    return encoded


def _validate_scalar_text(value: str, maximum: int) -> None:
    """Reject NUL, surrogates, and oversized Unicode scalar values."""
    if len(value) > maximum or any(
        character == "\x00" or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise PatchInputError(PatchInputErrorCode.MALFORMED)


def _validate_content(value: str, limits: PatchInputLimits) -> None:
    """Validate supported text spelling and its independent byte ceiling."""
    _validate_scalar_text(value, limits.max_scalars)
    if len(value.encode("utf-8")) > limits.max_content_bytes:
        raise PatchInputError(PatchInputErrorCode.OVERSIZED)
    if "\r" in value and (
        "\r" in value.replace("\r\n", "") or "\n" in value.replace("\r\n", "")
    ):
        raise PatchInputError(PatchInputErrorCode.MALFORMED)


def _validate_path(value: str, limits: PatchInputLimits) -> None:
    """Reject unsafe portable path spellings before target inspection."""
    _validate_scalar_text(value, limits.max_scalars)
    encoded = value.encode("utf-8")
    parts = value.split("/")
    if (
        not value
        or len(value) > limits.max_path_characters
        or len(encoded) > limits.max_path_bytes
        or len(parts) > limits.max_path_components
        or value.startswith(("/", "~", "$", "%", "\\"))
        or "\\" in value
        or fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*:.*", value) is not None
        or any(
            not part
            or part in {".", ".."}
            or part != part.strip()
            or ":" in part
            or len(part) > limits.max_component_characters
            or len(part.encode("utf-8")) > limits.max_component_bytes
            or part.split(".", 1)[0].casefold() in _RESERVED_DEVICES
            or any(
                category(character).startswith("C")
                or character in _BIDI_CONTROLS
                for character in part
            )
            for part in parts
        )
    ):
        raise PatchInputError(PatchInputErrorCode.PATH)


def _records(text: str, maximum: int) -> tuple[str, ...]:
    """Tokenize only uniformly separated nonblank grammar records."""
    if "\r" in text:
        remainder = text.replace("\r\n", "")
        if "\r\n" not in text or "\r" in remainder or "\n" in remainder:
            raise PatchInputError(PatchInputErrorCode.GRAMMAR)
        separator = "\r\n"
    else:
        separator = "\n"
    records = text.split(separator)
    if records[-1] == "":
        records.pop()
    if (
        not records
        or any(not record for record in records)
        or len(records) > maximum
    ):
        raise PatchInputError(PatchInputErrorCode.GRAMMAR)
    return tuple(records)


def _is_header(record: str) -> bool:
    """Return whether a record starts a declaration or terminal marker."""
    return record == _END or record.startswith((_ADD, _UPDATE, _DELETE))


def _is_boundary(record: str) -> bool:
    """Return whether a hunk must end before the next grammar construct."""
    return _is_header(record) or record.startswith("@@")


def _label(record: str) -> str | None:
    """Validate one non-authoritative presentation-safe optional label."""
    if record == "@@":
        return None
    if not record.startswith("@@ "):
        raise PatchInputError(PatchInputErrorCode.GRAMMAR)
    label = record[3:]
    if (
        not label
        or label != label.strip()
        or any(
            category(character).startswith("C") or character in _BIDI_CONTROLS
            for character in label
        )
    ):
        raise PatchInputError(PatchInputErrorCode.GRAMMAR)
    return label


def _members(
    value: _JsonValue, expected: tuple[str, ...]
) -> tuple[tuple[str, _JsonValue], ...]:
    """Require one exact duplicate-free closed JSON object member set."""
    if not isinstance(value, _JsonObject):
        raise PatchInputError(PatchInputErrorCode.SCHEMA)
    names = tuple(name for name, _ in value.members)
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise PatchInputError(PatchInputErrorCode.SCHEMA)
    return value.members


def _member(
    members: tuple[tuple[str, _JsonValue], ...], name: str
) -> _JsonValue:
    """Return one unique member whose exact closed schema was checked."""
    for candidate, value in members:
        if candidate == name:
            return value
    raise PatchInputError(PatchInputErrorCode.SCHEMA)


def _string_member(
    members: tuple[tuple[str, _JsonValue], ...], name: str
) -> str:
    """Return one schema-required JSON string member."""
    value = _member(members, name)
    if not isinstance(value, _JsonString):
        raise PatchInputError(PatchInputErrorCode.SCHEMA)
    return value.value


def _array_member(
    members: tuple[tuple[str, _JsonValue], ...], name: str
) -> _JsonArray:
    """Return one schema-required JSON array member."""
    value = _member(members, name)
    if not isinstance(value, _JsonArray):
        raise PatchInputError(PatchInputErrorCode.SCHEMA)
    return value


def _canonical_edit(value: StructuredEditSyntax) -> bytes:
    """Encode a closed structured edit request with stable JSON bytes."""
    pieces = [b'{"path":', _json_string(value.path.value), b',"edits":[']
    for index, edit in enumerate(value.edits):
        if index:
            pieces.append(b",")
        pieces.extend(
            (
                b'{"old_text":',
                _json_string(edit.old_text),
                b',"new_text":',
                _json_string(edit.new_text),
                b"}",
            )
        )
    pieces.extend((b"]}",))
    return b"".join(pieces)


def _json_string(value: str) -> bytes:
    """Encode one scalar string using deterministic JSON escape spelling."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return ('"' + escaped + '"').encode("utf-8")


class _JsonReader:
    """Read the duplicate-preserving JSON language needed by Version 1."""

    def __init__(self, text: str, max_depth: int = 256) -> None:
        """Initialize a zero-copy cursor over strict decoded JSON text."""
        if type(max_depth) is not int or max_depth < 1:
            raise PatchInputError(PatchInputErrorCode.MALFORMED)
        self._text = text
        self._max_depth = max_depth
        self._index = 0

    def parse(self) -> _JsonValue:
        """Parse exactly one non-whitespace JSON value."""
        self._space()
        value = self._value(0)
        self._space()
        if self._index != len(self._text):
            self._error()
        return value

    def _value(self, depth: int) -> _JsonValue:
        """Read only object, array, or string forms used by closed schemas."""
        character = self._peek()
        if character == "{":
            self._depth(depth)
            return self._object(depth + 1)
        if character == "[":
            self._depth(depth)
            return self._array(depth + 1)
        if character == '"':
            return _JsonString(self._string())
        return self._other()

    def _other(self) -> _JsonOther:
        """Read a closed non-container primitive for later schema rejection."""
        start = self._index
        while self._peek() and self._peek() not in {
            " ",
            "\n",
            "\r",
            "\t",
            ",",
            "]",
            "}",
        }:
            self._index += 1
        spelling = self._text[start : self._index]
        if spelling in {"true", "false", "null"} or fullmatch(
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?",
            spelling,
        ):
            return _JsonOther(spelling)
        self._error()

    def _object(self, depth: int) -> _JsonObject:
        """Read ordered duplicate-preserving JSON object members."""
        self._take("{")
        self._space()
        members: list[tuple[str, _JsonValue]] = []
        if self._peek() == "}":
            self._index += 1
            return _JsonObject(())
        while True:
            if self._peek() != '"':
                self._error()
            name = self._string()
            self._space()
            self._take(":")
            self._space()
            members.append((name, self._value(depth)))
            self._space()
            if self._peek() == "}":
                self._index += 1
                return _JsonObject(tuple(members))
            self._take(",")
            self._space()

    def _array(self, depth: int) -> _JsonArray:
        """Read one ordered JSON array without object-map collapse."""
        self._take("[")
        self._space()
        values: list[_JsonValue] = []
        if self._peek() == "]":
            self._index += 1
            return _JsonArray(())
        while True:
            values.append(self._value(depth))
            self._space()
            if self._peek() == "]":
                self._index += 1
                return _JsonArray(tuple(values))
            self._take(",")
            self._space()

    def _string(self) -> str:
        """Read one JSON string with strict scalar escape validation."""
        self._take('"')
        characters: list[str] = []
        while True:
            character = self._peek()
            if not character:
                self._error()
            self._index += 1
            if character == '"':
                value = "".join(characters)
                _validate_scalar_text(value, len(self._text))
                return value
            if ord(character) < 0x20:
                self._error()
            if character != "\\":
                characters.append(character)
                continue
            escape = self._peek()
            self._index += 1
            simple = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if escape in simple:
                characters.append(simple[escape])
            elif escape == "u":
                characters.append(self._unicode())
            else:
                self._error()

    def _unicode(self) -> str:
        """Decode one JSON Unicode escape while rejecting lone surrogates."""
        token = self._text[self._index : self._index + 4]
        if len(token) != 4 or fullmatch(r"[0-9A-Fa-f]{4}", token) is None:
            self._error()
        self._index += 4
        first = int(token, 16)
        if 0xD800 <= first <= 0xDBFF:
            if self._text[self._index : self._index + 2] != "\\u":
                self._error()
            self._index += 2
            second_token = self._text[self._index : self._index + 4]
            if fullmatch(r"[0-9A-Fa-f]{4}", second_token) is None:
                self._error()
            self._index += 4
            second = int(second_token, 16)
            if not 0xDC00 <= second <= 0xDFFF:
                self._error()
            return chr(0x10000 + ((first - 0xD800) << 10) + second - 0xDC00)
        if 0xDC00 <= first <= 0xDFFF:
            self._error()
        return chr(first)

    def _space(self) -> None:
        """Consume only JSON's four permitted whitespace code points."""
        while self._peek() in {" ", "\n", "\r", "\t"}:
            self._index += 1

    def _take(self, expected: str) -> None:
        """Require one exact JSON delimiter at the current cursor."""
        if self._peek() != expected:
            self._error()
        self._index += 1

    def _peek(self) -> str:
        """Return the next code point or an empty end sentinel."""
        return self._text[self._index : self._index + 1]

    def _depth(self, depth: int) -> None:
        """Reject the next container before interpreter recursion grows."""
        if depth >= self._max_depth:
            raise PatchInputError(PatchInputErrorCode.OVERSIZED)

    def _error(self) -> NoReturn:
        """Raise one non-content-bearing malformed-input diagnostic."""
        raise PatchInputError(PatchInputErrorCode.MALFORMED)
