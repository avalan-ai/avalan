"""Exercise closed raw patch ingress and Version 1 syntax parsing."""

from asyncio import run
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
from json import loads
from pathlib import Path
from random import Random
from traceback import TracebackException

import pytest

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    TextGenerationNonStreamToolCall,
    stream_channel_for_kind,
)
from avalan.patch import parser as parser_module
from avalan.patch.domain import AlgorithmDigest, LogicalPath, OperationType
from avalan.patch.parser import (
    DORMANT_PARAMETER_DESCRIPTORS,
    AddDeclarationSyntax,
    CanonicalPatchRequest,
    PatchDocumentSyntax,
    PatchHunkSyntax,
    PatchInputAccumulator,
    PatchInputError,
    PatchInputErrorCode,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderIngressAdapter,
    RawProviderProfile,
    RawToolCallId,
    StructuredEditSyntax,
    UpdateDeclarationSyntax,
)

_ROOT = Path(__file__).resolve().parents[2]


def _ingress(
    payload: bytes,
    kind: RawPatchInputKind = RawPatchInputKind.APPLY_JSON,
) -> RawPatchIngress:
    """Return one deterministic complete raw provider projection."""
    return RawPatchIngress(
        provider_profile=RawProviderProfile("test-provider"),
        tool_call_id=RawToolCallId("call_01"),
        kind=kind,
        state=RawPatchInputState.COMPLETE,
        raw_bytes=payload,
    )


def _document(*records: str, separator: str = "\n") -> bytes:
    """Encode exact grammar records with a single terminal separator."""
    return (separator.join(records) + separator).encode("utf-8")


def _apply(document: bytes) -> CanonicalPatchRequest:
    """Parse one structured JSON apply payload around raw patch bytes."""
    escaped = (
        document.decode("utf-8").replace("\\", "\\\\").replace("\n", "\\n")
    )
    return PatchRequestParser().parse(
        _ingress(f'{{"patch":"{escaped}"}}'.encode())
    )


async def _provider_items(
    *items: object,
) -> AsyncIterator[object]:
    """Yield a real canonical provider lifecycle without a dispatch target."""
    for item in items:
        yield item


def _provider_item(
    sequence: int,
    kind: StreamItemKind,
    tool_call_id: str,
    *,
    text_delta: str | None = None,
) -> CanonicalStreamItem:
    """Return one real canonical provider tool-call lifecycle item."""
    return CanonicalStreamItem(
        stream_session_id="provider-stream",
        run_id="provider-run",
        turn_id="provider-turn",
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        text_delta=text_delta,
        terminal_outcome=(
            StreamTerminalOutcome.CANCELLED
            if kind is StreamItemKind.STREAM_CANCELLED
            else (
                StreamTerminalOutcome.ERRORED
                if kind is StreamItemKind.STREAM_ERRORED
                else None
            )
        ),
    )


def test_patch_phase_2_requirements() -> None:
    """Accept every Appendix A golden syntax form as immutable values."""
    parser = PatchRequestParser()
    empty = _apply(
        _document(
            "*** Begin Patch v1", "*** Add File: empty.txt", "*** End Patch"
        )
    )
    no_newline = _apply(
        _document(
            "*** Begin Patch v1",
            "*** Add File: note.txt",
            "+hello",
            "\\ No newline at end of file",
            "*** End Patch",
        )
    )
    insertion = _apply(
        _document(
            "*** Begin Patch v1",
            "*** Update File: empty.txt",
            "@@",
            "+hello",
            "\\ No newline at end of file",
            "*** End of File",
            "*** End Patch",
        )
    )
    append = _apply(
        _document(
            "*** Begin Patch v1",
            "*** Update File: log.txt",
            "@@ label",
            " base",
            "+x",
            "*** End of File",
            "*** End Patch",
        )
    )
    move = _apply(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old/name.txt",
            "*** Move to: new/name.txt",
            "*** End Patch",
        )
    )

    assert empty.operation is OperationType.APPLY
    empty_syntax = empty.syntax
    no_newline_syntax = no_newline.syntax
    insertion_syntax = insertion.syntax
    append_syntax = append.syntax
    move_syntax = move.syntax
    assert isinstance(empty_syntax, PatchDocumentSyntax)
    assert isinstance(empty_syntax.declarations[0], AddDeclarationSyntax)
    assert empty_syntax.declarations[0].lines == ()
    assert isinstance(no_newline_syntax, PatchDocumentSyntax)
    assert isinstance(no_newline_syntax.declarations[0], AddDeclarationSyntax)
    assert not no_newline_syntax.declarations[0].lines[-1].has_newline
    assert isinstance(insertion_syntax, PatchDocumentSyntax)
    assert isinstance(
        insertion_syntax.declarations[0], UpdateDeclarationSyntax
    )
    assert insertion_syntax.declarations[0].hunks[0].eof_anchor
    assert isinstance(append_syntax, PatchDocumentSyntax)
    assert isinstance(append_syntax.declarations[0], UpdateDeclarationSyntax)
    assert append_syntax.declarations[0].hunks[0].label == "label"
    assert isinstance(move_syntax, PatchDocumentSyntax)
    assert isinstance(move_syntax.declarations[0], UpdateDeclarationSyntax)
    assert move_syntax.declarations[0].move_to is not None
    with pytest.raises(FrozenInstanceError):
        setattr(empty_syntax, "declarations", ())
    assert parser._limits == PatchInputLimits()


def test_patch_phase_2_structured_json_is_closed_and_canonical() -> None:
    """Preserve duplicates until schema rejection and canonicalize edits."""
    parser = PatchRequestParser()
    first = parser.parse(
        _ingress(
            b'{"path":"a.txt","edits":[{"old_text":"a","new_text":"b"}]}',
            RawPatchInputKind.EDIT_JSON,
        )
    )
    escaped = parser.parse(
        _ingress(
            b'{"edits":[{"new_text":"\\u0062","old_text":"\\u0061"}],"path":"a.txt"}',
            RawPatchInputKind.EDIT_JSON,
        )
    )

    assert isinstance(first.syntax, StructuredEditSyntax)
    assert first.canonical_bytes == escaped.canonical_bytes
    assert first.digest == escaped.digest
    assert (
        b'additionalProperties":false'
        in DORMANT_PARAMETER_DESCRIPTORS[0].schema_bytes
    )
    assert (
        b'additionalProperties":false'
        in DORMANT_PARAMETER_DESCRIPTORS[1].schema_bytes
    )
    assert tuple(item.operation for item in DORMANT_PARAMETER_DESCRIPTORS) == (
        OperationType.EDIT,
        OperationType.APPLY,
    )
    for payload in (
        b'{"path":"a","path":"b","edits":[{"old_text":"a","new_text":"b"}]}',
        b'{"path":"a","edits":[{"old_text":"a","old_text":"b","new_text":"c"}]}',
        b'{"path":"a","edits":[]}',
        b'{"path":"a","edits":[{"old_text":"","new_text":"b"}]}',
        b'{"path":"a","edits":[{"old_text":false,"new_text":"b"}]}',
        b'{"path":"a","edits":[{"old_text":"a","new_text":"b","extra":"x"}]}',
        b'{"patch":"x"}',
    ):
        with pytest.raises(PatchInputError) as error:
            parser.parse(_ingress(payload, RawPatchInputKind.EDIT_JSON))
        assert error.value.code is PatchInputErrorCode.SCHEMA


@pytest.mark.parametrize(
    "path",
    (
        "",
        "/absolute",
        "a/",
        "a//b",
        "a/./b",
        "a/../b",
        "a\\b",
        "~/x",
        "$HOME/x",
        "%USERPROFILE%/x",
        "file:/x",
        "C:/x",
        "//server/x",
        "a:stream",
        "CON",
        "trail /x",
        "x/ lead",
        "x/\x1b[31m",
        "x/\u202ehidden",
    ),
)
def test_patch_phase_2_rejects_unsafe_lexical_paths(path: str) -> None:
    """Reject every prohibited spelling before any target can be selected."""
    payload = (
        '{"path":"'
        + path.replace("\\", "\\\\")
        .replace("\x1b", "\\u001b")
        .replace('"', '\\"')
        + '","edits":[{"old_text":"a","new_text":"b"}]}'
    ).encode("utf-8")
    with pytest.raises(PatchInputError) as error:
        PatchRequestParser().parse(
            _ingress(payload, RawPatchInputKind.EDIT_JSON)
        )
    assert error.value.code is PatchInputErrorCode.PATH


def test_patch_phase_2_enforces_path_and_content_boundaries() -> None:
    """Check lexical limits at N-1, N, and N+1 including UTF-8 bytes."""
    parser = PatchRequestParser(
        PatchInputLimits(
            max_raw_bytes=1000,
            max_scalars=1000,
            max_records=20,
            max_paths=2,
            max_declarations=2,
            max_hunks=2,
            max_edits=2,
            max_path_characters=3,
            max_path_bytes=4,
            max_path_components=2,
            max_component_characters=3,
            max_component_bytes=4,
            max_content_bytes=4,
        )
    )
    for path in ("ab", "abc", "éé"):
        value = (
            '{"path":"' + path + '","edits":[{"old_text":"a","new_text":"b"}]}'
        ).encode()
        parser.parse(_ingress(value, RawPatchInputKind.EDIT_JSON))
    for path in ("abcd", "ééé", "a/b/c"):
        value = (
            '{"path":"' + path + '","edits":[{"old_text":"a","new_text":"b"}]}'
        ).encode()
        with pytest.raises(PatchInputError) as error:
            parser.parse(_ingress(value, RawPatchInputKind.EDIT_JSON))
        assert error.value.code is PatchInputErrorCode.PATH
    for text in ("abc", "abcd"):
        parser.parse(
            _ingress(
                (
                    '{"path":"a","edits":[{"old_text":"a","new_text":"'
                    + text
                    + '"}]}'
                ).encode(),
                RawPatchInputKind.EDIT_JSON,
            )
        )
    with pytest.raises(PatchInputError) as error:
        parser.parse(
            _ingress(
                b'{"path":"a","edits":[{"old_text":"a","new_text":"abcde"}]}',
                RawPatchInputKind.EDIT_JSON,
            )
        )
    assert error.value.code is PatchInputErrorCode.OVERSIZED


@pytest.mark.parametrize(
    "records",
    (
        ("*** Begin Patch", "*** Add File: a", "*** End Patch"),
        ("*** Begin Patch v1", "", "*** Add File: a", "*** End Patch"),
        ("*** Begin Patch v1", "*** Add File: a ", "*** End Patch"),
        ("*** Begin Patch v1", "*** Add File: a", "content", "*** End Patch"),
        ("*** Begin Patch v1", "*** Delete File: a", "+body", "*** End Patch"),
        ("*** Begin Patch v1", "*** Add File: a", "-body", "*** End Patch"),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "*** Move to: b",
            "*** Move to: c",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            " body",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "+x",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "\\ No newline at end of file",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Add File: a",
            "+x",
            "\\ No newline at end of file",
            "+y",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            " old",
            "*** End of File",
            "+x",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            " old",
            "*** End of File",
            "*** End of File",
            "*** End Patch",
        ),
        ("*** Begin Patch v1", "*** Move to: b", "*** End Patch"),
    ),
)
def test_patch_phase_2_rejects_exact_grammar_boundaries(
    records: tuple[str, ...],
) -> None:
    """Reject malformed declarations and hunks before target reads."""
    with pytest.raises(PatchInputError) as error:
        _apply(_document(*records))
    assert error.value.code in {
        PatchInputErrorCode.GRAMMAR,
        PatchInputErrorCode.PATH,
    }


def test_patch_phase_2_handles_hunk_newline_marker_sides() -> None:
    """Represent valid old, new, and shared no-newline marker attachments."""
    old = _apply(
        _document(
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-old",
            "\\ No newline at end of file",
            "+new",
            "*** End of File",
            "*** End Patch",
        )
    )
    new = _apply(
        _document(
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-old",
            "+new",
            "\\ No newline at end of file",
            "*** End of File",
            "*** End Patch",
        )
    )
    both = _apply(
        _document(
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-old",
            "+new",
            " shared",
            "\\ No newline at end of file",
            "*** End of File",
            "*** End Patch",
        )
    )
    for request in (old, new, both):
        syntax = request.syntax
        assert isinstance(syntax, PatchDocumentSyntax)
        declaration = syntax.declarations[0]
        assert isinstance(declaration, UpdateDeclarationSyntax)
        hunk = declaration.hunks[0]
        assert isinstance(hunk, PatchHunkSyntax)
        assert hunk.eof_anchor
    old_syntax = old.syntax
    new_syntax = new.syntax
    both_syntax = both.syntax
    assert isinstance(old_syntax, PatchDocumentSyntax)
    assert isinstance(new_syntax, PatchDocumentSyntax)
    assert isinstance(both_syntax, PatchDocumentSyntax)
    old_declaration = old_syntax.declarations[0]
    new_declaration = new_syntax.declarations[0]
    both_declaration = both_syntax.declarations[0]
    assert isinstance(old_declaration, UpdateDeclarationSyntax)
    assert isinstance(new_declaration, UpdateDeclarationSyntax)
    assert isinstance(both_declaration, UpdateDeclarationSyntax)
    assert not old_declaration.hunks[0].old_lines[-1].has_newline
    assert not new_declaration.hunks[0].new_lines[-1].has_newline
    assert not both_declaration.hunks[0].old_lines[-1].has_newline
    assert not both_declaration.hunks[0].new_lines[-1].has_newline


def test_patch_phase_2_streaming_freeform_and_json_are_byte_identical() -> (
    None
):
    """Parse every raw chunk boundary before generic mapping construction."""
    document = _document(
        "*** Begin Patch v1", "*** Add File: a", "+x", "*** End Patch"
    )
    structured = _apply(document)
    freeform = PatchRequestParser().parse(
        _ingress(document, RawPatchInputKind.VERIFIED_FREEFORM)
    )

    assert structured.canonical_bytes == freeform.canonical_bytes
    assert structured.digest == freeform.digest
    payload = b'{"path":"a","edits":[{"old_text":"x","new_text":"y"}]}'
    expected = PatchRequestParser().parse(
        _ingress(payload, RawPatchInputKind.EDIT_JSON)
    )
    for split in range(len(payload) + 1):
        accumulator = PatchInputAccumulator(
            PatchInputLimits(max_raw_bytes=1000)
        )
        accumulator.append(payload[:split])
        accumulator.append(payload[split:])
        actual = PatchRequestParser().parse(
            accumulator.finish(
                RawProviderProfile("test-provider"),
                RawToolCallId("call_01"),
                RawPatchInputKind.EDIT_JSON,
            )
        )
        assert actual.canonical_bytes == expected.canonical_bytes
    accumulator = PatchInputAccumulator(PatchInputLimits(max_raw_bytes=1))
    with pytest.raises(PatchInputError) as oversized:
        accumulator.append(b"xx")
    assert oversized.value.code is PatchInputErrorCode.OVERSIZED
    accumulator = PatchInputAccumulator(PatchInputLimits())
    accumulator.cancel()
    with pytest.raises(PatchInputError) as cancelled:
        accumulator.finish(
            RawProviderProfile("test-provider"),
            RawToolCallId("call_01"),
            RawPatchInputKind.APPLY_JSON,
        )
    assert cancelled.value.code is PatchInputErrorCode.CANCELLED
    adapter = RawProviderIngressAdapter(PatchInputLimits(max_raw_bytes=1000))
    profile = RawProviderProfile("test-provider")
    target_touches: list[str] = []
    non_streaming = adapter.non_streaming(
        TextGenerationNonStreamToolCall(
            call_id="call_01",
            name="dormant",
            arguments=payload.decode("utf-8"),
            provider_event_type="provider.tool_call",
        ),
        profile,
        RawPatchInputKind.EDIT_JSON,
    )
    assert PatchRequestParser().parse(non_streaming).canonical_bytes == (
        expected.canonical_bytes
    )
    with pytest.raises(PatchInputError) as decoded_mapping:
        adapter.non_streaming(
            {"arguments": payload.decode("utf-8")},
            profile,
            RawPatchInputKind.EDIT_JSON,
        )
    assert decoded_mapping.value.code is PatchInputErrorCode.MALFORMED
    mapping_arguments = TextGenerationNonStreamToolCall(
        call_id="call_01",
        name="dormant",
        arguments="safe",
        provider_event_type="provider.tool_call",
    )
    object.__setattr__(mapping_arguments, "arguments", {"path": "a"})
    with pytest.raises(PatchInputError) as mapping_only:
        adapter.non_streaming(
            mapping_arguments,
            profile,
            RawPatchInputKind.EDIT_JSON,
        )
    assert mapping_only.value.code is PatchInputErrorCode.MALFORMED
    for split in range(len(payload) + 1):
        ingress = run(
            adapter.streaming(
                _provider_items(
                    _provider_item(
                        0,
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        "call_01",
                        text_delta=payload[:split].decode("utf-8"),
                    ),
                    _provider_item(
                        1,
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        "call_01",
                        text_delta=payload[split:].decode("utf-8"),
                    ),
                    _provider_item(
                        2, StreamItemKind.TOOL_CALL_READY, "call_01"
                    ),
                    _provider_item(
                        3, StreamItemKind.TOOL_CALL_DONE, "call_01"
                    ),
                ),
                profile,
                RawToolCallId("call_01"),
                RawPatchInputKind.EDIT_JSON,
            )
        )
        assert PatchRequestParser().parse(ingress).canonical_bytes == (
            expected.canonical_bytes
        )
    for events, expected_code in (
        (
            (
                _provider_item(
                    0,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    "call_01",
                    text_delta='{"path":"a","path":"b","edits":[]}',
                ),
                _provider_item(1, StreamItemKind.TOOL_CALL_READY, "call_01"),
                _provider_item(2, StreamItemKind.TOOL_CALL_DONE, "call_01"),
            ),
            PatchInputErrorCode.SCHEMA,
        ),
        (
            (
                _provider_item(
                    0,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    "call_01",
                    text_delta="{",
                ),
            ),
            PatchInputErrorCode.INCOMPLETE,
        ),
        (
            (_provider_item(0, StreamItemKind.STREAM_ERRORED, "call_01"),),
            PatchInputErrorCode.MALFORMED,
        ),
        (
            (_provider_item(0, StreamItemKind.STREAM_CANCELLED, "call_01"),),
            PatchInputErrorCode.CANCELLED,
        ),
    ):
        with pytest.raises(PatchInputError) as failed:
            ingress = run(
                adapter.streaming(
                    _provider_items(*events),
                    profile,
                    RawToolCallId("call_01"),
                    RawPatchInputKind.EDIT_JSON,
                )
            )
            PatchRequestParser().parse(ingress)
        assert failed.value.code is expected_code
    tiny_adapter = RawProviderIngressAdapter(PatchInputLimits(max_raw_bytes=1))
    with pytest.raises(PatchInputError) as streamed_oversized:
        run(
            tiny_adapter.streaming(
                _provider_items(
                    _provider_item(
                        0,
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        "call_01",
                        text_delta="{}",
                    ),
                ),
                profile,
                RawToolCallId("call_01"),
                RawPatchInputKind.APPLY_JSON,
            )
        )
    assert streamed_oversized.value.code is PatchInputErrorCode.OVERSIZED
    missing_chunk = _provider_item(
        0,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        "call_01",
        text_delta="safe",
    )
    object.__setattr__(missing_chunk, "text_delta", None)
    stream_failures: tuple[tuple[object, ...], ...] = (
        (object(),),
        (missing_chunk,),
        (
            _provider_item(0, StreamItemKind.TOOL_CALL_READY, "call_01"),
            _provider_item(1, StreamItemKind.TOOL_CALL_READY, "call_01"),
        ),
        (_provider_item(0, StreamItemKind.TOOL_CALL_DONE, "call_01"),),
        (
            _provider_item(0, StreamItemKind.TOOL_CALL_READY, "other-call"),
            _provider_item(
                1,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                "call_01",
                text_delta="ignored",
            ),
        ),
    )
    for failure_events in stream_failures:
        with pytest.raises(PatchInputError):
            run(
                adapter.streaming(
                    _provider_items(*failure_events),
                    profile,
                    RawToolCallId("call_01"),
                    RawPatchInputKind.EDIT_JSON,
                )
            )
    assert target_touches == []


def test_patch_phase_2_rejects_transport_and_json_edge_cases() -> None:
    """Reject malformed UTF-8, BOM, non-scalars, and malformed JSON forms."""
    parser = PatchRequestParser()
    for payload, kind in (
        (b"\xef\xbb\xbf{}", RawPatchInputKind.APPLY_JSON),
        (b"\xff", RawPatchInputKind.APPLY_JSON),
        (b'{"patch":"\\ud800"}', RawPatchInputKind.APPLY_JSON),
        (b'{"patch":"\\udc00"}', RawPatchInputKind.APPLY_JSON),
        (b'{"patch":null}', RawPatchInputKind.APPLY_JSON),
        (b'{"patch":"x",}', RawPatchInputKind.APPLY_JSON),
        (b"[]", RawPatchInputKind.APPLY_JSON),
    ):
        with pytest.raises(PatchInputError):
            parser.parse(_ingress(payload, kind))
    with pytest.raises(PatchInputError) as incomplete:
        RawPatchIngress(
            RawProviderProfile("test-provider"),
            RawToolCallId("call_01"),
            RawPatchInputKind.APPLY_JSON,
            RawPatchInputState.INCOMPLETE,
            b"",
        )
    assert incomplete.value.code is PatchInputErrorCode.INCOMPLETE
    with pytest.raises(PatchInputError):
        RawProviderProfile("bad profile")
    with pytest.raises(PatchInputError):
        RawToolCallId("bad call")
    for old_text, new_text in (
        ("old\rnew", "new"),
        ("old\r\nnew\n", "new"),
        ("old", "new\rnext"),
        ("old", "new\r\nnext\n"),
    ):
        escaped_old = old_text.replace("\r", "\\r").replace("\n", "\\n")
        escaped_new = new_text.replace("\r", "\\r").replace("\n", "\\n")
        with pytest.raises(PatchInputError) as mixed_newlines:
            parser.parse(
                _ingress(
                    (
                        '{"path":"a","edits":[{"old_text":"'
                        + escaped_old
                        + '","new_text":"'
                        + escaped_new
                        + '"}]}'
                    ).encode(),
                    RawPatchInputKind.EDIT_JSON,
                )
            )
        assert mixed_newlines.value.code is PatchInputErrorCode.MALFORMED
    for old_text, new_text in (
        ("old\nnext", "new"),
        ("old\r\nnext", "new\r\nnext"),
    ):
        escaped_old = old_text.replace("\r", "\\r").replace("\n", "\\n")
        escaped_new = new_text.replace("\r", "\\r").replace("\n", "\\n")
        assert parser.parse(
            _ingress(
                (
                    '{"path":"a","edits":[{"old_text":"'
                    + escaped_old
                    + '","new_text":"'
                    + escaped_new
                    + '"}]}'
                ).encode(),
                RawPatchInputKind.EDIT_JSON,
            )
        )
    canary = b"raw-canary-should-not-escape-\xff"
    with pytest.raises(PatchInputError) as malformed_utf8:
        parser.parse(_ingress(canary, RawPatchInputKind.APPLY_JSON))
    error = malformed_utf8.value
    rendered = "".join(TracebackException.from_exception(error).format())
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "raw-canary-should-not-escape" not in repr(error)
    assert "raw-canary-should-not-escape" not in rendered
    raw_call = TextGenerationNonStreamToolCall(
        call_id="call_01",
        name="dormant",
        arguments="safe",
        provider_event_type="provider.tool_call",
    )
    object.__setattr__(raw_call, "arguments", "unicode-canary-\ud800")
    with pytest.raises(PatchInputError) as malformed_encode:
        RawProviderIngressAdapter(PatchInputLimits()).non_streaming(
            raw_call,
            RawProviderProfile("test-provider"),
            RawPatchInputKind.APPLY_JSON,
        )
    assert malformed_encode.value.__cause__ is None
    assert malformed_encode.value.__context__ is None
    assert "unicode-canary" not in repr(malformed_encode.value)


def test_patch_phase_2_fixed_seed_malformed_fuzz_never_calls_a_target() -> (
    None
):
    """Classify fixed-seed malformed input without target access."""
    parser = PatchRequestParser(
        PatchInputLimits(max_raw_bytes=128, max_scalars=128)
    )
    probe = parser.parse(
        _ingress(
            b'{"patch":"*** Begin Patch v1\\n*** Add File: probe.txt'
            b'\\n*** End Patch\\n"}'
        )
    )
    assert probe.operation is OperationType.APPLY
    generator = Random(20260808)
    alphabet = b'{}[],:"\\*+ -@\r\n\x00\xffabc'
    accepted = 0
    rejected = 0
    for _ in range(100):
        payload = bytes(
            generator.choice(alphabet) for _ in range(generator.randrange(129))
        )
        try:
            parser.parse(_ingress(payload, RawPatchInputKind.APPLY_JSON))
        except PatchInputError as error:
            rejected += 1
            assert error.code in PatchInputErrorCode
        else:
            accepted += 1
    assert accepted + rejected == 100


def test_patch_phase_2_appendix_a_corpus_is_tracked_and_parser_only() -> None:
    """Run the frozen Appendix A corpus without production fixture reads."""
    payload = loads(
        (_ROOT / "tests/fixtures/patch/appendix_a_corpus.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema_version"] == 1
    assert payload["seed"] == 20260808
    parser = PatchRequestParser()
    for text in payload["valid"]:
        assert parser.parse(
            _ingress(text.encode(), RawPatchInputKind.VERIFIED_FREEFORM)
        )
    for text in payload["invalid"]:
        with pytest.raises(PatchInputError):
            parser.parse(
                _ingress(text.encode(), RawPatchInputKind.VERIFIED_FREEFORM)
            )


def test_patch_phase_2_parser_boundary_failure_paths_remain_closed() -> None:
    """Exercise every bounded parser failure without a target dependency."""
    syntax = StructuredEditSyntax(
        path=LogicalPath("a"),
        edits=(parser_module.TextEditSyntax("a", "b"),),
    )
    with pytest.raises(PatchInputError):
        CanonicalPatchRequest(
            OperationType.EDIT,
            syntax,
            b"x",
            AlgorithmDigest.from_bytes(b"y"),
        )
    with pytest.raises(PatchInputError):
        PatchInputLimits(max_raw_bytes=0)
    parser = PatchRequestParser(PatchInputLimits(max_raw_bytes=1))
    with pytest.raises(PatchInputError) as oversized:
        parser.parse(_ingress(b"{}"))
    assert oversized.value.code is PatchInputErrorCode.OVERSIZED
    malformed_kind = RawPatchIngress(
        RawProviderProfile("test-provider"),
        RawToolCallId("call_01"),
        RawPatchInputKind.APPLY_JSON,
        RawPatchInputState.COMPLETE,
        b"{}",
    )
    object.__setattr__(malformed_kind, "kind", "unexpected")
    with pytest.raises(PatchInputError):
        PatchRequestParser().parse(malformed_kind)
    accumulator = PatchInputAccumulator(PatchInputLimits())
    accumulator.finish(
        RawProviderProfile("test-provider"),
        RawToolCallId("call_01"),
        RawPatchInputKind.APPLY_JSON,
    )
    with pytest.raises(PatchInputError) as double_append:
        accumulator.append(b"x")
    assert double_append.value.code is PatchInputErrorCode.MALFORMED
    for state, expected in (
        (RawPatchInputState.INCOMPLETE, PatchInputErrorCode.INCOMPLETE),
        (RawPatchInputState.CANCELLED, PatchInputErrorCode.CANCELLED),
        (RawPatchInputState.OVERSIZED, PatchInputErrorCode.OVERSIZED),
        (RawPatchInputState.MALFORMED, PatchInputErrorCode.MALFORMED),
        (RawPatchInputState.COMPLETE, PatchInputErrorCode.MALFORMED),
    ):
        assert parser_module._state_error(state) is expected
    for value in ("a\rb", "a\r\nb\r"):
        with pytest.raises(PatchInputError):
            parser_module._validate_content(value, PatchInputLimits())
    for value in ("a\rb", "a\r\nb\n"):
        with pytest.raises(PatchInputError):
            parser_module._records(value, 4)
    assert parser_module._records("a\r\nb\r\n", 4) == ("a", "b")
    assert parser_module._records("a\nb", 4) == ("a", "b")
    for value in ("@", "@@ ", "@@ bad ", "@@ \u202e"):
        with pytest.raises(PatchInputError):
            parser_module._label(value)
    members = (("a", parser_module._JsonString("x")),)
    with pytest.raises(PatchInputError):
        parser_module._member(members, "b")
    with pytest.raises(PatchInputError):
        parser_module._array_member(members, "a")
    reader_inputs = (
        "{}",
        "[",
        '{"a":1,}',
        '"unterminated',
        '"\x01"',
        '"\\q"',
        '"\\uZZZZ"',
        '"\\ud800"',
        '"\\ud800\\uZZZZ"',
        '"\\ud800\\u0000"',
        '"\\udc00"',
        "{} trailing",
    )
    for value in reader_inputs:
        reader = parser_module._JsonReader(value)
        if value == "{}":
            assert isinstance(reader.parse(), parser_module._JsonObject)
        else:
            with pytest.raises(PatchInputError):
                reader.parse()
    surrogate = parser_module._JsonReader('"\\ud83d\\ude00"').parse()
    assert isinstance(surrogate, parser_module._JsonString)
    assert surrogate.value == "😀"
    for depth in (2, 3):
        nested = parser_module._JsonReader(
            "[" * depth + "0" + "]" * depth, max_depth=3
        ).parse()
        assert isinstance(nested, parser_module._JsonArray)
    with pytest.raises(PatchInputError) as depth_overflow:
        parser_module._JsonReader("[" * 4 + "0" + "]" * 4, max_depth=3).parse()
    assert depth_overflow.value.code is PatchInputErrorCode.OVERSIZED
    with pytest.raises(PatchInputError):
        parser_module._JsonReader("[]", max_depth=0)
    with pytest.raises(PatchInputError) as deeply_nested:
        PatchRequestParser().parse(_ingress(b"[" * 1100 + b"0" + b"]" * 1100))
    assert deeply_nested.value.code is PatchInputErrorCode.OVERSIZED
    with pytest.raises(PatchInputError):
        parser_module._JsonReader("x")._take("y")
    two_edits = PatchRequestParser().parse(
        _ingress(
            b'{"path":"a","edits":[{"old_text":"a","new_text":"b"},{"old_text":"c","new_text":"d"}]}',
            RawPatchInputKind.EDIT_JSON,
        )
    )
    assert b"},{" in two_edits.canonical_bytes
    with pytest.raises(PatchInputError):
        _apply(
            _document(
                "*** Begin Patch v1",
                "*** Add File: a",
                "\\ No newline at end of file",
                "*** End Patch",
            )
        )
    global_hunk_limits = PatchInputLimits(max_hunks=2)

    def parse_global_hunks(count: int) -> CanonicalPatchRequest:
        """Parse updates under one request-wide hunk cap."""
        records = ["*** Begin Patch v1"]
        for index in range(count):
            records.extend(
                (
                    f"*** Update File: file-{index}",
                    "@@",
                    "-old",
                    "+new",
                )
            )
        records.append("*** End Patch")
        document = _document(*records).decode("utf-8")
        payload = (
            '{"patch":"' + document.replace("\n", "\\n") + '"}'
        ).encode()
        return PatchRequestParser(global_hunk_limits).parse(_ingress(payload))

    assert isinstance(parse_global_hunks(1).syntax, PatchDocumentSyntax)
    assert isinstance(parse_global_hunks(2).syntax, PatchDocumentSyntax)
    with pytest.raises(PatchInputError) as global_hunk_overflow:
        parse_global_hunks(3)
    assert global_hunk_overflow.value.code is PatchInputErrorCode.OVERSIZED


def test_patch_phase_2_parser_limit_and_hunk_failure_branches() -> None:
    """Reject bounded declaration and hunk states without target inspection."""
    limits = PatchInputLimits(
        max_raw_bytes=1000,
        max_scalars=1000,
        max_records=100,
        max_paths=10,
        max_declarations=1,
        max_hunks=1,
        max_edits=10,
        max_path_characters=100,
        max_path_bytes=100,
        max_path_components=10,
        max_component_characters=100,
        max_component_bytes=100,
        max_content_bytes=100,
    )

    def parse_records(*records: str) -> None:
        """Drive one exact document through the structured raw parser."""
        text = _document(*records).decode().replace("\\", "\\\\")
        payload = b'{"patch":"' + text.replace("\n", "\\n").encode() + b'"}'
        PatchRequestParser(limits).parse(_ingress(payload))

    cases = (
        (
            "*** Begin Patch v1",
            "*** Add File: a",
            "*** Add File: b",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-a",
            "+b",
            "@@",
            "-c",
            "+d",
            "*** End Patch",
        ),
        ("*** Begin Patch v1", "*** Update File: a", "*** End Patch"),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-a",
            "\\ No newline at end of file",
            " context",
            "*** End of File",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-a",
            "+b",
            "\\ No newline at end of file",
            "+c",
            "*** End of File",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-a",
            "*unknown",
            "*** End of File",
            "*** End Patch",
        ),
        (
            "*** Begin Patch v1",
            "*** Update File: a",
            "@@",
            "-a",
            "\\ No newline at end of file",
            "+b",
            "*** End Patch",
        ),
    )
    for case in cases:
        with pytest.raises(PatchInputError):
            parse_records(*case)
    one_path_limits = PatchInputLimits(
        max_raw_bytes=1000,
        max_scalars=1000,
        max_records=100,
        max_paths=1,
        max_declarations=10,
        max_hunks=10,
        max_edits=10,
        max_path_characters=100,
        max_path_bytes=100,
        max_path_components=10,
        max_component_characters=100,
        max_component_bytes=100,
        max_content_bytes=100,
    )
    with pytest.raises(PatchInputError):
        PatchRequestParser(one_path_limits).parse(
            _ingress(
                b'{"patch":"*** Begin Patch v1\\n*** Update File: a\\n'
                b'*** Move to: b\\n*** End Patch\\n"}'
            )
        )
    invalid = RawPatchIngress(
        RawProviderProfile("test-provider"),
        RawToolCallId("call_01"),
        RawPatchInputKind.APPLY_JSON,
        RawPatchInputState.COMPLETE,
        b"",
    )
    object.__setattr__(invalid, "raw_bytes", "wrong")
    with pytest.raises(PatchInputError):
        invalid.__post_init__()
    with pytest.raises(PatchInputError):
        _apply(
            _document(
                "*** Begin Patch v1",
                "*** Update File: a",
                "@@",
                " old",
                "\\ No newline at end of file",
                "*** End Patch",
            )
        )
    with pytest.raises(PatchInputError):
        _apply(
            _document(
                "*** Begin Patch v1",
                "*** Update File: a",
                "@@",
                "-old",
                "\\ No newline at end of file",
                "-later",
                "*** End of File",
                "*** End Patch",
            )
        )
