"""Exercise dormant lossless text planning over abstract snapshots."""

from asyncio import run
from json import dumps

import pytest

import avalan.patch.planner as planner_module
from avalan.patch.domain import (
    Capability,
    FileMode,
    LogicalPath,
    MetadataProfile,
    ProposedBytes,
    SourceBytes,
)
from avalan.patch.parser import (
    CanonicalPatchRequest,
    PatchHunkSyntax,
    PatchInputLimits,
    PatchLineSyntax,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
    UpdateDeclarationSyntax,
)
from avalan.patch.planner import (
    BoundedPlannerWorker,
    LogicalText,
    MatchKind,
    PlannerError,
    PlannerErrorCode,
    PlannerFacade,
    PlannerFile,
    PlannerLimits,
    PlannerParentMount,
    PlannerWorkspace,
    TextRepresentation,
    apply_replacements,
    find_match,
    plan,
    render_review_diff,
    supported_text,
)


def _request(payload: bytes, kind: RawPatchInputKind) -> CanonicalPatchRequest:
    """Parse one closed canonical request used by pure planner tests."""
    return PatchRequestParser(PatchInputLimits()).parse(
        RawPatchIngress(
            RawProviderProfile("test-provider"),
            RawToolCallId("call_03"),
            kind,
            RawPatchInputState.COMPLETE,
            payload,
        )
    )


def _document(*lines: str) -> bytes:
    """Encode one canonical JSON apply request around a Version 1 document."""
    text = "\n".join(lines) + "\n"
    return dumps({"patch": text}, separators=(",", ":")).encode()


def _file(path: str, value: bytes, *, mount: str = "mount-a") -> PlannerFile:
    """Return one portable abstract regular-file snapshot."""
    view = LogicalText.from_bytes(value)
    return PlannerFile(
        LogicalPath(path),
        SourceBytes(value),
        MetadataProfile(
            FileMode(0o644),
            view.has_bom,
            (
                view.representation.value
                if view.representation.value != "none"
                else "lf"
            ),
        ),
        LogicalPath(path.rsplit("/", 1)[0]) if "/" in path else None,
        mount,
        "identity-" + path,
    )


def _workspace(
    *files: PlannerFile,
    parents: tuple[str, ...] = (),
    parent_mounts: tuple[PlannerParentMount, ...] = (),
) -> PlannerWorkspace:
    """Return typed source and parent observations for one planner call."""
    return PlannerWorkspace(
        files,
        frozenset(LogicalPath(item) for item in parents),
        parent_mounts,
    )


def _edit(old: str, new: str, path: str = "note.txt") -> CanonicalPatchRequest:
    """Return one canonical structured edit request."""
    payload = (
        '{"path":"'
        + path
        + '","edits":[{"old_text":"'
        + old.replace("\n", "\\n").replace("\r", "\\r")
        + '","new_text":"'
        + new.replace("\n", "\\n").replace("\r", "\\r")
        + '"}]}'
    ).encode()
    return _request(payload, RawPatchInputKind.EDIT_JSON)


def _minimum_planner_memory(
    request: CanonicalPatchRequest, workspace: PlannerWorkspace
) -> int:
    """Return the deterministic smallest accepted planner-memory ceiling."""
    maximum = 1
    while True:
        try:
            plan(request, workspace, PlannerLimits(max_memory_bytes=maximum))
        except PlannerError as error:
            assert error.code is PlannerErrorCode.LIMIT
            maximum *= 2
        else:
            break
    minimum = (maximum // 2) + 1
    while minimum < maximum:
        midpoint = (minimum + maximum) // 2
        try:
            plan(request, workspace, PlannerLimits(max_memory_bytes=midpoint))
        except PlannerError as error:
            assert error.code is PlannerErrorCode.LIMIT
            minimum = midpoint + 1
        else:
            maximum = midpoint
    return minimum


def test_patch_phase_3_requirements() -> None:
    """Plan exact edit, create, update, move, and delete terminal effects."""
    edit = plan(
        _edit("green", "blue"), _workspace(_file("note.txt", b"green\n"))
    )
    create = plan(
        _request(
            _document(
                "*** Begin Patch v1",
                "*** Add File: made.txt",
                "+new",
                "*** End Patch",
            ),
            RawPatchInputKind.APPLY_JSON,
        ),
        _workspace(),
    )
    move = plan(
        _request(
            _document(
                "*** Begin Patch v1",
                "*** Update File: old.txt",
                "*** Move to: new.txt",
                "*** End Patch",
            ),
            RawPatchInputKind.APPLY_JSON,
        ),
        _workspace(
            _file("old.txt", b"old\n"),
            parent_mounts=(PlannerParentMount(None, "mount-a"),),
        ),
    )
    delete = plan(
        _request(
            _document(
                "*** Begin Patch v1",
                "*** Delete File: old.txt",
                "*** End Patch",
            ),
            RawPatchInputKind.APPLY_JSON,
        ),
        _workspace(_file("old.txt", b"old\n")),
    )

    assert edit.lineages[0].final.bytes_value is not None
    assert edit.lineages[0].final.bytes_value._value == b"blue\n"
    assert edit.lineages[0].capabilities == frozenset((Capability.UPDATE,))
    assert create.lineages[0].capabilities == frozenset((Capability.CREATE,))
    assert move.lineages[0].source_path == LogicalPath("old.txt")
    assert move.lineages[0].destination_path == LogicalPath("new.txt")
    assert not delete.lineages[0].final.present
    assert edit.diff.digest == edit.diff.digest.from_bytes(edit.diff.rendered)


@pytest.mark.parametrize(
    ("source", "old", "kind"),
    (
        (b"\xef\xbb\xbfalpha\r\nbeta\r\n", "beta\r\n", MatchKind.EXACT_TEXT),
        (b"alpha\r\nbeta\r\n", "beta\n", MatchKind.NEWLINE_COMPATIBLE),
        (b"alpha\nbeta", "beta", MatchKind.EXACT_TEXT),
    ),
)
def test_patch_phase_3_lossless_text_and_exact_first_matching(
    source: bytes, old: str, kind: MatchKind
) -> None:
    """Map supported source bytes to precise spans and exact-first matches."""
    view = supported_text(source)
    match = find_match(view, old, 4)

    assert match.kind is kind
    if kind is MatchKind.EXACT_TEXT:
        assert source[match.span.byte_start : match.span.byte_end].endswith(
            old.encode()
        )
    assert view.byte_offsets[0] == (
        3 if source.startswith(b"\xef\xbb\xbf") else 0
    )


@pytest.mark.parametrize(
    "value",
    (
        b"\xef\xbb\xbf\xef\xbb\xbfx",
        b"\xff",
        b"a\rb",
        b"a\nb\r\nc",
        "\x00",
        "\ud800",
    ),
)
def test_patch_phase_3_rejects_unsupported_text_at_shared_boundary(
    value: bytes | str,
) -> None:
    """Reject unsupported encoding, control, surrogate, and newline forms."""
    with pytest.raises(PlannerError) as error:
        supported_text(value)
    assert error.value.code in {
        PlannerErrorCode.CONTENT,
        PlannerErrorCode.REPRESENTATION,
    }


def test_patch_phase_3_simultaneous_replacements() -> None:
    """Resolve all edits against one source and preserve physical spans."""
    source = b"one\r\ntwo\r\nthree\r\n"
    view = LogicalText.from_bytes(source)
    first = find_match(view, "three\n", 8)
    second = find_match(view, "one\n", 8)
    value = apply_replacements(
        source,
        ((first, "THREE\n"), (second, "ONE\n")),
        TextRepresentation.CRLF,
        PlannerLimits(),
        preserve_final_newline=True,
    )

    assert value == b"ONE\r\ntwo\r\nTHREE\r\n"
    assert b"two\r\n" in value
    with pytest.raises(PlannerError) as error:
        apply_replacements(
            source,
            (
                (find_match(view, "one\n", 8), "x\n"),
                (find_match(view, "two\n", 8), "y\n"),
            ),
            TextRepresentation.CRLF,
            PlannerLimits(),
            preserve_final_newline=True,
        )
    assert error.value.code is PlannerErrorCode.OVERLAPPING_EDITS


def test_patch_phase_3_edit_failure_modes() -> None:
    """Keep matching authority exact and preserve edit final-newline state."""
    candidate = plan(_edit("a", "b"), _workspace(_file("note.txt", b"a\n")))
    final_bytes = candidate.lineages[0].final.bytes_value
    assert final_bytes is not None
    assert final_bytes._value == b"b\n"
    for request, source, code in (
        (_edit("a", "b"), b"a\na\n", PlannerErrorCode.AMBIGUOUS_MATCH),
        (_edit("missing", "b"), b"a\n", PlannerErrorCode.MATCH_NOT_FOUND),
        (_edit("a", "a"), b"a\n", PlannerErrorCode.NO_EFFECT),
        (_edit("a", "a\n"), b"a", PlannerErrorCode.CONFLICT),
    ):
        with pytest.raises(PlannerError) as error:
            plan(request, _workspace(_file("note.txt", source)))
        assert error.value.code is code


def test_patch_phase_3_apply_hunk_semantics() -> None:
    """Apply one declaration's hunks and support exact EOF state."""
    request = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: note.txt",
            "@@",
            "-one",
            "+ONE",
            "@@",
            "-three",
            "+THREE",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    candidate = plan(
        request, _workspace(_file("note.txt", b"one\ntwo\nthree\n"))
    )

    assert candidate.lineages[0].final.bytes_value is not None
    assert (
        candidate.lineages[0].final.bytes_value._value == b"ONE\ntwo\nTHREE\n"
    )
    assert len(candidate.lineages[0].matches) == 2


def test_patch_phase_3_virtual_workspace_transitions() -> None:
    """Collapse valid chains and reject unsafe transitions."""
    update_create = _request(
        _document(
            "*** Begin Patch v1",
            "*** Add File: new.txt",
            "+old",
            "*** Update File: new.txt",
            "@@",
            "-old",
            "+new",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    moved_updated = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old.txt",
            "*** Move to: new.txt",
            "*** Update File: new.txt",
            "@@",
            "-old",
            "+new",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    assert plan(update_create, _workspace()).lineages[0].capabilities
    assert plan(
        moved_updated,
        _workspace(
            _file("old.txt", b"old\n"),
            parent_mounts=(PlannerParentMount(None, "mount-a"),),
        ),
    ).lineages[0].destination_path == LogicalPath("new.txt")
    for request, workspace, code in (
        (
            _request(
                _document(
                    "*** Begin Patch v1",
                    "*** Add File: new.txt",
                    "+a",
                    "*** Update File: new.txt",
                    "*** Move to: other.txt",
                    "*** End Patch",
                ),
                RawPatchInputKind.APPLY_JSON,
            ),
            _workspace(),
            PlannerErrorCode.DESTINATION_EXISTS,
        ),
        (
            _request(
                _document(
                    "*** Begin Patch v1",
                    "*** Delete File: old.txt",
                    "*** Add File: old.txt",
                    "+new",
                    "*** End Patch",
                ),
                RawPatchInputKind.APPLY_JSON,
            ),
            _workspace(_file("old.txt", b"old\n")),
            PlannerErrorCode.CONFLICT,
        ),
        (
            _request(
                _document(
                    "*** Begin Patch v1",
                    "*** Add File: missing/new.txt",
                    "+new",
                    "*** End Patch",
                ),
                RawPatchInputKind.APPLY_JSON,
            ),
            _workspace(),
            PlannerErrorCode.PARENT_MISSING,
        ),
    ):
        with pytest.raises(PlannerError) as error:
            plan(request, workspace)
        assert error.value.code is code


def test_patch_phase_3_limits_and_async_facade() -> None:
    """Fail independent N-plus-one limits and return an async candidate."""
    request = _edit("a", "b")
    workspace = _workspace(_file("note.txt", b"a\n"))
    limits = PlannerLimits(
        max_file_snapshot_bytes=2,
        max_snapshot_bytes=2,
        max_file_proposed_bytes=2,
        max_proposed_bytes=2,
        max_changed_bytes=2,
        max_match_candidates=2,
        max_diff_work_bytes=500,
        max_diff_bytes=500,
        max_memory_bytes=100_000,
    )
    candidate = run(
        PlannerFacade(BoundedPlannerWorker(1), limits).plan(request, workspace)
    )
    assert render_review_diff(candidate, 1) == candidate.diff.rendered[:1]
    with pytest.raises(PlannerError) as error:
        plan(request, workspace, PlannerLimits(max_file_snapshot_bytes=1))
    assert error.value.code is PlannerErrorCode.LIMIT
    with pytest.raises(PlannerError) as error:
        render_review_diff(candidate, -1)
    assert error.value.code is PlannerErrorCode.LIMIT


def test_patch_phase_3_constructor_boundaries() -> None:
    """Reject invalid values before data can be published or scheduled."""
    file = _file("note.txt", b"a\n")
    with pytest.raises(PlannerError):
        PlannerLimits(creation_newline=TextRepresentation.NONE)
    with pytest.raises(PlannerError):
        PlannerFile(file.path, file.bytes_value, file.metadata, None, "", "id")
    with pytest.raises(PlannerError):
        PlannerWorkspace((file, file), frozenset())
    with pytest.raises(PlannerError):
        planner_module.PlannedFile(
            file.path, True, None, file.metadata, None, file.bytes_value.size()
        )
    with pytest.raises(PlannerError):
        planner_module.PlannedFile(
            file.path,
            True,
            file.bytes_value,
            file.metadata,
            None,
            file.bytes_value.size(),
        )
    with pytest.raises(PlannerError):
        BoundedPlannerWorker(0)
    with pytest.raises(PlannerError):
        LogicalText.from_request("\ufeffx")
    with pytest.raises(PlannerError):
        LogicalText.from_bytes(b"a").span(-1, 0)


def test_patch_phase_3_transition_conflicts() -> None:
    """Reject source, delete, move, EOF, and cross-mount failures."""
    missing_edit = _edit("a", "b", "missing.txt")
    with pytest.raises(PlannerError) as error:
        plan(missing_edit, _workspace())
    assert error.value.code is PlannerErrorCode.SOURCE_MISSING
    delete_updated = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old.txt",
            "@@",
            "-old",
            "+new",
            "*** Delete File: old.txt",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    with pytest.raises(PlannerError) as error:
        plan(delete_updated, _workspace(_file("old.txt", b"old\n")))
    assert error.value.code is PlannerErrorCode.CONFLICT
    eof_nonempty = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old.txt",
            "@@",
            "+new",
            "*** End of File",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    with pytest.raises(PlannerError) as error:
        plan(eof_nonempty, _workspace(_file("old.txt", b"old\n")))
    assert error.value.code is PlannerErrorCode.MATCH_NOT_FOUND
    moved = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old.txt",
            "*** Move to: other/new.txt",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    source = _file("old.txt", b"old\n", mount="mount-a")
    sibling = _file("other/present.txt", b"x\n", mount="mount-b")
    with pytest.raises(PlannerError) as error:
        plan(
            moved,
            _workspace(
                source,
                sibling,
                parents=("other",),
                parent_mounts=(
                    PlannerParentMount(LogicalPath("other"), "mount-b"),
                ),
            ),
        )
    assert error.value.code is PlannerErrorCode.MOUNT


def test_patch_phase_3_helper_bounds() -> None:
    """Exercise candidate, memory, absent, and physical helper edge states."""
    view = LogicalText.from_bytes(b"a\na\na")
    with pytest.raises(PlannerError) as error:
        find_match(view, "a", 4)
    assert error.value.code is PlannerErrorCode.AMBIGUOUS_MATCH
    with pytest.raises(PlannerError) as error:
        planner_module._find_all("a", "", 1)
    assert error.value.code is PlannerErrorCode.MATCH_NOT_FOUND
    with pytest.raises(PlannerError) as error:
        planner_module._find_all("aaa", "a", 2)
    assert error.value.code is PlannerErrorCode.LIMIT
    assert planner_module._parent(LogicalPath("flat.txt")) is None
    assert (
        planner_module._mount_for(
            None,
            _workspace(parent_mounts=(PlannerParentMount(None, "mount"),)),
        )
        == "mount"
    )
    assert planner_module._physical("a\n", TextRepresentation.LF) == "a\n"
    candidate = plan(_edit("a", "b"), _workspace(_file("note.txt", b"a\n")))
    with pytest.raises(PlannerError) as error:
        plan(
            _edit("a", "b"),
            _workspace(_file("note.txt", b"a\n")),
            PlannerLimits(max_memory_bytes=1),
        )
    assert error.value.code is PlannerErrorCode.LIMIT
    assert candidate.lineages[0].initial.present


def test_patch_phase_3_covers_terminal_ledger_and_eof_failure_branches() -> (
    None
):
    """Reject terminal path, hunk, and collapsed no-effect transitions."""
    existing = _workspace(_file("old.txt", b"old\n"))
    duplicate_add = _request(
        _document(
            "*** Begin Patch v1",
            "*** Add File: old.txt",
            "+new",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    missing_delete = _request(
        _document(
            "*** Begin Patch v1",
            "*** Delete File: gone.txt",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    missing_update = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: gone.txt",
            "@@",
            "-old",
            "+new",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    missing_move_parent = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old.txt",
            "*** Move to: missing/new.txt",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    for request, workspace, code in (
        (duplicate_add, existing, PlannerErrorCode.DESTINATION_EXISTS),
        (missing_delete, _workspace(), PlannerErrorCode.SOURCE_MISSING),
        (missing_update, _workspace(), PlannerErrorCode.SOURCE_MISSING),
        (missing_move_parent, existing, PlannerErrorCode.PARENT_MISSING),
    ):
        with pytest.raises(PlannerError) as error:
            plan(request, workspace)
        assert error.value.code is code
    empty_eof = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: empty.txt",
            "@@",
            "+new",
            "*** End of File",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    candidate = plan(empty_eof, _workspace(_file("empty.txt", b"")))
    assert candidate.lineages[0].final.bytes_value is not None
    nonterminal_eof = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old.txt",
            "@@",
            " old",
            "+new",
            "*** End of File",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    with pytest.raises(PlannerError) as error:
        plan(nonterminal_eof, _workspace(_file("old.txt", b"old\nmore\n")))
    assert error.value.code is PlannerErrorCode.MATCH_NOT_FOUND
    initial = _file("same.txt", b"same\n")
    lineages = {
        initial.path: planner_module._MutableLineage(
            initial,
            initial.path,
            initial.bytes_value._value,
            initial.metadata,
        )
    }
    assert planner_module._terminal_lineages(lineages, PlannerLimits()) == ()
    absent = planner_module._MutableLineage(None, None, None, None)
    assert (
        planner_module._terminal_lineages(
            {LogicalPath("void.txt"): absent}, PlannerLimits()
        )
        == ()
    )
    with pytest.raises(PlannerError) as error:
        planner_module._limit(2, 1)
    assert error.value.code is PlannerErrorCode.LIMIT
    with pytest.raises(PlannerError) as error:
        planner_module._reserve_memory(
            planner_module._ResourceUsage(), -1, PlannerLimits()
        )
    assert error.value.code is PlannerErrorCode.LIMIT
    with pytest.raises(PlannerError) as error:
        planner_module._reserve_container(
            planner_module._ResourceUsage(), -1, PlannerLimits()
        )
    assert error.value.code is PlannerErrorCode.LIMIT


def test_patch_phase_3_rejects_collapsed_and_tombstoned_no_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover no-effect, repeated update, and tombstoned destination rules."""
    request = _edit("a", "b")
    workspace = _workspace(_file("note.txt", b"a\n"))
    monkeypatch.setattr(
        planner_module, "_terminal_lineages", lambda _a, _b, _c: ()
    )
    with pytest.raises(PlannerError) as error:
        plan(request, workspace)
    assert error.value.code is PlannerErrorCode.NO_EFFECT
    monkeypatch.undo()
    item = _file("note.txt", b"old\n")
    virtual = {item.path: item}
    lineages = {
        item.path: planner_module._MutableLineage(
            item, item.path, item.bytes_value._value, item.metadata
        )
    }
    hunk = PatchHunkSyntax(
        None,
        (PatchLineSyntax("old", True),),
        (PatchLineSyntax("old", True),),
        False,
    )
    with pytest.raises(PlannerError) as error:
        planner_module._update(
            UpdateDeclarationSyntax(item.path, None, (hunk,)),
            virtual,
            lineages,
            set(),
            set(),
            _workspace(item),
            PlannerLimits(),
        )
    assert error.value.code is PlannerErrorCode.NO_EFFECT
    tombstoned_destination = _request(
        _document(
            "*** Begin Patch v1",
            "*** Delete File: dead.txt",
            "*** Update File: old.txt",
            "*** Move to: dead.txt",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    with pytest.raises(PlannerError) as error:
        plan(
            tombstoned_destination,
            _workspace(
                _file("old.txt", b"old\n"), _file("dead.txt", b"dead\n")
            ),
        )
    assert error.value.code is PlannerErrorCode.CONFLICT


def test_patch_phase_3_move_requires_authoritative_parent_mount() -> None:
    """Fail closed until an exact destination-parent mount fact is present."""
    request = _request(
        _document(
            "*** Begin Patch v1",
            "*** Update File: old.txt",
            "*** Move to: new.txt",
            "*** End Patch",
        ),
        RawPatchInputKind.APPLY_JSON,
    )
    source = _file("old.txt", b"old\n", mount="mount-a")
    with pytest.raises(PlannerError) as error:
        PlannerParentMount(None, "")
    assert error.value.code is PlannerErrorCode.MOUNT
    for workspace in (
        _workspace(source),
        _workspace(
            source,
            parents=("other",),
            parent_mounts=(
                PlannerParentMount(LogicalPath("other"), "mount-a"),
            ),
        ),
        _workspace(
            source,
            parent_mounts=(PlannerParentMount(None, "mount-b"),),
        ),
    ):
        with pytest.raises(PlannerError) as error:
            plan(request, workspace)
        assert error.value.code is PlannerErrorCode.MOUNT
    candidate = plan(
        request,
        _workspace(
            source,
            parent_mounts=(PlannerParentMount(None, "mount-a"),),
        ),
    )
    assert candidate.lineages[0].destination_path == LogicalPath("new.txt")


def test_patch_phase_3_diff_is_complete_content_not_digest_summary() -> None:
    """Render complete unified content with absent, move, and EOF markers."""
    changed = plan(
        _edit("alpha", "beta"), _workspace(_file("note.txt", b"alpha\n"))
    ).diff.rendered.decode()
    created = plan(
        _request(
            _document(
                "*** Begin Patch v1",
                "*** Add File: made.txt",
                "+made",
                "*** End Patch",
            ),
            RawPatchInputKind.APPLY_JSON,
        ),
        _workspace(),
    ).diff.rendered.decode()
    moved = plan(
        _request(
            _document(
                "*** Begin Patch v1",
                "*** Update File: old.txt",
                "*** Move to: new.txt",
                "*** End Patch",
            ),
            RawPatchInputKind.APPLY_JSON,
        ),
        _workspace(
            _file("old.txt", b"old\n"),
            parent_mounts=(PlannerParentMount(None, "mount-a"),),
        ),
    ).diff.rendered.decode()
    no_newline = plan(
        _edit("a", "b"), _workspace(_file("note.txt", b"a"))
    ).diff.rendered.decode()

    assert "--- note.txt\n+++ note.txt\n" in changed
    assert "-alpha\n+beta\n" in changed
    assert "--- <absent>\n+++ made.txt\n" in created
    assert "\\ initial: <absent>\n" in created
    assert "+made\n" in created
    assert "rename from old.txt\nrename to new.txt\n" in moved
    assert "\\ No newline at end of file (before)\n" in no_newline
    assert "\\ No newline at end of file (after)\n" in no_newline
    initial = planner_module._planned_initial(
        _file("bom.txt", b"same\n"), None
    )
    bom_bytes = ProposedBytes(b"\xef\xbb\xbfsame\n")
    final = planner_module.PlannedFile(
        LogicalPath("bom.txt"),
        True,
        bom_bytes,
        MetadataProfile(FileMode(0o644), True, "lf"),
        bom_bytes.digest(),
        bom_bytes.size(),
    )
    assert (
        "\\ UTF-8 BOM changed\n"
        in planner_module._diff_entry(
            initial, final, LogicalPath("bom.txt"), LogicalPath("bom.txt")
        ).decode()
    )
    assert not planner_module._file_final_newline(
        planner_module._planned_initial(None, LogicalPath("gone.txt"))
    )


def test_patch_phase_3_aggregates_changed_candidate_and_diff_resources() -> (
    None
):
    """Reject aggregate equal-length changes and retained planner buffers."""
    request = _request(
        dumps(
            {
                "path": "note.txt",
                "edits": [
                    {"old_text": "aa", "new_text": "AA"},
                    {"old_text": "bb", "new_text": "BB"},
                ],
            },
            separators=(",", ":"),
        ).encode(),
        RawPatchInputKind.EDIT_JSON,
    )
    workspace = _workspace(_file("note.txt", b"aa\nbb\n"))
    with pytest.raises(PlannerError) as error:
        plan(request, workspace, PlannerLimits(max_changed_bytes=3))
    assert error.value.code is PlannerErrorCode.LIMIT
    with pytest.raises(PlannerError) as error:
        plan(request, workspace, PlannerLimits(max_match_candidates=1))
    assert error.value.code is PlannerErrorCode.LIMIT
    with pytest.raises(PlannerError) as error:
        plan(request, workspace, PlannerLimits(max_diff_work_bytes=32))
    assert error.value.code is PlannerErrorCode.LIMIT
    candidate = plan(request, workspace, PlannerLimits(max_changed_bytes=4))
    final_bytes = candidate.lineages[0].final.bytes_value
    assert final_bytes is not None
    assert final_bytes._value == b"AA\nBB\n"


def test_patch_phase_3_memory_reserves_before_logical_allocations() -> None:
    """Reject an oversized equal-length edit before planner work."""
    source = b"a" * 1_000
    request = _edit("a" * 1_000, "b" * 1_000)
    workspace = _workspace(_file("note.txt", source))
    with pytest.raises(PlannerError) as error:
        plan(request, workspace, PlannerLimits(max_memory_bytes=60_000))
    assert error.value.code is PlannerErrorCode.LIMIT
    small_request = _edit("a", "b")
    small_workspace = _workspace(_file("note.txt", b"a\n"))
    minimum = _minimum_planner_memory(small_request, small_workspace)
    assert minimum > 1
    plan(
        small_request,
        small_workspace,
        PlannerLimits(max_memory_bytes=minimum),
    )
    with pytest.raises(PlannerError) as error:
        plan(
            small_request,
            small_workspace,
            PlannerLimits(max_memory_bytes=minimum - 1),
        )
    assert error.value.code is PlannerErrorCode.LIMIT
