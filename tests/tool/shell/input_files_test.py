from pathlib import Path

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallContext,
)
from avalan.tool.shell.input_files import (
    _add_alias,
    _cwd_relative_file_path,
    _effective_shell_cwd,
    _input_file_path_aliases,
    _is_relative_to,
    _iter_input_file_content,
    _iter_message_file_content,
    _path_alias,
    _path_has_part,
    _rewrite_path_argument,
    _rewrite_paths_argument,
    _rewrite_shell_input_file_paths,
    shell_input_file_filter,
)
from avalan.tool.shell.settings import ShellToolSettings


def test_shell_input_file_filter_rewrites_attachment_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")
    message = _message(source)
    call = ToolCall(
        id="call-1",
        name="shell.pdfinfo",
        arguments={
            "path": f"./{source.name}",
            "paths": [str(source.resolve()), "other.pdf"],
        },
    )
    settings = ShellToolSettings(workspace_root=str(tmp_path))
    tool_filter = shell_input_file_filter(settings)

    result = tool_filter.func(call, ToolCallContext(input=message))

    assert result is not None
    filtered_call, context = result
    assert context.input == message
    assert filtered_call.arguments == {
        "path": "nested/report.pdf",
        "paths": ["nested/report.pdf", "other.pdf"],
    }


def test_shell_input_file_filter_preserves_unmapped_arguments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")
    aliases = _input_file_path_aliases(
        [
            Message(role=MessageRole.USER, content="plain"),
            _message(source),
        ],
        ShellToolSettings(workspace_root=str(tmp_path)),
    )

    assert aliases[source.name] == "nested/report.pdf"
    assert _rewrite_path_argument(1, aliases) == (1, False)
    assert _rewrite_path_argument("missing.pdf", aliases) == (
        "missing.pdf",
        False,
    )
    assert _rewrite_path_argument("nested/report.pdf", aliases) == (
        "nested/report.pdf",
        False,
    )
    assert _rewrite_paths_argument("report.pdf", aliases) == (
        "report.pdf",
        False,
    )
    assert _rewrite_paths_argument(["missing.pdf"], aliases) == (
        ["missing.pdf"],
        False,
    )
    assert _rewrite_paths_argument((source.name,), aliases) == (
        ("nested/report.pdf",),
        True,
    )


def test_shell_input_file_aliases_are_relative_to_settings_cwd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")
    settings = ShellToolSettings(
        workspace_root=str(tmp_path),
        cwd="nested",
    )

    aliases = _input_file_path_aliases(_message(source), settings)

    assert aliases[source.name] == "report.pdf"
    assert _rewrite_path_argument(f"./{source.name}", aliases) == (
        "report.pdf",
        True,
    )
    assert aliases[str(source.resolve())] == "report.pdf"


def test_shell_input_file_filter_uses_per_call_cwd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")
    settings = ShellToolSettings(
        workspace_root=str(tmp_path),
        cwd="other",
    )
    call = ToolCall(
        id="call-1",
        name="shell.pdfinfo",
        arguments={
            "path": source.name,
            "paths": [str(source.resolve())],
            "cwd": "nested",
        },
    )

    result = _rewrite_shell_input_file_paths(
        call,
        ToolCallContext(input=_message(source)),
        settings,
    )

    assert result is not None
    filtered_call, _ = result
    assert filtered_call.arguments == {
        "path": "report.pdf",
        "paths": ["report.pdf"],
        "cwd": "nested",
    }


def test_shell_input_file_aliases_skip_files_outside_effective_cwd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")
    nested = tmp_path / "nested"
    nested.mkdir()

    assert (
        _input_file_path_aliases(
            _message(source),
            ShellToolSettings(workspace_root=str(tmp_path), cwd="nested"),
        )
        == {}
    )
    assert (
        _cwd_relative_file_path(nested.resolve(), source.resolve())
        is None
    )


def test_effective_shell_cwd_rejects_invalid_values(tmp_path: Path) -> None:
    workspace_root = tmp_path.resolve()
    settings = ShellToolSettings(workspace_root=str(tmp_path))
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")

    assert (
        _input_file_path_aliases(
            _message(source),
            settings,
            request_cwd=1,
        )
        == {}
    )
    assert _effective_shell_cwd(workspace_root, settings, 1) is None
    assert _effective_shell_cwd(workspace_root, settings, "") is None
    assert _effective_shell_cwd(workspace_root, settings, "../tmp") is None
    assert (
        _effective_shell_cwd(
            workspace_root,
            settings,
            str(workspace_root),
        )
        is None
    )
    assert (
        _effective_shell_cwd(
            workspace_root,
            ShellToolSettings(
                workspace_root=str(tmp_path),
                allow_absolute_paths=True,
            ),
            str(tmp_path.parent),
        )
        is None
    )
    assert _effective_shell_cwd(
        workspace_root,
        ShellToolSettings(
            workspace_root=str(tmp_path),
            allow_absolute_paths=True,
        ),
        str(workspace_root),
    ) == workspace_root


def test_shell_input_file_rewrite_noops(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")
    settings = ShellToolSettings(workspace_root=str(tmp_path))
    context = ToolCallContext(input=_message(source))

    assert (
        _rewrite_shell_input_file_paths(
            ToolCall(id="call-1", name="shell.pdfinfo", arguments=None),
            context,
            settings,
        )
        is None
    )
    assert (
        _rewrite_shell_input_file_paths(
            ToolCall(
                id="call-2",
                name="shell.pdfinfo",
                arguments={"path": source.name},
            ),
            ToolCallContext(),
            settings,
        )
        is None
    )
    assert (
        _rewrite_shell_input_file_paths(
            ToolCall(
                id="call-3",
                name="shell.pdfinfo",
                arguments={"path": source.name},
            ),
            context,
            settings,
        )
        is None
    )


def test_shell_input_file_aliases_skip_unsafe_or_ambiguous_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "same.pdf"
    second = tmp_path / "second" / "same.pdf"
    outside = tmp_path.parent / "outside.pdf"
    for path in (first, second, outside):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"%PDF-1.7")
    invalid_filename = MessageContentFile(
        type="file",
        file={"filename": "", "local_path": str(first.resolve())},
    )
    invalid_path = MessageContentFile(
        type="file",
        file={"filename": "empty.pdf", "local_path": ""},
    )

    aliases = _input_file_path_aliases(
        Message(
            role=MessageRole.USER,
            content=[
                invalid_filename,
                invalid_path,
                _file_content(first),
                _file_content(second),
                _file_content(outside),
            ],
        ),
        ShellToolSettings(workspace_root=str(tmp_path)),
    )

    assert "same.pdf" not in aliases
    assert "empty.pdf" not in aliases
    assert outside.name not in aliases
    assert aliases[str(first.resolve())] == "first/same.pdf"
    assert aliases[str(second.resolve())] == "second/same.pdf"


def test_input_file_iterators_cover_message_shapes(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")
    file_content = _file_content(source)
    single = Message(role=MessageRole.USER, content=file_content)
    text = Message(
        role=MessageRole.USER,
        content=MessageContentText(type="text", text="x"),
    )
    mixed = Message(
        role=MessageRole.USER,
        content=[MessageContentText(type="text", text="x"), file_content],
    )

    assert list(_iter_message_file_content(single)) == [file_content]
    assert list(_iter_message_file_content(text)) == []
    assert list(_iter_message_file_content(mixed)) == [file_content]
    assert list(_iter_input_file_content(None)) == []
    assert list(_iter_input_file_content("text")) == []
    assert list(_iter_input_file_content(single)) == [file_content]
    assert list(_iter_input_file_content([text, mixed])) == [file_content]


def test_add_alias_and_path_alias_helpers() -> None:
    aliases: dict[str, str] = {}
    conflicts: set[str] = set()

    _add_alias(aliases, conflicts, "", "a")
    _add_alias(aliases, conflicts, "name.pdf", "a")
    _add_alias(aliases, conflicts, "name.pdf", "a")
    _add_alias(aliases, conflicts, "name.pdf", "b")
    _add_alias(aliases, conflicts, "name.pdf", "c")

    assert aliases == {}
    assert conflicts == {"name.pdf"}
    assert _path_alias("././name.pdf") == "name.pdf"
    assert _is_relative_to(Path("a/b"), Path("a"))
    assert not _is_relative_to(Path("a/b"), Path("c"))
    assert _path_has_part(Path("a/../b"), "..")


def _message(path: Path) -> Message:
    return Message(
        role=MessageRole.USER,
        content=[
            MessageContentText(type="text", text="read this"),
            _file_content(path),
        ],
    )


def _file_content(path: Path) -> MessageContentFile:
    return MessageContentFile(
        type="file",
        file={
            "filename": path.name,
            "local_path": str(path.resolve()),
        },
    )
