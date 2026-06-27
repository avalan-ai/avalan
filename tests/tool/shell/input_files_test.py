from base64 import b64encode
from inspect import isawaitable
from pathlib import Path
from types import SimpleNamespace

import pytest

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
)
from avalan.tool.shell import input_files as shell_input_files
from avalan.tool.shell.entities import (
    ExecutionResult,
    GeneratedFile,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellOutputKind,
)
from avalan.tool.shell.input_files import (
    _add_alias,
    _add_generated_file_path_aliases,
    _cwd_relative_file_path,
    _decode_base64_payload,
    _effective_shell_cwd,
    _execution_result,
    _generated_file_filename,
    _input_file_path_alias_values,
    _input_file_path_aliases,
    _is_relative_to,
    _iter_generated_files,
    _iter_input_file_content,
    _iter_message_file_content,
    _make_directory_tree,
    _path_alias,
    _path_has_part,
    _path_suffix_aliases,
    _rewrite_path_argument,
    _rewrite_paths_argument,
    _rewrite_shell_input_file_paths,
    _shell_file_path_aliases,
    _workspace_relative_path,
    shell_input_file_filter,
)
from avalan.tool.shell.settings import ShellToolSettings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_shell_input_file_filter_rewrites_attachment_paths(
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

    result = await tool_filter.func(call, ToolCallContext(input=message))

    assert result is not None
    filtered_call, context = result
    assert context.input == message
    assert filtered_call.arguments == {
        "path": "nested/report.pdf",
        "paths": ["nested/report.pdf", "other.pdf"],
    }


async def test_shell_input_file_filter_preserves_unmapped_arguments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")
    aliases = await _input_file_path_aliases(
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


async def test_shell_input_file_aliases_are_relative_to_settings_cwd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nested" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")
    settings = ShellToolSettings(
        workspace_root=str(tmp_path),
        cwd="nested",
    )

    aliases = await _input_file_path_aliases(_message(source), settings)

    assert aliases[source.name] == "report.pdf"
    assert _rewrite_path_argument(f"./{source.name}", aliases) == (
        "report.pdf",
        True,
    )
    assert aliases[str(source.resolve())] == "report.pdf"


async def test_shell_input_file_filter_uses_per_call_cwd(
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

    result = await _rewrite_shell_input_file_paths(
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


async def test_shell_input_file_filter_rewrites_workspace_suffixes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "batches" / "client_docs" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7")
    settings = ShellToolSettings(workspace_root=str(tmp_path))
    call = ToolCall(
        id="call-1",
        name="shell.pdfinfo",
        arguments={
            "path": "client_docs/report.pdf",
            "paths": [
                "batches/client_docs/report.pdf",
                "./inputs/batches/client_docs/report.pdf",
            ],
        },
    )

    result = await _rewrite_shell_input_file_paths(
        call,
        ToolCallContext(input=_message(source)),
        settings,
    )

    assert result is not None
    filtered_call, _ = result
    workspace_path = "inputs/batches/client_docs/report.pdf"
    assert filtered_call.arguments == {
        "path": workspace_path,
        "paths": [workspace_path, workspace_path],
    }


async def test_shell_input_file_manifest_lists_workspace_relative_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "batches" / "client_docs" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7")

    manifest = await _shell_input_file_manifest(
        _message(source),
        ShellToolSettings(workspace_root=str(tmp_path)),
    )

    assert manifest is not None
    assert manifest.startswith("Attached files available to tools:\n")
    assert "Use these path values as tool arguments." in manifest
    assert "shell tools" not in manifest
    assert "JSON" not in manifest
    assert "report.pdf" in manifest
    assert "inputs/batches/client_docs/report.pdf" in manifest
    assert str(source.resolve()) not in manifest


async def test_shell_input_file_manifest_uses_custom_message(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")

    manifest = await _shell_input_file_manifest(
        _message(source),
        ShellToolSettings(
            workspace_root=str(tmp_path),
            input_file_manifest_message="Files for available tools:",
            input_file_manifest_path_message="Pass these paths to tools.",
        ),
    )

    assert manifest is not None
    assert manifest.startswith(
        "Files for available tools:\nPass these paths to tools.\n"
    )
    assert "inputs/report.pdf" in manifest


async def test_shell_input_file_manifest_can_be_disabled(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.7")

    manifest = await _shell_input_file_manifest(
        _message(source),
        ShellToolSettings(
            workspace_root=str(tmp_path),
            input_file_manifest_enabled=False,
        ),
    )

    assert manifest is None


async def test_shell_input_file_manifest_omits_unavailable_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside" / "report.pdf"
    outside.parent.mkdir()
    outside.write_bytes(b"%PDF-1.7")
    message = Message(
        role=MessageRole.USER,
        content=[
            MessageContentText(type="text", text="inspect attachment"),
            _file_content(outside),
        ],
    )

    manifest = await _shell_input_file_manifest(
        message,
        ShellToolSettings(workspace_root=str(workspace)),
    )

    assert manifest is None


async def test_shell_input_file_aliases_skip_files_outside_effective_cwd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")
    nested = tmp_path / "nested"
    nested.mkdir()

    assert (
        await _input_file_path_aliases(
            _message(source),
            ShellToolSettings(workspace_root=str(tmp_path), cwd="nested"),
        )
        == {}
    )
    assert _cwd_relative_file_path(nested.resolve(), source.resolve()) is None


async def test_effective_shell_cwd_rejects_invalid_values(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path.resolve()
    settings = ShellToolSettings(workspace_root=str(tmp_path))
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")

    assert (
        await _input_file_path_aliases(
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
    assert (
        _effective_shell_cwd(
            workspace_root,
            ShellToolSettings(
                workspace_root=str(tmp_path),
                allow_absolute_paths=True,
            ),
            str(workspace_root),
        )
        == workspace_root
    )


async def test_shell_file_path_aliases_skip_when_effective_cwd_invalid(
    tmp_path: Path,
) -> None:
    aliases = await _shell_file_path_aliases(
        ToolCallContext(input=_message(tmp_path / "missing.pdf")),
        ShellToolSettings(workspace_root=str(tmp_path)),
        request_cwd=object(),
    )

    assert aliases == {}


async def test_shell_input_file_rewrite_noops(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")
    settings = ShellToolSettings(workspace_root=str(tmp_path))
    context = ToolCallContext(input=_message(source))

    assert (
        await _rewrite_shell_input_file_paths(
            ToolCall(id="call-1", name="shell.pdfinfo", arguments=None),
            context,
            settings,
        )
        is None
    )
    assert (
        await _rewrite_shell_input_file_paths(
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
        await _rewrite_shell_input_file_paths(
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


async def test_shell_input_file_aliases_skip_unsafe_or_ambiguous_files(
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

    aliases = await _input_file_path_aliases(
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


async def test_shell_input_file_filter_materializes_base64_attachments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ids = iter(
        [
            SimpleNamespace(hex="fixed"),
            SimpleNamespace(hex="second"),
        ]
    )
    monkeypatch.setattr(
        "avalan.tool.shell.input_files.uuid4",
        lambda: next(ids),
    )
    file_data = b64encode(b"%PDF-1.7").decode("ascii")
    message = Message(
        role=MessageRole.USER,
        content=[
            MessageContentFile(
                type="file",
                file={
                    "file_data": f"data:application/pdf;base64,{file_data}",
                    "filename": "../report.pdf",
                    "local_path": str(tmp_path.parent / "report.pdf"),
                },
            )
        ],
    )
    settings = ShellToolSettings(
        workspace_root=str(tmp_path),
        materialized_input_files_dir="materialized/inputs",
    )
    call = ToolCall(
        id="call-1",
        name="shell.pdfinfo",
        arguments={"path": "report.pdf", "paths": ["./report.pdf"]},
    )

    result = await _rewrite_shell_input_file_paths(
        call,
        ToolCallContext(input=message),
        settings,
    )

    assert result is not None
    filtered_call, _ = result
    assert filtered_call.arguments == {
        "path": "materialized/inputs/fixed/report.pdf",
        "paths": ["materialized/inputs/fixed/report.pdf"],
    }
    materialized = (
        tmp_path / "materialized" / "inputs" / "fixed" / "report.pdf"
    )
    assert materialized.read_bytes() == b"%PDF-1.7"

    aliases = await _input_file_path_aliases(
        Message(
            role=MessageRole.USER,
            content=MessageContentFile(
                type="file",
                file={
                    "filename": "second.pdf",
                    "file_data": b64encode(b"second").decode("ascii"),
                },
            ),
        ),
        settings,
    )

    assert aliases["second.pdf"] == "materialized/inputs/second/second.pdf"
    second = tmp_path / "materialized" / "inputs" / "second" / "second.pdf"
    assert second.read_bytes() == b"second"


async def test_shell_input_file_filter_materializes_generated_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "avalan.tool.shell.input_files.uuid4",
        lambda: SimpleNamespace(hex="generated"),
    )
    generated_file = GeneratedFile(
        display_path="GENERATED_PREFIX-1.png",
        media_type="image/png",
        suffix=".png",
        bytes=5,
        content_base64=b64encode(b"image").decode("ascii"),
    )
    execution_result = ExecutionResult(
        backend="local",
        tool_name="shell.pdftoppm",
        command="pdftoppm",
        argv=("pdftoppm",),
        display_argv=("pdftoppm", "GENERATED_PREFIX"),
        cwd=str(tmp_path),
        display_cwd=".",
        status=ShellExecutionStatus.COMPLETED,
        exit_code=0,
        stdout="",
        stderr="",
        stdout_media_type="application/json",
        output_kind=ShellOutputKind.GENERATED_FILES,
        generated_files=(generated_file,),
    )
    previous = ToolCallResult(
        id="result-1",
        call=ToolCall(
            id="call-1",
            name="shell.pdftoppm",
            arguments={"path": "report.pdf"},
        ),
        name="shell.pdftoppm",
        arguments={"path": "report.pdf"},
        result=ShellFormattedResult("formatted", execution_result),
    )
    call = ToolCall(
        id="call-2",
        name="shell.tesseract",
        arguments={"path": "GENERATED_PREFIX-1.png"},
    )

    result = await _rewrite_shell_input_file_paths(
        call,
        ToolCallContext(calls=[previous]),
        ShellToolSettings(
            workspace_root=str(tmp_path),
            materialized_input_files_dir="generated-files",
        ),
    )

    assert result is not None
    filtered_call, _ = result
    assert filtered_call.arguments == {
        "path": "generated-files/generated/GENERATED_PREFIX-1.png"
    }
    materialized = (
        tmp_path / "generated-files" / "generated" / "GENERATED_PREFIX-1.png"
    )
    assert materialized.read_bytes() == b"image"


async def test_shell_input_file_aliases_skip_unmaterializable_files(
    tmp_path: Path,
) -> None:
    aliases = await _input_file_path_aliases(
        Message(
            role=MessageRole.USER,
            content=[
                MessageContentFile(
                    type="file",
                    file={
                        "filename": 1,
                        "file_data": b64encode(b"%PDF-1.7").decode("ascii"),
                    },
                ),
                MessageContentFile(
                    type="file",
                    file={"filename": "broken.pdf", "file_data": "data:"},
                ),
                MessageContentFile(
                    type="file",
                    file={"filename": "bad.pdf", "base64": "not base64!"},
                ),
                MessageContentFile(
                    type="file",
                    file={"filename": "blank.pdf", "data": " "},
                ),
            ],
        ),
        ShellToolSettings(workspace_root=str(tmp_path)),
    )

    assert aliases == {}
    assert not (tmp_path / "avalan-input-files").exists()


async def test_make_directory_tree_handles_stop_and_races(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[Path] = []

    async def make_directory(path: Path) -> None:
        calls.append(path)
        if len(calls) == 1:
            raise FileNotFoundError(path)
        if len(calls) == 3:
            raise FileExistsError(path)

    await _make_directory_tree(tmp_path, stop_at=tmp_path)

    monkeypatch.setattr(
        "avalan.tool.shell.input_files._make_directory",
        make_directory,
    )

    await _make_directory_tree(tmp_path / "parent" / "child", stop_at=tmp_path)

    assert calls == [
        tmp_path / "parent" / "child",
        tmp_path / "parent",
        tmp_path / "parent" / "child",
    ]


async def test_make_directory_tree_reraises_outside_stop_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def make_directory(_path: Path) -> None:
        raise FileNotFoundError(_path)

    monkeypatch.setattr(
        "avalan.tool.shell.input_files._make_directory",
        make_directory,
    )

    with pytest.raises(FileNotFoundError):
        await _make_directory_tree(tmp_path.parent, stop_at=tmp_path)


async def test_generated_file_aliases_skip_unusable_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ids = iter(
        [
            SimpleNamespace(hex="valid"),
            SimpleNamespace(hex="outside"),
        ]
    )
    monkeypatch.setattr(
        "avalan.tool.shell.input_files.uuid4",
        lambda: next(ids),
    )
    aliases: dict[str, str] = {}
    conflicts: set[str] = set()

    await _add_generated_file_path_aliases(
        aliases,
        conflicts,
        [
            _generated_result(
                GeneratedFile(
                    display_path="truncated.png",
                    media_type="image/png",
                    suffix=".png",
                    bytes=1,
                    truncated=True,
                )
            ),
            _generated_result(
                GeneratedFile(
                    display_path="missing-content.png",
                    media_type="image/png",
                    suffix=".png",
                    bytes=1,
                )
            ),
            _generated_result(
                GeneratedFile(
                    display_path=".",
                    media_type="image/png",
                    suffix=".png",
                    bytes=1,
                    content_base64=b64encode(b"dot").decode("ascii"),
                )
            ),
            _generated_result(
                GeneratedFile(
                    display_path="bad.png",
                    media_type="image/png",
                    suffix=".png",
                    bytes=1,
                    content_base64="not base64!",
                )
            ),
            _generated_result(
                GeneratedFile(
                    display_path="valid.png",
                    media_type="image/png",
                    suffix=".png",
                    bytes=1,
                    content_base64=b64encode(b"valid").decode("ascii"),
                )
            ),
        ],
        workspace_root=tmp_path,
        materialized_root=tmp_path / "generated",
        effective_cwd=tmp_path,
    )

    await _add_generated_file_path_aliases(
        aliases,
        conflicts,
        [
            _generated_result(
                GeneratedFile(
                    display_path="outside.png",
                    media_type="image/png",
                    suffix=".png",
                    bytes=1,
                    content_base64=b64encode(b"outside").decode("ascii"),
                )
            )
        ],
        workspace_root=tmp_path,
        materialized_root=tmp_path / "generated",
        effective_cwd=tmp_path / "nested",
    )

    assert aliases == {"valid.png": "generated/valid/valid.png"}
    assert conflicts == set()


def test_generated_file_helpers_skip_non_execution_results() -> None:
    plain_call = ToolCall(id="call-1", name="shell.cat", arguments={})
    plain_result = ToolCallResult(
        id="result-1",
        call=plain_call,
        name="shell.cat",
        arguments={},
        result="plain",
    )
    malformed_result = ToolCallResult(
        id="result-2",
        call=plain_call,
        name="shell.cat",
        arguments={},
        result=SimpleNamespace(execution_result="not-result"),
    )

    assert list(_iter_generated_files([plain_call])) == []
    assert list(_iter_generated_files([plain_result])) == []
    assert _execution_result(malformed_result.result) is None
    assert (
        _generated_file_filename(
            GeneratedFile(
                display_path="dir/..",
                media_type="image/png",
                suffix=".png",
                bytes=1,
                content_base64=b64encode(b"x").decode("ascii"),
            )
        )
        is None
    )
    assert _decode_base64_payload(" ") is None
    assert _decode_base64_payload("not base64!") is None


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
    assert _workspace_relative_path(
        Path("/workspace/a"), Path("/workspace")
    ) == (Path("a"))
    assert (
        _workspace_relative_path(Path("/other/a"), Path("/workspace")) is None
    )
    assert _path_suffix_aliases(Path("a/b/c.pdf")) == (
        "a/b/c.pdf",
        "b/c.pdf",
        "c.pdf",
    )
    assert _input_file_path_alias_values(
        "c.pdf",
        Path("/workspace/a/b/c.pdf"),
        Path("/workspace"),
    ) == (
        "c.pdf",
        "./c.pdf",
        "/workspace/a/b/c.pdf",
        "a/b/c.pdf",
        "./a/b/c.pdf",
        "b/c.pdf",
        "./b/c.pdf",
    )


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


async def _shell_input_file_manifest(
    input_value: object,
    settings: ShellToolSettings,
) -> str | None:
    builder = getattr(shell_input_files, "shell_input_file_manifest", None)
    assert callable(
        builder
    ), "Expected shell_input_file_manifest helper to be implemented"

    manifest = builder(input_value, settings)
    if isawaitable(manifest):
        manifest = await manifest

    if manifest is None:
        return None
    if isinstance(manifest, MessageContentText):
        return manifest.text
    assert isinstance(manifest, str)
    return manifest


def _generated_result(generated_file: GeneratedFile) -> ToolCallResult:
    execution_result = ExecutionResult(
        backend="local",
        tool_name="shell.pdftoppm",
        command="pdftoppm",
        argv=("pdftoppm",),
        display_argv=("pdftoppm",),
        cwd=".",
        display_cwd=".",
        status=ShellExecutionStatus.COMPLETED,
        exit_code=0,
        stdout="",
        stderr="",
        stdout_media_type="application/json",
        output_kind=ShellOutputKind.GENERATED_FILES,
        generated_files=(generated_file,),
    )
    return ToolCallResult(
        id=f"result-{generated_file.display_path}",
        call=ToolCall(
            id=f"call-{generated_file.display_path}",
            name="shell.pdftoppm",
            arguments={},
        ),
        name="shell.pdftoppm",
        arguments={},
        result=ShellFormattedResult("formatted", execution_result),
    )
