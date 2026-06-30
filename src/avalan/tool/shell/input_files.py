from ...entities import (
    Input,
    Message,
    MessageContentFile,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolFilter,
    ToolValue,
)
from ..input_files import message_file_content
from .entities import (
    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY,
    ExecutionResult,
    GeneratedFile,
)
from .filesystem import inspect_path as _inspect_path
from .filesystem import make_directory as _make_directory
from .filesystem import write_bytes as _write_bytes
from .settings import ShellToolSettings

from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Iterator, Sequence
from dataclasses import replace
from json import dumps as json_dumps
from os.path import relpath
from pathlib import Path
from typing import cast
from uuid import uuid4


async def shell_input_file_manifest(
    input_value: Input | None,
    settings: ShellToolSettings,
) -> str | None:
    """Return model-visible shell paths for attached input files."""
    assert isinstance(settings, ShellToolSettings)
    if not settings.input_file_manifest_enabled:
        return None

    paths = await _input_file_manifest_paths(input_value, settings)
    if not paths:
        return None

    lines = [
        settings.input_file_manifest_message,
        settings.input_file_manifest_path_message,
    ]
    lines.extend(f"- {json_dumps(path)}" for path in paths)
    return "\n".join(lines)


def shell_input_file_filter(settings: ShellToolSettings) -> ToolFilter:
    """Return a shell filter resolving attached input file names."""
    assert isinstance(settings, ShellToolSettings)

    async def filter_call(
        call: ToolCall, context: ToolCallContext
    ) -> tuple[ToolCall, ToolCallContext] | None:
        return await _rewrite_shell_input_file_paths(call, context, settings)

    return ToolFilter(func=filter_call, namespace="shell")


async def _rewrite_shell_input_file_paths(
    call: ToolCall,
    context: ToolCallContext,
    settings: ShellToolSettings,
) -> tuple[ToolCall, ToolCallContext] | None:
    arguments = call.arguments
    if not isinstance(arguments, dict):
        return None
    aliases = await _shell_file_path_aliases(
        context,
        settings,
        request_cwd=arguments.get("cwd"),
    )
    if not aliases:
        return None

    rewritten: dict[str, ToolValue] = dict(arguments)
    changed = False
    path_value, path_changed = _rewrite_path_argument(
        rewritten.get("path"),
        aliases,
    )
    if path_changed:
        rewritten["path"] = cast(ToolValue, path_value)
        changed = True

    paths_value, paths_changed = _rewrite_paths_argument(
        rewritten.get("paths"),
        aliases,
    )
    if paths_changed:
        rewritten["paths"] = cast(ToolValue, paths_value)
        changed = True

    if not changed:
        return None
    return (replace(call, arguments=rewritten), context)


def _rewrite_path_argument(
    value: object,
    aliases: dict[str, str],
) -> tuple[object, bool]:
    if not isinstance(value, str):
        return value, False
    replacement = aliases.get(_path_alias(value))
    if replacement is None or replacement == value:
        return value, False
    return replacement, True


def _rewrite_paths_argument(
    value: object,
    aliases: dict[str, str],
) -> tuple[object, bool]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return value, False

    rewritten: list[object] = []
    changed = False
    for item in value:
        next_item, item_changed = _rewrite_path_argument(item, aliases)
        rewritten.append(next_item)
        changed = changed or item_changed

    if not changed:
        return value, False
    if isinstance(value, tuple):
        return tuple(rewritten), True
    return rewritten, True


async def _input_file_path_aliases(
    input_value: Input | None,
    settings: ShellToolSettings,
    *,
    request_cwd: object | None = None,
) -> dict[str, str]:
    workspace_root = Path(settings.workspace_root).resolve()
    effective_cwd = _effective_shell_cwd(
        workspace_root,
        settings,
        request_cwd,
    )
    if effective_cwd is None:
        return {}
    materialized_root = _materialized_input_files_root(
        workspace_root,
        settings,
    )
    aliases: dict[str, str] = {}
    conflicts: set[str] = set()
    await _add_input_file_path_aliases(
        aliases,
        conflicts,
        input_value,
        workspace_root=workspace_root,
        materialized_root=materialized_root,
        effective_cwd=effective_cwd,
    )
    return aliases


async def _input_file_manifest_paths(
    input_value: Input | None,
    settings: ShellToolSettings,
) -> tuple[str, ...]:
    aliases = await _input_file_path_aliases(input_value, settings)
    return tuple(dict.fromkeys(aliases.values()))


async def _shell_file_path_aliases(
    context: ToolCallContext,
    settings: ShellToolSettings,
    *,
    request_cwd: object | None = None,
) -> dict[str, str]:
    workspace_root = Path(settings.workspace_root).resolve()
    effective_cwd = _effective_shell_cwd(
        workspace_root,
        settings,
        request_cwd,
    )
    if effective_cwd is None:
        return {}
    materialized_root = _materialized_input_files_root(
        workspace_root,
        settings,
    )
    aliases: dict[str, str] = {}
    conflicts: set[str] = set()
    await _add_input_file_path_aliases(
        aliases,
        conflicts,
        context.input,
        workspace_root=workspace_root,
        materialized_root=materialized_root,
        effective_cwd=effective_cwd,
    )
    await _add_generated_file_path_aliases(
        aliases,
        conflicts,
        context.calls or [],
        workspace_root=workspace_root,
        materialized_root=materialized_root,
        effective_cwd=effective_cwd,
    )
    return aliases


async def _add_input_file_path_aliases(
    aliases: dict[str, str],
    conflicts: set[str],
    input_value: Input | None,
    *,
    workspace_root: Path,
    materialized_root: Path,
    effective_cwd: Path,
) -> None:
    for file_content in _iter_input_file_content(input_value):
        filename = _input_file_filename(file_content)
        if filename is None:
            continue
        source_path = await _input_file_path(
            file_content,
            workspace_root,
            materialized_root,
            filename,
        )
        if source_path is None:
            continue
        relative_path = _cwd_relative_file_path(effective_cwd, source_path)
        if relative_path is None:
            continue
        for alias in _input_file_path_alias_values(
            filename,
            source_path,
            workspace_root,
        ):
            _add_alias(
                aliases,
                conflicts,
                _path_alias(alias),
                relative_path,
            )


def _input_file_filename(
    file_content: MessageContentFile,
) -> str | None:
    filename = file_content.file.get("filename")
    if not isinstance(filename, str):
        return None
    filename = Path(filename.strip()).name
    if not filename or filename in {".", ".."}:
        return None
    return filename


async def _input_file_path(
    file_content: MessageContentFile,
    workspace_root: Path,
    materialized_root: Path,
    filename: str,
) -> Path | None:
    local_path = file_content.file.get("local_path")
    if isinstance(local_path, str) and local_path:
        source_path = Path(local_path).resolve()
        if _is_relative_to(source_path, workspace_root):
            return source_path
    return await _materialize_input_file(
        file_content,
        workspace_root,
        materialized_root,
        filename,
    )


def _input_file_path_alias_values(
    filename: str,
    source_path: Path,
    workspace_root: Path,
) -> tuple[str, ...]:
    aliases = [filename, f"./{filename}", str(source_path)]
    relative_path = _workspace_relative_path(source_path, workspace_root)
    if relative_path is not None:
        for suffix in _path_suffix_aliases(relative_path):
            aliases.extend((suffix, f"./{suffix}"))
    return tuple(dict.fromkeys(aliases))


def _workspace_relative_path(path: Path, workspace_root: Path) -> Path | None:
    try:
        return path.relative_to(workspace_root)
    except ValueError:
        return None


def _path_suffix_aliases(path: Path) -> tuple[str, ...]:
    parts = path.parts
    return tuple(
        Path(*parts[index:]).as_posix() for index in range(len(parts))
    )


async def _materialize_input_file(
    file_content: MessageContentFile,
    workspace_root: Path,
    materialized_root: Path,
    filename: str,
) -> Path | None:
    raw = _decode_input_file_data(file_content)
    if raw is None:
        return None
    return await _materialize_bytes(
        raw,
        workspace_root,
        materialized_root,
        filename,
    )


async def _materialize_bytes(
    raw: bytes,
    workspace_root: Path,
    materialized_root: Path,
    filename: str,
) -> Path:
    await _make_directory_tree(materialized_root, stop_at=workspace_root)
    target_dir = materialized_root / uuid4().hex
    await _make_directory(target_dir)
    target_path = target_dir / _safe_materialized_filename(filename)
    await _write_bytes(target_path, raw)
    return target_path.resolve()


async def _make_directory_tree(path: Path, *, stop_at: Path) -> None:
    if path == stop_at:
        return
    try:
        await _make_directory(path)
    except FileNotFoundError:
        if path.parent == path or not _is_relative_to(path.parent, stop_at):
            raise
        await _make_directory_tree(path.parent, stop_at=stop_at)
        try:
            await _make_directory(path)
        except FileExistsError:
            pass
    except FileExistsError:
        # The shared materialization root may already exist from prior or
        # concurrent requests.
        pass


def _decode_input_file_data(
    file_content: MessageContentFile,
) -> bytes | None:
    value = _input_file_data(file_content)
    if value is None:
        return None
    payload = value.strip()
    if payload.startswith("data:"):
        _prefix, separator, payload = payload.partition(",")
        if not separator:
            return None
        payload = payload.strip()
    try:
        return b64decode(payload, validate=True)
    except (BinasciiError, ValueError):
        return None


def _input_file_data(file_content: MessageContentFile) -> str | None:
    for key in ("file_data", "data", "base64"):
        value = file_content.file.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


async def _add_generated_file_path_aliases(
    aliases: dict[str, str],
    conflicts: set[str],
    calls: list[ToolCall],
    *,
    workspace_root: Path,
    materialized_root: Path,
    effective_cwd: Path,
) -> None:
    materialized_root = materialized_root.resolve()
    for execution_result, generated_file in _iter_generated_file_sources(
        calls
    ):
        if generated_file.truncated:
            continue
        filename = _generated_file_filename(generated_file)
        if filename is None:
            continue
        source_path = await _generated_file_source_path(
            generated_file,
            workspace_root=workspace_root,
            materialized_root=materialized_root,
            filename=filename,
        )
        if source_path is None:
            continue
        relative_path = _cwd_relative_file_path(effective_cwd, source_path)
        if relative_path is None:
            continue
        prefix_alias = _single_generated_file_prefix_alias(execution_result)
        alias_values = (
            generated_file.display_path,
            f"./{generated_file.display_path}",
            filename,
            f"./{filename}",
            *(
                ()
                if prefix_alias is None
                else (prefix_alias, f"./{prefix_alias}")
            ),
        )
        for alias in alias_values:
            _add_alias(
                aliases,
                conflicts,
                _path_alias(alias),
                relative_path,
            )


async def _generated_file_source_path(
    generated_file: GeneratedFile,
    *,
    workspace_root: Path,
    materialized_root: Path,
    filename: str,
) -> Path | None:
    materialized_path = await _generated_file_materialized_path(
        generated_file,
        materialized_root=materialized_root,
    )
    if materialized_path is not None:
        return materialized_path
    if generated_file.content_base64 is None:
        return None
    raw = _decode_base64_payload(generated_file.content_base64)
    if raw is None:
        return None
    return await _materialize_bytes(
        raw,
        workspace_root,
        materialized_root,
        filename,
    )


async def _generated_file_materialized_path(
    generated_file: GeneratedFile,
    *,
    materialized_root: Path,
) -> Path | None:
    value = generated_file.metadata.get(
        GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY
    )
    if not isinstance(value, str) or not value:
        return None
    source_path = Path(value).resolve()
    if not _is_relative_to(source_path, materialized_root):
        return None
    try:
        metadata = await _inspect_path(source_path)
    except OSError:
        return None
    if (
        metadata.is_symlink
        or not metadata.is_file
        or not _is_relative_to(metadata.resolved_path, materialized_root)
    ):
        return None
    return source_path


def _iter_generated_files(
    calls: list[ToolCall],
) -> Iterator[GeneratedFile]:
    for _execution_result, generated_file in _iter_generated_file_sources(
        calls
    ):
        yield generated_file


def _iter_generated_file_sources(
    calls: list[ToolCall],
) -> Iterator[tuple[ExecutionResult, GeneratedFile]]:
    for call in calls:
        if not isinstance(call, ToolCallResult):
            continue
        execution_result = _execution_result(call.result)
        if execution_result is None:
            continue
        for generated_file in execution_result.generated_files:
            yield execution_result, generated_file


def _single_generated_file_prefix_alias(
    execution_result: ExecutionResult,
) -> str | None:
    if len(execution_result.generated_files) != 1:
        return None
    value = execution_result.metadata.get("generated_output_display_prefix")
    if not isinstance(value, str):
        return None
    alias = value.strip()
    if not alias or Path(alias).name != alias or alias in {".", ".."}:
        return None
    return alias


def _execution_result(value: object) -> ExecutionResult | None:
    execution_result = getattr(value, "execution_result", None)
    if not isinstance(execution_result, ExecutionResult):
        return None
    return execution_result


def _generated_file_filename(generated_file: GeneratedFile) -> str | None:
    filename = Path(generated_file.display_path.strip()).name
    if not filename or filename in {".", ".."}:
        return None
    return filename


def _decode_base64_payload(value: str) -> bytes | None:
    payload = value.strip()
    if not payload:
        return None
    try:
        return b64decode(payload, validate=True)
    except (BinasciiError, ValueError):
        return None


def _materialized_input_files_root(
    workspace_root: Path,
    settings: ShellToolSettings,
) -> Path:
    return workspace_root / settings.materialized_input_files_dir


def _safe_materialized_filename(filename: str) -> str:
    safe = filename.lstrip(".")
    return safe or "input"


def _effective_shell_cwd(
    workspace_root: Path,
    settings: ShellToolSettings,
    request_cwd: object | None,
) -> Path | None:
    cwd_value = request_cwd if request_cwd is not None else settings.cwd
    if not isinstance(cwd_value, str) or not cwd_value.strip():
        return None
    cwd_path = Path(cwd_value)
    if _path_has_part(cwd_path, ".."):
        return None
    if cwd_path.is_absolute() and not settings.allow_absolute_paths:
        return None
    if not cwd_path.is_absolute():
        cwd_path = workspace_root / cwd_path
    cwd_path = cwd_path.resolve()
    if not _is_relative_to(cwd_path, workspace_root):
        return None
    return cwd_path


def _cwd_relative_file_path(cwd: Path, source_path: Path) -> str | None:
    relative_path = Path(relpath(source_path, cwd))
    if _path_has_part(relative_path, ".."):
        return None
    return relative_path.as_posix()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_has_part(path: Path, part: str) -> bool:
    return any(path_part == part for path_part in path.parts)


def _iter_input_file_content(
    input_value: Input | None,
) -> Iterator[MessageContentFile]:
    if isinstance(input_value, Message):
        yield from _iter_message_file_content(input_value)
        return
    if not isinstance(input_value, Sequence) or isinstance(input_value, str):
        return
    for item in input_value:
        if isinstance(item, Message):
            yield from _iter_message_file_content(item)


def _iter_message_file_content(
    message: Message,
) -> Iterator[MessageContentFile]:
    yield from message_file_content(message)


def _add_alias(
    aliases: dict[str, str],
    conflicts: set[str],
    alias: str,
    target: str,
) -> None:
    if not alias or alias in conflicts:
        return
    existing = aliases.get(alias)
    if existing is None:
        aliases[alias] = target
        return
    if existing == target:
        return
    aliases.pop(alias, None)
    conflicts.add(alias)


def _path_alias(value: str) -> str:
    while value.startswith("./"):
        value = value[2:]
    return value
