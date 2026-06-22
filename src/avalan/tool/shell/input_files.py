from ...entities import (
    Input,
    Message,
    MessageContentFile,
    ToolCall,
    ToolCallContext,
    ToolFilter,
    ToolValue,
)
from .settings import ShellToolSettings

from collections.abc import Iterator, Sequence
from dataclasses import replace
from os.path import relpath
from pathlib import Path
from typing import cast


def shell_input_file_filter(settings: ShellToolSettings) -> ToolFilter:
    """Return a shell filter resolving attached input file names."""
    assert isinstance(settings, ShellToolSettings)

    def filter_call(
        call: ToolCall, context: ToolCallContext
    ) -> tuple[ToolCall, ToolCallContext] | None:
        return _rewrite_shell_input_file_paths(call, context, settings)

    return ToolFilter(func=filter_call, namespace="shell")


def _rewrite_shell_input_file_paths(
    call: ToolCall,
    context: ToolCallContext,
    settings: ShellToolSettings,
) -> tuple[ToolCall, ToolCallContext] | None:
    arguments = call.arguments
    if not isinstance(arguments, dict):
        return None
    aliases = _input_file_path_aliases(
        context.input,
        settings,
        request_cwd=arguments.get("cwd"),
    )
    if not aliases:
        return None

    rewritten = dict(arguments)
    changed = False
    path_value, path_changed = _rewrite_path_argument(
        rewritten.get("path"),
        aliases,
    )
    if path_changed:
        rewritten["path"] = path_value
        changed = True

    paths_value, paths_changed = _rewrite_paths_argument(
        rewritten.get("paths"),
        aliases,
    )
    if paths_changed:
        rewritten["paths"] = paths_value
        changed = True

    if not changed:
        return None
    return (
        replace(call, arguments=cast(dict[str, ToolValue], rewritten)),
        context,
    )


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


def _input_file_path_aliases(
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
    aliases: dict[str, str] = {}
    conflicts: set[str] = set()
    for file_content in _iter_input_file_content(input_value):
        filename = file_content.file.get("filename")
        local_path = file_content.file.get("local_path")
        if not isinstance(filename, str) or not filename:
            continue
        if not isinstance(local_path, str) or not local_path:
            continue
        source_path = Path(local_path).resolve()
        if not _is_relative_to(source_path, workspace_root):
            continue
        relative_path = _cwd_relative_file_path(effective_cwd, source_path)
        if relative_path is None:
            continue
        for alias in (filename, f"./{filename}", str(source_path)):
            _add_alias(
                aliases,
                conflicts,
                _path_alias(alias),
                relative_path,
            )
    return aliases


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
    content = message.content
    if isinstance(content, MessageContentFile):
        yield content
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, MessageContentFile):
            yield item


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
