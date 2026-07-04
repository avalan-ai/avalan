from ...filesystem import which_executable as _which_executable
from ...types import assert_non_empty_string as _assert_non_empty_string
from .commands.helpers import path_matches_sensitive_denylist
from .entities import (
    ExecutionSpec,
    ShellExecutionModeValue,
    ShellExecutionStatus,
    ShellOutputKind,
)
from .filesystem import list_directory as _list_directory
from .filesystem import read_bytes as _read_bytes
from .git import (
    ShellGitCapability,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitExecutionErrorCode,
    ShellGitPolicyDenied,
    shell_git_capability_for_request,
)
from .policy import ExecutionPolicy
from .settings import ShellGitToolSettings, ShellToolSettings

from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from re import compile as compile_pattern
from typing import cast, final

GitExecutableLookup = Callable[[tuple[str, ...]], Awaitable[str | None]]

_SAFE_GIT_ENVIRONMENT = {
    "LC_ALL": "C",
    "LANG": "C",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "CLICOLOR": "0",
    "CLICOLOR_FORCE": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": "/nonexistent",
    "GIT_CONFIG_SYSTEM": "/nonexistent",
    "GIT_PAGER": "/nonexistent",
    "PAGER": "/nonexistent",
    "GIT_EDITOR": "/nonexistent",
    "GIT_SEQUENCE_EDITOR": "/nonexistent",
    "GIT_MERGE_AUTOEDIT": "no",
    "VISUAL": "/nonexistent",
    "EDITOR": "/nonexistent",
    "GIT_ASKPASS": "/nonexistent",
    "SSH_ASKPASS": "/nonexistent",
    "GIT_SSH": "/nonexistent",
    "GIT_SSH_VARIANT": "ssh",
    "GIT_EXTERNAL_DIFF": "/nonexistent",
    "HOME": "/nonexistent",
    "XDG_CONFIG_HOME": "/nonexistent",
    "XDG_CACHE_HOME": "/nonexistent",
}
_UNSAFE_GLOBAL_OPTION_VALUES = (
    "-c",
    "--config",
    "--config-env",
    "--git-dir",
    "--work-tree",
    "-C",
    "--exec-path",
    "--namespace",
    "--paginate",
    "--literal-pathspecs",
    "--glob-pathspecs",
    "--noglob-pathspecs",
    "--icase-pathspecs",
)
_UNSAFE_GLOBAL_OPTIONS = frozenset(_UNSAFE_GLOBAL_OPTION_VALUES)
_UNSAFE_OPTION_NAME_VALUES = (
    "alias",
    "askpass",
    "config",
    "config-env",
    "credential",
    "editor",
    "exec-path",
    "external-diff",
    "filter",
    "fsmonitor",
    "git-dir",
    "gpg-sign",
    "gpgsign",
    "hook",
    "hooks-path",
    "namespace",
    "pager",
    "paginate",
    "prompt",
    "recurse-submodules",
    "sign",
    "signing",
    "ssh",
    "ssh-command",
    "submodule",
    "submodules",
    "textconv",
    "work-tree",
)
_UNSAFE_OPTION_NAMES = frozenset(_UNSAFE_OPTION_NAME_VALUES)
_UNSAFE_VALUE_MARKERS = (
    "alias.",
    "askpass",
    "credential.",
    "core.editor",
    "core.fsmonitor",
    "core.hookspath",
    "core.pager",
    "diff.external",
    "filter.",
    "gpg.",
    "pager.",
    "sshcommand",
    "textconv",
    "!",
)
_REVISION_PATTERN = compile_pattern(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")
_HEX_REVISION_PATTERN = compile_pattern(r"^[0-9a-fA-F]{7,64}$")
_ANCESTRY_REVISION_PATTERN = compile_pattern(
    r"^(?:HEAD|[0-9a-fA-F]{7,64}|[A-Za-z0-9][A-Za-z0-9._\-]*)"
    r"(?:(?:~[0-9]+)|(?:\^[0-9]*))*$"
)
_STASH_REF_PATTERN = compile_pattern(r"^stash@\{(?P<index>[0-9]+)\}$")
_REVISION_BASE_PATTERN = compile_pattern(r"^(?P<base>[^~^]+)")
_REMOTE_URL_PATTERN = compile_pattern(
    r"^(?P<protocol>[A-Za-z][A-Za-z0-9+.-]*)://"
    r"(?:(?P<userinfo>[^/@\s]+)@)?(?P<host>[^/@:\s]+)"
    r"(?::[0-9]+)?(?:/.*)?$"
)
_CONTROL_CHARACTERS = tuple(chr(value) for value in (*range(0, 32), 127))
_DANGEROUS_CONFIG_MARKERS = (
    "fsmonitor",
    "pager",
    "editor",
    "credential",
    "askpass",
    "sshcommand",
    "hookspath",
    "diff.external",
    "[include",
    "include.path",
    "includeif",
    "[filter",
    "[diff",
    "textconv",
    "gpg",
    "gpgsign",
    "showsignature",
    "ignorerevsfile",
    "[merge",
    "merge.",
    "signing",
    "worktree",
)
_DANGEROUS_ATTRIBUTES_MARKERS = (
    "filter",
    "merge=",
    "textconv",
    "diff=",
)
_DESCRIBE_OPTION_KEYS = frozenset(("target", "mode", "max_candidates"))
_DIFF_OPTION_KEYS = frozenset(("mode", "base_revision", "head_revision"))
_RESET_HISTORY_MODES = ("soft", "mixed", "hard")
_RESET_MODES = ("paths", *_RESET_HISTORY_MODES)
_ALLOWED_GIT_OPTION_KEYS = {
    ShellGitCommandName.STATUS: frozenset({"mode", "include_branch"}),
    ShellGitCommandName.REV_PARSE: frozenset({"fact"}),
    ShellGitCommandName.BRANCH: frozenset({"mode", "contains"}),
    ShellGitCommandName.TAG: frozenset({"mode", "name", "max_count"}),
    ShellGitCommandName.DESCRIBE: _DESCRIBE_OPTION_KEYS,
    ShellGitCommandName.LS_FILES: frozenset({"mode"}),
    ShellGitCommandName.LOG: frozenset({"max_count", "revision", "format"}),
    ShellGitCommandName.DIFF: _DIFF_OPTION_KEYS,
    ShellGitCommandName.SHOW: frozenset({"mode", "revision"}),
    ShellGitCommandName.BLAME: frozenset({"start_line", "end_line"}),
    ShellGitCommandName.GREP: frozenset({"pattern", "case", "max_matches"}),
    ShellGitCommandName.STASH_LIST: frozenset({"max_count"}),
    ShellGitCommandName.STASH_SHOW: frozenset({"stash", "mode"}),
    ShellGitCommandName.ADD: frozenset({"mode"}),
    ShellGitCommandName.RESTORE: frozenset(
        {
            "source_revision",
            "staged",
            "worktree",
        }
    ),
    ShellGitCommandName.CHECKOUT: frozenset({"target"}),
    ShellGitCommandName.SWITCH: frozenset({"branch"}),
    ShellGitCommandName.RESET: frozenset(
        {"mode", "revision", "confirm_revision", "confirm_hard"},
    ),
    ShellGitCommandName.RM: frozenset({"cached"}),
    ShellGitCommandName.MV: frozenset({"source", "destination"}),
    ShellGitCommandName.STASH_PUSH: frozenset(
        {
            "message",
            "include_untracked",
        }
    ),
    ShellGitCommandName.STASH_APPLY: frozenset({"stash"}),
    ShellGitCommandName.COMMIT: frozenset({"message"}),
    ShellGitCommandName.BRANCH_CREATE: frozenset(
        {"name", "start_point"},
    ),
    ShellGitCommandName.BRANCH_DELETE: frozenset(
        {"name", "confirm_name"},
    ),
    ShellGitCommandName.BRANCH_RENAME: frozenset(
        {"old_name", "new_name", "confirm_old_name"},
    ),
    ShellGitCommandName.TAG_CREATE: frozenset(
        {"name", "target", "message"},
    ),
    ShellGitCommandName.TAG_DELETE: frozenset({"name", "confirm_name"}),
    ShellGitCommandName.MERGE: frozenset(
        {"revision", "mode", "confirm_revision"},
    ),
    ShellGitCommandName.REBASE: frozenset(
        {"upstream", "branch", "confirm_upstream"},
    ),
    ShellGitCommandName.CHERRY_PICK: frozenset(
        {"revision", "confirm_revision"},
    ),
    ShellGitCommandName.REVERT: frozenset({"revision", "confirm_revision"}),
    ShellGitCommandName.CLEAN: frozenset(
        {"dry_run", "confirm_paths"},
    ),
    ShellGitCommandName.STASH_POP: frozenset(
        {"stash", "index", "confirm_stash"},
    ),
    ShellGitCommandName.STASH_DROP: frozenset({"stash", "confirm_stash"}),
}


@final
class GitExecutionPolicy:
    _executable_lookup: GitExecutableLookup
    _shell_policy: ExecutionPolicy
    _settings: ShellToolSettings

    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        *,
        executable_lookup: GitExecutableLookup | None = None,
    ) -> None:
        self._settings = settings or ShellToolSettings()
        self._shell_policy = ExecutionPolicy(settings=self._settings)
        self._executable_lookup = executable_lookup or _lookup_git_executable

    async def normalize(
        self,
        request: ShellGitCommandRequest,
    ) -> ExecutionSpec:
        assert isinstance(
            request,
            ShellGitCommandRequest,
        ), "request must be a shell Git command request"
        git_settings = self._git_settings()
        _validate_tool_name(request)
        _validate_authorization(request, git_settings)
        workspace_root = _resolve_workspace_root(git_settings.workspace_root)
        effective_cwd = _resolve_effective_cwd(
            request.cwd or git_settings.cwd,
            workspace_root=workspace_root,
        )
        repo_root, git_dir = _discover_repository(
            effective_cwd,
            workspace_root=workspace_root,
        )
        await _validate_repository_form(
            repo_root,
            git_dir,
            workspace_root=workspace_root,
            settings=git_settings,
        )
        pathspecs = _validated_pathspecs(
            request.pathspecs,
            repo_root=repo_root,
            settings=git_settings,
            allow_hidden=self._settings.allow_hidden,
        )
        _validate_git_option_keys(request)
        _validate_revisions(request, git_settings)
        _validate_git_option_values(request.options)
        await _validate_history_ref_state(
            request,
            git_dir=git_dir,
            settings=git_settings,
        )
        _validate_content_pathspec_scope(request, pathspecs, repo_root)
        _validate_mutation_pathspec_scope(request, pathspecs, repo_root)
        argv = _argv_for_request(request, pathspecs, git_settings)
        _validate_argv_budgets(argv, self._settings)
        timeout_seconds = _bounded_float(
            request.timeout_seconds,
            default_value=git_settings.default_timeout_seconds,
            max_value=git_settings.max_timeout_seconds,
        )
        max_stdout_bytes = _bounded_stdout_bytes(request, git_settings)
        max_stderr_bytes = _bounded_int(
            request.max_stderr_bytes,
            git_settings.max_stderr_bytes,
        )
        capability = shell_git_capability_for_request(request)
        display_argv = _redacted_argv(
            argv,
            git_settings,
            redact_messages=capability is ShellGitCapability.HISTORY,
        )
        metadata = _metadata(
            request,
            workspace_root=workspace_root,
            effective_cwd=effective_cwd,
            repo_root=repo_root,
            display_argv=display_argv,
            settings=git_settings,
        )
        executable = await self._executable_lookup(
            tuple(self._settings.executable_search_paths)
        )
        return self._shell_policy.create_execution_spec(
            backend=cast(
                ShellExecutionModeValue,
                self._settings.execution_mode,
            ),
            tool_name=request.tool_name,
            command=f"git.{request.command.value}",
            executable=executable,
            argv=argv,
            display_argv=display_argv,
            cwd=str(repo_root),
            display_cwd=_display_path(workspace_root, effective_cwd),
            env=_safe_git_environment(git_settings),
            stdin=None,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            resource_class="standard",
            output_plan=None,
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            metadata=metadata,
        )

    def _git_settings(self) -> ShellGitToolSettings:
        git_settings = self._settings.git
        assert isinstance(
            git_settings,
            ShellGitToolSettings,
        ), "git must be shell Git tool settings"
        return git_settings


async def _lookup_git_executable(
    search_paths: tuple[str, ...],
) -> str | None:
    if not search_paths:
        return None
    return await _which_executable("git", search_paths)


def _validate_tool_name(request: ShellGitCommandRequest) -> None:
    expected = f"shell.git_{request.command.value.replace('-', '_')}"
    if request.tool_name != expected:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.COMMAND_DISABLED,
            "tool name does not match shell Git command",
        )


def _validate_authorization(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> None:
    if request.command is ShellGitCommandName.RESET:
        mode = _reset_mode_option(request.options)
        if mode != "paths" and request.pathspecs:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "git reset ref-moving modes do not accept pathspecs",
            )
    capability = shell_git_capability_for_request(request)
    if request.command.value not in settings.allowed_commands:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.COMMAND_DISABLED,
            "shell Git command is disabled",
        )
    if capability.value not in settings.capabilities:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
            f"shell Git command requires capability {capability.value}; "
            f"configured capabilities: {', '.join(settings.capabilities)}",
        )
    if capability is ShellGitCapability.REMOTE:
        _validate_remote_policy(request, settings)


def _validate_remote_policy(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> None:
    if request.command is ShellGitCommandName.SUBMODULE_UPDATE:
        if not settings.allow_submodule_update:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.SUBMODULE_DENIED,
                "submodule update is disabled",
            )
        if bool(request.options.get("recursive")):
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.SUBMODULE_DENIED,
                "recursive submodule update is disabled",
            )
    if not settings.allowed_remote_hosts:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
            "remote hosts are not allowlisted",
        )
    url = request.options.get("url")
    if isinstance(url, str):
        _validate_remote_url(url, settings)


def _validate_remote_url(url: str, settings: ShellGitToolSettings) -> None:
    match = _REMOTE_URL_PATTERN.match(url)
    if not match:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "remote URL protocol is unsupported",
        )
    protocol = match.group("protocol").lower()
    host = match.group("host").lower()
    if protocol not in settings.allowed_remote_protocols:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "remote URL protocol is not allowlisted",
        )
    if host not in settings.allowed_remote_hosts:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
            "remote URL host is not allowlisted",
        )
    if not settings.allow_remote_credentials and "@" in url.split("://", 1)[1]:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
            "remote URL credentials are disabled",
        )


def _resolve_workspace_root(value: str) -> Path:
    _assert_non_empty_string(value, "git.workspace_root")
    root = Path(value).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
            "git workspace root is unavailable",
        )
    return root


def _resolve_effective_cwd(value: str, *, workspace_root: Path) -> Path:
    _assert_non_empty_string(value, "git.cwd")
    path = Path(value)
    if _contains_unsafe_path_text(value):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
            "git cwd is unsafe",
        )
    candidate = path if path.is_absolute() else workspace_root / path
    resolved = candidate.resolve()
    if not _is_relative_to(resolved, workspace_root):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
            "git cwd escapes workspace root",
        )
    if not resolved.exists() or not resolved.is_dir():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REPO_NOT_FOUND,
            "git cwd is not a repository",
        )
    return resolved


def _discover_repository(
    cwd: Path,
    *,
    workspace_root: Path,
) -> tuple[Path, Path]:
    current = cwd
    while True:
        if _looks_like_bare_repository(current):
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.BARE_REPO_DENIED,
                "bare Git repositories are disabled",
            )
        git_marker = current / ".git"
        if git_marker.exists() or git_marker.is_symlink():
            if git_marker.is_file():
                raise ShellGitPolicyDenied(
                    ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
                    "Git repository indirection is disabled",
                )
            if git_marker.is_symlink() or not git_marker.is_dir():
                raise ShellGitPolicyDenied(
                    ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
                    "Git repository metadata is unsafe",
                )
            repo_root = current.resolve()
            git_dir = git_marker.resolve()
            return repo_root, git_dir
        if current == workspace_root:
            break
        current = current.parent
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.REPO_NOT_FOUND,
        "git repository was not found inside workspace root",
    )


async def _validate_repository_form(
    repo_root: Path,
    git_dir: Path,
    *,
    workspace_root: Path,
    settings: ShellGitToolSettings,
) -> None:
    config = await _read_text(git_dir / "config")
    _validate_git_config(config)
    if _config_declares_bare(config) and not settings.allow_bare_repositories:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.BARE_REPO_DENIED,
            "bare Git repositories are disabled",
        )
    common_dir = git_dir / "commondir"
    if common_dir.exists():
        if not settings.allow_linked_worktrees:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
                "linked worktrees are disabled",
            )
        common_path = _resolve_gitdir_reference(
            (await _read_text(common_dir)).strip(),
            base=git_dir,
        )
        if not _is_relative_to(common_path, workspace_root):
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
                "Git common directory escapes workspace root",
            )
    alternates = git_dir / "objects" / "info" / "alternates"
    if alternates.exists():
        await _validate_alternates(
            alternates,
            workspace_root=workspace_root,
            settings=settings,
        )
    await _validate_attributes(repo_root / ".gitattributes")
    await _validate_attributes(git_dir / "info" / "attributes")
    await _validate_hooks(git_dir / "hooks")


async def _validate_alternates(
    path: Path,
    *,
    workspace_root: Path,
    settings: ShellGitToolSettings,
) -> None:
    if not settings.allow_alternates:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.ALTERNATE_DENIED,
            "Git alternates are disabled",
        )
    base = path.parent
    for line in (await _read_text(path)).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        alternate = _resolve_gitdir_reference(stripped, base=base)
        if not _is_relative_to(alternate, workspace_root):
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.ALTERNATE_DENIED,
                "Git alternate object directory escapes workspace root",
            )


def _validate_git_config(config: str) -> None:
    lowered = config.lower()
    if any(marker in lowered for marker in _DANGEROUS_CONFIG_MARKERS):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            "Git repository configuration is unsafe",
        )


async def _validate_attributes(path: Path) -> None:
    if not path.exists():
        return
    lowered = (await _read_text(path)).lower()
    if any(marker in lowered for marker in _DANGEROUS_ATTRIBUTES_MARKERS):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            "Git attributes can trigger external processing",
        )


async def _validate_hooks(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            "Git hooks path is unsafe",
        )
    for child in await _list_directory(path):
        if child.name.endswith(".sample"):
            continue
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            "Git hooks can trigger external processing",
        )


def _validated_pathspecs(
    values: tuple[str, ...],
    *,
    repo_root: Path,
    settings: ShellGitToolSettings,
    allow_hidden: bool,
) -> tuple[str, ...]:
    if len(values) > settings.max_pathspecs:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "too many Git pathspecs",
        )
    pathspecs: list[str] = []
    total_bytes = 0
    for value in values:
        _validate_pathspec(
            value,
            repo_root=repo_root,
            allow_hidden=allow_hidden,
        )
        total_bytes += len(value.encode("utf-8"))
        if total_bytes > settings.max_pathspec_bytes:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                "Git pathspecs exceed byte limit",
            )
        pathspecs.append(value)
    return tuple(pathspecs)


def _validate_pathspec(
    value: str,
    *,
    repo_root: Path,
    allow_hidden: bool,
) -> None:
    if _contains_unsafe_path_text(value):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git pathspec is unsafe",
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git pathspec must be repo-relative",
        )
    if (
        value.startswith("-")
        or value.startswith(":")
        or ":" in value
        or "\\" in value
        or any(character in value for character in ("*", "?", "[", "]"))
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git pathspec form is unsupported",
        )
    if any(part == ".git" for part in path.parts):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git metadata pathspecs are unsupported",
        )
    display_path = path.as_posix()
    if not allow_hidden and _has_hidden_pathspec_component(display_path):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "hidden Git pathspecs are unsupported",
        )
    if path_matches_sensitive_denylist(display_path):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "sensitive Git pathspecs are unsupported",
        )
    resolved = (repo_root / Path(*path.parts)).resolve()
    if not _is_relative_to(resolved, repo_root):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git pathspec escapes repository root",
        )


def _has_hidden_pathspec_component(display_path: str) -> bool:
    return any(
        part not in ("", ".", "..") and part.startswith(".")
        for part in PurePosixPath(display_path).parts
    )


def _validate_git_option_keys(request: ShellGitCommandRequest) -> None:
    allowed_options = _ALLOWED_GIT_OPTION_KEYS.get(request.command)
    for name in request.options:
        _validate_git_option_name(name)
        if allowed_options is not None and name not in allowed_options:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "Git option is unsupported for this command",
            )


def _validate_git_option_values(options: Mapping[str, object]) -> None:
    for value in options.values():
        _validate_git_option_value(value)


def _validate_git_option_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "Git option name is unsupported",
        )
    normalized = name.lower().replace("_", "-")
    if normalized in _UNSAFE_OPTION_NAMES:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "Git option surface is unsupported",
        )


def _validate_git_option_value(value: object) -> None:
    if isinstance(value, str):
        _validate_git_option_string(value)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            _validate_git_option_value(item)


def _validate_git_option_string(value: str) -> None:
    lowered = value.lower()
    if (
        _is_unsafe_global_option(value)
        or any(marker in lowered for marker in _UNSAFE_VALUE_MARKERS)
        or _contains_control(value)
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "Git option surface is unsupported",
        )


def _validate_content_pathspec_scope(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    repo_root: Path,
) -> None:
    if not pathspecs or not _needs_file_scoped_content_pathspecs(request):
        return
    for value in pathspecs:
        path = PurePosixPath(value)
        if path.as_posix() == ".":
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                "Git content pathspec must name a file path",
            )
        resolved = (repo_root / Path(*path.parts)).resolve()
        if not resolved.exists() or not resolved.is_file():
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                "Git content pathspec must name an existing file",
            )


def _needs_file_scoped_content_pathspecs(
    request: ShellGitCommandRequest,
) -> bool:
    if request.command in (
        ShellGitCommandName.BLAME,
        ShellGitCommandName.DIFF,
        ShellGitCommandName.GREP,
        ShellGitCommandName.STASH_APPLY,
        ShellGitCommandName.STASH_SHOW,
    ):
        return True
    if request.command is ShellGitCommandName.SHOW:
        mode = request.options.get("mode", "summary")
        return mode in ("stat", "patch")
    return False


def _validate_mutation_pathspec_scope(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    repo_root: Path,
) -> None:
    if (
        request.command is ShellGitCommandName.RESET
        and _reset_mode_option(request.options) != "paths"
    ):
        _reject_pathspecs(pathspecs, "git reset ref-moving mode")
        return
    if request.command is ShellGitCommandName.MV:
        _validate_mv_pathspec_scope(request, pathspecs, repo_root)
        return
    if request.command not in (
        ShellGitCommandName.ADD,
        ShellGitCommandName.RESTORE,
        ShellGitCommandName.CHECKOUT,
        ShellGitCommandName.RESET,
        ShellGitCommandName.RM,
        ShellGitCommandName.STASH_PUSH,
        ShellGitCommandName.STASH_APPLY,
        ShellGitCommandName.CLEAN,
    ):
        return
    for value in pathspecs:
        _validate_file_scoped_mutation_pathspec(value, repo_root)


def _validate_file_scoped_mutation_pathspec(
    value: str,
    repo_root: Path,
) -> None:
    path = PurePosixPath(value)
    if path.as_posix() == ".":
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git mutation pathspec must name a file path",
        )
    resolved = _resolve_repo_path(repo_root, path)
    if resolved.exists() and resolved.is_dir():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git mutation pathspec must not name a directory",
        )


def _validate_mv_pathspec_scope(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    repo_root: Path,
) -> None:
    if len(pathspecs) != 2:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv requires source and destination paths",
        )
    source = _required_string_option(
        request.options,
        "source",
        message="git mv source must be a string",
    )
    destination = _required_string_option(
        request.options,
        "destination",
        message="git mv destination must be a string",
    )
    if pathspecs != (source, destination):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv pathspecs must match source and destination",
        )
    source_pathspec = PurePosixPath(source)
    destination_pathspec = PurePosixPath(destination)
    _validate_file_scoped_mutation_pathspec(source, repo_root)
    source_path = _resolve_repo_path(repo_root, source_pathspec)
    if not source_path.exists() or not source_path.is_file():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv source must name an existing file",
        )
    _validate_mv_destination_pathspec(destination_pathspec, repo_root)


def _validate_mv_destination_pathspec(
    pathspec: PurePosixPath,
    repo_root: Path,
) -> None:
    if pathspec.as_posix() == ".":
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv destination must name a file path",
        )
    destination = _resolve_repo_path(repo_root, pathspec)
    if destination.exists():
        if destination.is_dir():
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                "git mv destination must not name a directory",
            )
        return
    parent = _resolve_repo_path(repo_root, PurePosixPath(pathspec.parent))
    if not _is_relative_to(parent, repo_root):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv destination parent escapes repository root",
        )
    if parent.exists() and not parent.is_dir():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv destination parent must be a directory",
        )


def _is_unsafe_global_option(value: str) -> bool:
    return any(
        value == option or value.startswith(f"{option}=")
        for option in _UNSAFE_GLOBAL_OPTIONS
    )


def _validate_revisions(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> None:
    for name, value in request.options.items():
        if not _is_revision_option(request.command, name):
            continue
        if value is None:
            continue
        if not isinstance(value, str):
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REVISION_DENIED,
                "Git revision must be a string",
            )
        _validate_revision(value, settings)


async def _validate_history_ref_state(
    request: ShellGitCommandRequest,
    *,
    git_dir: Path,
    settings: ShellGitToolSettings,
) -> None:
    if (
        shell_git_capability_for_request(request)
        is not ShellGitCapability.HISTORY
    ):
        return
    heads, tags = await _local_ref_names(git_dir)
    match request.command:
        case ShellGitCommandName.BRANCH_CREATE:
            name = _required_string_option(
                request.options,
                "name",
                message="git branch create name must be a string",
            )
            _deny_existing_ref_name(name, heads, tags, "branch name")
            start_point = _optional_string_option(
                request.options,
                "start_point",
                message="git branch create start point must be a string",
            )
            if start_point is not None:
                await _validate_resolvable_revision(
                    start_point,
                    heads,
                    tags,
                    git_dir=git_dir,
                    settings=settings,
                )
            else:
                await _require_head_ref(git_dir, heads)
        case ShellGitCommandName.BRANCH_DELETE:
            name = _required_string_option(
                request.options,
                "name",
                message="git branch delete name must be a string",
            )
            _require_branch_ref(name, heads)
        case ShellGitCommandName.BRANCH_RENAME:
            old_name = _required_string_option(
                request.options,
                "old_name",
                message="git branch rename old_name must be a string",
            )
            new_name = _required_string_option(
                request.options,
                "new_name",
                message="git branch rename new_name must be a string",
            )
            _require_branch_ref(old_name, heads)
            _deny_existing_ref_name(new_name, heads, tags, "new branch name")
        case ShellGitCommandName.TAG_CREATE:
            name = _required_string_option(
                request.options,
                "name",
                message="git tag create name must be a string",
            )
            _deny_existing_ref_name(name, heads, tags, "tag name")
            target = _optional_string_option(
                request.options,
                "target",
                message="git tag create target must be a string",
            )
            if target is not None:
                await _validate_resolvable_revision(
                    target,
                    heads,
                    tags,
                    git_dir=git_dir,
                    settings=settings,
                )
            else:
                await _require_head_ref(git_dir, heads)
        case ShellGitCommandName.TAG_DELETE:
            name = _required_string_option(
                request.options,
                "name",
                message="git tag delete name must be a string",
            )
            _require_tag_ref(name, tags)
        case ShellGitCommandName.RESET:
            if _reset_mode_option(request.options) != "paths":
                await _validate_required_revision_option(
                    request,
                    "revision",
                    heads,
                    tags,
                    git_dir=git_dir,
                    settings=settings,
                )
        case ShellGitCommandName.MERGE:
            await _validate_required_revision_option(
                request,
                "revision",
                heads,
                tags,
                git_dir=git_dir,
                settings=settings,
            )
        case ShellGitCommandName.REBASE:
            await _validate_required_revision_option(
                request,
                "upstream",
                heads,
                tags,
                git_dir=git_dir,
                settings=settings,
            )
            branch = _optional_string_option(
                request.options,
                "branch",
                message="git rebase branch must be a string",
            )
            if branch is not None:
                _require_branch_ref(branch, heads)
        case ShellGitCommandName.CHERRY_PICK | ShellGitCommandName.REVERT:
            await _validate_required_revision_option(
                request,
                "revision",
                heads,
                tags,
                git_dir=git_dir,
                settings=settings,
            )
        case ShellGitCommandName.STASH_POP | ShellGitCommandName.STASH_DROP:
            stash = _required_string_option(
                request.options,
                "stash",
                message="git stash reference must be a string",
            )
            await _validate_existing_stash_ref(stash, git_dir, settings)


async def _local_ref_names(
    git_dir: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    heads = await _ref_names(git_dir, "heads")
    tags = await _ref_names(git_dir, "tags")
    packed = await _packed_ref_names(git_dir / "packed-refs")
    return (
        frozenset((*heads, *packed[0])),
        frozenset((*tags, *packed[1])),
    )


async def _ref_names(git_dir: Path, namespace: str) -> frozenset[str]:
    root = git_dir / "refs" / namespace
    if not root.exists():
        return frozenset()
    names: set[str] = set()
    await _collect_ref_names(root, root, names)
    return frozenset(names)


async def _collect_ref_names(
    root: Path,
    current: Path,
    names: set[str],
) -> None:
    if not current.exists():
        return
    if not current.is_dir():
        return
    for child in await _list_directory(current):
        if child.is_dir():
            await _collect_ref_names(root, child, names)
            continue
        if not child.is_file():
            continue
        names.add(child.relative_to(root).as_posix())


async def _packed_ref_names(
    path: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    heads: set[str] = set()
    tags: set[str] = set()
    for line in (await _read_text(path)).splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("^")
        ):
            continue
        parts = stripped.split(" ", maxsplit=1)
        if len(parts) != 2:
            continue
        ref = parts[1]
        if ref.startswith("refs/heads/"):
            heads.add(ref.removeprefix("refs/heads/"))
        elif ref.startswith("refs/tags/"):
            tags.add(ref.removeprefix("refs/tags/"))
    return frozenset(heads), frozenset(tags)


async def _validate_required_revision_option(
    request: ShellGitCommandRequest,
    name: str,
    heads: frozenset[str],
    tags: frozenset[str],
    *,
    git_dir: Path,
    settings: ShellGitToolSettings,
) -> None:
    revision = _required_string_option(
        request.options,
        name,
        message=f"git {name} must be a string",
    )
    await _validate_resolvable_revision(
        revision,
        heads,
        tags,
        git_dir=git_dir,
        settings=settings,
    )


async def _validate_resolvable_revision(
    value: str,
    heads: frozenset[str],
    tags: frozenset[str],
    *,
    git_dir: Path,
    settings: ShellGitToolSettings,
) -> None:
    _validate_revision(value, settings)
    base = _revision_base(value)
    if _HEX_REVISION_PATTERN.match(base):
        return
    if base == "HEAD":
        await _require_head_ref(git_dir, heads)
        return
    has_head = base in heads
    has_tag = base in tags
    if has_head and has_tag:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.AMBIGUOUS_REVISION,
            "Git revision is ambiguous between local refs",
        )
    if not has_head and not has_tag:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            "Git revision was not found",
        )


def _revision_base(value: str) -> str:
    match = _REVISION_BASE_PATTERN.match(value)
    return value if match is None else match.group("base")


async def _require_head_ref(
    git_dir: Path,
    heads: frozenset[str],
) -> None:
    head = (await _read_text(git_dir / "HEAD")).strip()
    if head.startswith("ref: refs/heads/"):
        name = head.removeprefix("ref: refs/heads/")
        if name in heads:
            return
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            "Git HEAD revision was not found",
        )
    if _HEX_REVISION_PATTERN.match(head):
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        "Git HEAD revision was not found",
    )


def _require_branch_ref(name: str, heads: frozenset[str]) -> None:
    if name in heads:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        "Git branch was not found",
    )


def _require_tag_ref(name: str, tags: frozenset[str]) -> None:
    if name in tags:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        "Git tag was not found",
    )


def _deny_existing_ref_name(
    name: str,
    heads: frozenset[str],
    tags: frozenset[str],
    field_name: str,
) -> None:
    if name not in heads and name not in tags:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.AMBIGUOUS_REVISION,
        f"Git {field_name} already exists",
    )


async def _validate_existing_stash_ref(
    value: str,
    git_dir: Path,
    settings: ShellGitToolSettings,
) -> None:
    _validate_stash_ref(value, settings)
    match = _STASH_REF_PATTERN.match(value)
    assert match is not None, "validated stash reference must match"
    index = int(match.group("index"))
    if index == 0 and (git_dir / "refs" / "stash").exists():
        return
    log = git_dir / "logs" / "refs" / "stash"
    if log.exists() and len((await _read_text(log)).splitlines()) > index:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        "Git stash reference was not found",
    )


def _is_revision_option(command: ShellGitCommandName, name: str) -> bool:
    if command in (ShellGitCommandName.CLONE,):
        return False
    return name in {
        "revision",
        "base_revision",
        "head_revision",
        "source_revision",
        "target",
        "start_point",
        "upstream",
        "contains",
        "branch",
    }


def _validate_revision(
    value: str,
    settings: ShellGitToolSettings,
) -> None:
    if (
        not value
        or _contains_control(value)
        or "\x00" in value
        or value.startswith("-")
        or ":" in value
        or "@" in value
        or "\\" in value
        or "{" in value
        or "}" in value
        or value.startswith("/")
        or "/" in value
        or ".." in value
        or len(value.encode("utf-8")) > settings.max_revision_bytes
        or (
            not _HEX_REVISION_PATTERN.match(value)
            and value != "HEAD"
            and not _REVISION_PATTERN.match(value)
            and not _ANCESTRY_REVISION_PATTERN.match(value)
        )
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_DENIED,
            "Git revision form is unsupported",
        )


def _argv_for_request(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    if request.command is ShellGitCommandName.STATUS:
        return _status_argv(request, pathspecs)
    if request.command is ShellGitCommandName.REV_PARSE:
        return _rev_parse_argv(request)
    if request.command is ShellGitCommandName.BRANCH:
        return _branch_argv(request)
    if request.command is ShellGitCommandName.TAG:
        return _tag_argv(request, settings)
    if request.command is ShellGitCommandName.DESCRIBE:
        return _describe_argv(request, settings)
    if request.command is ShellGitCommandName.LS_FILES:
        return _ls_files_argv(request, pathspecs)
    if request.command is ShellGitCommandName.LOG:
        return _log_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.DIFF:
        return _diff_argv(request, pathspecs)
    if request.command is ShellGitCommandName.SHOW:
        return _show_argv(request, pathspecs)
    if request.command is ShellGitCommandName.BLAME:
        return _blame_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.GREP:
        return _grep_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.STASH_LIST:
        return _stash_list_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.STASH_SHOW:
        return _stash_show_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.ADD:
        return _add_argv(request, pathspecs)
    if request.command is ShellGitCommandName.RESTORE:
        return _restore_argv(request, pathspecs)
    if request.command is ShellGitCommandName.CHECKOUT:
        return _checkout_argv(request, pathspecs)
    if request.command is ShellGitCommandName.SWITCH:
        return _switch_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.RESET:
        return _reset_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.RM:
        return _rm_argv(request, pathspecs)
    if request.command is ShellGitCommandName.MV:
        return _mv_argv(request, pathspecs)
    if request.command is ShellGitCommandName.STASH_PUSH:
        return _stash_push_argv(request, pathspecs)
    if request.command is ShellGitCommandName.STASH_APPLY:
        return _stash_apply_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.COMMIT:
        return _commit_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.BRANCH_CREATE:
        return _branch_create_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.BRANCH_DELETE:
        return _branch_delete_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.BRANCH_RENAME:
        return _branch_rename_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.TAG_CREATE:
        return _tag_create_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.TAG_DELETE:
        return _tag_delete_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.MERGE:
        return _merge_argv(request, pathspecs)
    if request.command is ShellGitCommandName.REBASE:
        return _rebase_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.CHERRY_PICK:
        return _cherry_pick_argv(request, pathspecs)
    if request.command is ShellGitCommandName.REVERT:
        return _revert_argv(request, pathspecs)
    if request.command is ShellGitCommandName.CLEAN:
        return _clean_argv(request, pathspecs)
    if request.command is ShellGitCommandName.STASH_POP:
        return _stash_pop_argv(request, pathspecs, settings)
    if request.command is ShellGitCommandName.STASH_DROP:
        return _stash_drop_argv(request, pathspecs, settings)
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.COMMAND_DISABLED,
        "shell Git command is not executable in this phase",
    )


def _base_argv(subcommand: str) -> list[str]:
    return ["git", "--no-pager", "--no-optional-locks", subcommand]


def _status_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    mode = _string_option(
        request.options,
        "mode",
        default_value="porcelain_v2",
        allowed_values=("porcelain_v2", "short"),
        message="git status mode is unsupported",
    )
    include_branch = _bool_option(
        request.options,
        "include_branch",
        default_value=True,
        message="git status include_branch must be boolean",
    )
    argv = _base_argv("status")
    argv.append("--porcelain=v2" if mode == "porcelain_v2" else "--short")
    if include_branch:
        argv.append("--branch")
    argv.extend(("--untracked-files=all", "--ignore-submodules=all"))
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _rev_parse_argv(request: ShellGitCommandRequest) -> tuple[str, ...]:
    fact = _string_option(
        request.options,
        "fact",
        default_value="head",
        allowed_values=("head", "short_head", "current_branch", "repo_root"),
        message="git rev-parse fact is unsupported",
    )
    argv = _base_argv("rev-parse")
    match fact:
        case "head":
            argv.extend(("--verify", "HEAD^{commit}"))
        case "short_head":
            argv.extend(("--short=12", "--verify", "HEAD^{commit}"))
        case "current_branch":
            argv.append("--abbrev-ref")
            argv.append("HEAD")
        case "repo_root":
            argv.append("--show-toplevel")
    return tuple(argv)


def _branch_argv(request: ShellGitCommandRequest) -> tuple[str, ...]:
    mode = _string_option(
        request.options,
        "mode",
        default_value="current",
        allowed_values=("current", "list"),
        message="git branch mode is unsupported",
    )
    contains = _optional_string_option(
        request.options,
        "contains",
        message="git branch contains revision must be a string",
    )
    argv = _base_argv("branch")
    argv.append("--no-color")
    if mode == "current":
        if contains is not None:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "git branch current mode does not support filters",
            )
        argv.append("--show-current")
        return tuple(argv)
    argv.extend(("--list", "--format=%(refname:short)"))
    if contains is not None:
        argv.extend(("--contains", contains))
    return tuple(argv)


def _tag_argv(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    mode = _string_option(
        request.options,
        "mode",
        default_value="list",
        allowed_values=("list", "show"),
        message="git tag mode is unsupported",
    )
    name = _optional_string_option(
        request.options,
        "name",
        message="git tag name must be a string",
    )
    if mode == "list":
        if name is not None:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "git tag list mode does not accept tag names",
            )
        count = _bounded_count(
            request.options.get("max_count"),
            default_value=settings.max_log_count,
            max_value=settings.max_log_count,
            option_name="max_count",
        )
        return (
            "git",
            "--no-pager",
            "--no-optional-locks",
            "for-each-ref",
            "--format=%(refname:short)",
            "--sort=refname",
            f"--count={count}",
            "refs/tags",
        )
    if name is None:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "git tag show mode requires a tag name",
        )
    _validate_ref_name(name, settings, "tag name")
    if request.options.get("max_count") is not None:
        _bounded_count(
            request.options.get("max_count"),
            default_value=1,
            max_value=1,
            option_name="max_count",
        )
    return (
        "git",
        "--no-pager",
        "--no-optional-locks",
        "for-each-ref",
        "--format=%(refname:short)%09%(objectname:short)%09%(subject)",
        "--count=1",
        f"refs/tags/{name}",
    )


def _describe_argv(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    mode = _string_option(
        request.options,
        "mode",
        default_value="tags",
        allowed_values=("tags", "always"),
        message="git describe mode is unsupported",
    )
    max_candidates = _bounded_count(
        request.options.get("max_candidates"),
        default_value=10,
        max_value=settings.max_log_count,
        option_name="max_candidates",
        allow_zero=True,
    )
    target = _optional_string_option(
        request.options,
        "target",
        message="git describe target must be a string",
    )
    argv = _base_argv("describe")
    argv.append("--tags")
    argv.append(f"--candidates={max_candidates}")
    if mode == "always":
        argv.append("--always")
    if target is not None:
        argv.append(target)
    return tuple(argv)


def _ls_files_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    mode = _string_option(
        request.options,
        "mode",
        default_value="tracked",
        allowed_values=("tracked", "modified", "deleted", "others"),
        message="git ls-files mode is unsupported",
    )
    argv = _base_argv("ls-files")
    match mode:
        case "tracked":
            argv.extend(("--cached", "--deduplicate"))
        case "modified":
            argv.extend(("--modified", "--deduplicate"))
        case "deleted":
            argv.extend(("--deleted", "--deduplicate"))
        case "others":
            argv.extend(("--others", "--exclude-standard"))
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _log_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    max_count = _bounded_count(
        request.options.get("max_count"),
        default_value=10,
        max_value=settings.max_log_count,
        option_name="max_count",
    )
    format_name = _string_option(
        request.options,
        "format",
        default_value="summary",
        allowed_values=("summary", "oneline"),
        message="git log format is unsupported",
    )
    revision = _optional_string_option(
        request.options,
        "revision",
        message="git log revision must be a string",
    )
    format_value = (
        "%H%x09%an%x09%ae%x09%ad%x09%s"
        if format_name == "summary"
        else "%h %s"
    )
    log_args = (
        f"--max-count={max_count}",
        "--no-decorate",
        "--no-color",
        "--no-ext-diff",
        "--date=iso-strict",
        f"--format={format_value}",
    )
    argv = _base_argv("log")
    argv.extend(log_args)
    if revision is not None:
        argv.append(revision)
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _diff_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    mode = _string_option(
        request.options,
        "mode",
        default_value="worktree",
        allowed_values=("worktree", "staged", "range", "stat", "name_only"),
        message="git diff mode is unsupported",
    )
    base_revision = _optional_string_option(
        request.options,
        "base_revision",
        message="git diff base revision must be a string",
    )
    head_revision = _optional_string_option(
        request.options,
        "head_revision",
        message="git diff head revision must be a string",
    )
    diff_args = (
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-renames",
    )
    argv = _base_argv("diff")
    argv.extend(diff_args)
    match mode:
        case "worktree":
            _reject_revisions_for_mode(base_revision, head_revision, mode)
            argv.append("--patch")
        case "staged":
            _reject_revisions_for_mode(base_revision, head_revision, mode)
            argv.extend(("--cached", "--patch"))
        case "range":
            if base_revision is None or head_revision is None:
                raise ShellGitPolicyDenied(
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                    "git diff range mode requires base and head revisions",
                )
            range_args = (
                "--patch",
                _commit_revision(base_revision),
                _commit_revision(head_revision),
            )
            argv.extend(range_args)
        case "stat":
            _reject_revisions_for_mode(base_revision, head_revision, mode)
            argv.append("--stat")
        case "name_only":
            _reject_revisions_for_mode(base_revision, head_revision, mode)
            argv.append("--name-only")
    _require_pathspecs_for_content(pathspecs, "git diff")
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _show_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    revision = _required_string_option(
        request.options,
        "revision",
        message="git show revision must be a string",
    )
    mode = _string_option(
        request.options,
        "mode",
        default_value="summary",
        allowed_values=("summary", "stat", "patch"),
        message="git show mode is unsupported",
    )
    show_args = (
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-decorate",
        "--no-show-signature",
        "--no-renames",
        "--date=iso-strict",
        "--format=%H%x09%an%x09%ae%x09%ad%x09%s",
    )
    argv = _base_argv("show")
    argv.extend(show_args)
    match mode:
        case "summary":
            argv.append("--no-patch")
        case "stat":
            _require_pathspecs_for_content(pathspecs, "git show stat")
            argv.append("--stat")
        case "patch":
            _require_pathspecs_for_content(pathspecs, "git show patch")
            argv.append("--patch")
    argv.append(_commit_revision(revision))
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _blame_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    if len(pathspecs) != 1:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git blame requires exactly one repo-relative file path",
        )
    start_line = _optional_positive_int_option(
        request.options,
        "start_line",
        message="git blame start line must be a positive integer",
    )
    end_line = _optional_positive_int_option(
        request.options,
        "end_line",
        message="git blame end line must be a positive integer",
    )
    blame_args = (
        "--no-textconv",
        "--date=iso-strict",
        "--line-porcelain",
        "--no-progress",
        "--no-ignore-revs-file",
        "--no-color-lines",
        "--no-color-by-age",
    )
    argv = _base_argv("blame")
    argv.extend(blame_args)
    if start_line is None and end_line is not None:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "git blame end line requires a start line",
        )
    if start_line is not None:
        final_line = start_line if end_line is None else end_line
        if final_line < start_line:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "git blame line range is invalid",
            )
        if final_line - start_line + 1 > settings.max_grep_matches:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "git blame line range exceeds the allowed limit",
            )
        argv.extend(("-L", f"{start_line},{final_line}"))
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _grep_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    pattern = _required_string_option(
        request.options,
        "pattern",
        message="git grep pattern must be a non-empty string",
    )
    case_mode = _string_option(
        request.options,
        "case",
        default_value="sensitive",
        allowed_values=("sensitive", "insensitive"),
        message="git grep case mode is unsupported",
    )
    max_matches = _bounded_count(
        request.options.get("max_matches"),
        default_value=settings.max_grep_matches,
        max_value=settings.max_grep_matches,
        option_name="max_matches",
    )
    _require_pathspecs_for_content(pathspecs, "git grep")
    grep_args = (
        "--index",
        "--no-recurse-submodules",
        "--no-textconv",
        "--fixed-strings",
        "--line-number",
        "--full-name",
        "--no-color",
        f"--max-count={max_matches}",
    )
    argv = _base_argv("grep")
    argv.extend(grep_args)
    if case_mode == "insensitive":
        argv.append("--ignore-case")
    argv.extend(("-e", pattern))
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _stash_list_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git stash list")
    max_count = _bounded_count(
        request.options.get("max_count"),
        default_value=10,
        max_value=settings.max_log_count,
        option_name="max_count",
    )
    stash_list_args = (
        "list",
        "--format=%gd%x09%gs",
        f"--max-count={max_count}",
    )
    argv = _base_argv("stash")
    argv.extend(stash_list_args)
    return tuple(argv)


def _stash_show_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    stash = _required_string_option(
        request.options,
        "stash",
        message="git stash show stash reference must be a string",
    )
    _validate_stash_ref(stash, settings)
    mode = _string_option(
        request.options,
        "mode",
        default_value="stat",
        allowed_values=("stat", "patch"),
        message="git stash show mode is unsupported",
    )
    _require_pathspecs_for_content(pathspecs, "git stash show")
    diff_args = (
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-renames",
    )
    argv = _base_argv("diff")
    argv.extend(diff_args)
    argv.append("--stat" if mode == "stat" else "--patch")
    argv.append(f"{stash}^1")
    argv.append(stash)
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _add_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _require_pathspecs_for_mutation(pathspecs, "git add")
    mode = _string_option(
        request.options,
        "mode",
        default_value="normal",
        allowed_values=("normal", "intent_to_add"),
        message="git add mode is unsupported",
    )
    argv = _base_argv("add")
    if mode == "intent_to_add":
        argv.append("--intent-to-add")
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _restore_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _require_pathspecs_for_mutation(pathspecs, "git restore")
    staged = _bool_option(
        request.options,
        "staged",
        default_value=False,
        message="git restore staged must be boolean",
    )
    worktree = _bool_option(
        request.options,
        "worktree",
        default_value=True,
        message="git restore worktree must be boolean",
    )
    if not staged and not worktree:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "git restore requires staged or worktree mode",
        )
    source_revision = _optional_string_option(
        request.options,
        "source_revision",
        message="git restore source revision must be a string",
    )
    argv = _base_argv("restore")
    argv.append("--no-overlay")
    if staged:
        argv.append("--staged")
    if worktree:
        argv.append("--worktree")
    if source_revision is not None:
        argv.extend(("--source", source_revision))
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _checkout_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    target = _optional_string_option(
        request.options,
        "target",
        message="git checkout target must be a string",
    )
    _require_pathspecs_for_mutation(pathspecs, "git checkout")
    argv = _base_argv("checkout")
    argv.append("--no-recurse-submodules")
    if target is not None:
        argv.append(target)
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _switch_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git switch")
    branch = _required_string_option(
        request.options,
        "branch",
        message="git switch branch must be a string",
    )
    _validate_local_branch_target(branch, settings, "switch branch")
    return (
        *_base_argv("switch"),
        "--no-guess",
        "--no-recurse-submodules",
        branch,
    )


def _reset_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    mode = _reset_mode_option(request.options)
    if mode == "paths":
        _require_pathspecs_for_mutation(pathspecs, "git reset")
        argv = _base_argv("reset")
        argv.append("--")
        argv.extend(pathspecs)
        return tuple(argv)
    _reject_pathspecs(pathspecs, "git reset ref-moving mode")
    revision = _required_string_option(
        request.options,
        "revision",
        message="git reset revision must be a string",
    )
    _require_confirmation(
        request.options,
        "confirm_revision",
        revision,
        "git reset",
    )
    confirm_hard = _bool_option(
        request.options,
        "confirm_hard",
        default_value=False,
        message="git reset confirm_hard must be boolean",
    )
    if mode == "hard" and not confirm_hard:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "git reset hard mode requires confirm_hard",
        )
    _validate_revision(revision, settings)
    argv = _base_argv("reset")
    argv.extend((f"--{mode}", "--no-recurse-submodules", revision))
    return tuple(argv)


def _rm_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _require_pathspecs_for_mutation(pathspecs, "git rm")
    cached = _bool_option(
        request.options,
        "cached",
        default_value=False,
        message="git rm cached must be boolean",
    )
    argv = _base_argv("rm")
    if cached:
        argv.append("--cached")
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _mv_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    if len(pathspecs) != 2:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv requires source and destination paths",
        )
    source = _required_string_option(
        request.options,
        "source",
        message="git mv source must be a string",
    )
    destination = _required_string_option(
        request.options,
        "destination",
        message="git mv destination must be a string",
    )
    if pathspecs != (source, destination):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git mv pathspecs must match source and destination",
        )
    argv = _base_argv("mv")
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _stash_push_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _require_pathspecs_for_mutation(pathspecs, "git stash push")
    message = _optional_string_option(
        request.options,
        "message",
        message="git stash push message must be a string",
    )
    include_untracked = _bool_option(
        request.options,
        "include_untracked",
        default_value=False,
        message="git stash push include_untracked must be boolean",
    )
    argv = _base_argv("stash")
    argv.append("push")
    if message is not None:
        argv.extend(("--message", message))
    if include_untracked:
        argv.append("--include-untracked")
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _stash_apply_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    stash = _required_string_option(
        request.options,
        "stash",
        message="git stash apply stash reference must be a string",
    )
    _validate_stash_ref(stash, settings)
    _require_pathspecs_for_mutation(pathspecs, "git stash apply")
    argv = _base_argv("restore")
    argv.extend(("--no-overlay", "--worktree", "--source", stash, "--"))
    argv.extend(pathspecs)
    return tuple(argv)


def _commit_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git commit")
    message = _required_string_option(
        request.options,
        "message",
        message="git commit message must be a string",
    )
    _validate_message(message, settings, "commit message")
    return (
        *_base_argv("commit"),
        "--no-edit",
        "--no-gpg-sign",
        "--no-verify",
        "--no-post-rewrite",
        "--no-status",
        "--cleanup=strip",
        "--message",
        message,
    )


def _branch_create_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git branch create")
    name = _required_string_option(
        request.options,
        "name",
        message="git branch create name must be a string",
    )
    _validate_local_branch_target(name, settings, "branch name")
    start_point = _optional_string_option(
        request.options,
        "start_point",
        message="git branch create start point must be a string",
    )
    argv = _base_argv("branch")
    argv.append("--no-track")
    argv.append(name)
    if start_point is not None:
        argv.append(start_point)
    return tuple(argv)


def _branch_delete_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git branch delete")
    name = _required_string_option(
        request.options,
        "name",
        message="git branch delete name must be a string",
    )
    _validate_local_branch_target(name, settings, "branch name")
    _require_confirmation(
        request.options,
        "confirm_name",
        name,
        "git branch delete",
    )
    return (*_base_argv("branch"), "--delete", name)


def _branch_rename_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git branch rename")
    old_name = _required_string_option(
        request.options,
        "old_name",
        message="git branch rename old_name must be a string",
    )
    new_name = _required_string_option(
        request.options,
        "new_name",
        message="git branch rename new_name must be a string",
    )
    _validate_local_branch_target(old_name, settings, "old branch name")
    _validate_local_branch_target(new_name, settings, "new branch name")
    _require_confirmation(
        request.options,
        "confirm_old_name",
        old_name,
        "git branch rename",
    )
    if old_name == new_name:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "git branch rename requires different branch names",
        )
    return (*_base_argv("branch"), "--move", old_name, new_name)


def _tag_create_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git tag create")
    name = _required_string_option(
        request.options,
        "name",
        message="git tag create name must be a string",
    )
    _validate_ref_name(name, settings, "tag name")
    target = _optional_string_option(
        request.options,
        "target",
        message="git tag create target must be a string",
    )
    message = _optional_string_option(
        request.options,
        "message",
        message="git tag create message must be a string",
    )
    argv = _base_argv("tag")
    if message is not None:
        _validate_message(message, settings, "tag message")
        argv.extend(("--annotate", "--no-sign", "--message", message))
    argv.append(name)
    if target is not None:
        argv.append(target)
    return tuple(argv)


def _tag_delete_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git tag delete")
    name = _required_string_option(
        request.options,
        "name",
        message="git tag delete name must be a string",
    )
    _validate_ref_name(name, settings, "tag name")
    _require_confirmation(
        request.options,
        "confirm_name",
        name,
        "git tag delete",
    )
    return (*_base_argv("tag"), "--delete", name)


def _merge_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git merge")
    revision = _required_string_option(
        request.options,
        "revision",
        message="git merge revision must be a string",
    )
    _require_confirmation(
        request.options,
        "confirm_revision",
        revision,
        "git merge",
    )
    mode = _string_option(
        request.options,
        "mode",
        default_value="ff_only",
        allowed_values=("ff_only", "no_ff"),
        message="git merge mode is unsupported",
    )
    argv = _base_argv("merge")
    argv.extend(
        (
            "--no-verify",
            "--no-gpg-sign",
            "--no-stat",
            "--no-edit",
            "--no-autostash",
        )
    )
    argv.append("--ff-only" if mode == "ff_only" else "--no-ff")
    argv.append(revision)
    return tuple(argv)


def _rebase_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git rebase")
    upstream = _required_string_option(
        request.options,
        "upstream",
        message="git rebase upstream must be a string",
    )
    _require_confirmation(
        request.options,
        "confirm_upstream",
        upstream,
        "git rebase",
    )
    branch = _optional_string_option(
        request.options,
        "branch",
        message="git rebase branch must be a string",
    )
    if branch is not None:
        _validate_local_branch_target(branch, settings, "rebase branch")
    argv = _base_argv("rebase")
    argv.extend(
        (
            "--no-verify",
            "--no-gpg-sign",
            "--no-stat",
            "--no-autostash",
            "--no-rebase-merges",
            "--empty=stop",
            upstream,
        )
    )
    if branch is not None:
        argv.append(branch)
    return tuple(argv)


def _cherry_pick_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git cherry-pick")
    revision = _required_string_option(
        request.options,
        "revision",
        message="git cherry-pick revision must be a string",
    )
    _require_confirmation(
        request.options,
        "confirm_revision",
        revision,
        "git cherry-pick",
    )
    return (
        *_base_argv("cherry-pick"),
        "--no-edit",
        "--no-gpg-sign",
        revision,
    )


def _revert_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git revert")
    revision = _required_string_option(
        request.options,
        "revision",
        message="git revert revision must be a string",
    )
    _require_confirmation(
        request.options,
        "confirm_revision",
        revision,
        "git revert",
    )
    return (
        *_base_argv("revert"),
        "--no-edit",
        "--no-gpg-sign",
        revision,
    )


def _clean_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    dry_run = _bool_option(
        request.options,
        "dry_run",
        default_value=True,
        message="git clean dry_run must be boolean",
    )
    _require_pathspecs_for_mutation(pathspecs, "git clean")
    argv = _base_argv("clean")
    argv.extend(("--dry-run",) if dry_run else ("--force",))
    if not dry_run:
        confirm_paths = _string_tuple_option(
            request.options,
            "confirm_paths",
            message="git clean confirm_paths must be a sequence of strings",
        )
        if confirm_paths != pathspecs:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "git clean confirm_paths must match paths",
            )
    argv.append("--")
    argv.extend(pathspecs)
    return tuple(argv)


def _stash_pop_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git stash pop")
    stash = _required_string_option(
        request.options,
        "stash",
        message="git stash pop stash reference must be a string",
    )
    _validate_stash_ref(stash, settings)
    _require_confirmation(
        request.options,
        "confirm_stash",
        stash,
        "git stash pop",
    )
    index = _bool_option(
        request.options,
        "index",
        default_value=False,
        message="git stash pop index must be boolean",
    )
    argv = _base_argv("stash")
    argv.append("pop")
    if index:
        argv.append("--index")
    argv.append(stash)
    return tuple(argv)


def _stash_drop_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    _reject_pathspecs(pathspecs, "git stash drop")
    stash = _required_string_option(
        request.options,
        "stash",
        message="git stash drop stash reference must be a string",
    )
    _validate_stash_ref(stash, settings)
    _require_confirmation(
        request.options,
        "confirm_stash",
        stash,
        "git stash drop",
    )
    return (*_base_argv("stash"), "drop", stash)


def _reject_revisions_for_mode(
    base_revision: str | None,
    head_revision: str | None,
    mode: str,
) -> None:
    if base_revision is None and head_revision is None:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.INVALID_OPTION,
        f"git diff {mode} mode does not accept revisions",
    )


def _reject_pathspecs(pathspecs: tuple[str, ...], command: str) -> None:
    if not pathspecs:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        f"{command} does not accept pathspecs",
    )


def _require_pathspecs_for_content(
    pathspecs: tuple[str, ...],
    command: str,
) -> None:
    if pathspecs:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        f"{command} requires explicit safe pathspecs",
    )


def _require_pathspecs_for_mutation(
    pathspecs: tuple[str, ...],
    command: str,
) -> None:
    if pathspecs:
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        f"{command} requires explicit safe pathspecs",
    )


def _commit_revision(revision: str) -> str:
    return f"{revision}^{{commit}}"


def _string_option(
    options: Mapping[str, object],
    name: str,
    *,
    default_value: str,
    allowed_values: tuple[str, ...],
    message: str,
) -> str:
    value = options.get(name, default_value)
    if not isinstance(value, str) or value not in allowed_values:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            message,
        )
    return value


def _reset_mode_option(options: Mapping[str, object]) -> str:
    return _string_option(
        options,
        "mode",
        default_value="paths",
        allowed_values=_RESET_MODES,
        message="git reset mode is unsupported",
    )


def _required_string_option(
    options: Mapping[str, object],
    name: str,
    *,
    message: str,
) -> str:
    value = options.get(name)
    if not isinstance(value, str) or not value:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            message,
        )
    return value


def _optional_string_option(
    options: Mapping[str, object],
    name: str,
    *,
    message: str,
) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            message,
        )
    return value


def _optional_positive_int_option(
    options: Mapping[str, object],
    name: str,
    *,
    message: str,
) -> int | None:
    value = options.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            message,
        )
    return value


def _bool_option(
    options: Mapping[str, object],
    name: str,
    *,
    default_value: bool,
    message: str,
) -> bool:
    value = options.get(name, default_value)
    if not isinstance(value, bool):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            message,
        )
    return value


def _string_tuple_option(
    options: Mapping[str, object],
    name: str,
    *,
    message: str,
) -> tuple[str, ...]:
    value = options.get(name, ())
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            message,
        )
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                message,
            )
        values.append(item)
    return tuple(values)


def _require_confirmation(
    options: Mapping[str, object],
    name: str,
    expected: str,
    command: str,
) -> None:
    value = options.get(name)
    if value != expected:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            f"{command} requires {name} to match the requested target",
        )


def _validate_message(
    value: str,
    settings: ShellGitToolSettings,
    field_name: str,
) -> None:
    if (
        not value
        or _contains_control(value)
        or "\x00" in value
        or len(value.encode("utf-8")) > settings.max_commit_message_bytes
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            f"Git {field_name} form is unsupported",
        )


def _bounded_count(
    value: object,
    *,
    default_value: int,
    max_value: int,
    option_name: str,
    allow_zero: bool = False,
) -> int:
    count = default_value if value is None else value
    minimum = 0 if allow_zero else 1
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < minimum
        or count > max_value
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            f"git {option_name} is outside the allowed range",
        )
    return count


def _validate_ref_name(
    value: str,
    settings: ShellGitToolSettings,
    field_name: str,
) -> None:
    if (
        not value
        or _contains_control(value)
        or "\x00" in value
        or value.startswith("-")
        or ":" in value
        or "@" in value
        or "\\" in value
        or "{" in value
        or "}" in value
        or "/" in value
        or ".." in value
        or " " in value
        or value == "HEAD"
        or value.endswith(".")
        or value.endswith(".lock")
        or any(character in value for character in ("*", "?", "[", "]"))
        or len(value.encode("utf-8")) > settings.max_revision_bytes
        or _HEX_REVISION_PATTERN.match(value)
        or not _REVISION_PATTERN.match(value)
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_DENIED,
            f"Git {field_name} form is unsupported",
        )


def _validate_local_branch_target(
    value: str,
    settings: ShellGitToolSettings,
    field_name: str,
) -> None:
    if (
        value == "HEAD"
        or value.startswith(".")
        or value.endswith(".")
        or value.endswith(".lock")
        or "~" in value
        or "^" in value
        or _HEX_REVISION_PATTERN.match(value)
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_DENIED,
            f"Git {field_name} form is unsupported",
        )
    _validate_ref_name(value, settings, field_name)


def _validate_stash_ref(
    value: str,
    settings: ShellGitToolSettings,
) -> None:
    match = _STASH_REF_PATTERN.match(value)
    if not match:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_DENIED,
            "Git stash reference form is unsupported",
        )
    index = int(match.group("index"))
    if index >= settings.max_log_count:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_DENIED,
            "Git stash reference exceeds the allowed range",
        )


def _validate_argv_budgets(
    argv: tuple[str, ...],
    settings: ShellToolSettings,
) -> None:
    if len(argv) > settings.max_arguments:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "Git argv exceeds argument count limit",
        )
    command_bytes = 0
    for argument in argv:
        argument_bytes = len(argument.encode("utf-8"))
        if argument_bytes > settings.max_argument_bytes:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.INVALID_OPTION,
                "Git argv argument exceeds byte limit",
            )
        command_bytes += argument_bytes
    command_bytes += max(len(argv) - 1, 0)
    if command_bytes > settings.max_command_bytes:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            "Git argv exceeds command byte limit",
        )


def _safe_git_environment(
    settings: ShellGitToolSettings,
) -> dict[str, str]:
    environment = dict(_SAFE_GIT_ENVIRONMENT)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _bounded_float(
    value: float | None,
    *,
    default_value: float,
    max_value: float,
) -> float:
    return min(default_value if value is None else value, max_value)


def _bounded_int(value: int | None, max_value: int) -> int:
    return min(max_value if value is None else value, max_value)


def _bounded_stdout_bytes(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> int:
    max_value = settings.max_stdout_bytes
    if request.command in (
        ShellGitCommandName.DIFF,
        ShellGitCommandName.SHOW,
        ShellGitCommandName.STASH_SHOW,
    ):
        max_value = min(max_value, settings.max_diff_bytes)
    return _bounded_int(request.max_stdout_bytes, max_value)


def _metadata(
    request: ShellGitCommandRequest,
    *,
    workspace_root: Path,
    effective_cwd: Path,
    repo_root: Path,
    display_argv: tuple[str, ...],
    settings: ShellGitToolSettings,
) -> dict[str, object]:
    capability = shell_git_capability_for_request(request)
    metadata: dict[str, object] = {
        "git_tool_name": request.tool_name,
        "git_command": request.command.value,
        "git_capability_required": request.capability_required.value,
        "git_capability_used": capability.value,
        "git_effective_cwd": _display_path(workspace_root, effective_cwd),
        "git_repo_root": _display_path(workspace_root, repo_root),
        "git_display_argv": display_argv,
        "git_request_options": _redacted_metadata(
            request.options,
            settings,
        ),
        "git_request_pathspecs": _redacted_metadata(
            request.pathspecs,
            settings,
        ),
    }
    if capability is ShellGitCapability.WORKTREE:
        metadata.update(
            {
                "git_mutation_attempted": True,
                "git_mutation_scope": "worktree",
            }
        )
    elif capability is ShellGitCapability.HISTORY:
        metadata.update(
            {
                "git_mutation_attempted": True,
                "git_mutation_scope": "history",
            }
        )
    if request.command is ShellGitCommandName.GREP:
        metadata["exit_code_statuses"] = {
            1: ShellExecutionStatus.NO_MATCHES.value,
        }
    return metadata


def _redacted_argv(
    argv: tuple[str, ...],
    settings: ShellGitToolSettings,
    *,
    redact_messages: bool = False,
) -> tuple[str, ...]:
    redacted: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            redacted.append("[redacted]")
            redact_next = False
            continue
        if redact_messages and argument == "--message":
            redacted.append(argument)
            redact_next = True
            continue
        if redact_messages and argument.startswith("--message="):
            redacted.append("--message=[redacted]")
            continue
        redacted.append(_redact_text(argument, settings))
    return tuple(redacted)


def _redacted_metadata(
    value: object,
    settings: ShellGitToolSettings,
) -> object:
    if isinstance(value, str):
        return _redact_text(value, settings)
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = (
                "[redacted]"
                if key_text == "message" and isinstance(item, str)
                else _redacted_metadata(item, settings)
            )
        return redacted
    if isinstance(value, tuple):
        return tuple(_redacted_metadata(item, settings) for item in value)
    if isinstance(value, list):
        return [_redacted_metadata(item, settings) for item in value]
    return value


def redact_git_text(value: str, settings: ShellGitToolSettings) -> str:
    return _redact_text(value, settings)


def _redact_text(value: str, settings: ShellGitToolSettings) -> str:
    redacted = value
    if settings.redact_credentials:
        redacted = compile_pattern(
            r"([A-Za-z][A-Za-z0-9+.-]*://)([^/@\s]+)@"
        ).sub(r"\1[redacted]@", redacted)
    if settings.redact_remote_urls:
        redacted = compile_pattern(
            r"([A-Za-z][A-Za-z0-9+.-]*://)(?:[^/@\s]+@)?([^/\s]+)(/[^\s]*)?"
        ).sub(r"\1\2/[redacted]", redacted)
    return redacted


def _display_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "[outside-workspace]"
    return "." if str(relative) == "." else relative.as_posix()


def _looks_like_bare_repository(path: Path) -> bool:
    return (
        (path / "HEAD").is_file()
        and (path / "objects").is_dir()
        and (path / "refs").is_dir()
    )


async def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return (await _read_bytes(path)).decode("utf-8", errors="replace")


def _config_declares_bare(config: str) -> bool:
    for line in config.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("bare") and "true" in stripped:
            return True
    return False


def _resolve_gitdir_reference(value: str, *, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_repo_path(repo_root: Path, path: PurePosixPath) -> Path:
    return (repo_root / Path(*path.parts)).resolve()


def _contains_unsafe_path_text(value: str) -> bool:
    return (
        not value
        or _contains_control(value)
        or "\x00" in value
        or value.startswith("~")
        or "$" in value
        or "{" in value
        or "}" in value
    )


def _contains_control(value: str) -> bool:
    return any(character in value for character in _CONTROL_CHARACTERS)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
