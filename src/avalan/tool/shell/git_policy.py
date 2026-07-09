from ...filesystem import which_executable as _which_executable
from ...types import assert_non_empty_string as _assert_non_empty_string
from .commands.helpers import path_matches_sensitive_denylist
from .entities import (
    ExecutionSpec,
    ShellExecutionModeValue,
    ShellExecutionStatus,
    ShellOutputKind,
)
from .filesystem import inspect_path as _inspect_path
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
from dataclasses import dataclass
from hashlib import sha1, sha256
from hmac import compare_digest
from pathlib import Path, PurePosixPath
from re import compile as compile_pattern
from stat import S_IXGRP, S_IXOTH, S_IXUSR
from typing import NoReturn, cast, final
from urllib.parse import unquote, urlsplit
from zlib import decompressobj as zlib_decompressobj
from zlib import error as ZlibError

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
    "receive-pack",
    "receivepack",
    "upload-pack",
    "uploadpack",
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
    "uploadpack",
    "receivepack",
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
_AUTHOR_SUMMARY_EMAIL_PATTERN = compile_pattern(
    r"(?m)^([0-9a-fA-F]{7,64}\t[^\t\r\n]*\t)" r"[^\t\r\n]*@[^\t\r\n]*(\t)"
)
_AUTHOR_HEADER_EMAIL_PATTERN = compile_pattern(
    r"(?im)^((?:author|commit|committer):[^\r\n<]*<)"
    r"[^<>\r\n]*@[^<>\r\n]*(>)"
)
_AUTHOR_PORCELAIN_EMAIL_PATTERN = compile_pattern(
    r"(?im)^((?:author|committer)-mail\s+<)" r"[^<>\r\n]*@[^<>\r\n]*(>)"
)
_GIT_INDEX_FILE_MODES = frozenset((0o100644, 0o100755, 0o120000))
_GIT_INDEX_MAX_PROBE_BYTES = 8 * 1024 * 1024
_GIT_INDEX_LINK_EXTENSION = b"link"
_GIT_INDEX_SIGNATURE = b"DIRC"
_GIT_INDEX_SPARSE_EXTENSION = b"sdir"
_GIT_INDEX_SUPPORTED_VERSIONS = (2, 3)
_GIT_LOOSE_OBJECT_MAX_PROBE_BYTES = 8 * 1024 * 1024
_GIT_OBJECT_SIZE_MAX_DIGITS = 10
_GIT_OBJECT_FORMAT_SHA1 = "sha1"
_GIT_OBJECT_FORMAT_SHA256 = "sha256"
_GIT_OBJECT_HASH_SIZE_SHA1 = 20
_GIT_OBJECT_HASH_SIZE_SHA256 = 32
_GIT_TREE_DIRECTORY_MODE = 0o40000
_CONTROL_CHARACTERS = tuple(chr(value) for value in (*range(0, 32), 127))
_DANGEROUS_CONFIG_SECTION_NAMES = frozenset((
    "credential",
    "diff",
    "filter",
    "gpg",
    "hook",
    "include",
    "includeif",
    "merge",
    "pager",
))
_DANGEROUS_CONFIG_SECTION_PREFIXES = frozenset(
    f"{section}." for section in _DANGEROUS_CONFIG_SECTION_NAMES
)
_DANGEROUS_CONFIG_KEY_NAMES = frozenset((
    "askpass",
    "editor",
    "external",
    "fsmonitor",
    "gpgsign",
    "hookspath",
    "ignorerevsfile",
    "insteadof",
    "pager",
    "pushinsteadof",
    "receivepack",
    "showsignature",
    "sshcommand",
    "textconv",
    "uploadpack",
    "worktree",
    "worktreeconfig",
))
_DANGEROUS_CONFIG_KEY_MARKERS = (
    "credential",
    "gpg",
    "proxy",
    "signing",
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
    ShellGitCommandName.RESTORE: frozenset({
        "source_revision",
        "staged",
        "worktree",
    }),
    ShellGitCommandName.CHECKOUT: frozenset({"target"}),
    ShellGitCommandName.SWITCH: frozenset({"branch"}),
    ShellGitCommandName.RESET: frozenset(
        {"mode", "revision", "confirm_revision", "confirm_hard"},
    ),
    ShellGitCommandName.RM: frozenset({"cached"}),
    ShellGitCommandName.MV: frozenset({"source", "destination"}),
    ShellGitCommandName.STASH_PUSH: frozenset({
        "message",
        "include_untracked",
    }),
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
    ShellGitCommandName.FETCH: frozenset(
        {"remote", "ref_type", "ref_name"},
    ),
    ShellGitCommandName.PULL: frozenset({"remote", "branch"}),
    ShellGitCommandName.PUSH: frozenset(
        {"remote", "ref_type", "ref_name"},
    ),
    ShellGitCommandName.CLONE: frozenset(
        {"url", "destination", "branch"},
    ),
    ShellGitCommandName.REMOTE_LIST: frozenset(),
    ShellGitCommandName.REMOTE_ADD: frozenset({"name", "url"}),
    ShellGitCommandName.REMOTE_SET_URL: frozenset({"name", "url"}),
    ShellGitCommandName.REMOTE_REMOVE: frozenset({"name"}),
    ShellGitCommandName.REMOTE_RENAME: frozenset({"old_name", "new_name"}),
    ShellGitCommandName.SUBMODULE_UPDATE: frozenset({"init"}),
}
_REMOTE_REPOSITORY_COMMANDS = frozenset((
    ShellGitCommandName.FETCH,
    ShellGitCommandName.PULL,
    ShellGitCommandName.PUSH,
    ShellGitCommandName.REMOTE_LIST,
    ShellGitCommandName.REMOTE_ADD,
    ShellGitCommandName.REMOTE_SET_URL,
    ShellGitCommandName.REMOTE_REMOVE,
    ShellGitCommandName.REMOTE_RENAME,
    ShellGitCommandName.SUBMODULE_UPDATE,
))
_REMOTE_NETWORK_COMMANDS = frozenset((
    ShellGitCommandName.FETCH,
    ShellGitCommandName.PULL,
    ShellGitCommandName.PUSH,
    ShellGitCommandName.CLONE,
    ShellGitCommandName.SUBMODULE_UPDATE,
))
_PULL_REQUIRED_CAPABILITIES = (
    ShellGitCapability.REMOTE,
    ShellGitCapability.WORKTREE,
    ShellGitCapability.HISTORY,
)
_REMOTE_STATE_MUTATING_COMMANDS = frozenset((ShellGitCommandName.PUSH,))
_SERVER_SIDE_PUSH_HOOK_NAMES = frozenset((
    "pre-receive",
    "update",
    "post-receive",
    "post-update",
    "proc-receive",
    "reference-transaction",
    "push-to-checkout",
))
_EXECUTABLE_PERMISSION_BITS = S_IXUSR | S_IXGRP | S_IXOTH
_LOCAL_REMOTE_CONFIG_MUTATING_COMMANDS = frozenset((
    ShellGitCommandName.REMOTE_ADD,
    ShellGitCommandName.REMOTE_SET_URL,
    ShellGitCommandName.REMOTE_REMOVE,
    ShellGitCommandName.REMOTE_RENAME,
))
_REMOTE_CONFIG_DENIED_KEYS = frozenset((
    "pushurl",
    "receivepack",
    "serveroption",
    "uploadpack",
    "vcs",
))
_REMOTE_CONFIG_DENIED_BOOL_KEYS = frozenset((
    "mirror",
    "prune",
    "prunetags",
))
_HTTP_CREDENTIAL_CONFIG_KEYS = frozenset((
    "cookiefile",
    "delegation",
    "emptyauth",
    "extraheader",
    "savecookies",
    "sslcert",
    "sslcertpasswordprotected",
    "sslkey",
))
_HTTP_TLS_CONFIG_KEYS = frozenset((
    "pinnedpubkey",
    "schannelcheckrevoke",
    "schannelusesslcainfo",
    "sslbackend",
    "sslcainfo",
    "sslcapath",
    "sslcipherlist",
    "ssltry",
    "sslverify",
    "sslversion",
))
_FETCH_CONFIG_DENIED_BOOL_KEYS = frozenset(("prune", "prunetags"))
_PULL_CONFIG_DENIED_KEYS = frozenset((
    "ff",
    "octopus",
    "rebase",
    "twohead",
))
_PUSH_CONFIG_DENIED_KEYS = frozenset((
    "default",
    "followtags",
    "gpgsign",
    "pushoption",
))
_GIT_CONFIG_EXPLICIT_FALSE_VALUES = frozenset(("false", "no", "off", "0"))
_GIT_CONFIG_EXPLICIT_TRUE_VALUES = frozenset(("true", "yes", "on", "1"))
_REMOTE_NAME_PATTERN = compile_pattern(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REMOTE_REF_TYPES = ("branch", "tag")
_REMOTE_REF_PATTERN = compile_pattern(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._/-]*[A-Za-z0-9._-])?$"
)
_URL_SCHEME_PATTERN = compile_pattern(r"^[A-Za-z][A-Za-z0-9+.-]*$")


@final
@dataclass(frozen=True, slots=True)
class _GitConfigEntry:
    section: str
    subsection: str | None
    key: str
    value: str


@final
@dataclass(frozen=True, slots=True)
class _RemoteConfig:
    name: str
    urls: tuple[str, ...]
    entries: tuple[_GitConfigEntry, ...]


@dataclass(frozen=True, slots=True)
class _GitIndexEntry:
    mode: int
    path: bytes


@dataclass(frozen=True, slots=True)
class _GitIndexData:
    entries: tuple[_GitIndexEntry, ...]
    hash_size: int
    split_index: bool = False
    sparse: bool = False


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
        if request.command is ShellGitCommandName.CLONE:
            effective_cwd = _resolve_workspace_cwd(
                request.cwd or git_settings.cwd,
                workspace_root=workspace_root,
            )
        else:
            effective_cwd = _resolve_effective_cwd(
                request.cwd or git_settings.cwd,
                workspace_root=workspace_root,
            )
        _validate_git_option_keys(request)
        _validate_revisions(request, git_settings)
        _validate_git_option_values(request.options)
        if request.command is ShellGitCommandName.CLONE:
            repo_root = None
            git_dir = None
            execution_cwd = workspace_root
            pathspecs: tuple[str, ...] = ()
            argv_request = request
            _validate_clone_request(
                request,
                workspace_root=workspace_root,
                settings=git_settings,
                allow_hidden=self._settings.allow_hidden,
            )
            clone_url = _required_string_option(
                request.options,
                "url",
                message="git clone url must be a string",
            )
            remote_urls: dict[str, str] = {"origin": clone_url}
        else:
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
            await _validate_history_ref_state(
                request,
                git_dir=git_dir,
                settings=git_settings,
            )
            remote_urls = await _validate_remote_repository_state(
                request,
                git_dir=git_dir,
                repo_root=repo_root,
                workspace_root=workspace_root,
                settings=git_settings,
            )
            argv_request = request
            await _validate_content_pathspec_scope(
                request,
                pathspecs,
                repo_root,
                git_dir=git_dir,
            )
            _validate_mutation_pathspec_scope(request, pathspecs, repo_root)
            execution_cwd = repo_root
        argv = _argv_for_request(
            argv_request,
            pathspecs,
            git_settings,
            workspace_root=workspace_root,
            git_dir=git_dir,
            allow_hidden=self._settings.allow_hidden,
        )
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
            argv_request,
            workspace_root=workspace_root,
            effective_cwd=effective_cwd,
            repo_root=repo_root,
            display_argv=display_argv,
            settings=git_settings,
            remote_urls=remote_urls,
        )
        executable = git_settings.executable_path or (
            await self._executable_lookup(
                tuple(self._settings.executable_search_paths)
            )
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
            cwd=str(execution_cwd),
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
            f"{request.tool_name} requires capability {capability.value}; "
            f"configured capabilities: {', '.join(settings.capabilities)}",
        )
    _validate_multi_capability_authorization(request, settings)
    if capability is ShellGitCapability.REMOTE:
        _validate_remote_policy(request, settings)


def _validate_multi_capability_authorization(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> None:
    if request.command is not ShellGitCommandName.PULL:
        return
    missing = tuple(
        capability.value
        for capability in _PULL_REQUIRED_CAPABILITIES
        if capability.value not in settings.capabilities
    )
    if missing:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
            f"{request.tool_name} requires capabilities "
            + ", ".join(
                capability.value for capability in _PULL_REQUIRED_CAPABILITIES
            )
            + "; configured capabilities: "
            + ", ".join(settings.capabilities),
        )


def _validate_remote_policy(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> None:
    if not settings.allowed_remote_protocols:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "remote protocols are not allowlisted",
        )
    if not settings.allowed_remote_hosts:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
            "remote hosts are not allowlisted",
        )
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
    url = request.options.get("url")
    if isinstance(url, str):
        _validate_remote_url(url, settings)


def _validate_remote_url(
    url: str,
    settings: ShellGitToolSettings,
    *,
    workspace_root: Path | None = None,
) -> None:
    try:
        parts = urlsplit(url)
    except ValueError as error:
        protocol = _url_scheme_prefix(url)
        if protocol and protocol not in settings.allowed_remote_protocols:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
                "remote URL protocol is not allowlisted",
            ) from error
        if protocol:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
                "remote URL host is unsupported",
            ) from error
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "remote URL protocol is unsupported",
        ) from error
    if (
        parts.query
        or parts.fragment
        or _contains_control(url)
        or "\x00" in url
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "remote URL protocol is unsupported",
        )
    protocol = parts.scheme.lower()
    if protocol == "file" and not parts.netloc:
        if protocol not in settings.allowed_remote_protocols:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
                "remote URL protocol is not allowlisted",
            )
        if "localhost" not in settings.allowed_remote_hosts:
            raise ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
                "remote URL host is not allowlisted",
            )
        _validate_hostless_file_remote_url(
            url,
            parts.path,
            workspace_root=workspace_root,
        )
        return
    if not parts.scheme or not parts.netloc or not parts.hostname:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "remote URL protocol is unsupported",
        )
    host = parts.hostname.lower()
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
    if not settings.allow_remote_credentials and (
        parts.username is not None or parts.password is not None
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
            "remote URL credentials are disabled",
        )
    if protocol == "file":
        _validate_file_remote_url(parts.path, workspace_root=workspace_root)


async def _validate_remote_url_for_command(
    url: str,
    settings: ShellGitToolSettings,
    *,
    workspace_root: Path,
    command: ShellGitCommandName,
) -> None:
    _validate_remote_url(url, settings, workspace_root=workspace_root)
    if command is ShellGitCommandName.PUSH:
        await _validate_file_push_remote_url(
            url,
            workspace_root=workspace_root,
        )


def _validate_file_remote_url(
    value: str,
    *,
    workspace_root: Path | None,
) -> None:
    _validated_file_remote_path(value, workspace_root=workspace_root)


def _validate_hostless_file_remote_url(
    url: str,
    value: str,
    *,
    workspace_root: Path | None,
) -> None:
    decoded_value = _decoded_file_remote_path(value)
    if (
        not url.startswith("file:///")
        or value.startswith("//")
        or decoded_value.startswith("//")
        or not Path(decoded_value).is_absolute()
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "remote URL protocol is unsupported",
        )
    _validate_file_remote_url(value, workspace_root=workspace_root)


def _validated_file_remote_path(
    value: str,
    *,
    workspace_root: Path | None,
) -> Path | None:
    if not value:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "file remote URL path is unsupported",
        )
    decoded_value = _decoded_file_remote_path(value)
    if workspace_root is None:
        return None
    path = Path(decoded_value)
    if not path.is_absolute():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
            "file remote URL path must be absolute",
        )
    resolved = path.resolve()
    if not _is_relative_to(resolved, workspace_root):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
            "file remote URL escapes workspace root",
        )
    return resolved


def _decoded_file_remote_path(value: str) -> str:
    decoded_value = unquote(value)
    if _contains_control(decoded_value) or "\x00" in decoded_value:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            "file remote URL path is unsupported",
        )
    return decoded_value


async def _validate_file_push_remote_url(
    url: str,
    *,
    workspace_root: Path,
) -> None:
    parts = urlsplit(url)
    if parts.scheme.lower() != "file":
        return
    target = _validated_file_remote_path(
        parts.path,
        workspace_root=workspace_root,
    )
    assert target is not None, "workspace root must resolve file remote path"
    await _validate_file_push_remote_target(target)


async def _validate_file_push_remote_target(target: Path) -> None:
    if not target.exists() or not target.is_dir():
        _deny_unsafe_file_push_target()
    git_marker = target / ".git"
    if git_marker.exists() or git_marker.is_symlink():
        _deny_unsafe_file_push_target()
    if not _looks_like_bare_repository(target):
        _deny_unsafe_file_push_target()
    common_dir = target / "commondir"
    if common_dir.exists() or common_dir.is_symlink():
        _deny_unsafe_file_push_target()
    config_path = target / "config"
    if config_path.is_symlink() or not config_path.is_file():
        _deny_unsafe_file_push_target()
    try:
        config = await _read_text(config_path)
    except OSError as error:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            "Git file push target cannot be proven safe",
        ) from error
    _validate_git_config(config)
    if not _config_strictly_declares_bare(config):
        _deny_unsafe_file_push_target()
    await _validate_server_side_push_hooks(target / "hooks")


async def _validate_server_side_push_hooks(path: Path) -> None:
    if not path.exists():
        if path.is_symlink():
            _deny_unsafe_file_push_target_hooks()
        return
    if path.is_symlink() or not path.is_dir():
        _deny_unsafe_file_push_target_hooks()
    try:
        children = await _list_directory(path)
    except OSError as error:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            "Git file push target cannot be proven safe",
        ) from error
    for child in children:
        if child.name.endswith(".sample"):
            continue
        if child.name not in _SERVER_SIDE_PUSH_HOOK_NAMES:
            continue
        await _validate_server_side_push_hook(child)


async def _validate_server_side_push_hook(path: Path) -> None:
    if path.is_symlink() or not path.exists() or not path.is_file():
        _deny_unsafe_file_push_target_hooks()
    try:
        metadata = await _inspect_path(path)
    except OSError as error:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            "Git file push target cannot be proven safe",
        ) from error
    if metadata.mode & _EXECUTABLE_PERMISSION_BITS:
        _deny_unsafe_file_push_target_hooks()


def _deny_unsafe_file_push_target() -> NoReturn:
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        "Git file push target cannot be proven safe",
    )


def _deny_unsafe_file_push_target_hooks() -> NoReturn:
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        "Git file push target hooks can trigger external processing",
    )


def _remote_url_parts(url: str) -> tuple[str, str, str]:
    try:
        parts = urlsplit(url)
    except ValueError:
        return "", "", url
    if (
        parts.scheme.lower() == "file"
        and not parts.netloc
        and url.lower().startswith("file:///")
    ):
        return "file", "", url
    if not parts.scheme or not parts.hostname:
        return "", "", url
    protocol = parts.scheme.lower()
    host = parts.hostname.lower()
    return protocol, host, url


def _url_scheme_prefix(url: str) -> str:
    scheme, separator, _ = url.partition("://")
    if not separator or not _URL_SCHEME_PATTERN.match(scheme):
        return ""
    return scheme.lower()


def _validate_remote_name(value: str, field_name: str) -> None:
    if (
        not value
        or _contains_control(value)
        or "\x00" in value
        or value.startswith("-")
        or value.startswith(".")
        or value.endswith(".")
        or value.endswith(".lock")
        or value == "HEAD"
        or "/" in value
        or "\\" in value
        or ":" in value
        or "@" in value
        or ".." in value
        or " " in value
        or any(character in value for character in ("*", "?", "[", "]"))
        or not _REMOTE_NAME_PATTERN.match(value)
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            f"Git {field_name} form is unsupported",
        )


async def _validate_remote_repository_state(
    request: ShellGitCommandRequest,
    *,
    git_dir: Path,
    repo_root: Path,
    workspace_root: Path,
    settings: ShellGitToolSettings,
) -> dict[str, str]:
    if request.command not in _REMOTE_REPOSITORY_COMMANDS:
        return {}
    config = await _read_text(git_dir / "config")
    entries = _git_config_entries(config)
    _validate_remote_command_config(entries, request.command, settings)
    remotes = _remote_configs(entries)
    match request.command:
        case ShellGitCommandName.FETCH | ShellGitCommandName.PULL:
            remote = _remote_option(request.options)
            url = await _require_existing_remote(
                remote,
                remotes,
                settings,
                workspace_root=workspace_root,
                command=request.command,
            )
            return {remote: url}
        case ShellGitCommandName.PUSH:
            remote = _remote_option(request.options)
            url = await _require_existing_remote(
                remote,
                remotes,
                settings,
                workspace_root=workspace_root,
                command=request.command,
            )
            return {remote: url}
        case ShellGitCommandName.REMOTE_LIST:
            for name in remotes:
                await _require_existing_remote(
                    name,
                    remotes,
                    settings,
                    workspace_root=workspace_root,
                    command=request.command,
                )
            return {
                name: _remote_config_url(remote_config)
                for name, remote_config in remotes.items()
            }
        case ShellGitCommandName.REMOTE_ADD:
            name = _remote_name_option(request.options, "name")
            if name in remotes:
                raise ShellGitPolicyDenied(
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                    "Git remote entry already exists",
                )
            url = _required_string_option(
                request.options,
                "url",
                message="git remote add url must be a string",
            )
            await _validate_remote_url_for_command(
                url,
                settings,
                workspace_root=workspace_root,
                command=request.command,
            )
            return {name: url}
        case ShellGitCommandName.REMOTE_SET_URL:
            name = _remote_name_option(request.options, "name")
            await _require_existing_remote(
                name,
                remotes,
                settings,
                workspace_root=workspace_root,
                command=request.command,
            )
            url = _required_string_option(
                request.options,
                "url",
                message="git remote set-url url must be a string",
            )
            await _validate_remote_url_for_command(
                url,
                settings,
                workspace_root=workspace_root,
                command=request.command,
            )
            return {name: url}
        case ShellGitCommandName.REMOTE_REMOVE:
            name = _remote_name_option(request.options, "name")
            url = await _require_existing_remote(
                name,
                remotes,
                settings,
                workspace_root=workspace_root,
                command=request.command,
            )
            return {name: url}
        case ShellGitCommandName.REMOTE_RENAME:
            old_name = _remote_name_option(request.options, "old_name")
            new_name = _remote_name_option(request.options, "new_name")
            url = await _require_existing_remote(
                old_name,
                remotes,
                settings,
                workspace_root=workspace_root,
                command=request.command,
            )
            if old_name == new_name:
                raise ShellGitPolicyDenied(
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                    "git remote rename requires different remote names",
                )
            if new_name in remotes:
                raise ShellGitPolicyDenied(
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                    "Git destination remote entry already exists",
                )
            return {old_name: url, new_name: url}
    assert request.command is ShellGitCommandName.SUBMODULE_UPDATE
    _require_pathspecs_for_mutation(
        request.pathspecs,
        "git submodule update",
    )
    return await _validate_submodule_urls(
        repo_root,
        git_dir,
        settings,
        workspace_root=workspace_root,
    )


def _git_config_entries(config: str) -> tuple[_GitConfigEntry, ...]:
    entries: list[_GitConfigEntry] = []
    section: str | None = None
    subsection: str | None = None
    for line in config.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("["):
            section, subsection = _git_config_section(stripped)
            continue
        if section is None:
            continue
        key, value = _git_config_key_value(stripped)
        if not key:
            continue
        entries.append(
            _GitConfigEntry(
                section=section,
                subsection=subsection,
                key=key,
                value=value.strip(),
            )
        )
    return tuple(entries)


def _git_config_section_names(config: str) -> tuple[str, ...]:
    sections: list[str] = []
    for line in config.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("["):
            section, _ = _git_config_section(stripped)
            if section is not None:
                sections.append(section)
    return tuple(sections)


def _git_config_key_value(line: str) -> tuple[str, str]:
    if "=" in line:
        key, value = line.split("=", maxsplit=1)
        return key.strip().lower(), value.strip()
    return _git_config_bare_key(line), "true"


def _git_config_bare_key(line: str) -> str:
    key = _git_config_unquoted_comment_prefix(line).strip().lower()
    return key


def _git_config_unquoted_comment_prefix(line: str) -> str:
    quoted = False
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if quoted and character == "\\":
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if not quoted and character in ("#", ";"):
            return line[:index]
    return line


def _git_config_section(section: str) -> tuple[str | None, str | None]:
    closing_index = _git_config_section_closing_index(section)
    if closing_index is None:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            "Git repository configuration is unsafe",
        )
    trailing = section[closing_index + 1 :].strip()
    if trailing and not trailing.startswith(("#", ";")):
        _deny_unsafe_git_config("Git repository configuration is unsafe")
    content = section[1:closing_index].strip()
    if not content:
        return None, None
    parts = content.split(maxsplit=1)
    if len(parts) == 1:
        legacy_section = _git_config_legacy_subsection(content)
        if legacy_section is not None:
            return legacy_section
        return content.lower(), None
    name, subsection = parts
    subsection = subsection.strip()
    if subsection.startswith('"') and subsection.endswith('"'):
        subsection = subsection[1:-1]
    return name.lower(), subsection


def _git_config_section_closing_index(section: str) -> int | None:
    quoted = False
    escaped = False
    for index, character in enumerate(section[1:], start=1):
        if escaped:
            escaped = False
            continue
        if quoted and character == "\\":
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if character == "]" and not quoted:
            return index
    return None


def _git_config_legacy_subsection(
    content: str,
) -> tuple[str, str] | None:
    lowered = content.lower()
    for section in ("http", "remote", "submodule"):
        prefix = f"{section}."
        if lowered.startswith(prefix):
            return section, content[len(prefix) :]
    return None


def _remote_configs(
    entries: tuple[_GitConfigEntry, ...],
) -> dict[str, _RemoteConfig]:
    remote_entries: dict[str, list[_GitConfigEntry]] = {}
    urls: dict[str, list[str]] = {}
    for entry in entries:
        if entry.section != "remote" or entry.subsection is None:
            continue
        _validate_remote_name(entry.subsection, "remote name")
        remote_entries.setdefault(entry.subsection, []).append(entry)
        if entry.key == "url":
            urls.setdefault(entry.subsection, []).append(entry.value)
    return {
        name: _RemoteConfig(
            name=name,
            urls=tuple(urls.get(name, ())),
            entries=tuple(values),
        )
        for name, values in remote_entries.items()
    }


def _validate_remote_command_config(
    entries: tuple[_GitConfigEntry, ...],
    command: ShellGitCommandName,
    settings: ShellGitToolSettings,
) -> None:
    assert isinstance(settings, ShellGitToolSettings)
    for entry in entries:
        if entry.section == "http":
            _validate_http_remote_config(entry)
        if entry.section == "fetch" and command in (
            ShellGitCommandName.FETCH,
            ShellGitCommandName.PULL,
        ):
            if (
                entry.key in _FETCH_CONFIG_DENIED_BOOL_KEYS
                and _git_config_bool_enabled(entry.value)
            ):
                _deny_unsafe_git_config("Git fetch configuration is unsafe")
        if entry.section == "pull" and command is ShellGitCommandName.PULL:
            if entry.key in _PULL_CONFIG_DENIED_KEYS:
                _deny_unsafe_git_config("Git pull configuration is unsafe")
        if entry.section == "push" and command is ShellGitCommandName.PUSH:
            if entry.key in _PUSH_CONFIG_DENIED_KEYS:
                _deny_unsafe_git_config("Git push configuration is unsafe")


def _validate_http_remote_config(entry: _GitConfigEntry) -> None:
    if (
        _http_config_subsection_has_credentials(entry)
        or entry.key in _HTTP_CREDENTIAL_CONFIG_KEYS
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
            "Git HTTP credential configuration is disabled",
        )
    if entry.key in _HTTP_TLS_CONFIG_KEYS or entry.key.startswith("ssl"):
        _deny_unsafe_git_config("Git HTTP TLS configuration is unsafe")


def _http_config_subsection_has_credentials(entry: _GitConfigEntry) -> bool:
    if entry.subsection is None:
        return False
    has_userinfo_marker = "@" in entry.subsection
    try:
        parts = urlsplit(entry.subsection)
    except ValueError as error:
        if has_userinfo_marker:
            return True
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            "Git HTTP URL configuration is unsafe",
        ) from error
    if parts.username is not None or parts.password is not None:
        return True
    if has_userinfo_marker:
        return True
    return False


def _validate_remote_config(
    remote_config: _RemoteConfig,
    command: ShellGitCommandName,
) -> None:
    for entry in remote_config.entries:
        if entry.key in _REMOTE_CONFIG_DENIED_KEYS:
            _deny_unsafe_git_config("Git remote configuration is unsafe")
        if (
            entry.key in _REMOTE_CONFIG_DENIED_BOOL_KEYS
            and _git_config_bool_enabled(entry.value)
        ):
            _deny_unsafe_git_config("Git remote configuration is unsafe")
        if entry.key == "tagopt" and entry.value.strip() != "--no-tags":
            _deny_unsafe_git_config("Git remote tag configuration is unsafe")
        if (
            command is ShellGitCommandName.PUSH
            and entry.key == "tagopt"
            and entry.value.strip()
        ):
            _deny_unsafe_git_config("Git remote tag configuration is unsafe")


def _git_config_bool_enabled(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered or lowered in _GIT_CONFIG_EXPLICIT_FALSE_VALUES:
        return False
    if lowered in _GIT_CONFIG_EXPLICIT_TRUE_VALUES:
        return True
    return True


def _deny_unsafe_git_config(message: str) -> None:
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        message,
    )


def _remote_option(options: Mapping[str, object]) -> str:
    return _remote_name_option(options, "remote", default_value="origin")


def _remote_name_option(
    options: Mapping[str, object],
    name: str,
    *,
    default_value: str | None = None,
) -> str:
    value = options.get(name, default_value)
    if not isinstance(value, str) or not value:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.INVALID_OPTION,
            f"git remote {name} must be a string",
        )
    _validate_remote_name(value, f"remote {name}")
    return value


async def _require_existing_remote(
    name: str,
    remotes: Mapping[str, _RemoteConfig],
    settings: ShellGitToolSettings,
    *,
    workspace_root: Path,
    command: ShellGitCommandName,
) -> str:
    remote_config = remotes.get(name)
    if remote_config is None:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            "Git remote entry was not found",
        )
    _validate_remote_config(remote_config, command)
    url = _remote_config_url(remote_config)
    await _validate_remote_url_for_command(
        url,
        settings,
        workspace_root=workspace_root,
        command=command,
    )
    return url


def _remote_config_url(remote_config: _RemoteConfig) -> str:
    if not remote_config.urls:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            "Git remote URL was not found",
        )
    if len(remote_config.urls) != 1:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            "Git remote URL configuration is unsupported",
        )
    return remote_config.urls[0]


async def _validate_submodule_urls(
    repo_root: Path,
    git_dir: Path,
    settings: ShellGitToolSettings,
    *,
    workspace_root: Path,
) -> dict[str, str]:
    gitmodules_entries = _git_config_entries(
        await _read_text(repo_root / ".gitmodules"),
    )
    repo_entries = _git_config_entries(await _read_text(git_dir / "config"))
    _validate_submodule_effective_config(gitmodules_entries, repo_entries)
    urls = _submodule_urls(gitmodules_entries)
    if not urls:
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.SUBMODULE_DENIED,
            "git submodule update requires validated submodule URLs",
        )
    for url in urls.values():
        _validate_remote_url(url, settings, workspace_root=workspace_root)
    return urls


def _validate_submodule_effective_config(
    gitmodules_entries: tuple[_GitConfigEntry, ...],
    repo_entries: tuple[_GitConfigEntry, ...],
) -> None:
    for entry in gitmodules_entries:
        if entry.section == "submodule":
            if entry.key == "update":
                _deny_unsafe_git_config(
                    "Git submodule update configuration is unsafe"
                )
            if (
                entry.key == "recurse"
                and not _git_config_bool_explicitly_disabled(entry.value)
            ):
                _deny_unsafe_git_config(
                    "Git submodule recurse configuration is unsafe"
                )
    for entry in repo_entries:
        if entry.section != "submodule":
            continue
        if entry.key in ("url", "update"):
            _deny_unsafe_git_config(
                "Git effective submodule configuration is unsafe"
            )
        if entry.key == "recurse" and not _git_config_bool_explicitly_disabled(
            entry.value
        ):
            _deny_unsafe_git_config(
                "Git effective submodule configuration is unsafe"
            )


def _git_config_bool_explicitly_disabled(value: str) -> bool:
    return value.strip().lower() in _GIT_CONFIG_EXPLICIT_FALSE_VALUES


def _git_config_bool_explicitly_enabled(value: str) -> bool:
    return value.strip().lower() in _GIT_CONFIG_EXPLICIT_TRUE_VALUES


def _submodule_urls(
    gitmodules_entries: tuple[_GitConfigEntry, ...],
) -> dict[str, str]:
    urls: dict[str, str] = {}
    index = 0
    for entry in gitmodules_entries:
        if (
            entry.section != "submodule"
            or entry.subsection is None
            or entry.key != "url"
        ):
            continue
        key = f"gitmodules:submodule:{entry.subsection or index}"
        if key in urls:
            key = f"{key}:{index}"
        urls[key] = entry.value
        index += 1
    return urls


def _validate_clone_request(
    request: ShellGitCommandRequest,
    *,
    workspace_root: Path,
    settings: ShellGitToolSettings,
    allow_hidden: bool,
) -> None:
    url = _required_string_option(
        request.options,
        "url",
        message="git clone url must be a string",
    )
    _validate_remote_url(url, settings, workspace_root=workspace_root)
    destination = _required_string_option(
        request.options,
        "destination",
        message="git clone destination must be a string",
    )
    if request.pathspecs and request.pathspecs != (destination,):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "git clone pathspecs must match destination",
        )
    _validate_workspace_destination(
        destination,
        workspace_root=workspace_root,
        allow_hidden=allow_hidden,
    )
    branch = _optional_string_option(
        request.options,
        "branch",
        message="git clone branch must be a string",
    )
    if branch is not None:
        _validate_remote_ref(branch, settings, "clone branch")


def _validate_workspace_destination(
    value: str,
    *,
    workspace_root: Path,
    allow_hidden: bool,
) -> None:
    if _contains_unsafe_path_text(value):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git clone destination is unsafe",
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() == ".":
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git clone destination must be workspace-relative",
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
            "Git clone destination form is unsupported",
        )
    if any(part == ".git" for part in path.parts):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git clone destination must not target Git metadata",
        )
    display_path = path.as_posix()
    if not allow_hidden and _has_hidden_pathspec_component(display_path):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "hidden Git clone destinations are unsupported",
        )
    if path_matches_sensitive_denylist(display_path):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "sensitive Git clone destinations are unsupported",
        )
    resolved = (workspace_root / Path(*path.parts)).resolve()
    if not _is_relative_to(resolved, workspace_root):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git clone destination escapes workspace root",
        )
    if resolved.exists():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git clone destination already exists",
        )
    parent = resolved.parent
    if not parent.exists() or not parent.is_dir():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git clone destination parent must exist",
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


def _resolve_workspace_cwd(value: str, *, workspace_root: Path) -> Path:
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
            "git cwd is unavailable",
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
    for section in _git_config_section_names(config):
        if _git_config_section_is_dangerous(section):
            _deny_unsafe_git_config("Git repository configuration is unsafe")
    for entry in _git_config_entries(config):
        if _git_config_entry_is_dangerous(entry):
            _deny_unsafe_git_config("Git repository configuration is unsafe")


def _git_config_entry_is_dangerous(entry: _GitConfigEntry) -> bool:
    key = entry.key.replace("-", "")
    return key in _DANGEROUS_CONFIG_KEY_NAMES or any(
        marker in key for marker in _DANGEROUS_CONFIG_KEY_MARKERS
    )


def _git_config_section_is_dangerous(section: str) -> bool:
    return section in _DANGEROUS_CONFIG_SECTION_NAMES or any(
        section.startswith(prefix)
        for prefix in _DANGEROUS_CONFIG_SECTION_PREFIXES
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


async def _validate_content_pathspec_scope(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
    repo_root: Path,
    *,
    git_dir: Path,
) -> None:
    if not pathspecs:
        return
    if request.command in (
        ShellGitCommandName.BLAME,
        ShellGitCommandName.GREP,
        ShellGitCommandName.STASH_APPLY,
    ):
        _validate_existing_file_content_pathspecs(pathspecs, repo_root)
        return
    if request.command is ShellGitCommandName.DIFF:
        mode = _diff_mode_option(request.options)
        if mode == "worktree":
            await _validate_diff_current_pathspecs(
                pathspecs,
                repo_root,
                git_dir=git_dir,
            )
        elif mode == "staged":
            await _validate_staged_diff_pathspecs(
                pathspecs,
                repo_root,
                git_dir=git_dir,
            )
        elif mode == "range":
            _validate_existing_file_content_pathspecs(pathspecs, repo_root)
        return
    if request.command is ShellGitCommandName.SHOW:
        if _show_mode_option(request.options) == "patch":
            _validate_existing_file_content_pathspecs(pathspecs, repo_root)
        return
    if request.command is ShellGitCommandName.STASH_SHOW:
        if _stash_show_mode_option(request.options) == "patch":
            _validate_existing_file_content_pathspecs(pathspecs, repo_root)


async def _validate_diff_current_pathspecs(
    pathspecs: tuple[str, ...],
    repo_root: Path,
    *,
    git_dir: Path,
) -> None:
    for value in pathspecs:
        await _validate_diff_current_pathspec(
            value,
            repo_root,
            git_dir=git_dir,
        )


async def _validate_staged_diff_pathspecs(
    pathspecs: tuple[str, ...],
    repo_root: Path,
    *,
    git_dir: Path,
) -> None:
    for value in pathspecs:
        await _validate_staged_diff_pathspec(
            value,
            repo_root,
            git_dir=git_dir,
        )


async def _validate_diff_current_pathspec(
    value: str,
    repo_root: Path,
    *,
    git_dir: Path,
) -> None:
    path = PurePosixPath(value)
    if path.as_posix() == ".":
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git content pathspec must name a file path",
        )
    resolved = (repo_root / Path(*path.parts)).resolve()
    if resolved.exists() and not resolved.is_file():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git content pathspec must name a file path",
        )
    if resolved.exists():
        return
    if await _current_index_proves_missing_file_pathspec(
        value,
        git_dir,
    ):
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        "Git content pathspec must name a deleted tracked file",
    )


async def _validate_staged_diff_pathspec(
    value: str,
    repo_root: Path,
    *,
    git_dir: Path,
) -> None:
    path = PurePosixPath(value)
    if path.as_posix() == ".":
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git content pathspec must name a file path",
        )
    resolved = (repo_root / Path(*path.parts)).resolve()
    if resolved.exists() and not resolved.is_file():
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            "Git content pathspec must name a file path",
        )
    if resolved.exists():
        return
    if await _head_tree_proves_file_pathspec(value, git_dir):
        return
    raise ShellGitPolicyDenied(
        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        "Git content pathspec must name a deleted tracked file",
    )


async def _current_index_proves_missing_file_pathspec(
    value: str,
    git_dir: Path,
) -> bool:
    index_path = git_dir / "index"
    index = await _read_bounded_regular_file(
        index_path,
        max_bytes=_GIT_INDEX_MAX_PROBE_BYTES,
    )
    if index is None:
        return False
    config = await _read_text(git_dir / "config")
    hash_size = _git_index_hash_size_from_config(config)
    if hash_size is None:
        return False
    return _git_index_data_has_exact_file_pathspec(
        index,
        value,
        hash_size=hash_size,
    )


async def _head_tree_proves_file_pathspec(
    value: str,
    git_dir: Path,
) -> bool:
    hash_size = await _git_object_hash_size(git_dir)
    if hash_size is None:
        return False
    commit_oid = await _head_commit_oid(git_dir, hash_size=hash_size)
    if commit_oid is None:
        return False
    commit = await _read_loose_git_object(
        git_dir,
        commit_oid,
        hash_size=hash_size,
    )
    if commit is None or commit[0] != "commit":
        return False
    tree_oid = _git_commit_tree_oid(commit[1], hash_size=hash_size)
    if tree_oid is None:
        return False
    target = value.encode("utf-8", "surrogateescape")
    return await _git_tree_has_file_pathspec(
        git_dir,
        tree_oid,
        target,
        hash_size=hash_size,
    )


def _git_index_data_has_exact_file_pathspec(
    index: bytes,
    value: str,
    *,
    hash_size: int,
) -> bool:
    parsed = _parse_git_index_data_with_hash_size(index, hash_size=hash_size)
    if parsed is None or parsed.sparse or parsed.split_index:
        return False
    target = value.encode("utf-8", "surrogateescape")
    return _git_index_entries_have_exact_file_pathspec(
        parsed.entries,
        target,
    )


async def _git_object_hash_size(git_dir: Path) -> int | None:
    config = await _read_text(git_dir / "config")
    return _git_index_hash_size_from_config(config)


def _git_index_hash_size_from_config(config: str) -> int | None:
    hash_size: int | None = _GIT_OBJECT_HASH_SIZE_SHA1
    for entry in _git_config_entries(config):
        if (
            entry.section == "extensions"
            and entry.subsection is None
            and entry.key == "objectformat"
        ):
            value = _git_config_unquoted_value(entry.value).lower()
            if value == _GIT_OBJECT_FORMAT_SHA1:
                hash_size = _GIT_OBJECT_HASH_SIZE_SHA1
            elif value == _GIT_OBJECT_FORMAT_SHA256:
                hash_size = _GIT_OBJECT_HASH_SIZE_SHA256
            else:
                hash_size = None
    return hash_size


def _git_config_unquoted_value(value: str) -> str:
    unquoted = _git_config_unquoted_comment_prefix(value).strip()
    if len(unquoted) < 2 or not (
        unquoted.startswith('"') and unquoted.endswith('"')
    ):
        return unquoted.strip()
    inner = unquoted[1:-1]
    result: list[str] = []
    escaped = False
    for character in inner:
        if escaped:
            if character in ('"', "\\"):
                result.append(character)
            else:
                result.append("\\")
                result.append(character)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        result.append(character)
    if escaped:
        result.append("\\")
    return "".join(result).strip()


async def _head_commit_oid(git_dir: Path, *, hash_size: int) -> str | None:
    head = await _read_bounded_regular_text(git_dir / "HEAD")
    if head is None:
        return None
    value = head.strip()
    if value.startswith("ref: "):
        ref = value.removeprefix("ref: ").strip()
        ref_path = _safe_head_ref_path(git_dir, ref)
        if ref_path is None:
            return None
        value = (await _read_bounded_regular_text(ref_path) or "").strip()
    return value.lower() if _is_git_object_oid(value, hash_size) else None


def _safe_head_ref_path(git_dir: Path, ref: str) -> Path | None:
    if not ref.startswith("refs/heads/") or _contains_control(ref):
        return None
    relative = PurePosixPath(ref)
    if (
        relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or any(part.endswith(".lock") for part in relative.parts)
    ):
        return None
    path = (git_dir / Path(*relative.parts)).resolve()
    return path if _is_relative_to(path, git_dir) else None


def _is_git_object_oid(value: str, hash_size: int) -> bool:
    if len(value) != hash_size * 2:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


async def _read_loose_git_object(
    git_dir: Path,
    oid: str,
    *,
    hash_size: int,
) -> tuple[str, bytes] | None:
    if not _is_git_object_oid(oid, hash_size):
        return None
    object_path = _safe_loose_git_object_path(git_dir, oid)
    if object_path is None:
        return None
    compressed = await _read_bounded_regular_file(
        object_path,
        max_bytes=_GIT_LOOSE_OBJECT_MAX_PROBE_BYTES,
    )
    if compressed is None:
        return None
    data = _zlib_decompress_bounded(
        compressed,
        max_bytes=_GIT_LOOSE_OBJECT_MAX_PROBE_BYTES,
    )
    if data is None:
        return None
    parts = _git_object_parts(data)
    if parts is None:
        return None
    object_type, payload = parts
    if not _git_loose_object_digest_matches(
        oid,
        object_type,
        payload,
        hash_size=hash_size,
    ):
        return None
    return parts


def _git_loose_object_digest_matches(
    oid: str,
    object_type: str,
    payload: bytes,
    *,
    hash_size: int,
) -> bool:
    data = f"{object_type} {len(payload)}\0".encode("ascii") + payload
    if hash_size == _GIT_OBJECT_HASH_SIZE_SHA1:
        actual = sha1(data).hexdigest()
    elif hash_size == _GIT_OBJECT_HASH_SIZE_SHA256:
        actual = sha256(data).hexdigest()
    else:
        return False
    return compare_digest(actual, oid.lower())


def _safe_loose_git_object_path(git_dir: Path, oid: str) -> Path | None:
    objects_dir = git_dir / "objects"
    object_prefix_dir = objects_dir / oid[:2]
    object_path = object_prefix_dir / oid[2:]
    if any(
        component.is_symlink()
        for component in (objects_dir, object_prefix_dir, object_path)
    ):
        return None
    try:
        resolved_objects_dir = objects_dir.resolve()
        resolved_object_path = object_path.resolve()
    except (OSError, RuntimeError):
        return None
    if not _is_relative_to(resolved_object_path, resolved_objects_dir):
        return None
    return object_path


def _zlib_decompress_bounded(data: bytes, *, max_bytes: int) -> bytes | None:
    decompressor = zlib_decompressobj()
    try:
        result = decompressor.decompress(data, max_bytes + 1)
        if decompressor.unconsumed_tail:
            return None
        result += decompressor.flush()
    except ZlibError:
        return None
    if (
        len(result) > max_bytes
        or not decompressor.eof
        or decompressor.unused_data
    ):
        return None
    return result


def _git_object_parts(data: bytes) -> tuple[str, bytes] | None:
    header_end = data.find(b"\x00")
    if header_end < 0:
        return None
    header = data[:header_end]
    header_parts = header.split(b" ", maxsplit=1)
    if len(header_parts) != 2:
        return None
    try:
        object_type = header_parts[0].decode("ascii", errors="strict")
        size_text = header_parts[1].decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None
    if (
        not size_text.isdecimal()
        or len(size_text) > _GIT_OBJECT_SIZE_MAX_DIGITS
    ):
        return None
    payload = data[header_end + 1 :]
    if int(size_text) != len(payload):
        return None
    return object_type, payload


def _git_commit_tree_oid(commit: bytes, *, hash_size: int) -> str | None:
    for line in commit.splitlines():
        if not line:
            return None
        if not line.startswith(b"tree "):
            continue
        try:
            value = line.removeprefix(b"tree ").decode(
                "ascii",
                errors="strict",
            )
        except UnicodeDecodeError:
            return None
        return value.lower() if _is_git_object_oid(value, hash_size) else None
    return None


async def _git_tree_has_file_pathspec(
    git_dir: Path,
    tree_oid: str,
    target: bytes,
    *,
    hash_size: int,
) -> bool:
    parts = target.split(b"/")
    if not parts or any(not part for part in parts):
        return False
    oid = tree_oid
    found = False
    for index, part in enumerate(parts):
        tree = await _read_loose_git_object(
            git_dir,
            oid,
            hash_size=hash_size,
        )
        if tree is None or tree[0] != "tree":
            return False
        entry = _git_tree_entry(tree[1], part, hash_size=hash_size)
        if entry is None:
            return False
        mode, raw_oid = entry
        final = index == len(parts) - 1
        if final:
            found = mode in _GIT_INDEX_FILE_MODES
            break
        if mode != _GIT_TREE_DIRECTORY_MODE:
            return False
        oid = raw_oid.hex()
    return found


def _git_tree_entry(
    tree: bytes,
    target_name: bytes,
    *,
    hash_size: int,
) -> tuple[int, bytes] | None:
    offset = 0
    while offset < len(tree):
        mode_end = tree.find(b" ", offset)
        if mode_end < 0:
            return None
        name_end = tree.find(b"\x00", mode_end + 1)
        if name_end < 0:
            return None
        oid_offset = name_end + 1
        next_offset = oid_offset + hash_size
        if next_offset > len(tree):
            return None
        mode_bytes = tree[offset:mode_end]
        if not mode_bytes or any(
            byte not in b"01234567" for byte in mode_bytes
        ):
            return None
        name = tree[mode_end + 1 : name_end]
        if name == target_name:
            return (
                int(mode_bytes.decode("ascii"), 8),
                tree[oid_offset:next_offset],
            )
        offset = next_offset
    return None


async def _read_bounded_regular_text(path: Path) -> str | None:
    data = await _read_bounded_regular_file(
        path,
        max_bytes=_GIT_LOOSE_OBJECT_MAX_PROBE_BYTES,
    )
    if data is None:
        return None
    return data.decode("utf-8", errors="replace")


async def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
) -> bytes | None:
    if path.is_symlink():
        return None
    try:
        metadata = await _inspect_path(path)
    except OSError:
        return None
    if not metadata.is_file or metadata.size > max_bytes:
        return None
    try:
        data = await _read_bytes(path)
    except OSError:
        return None
    return data if len(data) <= max_bytes else None


def _parse_git_index_data_with_hash_size(
    index: bytes,
    *,
    hash_size: int,
) -> _GitIndexData | None:
    if (
        hash_size
        not in (
            _GIT_OBJECT_HASH_SIZE_SHA1,
            _GIT_OBJECT_HASH_SIZE_SHA256,
        )
        or len(index) < 12 + hash_size
        or index[:4] != _GIT_INDEX_SIGNATURE
        or not _git_index_checksum_is_valid(index, hash_size=hash_size)
    ):
        return None
    version = int.from_bytes(index[4:8], "big")
    if version not in _GIT_INDEX_SUPPORTED_VERSIONS:
        return None
    content_end = len(index) - hash_size
    count = int.from_bytes(index[8:12], "big")
    entries: list[_GitIndexEntry] = []
    offset = 12
    for _ in range(count):
        entry = _git_index_entry(
            index,
            offset,
            content_end=content_end,
            hash_size=hash_size,
            version=version,
        )
        if entry is None:
            return None
        mode, path, offset = entry
        entries.append(_GitIndexEntry(mode=mode, path=path))
    extensions = _git_index_extensions(
        index,
        offset,
        content_end,
    )
    if extensions is None:
        return None
    split_index, sparse = extensions
    return _GitIndexData(
        entries=tuple(entries),
        hash_size=hash_size,
        split_index=split_index,
        sparse=sparse,
    )


def _git_index_checksum_is_valid(index: bytes, *, hash_size: int) -> bool:
    payload = index[:-hash_size]
    expected = index[-hash_size:]
    if hash_size == _GIT_OBJECT_HASH_SIZE_SHA256:
        actual = sha256(payload).digest()
    else:
        actual = sha1(payload).digest()
    return compare_digest(actual, expected)


def _git_index_entry(
    index: bytes,
    offset: int,
    *,
    content_end: int,
    hash_size: int,
    version: int,
) -> tuple[int, bytes, int] | None:
    flags_offset = offset + 40 + hash_size
    if flags_offset + 2 > content_end:
        return None
    mode = int.from_bytes(index[offset + 24 : offset + 28], "big")
    flags = int.from_bytes(index[flags_offset : flags_offset + 2], "big")
    path_offset = flags_offset + 2
    if flags & 0x4000:
        if version < 3:
            return None
        if path_offset + 2 > content_end:
            return None
        path_offset += 2
    path_end = index.find(b"\x00", path_offset, content_end)
    if path_end < 0:
        return None
    next_offset = offset + _padded_git_index_entry_size(
        path_end + 1 - offset,
    )
    if next_offset > content_end:
        return None
    return (
        mode,
        index[path_offset:path_end],
        next_offset,
    )


def _git_index_extensions(
    index: bytes,
    offset: int,
    extension_end: int,
) -> tuple[bool, bool] | None:
    split_index = False
    sparse = False
    while offset < extension_end:
        if offset + 8 > extension_end:
            return None
        signature = index[offset : offset + 4]
        size = int.from_bytes(index[offset + 4 : offset + 8], "big")
        data_offset = offset + 8
        next_offset = data_offset + size
        if next_offset > extension_end:
            return None
        if signature == _GIT_INDEX_LINK_EXTENSION:
            if split_index:
                return None
            split_index = True
        elif signature == _GIT_INDEX_SPARSE_EXTENSION:
            sparse = True
        elif signature[:1].islower():
            return None
        offset = next_offset
    return split_index, sparse


def _git_index_entries_have_exact_file_pathspec(
    entries: tuple[_GitIndexEntry, ...],
    target: bytes,
) -> bool:
    return any(
        _git_index_entry_matches_file_pathspec(entry, target)
        for entry in entries
    )


def _git_index_entry_matches_file_pathspec(
    entry: _GitIndexEntry,
    target: bytes,
) -> bool:
    return entry.path == target and entry.mode in _GIT_INDEX_FILE_MODES


def _padded_git_index_entry_size(size: int) -> int:
    remainder = size % 8
    return size if remainder == 0 else size + (8 - remainder)


def _validate_existing_file_content_pathspecs(
    pathspecs: tuple[str, ...],
    repo_root: Path,
) -> None:
    for value in pathspecs:
        _validate_existing_file_content_pathspec(value, repo_root)


def _validate_existing_file_content_pathspec(
    value: str,
    repo_root: Path,
) -> None:
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
    if command in (
        ShellGitCommandName.FETCH,
        ShellGitCommandName.PULL,
        ShellGitCommandName.PUSH,
        ShellGitCommandName.CLONE,
        ShellGitCommandName.REMOTE_LIST,
        ShellGitCommandName.REMOTE_ADD,
        ShellGitCommandName.REMOTE_SET_URL,
        ShellGitCommandName.REMOTE_REMOVE,
        ShellGitCommandName.REMOTE_RENAME,
        ShellGitCommandName.SUBMODULE_UPDATE,
    ):
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
    *,
    workspace_root: Path | None = None,
    git_dir: Path | None = None,
    allow_hidden: bool = False,
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
    if request.command is ShellGitCommandName.FETCH:
        assert git_dir is not None, "fetch requires a repository"
        return _fetch_argv(request, git_dir, settings)
    if request.command is ShellGitCommandName.PULL:
        return _pull_argv(request, settings)
    if request.command is ShellGitCommandName.PUSH:
        return _push_argv(request, settings)
    if request.command is ShellGitCommandName.CLONE:
        assert workspace_root is not None, "clone requires a workspace"
        return _clone_argv(
            request,
            workspace_root,
            settings,
            allow_hidden=allow_hidden,
        )
    if request.command is ShellGitCommandName.REMOTE_LIST:
        return _remote_list_argv(request)
    if request.command is ShellGitCommandName.REMOTE_ADD:
        return _remote_add_argv(request)
    if request.command is ShellGitCommandName.REMOTE_SET_URL:
        return _remote_set_url_argv(request)
    if request.command is ShellGitCommandName.REMOTE_REMOVE:
        return _remote_remove_argv(request)
    if request.command is ShellGitCommandName.REMOTE_RENAME:
        return _remote_rename_argv(request)
    assert request.command is ShellGitCommandName.SUBMODULE_UPDATE
    return _submodule_update_argv(request, pathspecs)


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
    mode = _diff_mode_option(request.options)
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
    requires_pathspecs = False
    match mode:
        case "worktree":
            _reject_revisions_for_mode(base_revision, head_revision, mode)
            requires_pathspecs = True
            argv.append("--patch")
        case "staged":
            _reject_revisions_for_mode(base_revision, head_revision, mode)
            requires_pathspecs = True
            argv.extend(("--cached", "--patch"))
        case "range":
            if base_revision is None or head_revision is None:
                raise ShellGitPolicyDenied(
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                    "git diff range mode requires base and head revisions",
                )
            requires_pathspecs = True
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
    if requires_pathspecs:
        _require_pathspecs_for_content(pathspecs, "git diff")
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _diff_mode_option(options: Mapping[str, object]) -> str:
    return _string_option(
        options,
        "mode",
        default_value="worktree",
        allowed_values=("worktree", "staged", "range", "stat", "name_only"),
        message="git diff mode is unsupported",
    )


def _show_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    revision = _required_string_option(
        request.options,
        "revision",
        message="git show revision must be a string",
    )
    mode = _show_mode_option(request.options)
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
            argv.append("--stat")
        case "patch":
            _require_pathspecs_for_content(pathspecs, "git show patch")
            argv.append("--patch")
    argv.append(_commit_revision(revision))
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _show_mode_option(options: Mapping[str, object]) -> str:
    return _string_option(
        options,
        "mode",
        default_value="summary",
        allowed_values=("summary", "stat", "patch"),
        message="git show mode is unsupported",
    )


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
    mode = _stash_show_mode_option(request.options)
    if mode == "patch":
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
    if pathspecs:
        argv.append("--")
        argv.extend(pathspecs)
    return tuple(argv)


def _stash_show_mode_option(options: Mapping[str, object]) -> str:
    return _string_option(
        options,
        "mode",
        default_value="stat",
        allowed_values=("stat", "patch"),
        message="git stash show mode is unsupported",
    )


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
    argv.extend((
        "--no-verify",
        "--no-gpg-sign",
        "--no-stat",
        "--no-edit",
        "--no-autostash",
    ))
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
    argv.extend((
        "--no-verify",
        "--no-gpg-sign",
        "--no-stat",
        "--no-autostash",
        "--no-rebase-merges",
        "--empty=stop",
        upstream,
    ))
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


def _fetch_argv(
    request: ShellGitCommandRequest,
    git_dir: Path,
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    remote = _remote_option(request.options)
    refspecs = _fetch_refspecs(request, remote, git_dir, settings)
    argv = _base_argv("fetch")
    argv.extend((
        "--no-tags",
        "--no-prune",
        "--no-recurse-submodules",
        "--no-write-fetch-head",
        remote,
    ))
    argv.extend(refspecs)
    return tuple(argv)


def _pull_argv(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    remote = _remote_option(request.options)
    branch = _required_string_option(
        request.options,
        "branch",
        message="git pull branch must be a string",
    )
    _validate_remote_ref(branch, settings, "pull branch")
    argv = _base_argv("pull")
    argv.extend((
        "--ff-only",
        "--no-verify",
        "--no-tags",
        "--no-prune",
        "--no-recurse-submodules",
        remote,
        branch,
    ))
    return tuple(argv)


def _push_argv(
    request: ShellGitCommandRequest,
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    remote = _remote_option(request.options)
    ref_type = _string_option(
        request.options,
        "ref_type",
        default_value="branch",
        allowed_values=_REMOTE_REF_TYPES,
        message="git push ref_type is unsupported",
    )
    ref_name = _required_string_option(
        request.options,
        "ref_name",
        message="git push ref_name must be a string",
    )
    _validate_remote_ref(ref_name, settings, f"push {ref_type}")
    selected_refspec = (
        f"refs/tags/{ref_name}:refs/tags/{ref_name}"
        if ref_type == "tag"
        else f"refs/heads/{ref_name}:refs/heads/{ref_name}"
    )
    argv = _base_argv("push")
    argv.extend(("--no-verify", "--porcelain"))
    argv.extend((remote, selected_refspec))
    return tuple(argv)


def _clone_argv(
    request: ShellGitCommandRequest,
    workspace_root: Path,
    settings: ShellGitToolSettings,
    *,
    allow_hidden: bool,
) -> tuple[str, ...]:
    url = _required_string_option(
        request.options,
        "url",
        message="git clone url must be a string",
    )
    destination = _required_string_option(
        request.options,
        "destination",
        message="git clone destination must be a string",
    )
    _validate_workspace_destination(
        destination,
        workspace_root=workspace_root,
        allow_hidden=allow_hidden,
    )
    branch = _optional_string_option(
        request.options,
        "branch",
        message="git clone branch must be a string",
    )
    argv = _base_argv("clone")
    argv.extend(("--no-tags", "--no-recurse-submodules", "--single-branch"))
    if branch is not None:
        _validate_remote_ref(branch, settings, "clone branch")
        argv.extend(("--branch", branch))
    argv.extend((url, destination))
    return tuple(argv)


def _remote_list_argv(request: ShellGitCommandRequest) -> tuple[str, ...]:
    assert isinstance(request, ShellGitCommandRequest)
    argv = _base_argv("remote")
    argv.append("--verbose")
    return tuple(argv)


def _remote_add_argv(request: ShellGitCommandRequest) -> tuple[str, ...]:
    name = _remote_name_option(request.options, "name")
    url = _required_string_option(
        request.options,
        "url",
        message="git remote add url must be a string",
    )
    return (*_base_argv("remote"), "add", name, url)


def _remote_set_url_argv(request: ShellGitCommandRequest) -> tuple[str, ...]:
    name = _remote_name_option(request.options, "name")
    url = _required_string_option(
        request.options,
        "url",
        message="git remote set-url url must be a string",
    )
    return (*_base_argv("remote"), "set-url", name, url)


def _remote_remove_argv(request: ShellGitCommandRequest) -> tuple[str, ...]:
    name = _remote_name_option(request.options, "name")
    return (*_base_argv("remote"), "remove", name)


def _remote_rename_argv(request: ShellGitCommandRequest) -> tuple[str, ...]:
    old_name = _remote_name_option(request.options, "old_name")
    new_name = _remote_name_option(request.options, "new_name")
    return (*_base_argv("remote"), "rename", old_name, new_name)


def _submodule_update_argv(
    request: ShellGitCommandRequest,
    pathspecs: tuple[str, ...],
) -> tuple[str, ...]:
    _require_pathspecs_for_mutation(pathspecs, "git submodule update")
    init = _bool_option(
        request.options,
        "init",
        default_value=False,
        message="git submodule update init must be boolean",
    )
    argv = _base_argv("submodule")
    argv.append("update")
    if init:
        argv.append("--init")
    argv.extend(("--no-recommend-shallow", "--depth=1", "--"))
    argv.extend(pathspecs)
    return tuple(argv)


def _fetch_refspecs(
    request: ShellGitCommandRequest,
    remote: str,
    git_dir: Path,
    settings: ShellGitToolSettings,
) -> tuple[str, ...]:
    assert isinstance(git_dir, Path)
    ref_type = _string_option(
        request.options,
        "ref_type",
        default_value="branch",
        allowed_values=_REMOTE_REF_TYPES,
        message="git fetch ref_type is unsupported",
    )
    ref_name = _required_string_option(
        request.options,
        "ref_name",
        message="git fetch ref_name must be a string",
    )
    _validate_remote_ref(ref_name, settings, f"fetch {ref_type}")
    if ref_type == "tag":
        return (f"refs/tags/{ref_name}:refs/tags/{ref_name}",)
    return (f"refs/heads/{ref_name}:refs/remotes/{remote}/{ref_name}",)


def _validate_remote_ref(
    value: str,
    settings: ShellGitToolSettings,
    field_name: str,
) -> None:
    if (
        not value
        or _contains_control(value)
        or "\x00" in value
        or value.startswith(("-", "/", "."))
        or value.endswith((".", "/", ".lock"))
        or value == "HEAD"
        or ":" in value
        or "@" in value
        or "\\" in value
        or "{" in value
        or "}" in value
        or ".." in value
        or " " in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(part.endswith(".lock") for part in value.split("/"))
        or any(character in value for character in ("*", "?", "[", "]"))
        or len(value.encode("utf-8")) > settings.max_revision_bytes
        or _HEX_REVISION_PATTERN.match(value)
        or not _REMOTE_REF_PATTERN.match(value)
    ):
        raise ShellGitPolicyDenied(
            ShellGitExecutionErrorCode.REVISION_DENIED,
            f"Git {field_name} form is unsupported",
        )


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
    if settings.allowed_remote_protocols:
        environment["GIT_ALLOW_PROTOCOL"] = ":".join(
            settings.allowed_remote_protocols
        )
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
    repo_root: Path | None,
    display_argv: tuple[str, ...],
    settings: ShellGitToolSettings,
    remote_urls: Mapping[str, str] | None = None,
) -> dict[str, object]:
    capability = shell_git_capability_for_request(request)
    metadata: dict[str, object] = {
        "git_tool_name": request.tool_name,
        "git_command": request.command.value,
        "git_capability_required": request.capability_required.value,
        "git_capability_used": capability.value,
        "git_effective_cwd": _display_path(workspace_root, effective_cwd),
        "git_repo_root": (
            None
            if repo_root is None
            else _display_path(workspace_root, repo_root)
        ),
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
        metadata.update({
            "git_mutation_attempted": True,
            "git_mutation_scope": "worktree",
        })
    elif capability is ShellGitCapability.HISTORY:
        metadata.update({
            "git_mutation_attempted": True,
            "git_mutation_scope": "history",
        })
    if request.command is ShellGitCommandName.GREP:
        metadata["exit_code_statuses"] = {
            1: ShellExecutionStatus.NO_MATCHES.value,
        }
    if capability is ShellGitCapability.REMOTE:
        metadata.update(
            git_remote_audit_metadata(
                request,
                settings=settings,
                remote_urls={} if remote_urls is None else remote_urls,
            )
        )
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


def git_remote_audit_metadata(
    request: ShellGitCommandRequest,
    *,
    settings: ShellGitToolSettings,
    remote_urls: Mapping[str, str] | None = None,
) -> dict[str, object]:
    url_values = _remote_audit_urls(
        request,
        {} if remote_urls is None else remote_urls,
    )
    primary_url = url_values[0] if url_values else None
    protocol, host, _ = (
        ("", "", "") if primary_url is None else _remote_url_parts(primary_url)
    )
    selected_refs = _selected_remote_refs(request)
    metadata: dict[str, object] = {
        "git_network_command_type": _remote_command_type(request.command),
        "git_remote_protocol": protocol or None,
        "git_remote_host": host or None,
        "git_remote_url": (
            None
            if primary_url is None
            else _redact_text(primary_url, settings)
        ),
        "git_remote_urls": tuple(
            _redact_text(url, settings) for url in url_values
        ),
        "git_selected_refs": selected_refs,
        "git_network_policy": {
            "allowed_remote_protocols": settings.allowed_remote_protocols,
            "allowed_remote_hosts": settings.allowed_remote_hosts,
            "terminal_prompts": "denied",
            "askpass": "denied",
            "credential_helpers": "denied",
            "custom_transports": "denied",
        },
        "git_credential_mode": settings.credential_policy,
        "git_remote_state_may_mutate": (
            request.command in _REMOTE_STATE_MUTATING_COMMANDS
        ),
        "git_local_remote_config_may_mutate": (
            request.command in _LOCAL_REMOTE_CONFIG_MUTATING_COMMANDS
        ),
        "git_network_may_run": request.command in _REMOTE_NETWORK_COMMANDS,
    }
    if request.command is ShellGitCommandName.SUBMODULE_UPDATE:
        submodule_parts = tuple(_remote_url_parts(url) for url in url_values)
        metadata.update({
            "git_submodule_protocols": tuple(
                protocol or None for protocol, _, _ in submodule_parts
            ),
            "git_submodule_hosts": tuple(
                host or None for _, host, _ in submodule_parts
            ),
            "git_submodule_urls": tuple(
                _redact_text(url, settings) for url in url_values
            ),
        })
    return metadata


def _remote_audit_urls(
    request: ShellGitCommandRequest,
    remote_urls: Mapping[str, str],
) -> tuple[str, ...]:
    url = request.options.get("url")
    urls: list[str] = []
    if isinstance(url, str):
        urls.append(url)
    for value in remote_urls.values():
        if value not in urls:
            urls.append(value)
    return tuple(urls)


def _remote_command_type(command: ShellGitCommandName) -> str:
    if command is ShellGitCommandName.CLONE:
        return "clone"
    if command in _LOCAL_REMOTE_CONFIG_MUTATING_COMMANDS:
        return "remote_management"
    if command is ShellGitCommandName.REMOTE_LIST:
        return "remote_inspection"
    if command is ShellGitCommandName.SUBMODULE_UPDATE:
        return "submodule_update"
    return "network"


def _selected_remote_refs(
    request: ShellGitCommandRequest,
) -> tuple[str, ...]:
    if request.command in (
        ShellGitCommandName.FETCH,
        ShellGitCommandName.PUSH,
    ):
        ref_type = request.options.get("ref_type", "branch")
        ref_name = request.options.get("ref_name")
        if ref_type == "tag" and isinstance(ref_name, str):
            return (f"refs/tags/{ref_name}",)
        if isinstance(ref_name, str):
            return (f"refs/heads/{ref_name}",)
    if request.command in (
        ShellGitCommandName.PULL,
        ShellGitCommandName.CLONE,
    ):
        branch = request.options.get("branch")
        if isinstance(branch, str):
            return (f"refs/heads/{branch}",)
    return ()


def redact_git_text(value: str, settings: ShellGitToolSettings) -> str:
    return _redact_text(value, settings)


def _redact_text(value: str, settings: ShellGitToolSettings) -> str:
    redacted = value
    if settings.redact_credentials:
        redacted = compile_pattern(
            r"([A-Za-z][A-Za-z0-9+.-]*://)([^/@\s]+)@"
        ).sub(r"\1[redacted]@", redacted)
        redacted = compile_pattern(
            r"([?&](?:access_)?token=)[^&#\s]+",
            flags=0,
        ).sub(r"\1[redacted]", redacted)
        redacted = compile_pattern(
            r"([?&](?:password|secret|credential)=)[^&#\s]+",
        ).sub(r"\1[redacted]", redacted)
    if settings.redact_remote_urls:
        redacted = compile_pattern(
            r"(?<![A-Za-z0-9+.-])([Ff][Ii][Ll][Ee]:///)"
            r"(?:[^?#\s]*)?(?:[?#][^\s]*)?"
        ).sub(r"\1[redacted]", redacted)
        redacted = compile_pattern(
            r"([A-Za-z][A-Za-z0-9+.-]*://)(?:[^/@\s]+@)?"
            r"([^/?#\s]+)(?:[/?#][^\s]*)?"
        ).sub(r"\1\2/[redacted]", redacted)
    if settings.redact_author_emails:
        redacted = _redact_author_email_fields(redacted)
    return redacted


def _redact_author_email_fields(value: str) -> str:
    redacted = _AUTHOR_SUMMARY_EMAIL_PATTERN.sub(
        r"\1[redacted]\2",
        value,
    )
    redacted = _AUTHOR_HEADER_EMAIL_PATTERN.sub(
        r"\1[redacted]\2",
        redacted,
    )
    return _AUTHOR_PORCELAIN_EMAIL_PATTERN.sub(
        r"\1[redacted]\2",
        redacted,
    )


def _display_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "[outside-workspace]"
    return "." if str(relative) == "." else relative.as_posix()


def _looks_like_bare_repository(path: Path) -> bool:
    return (
        _path_is_file_without_following_symlinks(path / "HEAD")
        and _path_is_dir_without_following_symlinks(path / "objects")
        and _path_is_dir_without_following_symlinks(path / "refs")
    )


def _path_is_file_without_following_symlinks(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _path_is_dir_without_following_symlinks(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir()


async def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return (await _read_bytes(path)).decode("utf-8", errors="replace")


def _config_declares_bare(config: str) -> bool:
    for entry in _git_config_entries(config):
        if (
            entry.section == "core"
            and entry.subsection is None
            and entry.key == "bare"
            and _git_config_bool_enabled(
                _git_config_unquoted_comment_prefix(entry.value)
            )
        ):
            return True
    return False


def _config_strictly_declares_bare(config: str) -> bool:
    bare_values: list[str] = []
    for entry in _git_config_entries(config):
        if (
            entry.section == "core"
            and entry.subsection is None
            and entry.key == "bare"
        ):
            bare_values.append(
                _git_config_unquoted_comment_prefix(entry.value)
            )
    return bool(bare_values) and all(
        _git_config_bool_explicitly_enabled(value) for value in bare_values
    )


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
