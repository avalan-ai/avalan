from ...types import assert_bool as _assert_bool
from ...types import assert_int as _assert_int
from ...types import assert_non_empty_string as _assert_non_empty_string
from ...types import assert_non_negative_int as _assert_non_negative_int
from ...types import (
    assert_optional_bounded_number as _assert_optional_bounded_number,
)
from ...types import (
    assert_optional_non_negative_int as _assert_optional_non_negative_int,
)
from ...types import assert_string_tuple as _assert_string_tuple

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, final

ShellGitExecutionMode = Literal[
    "disabled",
    "policy",
    "local",
    "sandbox",
    "container",
]
_SHELL_GIT_EXECUTION_MODES: tuple[ShellGitExecutionMode, ...] = (
    "disabled",
    "policy",
    "local",
    "sandbox",
    "container",
)


class ShellGitCapability(StrEnum):
    READ = "read"
    WORKTREE = "worktree"
    HISTORY = "history"
    REMOTE = "remote"


class ShellGitCommandName(StrEnum):
    STATUS = "status"
    REV_PARSE = "rev-parse"
    BRANCH = "branch"
    TAG = "tag"
    DESCRIBE = "describe"
    LS_FILES = "ls-files"
    LOG = "log"
    DIFF = "diff"
    SHOW = "show"
    BLAME = "blame"
    GREP = "grep"
    STASH_LIST = "stash-list"
    STASH_SHOW = "stash-show"
    ADD = "add"
    RESTORE = "restore"
    CHECKOUT = "checkout"
    SWITCH = "switch"
    RESET = "reset"
    RM = "rm"
    MV = "mv"
    STASH_PUSH = "stash-push"
    STASH_APPLY = "stash-apply"
    COMMIT = "commit"
    BRANCH_CREATE = "branch-create"
    BRANCH_DELETE = "branch-delete"
    BRANCH_RENAME = "branch-rename"
    TAG_CREATE = "tag-create"
    TAG_DELETE = "tag-delete"
    MERGE = "merge"
    REBASE = "rebase"
    CHERRY_PICK = "cherry-pick"
    REVERT = "revert"
    CLEAN = "clean"
    STASH_POP = "stash-pop"
    STASH_DROP = "stash-drop"
    FETCH = "fetch"
    PULL = "pull"
    PUSH = "push"
    CLONE = "clone"
    REMOTE_LIST = "remote-list"
    REMOTE_ADD = "remote-add"
    REMOTE_SET_URL = "remote-set-url"
    REMOTE_REMOVE = "remote-remove"
    REMOTE_RENAME = "remote-rename"
    SUBMODULE_UPDATE = "submodule-update"


class ShellGitExecutionStatus(StrEnum):
    SUCCESS = "success"
    POLICY_DENIED = "policy_denied"
    COMMAND_UNAVAILABLE = "command_unavailable"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ShellGitExecutionErrorCode(StrEnum):
    CAPABILITY_REQUIRED = "capability_required"
    COMMAND_DISABLED = "command_disabled"
    REPO_NOT_FOUND = "repo_not_found"
    REPO_BOUNDARY_DENIED = "repo_boundary_denied"
    BARE_REPO_DENIED = "bare_repo_denied"
    SUBMODULE_DENIED = "submodule_denied"
    ALTERNATE_DENIED = "alternate_denied"
    PATHSPEC_DENIED = "pathspec_denied"
    REVISION_DENIED = "revision_denied"
    REVISION_NOT_FOUND = "revision_not_found"
    AMBIGUOUS_REVISION = "ambiguous_revision"
    INVALID_OPTION = "invalid_option"
    UNSAFE_GIT_CONFIG = "unsafe_git_config"
    EXTERNAL_PROCESS_DENIED = "external_process_denied"
    OPTIONAL_LOCK_DENIED = "optional_lock_denied"
    CREDENTIAL_DENIED = "credential_denied"
    REMOTE_PROTOCOL_DENIED = "remote_protocol_denied"
    REMOTE_HOST_DENIED = "remote_host_denied"
    OUTPUT_TRUNCATED = "output_truncated"
    TIMEOUT = "timeout"
    NONZERO_EXIT = "nonzero_exit"
    COMMAND_UNAVAILABLE = "command_unavailable"


SHELL_GIT_READ_COMMANDS: tuple[ShellGitCommandName, ...] = (
    ShellGitCommandName.STATUS,
    ShellGitCommandName.REV_PARSE,
    ShellGitCommandName.BRANCH,
    ShellGitCommandName.TAG,
    ShellGitCommandName.DESCRIBE,
    ShellGitCommandName.LS_FILES,
    ShellGitCommandName.LOG,
    ShellGitCommandName.DIFF,
    ShellGitCommandName.SHOW,
    ShellGitCommandName.BLAME,
    ShellGitCommandName.GREP,
    ShellGitCommandName.STASH_LIST,
    ShellGitCommandName.STASH_SHOW,
)
SHELL_GIT_WORKTREE_COMMANDS: tuple[ShellGitCommandName, ...] = (
    ShellGitCommandName.ADD,
    ShellGitCommandName.RESTORE,
    ShellGitCommandName.CHECKOUT,
    ShellGitCommandName.SWITCH,
    ShellGitCommandName.RESET,
    ShellGitCommandName.RM,
    ShellGitCommandName.MV,
    ShellGitCommandName.STASH_PUSH,
    ShellGitCommandName.STASH_APPLY,
)
SHELL_GIT_HISTORY_COMMANDS: tuple[ShellGitCommandName, ...] = (
    ShellGitCommandName.COMMIT,
    ShellGitCommandName.BRANCH_CREATE,
    ShellGitCommandName.BRANCH_DELETE,
    ShellGitCommandName.BRANCH_RENAME,
    ShellGitCommandName.TAG_CREATE,
    ShellGitCommandName.TAG_DELETE,
    ShellGitCommandName.MERGE,
    ShellGitCommandName.REBASE,
    ShellGitCommandName.CHERRY_PICK,
    ShellGitCommandName.REVERT,
    ShellGitCommandName.CLEAN,
    ShellGitCommandName.STASH_POP,
    ShellGitCommandName.STASH_DROP,
)
SHELL_GIT_REMOTE_COMMANDS: tuple[ShellGitCommandName, ...] = (
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
)
SHELL_GIT_DEFAULT_ALLOWED_COMMANDS = SHELL_GIT_READ_COMMANDS
SHELL_GIT_COMMAND_CAPABILITIES: dict[
    ShellGitCommandName,
    ShellGitCapability,
] = {
    **{
        command: ShellGitCapability.READ for command in SHELL_GIT_READ_COMMANDS
    },
    **{
        command: ShellGitCapability.WORKTREE
        for command in SHELL_GIT_WORKTREE_COMMANDS
    },
    **{
        command: ShellGitCapability.HISTORY
        for command in SHELL_GIT_HISTORY_COMMANDS
    },
    **{
        command: ShellGitCapability.REMOTE
        for command in SHELL_GIT_REMOTE_COMMANDS
    },
}
SHELL_GIT_CAPABILITY_IDS: tuple[str, ...] = tuple(
    capability.value for capability in ShellGitCapability
)
SHELL_GIT_COMMAND_IDS: tuple[str, ...] = tuple(
    command.value for command in ShellGitCommandName
)
SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS: tuple[str, ...] = tuple(
    command.value for command in SHELL_GIT_DEFAULT_ALLOWED_COMMANDS
)
SHELL_GIT_TOOL_NAMES: tuple[str, ...] = tuple(
    f"git_{command.value.replace('-', '_')}" for command in ShellGitCommandName
)
SHELL_GIT_TOOL_COMMANDS: dict[str, ShellGitCommandName] = {
    f"git_{command.value.replace('-', '_')}": command
    for command in ShellGitCommandName
}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellGitCommandRequest:
    tool_name: str
    command: ShellGitCommandName
    capability_required: ShellGitCapability
    options: dict[str, object]
    pathspecs: tuple[str, ...] = ()
    cwd: str | None = None
    timeout_seconds: float | None = None
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.tool_name, "tool_name")
        assert isinstance(
            self.command,
            ShellGitCommandName,
        ), "command must be a shell Git command"
        assert isinstance(
            self.capability_required,
            ShellGitCapability,
        ), "capability_required must be a shell Git capability"
        assert isinstance(self.options, dict), "options must be a dictionary"
        _assert_string_tuple(self.pathspecs, "pathspecs")
        if self.cwd is not None:
            _assert_non_empty_string(self.cwd, "cwd")
        _assert_optional_bounded_number(
            self.timeout_seconds,
            "timeout_seconds",
            min_value=0,
            min_inclusive=False,
        )
        _assert_optional_non_negative_int(
            self.max_stdout_bytes,
            "max_stdout_bytes",
        )
        _assert_optional_non_negative_int(
            self.max_stderr_bytes,
            "max_stderr_bytes",
        )
        assert isinstance(self.metadata, dict), "metadata must be a dictionary"
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "pathspecs", tuple(self.pathspecs))
        object.__setattr__(self, "metadata", dict(self.metadata))


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellGitCommandResult:
    tool_name: str
    command: ShellGitCommandName
    display_argv: tuple[str, ...]
    effective_cwd: str
    resolved_repo_root: str | None
    capability_required: ShellGitCapability
    capability_used: ShellGitCapability | None
    execution_mode: ShellGitExecutionMode
    status: ShellGitExecutionStatus
    exit_code: int | None
    stdout_snippet: str
    stderr_snippet: str
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False
    duration_ms: int = 0
    error_code: ShellGitExecutionErrorCode | None = None
    error_message: str | None = None
    audit_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.tool_name, "tool_name")
        assert isinstance(
            self.command,
            ShellGitCommandName,
        ), "command must be a shell Git command"
        _assert_string_tuple(self.display_argv, "display_argv")
        _assert_non_empty_string(self.effective_cwd, "effective_cwd")
        if self.resolved_repo_root is not None:
            _assert_non_empty_string(
                self.resolved_repo_root,
                "resolved_repo_root",
            )
        assert isinstance(
            self.capability_required,
            ShellGitCapability,
        ), "capability_required must be a shell Git capability"
        if self.capability_used is not None:
            assert isinstance(
                self.capability_used,
                ShellGitCapability,
            ), "capability_used must be a shell Git capability"
        assert (
            self.execution_mode in _SHELL_GIT_EXECUTION_MODES
        ), "execution_mode must be a shell Git execution mode"
        assert isinstance(
            self.status,
            ShellGitExecutionStatus,
        ), "status must be a shell Git execution status"
        if self.exit_code is not None:
            _assert_int(self.exit_code, "exit_code")
        assert isinstance(
            self.stdout_snippet,
            str,
        ), "stdout_snippet must be a string"
        assert isinstance(
            self.stderr_snippet,
            str,
        ), "stderr_snippet must be a string"
        _assert_non_negative_int(self.stdout_bytes, "stdout_bytes")
        _assert_non_negative_int(self.stderr_bytes, "stderr_bytes")
        _assert_bool(self.stdout_truncated, "stdout_truncated")
        _assert_bool(self.stderr_truncated, "stderr_truncated")
        _assert_bool(self.timed_out, "timed_out")
        _assert_bool(self.cancelled, "cancelled")
        _assert_non_negative_int(self.duration_ms, "duration_ms")
        if self.error_code is not None:
            assert isinstance(
                self.error_code,
                ShellGitExecutionErrorCode,
            ), "error_code must be a shell Git execution error code"
        if self.error_message is not None:
            _assert_non_empty_string(self.error_message, "error_message")
        assert isinstance(
            self.audit_metadata,
            dict,
        ), "audit_metadata must be a dictionary"
        object.__setattr__(self, "display_argv", tuple(self.display_argv))
        object.__setattr__(self, "audit_metadata", dict(self.audit_metadata))


class ShellGitFormattedResult(str):
    git_result: ShellGitCommandResult

    def __new__(
        cls,
        value: str,
        git_result: ShellGitCommandResult,
    ) -> "ShellGitFormattedResult":
        assert isinstance(value, str), "value must be a string"
        assert isinstance(
            git_result,
            ShellGitCommandResult,
        ), "git_result must be a shell Git command result"
        formatted = str.__new__(cls, value)
        formatted.git_result = git_result
        return formatted

    def __copy__(self) -> "ShellGitFormattedResult":
        return self

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> "ShellGitFormattedResult":
        return self

    def __reduce__(
        self,
    ) -> tuple[
        type["ShellGitFormattedResult"],
        tuple[str, ShellGitCommandResult],
    ]:
        return (self.__class__, (str(self), self.git_result))
