from collections.abc import Awaitable, Callable
from os import devnull, environ
from pathlib import Path, PurePosixPath
from shutil import which
from subprocess import CompletedProcess, run
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, skipIf

from avalan.entities import ToolCallContext, ToolExecutionStreamEvent
from avalan.tool.shell import ShellGitFormattedResult
from avalan.tool.shell.entities import (
    ExecutionResult,
    ExecutionSpec,
    ShellExecutionStatus,
    ShellOutputKind,
)
from avalan.tool.shell.git import (
    SHELL_GIT_COMMAND_CAPABILITIES,
    ShellGitCapability,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitExecutionErrorCode,
    ShellGitExecutionStatus,
    ShellGitPolicyDenied,
)
from avalan.tool.shell.git_policy import (
    GitExecutionPolicy,
    _argv_for_request,
    _validate_mv_destination_pathspec,
)
from avalan.tool.shell.settings import ShellGitToolSettings, ShellToolSettings
from avalan.tool.shell.toolset import ShellToolSet

_GIT_BINARY = which("git")
_PHASE5_COMMANDS = (
    "add",
    "restore",
    "checkout",
    "switch",
    "reset",
    "rm",
    "mv",
    "stash-push",
    "stash-apply",
)
_PHASE5_TOOL_NAMES = [
    "shell.git_add",
    "shell.git_restore",
    "shell.git_checkout",
    "shell.git_switch",
    "shell.git_reset",
    "shell.git_rm",
    "shell.git_mv",
    "shell.git_stash_push",
    "shell.git_stash_apply",
]
_HISTORY_TOOL_NAMES = {
    "shell.git_commit",
    "shell.git_branch_create",
    "shell.git_branch_delete",
    "shell.git_branch_rename",
    "shell.git_tag_create",
    "shell.git_tag_delete",
    "shell.git_merge",
    "shell.git_rebase",
    "shell.git_cherry_pick",
    "shell.git_revert",
    "shell.git_clean",
    "shell.git_stash_pop",
    "shell.git_stash_drop",
}
_REMOTE_TOOL_NAMES = {
    "shell.git_fetch",
    "shell.git_pull",
    "shell.git_push",
    "shell.git_clone",
    "shell.git_remote_list",
    "shell.git_remote_add",
    "shell.git_remote_set_url",
    "shell.git_remote_remove",
    "shell.git_remote_rename",
    "shell.git_submodule_update",
}
_GitToolCallable = Callable[..., Awaitable[str]]
_PolicyFactory = Callable[[Path], GitExecutionPolicy]


class GitWorktreePolicyPhase5Test(IsolatedAsyncioTestCase):
    async def test_worktree_commands_build_fixed_argv(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.ADD,
                        options={"mode": "normal"},
                        pathspecs=("src/app.py",),
                    ),
                    (*_git_prefix(), "add", "--", "src/app.py"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.ADD,
                        options={"mode": "intent_to_add"},
                        pathspecs=("new.txt",),
                    ),
                    (
                        *_git_prefix(),
                        "add",
                        "--intent-to-add",
                        "--",
                        "new.txt",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.RESTORE,
                        options={
                            "source_revision": "HEAD",
                            "staged": True,
                            "worktree": True,
                        },
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "restore",
                        "--no-overlay",
                        "--staged",
                        "--worktree",
                        "--source",
                        "HEAD",
                        "--",
                        "src/app.py",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.CHECKOUT,
                        options={"target": "HEAD"},
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "checkout",
                        "--no-recurse-submodules",
                        "HEAD",
                        "--",
                        "src/app.py",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.SWITCH,
                        options={"branch": "side"},
                    ),
                    (
                        *_git_prefix(),
                        "switch",
                        "--no-guess",
                        "--no-recurse-submodules",
                        "side",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.RESET,
                        options={"mode": "paths"},
                        pathspecs=("new.txt",),
                    ),
                    (*_git_prefix(), "reset", "--", "new.txt"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.RM,
                        options={"cached": True},
                        pathspecs=("new.txt",),
                    ),
                    (*_git_prefix(), "rm", "--cached", "--", "new.txt"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.MV,
                        options={
                            "source": "old.txt",
                            "destination": "new.txt",
                        },
                        pathspecs=("old.txt", "new.txt"),
                    ),
                    (*_git_prefix(), "mv", "--", "old.txt", "new.txt"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.MV,
                        options={
                            "source": "old.txt",
                            "destination": "README.md",
                        },
                        pathspecs=("old.txt", "README.md"),
                    ),
                    (*_git_prefix(), "mv", "--", "old.txt", "README.md"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_PUSH,
                        options={
                            "message": "save phase5",
                            "include_untracked": True,
                        },
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "stash",
                        "push",
                        "--message",
                        "save phase5",
                        "--include-untracked",
                        "--",
                        "src/app.py",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_APPLY,
                        options={"stash": "stash@{0}"},
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "restore",
                        "--no-overlay",
                        "--worktree",
                        "--source",
                        "stash@{0}",
                        "--",
                        "src/app.py",
                    ),
                ),
            )

            for request, expected_argv in cases:
                with self.subTest(command=request.command.value):
                    spec = await policy.normalize(request)

                    self.assertEqual(spec.argv, expected_argv)
                    self.assertEqual(spec.display_argv, expected_argv)
                    self.assertEqual(spec.output_kind, ShellOutputKind.TEXT)
                    self.assertEqual(
                        spec.metadata["git_capability_used"],
                        ShellGitCapability.WORKTREE.value,
                    )
                    self.assertEqual(
                        spec.metadata["git_mutation_attempted"],
                        True,
                    )
                    self.assertEqual(
                        spec.metadata["git_mutation_scope"],
                        "worktree",
                    )
                    self.assertEqual(
                        spec.metadata["git_request_pathspecs"],
                        request.pathspecs,
                    )

    async def test_worktree_unsafe_forms_fail_closed(self) -> None:
        cases = (
            (
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "patch"},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=(".env",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=("credentials",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.RESTORE,
                    options={"staged": False, "worktree": False},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.RESTORE,
                    options={"source_revision": "HEAD:src/app.py"},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.CHECKOUT,
                    options={"mode": "branch", "target": "side"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.CHECKOUT,
                    options={"mode": "branch", "target": "feature/new"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.CHECKOUT,
                    options={"target": "HEAD"},
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.CHECKOUT,
                    options={"mode": "branch"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.SWITCH,
                    options={"branch": "side", "create": True},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.SWITCH,
                    options={"branch": "HEAD"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.RESET,
                    options={"mode": "hard"},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.RESET,
                    options={"mode": "paths"},
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.RM,
                    options={"force": True},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.RM,
                    options={"recursive": True},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.MV,
                    options={
                        "source": "old.txt",
                        "destination": "new.txt",
                    },
                    pathspecs=("old.txt", "other.txt"),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.MV,
                    options={
                        "source": "old.txt",
                        "destination": "new.txt",
                    },
                    pathspecs=("old.txt",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_PUSH,
                    options={"message": "save"},
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_APPLY,
                    options={"stash": "stash@{999}"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_APPLY,
                    options={"stash": "stash@{0}"},
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_APPLY,
                    options={"stash": "stash@{0}", "index": False},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_APPLY,
                    options={"stash": "stash@{0}"},
                    pathspecs=("src",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
        )

        for request, error_code in cases:
            with self.subTest(
                command=request.command.value,
                options=request.options,
                pathspecs=request.pathspecs,
            ):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(_policy(root), request)
                    self.assertEqual(error.error_code, error_code)

    async def test_worktree_mutation_pathspecs_are_file_scoped(
        self,
    ) -> None:
        command_cases: tuple[
            tuple[str, ShellGitCommandName, dict[str, object]],
            ...,
        ] = (
            (
                "add",
                ShellGitCommandName.ADD,
                {"mode": "normal"},
            ),
            (
                "restore",
                ShellGitCommandName.RESTORE,
                {},
            ),
            (
                "checkout",
                ShellGitCommandName.CHECKOUT,
                {"target": "HEAD"},
            ),
            (
                "reset",
                ShellGitCommandName.RESET,
                {"mode": "paths"},
            ),
            (
                "rm",
                ShellGitCommandName.RM,
                {"cached": False},
            ),
            (
                "stash_push",
                ShellGitCommandName.STASH_PUSH,
                {"message": "save"},
            ),
            (
                "stash_apply",
                ShellGitCommandName.STASH_APPLY,
                {"stash": "stash@{0}"},
            ),
        )

        for pathspec in ("src", "."):
            for label, command, options in command_cases:
                with self.subTest(command=label, pathspec=pathspec):
                    with TemporaryDirectory() as workspace:
                        root = Path(workspace)
                        _write_minimal_git_repo(root / "repo")
                        error = await _policy_error(
                            _policy(root),
                            _request(
                                command=command,
                                options=options,
                                pathspecs=(pathspec,),
                            ),
                        )

                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

    async def test_git_mv_source_and_destination_are_file_scoped(
        self,
    ) -> None:
        cases: tuple[
            tuple[str, dict[str, object], tuple[str, str]],
            ...,
        ] = (
            (
                "source_directory",
                {"source": "src", "destination": "new.txt"},
                ("src", "new.txt"),
            ),
            (
                "source_dot",
                {"source": ".", "destination": "new.txt"},
                (".", "new.txt"),
            ),
            (
                "source_missing",
                {"source": "missing.txt", "destination": "new.txt"},
                ("missing.txt", "new.txt"),
            ),
            (
                "destination_dot",
                {"source": "old.txt", "destination": "."},
                ("old.txt", "."),
            ),
            (
                "destination_directory",
                {"source": "old.txt", "destination": "src"},
                ("old.txt", "src"),
            ),
            (
                "destination_hidden",
                {"source": "old.txt", "destination": "src/.env"},
                ("old.txt", "src/.env"),
            ),
            (
                "destination_sensitive_parent",
                {"source": "old.txt", "destination": "credentials/new.txt"},
                ("old.txt", "credentials/new.txt"),
            ),
            (
                "destination_parent_file",
                {"source": "old.txt", "destination": "README.md/new.txt"},
                ("old.txt", "README.md/new.txt"),
            ),
        )

        for label, options, pathspecs in cases:
            with self.subTest(label=label):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(
                        _policy(root),
                        _request(
                            command=ShellGitCommandName.MV,
                            options=options,
                            pathspecs=pathspecs,
                        ),
                    )

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                )

    def test_git_mv_internal_guards_fail_closed(self) -> None:
        request = _request(
            command=ShellGitCommandName.MV,
            options={"source": "old.txt", "destination": "new.txt"},
            pathspecs=("old.txt", "new.txt"),
        )
        settings = ShellGitToolSettings()

        for pathspecs in (
            ("old.txt",),
            ("old.txt", "other.txt"),
        ):
            with self.subTest(pathspecs=pathspecs):
                with self.assertRaises(ShellGitPolicyDenied) as raised:
                    _argv_for_request(request, pathspecs, settings)

                self.assertEqual(
                    raised.exception.error_code,
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")

            with self.assertRaises(ShellGitPolicyDenied) as raised:
                _validate_mv_destination_pathspec(
                    PurePosixPath("../outside/new.txt"),
                    repo,
                )

        self.assertEqual(
            raised.exception.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )

    async def test_worktree_pathspecs_reject_symlink_escapes(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_text("secret\n")
            try:
                (repo / "linked").symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")

            policy_error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=("linked/secret.txt",),
                ),
            )

        self.assertEqual(
            policy_error.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )

    async def test_worktree_capability_cannot_run_history_or_remote_policy(
        self,
    ) -> None:
        commands = (
            ShellGitCommandName.COMMIT,
            ShellGitCommandName.BRANCH_DELETE,
            ShellGitCommandName.TAG_DELETE,
            ShellGitCommandName.FETCH,
            ShellGitCommandName.PULL,
            ShellGitCommandName.PUSH,
            ShellGitCommandName.CLONE,
            ShellGitCommandName.SUBMODULE_UPDATE,
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(
                root,
                allowed_commands=(
                    *_PHASE5_COMMANDS,
                    "commit",
                    "branch-delete",
                    "tag-delete",
                    "fetch",
                    "pull",
                    "push",
                    "clone",
                    "submodule-update",
                ),
                allowed_remote_hosts=("github.com",),
                allow_submodule_update=True,
            )

            for command in commands:
                with self.subTest(command=command.value):
                    error = await _policy_error(
                        policy,
                        _request(
                            command=command,
                            options=_disabled_command_options(command),
                        ),
                    )

                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
                    )

    async def test_worktree_pathspec_boundaries_are_enforced(self) -> None:
        cases: tuple[
            tuple[
                _PolicyFactory,
                ShellGitCommandRequest,
                ShellGitExecutionErrorCode | None,
            ],
            ...,
        ] = (
            (
                lambda root: _policy(root, max_pathspecs=1),
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=("src/app.py", "README.md"),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                lambda root: _policy(root, max_pathspec_bytes=3),
                _request(
                    command=ShellGitCommandName.RM,
                    options={"cached": True},
                    pathspecs=("README.md",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                lambda root: _policy(root, max_stdout_bytes=21),
                _request(
                    command=ShellGitCommandName.STASH_APPLY,
                    options={"stash": "stash@{0}"},
                    pathspecs=("src/app.py",),
                    max_stdout_bytes=999,
                ),
                None,
            ),
            (
                lambda root: _policy(
                    root,
                    default_timeout_seconds=5,
                    max_timeout_seconds=7,
                    max_stderr_bytes=13,
                ),
                _request(
                    command=ShellGitCommandName.RESET,
                    options={"mode": "paths"},
                    pathspecs=("src/app.py",),
                    timeout_seconds=99,
                    max_stderr_bytes=99,
                ),
                None,
            ),
        )

        for policy_factory, request, error_code in cases:
            with self.subTest(
                command=request.command.value,
            ):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    policy = policy_factory(root)
                    if error_code is not None:
                        error = await _policy_error(policy, request)
                        self.assertEqual(error.error_code, error_code)
                    else:
                        spec = await policy.normalize(request)
                        if request.max_stdout_bytes is not None:
                            self.assertEqual(spec.max_stdout_bytes, 21)
                        if request.timeout_seconds is not None:
                            self.assertEqual(spec.timeout_seconds, 7)
                            self.assertEqual(spec.max_stderr_bytes, 13)

    async def test_worktree_option_types_fail_closed_and_defaults_apply(
        self,
    ) -> None:
        cases = (
            (
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": 1},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.SWITCH,
                    options={},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.RM,
                    options={"cached": "yes"},
                    pathspecs=("README.md",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_PUSH,
                    options={"message": 1},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.MV,
                    options={"destination": "new.txt"},
                    pathspecs=("old.txt", "new.txt"),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_APPLY,
                    options={"stash": "stash@{abc}"},
                    pathspecs=("src/app.py",),
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
        )

        for request, error_code in cases:
            with self.subTest(
                command=request.command.value,
                options=request.options,
            ):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(_policy(root), request)
                    self.assertEqual(error.error_code, error_code)

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            restore = await policy.normalize(
                _request(
                    command=ShellGitCommandName.RESTORE,
                    options={},
                    pathspecs=("src/app.py",),
                )
            )
            checkout = await policy.normalize(
                _request(
                    command=ShellGitCommandName.CHECKOUT,
                    options={},
                    pathspecs=("src/app.py",),
                )
            )

        self.assertEqual(
            restore.argv,
            (
                *_git_prefix(),
                "restore",
                "--no-overlay",
                "--worktree",
                "--",
                "src/app.py",
            ),
        )
        self.assertEqual(
            checkout.argv,
            (
                *_git_prefix(),
                "checkout",
                "--no-recurse-submodules",
                "--",
                "src/app.py",
            ),
        )

    async def test_worktree_argv_budgets_are_enforced(self) -> None:
        cases: tuple[
            tuple[_PolicyFactory, ShellGitExecutionErrorCode],
            ...,
        ] = (
            (
                lambda root: _policy(root, max_arguments=5),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                lambda root: _policy(root, max_argument_bytes=2),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                lambda root: _policy(root, max_command_bytes=10),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
        )

        for policy_factory, error_code in cases:
            with self.subTest(error_code=error_code.value):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(
                        policy_factory(root),
                        _request(
                            command=ShellGitCommandName.ADD,
                            options={"mode": "normal"},
                            pathspecs=("src/app.py",),
                        ),
                    )

                self.assertEqual(error.error_code, error_code)

    async def test_worktree_redacts_mutation_metadata_and_output(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                stdout="saved https://token@github.com/acme/repo\n",
                stderr="remote https://token@github.com/acme/repo\n",
                status=ShellExecutionStatus.COMPLETED,
                exit_code=0,
            )
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_stash_push",
                message="save https://token@github.com/acme/repo",
                paths=("src/app.py",),
            )

        formatted = str(result)
        audit_metadata = str(result.git_result.audit_metadata)
        self.assertNotIn("token", formatted)
        self.assertNotIn("token", audit_metadata)
        self.assertIn("https://github.com/[redacted]", formatted)
        self.assertIn("https://github.com/[redacted]", audit_metadata)

    async def test_worktree_repository_form_edges_are_enforced(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").unlink()

            spec = await _policy(root).normalize(
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=("src/app.py",),
                )
            )

        self.assertEqual(spec.metadata["git_repo_root"], "repo")

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").write_text("[core]\n\tbare = true\n")

            error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=("src/app.py",),
                ),
            )

        self.assertEqual(
            error.error_code, ShellGitExecutionErrorCode.BARE_REPO_DENIED
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (root / "common.git").mkdir()
            (repo / ".git" / "commondir").write_text("../common.git\n")

            linked_spec = await _policy(
                root,
                allow_linked_worktrees=True,
            ).normalize(
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=("src/app.py",),
                )
            )

        self.assertEqual(linked_spec.metadata["git_repo_root"], "repo")

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            outside = root.parent / "outside-common.git"
            (repo / ".git" / "commondir").write_text(str(outside))

            error = await _policy_error(
                _policy(root, allow_linked_worktrees=True),
                _request(
                    command=ShellGitCommandName.ADD,
                    options={"mode": "normal"},
                    pathspecs=("src/app.py",),
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

    async def test_worktree_denials_do_not_build_executor_specs(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor()
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_add",
                paths=("../outside",),
            )

        self.assertEqual(executor.calls, 0)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertIn("status: policy_denied", result)
        self.assertIn("error_code: pathspec_denied", result)
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_attempted"],
            True,
        )
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_scope"],
            "worktree",
        )

    async def test_worktree_successes_include_audit_metadata(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                stdout="updated\n",
                status=ShellExecutionStatus.COMPLETED,
                exit_code=0,
            )
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_add",
                paths=("src/app.py",),
            )

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        self.assertEqual(
            result.git_result.capability_used,
            ShellGitCapability.WORKTREE,
        )
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_attempted"],
            True,
        )
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_scope"],
            "worktree",
        )


class GitWorktreeToolExposurePhase5Test(TestCase):
    def test_read_only_config_exposes_no_worktree_tools(self) -> None:
        toolset = ShellToolSet().with_enabled_tools(["shell.*"])
        names = set(_schema_names(toolset))

        self.assertTrue(set(_PHASE5_TOOL_NAMES).isdisjoint(names))

    def test_explicit_read_only_config_exposes_no_worktree_tools(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(allowed_commands=("add",))
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.git_add"]
        )
        selected_names = set(_schema_names(toolset))

        self.assertNotIn("shell.git_add", selected_names)

    def test_worktree_config_exposes_no_history_or_remote_tools(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("worktree",),
                allowed_commands=_PHASE5_COMMANDS,
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )
        names = set(_git_schema_names(toolset))

        self.assertEqual(names, set(_PHASE5_TOOL_NAMES))
        self.assertTrue(_HISTORY_TOOL_NAMES.isdisjoint(names))
        self.assertTrue(_REMOTE_TOOL_NAMES.isdisjoint(names))

    def test_worktree_capability_cannot_enable_history_or_remote(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("worktree",),
                allowed_commands=(
                    *_PHASE5_COMMANDS,
                    "commit",
                    "branch-delete",
                    "tag-delete",
                    "fetch",
                    "pull",
                    "push",
                    "clone",
                    "submodule-update",
                ),
                allowed_remote_hosts=("github.com",),
                allow_submodule_update=True,
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )
        names = set(_git_schema_names(toolset))

        self.assertEqual(names, set(_PHASE5_TOOL_NAMES))
        self.assertNotIn("shell.git_commit", names)
        self.assertNotIn("shell.git_branch_delete", names)
        self.assertNotIn("shell.git_tag_delete", names)
        self.assertNotIn("shell.git_push", names)
        self.assertNotIn("shell.git_pull", names)
        self.assertNotIn("shell.git_fetch", names)
        self.assertNotIn("shell.git_clone", names)
        self.assertNotIn("shell.git_submodule_update", names)

    def test_worktree_schemas_require_mutation_arguments(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("worktree",),
                allowed_commands=_PHASE5_COMMANDS,
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )
        parameters = _parameter_schemas(toolset)
        required_fields = {
            "shell.git_add": {"paths"},
            "shell.git_restore": {"paths"},
            "shell.git_checkout": {"paths"},
            "shell.git_switch": {"branch"},
            "shell.git_reset": {"paths"},
            "shell.git_rm": {"paths"},
            "shell.git_mv": {"source", "destination"},
            "shell.git_stash_push": {"paths"},
            "shell.git_stash_apply": {"stash", "paths"},
        }

        for tool_name, fields in required_fields.items():
            with self.subTest(tool_name=tool_name):
                self.assertTrue(
                    fields.issubset(set(parameters[tool_name]["required"]))
                )

        reset_properties = parameters["shell.git_reset"]["properties"]
        self.assertNotIn("mode", reset_properties)
        self.assertNotIn("revision", reset_properties)
        checkout_properties = parameters["shell.git_checkout"]["properties"]
        self.assertNotIn("mode", checkout_properties)
        self.assertNotIn("branch", checkout_properties)
        switch_properties = parameters["shell.git_switch"]["properties"]
        self.assertNotIn("create", switch_properties)
        rm_properties = parameters["shell.git_rm"]["properties"]
        self.assertNotIn("recursive", rm_properties)
        stash_apply_properties = parameters["shell.git_stash_apply"][
            "properties"
        ]
        self.assertNotIn("index", stash_apply_properties)


@skipIf(_GIT_BINARY is None, "git executable is not available")
class GitWorktreeSmokePhase5Test(IsolatedAsyncioTestCase):
    async def test_git_add_rejects_directory_pathspecs_before_staging_children(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)
            (repo / "src").mkdir()
            (repo / "src" / "safe.txt").write_text("safe\n")
            (repo / "src" / ".env").write_text("secret\n")

            denied_directory = await _call_tool(
                toolset,
                "git_add",
                paths=("src",),
            )
            denied_root = await _call_tool(
                toolset,
                "git_add",
                paths=(".",),
            )
            staged_paths = _git_output(
                repo,
                _GIT_BINARY,
                "diff",
                "--cached",
                "--name-only",
                "--",
                "src/safe.txt",
                "src/.env",
            )

        self.assertEqual(
            denied_directory.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            denied_directory.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertEqual(
            denied_root.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            denied_root.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertNotIn("src/safe.txt", staged_paths)
        self.assertNotIn("src/.env", staged_paths)

    async def test_stash_apply_restores_only_explicit_safe_paths(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)
            (repo / ".env").write_text("base secret\n")
            _git(repo, _GIT_BINARY, "add", ".env")
            _git(repo, _GIT_BINARY, "commit", "-m", "track env")
            (repo / "tracked.txt").write_text("stashed safe\n")
            (repo / ".env").write_text("stashed secret\n")
            _git(
                repo,
                _GIT_BINARY,
                "stash",
                "push",
                "--message",
                "mixed payload",
                "--",
                "tracked.txt",
                ".env",
            )

            denied_hidden = await _call_tool(
                toolset,
                "git_stash_apply",
                stash="stash@{0}",
                paths=(".env",),
            )
            denied_broad = await _call_tool(
                toolset,
                "git_stash_apply",
                stash="stash@{0}",
                paths=(".",),
            )
            restored_safe = await _call_tool(
                toolset,
                "git_stash_apply",
                stash="stash@{0}",
                paths=("tracked.txt",),
            )
            tracked_text = (repo / "tracked.txt").read_text()
            env_text = (repo / ".env").read_text()

        self.assertEqual(
            denied_hidden.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            denied_hidden.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertEqual(
            denied_broad.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            denied_broad.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertEqual(
            restored_safe.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        self.assertEqual(restored_safe.git_result.error_code, None)
        self.assertEqual(tracked_text, "stashed safe\n")
        self.assertEqual(env_text, "base secret\n")

    async def test_checkout_branch_mode_cannot_restore_same_named_file(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            target = "topic"
            (repo / target).write_text("base\n")
            _git(repo, _GIT_BINARY, "add", target)
            _git(repo, _GIT_BINARY, "commit", "-m", "track topic")
            (repo / target).write_text("modified\n")

            error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.CHECKOUT,
                    options={"mode": "branch", "target": target},
                ),
            )

            self.assertEqual(
                error.error_code,
                ShellGitExecutionErrorCode.INVALID_OPTION,
            )
            self.assertEqual((repo / target).read_text(), "modified\n")

    async def test_worktree_tools_mutate_temporary_repo_state(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            (repo / "new.txt").write_text("new\n")
            add = await _call_tool(
                toolset,
                "git_add",
                paths=("new.txt",),
            )
            self.assertIn(
                "new.txt",
                _git_output(
                    repo,
                    _GIT_BINARY,
                    "diff",
                    "--cached",
                    "--name-only",
                ),
            )

            reset = await _call_tool(
                toolset,
                "git_reset",
                paths=("new.txt",),
            )
            self.assertNotIn(
                "new.txt",
                _git_output(
                    repo,
                    _GIT_BINARY,
                    "diff",
                    "--cached",
                    "--name-only",
                ),
            )

            (repo / "tracked.txt").write_text("changed by checkout\n")
            checkout_paths = await _call_tool(
                toolset,
                "git_checkout",
                target="HEAD",
                paths=("tracked.txt",),
            )
            self.assertEqual((repo / "tracked.txt").read_text(), "base\n")

            (repo / "tracked.txt").write_text("changed by restore\n")
            restore = await _call_tool(
                toolset,
                "git_restore",
                paths=("tracked.txt",),
            )
            self.assertEqual((repo / "tracked.txt").read_text(), "base\n")

            switch = await _call_tool(toolset, "git_switch", branch="side")
            self.assertEqual(
                _git_output(repo, _GIT_BINARY, "branch", "--show-current"),
                "side\n",
            )
            switch_back = await _call_tool(
                toolset,
                "git_switch",
                branch="main",
            )
            self.assertEqual(
                _git_output(repo, _GIT_BINARY, "branch", "--show-current"),
                "main\n",
            )

            mv = await _call_tool(
                toolset,
                "git_mv",
                source="move.txt",
                destination="moved.txt",
            )
            self.assertFalse((repo / "move.txt").exists())
            self.assertTrue((repo / "moved.txt").exists())

            rm = await _call_tool(
                toolset,
                "git_rm",
                paths=("remove.txt",),
            )
            self.assertFalse((repo / "remove.txt").exists())

            (repo / "tracked.txt").write_text("stashed\n")
            stash_push = await _call_tool(
                toolset,
                "git_stash_push",
                message="phase5",
                paths=("tracked.txt",),
            )
            self.assertEqual((repo / "tracked.txt").read_text(), "base\n")
            stash_apply = await _call_tool(
                toolset,
                "git_stash_apply",
                stash="stash@{0}",
                paths=("tracked.txt",),
            )
            self.assertEqual((repo / "tracked.txt").read_text(), "stashed\n")

            denied = await _call_tool(
                toolset,
                "git_add",
                paths=("../outside",),
            )

        for result in (
            add,
            reset,
            checkout_paths,
            restore,
            switch,
            switch_back,
            mv,
            rm,
            stash_push,
            stash_apply,
        ):
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertEqual(result.git_result.error_code, None)
            self.assertEqual(
                result.git_result.capability_used,
                ShellGitCapability.WORKTREE,
            )
            self.assertEqual(
                result.git_result.audit_metadata["git_mutation_attempted"],
                True,
            )

        self.assertEqual(
            denied.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            denied.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )


def _git_prefix() -> tuple[str, str, str]:
    return ("git", "--no-pager", "--no-optional-locks")


def _request(
    *,
    command: ShellGitCommandName,
    options: dict[str, object] | None = None,
    pathspecs: tuple[str, ...] = (),
    timeout_seconds: float | None = None,
    max_stdout_bytes: int | None = None,
    max_stderr_bytes: int | None = None,
) -> ShellGitCommandRequest:
    return ShellGitCommandRequest(
        tool_name=f"shell.git_{command.value.replace('-', '_')}",
        command=command,
        capability_required=SHELL_GIT_COMMAND_CAPABILITIES[command],
        options={} if options is None else options,
        pathspecs=pathspecs,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _policy(
    workspace_root: Path,
    *,
    default_timeout_seconds: float = 10.0,
    max_timeout_seconds: float = 60.0,
    max_stdout_bytes: int = 65536,
    max_stderr_bytes: int = 32768,
    max_pathspecs: int = 64,
    max_pathspec_bytes: int = 4096,
    max_arguments: int = 128,
    max_argument_bytes: int = 8192,
    max_command_bytes: int = 32768,
    allowed_commands: tuple[str, ...] = _PHASE5_COMMANDS,
    allowed_remote_hosts: tuple[str, ...] = (),
    allow_linked_worktrees: bool = False,
    allow_submodule_update: bool = False,
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=ShellToolSettings(
            max_arguments=max_arguments,
            max_argument_bytes=max_argument_bytes,
            max_command_bytes=max_command_bytes,
            git=ShellGitToolSettings(
                workspace_root=str(workspace_root),
                cwd="repo",
                capabilities=("worktree",),
                allowed_commands=allowed_commands,
                default_timeout_seconds=default_timeout_seconds,
                max_timeout_seconds=max_timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
                max_pathspecs=max_pathspecs,
                max_pathspec_bytes=max_pathspec_bytes,
                allowed_remote_hosts=allowed_remote_hosts,
                allow_linked_worktrees=allow_linked_worktrees,
                allow_submodule_update=allow_submodule_update,
            ),
        ),
        executable_lookup=_fake_executable,
    )


def _disabled_command_options(
    command: ShellGitCommandName,
) -> dict[str, object]:
    if command is ShellGitCommandName.COMMIT:
        return {"message": "denied"}
    if command in {
        ShellGitCommandName.BRANCH_DELETE,
        ShellGitCommandName.TAG_DELETE,
    }:
        return {"name": "main"}
    if command is ShellGitCommandName.SUBMODULE_UPDATE:
        return {"recursive": False}
    if command is ShellGitCommandName.CLONE:
        return {"url": "https://github.com/acme/repo.git"}
    return {}


async def _fake_executable(search_paths: tuple[str, ...]) -> str | None:
    assert isinstance(search_paths, tuple)
    return "/usr/bin/git"


async def _policy_error(
    policy: GitExecutionPolicy,
    request: ShellGitCommandRequest,
) -> ShellGitPolicyDenied:
    try:
        await policy.normalize(request)
    except ShellGitPolicyDenied as error:
        return error
    raise AssertionError("Git policy should have denied the request")


def _write_minimal_git_repo(repo: Path) -> Path:
    git_dir = repo / ".git"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("hello\n")
    (repo / "old.txt").write_text("old\n")
    (repo / "src" / "app.py").write_text("print('hello')\n")
    (git_dir / "objects" / "info").mkdir(parents=True)
    (git_dir / "refs").mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    )
    return repo


def _fake_toolset(root: Path, executor: "_FakeGitExecutor") -> ShellToolSet:
    return ShellToolSet(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=str(root),
                cwd="repo",
                capabilities=("worktree",),
                allowed_commands=_PHASE5_COMMANDS,
            ),
        ),
        executor=executor,
    ).with_enabled_tools(_PHASE5_TOOL_NAMES)


def _real_git_toolset(
    root: Path,
    repo: Path,
    git_binary: str,
) -> ShellToolSet:
    return ShellToolSet(
        settings=ShellToolSettings(
            executable_search_paths=(str(Path(git_binary).parent),),
            git=ShellGitToolSettings(
                workspace_root=str(root),
                cwd=repo.name,
                capabilities=("worktree",),
                allowed_commands=_PHASE5_COMMANDS,
            ),
        )
    ).with_enabled_tools(_PHASE5_TOOL_NAMES)


async def _call_tool(
    toolset: ShellToolSet,
    command_id: str,
    **kwargs: Any,
) -> ShellGitFormattedResult:
    tool = _tool_by_name(toolset, command_id)
    result = await tool(context=ToolCallContext(), **kwargs)
    assert isinstance(result, ShellGitFormattedResult)
    return result


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> _GitToolCallable:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            assert callable(tool), "Git tool must be callable"
            return cast(_GitToolCallable, tool)
    raise AssertionError(f"Git tool not found: {command_id}")


def _schema_names(toolset: ShellToolSet) -> tuple[str, ...]:
    names: list[str] = []
    for tool in toolset.tools:
        assert callable(tool), "Schema tool must be callable"
        name = getattr(tool, "__name__", "")
        assert isinstance(name, str)
        names.append(f"shell.{name}")
    return tuple(names)


def _git_schema_names(toolset: ShellToolSet) -> tuple[str, ...]:
    return tuple(
        name
        for name in _schema_names(toolset)
        if name.startswith("shell.git_")
    )


def _parameter_schemas(toolset: ShellToolSet) -> dict[str, dict[str, Any]]:
    schemas = toolset.json_schemas()
    return {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in schemas or ()
        if schema["function"]["name"].startswith("shell.git_")
    }


class _FakeGitExecutor:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        status: ShellExecutionStatus = ShellExecutionStatus.NONZERO_EXIT,
        exit_code: int | None = 128,
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        timed_out: bool = False,
        error_message: str | None = None,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._status = status
        self._exit_code = exit_code
        self._stdout_truncated = stdout_truncated
        self._stderr_truncated = stderr_truncated
        self._timed_out = timed_out
        self._error_message = error_message
        self.calls = 0

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        assert stream is None
        self.calls += 1
        return _execution_result(
            spec,
            status=self._status,
            stdout=self._stdout,
            stderr=self._stderr,
            exit_code=self._exit_code,
            stdout_truncated=self._stdout_truncated,
            stderr_truncated=self._stderr_truncated,
            timed_out=self._timed_out,
            error_message=self._error_message,
        )


def _execution_result(
    spec: ExecutionSpec,
    *,
    status: ShellExecutionStatus,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 128,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
    error_message: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        backend=spec.backend,
        tool_name=spec.tool_name,
        command=spec.command,
        argv=spec.argv,
        display_argv=spec.display_argv,
        cwd=spec.cwd,
        display_cwd=spec.display_cwd,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        stdout_bytes=len(stdout.encode("utf-8")),
        stderr_bytes=len(stderr.encode("utf-8")),
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
        duration_ms=3,
        error_message=error_message,
        metadata=spec.metadata,
    )


def _write_real_git_repo(root: Path, git_binary: str) -> Path:
    repo = root / "repo"
    _git(root, git_binary, "init", "repo")
    _git(repo, git_binary, "checkout", "-b", "main")
    _git(repo, git_binary, "config", "user.name", "Avalan Test")
    _git(repo, git_binary, "config", "user.email", "avalan@example.test")
    (repo / "tracked.txt").write_text("base\n")
    (repo / "move.txt").write_text("move\n")
    (repo / "remove.txt").write_text("remove\n")
    _git(repo, git_binary, "add", "tracked.txt", "move.txt", "remove.txt")
    _git(repo, git_binary, "commit", "-m", "phase5 setup")
    _git(repo, git_binary, "checkout", "-b", "side")
    _git(repo, git_binary, "checkout", "main")
    return repo


def _git(cwd: Path, git_binary: str, *args: str) -> None:
    _git_run(cwd, git_binary, *args)


def _git_output(cwd: Path, git_binary: str, *args: str) -> str:
    return _git_run(cwd, git_binary, *args).stdout


def _git_run(cwd: Path, git_binary: str, *args: str) -> CompletedProcess[str]:
    isolation_root = _git_isolation_root(cwd)
    return run(
        (
            git_binary,
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            "-c",
            "credential.helper=",
            "-c",
            f"core.hooksPath={isolation_root / 'hooks'}",
            "-c",
            f"init.templateDir={isolation_root / 'templates'}",
            *args,
        ),
        cwd=cwd,
        env=_git_env(isolation_root),
        check=True,
        capture_output=True,
        text=True,
    )


def _git_isolation_root(cwd: Path) -> Path:
    if (cwd / ".git").exists():
        return cwd.parent / ".git-test-env"
    return cwd / ".git-test-env"


def _git_env(isolation_root: Path) -> dict[str, str]:
    home = isolation_root / "home"
    paths = (
        home,
        isolation_root / "hooks",
        isolation_root / "templates",
        isolation_root / "tmp",
        isolation_root / "xdg-cache",
        isolation_root / "xdg-config",
        isolation_root / "xdg-data",
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    env = {
        "GIT_ASKPASS": "",
        "GIT_CONFIG_GLOBAL": devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_EDITOR": "true",
        "GIT_PAGER": "cat",
        "GIT_TEMPLATE_DIR": str(isolation_root / "templates"),
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(home),
        "SSH_ASKPASS": "",
        "TEMP": str(isolation_root / "tmp"),
        "TMP": str(isolation_root / "tmp"),
        "TMPDIR": str(isolation_root / "tmp"),
        "XDG_CACHE_HOME": str(isolation_root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(isolation_root / "xdg-config"),
        "XDG_DATA_HOME": str(isolation_root / "xdg-data"),
    }
    current_path = environ.get("PATH")
    if current_path is not None:
        env["PATH"] = current_path

    for key in ("LANG", "LC_ALL", "LC_CTYPE"):
        value = environ.get(key)
        if value is not None:
            env[key] = value

    return env
