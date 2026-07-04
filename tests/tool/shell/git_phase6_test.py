from collections.abc import Awaitable, Callable
from os import devnull, environ
from pathlib import Path
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
    ShellGitCapability,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitExecutionErrorCode,
    ShellGitExecutionStatus,
    ShellGitPolicyDenied,
    shell_git_capability_for_request,
)
from avalan.tool.shell.git_policy import (
    GitExecutionPolicy,
    _argv_for_request,
    _collect_ref_names,
    _packed_ref_names,
    _redacted_argv,
    _ref_names,
)
from avalan.tool.shell.settings import ShellGitToolSettings, ShellToolSettings
from avalan.tool.shell.toolset import ShellToolSet

_GIT_BINARY = which("git")
_PHASE6_COMMANDS = (
    "commit",
    "branch-create",
    "branch-delete",
    "branch-rename",
    "tag-create",
    "tag-delete",
    "merge",
    "rebase",
    "cherry-pick",
    "revert",
    "reset",
    "clean",
    "stash-pop",
    "stash-drop",
)
_PHASE6_TOOL_NAMES = {
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
    "shell.git_reset",
    "shell.git_clean",
    "shell.git_stash_pop",
    "shell.git_stash_drop",
}
_WORKTREE_TOOL_NAMES = {
    "shell.git_add",
    "shell.git_restore",
    "shell.git_checkout",
    "shell.git_switch",
    "shell.git_rm",
    "shell.git_mv",
    "shell.git_stash_push",
    "shell.git_stash_apply",
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
_HEAD = "1111111111111111111111111111111111111111"
_SIDE = "2222222222222222222222222222222222222222"
_OLD = "3333333333333333333333333333333333333333"
_GitToolCallable = Callable[..., Awaitable[str]]


class GitHistoryPolicyPhase6Test(IsolatedAsyncioTestCase):
    async def test_history_commands_build_fixed_argv_and_audit_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.COMMIT,
                        options={"message": "phase6 commit"},
                    ),
                    (
                        *_git_prefix(),
                        "commit",
                        "--no-edit",
                        "--no-gpg-sign",
                        "--no-verify",
                        "--no-post-rewrite",
                        "--no-status",
                        "--cleanup=strip",
                        "--message",
                        "phase6 commit",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.BRANCH_CREATE,
                        options={"name": "new", "start_point": "HEAD"},
                    ),
                    (*_git_prefix(), "branch", "--no-track", "new", "HEAD"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.BRANCH_DELETE,
                        options={"name": "old", "confirm_name": "old"},
                    ),
                    (*_git_prefix(), "branch", "--delete", "old"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.BRANCH_RENAME,
                        options={
                            "old_name": "old",
                            "new_name": "new",
                            "confirm_old_name": "old",
                        },
                    ),
                    (*_git_prefix(), "branch", "--move", "old", "new"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.TAG_CREATE,
                        options={
                            "name": "v2.0",
                            "target": "HEAD",
                            "message": "release tag",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "tag",
                        "--annotate",
                        "--no-sign",
                        "--message",
                        "release tag",
                        "v2.0",
                        "HEAD",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.TAG_DELETE,
                        options={"name": "v0.1", "confirm_name": "v0.1"},
                    ),
                    (*_git_prefix(), "tag", "--delete", "v0.1"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.MERGE,
                        options={
                            "revision": "feature",
                            "confirm_revision": "feature",
                            "mode": "no_ff",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "merge",
                        "--no-verify",
                        "--no-gpg-sign",
                        "--no-stat",
                        "--no-edit",
                        "--no-autostash",
                        "--no-ff",
                        "feature",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.REBASE,
                        options={
                            "upstream": "main",
                            "confirm_upstream": "main",
                            "branch": "feature",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "rebase",
                        "--no-verify",
                        "--no-gpg-sign",
                        "--no-stat",
                        "--no-autostash",
                        "--no-rebase-merges",
                        "--empty=stop",
                        "main",
                        "feature",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.CHERRY_PICK,
                        options={
                            "revision": "feature",
                            "confirm_revision": "feature",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "cherry-pick",
                        "--no-edit",
                        "--no-gpg-sign",
                        "feature",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.REVERT,
                        options={
                            "revision": "feature",
                            "confirm_revision": "feature",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "revert",
                        "--no-edit",
                        "--no-gpg-sign",
                        "feature",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.RESET,
                        options={
                            "mode": "hard",
                            "revision": "HEAD",
                            "confirm_revision": "HEAD",
                            "confirm_hard": True,
                        },
                    ),
                    (
                        *_git_prefix(),
                        "reset",
                        "--hard",
                        "--no-recurse-submodules",
                        "HEAD",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.CLEAN,
                        options={
                            "dry_run": False,
                            "confirm_paths": ("scratch.txt",),
                        },
                        pathspecs=("scratch.txt",),
                    ),
                    (
                        *_git_prefix(),
                        "clean",
                        "--force",
                        "--",
                        "scratch.txt",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_POP,
                        options={
                            "stash": "stash@{0}",
                            "confirm_stash": "stash@{0}",
                            "index": True,
                        },
                    ),
                    (
                        *_git_prefix(),
                        "stash",
                        "pop",
                        "--index",
                        "stash@{0}",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_DROP,
                        options={
                            "stash": "stash@{0}",
                            "confirm_stash": "stash@{0}",
                        },
                    ),
                    (*_git_prefix(), "stash", "drop", "stash@{0}"),
                ),
            )

            for request, expected_argv in cases:
                with self.subTest(command=request.command.value):
                    spec = await policy.normalize(request)

                    self.assertEqual(spec.argv, expected_argv)
                    if "--message" in expected_argv:
                        self.assertIn("[redacted]", spec.display_argv)
                    else:
                        self.assertEqual(spec.display_argv, expected_argv)
                    self.assertEqual(
                        spec.metadata["git_capability_used"],
                        ShellGitCapability.HISTORY.value,
                    )
                    self.assertEqual(
                        spec.metadata["git_mutation_attempted"],
                        True,
                    )
                    self.assertEqual(
                        spec.metadata["git_mutation_scope"],
                        "history",
                    )
                    self.assertEqual(spec.env["GIT_EDITOR"], "/nonexistent")
                    self.assertEqual(
                        spec.env["GIT_SEQUENCE_EDITOR"],
                        "/nonexistent",
                    )
                    self.assertEqual(spec.env["GIT_MERGE_AUTOEDIT"], "no")

    async def test_history_capability_does_not_authorize_worktree_reset_mode(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")

            history_error = await _policy_error(
                _policy(
                    root,
                    capabilities=("history",),
                    allowed_commands=("reset",),
                ),
                _request(
                    command=ShellGitCommandName.RESET,
                    options={"mode": "paths"},
                    pathspecs=("tracked.txt",),
                ),
            )
            worktree_error = await _policy_error(
                _policy(
                    root,
                    capabilities=("worktree",),
                    allowed_commands=("reset",),
                ),
                _request(
                    command=ShellGitCommandName.RESET,
                    options={
                        "mode": "hard",
                        "revision": "HEAD",
                        "confirm_revision": "HEAD",
                        "confirm_hard": True,
                    },
                ),
            )

        self.assertEqual(
            history_error.error_code,
            ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
        )
        self.assertEqual(
            worktree_error.error_code,
            ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
        )

    async def test_history_confirmation_and_unsafe_forms_fail_closed(
        self,
    ) -> None:
        cases = (
            (
                _request(
                    command=ShellGitCommandName.BRANCH_DELETE,
                    options={"name": "old"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG_DELETE,
                    options={"name": "v0.1", "confirm_name": "other"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.MERGE,
                    options={"revision": "feature"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.REBASE,
                    options={
                        "upstream": "main",
                        "confirm_upstream": "other",
                    },
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.RESET,
                    options={
                        "mode": "hard",
                        "revision": "HEAD",
                        "confirm_revision": "HEAD",
                    },
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.CLEAN,
                    options={
                        "dry_run": False,
                        "confirm_paths": ("other.txt",),
                    },
                    pathspecs=("scratch.txt",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_POP,
                    options={"stash": "stash@{0}"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH_CREATE,
                    options={"name": "HEAD"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH_CREATE,
                    options={"name": "topic.lock"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH_CREATE,
                    options={"name": "feature~1"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG_CREATE,
                    options={"name": "deadbee"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.REBASE,
                    options={
                        "upstream": "main",
                        "confirm_upstream": "main",
                        "exec": "echo unsafe",
                    },
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
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
            error = await _policy_error(
                _policy(root, max_commit_message_bytes=4),
                _request(
                    command=ShellGitCommandName.COMMIT,
                    options={"message": "too long"},
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.INVALID_OPTION,
        )

    async def test_history_ref_state_denies_missing_and_ambiguous_refs(
        self,
    ) -> None:
        cases = (
            (
                _request(
                    command=ShellGitCommandName.BRANCH_DELETE,
                    options={
                        "name": "missing",
                        "confirm_name": "missing",
                    },
                ),
                ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG_DELETE,
                    options={
                        "name": "missing",
                        "confirm_name": "missing",
                    },
                ),
                ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            ),
            (
                _request(
                    command=ShellGitCommandName.MERGE,
                    options={
                        "revision": "missing",
                        "confirm_revision": "missing",
                    },
                ),
                ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            ),
            (
                _request(
                    command=ShellGitCommandName.MERGE,
                    options={"revision": "dup", "confirm_revision": "dup"},
                ),
                ShellGitExecutionErrorCode.AMBIGUOUS_REVISION,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_DROP,
                    options={
                        "stash": "stash@{3}",
                        "confirm_stash": "stash@{3}",
                    },
                ),
                ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH_CREATE,
                    options={"name": "v0.1", "start_point": "HEAD"},
                ),
                ShellGitExecutionErrorCode.AMBIGUOUS_REVISION,
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

    async def test_history_ref_collection_edges_are_bounded(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            git_dir = repo / ".git"
            heads_root = git_dir / "refs" / "heads"
            names: set[str] = set()

            await _collect_ref_names(heads_root, heads_root / "missing", names)
            plain_file = heads_root / "plain-file"
            plain_file.write_text(f"{_HEAD}\n")
            await _collect_ref_names(heads_root, plain_file, names)
            await _collect_ref_names(heads_root, heads_root, names)
            missing_names = await _ref_names(git_dir, "remotes")
            packed_heads, packed_tags = await _packed_ref_names(
                git_dir / "packed-refs"
            )

        self.assertEqual(missing_names, frozenset())
        self.assertIn("nested/branch", names)
        self.assertIn("packed", packed_heads)
        self.assertIn("packed-tag", packed_tags)

    async def test_history_head_and_stash_reference_edges(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "HEAD").write_text("ref: refs/heads/missing\n")

            missing_head = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.BRANCH_CREATE,
                    options={"name": "headless"},
                ),
            )

        self.assertEqual(
            missing_head.error_code,
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "HEAD").write_text("not-a-valid-head\n")

            invalid_head = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.TAG_CREATE,
                    options={"name": "headless-tag"},
                ),
            )

        self.assertEqual(
            invalid_head.error_code,
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "HEAD").write_text(f"{_HEAD}\n")
            policy = _policy(root)

            detached = await policy.normalize(
                _request(
                    command=ShellGitCommandName.TAG_CREATE,
                    options={"name": "detached"},
                )
            )
            stash_from_log = await policy.normalize(
                _request(
                    command=ShellGitCommandName.STASH_DROP,
                    options={
                        "stash": "stash@{1}",
                        "confirm_stash": "stash@{1}",
                    },
                )
            )

        self.assertEqual(
            detached.argv,
            (*_git_prefix(), "tag", "detached"),
        )
        self.assertEqual(
            stash_from_log.argv,
            (*_git_prefix(), "stash", "drop", "stash@{1}"),
        )

    async def test_history_internal_guards_fail_closed(self) -> None:
        settings = ShellGitToolSettings()
        internal_cases = (
            (
                _request(
                    command=ShellGitCommandName.BRANCH_RENAME,
                    options={
                        "old_name": "old",
                        "new_name": "old",
                        "confirm_old_name": "old",
                    },
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH_CREATE,
                    options={"name": "HEAD"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
        )

        for request, error_code in internal_cases:
            with self.subTest(command=request.command.value):
                with self.assertRaises(ShellGitPolicyDenied) as raised:
                    _argv_for_request(request, (), settings)

                self.assertEqual(raised.exception.error_code, error_code)

        redacted = _redacted_argv(
            ("git", "tag", "--message=secret"),
            settings,
            redact_messages=True,
        )
        self.assertEqual(redacted, ("git", "tag", "--message=[redacted]"))

    async def test_history_clean_confirm_paths_types_fail_closed(self) -> None:
        cases = (
            "scratch.txt",
            ("scratch.txt", 1),
        )

        for confirm_paths in cases:
            with self.subTest(confirm_paths=confirm_paths):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(
                        _policy(root),
                        _request(
                            command=ShellGitCommandName.CLEAN,
                            options={
                                "dry_run": False,
                                "confirm_paths": confirm_paths,
                            },
                            pathspecs=("scratch.txt",),
                        ),
                    )

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                )


class GitHistoryToolExposurePhase6Test(TestCase):
    def test_history_config_exposes_no_worktree_or_remote_tools(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("history",),
                allowed_commands=(
                    *_PHASE6_COMMANDS,
                    "add",
                    "fetch",
                    "push",
                ),
                allowed_remote_hosts=("github.com",),
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )
        names = set(_git_schema_names(toolset))

        self.assertEqual(names, _PHASE6_TOOL_NAMES)
        self.assertTrue(_WORKTREE_TOOL_NAMES.isdisjoint(names))
        self.assertTrue(_REMOTE_TOOL_NAMES.isdisjoint(names))

    def test_history_schemas_require_guard_fields(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("history",),
                allowed_commands=_PHASE6_COMMANDS,
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )
        parameters = _parameter_schemas(toolset)
        required_fields = {
            "shell.git_commit": {"message"},
            "shell.git_branch_create": {"name"},
            "shell.git_branch_delete": {"name", "confirm_name"},
            "shell.git_branch_rename": {
                "old_name",
                "new_name",
                "confirm_old_name",
            },
            "shell.git_tag_create": {"name"},
            "shell.git_tag_delete": {"name", "confirm_name"},
            "shell.git_merge": {"revision", "confirm_revision"},
            "shell.git_rebase": {"upstream", "confirm_upstream"},
            "shell.git_cherry_pick": {"revision", "confirm_revision"},
            "shell.git_revert": {"revision", "confirm_revision"},
            "shell.git_clean": {"paths"},
            "shell.git_stash_pop": {"confirm_stash"},
            "shell.git_stash_drop": {"confirm_stash"},
        }

        for tool_name, fields in required_fields.items():
            with self.subTest(tool_name=tool_name):
                self.assertTrue(
                    fields.issubset(set(parameters[tool_name]["required"]))
                )

        reset_parameters = parameters["shell.git_reset"]
        reset_properties = reset_parameters["properties"]
        self.assertNotIn("paths", reset_properties)
        self.assertEqual(
            reset_properties["mode"]["enum"],
            ["soft", "mixed", "hard"],
        )
        self.assertNotIn("default", reset_properties["mode"])
        self.assertTrue(
            {
                "mode",
                "revision",
                "confirm_revision",
            }.issubset(set(reset_parameters["required"]))
        )


class GitHistoryExecutionPhase6Test(IsolatedAsyncioTestCase):
    async def test_history_test_helpers_report_unexpected_success(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            with self.assertRaises(AssertionError):
                await _policy_error(
                    policy,
                    _request(
                        command=ShellGitCommandName.BRANCH_CREATE,
                        options={"name": "helper-success"},
                    ),
                )

        toolset = ShellToolSet(settings=ShellToolSettings())
        with self.assertRaises(AssertionError):
            _tool_by_name(toolset, "missing_git_tool")

    async def test_history_denials_include_audit_metadata(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor()
            toolset = _fake_toolset(
                root,
                executor,
                max_commit_message_bytes=4,
            )

            result = await _call_tool(
                toolset,
                "git_commit",
                message="secret message",
            )

        self.assertEqual(executor.calls, 0)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_attempted"],
            True,
        )
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_scope"],
            "history",
        )
        self.assertNotIn(
            "secret message", str(result.git_result.audit_metadata)
        )

    async def test_history_successes_include_audit_metadata(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                stdout="deleted\n",
                status=ShellExecutionStatus.COMPLETED,
                exit_code=0,
            )
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_branch_delete",
                name="old",
                confirm_name="old",
            )

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        self.assertEqual(
            result.git_result.capability_used,
            ShellGitCapability.HISTORY,
        )
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_attempted"],
            True,
        )
        self.assertEqual(
            result.git_result.audit_metadata["git_mutation_scope"],
            "history",
        )


@skipIf(_GIT_BINARY is None, "git executable is not available")
class GitHistorySmokePhase6Test(IsolatedAsyncioTestCase):
    async def test_history_tools_mutate_temporary_repo_state(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            (repo / "commit.txt").write_text("commit\n")
            _git(repo, _GIT_BINARY, "add", "commit.txt")
            commit = await _call_tool(
                toolset,
                "git_commit",
                message="phase6 commit",
            )

            branch_create = await _call_tool(
                toolset,
                "git_branch_create",
                name="topic",
                start_point="HEAD",
            )
            branch_rename = await _call_tool(
                toolset,
                "git_branch_rename",
                old_name="topic",
                new_name="topic2",
                confirm_old_name="topic",
            )
            branch_delete = await _call_tool(
                toolset,
                "git_branch_delete",
                name="topic2",
                confirm_name="topic2",
            )

            tag_create = await _call_tool(
                toolset,
                "git_tag_create",
                name="v2.0",
                target="HEAD",
                message="phase6 tag",
            )
            tag_delete = await _call_tool(
                toolset,
                "git_tag_delete",
                name="v2.0",
                confirm_name="v2.0",
            )

            (repo / "scratch.txt").write_text("scratch\n")
            clean = await _call_tool(
                toolset,
                "git_clean",
                paths=("scratch.txt",),
                dry_run=False,
                confirm_paths=("scratch.txt",),
            )

            (repo / "tracked.txt").write_text("popped\n")
            _git(
                repo,
                _GIT_BINARY,
                "stash",
                "push",
                "--message",
                "phase6 pop",
                "--",
                "tracked.txt",
            )
            stash_pop = await _call_tool(
                toolset,
                "git_stash_pop",
                stash="stash@{0}",
                confirm_stash="stash@{0}",
            )
            popped_text = (repo / "tracked.txt").read_text()

            (repo / "tracked.txt").write_text("dropped\n")
            _git(
                repo,
                _GIT_BINARY,
                "stash",
                "push",
                "--message",
                "phase6 drop",
                "--",
                "tracked.txt",
            )
            stash_drop = await _call_tool(
                toolset,
                "git_stash_drop",
                stash="stash@{0}",
                confirm_stash="stash@{0}",
            )

            head_before_reset = _git_output(
                repo,
                _GIT_BINARY,
                "rev-parse",
                "HEAD",
            ).strip()
            (repo / "reset.txt").write_text("reset\n")
            _git(repo, _GIT_BINARY, "add", "reset.txt")
            _git(repo, _GIT_BINARY, "commit", "-m", "raw reset target")
            reset = await _call_tool(
                toolset,
                "git_reset",
                mode="hard",
                revision=head_before_reset,
                confirm_revision=head_before_reset,
                confirm_hard=True,
            )
            head_after_reset = _git_output(
                repo,
                _GIT_BINARY,
                "rev-parse",
                "HEAD",
            ).strip()
            branches = _git_output(repo, _GIT_BINARY, "branch", "--list")
            tags = _git_output(repo, _GIT_BINARY, "tag", "--list")
            stashes = _git_output(repo, _GIT_BINARY, "stash", "list")
            scratch_exists = (repo / "scratch.txt").exists()
            tracked_text = (repo / "tracked.txt").read_text()
            reset_exists = (repo / "reset.txt").exists()

        for result in (
            commit,
            branch_create,
            branch_rename,
            branch_delete,
            tag_create,
            tag_delete,
            clean,
            stash_pop,
            stash_drop,
            reset,
        ):
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertEqual(result.git_result.error_code, None)
            self.assertEqual(
                result.git_result.capability_used,
                ShellGitCapability.HISTORY,
            )
            self.assertEqual(
                result.git_result.audit_metadata["git_mutation_scope"],
                "history",
            )

        self.assertFalse(scratch_exists)
        self.assertEqual(popped_text, "popped\n")
        self.assertEqual(tracked_text, "base\n")
        self.assertEqual(head_after_reset, head_before_reset)
        self.assertFalse(reset_exists)
        self.assertNotIn("topic", branches)
        self.assertNotIn("v2.0", tags)
        self.assertEqual(stashes, "")

    async def test_merge_tool_fast_forwards_temporary_repo(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            base_head = _git_head(repo, _GIT_BINARY)
            _git(repo, _GIT_BINARY, "checkout", "-b", "merge-source")
            (repo / "merge.txt").write_text("merged\n")
            _git(repo, _GIT_BINARY, "add", "merge.txt")
            _git(repo, _GIT_BINARY, "commit", "-m", "merge source")
            source_head = _git_head(repo, _GIT_BINARY)
            _git(repo, _GIT_BINARY, "checkout", "main")

            result = await _call_tool(
                toolset,
                "git_merge",
                revision="merge-source",
                confirm_revision="merge-source",
            )

            head = _git_head(repo, _GIT_BINARY)
            current_branch = _git_output(
                repo,
                _GIT_BINARY,
                "branch",
                "--show-current",
            ).strip()
            merged_text = (repo / "merge.txt").read_text()

        _assert_history_success(self, result)
        self.assertNotEqual(head, base_head)
        self.assertEqual(head, source_head)
        self.assertEqual(current_branch, "main")
        self.assertEqual(merged_text, "merged\n")

    async def test_rebase_tool_rewrites_temporary_repo_branch(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            _git(repo, _GIT_BINARY, "checkout", "-b", "feature-rebase")
            (repo / "feature.txt").write_text("feature\n")
            _git(repo, _GIT_BINARY, "add", "feature.txt")
            _git(repo, _GIT_BINARY, "commit", "-m", "feature rebase")
            feature_head_before = _git_head(repo, _GIT_BINARY)
            _git(repo, _GIT_BINARY, "checkout", "main")
            (repo / "main.txt").write_text("main\n")
            _git(repo, _GIT_BINARY, "add", "main.txt")
            _git(repo, _GIT_BINARY, "commit", "-m", "main rebase")
            main_head = _git_head(repo, _GIT_BINARY)

            result = await _call_tool(
                toolset,
                "git_rebase",
                upstream="main",
                confirm_upstream="main",
                branch="feature-rebase",
            )

            feature_head_after = _git_head(repo, _GIT_BINARY)
            merge_base = _git_output(
                repo,
                _GIT_BINARY,
                "merge-base",
                "main",
                "feature-rebase",
            ).strip()
            current_branch = _git_output(
                repo,
                _GIT_BINARY,
                "branch",
                "--show-current",
            ).strip()
            main_text = (repo / "main.txt").read_text()
            feature_text = (repo / "feature.txt").read_text()

        _assert_history_success(self, result)
        self.assertEqual(current_branch, "feature-rebase")
        self.assertEqual(merge_base, main_head)
        self.assertNotEqual(feature_head_after, feature_head_before)
        self.assertEqual(main_text, "main\n")
        self.assertEqual(feature_text, "feature\n")

    async def test_cherry_pick_tool_applies_temporary_repo_commit(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            _git(repo, _GIT_BINARY, "checkout", "-b", "pick-source")
            (repo / "picked.txt").write_text("picked\n")
            _git(repo, _GIT_BINARY, "add", "picked.txt")
            _git(repo, _GIT_BINARY, "commit", "-m", "pick source")
            pick_commit = _git_head(repo, _GIT_BINARY)
            _git(repo, _GIT_BINARY, "checkout", "main")

            result = await _call_tool(
                toolset,
                "git_cherry_pick",
                revision=pick_commit,
                confirm_revision=pick_commit,
            )

            head_tree = _git_output(
                repo,
                _GIT_BINARY,
                "show",
                "--format=%T",
                "--no-patch",
                "HEAD",
            ).strip()
            pick_tree = _git_output(
                repo,
                _GIT_BINARY,
                "show",
                "--format=%T",
                "--no-patch",
                pick_commit,
            ).strip()
            subject = _git_output(
                repo,
                _GIT_BINARY,
                "log",
                "--format=%s",
                "-1",
            ).strip()
            picked_text = (repo / "picked.txt").read_text()
            current_branch = _git_output(
                repo,
                _GIT_BINARY,
                "branch",
                "--show-current",
            ).strip()

        _assert_history_success(self, result)
        self.assertEqual(current_branch, "main")
        self.assertEqual(subject, "pick source")
        self.assertEqual(picked_text, "picked\n")
        self.assertEqual(head_tree, pick_tree)

    async def test_revert_tool_confirms_and_reverts_temporary_repo_commit(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)
            base_head = _git_head(repo, _GIT_BINARY)

            (repo / "tracked.txt").write_text("revert target\n")
            _git(repo, _GIT_BINARY, "add", "tracked.txt")
            _git(repo, _GIT_BINARY, "commit", "-m", "revert target")
            revert_target = _git_head(repo, _GIT_BINARY)

            denied = await _call_tool(
                toolset,
                "git_revert",
                revision=revert_target,
                confirm_revision=base_head,
            )
            head_after_denial = _git_head(repo, _GIT_BINARY)
            text_after_denial = (repo / "tracked.txt").read_text()

            result = await _call_tool(
                toolset,
                "git_revert",
                revision=revert_target,
                confirm_revision=revert_target,
            )

            head_after_revert = _git_head(repo, _GIT_BINARY)
            parent = _git_output(
                repo,
                _GIT_BINARY,
                "rev-parse",
                "HEAD^",
            ).strip()
            subject = _git_output(
                repo,
                _GIT_BINARY,
                "log",
                "--format=%s",
                "-1",
            ).strip()
            reverted_text = (repo / "tracked.txt").read_text()

        self.assertEqual(
            denied.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(head_after_denial, revert_target)
        self.assertEqual(text_after_denial, "revert target\n")
        _assert_history_success(self, result)
        self.assertNotEqual(head_after_revert, revert_target)
        self.assertEqual(parent, revert_target)
        self.assertEqual(subject, 'Revert "revert target"')
        self.assertEqual(reverted_text, "base\n")


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
    request = ShellGitCommandRequest(
        tool_name=f"shell.git_{command.value.replace('-', '_')}",
        command=command,
        capability_required=ShellGitCapability.READ,
        options={} if options is None else options,
        pathspecs=pathspecs,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )
    return ShellGitCommandRequest(
        tool_name=request.tool_name,
        command=request.command,
        capability_required=shell_git_capability_for_request(request),
        options=request.options,
        pathspecs=request.pathspecs,
        timeout_seconds=request.timeout_seconds,
        max_stdout_bytes=request.max_stdout_bytes,
        max_stderr_bytes=request.max_stderr_bytes,
    )


def _policy(
    workspace_root: Path,
    *,
    capabilities: tuple[str, ...] = ("history",),
    allowed_commands: tuple[str, ...] = _PHASE6_COMMANDS,
    max_commit_message_bytes: int = 4096,
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=str(workspace_root),
                cwd="repo",
                capabilities=capabilities,
                allowed_commands=allowed_commands,
                max_commit_message_bytes=max_commit_message_bytes,
                allowed_remote_hosts=("github.com",),
            ),
        ),
        executable_lookup=_fake_executable,
    )


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
    (repo / "tracked.txt").write_text("base\n")
    (repo / "scratch.txt").write_text("scratch\n")
    (git_dir / "objects" / "info").mkdir(parents=True)
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "refs" / "tags").mkdir(parents=True)
    (git_dir / "logs" / "refs").mkdir(parents=True)
    (git_dir / "refs" / "heads" / "main").write_text(f"{_HEAD}\n")
    (git_dir / "refs" / "heads" / "feature").write_text(f"{_SIDE}\n")
    (git_dir / "refs" / "heads" / "old").write_text(f"{_OLD}\n")
    (git_dir / "refs" / "heads" / "dup").write_text(f"{_SIDE}\n")
    (git_dir / "refs" / "heads" / "nested").mkdir()
    (git_dir / "refs" / "heads" / "nested" / "branch").write_text(f"{_SIDE}\n")
    (git_dir / "refs" / "heads" / "dangling").symlink_to("missing")
    (git_dir / "refs" / "tags" / "v0.1").write_text(f"{_HEAD}\n")
    (git_dir / "refs" / "tags" / "dup").write_text(f"{_HEAD}\n")
    (git_dir / "refs" / "stash").write_text(f"{_HEAD}\n")
    (git_dir / "logs" / "refs" / "stash").write_text(
        f"{_OLD} {_HEAD} Test <test@example.invalid> 0 +0000\tstash@{{1}}\n"
        f"{_HEAD} {_SIDE} Test <test@example.invalid> 1 +0000\tstash@{{0}}\n"
    )
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        "\n"
        f"{_SIDE} refs/heads/packed\n"
        f"{_HEAD} refs/tags/packed-tag\n"
        f"^{_OLD}\n"
        "malformed\n"
    )
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    )
    return repo


def _fake_toolset(
    root: Path,
    executor: "_FakeGitExecutor",
    *,
    max_commit_message_bytes: int = 4096,
) -> ShellToolSet:
    return ShellToolSet(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=str(root),
                cwd="repo",
                capabilities=("history",),
                allowed_commands=_PHASE6_COMMANDS,
                max_commit_message_bytes=max_commit_message_bytes,
            ),
        ),
        executor=executor,
    ).with_enabled_tools(list(_PHASE6_TOOL_NAMES))


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
                capabilities=("history",),
                allowed_commands=_PHASE6_COMMANDS,
            ),
        )
    ).with_enabled_tools(list(_PHASE6_TOOL_NAMES))


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
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._status = status
        self._exit_code = exit_code
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
        )


def _execution_result(
    spec: ExecutionSpec,
    *,
    status: ShellExecutionStatus,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 128,
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
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        duration_ms=3,
        error_message=None,
        metadata=spec.metadata,
    )


def _write_real_git_repo(root: Path, git_binary: str) -> Path:
    repo = root / "repo"
    _git(root, git_binary, "init", "repo")
    _git(repo, git_binary, "checkout", "-b", "main")
    _git(repo, git_binary, "config", "user.name", "Avalan Test")
    _git(repo, git_binary, "config", "user.email", "avalan@example.test")
    (repo / "tracked.txt").write_text("base\n")
    _git(repo, git_binary, "add", "tracked.txt")
    _git(repo, git_binary, "commit", "-m", "phase6 setup")
    return repo


def _git(cwd: Path, git_binary: str, *args: str) -> None:
    _git_run(cwd, git_binary, *args)


def _git_head(cwd: Path, git_binary: str) -> str:
    return _git_output(cwd, git_binary, "rev-parse", "HEAD").strip()


def _git_output(cwd: Path, git_binary: str, *args: str) -> str:
    return _git_run(cwd, git_binary, *args).stdout


def _assert_history_success(
    test_case: TestCase,
    result: ShellGitFormattedResult,
) -> None:
    test_case.assertEqual(
        result.git_result.status,
        ShellGitExecutionStatus.SUCCESS,
    )
    test_case.assertEqual(result.git_result.error_code, None)
    test_case.assertEqual(
        result.git_result.capability_used,
        ShellGitCapability.HISTORY,
    )
    test_case.assertEqual(
        result.git_result.audit_metadata["git_mutation_scope"],
        "history",
    )


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
