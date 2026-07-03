from collections.abc import Awaitable, Callable
from os import devnull, environ
from pathlib import Path
from shutil import which
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, skipIf

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
)
from avalan.tool.shell.git_policy import GitExecutionPolicy
from avalan.tool.shell.settings import ShellGitToolSettings, ShellToolSettings
from avalan.tool.shell.toolset import ShellToolSet

_GIT_BINARY = which("git")
_PHASE3_COMMANDS = (
    "status",
    "rev-parse",
    "branch",
    "tag",
    "describe",
    "ls-files",
    "log",
)
_PHASE3_TOOL_NAMES = [
    "shell.git_status",
    "shell.git_rev_parse",
    "shell.git_branch",
    "shell.git_tag",
    "shell.git_describe",
    "shell.git_ls_files",
    "shell.git_log",
]
_GitToolCallable = Callable[..., Awaitable[str]]


class GitMetadataPolicyPhase3Test(IsolatedAsyncioTestCase):
    async def test_metadata_commands_build_fixed_argv(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.STATUS,
                        options={
                            "mode": "porcelain_v2",
                            "include_branch": True,
                        },
                        pathspecs=("src",),
                    ),
                    (
                        "git",
                        "--no-pager",
                        "--no-optional-locks",
                        "status",
                        "--porcelain=v2",
                        "--branch",
                        "--untracked-files=all",
                        "--ignore-submodules=all",
                        "--",
                        "src",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.REV_PARSE,
                        options={"fact": "short_head"},
                    ),
                    (
                        "git",
                        "--no-pager",
                        "--no-optional-locks",
                        "rev-parse",
                        "--short=12",
                        "--verify",
                        "HEAD^{commit}",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.BRANCH,
                        options={"mode": "list", "contains": "HEAD"},
                    ),
                    (
                        "git",
                        "--no-pager",
                        "--no-optional-locks",
                        "branch",
                        "--no-color",
                        "--list",
                        "--format=%(refname:short)",
                        "--contains",
                        "HEAD",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.TAG,
                        options={"mode": "list", "max_count": 3},
                    ),
                    (
                        "git",
                        "--no-pager",
                        "--no-optional-locks",
                        "for-each-ref",
                        "--format=%(refname:short)",
                        "--sort=refname",
                        "--count=3",
                        "refs/tags",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.DESCRIBE,
                        options={
                            "mode": "always",
                            "target": "HEAD",
                            "max_candidates": 5,
                        },
                    ),
                    (
                        "git",
                        "--no-pager",
                        "--no-optional-locks",
                        "describe",
                        "--tags",
                        "--candidates=5",
                        "--always",
                        "HEAD",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.LS_FILES,
                        options={"mode": "others"},
                        pathspecs=("src",),
                    ),
                    (
                        "git",
                        "--no-pager",
                        "--no-optional-locks",
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                        "--",
                        "src",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.LOG,
                        options={
                            "max_count": 2,
                            "revision": "HEAD",
                            "format": "oneline",
                        },
                        pathspecs=("src",),
                    ),
                    (
                        "git",
                        "--no-pager",
                        "--no-optional-locks",
                        "log",
                        "--max-count=2",
                        "--no-decorate",
                        "--no-color",
                        "--no-ext-diff",
                        "--date=iso-strict",
                        "--format=%h %s",
                        "HEAD",
                        "--",
                        "src",
                    ),
                ),
            )

            for request, expected_argv in cases:
                with self.subTest(command=request.command.value):
                    spec = await policy.normalize(request)
                    self.assertEqual(spec.argv, expected_argv)
                    self.assertEqual(
                        spec.argv[:3],
                        ("git", "--no-pager", "--no-optional-locks"),
                    )

    async def test_metadata_modes_build_conservative_argv(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.REV_PARSE,
                        options={"fact": "head"},
                    ),
                    ("rev-parse", "--verify", "HEAD^{commit}"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.REV_PARSE,
                        options={"fact": "current_branch"},
                    ),
                    ("rev-parse", "--abbrev-ref", "HEAD"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.REV_PARSE,
                        options={"fact": "repo_root"},
                    ),
                    ("rev-parse", "--show-toplevel"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.BRANCH,
                        options={"mode": "current", "contains": None},
                    ),
                    ("branch", "--no-color", "--show-current"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.TAG,
                        options={
                            "mode": "show",
                            "name": "v1.0",
                            "max_count": 1,
                        },
                    ),
                    (
                        "for-each-ref",
                        (
                            "--format=%(refname:short)%09%(objectname:short)%09"
                            "%(subject)"
                        ),
                        "--count=1",
                        "refs/tags/v1.0",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.DESCRIBE,
                        options={"mode": "tags", "max_candidates": 0},
                    ),
                    ("describe", "--tags", "--candidates=0"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.LS_FILES,
                        options={"mode": "tracked"},
                    ),
                    ("ls-files", "--cached", "--deduplicate"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.LS_FILES,
                        options={"mode": "modified"},
                    ),
                    ("ls-files", "--modified", "--deduplicate"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.LS_FILES,
                        options={"mode": "deleted"},
                    ),
                    ("ls-files", "--deleted", "--deduplicate"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.LOG,
                        options={"max_count": 1, "format": "summary"},
                    ),
                    (
                        "log",
                        "--max-count=1",
                        "--no-decorate",
                        "--no-color",
                        "--no-ext-diff",
                        "--date=iso-strict",
                        "--format=%H%x09%an%x09%ae%x09%ad%x09%s",
                    ),
                ),
            )

            for request, expected_tail in cases:
                with self.subTest(command=request.command.value):
                    spec = await policy.normalize(request)
                    self.assertEqual(spec.argv[:3], _git_prefix())
                    self.assertEqual(spec.argv[3:], expected_tail)

    async def test_metadata_unknown_options_fail_closed(self) -> None:
        cases = (
            _request(
                command=ShellGitCommandName.STATUS,
                options={
                    "mode": "porcelain_v2",
                    "include_branch": True,
                    "ignored": "benign",
                },
            ),
            _request(
                command=ShellGitCommandName.STATUS,
                options={"mode": "short", "revision": 123},
            ),
            _request(
                command=ShellGitCommandName.REV_PARSE,
                options={"fact": "head", "ignored": "benign"},
            ),
            _request(
                command=ShellGitCommandName.BRANCH,
                options={"mode": "list", "ignored": "benign"},
            ),
            _request(
                command=ShellGitCommandName.TAG,
                options={"mode": "list", "ignored": "benign"},
            ),
            _request(
                command=ShellGitCommandName.DESCRIBE,
                options={"mode": "tags", "ignored": "benign"},
            ),
            _request(
                command=ShellGitCommandName.LS_FILES,
                options={"mode": "tracked", "ignored": "benign"},
            ),
            _request(
                command=ShellGitCommandName.LOG,
                options={
                    "max_count": 1,
                    "format": "summary",
                    "ignored": "benign",
                },
            ),
        )

        for request in cases:
            with self.subTest(command=request.command.value):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(_policy(root), request)
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.INVALID_OPTION,
                    )

    async def test_metadata_mutation_and_unsafe_forms_fail_closed(
        self,
    ) -> None:
        cases: tuple[
            tuple[ShellGitCommandRequest, ShellGitExecutionErrorCode],
            ...,
        ] = (
            (
                _request(
                    command=ShellGitCommandName.REV_PARSE,
                    options={"fact": "git_dir"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH,
                    options={"mode": "create"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH,
                    options={"mode": "current", "contains": "HEAD"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.BRANCH,
                    options={"mode": "list", "contains": 1},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG,
                    options={"mode": "create", "name": "v2.0"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG,
                    options={"mode": "list", "name": "v1.0"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG,
                    options={"mode": "show"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG,
                    options={"mode": "show", "name": 1},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.TAG,
                    options={"mode": "show", "name": "release/v1"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.DESCRIBE,
                    options={"mode": "dirty"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DESCRIBE,
                    options={"max_candidates": True},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.LS_FILES,
                    options={"mode": "submodules"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.LS_FILES,
                    options={"mode": "tracked", "recurse_submodules": True},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.LOG,
                    options={"max_count": 1, "format": "%H"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.LOG,
                    options={"max_count": 0, "format": "summary"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.LOG,
                    options={"max_count": 1, "revision": "HEAD@{1}"},
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

    async def test_metadata_count_boundaries_are_enforced(self) -> None:
        cases = (
            _request(
                command=ShellGitCommandName.TAG,
                options={"mode": "list", "max_count": 3},
            ),
            _request(
                command=ShellGitCommandName.DESCRIBE,
                options={"mode": "tags", "max_candidates": 3},
            ),
            _request(
                command=ShellGitCommandName.LOG,
                options={"max_count": 3, "format": "summary"},
            ),
        )

        for request in cases:
            with self.subTest(command=request.command.value):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(
                        _policy(root, max_log_count=2),
                        request,
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.INVALID_OPTION,
                    )

    async def test_metadata_pathspec_caps_are_enforced_for_log(self) -> None:
        cases = (
            (
                {"max_pathspecs": 1},
                ("src", "README.md"),
            ),
            (
                {"max_pathspec_bytes": 3},
                ("README.md",),
            ),
        )

        for policy_kwargs, pathspecs in cases:
            with self.subTest(policy_kwargs=policy_kwargs):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(
                        _policy(root, **policy_kwargs),
                        _request(
                            command=ShellGitCommandName.LOG,
                            options={"max_count": 1, "format": "summary"},
                            pathspecs=pathspecs,
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

    async def test_metadata_execution_caps_are_clamped(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            spec = await _policy(
                root,
                default_timeout_seconds=5,
                max_timeout_seconds=7,
                max_stdout_bytes=11,
                max_stderr_bytes=13,
            ).normalize(
                _request(
                    command=ShellGitCommandName.TAG,
                    options={"mode": "list"},
                    timeout_seconds=99,
                    max_stdout_bytes=99,
                    max_stderr_bytes=99,
                )
            )

        self.assertEqual(spec.timeout_seconds, 7)
        self.assertEqual(spec.max_stdout_bytes, 11)
        self.assertEqual(spec.max_stderr_bytes, 13)

    async def test_metadata_network_and_blob_commands_are_disabled(
        self,
    ) -> None:
        cases = (
            _request(
                command=ShellGitCommandName.FETCH,
                options={"url": "https://github.com/acme/repo.git"},
            ),
            _request(
                command=ShellGitCommandName.CLONE,
                options={"url": "https://github.com/acme/repo.git"},
            ),
            _request(
                command=ShellGitCommandName.SHOW,
                options={"revision": "HEAD:README.md"},
            ),
        )

        for request in cases:
            with self.subTest(command=request.command.value):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(_policy(root), request)
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.COMMAND_DISABLED,
                    )


class GitMetadataResultPhase3Test(IsolatedAsyncioTestCase):
    async def test_ambiguous_revision_result_is_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd="repo",
                        allowed_commands=_PHASE3_COMMANDS,
                    ),
                ),
                executor=_FakeGitExecutor(
                    "warning: refname 'topic' is ambiguous.\n"
                    "fatal: ambiguous object name: 'topic'\n"
                ),
            ).with_enabled_tools(["shell.git_log"])

            result = await _call_tool(
                toolset,
                "git_log",
                max_count=1,
                revision="topic",
            )

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.AMBIGUOUS_REVISION,
        )
        self.assertIn("error_code: ambiguous_revision", result)
        self.assertIn("Git revision is ambiguous", result)

    async def test_revision_not_found_result_is_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd="repo",
                        allowed_commands=_PHASE3_COMMANDS,
                    ),
                ),
                executor=_FakeGitExecutor("fatal: bad revision 'BADREF'\n"),
            ).with_enabled_tools(["shell.git_log"])

            result = await _call_tool(
                toolset,
                "git_log",
                max_count=1,
                revision="BADREF",
            )

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        )
        self.assertIn("error_code: revision_not_found", result)
        self.assertIn("Git revision was not found", result)

    async def test_unclassified_shell_failure_is_nonzero_exit(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd="repo",
                        allowed_commands=_PHASE3_COMMANDS,
                    ),
                ),
                executor=_FakeGitExecutor(
                    "tool execution failed\n",
                    status=ShellExecutionStatus.TOOL_ERROR,
                ),
            ).with_enabled_tools(["shell.git_status"])

            result = await _call_tool(toolset, "git_status")

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.NONZERO_EXIT,
        )
        self.assertIn("error_code: nonzero_exit", result)

    async def test_blob_and_network_like_options_denied_before_execution(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor()
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd="repo",
                        allowed_commands=_PHASE3_COMMANDS,
                    ),
                ),
                executor=executor,
            ).with_enabled_tools(_PHASE3_TOOL_NAMES)

            cases: tuple[
                tuple[
                    str,
                    dict[str, object],
                    ShellGitExecutionErrorCode,
                ],
                ...,
            ] = (
                (
                    "git_log",
                    {"max_count": 1, "revision": "HEAD:README.md"},
                    ShellGitExecutionErrorCode.REVISION_DENIED,
                ),
                (
                    "git_log",
                    {
                        "max_count": 1,
                        "revision": "https://github.com/acme/repo",
                    },
                    ShellGitExecutionErrorCode.REVISION_DENIED,
                ),
                (
                    "git_describe",
                    {"target": "HEAD:README.md"},
                    ShellGitExecutionErrorCode.REVISION_DENIED,
                ),
                (
                    "git_branch",
                    {"mode": "list", "contains": "origin/main"},
                    ShellGitExecutionErrorCode.REVISION_DENIED,
                ),
                (
                    "git_tag",
                    {"mode": "show", "name": "HEAD:README.md"},
                    ShellGitExecutionErrorCode.REVISION_DENIED,
                ),
                (
                    "git_rev_parse",
                    {"fact": "git_dir"},
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
            )

            for command_id, kwargs, error_code in cases:
                with self.subTest(command_id=command_id, kwargs=kwargs):
                    result = await _call_tool(toolset, command_id, **kwargs)
                    self.assertEqual(
                        result.git_result.status,
                        ShellGitExecutionStatus.POLICY_DENIED,
                    )
                    self.assertEqual(result.git_result.error_code, error_code)
                    self.assertIn("status: policy_denied", result)
                    self.assertEqual(executor.calls, 0)

    async def test_metadata_timeout_result_is_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                status=ShellExecutionStatus.TIMEOUT,
                exit_code=None,
                timed_out=True,
                error_message="process timed out",
            )
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd="repo",
                        allowed_commands=_PHASE3_COMMANDS,
                    ),
                ),
                executor=executor,
            ).with_enabled_tools(["shell.git_tag"])

            result = await _call_tool(toolset, "git_tag", mode="list")

        self.assertEqual(executor.calls, 1)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.TIMEOUT,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.TIMEOUT,
        )
        self.assertTrue(result.git_result.timed_out)
        self.assertIn("error_code: timeout", result)

    async def test_metadata_output_truncated_result_is_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                stdout="v1.0\n",
                status=ShellExecutionStatus.COMPLETED,
                exit_code=0,
                stdout_truncated=True,
            )
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd="repo",
                        allowed_commands=_PHASE3_COMMANDS,
                    ),
                ),
                executor=executor,
            ).with_enabled_tools(["shell.git_tag"])

            result = await _call_tool(toolset, "git_tag", mode="list")

        self.assertEqual(executor.calls, 1)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.OUTPUT_TRUNCATED,
        )
        self.assertTrue(result.git_result.stdout_truncated)
        self.assertIn("error_code: output_truncated", result)


@skipIf(_GIT_BINARY is None, "git executable is not available")
class GitMetadataSmokePhase3Test(IsolatedAsyncioTestCase):
    async def test_metadata_tools_execute_against_temporary_repo(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            status = await _call_tool(toolset, "git_status")
            rev_parse = await _call_tool(
                toolset,
                "git_rev_parse",
                fact="current_branch",
            )
            branch = await _call_tool(toolset, "git_branch")
            tag = await _call_tool(
                toolset,
                "git_tag",
                mode="list",
                max_count=2,
            )
            describe = await _call_tool(
                toolset,
                "git_describe",
                mode="always",
            )
            ls_files = await _call_tool(toolset, "git_ls_files")
            log = await _call_tool(
                toolset,
                "git_log",
                max_count=2,
                format="oneline",
            )
            bad_ref = await _call_tool(
                toolset,
                "git_log",
                max_count=1,
                revision="BADREF",
            )

        for result in (
            status,
            rev_parse,
            branch,
            tag,
            describe,
            ls_files,
            log,
        ):
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertIn("status: success", result)

        self.assertIn("# branch.head main", status)
        self.assertIn("main", rev_parse)
        self.assertIn("main", branch)
        self.assertIn("v1.0", tag)
        self.assertIn("v1.0", describe)
        self.assertIn("README.md", ls_files)
        self.assertIn("src/app.py", ls_files)
        self.assertIn("second", log)
        self.assertIn("initial", log)
        self.assertEqual(
            bad_ref.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            bad_ref.git_result.error_code,
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        )
        self.assertIn("status: failed", bad_ref)
        self.assertIn("error_code: revision_not_found", bad_ref)

    async def test_rev_parse_facts_execute_against_temporary_repo(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            head = await _call_tool(toolset, "git_rev_parse", fact="head")
            short_head = await _call_tool(
                toolset,
                "git_rev_parse",
                fact="short_head",
            )
            repo_root = await _call_tool(
                toolset,
                "git_rev_parse",
                fact="repo_root",
            )

        for result in (head, short_head, repo_root):
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )

        head_text = head.git_result.stdout_snippet.strip()
        short_head_text = short_head.git_result.stdout_snippet.strip()
        repo_root_text = repo_root.git_result.stdout_snippet.strip()
        self.assertRegex(head_text, r"^[0-9a-fA-F]{40,64}$")
        self.assertRegex(short_head_text, r"^[0-9a-fA-F]{12}$")
        self.assertTrue(head_text.lower().startswith(short_head_text.lower()))
        self.assertEqual(Path(repo_root_text).resolve(), repo.resolve())

    async def test_tag_show_executes_against_temporary_repo(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            tag = await _call_tool(
                toolset,
                "git_tag",
                mode="show",
                name="v1.0",
                max_count=1,
            )

        self.assertEqual(
            tag.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        self.assertIn("v1.0\t", tag.git_result.stdout_snippet)
        self.assertIn("initial", tag.git_result.stdout_snippet)

    async def test_branch_contains_executes_against_temporary_repo(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            _git(repo, _GIT_BINARY, "branch", "topic")
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            branches = await _call_tool(
                toolset,
                "git_branch",
                mode="list",
                contains="HEAD",
            )

        self.assertEqual(
            branches.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        branch_names = set(
            branches.git_result.stdout_snippet.strip().splitlines()
        )
        self.assertIn("main", branch_names)
        self.assertIn("topic", branch_names)

    async def test_ls_files_modes_execute_against_worktree_states(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            (repo / "src" / "app.py").write_text("print('changed')\n")
            (repo / "README.md").unlink()
            (repo / "scratch.txt").write_text("draft\n")
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            tracked = await _call_tool(
                toolset,
                "git_ls_files",
                mode="tracked",
            )
            modified = await _call_tool(
                toolset,
                "git_ls_files",
                mode="modified",
            )
            deleted = await _call_tool(
                toolset,
                "git_ls_files",
                mode="deleted",
            )
            others = await _call_tool(
                toolset,
                "git_ls_files",
                mode="others",
            )

        for result in (tracked, modified, deleted, others):
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )

        self.assertIn("README.md", tracked.git_result.stdout_snippet)
        self.assertIn("src/app.py", tracked.git_result.stdout_snippet)
        self.assertIn("src/app.py", modified.git_result.stdout_snippet)
        self.assertIn("README.md", deleted.git_result.stdout_snippet)
        self.assertIn("scratch.txt", others.git_result.stdout_snippet)
        self.assertNotIn("README.md", others.git_result.stdout_snippet)


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
                allowed_commands=_PHASE3_COMMANDS,
            ),
        )
    ).with_enabled_tools(_PHASE3_TOOL_NAMES)


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
        capability_required=ShellGitCapability.READ,
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
    max_log_count: int = 50,
    max_pathspecs: int = 64,
    max_pathspec_bytes: int = 4096,
    max_timeout_seconds: float = 60.0,
    max_stdout_bytes: int = 65536,
    max_stderr_bytes: int = 32768,
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=str(workspace_root),
                cwd="repo",
                allowed_commands=_PHASE3_COMMANDS,
                default_timeout_seconds=default_timeout_seconds,
                max_log_count=max_log_count,
                max_pathspecs=max_pathspecs,
                max_pathspec_bytes=max_pathspec_bytes,
                max_timeout_seconds=max_timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            )
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
    (git_dir / "objects" / "info").mkdir(parents=True)
    (git_dir / "refs").mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    )
    return repo


def _write_real_git_repo(root: Path, git_binary: str) -> Path:
    repo = root / "repo"
    _git(root, git_binary, "init", "repo")
    _git(repo, git_binary, "checkout", "-b", "main")
    _git(repo, git_binary, "config", "user.name", "Avalan Test")
    _git(repo, git_binary, "config", "user.email", "avalan@example.test")
    (repo / "README.md").write_text("hello\n")
    _git(repo, git_binary, "add", "README.md")
    _git(repo, git_binary, "commit", "-m", "initial")
    _git(repo, git_binary, "tag", "v1.0")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('hello')\n")
    _git(repo, git_binary, "add", "src/app.py")
    _git(repo, git_binary, "commit", "-m", "second")
    return repo


def _git(cwd: Path, git_binary: str, *args: str) -> None:
    isolation_root = _git_isolation_root(cwd)
    env = _git_env(isolation_root)
    run(
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
        env=env,
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


class _FakeGitExecutor:
    def __init__(
        self,
        stderr: str = "",
        *,
        stdout: str = "",
        status: ShellExecutionStatus = ShellExecutionStatus.NONZERO_EXIT,
        exit_code: int | None = 128,
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        timed_out: bool = False,
        error_message: str | None = None,
    ) -> None:
        self._stderr = stderr
        self._stdout = stdout
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
            return cast(_GitToolCallable, tool)
    raise AssertionError(f"tool {command_id} was not enabled")
