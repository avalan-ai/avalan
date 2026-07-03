from collections.abc import Awaitable, Callable
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


@skipIf(_GIT_BINARY is None, "git executable is not available")
class GitMetadataSmokePhase3Test(IsolatedAsyncioTestCase):
    async def test_metadata_tools_execute_against_temporary_repo(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY)
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    executable_search_paths=(str(Path(_GIT_BINARY).parent),),
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd=repo.name,
                        allowed_commands=_PHASE3_COMMANDS,
                    ),
                )
            ).with_enabled_tools(_PHASE3_TOOL_NAMES)

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


def _git_prefix() -> tuple[str, str, str]:
    return ("git", "--no-pager", "--no-optional-locks")


def _request(
    *,
    command: ShellGitCommandName,
    options: dict[str, object] | None = None,
    pathspecs: tuple[str, ...] = (),
) -> ShellGitCommandRequest:
    return ShellGitCommandRequest(
        tool_name=f"shell.git_{command.value.replace('-', '_')}",
        command=command,
        capability_required=ShellGitCapability.READ,
        options={} if options is None else options,
        pathspecs=pathspecs,
    )


def _policy(
    workspace_root: Path,
    *,
    max_log_count: int = 50,
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=str(workspace_root),
                cwd="repo",
                allowed_commands=_PHASE3_COMMANDS,
                max_log_count=max_log_count,
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
    run(
        (git_binary, *args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


class _FakeGitExecutor:
    def __init__(
        self,
        stderr: str,
        *,
        status: ShellExecutionStatus = ShellExecutionStatus.NONZERO_EXIT,
    ) -> None:
        self._stderr = stderr
        self._status = status

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        assert stream is None
        return _execution_result(
            spec,
            status=self._status,
            stderr=self._stderr,
        )


def _execution_result(
    spec: ExecutionSpec,
    *,
    status: ShellExecutionStatus,
    stderr: str,
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
        exit_code=128,
        stdout="",
        stderr=stderr,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        stdout_bytes=0,
        stderr_bytes=len(stderr.encode("utf-8")),
        duration_ms=3,
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
