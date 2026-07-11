from ast import (
    Call,
    Constant,
    Import,
    ImportFrom,
    keyword,
    parse,
    walk,
)
from asyncio import CancelledError
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import ToolCallContext, ToolExecutionStreamEvent
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
    ShellGitFormattedResult,
    ShellGitPolicyDenied,
)
from avalan.tool.shell.git_policy import GitExecutionPolicy
from avalan.tool.shell.settings import ShellGitToolSettings, ShellToolSettings
from avalan.tool.shell.toolset import ShellToolSet

_GitResultProvider = Callable[[ExecutionSpec], ExecutionResult]
_RepoWriter = Callable[[Path], None]


class GitExecutionPolicyPhase2Test(IsolatedAsyncioTestCase):
    async def test_safe_repo_discovery_normalizes_status_spec(self) -> None:
        with TemporaryDirectory() as workspace:
            repo = _write_minimal_git_repo(Path(workspace) / "repo")
            (repo / "src").mkdir()
            policy = _policy(workspace)

            spec = await policy.normalize(
                _request(cwd="repo", pathspecs=("src",))
            )

            self.assertEqual(Path(spec.cwd), repo.resolve())
            self.assertEqual(spec.display_cwd, "repo")
            self.assertEqual(
                spec.argv,
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
            )
            self.assertEqual(spec.executable, "/usr/bin/git")
            self.assertEqual(spec.metadata["git_repo_root"], "repo")

    async def test_non_repo_is_repo_not_found(self) -> None:
        with TemporaryDirectory() as workspace:
            (Path(workspace) / "plain").mkdir()
            policy = _policy(workspace, cwd="plain")

            with self.assertRaises(ShellGitPolicyDenied) as raised:
                await policy.normalize(_request())

            self.assertEqual(
                raised.exception.error_code,
                ShellGitExecutionErrorCode.REPO_NOT_FOUND,
            )

    async def test_repo_discovery_does_not_climb_above_workspace(self) -> None:
        with TemporaryDirectory() as parent:
            _write_minimal_git_repo(Path(parent))
            workspace = Path(parent) / "child"
            workspace.mkdir()
            policy = _policy(str(workspace))

            with self.assertRaises(ShellGitPolicyDenied) as raised:
                await policy.normalize(_request())

            self.assertEqual(
                raised.exception.error_code,
                ShellGitExecutionErrorCode.REPO_NOT_FOUND,
            )

    async def test_hostile_repo_forms_fail_closed(self) -> None:
        cases: tuple[
            tuple[str, _RepoWriter, ShellGitExecutionErrorCode],
            ...,
        ] = (
            (
                "bare",
                _write_bare_repo,
                ShellGitExecutionErrorCode.BARE_REPO_DENIED,
            ),
            (
                "git_file",
                _write_git_file_repo,
                ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
            ),
            (
                "alternates",
                _write_alternate_repo,
                ShellGitExecutionErrorCode.ALTERNATE_DENIED,
            ),
            (
                "linked",
                _write_linked_worktree_repo,
                ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
            ),
            (
                "config",
                _write_unsafe_config_repo,
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "attributes",
                _write_unsafe_attributes_repo,
                ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            ),
        )

        for name, writer, error_code in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    writer(Path(workspace) / "repo")
                    policy = _policy(workspace, cwd="repo")

                    with self.assertRaises(ShellGitPolicyDenied) as raised:
                        await policy.normalize(_request())

                    self.assertEqual(raised.exception.error_code, error_code)

    async def test_unsafe_pathspecs_fail_closed(self) -> None:
        pathspecs = (
            "../outside",
            "/absolute",
            ".git/config",
            ":magic",
            "-n",
            "src/{bad}",
        )

        for pathspec in pathspecs:
            with self.subTest(pathspec=pathspec):
                with TemporaryDirectory() as workspace:
                    _write_minimal_git_repo(Path(workspace) / "repo")
                    policy = _policy(workspace, cwd="repo")

                    with self.assertRaises(ShellGitPolicyDenied) as raised:
                        await policy.normalize(_request(pathspecs=(pathspec,)))

                    self.assertEqual(
                        raised.exception.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

    async def test_unsafe_revisions_fail_closed(self) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            policy = _policy(workspace, cwd="repo")

            with self.assertRaises(ShellGitPolicyDenied) as raised:
                await policy.normalize(
                    _request(
                        command=ShellGitCommandName.SHOW,
                        options={"revision": "HEAD:secret"},
                    )
                )

            self.assertEqual(
                raised.exception.error_code,
                ShellGitExecutionErrorCode.REVISION_DENIED,
            )

    async def test_unsupported_safe_commands_deny_without_executor_argv(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            policy = GitExecutionPolicy(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=workspace,
                        cwd="repo",
                        capabilities=("read",),
                        allowed_commands=("status",),
                    )
                ),
                executable_lookup=_fake_executable,
            )

            with self.assertRaises(ShellGitPolicyDenied) as raised:
                await policy.normalize(
                    _request(
                        command=ShellGitCommandName.SHOW,
                        options={"revision": "HEAD"},
                    )
                )

            self.assertEqual(
                raised.exception.error_code,
                ShellGitExecutionErrorCode.COMMAND_DISABLED,
            )

    async def test_global_options_are_denied_and_env_is_scrubbed(self) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            policy = _policy(workspace, cwd="repo")

            spec = await policy.normalize(_request())

            self.assertEqual(spec.env["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(spec.env["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(spec.env["GIT_ASKPASS"], "/nonexistent")
            self.assertEqual(spec.env["SSH_ASKPASS"], "/nonexistent")
            self.assertEqual(spec.env["GIT_SSH"], "/nonexistent")
            self.assertNotIn("GIT_CONFIG_PARAMETERS", spec.env)
            self.assertNotIn("GIT_SSH_COMMAND", spec.env)
            self.assertNotIn("-c", spec.argv)

            with self.assertRaises(ShellGitPolicyDenied) as raised:
                await policy.normalize(_request(options={"mode": "-c"}))

            self.assertEqual(
                raised.exception.error_code,
                ShellGitExecutionErrorCode.INVALID_OPTION,
            )

    async def test_remote_credentials_fail_closed(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=("remote-add",),
                allowed_remote_hosts=("github.com",),
            )
        )
        policy = GitExecutionPolicy(settings=settings)

        with self.assertRaises(ShellGitPolicyDenied) as raised:
            await policy.normalize(
                _request(
                    command=ShellGitCommandName.REMOTE_ADD,
                    options={
                        "name": "origin",
                        "url": "https://token@github.com/acme/repo.git",
                    },
                )
            )

        self.assertEqual(
            raised.exception.error_code,
            ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
        )

    async def test_limits_and_caps_are_enforced(self) -> None:
        await self._assert_policy_error(
            ShellToolSettings(git=ShellGitToolSettings(max_pathspecs=1)),
            _request(pathspecs=("one", "two")),
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        await self._assert_policy_error(
            ShellToolSettings(git=ShellGitToolSettings(max_pathspec_bytes=3)),
            _request(pathspecs=("long",)),
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        await self._assert_policy_error(
            ShellToolSettings(max_arguments=2),
            _request(),
            ShellGitExecutionErrorCode.INVALID_OPTION,
        )
        await self._assert_policy_error(
            ShellToolSettings(max_argument_bytes=2),
            _request(),
            ShellGitExecutionErrorCode.INVALID_OPTION,
        )
        await self._assert_policy_error(
            ShellToolSettings(max_command_bytes=4),
            _request(),
            ShellGitExecutionErrorCode.INVALID_OPTION,
        )

        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            settings = ShellToolSettings(
                git=ShellGitToolSettings(
                    workspace_root=workspace,
                    cwd="repo",
                    default_timeout_seconds=5,
                    max_timeout_seconds=7,
                    max_stdout_bytes=11,
                    max_stderr_bytes=12,
                )
            )
            spec = await GitExecutionPolicy(
                settings=settings,
                executable_lookup=_fake_executable,
            ).normalize(
                _request(
                    timeout_seconds=99,
                    max_stdout_bytes=999,
                    max_stderr_bytes=999,
                )
            )

            self.assertEqual(spec.timeout_seconds, 7)
            self.assertEqual(spec.max_stdout_bytes, 11)
            self.assertEqual(spec.max_stderr_bytes, 12)

    async def _assert_policy_error(
        self,
        settings: ShellToolSettings,
        request: ShellGitCommandRequest,
        error_code: ShellGitExecutionErrorCode,
    ) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            settings = _with_repo_settings(settings, workspace)
            policy = GitExecutionPolicy(
                settings=settings,
                executable_lookup=_fake_executable,
            )

            with self.assertRaises(ShellGitPolicyDenied) as raised:
                await policy.normalize(request)

            self.assertEqual(raised.exception.error_code, error_code)


class GitToolExecutionPhase2Test(IsolatedAsyncioTestCase):
    async def test_missing_git_is_stable_command_unavailable(self) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            tool = _status_tool(workspace)

            result = await tool(context=ToolCallContext())

            assert isinstance(result, ShellGitFormattedResult)
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.COMMAND_UNAVAILABLE,
            )
            self.assertEqual(
                result.git_result.error_code,
                ShellGitExecutionErrorCode.COMMAND_UNAVAILABLE,
            )
            self.assertIn("status: command_unavailable", result)

    async def test_success_result_uses_policy_spec(self) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")

            def result_provider(spec: ExecutionSpec) -> ExecutionResult:
                return _execution_result(spec, stdout="# branch.oid abc\n")

            executor = _FakeGitExecutor(result_provider)
            tool = _status_tool(workspace, executor=executor)

            result = await tool(context=ToolCallContext())

            assert isinstance(result, ShellGitFormattedResult)
            self.assertEqual(len(executor.specs), 1)
            self.assertEqual(
                executor.specs[0].argv[:4],
                ("git", "--no-pager", "--no-optional-locks", "status"),
            )
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertEqual(result.git_result.error_code, None)
            self.assertIn("# branch.oid abc", result)

    async def test_timeout_nonzero_and_truncation_are_stable(self) -> None:
        cases: tuple[
            tuple[
                str,
                ShellExecutionStatus,
                int | None,
                bool,
                bool,
                ShellGitExecutionStatus,
                ShellGitExecutionErrorCode,
            ],
            ...,
        ] = (
            (
                "timeout",
                ShellExecutionStatus.TIMEOUT,
                None,
                True,
                False,
                ShellGitExecutionStatus.TIMEOUT,
                ShellGitExecutionErrorCode.TIMEOUT,
            ),
            (
                "nonzero",
                ShellExecutionStatus.NONZERO_EXIT,
                2,
                False,
                False,
                ShellGitExecutionStatus.FAILED,
                ShellGitExecutionErrorCode.NONZERO_EXIT,
            ),
            (
                "truncated",
                ShellExecutionStatus.COMPLETED,
                0,
                False,
                True,
                ShellGitExecutionStatus.FAILED,
                ShellGitExecutionErrorCode.OUTPUT_TRUNCATED,
            ),
        )

        for (
            name,
            shell_status,
            exit_code,
            timed_out,
            stdout_truncated,
            git_status,
            error_code,
        ) in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    _write_minimal_git_repo(Path(workspace) / "repo")

                    def result_provider(
                        spec: ExecutionSpec,
                    ) -> ExecutionResult:
                        return _execution_result(
                            spec,
                            status=shell_status,
                            exit_code=exit_code,
                            timed_out=timed_out,
                            stdout_truncated=stdout_truncated,
                        )

                    executor = _FakeGitExecutor(result_provider)
                    tool = _status_tool(workspace, executor=executor)

                    result = await tool(context=ToolCallContext())

                    assert isinstance(result, ShellGitFormattedResult)
                    self.assertEqual(result.git_result.status, git_status)
                    self.assertEqual(result.git_result.error_code, error_code)

    async def test_cancellation_is_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            tool = _status_tool(
                workspace,
                executor=_FakeGitExecutor(CancelledError()),
            )

            result = await tool(context=ToolCallContext())

            assert isinstance(result, ShellGitFormattedResult)
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.CANCELLED,
            )
            self.assertTrue(result.git_result.cancelled)
            self.assertIn("status: cancelled", result)

    async def test_policy_denial_does_not_call_executor(self) -> None:
        with TemporaryDirectory() as workspace:

            def result_provider(spec: ExecutionSpec) -> ExecutionResult:
                return _execution_result(spec)

            executor = _FakeGitExecutor(result_provider)
            tool = _status_tool(workspace, executor=executor)

            result = await tool(context=ToolCallContext())

            assert isinstance(result, ShellGitFormattedResult)
            self.assertEqual(executor.specs, [])
            self.assertEqual(
                result.git_result.error_code,
                ShellGitExecutionErrorCode.REPO_NOT_FOUND,
            )

    async def test_git_output_and_metadata_are_redacted(self) -> None:
        secret_url = "https://token@github.com/acme/private.git"

        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")

            def result_provider(spec: ExecutionSpec) -> ExecutionResult:
                return _execution_result(
                    spec,
                    stdout=f"origin {secret_url}\n",
                    stderr=f"diagnostic {secret_url}\n",
                    error_message=f"failed {secret_url}",
                    metadata={"remote": secret_url, "mirrors": [secret_url]},
                )

            executor = _FakeGitExecutor(result_provider)
            tool = _status_tool(workspace, executor=executor)

            result = await tool(context=ToolCallContext())

            assert isinstance(result, ShellGitFormattedResult)
            self.assertNotIn("token", result)
            self.assertNotIn("private.git", result)
            self.assertNotIn("token", str(result.git_result.audit_metadata))
            metadata = str(result.git_result.audit_metadata)
            self.assertNotIn("private.git", metadata)
            self.assertIn("https://github.com/[redacted]", result)


class GitGuardrailPhase2Test(TestCase):
    def test_git_implementation_does_not_bypass_command_executor(self) -> None:
        for source in _git_sources():
            tree = parse(source.read_text())
            for node in walk(tree):
                if isinstance(node, Import):
                    self.assertNotIn(
                        "subprocess",
                        {alias.name for alias in node.names},
                    )
                if isinstance(node, ImportFrom):
                    self.assertNotEqual(node.module, "subprocess")
                if isinstance(node, Call):
                    self.assertNotIn(_call_name(node), _SUBPROCESS_CALLS)

    def test_git_implementation_does_not_request_shell_eval(self) -> None:
        for source in _git_sources():
            tree = parse(source.read_text())
            for node in walk(tree):
                if isinstance(node, Call):
                    self.assertNotEqual(
                        _call_name(node),
                        "create_subprocess_shell",
                    )
                    for item in node.keywords:
                        self.assertFalse(_is_shell_true_keyword(item))


class _FakeGitExecutor:
    specs: list[ExecutionSpec]

    def __init__(
        self,
        result: ExecutionResult | CancelledError | _GitResultProvider,
    ) -> None:
        self._result = result
        self.specs = []

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self.specs.append(spec)
        if isinstance(self._result, CancelledError):
            raise self._result
        if callable(self._result):
            result = self._result(spec)
        else:
            result = self._result
        assert isinstance(result, ExecutionResult)
        return result


async def _fake_executable(search_paths: tuple[str, ...]) -> str | None:
    assert isinstance(search_paths, tuple)
    return "/usr/bin/git"


def _policy(
    workspace: str,
    *,
    cwd: str = "repo",
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=workspace,
                cwd=cwd,
            )
        ),
        executable_lookup=_fake_executable,
    )


def _request(
    *,
    command: ShellGitCommandName = ShellGitCommandName.STATUS,
    options: dict[str, object] | None = None,
    pathspecs: tuple[str, ...] = (),
    cwd: str | None = None,
    timeout_seconds: float | None = None,
    max_stdout_bytes: int | None = None,
    max_stderr_bytes: int | None = None,
) -> ShellGitCommandRequest:
    return ShellGitCommandRequest(
        tool_name=f"shell.git_{command.value.replace('-', '_')}",
        command=command,
        capability_required=_capability(command),
        options=options or {"mode": "porcelain_v2"},
        pathspecs=pathspecs,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _capability(command: ShellGitCommandName) -> ShellGitCapability:
    if command in {
        ShellGitCommandName.ADD,
        ShellGitCommandName.RESTORE,
        ShellGitCommandName.CHECKOUT,
        ShellGitCommandName.SWITCH,
        ShellGitCommandName.RESET,
        ShellGitCommandName.RM,
        ShellGitCommandName.MV,
        ShellGitCommandName.STASH_PUSH,
        ShellGitCommandName.STASH_APPLY,
    }:
        return ShellGitCapability.WORKTREE
    if command in {
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
    }:
        return ShellGitCapability.REMOTE
    if command in {
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
    }:
        return ShellGitCapability.HISTORY
    return ShellGitCapability.READ


def _with_repo_settings(
    settings: ShellToolSettings,
    workspace: str,
) -> ShellToolSettings:
    git_settings = cast(ShellGitToolSettings, settings.git)
    return ShellToolSettings(
        max_arguments=settings.max_arguments,
        max_argument_bytes=settings.max_argument_bytes,
        max_command_bytes=settings.max_command_bytes,
        git=ShellGitToolSettings(
            workspace_root=workspace,
            cwd="repo",
            max_pathspecs=git_settings.max_pathspecs,
            max_pathspec_bytes=git_settings.max_pathspec_bytes,
        ),
    )


def _status_tool(
    workspace: str,
    *,
    executor: _FakeGitExecutor | None = None,
) -> Any:
    settings = ShellToolSettings(
        git=ShellGitToolSettings(
            workspace_root=workspace,
            cwd="repo",
        )
    )
    toolset = ShellToolSet(
        settings=settings,
        executor=executor,
    ).with_enabled_tools(["shell.git_status"])
    return cast(Any, toolset.tools[0])


def _execution_result(
    spec: ExecutionSpec,
    *,
    status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
    exit_code: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
    error_message: str | None = None,
    metadata: dict[str, object] | None = None,
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
        cancelled=status is ShellExecutionStatus.CANCELLED,
        duration_ms=3,
        error_message=error_message,
        metadata=metadata or {},
    )


def _write_minimal_git_repo(repo: Path) -> Path:
    git_dir = repo / ".git"
    (git_dir / "objects" / "info").mkdir(parents=True)
    (git_dir / "refs").mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    )
    return repo


def _write_bare_repo(repo: Path) -> None:
    repo.mkdir()
    (repo / "objects").mkdir()
    (repo / "refs").mkdir()
    (repo / "HEAD").write_text("ref: refs/heads/main\n")


def _write_git_file_repo(repo: Path) -> None:
    repo.mkdir()
    (repo / ".git").write_text("gitdir: ../outside.git\n")


def _write_alternate_repo(repo: Path) -> None:
    _write_minimal_git_repo(repo)
    alternates = repo / ".git" / "objects" / "info" / "alternates"
    alternates.write_text("/outside/objects\n")


def _write_linked_worktree_repo(repo: Path) -> None:
    _write_minimal_git_repo(repo)
    (repo / ".git" / "commondir").write_text("../common.git\n")


def _write_unsafe_config_repo(repo: Path) -> None:
    _write_minimal_git_repo(repo)
    (repo / ".git" / "config").write_text("[core]\n\tfsmonitor = dangerous\n")


def _write_unsafe_attributes_repo(repo: Path) -> None:
    _write_minimal_git_repo(repo)
    (repo / ".gitattributes").write_text("*.bin diff=external\n")


def _git_sources() -> tuple[Path, ...]:
    root = Path(__file__).parents[3]
    tools_directory = root / "src" / "avalan" / "tool" / "shell" / "tools"
    return (
        root / "src" / "avalan" / "tool" / "shell" / "git.py",
        root / "src" / "avalan" / "tool" / "shell" / "git_policy.py",
        *sorted(tools_directory.rglob("*.py")),
        root / "src" / "avalan" / "tool" / "shell" / "toolset.py",
    )


_SUBPROCESS_CALLS = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "create_subprocess_exec",
    "create_subprocess_shell",
    "run",
    "system",
}


def _call_name(node: Call) -> str:
    function = node.func
    name = getattr(function, "attr", None)
    if isinstance(name, str):
        return name
    name = getattr(function, "id", None)
    return name if isinstance(name, str) else ""


def _is_shell_true_keyword(node: keyword) -> bool:
    return (
        node.arg == "shell"
        and isinstance(node.value, Constant)
        and node.value.value is True
    )


if __name__ == "__main__":
    main()
