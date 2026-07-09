from collections.abc import Awaitable, Callable
from os import devnull, environ
from pathlib import Path
from shutil import which
from subprocess import DEVNULL, CompletedProcess, run
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, skipIf
from unittest.mock import patch

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
from avalan.tool.shell.git_policy import (
    GitExecutionPolicy,
    _validate_file_remote_url,
    git_remote_audit_metadata,
)
from avalan.tool.shell.settings import ShellGitToolSettings, ShellToolSettings
from avalan.tool.shell.toolset import ShellToolSet

_GIT_BINARY = which("git")
_REMOTE_COMMANDS = (
    "fetch",
    "pull",
    "push",
    "clone",
    "remote-list",
    "remote-add",
    "remote-set-url",
    "remote-remove",
    "remote-rename",
    "submodule-update",
)
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
_REMOTE_TOOL_NAMES_WITHOUT_PULL = _REMOTE_TOOL_NAMES - {"shell.git_pull"}
_LOCAL_MUTATION_TOOL_NAMES = {
    "shell.git_add",
    "shell.git_reset",
    "shell.git_commit",
    "shell.git_branch_delete",
}
_GitToolCallable = Callable[..., Awaitable[str]]


class GitRemoteExposurePhase7Test(TestCase):
    def test_remote_tools_are_absent_without_remote_capability(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("read",),
                allowed_commands=_REMOTE_COMMANDS,
                allowed_remote_protocols=("file",),
                allowed_remote_hosts=("localhost",),
                allow_submodule_update=True,
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )

        self.assertTrue(
            _REMOTE_TOOL_NAMES.isdisjoint(set(_schema_names(toolset)))
        )

    def test_remote_capability_does_not_expose_local_mutation_tools(
        self,
    ) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=(
                    *_REMOTE_COMMANDS,
                    "add",
                    "reset",
                    "commit",
                    "branch-delete",
                ),
                allowed_remote_protocols=("file",),
                allowed_remote_hosts=("localhost",),
                allow_submodule_update=True,
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )
        names = set(_schema_names(toolset))

        self.assertTrue(_REMOTE_TOOL_NAMES_WITHOUT_PULL.issubset(names))
        self.assertNotIn("shell.git_pull", names)
        self.assertTrue(_LOCAL_MUTATION_TOOL_NAMES.isdisjoint(names))

    def test_pull_exposure_requires_remote_worktree_and_history(self) -> None:
        remote_only_settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=("pull",),
                allowed_remote_protocols=("file",),
                allowed_remote_hosts=("localhost",),
            )
        )
        remote_only_names = set(
            _schema_names(
                ShellToolSet(settings=remote_only_settings).with_enabled_tools(
                    ["shell.*"]
                )
            )
        )
        remote_only_explicit_names = set(
            _schema_names(
                ShellToolSet(settings=remote_only_settings).with_enabled_tools(
                    ["shell.git_pull"]
                )
            )
        )
        full_settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote", "worktree", "history"),
                allowed_commands=("pull",),
                allowed_remote_protocols=("file",),
                allowed_remote_hosts=("localhost",),
            )
        )
        full_names = set(
            _schema_names(
                ShellToolSet(settings=full_settings).with_enabled_tools(
                    ["shell.*"]
                )
            )
        )
        full_explicit_names = set(
            _schema_names(
                ShellToolSet(settings=full_settings).with_enabled_tools(
                    ["shell.git_pull"]
                )
            )
        )

        self.assertNotIn("shell.git_pull", remote_only_names)
        self.assertNotIn("shell.git_pull", remote_only_explicit_names)
        self.assertIn("shell.git_pull", full_names)
        self.assertIn("shell.git_pull", full_explicit_names)

    def test_remote_schemas_do_not_expose_denied_raw_fields(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=_REMOTE_COMMANDS,
                allowed_remote_protocols=("file",),
                allowed_remote_hosts=("localhost",),
                allow_submodule_update=True,
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )
        schemas = toolset.json_schemas()
        assert schemas is not None

        fields = {
            field
            for schema in schemas
            if schema["function"]["name"].startswith("shell.git_")
            for field in _schema_field_names(schema["function"]["parameters"])
        }

        self.assertFalse(
            {
                "refspec",
                "refspecs",
                "prune",
                "force",
                "mirror",
                "tags_all",
                "ff_only",
                "set_upstream",
                "redact_urls",
                "recursive",
                "upload_pack",
                "receive_pack",
            }
            & fields
        )
        self.assertIn("ref_type", fields)
        self.assertIn("ref_name", fields)


class GitRemotePolicyPhase7Test(IsolatedAsyncioTestCase):
    async def test_empty_allowlists_deny_before_network(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            protocol_error = await _policy_error(
                _policy(
                    root,
                    allowed_remote_protocols=(),
                    allowed_remote_hosts=("localhost",),
                ),
                _request(command=ShellGitCommandName.CLONE),
            )
            host_error = await _policy_error(
                _policy(
                    root,
                    allowed_remote_protocols=("file",),
                    allowed_remote_hosts=(),
                ),
                _request(command=ShellGitCommandName.CLONE),
            )

        self.assertEqual(
            protocol_error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )
        self.assertEqual(
            host_error.error_code,
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
        )

    async def test_credentials_are_denied_and_redacted_when_disabled(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(
                    root,
                    allowed_remote_protocols=("https",),
                    allowed_remote_hosts=("github.com",),
                ),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": "https://token@github.com/acme/repo.git",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
        )

    async def test_core_bare_truthy_variants_are_denied(self) -> None:
        cases = (
            ("bare_key", "\tbare\n"),
            ("true", "\tbare = true\n"),
            ("yes", "\tbare = yes\n"),
            ("on", "\tbare = on\n"),
            ("one", "\tbare = 1\n"),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            remote_config = _remote_config(_file_url(root / "remote.git"))
            policy = _policy(root, allowed_commands=("fetch",))

            for name, bare_line in cases:
                with self.subTest(name=name):
                    (repo / ".git" / "config").write_text(
                        "[core]\n"
                        + "\trepositoryformatversion = 0\n"
                        + bare_line
                        + remote_config
                    )
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.FETCH,
                            options={
                                "remote": "origin",
                                "ref_type": "branch",
                                "ref_name": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.BARE_REPO_DENIED,
                    )

    async def test_core_bare_false_variants_are_allowed(self) -> None:
        cases = (
            ("false", "\tbare = false\n"),
            ("no", "\tbare = no\n"),
            ("off", "\tbare = off\n"),
            ("zero", "\tbare = 0\n"),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            remote_config = _remote_config(_file_url(root / "remote.git"))
            policy = _policy(root, allowed_commands=("fetch",))

            for name, bare_line in cases:
                with self.subTest(name=name):
                    (repo / ".git" / "config").write_text(
                        "[core]\n"
                        + "\trepositoryformatversion = 0\n"
                        + bare_line
                        + remote_config
                    )
                    spec = await policy.normalize(
                        _request(
                            command=ShellGitCommandName.FETCH,
                            options={
                                "remote": "origin",
                                "ref_type": "branch",
                                "ref_name": "main",
                            },
                        ),
                    )
                    self.assertEqual(spec.command, "git.fetch")

    async def test_malformed_request_remote_url_denies_stably(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(
                    root,
                    allowed_commands=("clone",),
                    allowed_remote_protocols=("https",),
                    allowed_remote_hosts=("github.com",),
                ),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": "https://[github.com]/repo",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
        )

    async def test_malformed_request_remote_url_protocol_edges(
        self,
    ) -> None:
        cases = (
            (
                "git://[github.com]/repo",
                ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            ),
            (
                "//[github.com]/repo",
                ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            policy = _policy(
                root,
                allowed_commands=("clone",),
                allowed_remote_protocols=("https",),
                allowed_remote_hosts=("github.com",),
            )

            for url, error_code in cases:
                with self.subTest(url=url):
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.CLONE,
                            options={
                                "url": url,
                                "destination": "repo-copy",
                                "branch": "main",
                            },
                        ),
                    )
                    self.assertEqual(error.error_code, error_code)

    async def test_hostless_file_clone_url_inside_workspace_is_allowed(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_bare_git_repo(root / "remote.git")
            url = _hostless_file_url(remote)
            spec = await _policy(
                root,
                cwd=".",
                allowed_commands=("clone",),
            ).normalize(
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": url,
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                )
            )

        self.assertIn(url, spec.argv)
        self.assertEqual(spec.metadata["git_remote_protocol"], "file")
        self.assertIsNone(spec.metadata["git_remote_host"])
        self.assertEqual(spec.metadata["git_remote_url"], "file:///[redacted]")

    async def test_hostless_file_clone_url_accepts_percent_encoded_space(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_bare_git_repo(root / "remote repo.git")
            url = f"file://{remote.resolve().as_posix().replace(' ', '%20')}"
            spec = await _policy(
                root,
                cwd=".",
                allowed_commands=("clone",),
            ).normalize(
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": url,
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                )
            )

        self.assertEqual(spec.argv[-2], url)

    async def test_hostless_file_clone_url_accepts_dot_dot_inside_workspace(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            (root / "remotes").mkdir()
            _write_minimal_bare_git_repo(root / "remotes" / "remote.git")
            url_path = root / "remotes" / ".." / "remotes" / "remote.git"
            url = f"file://{url_path.as_posix()}"
            spec = await _policy(
                root,
                cwd=".",
                allowed_commands=("clone",),
            ).normalize(
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": url,
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                )
            )

        self.assertEqual(spec.argv[-2], url)

    async def test_hostless_file_clone_url_rejects_raw_double_slash_path(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = root / "remote.git"
            error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": f"file:///{remote.resolve().as_posix()}",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )

    async def test_hostless_file_clone_url_rejects_encoded_leading_slash(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = root / "remote.git"
            encoded_path = remote.resolve().as_posix().lstrip("/")
            error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": f"file:///%2F{encoded_path}",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )

    async def test_hostless_file_clone_url_rejects_encoded_control_character(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": f"{_hostless_file_url(root / 'remote.git')}%0A",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )

    async def test_hostless_file_clone_url_rejects_file_absolute_form(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = root / "remote.git"
            error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": f"file:{remote.resolve().as_posix()}",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )

    async def test_hostless_file_clone_url_rejects_query(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": (
                            f"{_hostless_file_url(root / 'remote.git')}?q=1"
                        ),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )

    async def test_hostless_file_clone_url_must_stay_inside_workspace(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _hostless_file_url(root.parent / "remote.git"),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

    async def test_hostless_file_clone_url_requires_localhost_allowlist(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(
                    root,
                    cwd=".",
                    allowed_commands=("clone",),
                    allowed_remote_protocols=("file",),
                    allowed_remote_hosts=("github.com",),
                ),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _hostless_file_url(root / "remote.git"),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
        )

    async def test_hostless_file_clone_url_requires_file_protocol_allowlist(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(
                    root,
                    cwd=".",
                    allowed_commands=("clone",),
                    allowed_remote_protocols=("https",),
                    allowed_remote_hosts=("localhost",),
                ),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _hostless_file_url(root / "remote.git"),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )

    async def test_file_clone_url_requires_path_inside_workspace(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            missing_path_error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": "file://localhost",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )
            outside_error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _file_url(root.parent / "remote.git"),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            missing_path_error.error_code,
            ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
        )
        self.assertEqual(
            outside_error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

    def test_file_remote_url_path_must_be_absolute(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            with self.assertRaises(ShellGitPolicyDenied) as raised:
                _validate_file_remote_url(
                    "relative/repo.git",
                    workspace_root=root,
                )

        self.assertEqual(
            raised.exception.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

    async def test_file_push_denies_active_server_side_hooks(self) -> None:
        hook_names = (
            "pre-receive",
            "update",
            "post-receive",
            "post-update",
            "proc-receive",
            "reference-transaction",
            "push-to-checkout",
        )
        for hook_name in hook_names:
            with self.subTest(hook_name=hook_name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    remote = _write_minimal_bare_git_repo(root / "remote.git")
                    hook = remote / "hooks" / hook_name
                    hook.write_text("#!/bin/sh\nexit 0\n")
                    hook.chmod(0o755)
                    _write_minimal_git_repo(
                        root / "repo",
                        remote_url=_file_url(remote),
                    )
                    error = await _policy_error(
                        _policy(root, allowed_commands=("push",)),
                        _request(
                            command=ShellGitCommandName.PUSH,
                            options=_remote_command_options(
                                ShellGitCommandName.PUSH
                            ),
                        ),
                    )

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
                )
                self.assertEqual(
                    str(error),
                    "Git file push target hooks can trigger external "
                    "processing",
                )

    async def test_file_push_allows_sample_and_inactive_hooks(self) -> None:
        cases = ("no_hooks", "sample_hook", "inactive_hook")
        for name in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    remote = _write_minimal_bare_git_repo(root / "remote.git")
                    if name == "sample_hook":
                        hook = remote / "hooks" / "pre-receive.sample"
                        hook.write_text("#!/bin/sh\nexit 1\n")
                        hook.chmod(0o755)
                    if name == "inactive_hook":
                        hook = remote / "hooks" / "pre-receive"
                        hook.write_text("#!/bin/sh\nexit 1\n")
                        hook.chmod(0o644)
                    _write_minimal_git_repo(
                        root / "repo",
                        remote_url=_file_url(remote),
                    )
                    spec = await _policy(
                        root,
                        allowed_commands=("push",),
                    ).normalize(
                        _request(
                            command=ShellGitCommandName.PUSH,
                            options=_remote_command_options(
                                ShellGitCommandName.PUSH
                            ),
                        ),
                    )

                self.assertEqual(spec.command, "git.push")

    async def test_https_push_skips_local_hook_scan(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(
                root / "repo",
                remote_url="https://github.com/acme/repo.git",
            )
            spec = await _policy(
                root,
                allowed_commands=("push",),
                allowed_remote_protocols=("https",),
                allowed_remote_hosts=("github.com",),
            ).normalize(_push_request())

        self.assertEqual(spec.command, "git.push")

    async def test_file_push_denies_non_bare_remote_before_execution(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_git_repo(root / "remote-repo")
            error = await _push_policy_error(root, remote)

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )
        self.assertEqual(
            str(error),
            "Git file push target cannot be proven safe",
        )

    async def test_fetch_does_not_validate_file_push_target_shape(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_git_repo(root / "remote-repo")
            _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(remote),
            )
            fetch_spec = await _policy(
                root,
                allowed_commands=("fetch",),
            ).normalize(
                _request(
                    command=ShellGitCommandName.FETCH,
                    options=_remote_command_options(ShellGitCommandName.FETCH),
                ),
            )
            push_error = await _policy_error(
                _policy(root, allowed_commands=("push",)),
                _push_request(),
            )

        self.assertEqual(fetch_spec.command, "git.fetch")
        self.assertEqual(
            push_error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )

    async def test_file_push_target_must_be_verifiable_git_repo(
        self,
    ) -> None:
        cases = ("missing", "file", "plain_directory", "git_file")
        for name in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    remote = root / "remote.git"
                    if name == "file":
                        remote.write_text("not a repository\n")
                    if name == "plain_directory":
                        remote.mkdir()
                    if name == "git_file":
                        remote.mkdir()
                        (remote / ".git").write_text("gitdir: ../actual.git\n")
                    error = await _push_policy_error(root, remote)

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
                )
                self.assertEqual(
                    str(error),
                    "Git file push target cannot be proven safe",
                )

    async def test_file_push_target_config_edges_fail_closed(self) -> None:
        bare_config_edges = {
            "bare_missing": "[core]\n\trepositoryformatversion = 0\n",
            "bare_false": (
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = false\n"
            ),
            "bare_true_then_false": (
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = true\n"
                + "\tbare = false\n"
            ),
            "bare_no": (
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = no\n"
            ),
            "bare_off": (
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = off\n"
            ),
            "bare_zero": (
                "[core]\n" + "\trepositoryformatversion = 0\n" + "\tbare = 0\n"
            ),
            "bare_unknown": (
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = maybe\n"
            ),
            "bare_malformed": (
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = trueish\n"
            ),
            "bare_quoted_false": (
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + '\tbare = "false"\n'
            ),
        }
        cases = (
            "config_missing",
            "config_directory",
            "config_symlink",
            *bare_config_edges,
        )
        for name in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    remote = _write_minimal_bare_git_repo(root / "remote.git")
                    config = remote / "config"
                    if name in (
                        "config_missing",
                        "config_directory",
                        "config_symlink",
                    ):
                        config.unlink()
                    if name == "config_directory":
                        config.mkdir()
                    if name == "config_symlink":
                        try:
                            config.symlink_to(root / "external-config")
                        except OSError as error:
                            self.skipTest(f"symlink unavailable: {error}")
                    if name in bare_config_edges:
                        config.write_text(bare_config_edges[name])
                    error = await _push_policy_error(root, remote)

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
                )
                self.assertEqual(
                    str(error),
                    "Git file push target cannot be proven safe",
                )

    async def test_file_push_target_config_read_errors_fail_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_bare_git_repo(root / "remote.git")
            remote_config = remote / "config"

            async def failing_remote_config_read(path: Path) -> str:
                if path.resolve() == remote_config.resolve():
                    raise OSError("read failed")
                return path.read_text()

            with patch(
                "avalan.tool.shell.git_policy._read_text",
                failing_remote_config_read,
            ):
                error = await _push_policy_error(root, remote)

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )
        self.assertEqual(
            str(error),
            "Git file push target cannot be proven safe",
        )

    async def test_file_push_target_bare_layout_symlinks_fail_closed(
        self,
    ) -> None:
        cases = ("HEAD", "objects", "refs")
        for name in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    remote = _write_minimal_bare_git_repo(root / "remote.git")
                    entry = remote / name
                    symlink_target = root / f"linked-{name}"
                    entry.rename(symlink_target)
                    try:
                        entry.symlink_to(
                            symlink_target,
                            target_is_directory=symlink_target.is_dir(),
                        )
                    except OSError as error:
                        self.skipTest(f"symlink unavailable: {error}")
                    error = await _push_policy_error(root, remote)

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
                )
                self.assertEqual(
                    str(error),
                    "Git file push target cannot be proven safe",
                )

    async def test_file_push_target_with_commondir_fails_closed(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_bare_git_repo(root / "remote.git")
            (remote / "commondir").write_text("../common.git\n")
            error = await _push_policy_error(root, remote)

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )
        self.assertEqual(
            str(error),
            "Git file push target cannot be proven safe",
        )

    async def test_file_push_hook_directory_edges_fail_closed(self) -> None:
        cases = ("missing_hooks", "hooks_symlink", "hooks_file")
        for name in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    remote = _write_minimal_bare_git_repo(root / "remote.git")
                    hooks = remote / "hooks"
                    hooks.rmdir()
                    if name == "hooks_symlink":
                        try:
                            hooks.symlink_to(root / "missing-hooks")
                        except OSError as error:
                            self.skipTest(f"symlink unavailable: {error}")
                    if name == "hooks_file":
                        hooks.write_text("not a hooks directory\n")

                    if name == "missing_hooks":
                        spec = await _policy(
                            root,
                            allowed_commands=("push",),
                        ).normalize(
                            _write_push_repo_request(root, remote),
                        )
                        self.assertEqual(spec.command, "git.push")
                        continue

                    error = await _push_policy_error(root, remote)

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
                )
                self.assertEqual(
                    str(error),
                    "Git file push target hooks can trigger external "
                    "processing",
                )

    async def test_file_push_hook_entry_edges_fail_closed(self) -> None:
        cases = ("non_server_hook", "hook_directory", "hook_symlink")
        for name in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    remote = _write_minimal_bare_git_repo(root / "remote.git")
                    if name == "non_server_hook":
                        hook = remote / "hooks" / "pre-commit"
                        hook.write_text("#!/bin/sh\nexit 1\n")
                        hook.chmod(0o755)
                        spec = await _policy(
                            root,
                            allowed_commands=("push",),
                        ).normalize(
                            _write_push_repo_request(root, remote),
                        )
                        self.assertEqual(spec.command, "git.push")
                        continue
                    if name == "hook_directory":
                        (remote / "hooks" / "pre-receive").mkdir()
                    if name == "hook_symlink":
                        try:
                            (remote / "hooks" / "pre-receive").symlink_to(
                                root / "missing-hook"
                            )
                        except OSError as error:
                            self.skipTest(f"symlink unavailable: {error}")
                    error = await _push_policy_error(root, remote)

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
                )
                self.assertEqual(
                    str(error),
                    "Git file push target hooks can trigger external "
                    "processing",
                )

    async def test_file_push_hook_filesystem_errors_fail_closed(self) -> None:
        async def failing_list_directory(path: Path) -> tuple[Path, ...]:
            assert isinstance(path, Path)
            raise OSError("list failed")

        async def failing_inspect_path(path: Path) -> object:
            assert isinstance(path, Path)
            raise OSError("stat failed")

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_bare_git_repo(root / "remote.git")
            with patch(
                "avalan.tool.shell.git_policy._list_directory",
                failing_list_directory,
            ):
                list_error = await _push_policy_error(root, remote)

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_minimal_bare_git_repo(root / "remote.git")
            hook = remote / "hooks" / "pre-receive"
            hook.write_text("#!/bin/sh\nexit 0\n")
            with patch(
                "avalan.tool.shell.git_policy._inspect_path",
                failing_inspect_path,
            ):
                inspect_error = await _push_policy_error(root, remote)

        self.assertEqual(
            list_error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )
        self.assertEqual(
            inspect_error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )
        self.assertEqual(
            str(list_error),
            "Git file push target cannot be proven safe",
        )
        self.assertEqual(
            str(inspect_error),
            "Git file push target cannot be proven safe",
        )

    async def test_malformed_config_remote_url_denies_stably(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(
                root / "repo",
                remote_url="https://[github.com]/repo",
            )
            error = await _policy_error(
                _policy(
                    root,
                    allowed_commands=("fetch",),
                    allowed_remote_protocols=("https",),
                    allowed_remote_hosts=("github.com",),
                ),
                _request(
                    command=ShellGitCommandName.FETCH,
                    options={
                        "remote": "origin",
                        "ref_type": "branch",
                        "ref_name": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
        )

    async def test_malformed_http_url_scoped_config_denies_stably(
        self,
    ) -> None:
        cases = (
            (
                "quoted",
                '[http "https://[github.com"]\n'
                + "\tuserAgent = avalan-test\n",
            ),
            (
                "dotted",
                "[http.https://[github.com]\n" + "\tuserAgent = avalan-test\n",
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            base_config = (repo / ".git" / "config").read_text()
            policy = _policy(root, allowed_commands=("fetch",))

            for name, config in cases:
                with self.subTest(name=name):
                    (repo / ".git" / "config").write_text(base_config + config)
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.FETCH,
                            options={
                                "remote": "origin",
                                "ref_type": "branch",
                                "ref_name": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
                    )

    async def test_remote_name_and_existing_remote_edges(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            (repo / ".git" / "config").write_text(
                (repo / ".git" / "config").read_text()
                + _remote_config(_file_url(root / "upstream.git")).replace(
                    "origin",
                    "upstream",
                )
            )
            policy = _policy(
                root,
                allowed_commands=(
                    "fetch",
                    "remote-add",
                    "remote-rename",
                ),
            )
            cases = (
                (
                    _request(
                        command=ShellGitCommandName.REMOTE_ADD,
                        options={
                            "name": "-origin",
                            "url": _file_url(root / "other.git"),
                        },
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _request(
                        command=ShellGitCommandName.FETCH,
                        options={
                            "remote": 123,
                            "ref_type": "branch",
                            "ref_name": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _request(
                        command=ShellGitCommandName.FETCH,
                        options={
                            "remote": "missing",
                            "ref_type": "branch",
                            "ref_name": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
                ),
                (
                    _request(
                        command=ShellGitCommandName.REMOTE_ADD,
                        options={
                            "name": "origin",
                            "url": _file_url(root / "other.git"),
                        },
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _request(
                        command=ShellGitCommandName.REMOTE_RENAME,
                        options={
                            "old_name": "origin",
                            "new_name": "origin",
                        },
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _request(
                        command=ShellGitCommandName.REMOTE_RENAME,
                        options={
                            "old_name": "origin",
                            "new_name": "upstream",
                        },
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
            )

            for request, error_code in cases:
                with self.subTest(
                    command=request.command,
                    options=request.options,
                ):
                    error = await _policy_error(policy, request)
                    self.assertEqual(error.error_code, error_code)

    async def test_existing_remote_requires_a_single_url(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").write_text(
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = false\n"
                + '[remote "origin"]\n'
                + "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
            )
            error = await _policy_error(
                _policy(root, allowed_commands=("fetch",)),
                _request(
                    command=ShellGitCommandName.FETCH,
                    options={
                        "remote": "origin",
                        "ref_type": "branch",
                        "ref_name": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        )

    async def test_remote_only_cannot_run_local_or_pull_mutation_commands(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            pull_request = _request(
                command=ShellGitCommandName.PULL,
                options={"remote": "origin", "branch": "main"},
            )
            commit_error = await _policy_error(
                _policy(
                    root,
                    allowed_commands=("commit",),
                ),
                ShellGitCommandRequest(
                    tool_name="shell.git_commit",
                    command=ShellGitCommandName.COMMIT,
                    capability_required=ShellGitCapability.HISTORY,
                    options={"message": "blocked"},
                ),
            )
            pull_error = await _policy_error(
                _policy(
                    root,
                    allowed_commands=("pull",),
                ),
                pull_request,
            )
            pull_spec = await _policy(
                root,
                capabilities=("remote", "worktree", "history"),
                allowed_commands=("pull",),
            ).normalize(
                pull_request,
            )

        self.assertEqual(
            commit_error.error_code,
            ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
        )
        self.assertEqual(
            pull_error.error_code,
            ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
        )
        self.assertEqual(
            pull_spec.argv,
            (
                *_git_prefix(),
                "pull",
                "--ff-only",
                "--no-verify",
                "--no-tags",
                "--no-prune",
                "--no-recurse-submodules",
                "origin",
                "main",
            ),
        )

    async def test_denied_remote_options_fail_closed(self) -> None:
        cases: tuple[tuple[ShellGitCommandName, dict[str, object]], ...] = (
            (
                ShellGitCommandName.FETCH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "prune": True,
                },
            ),
            (
                ShellGitCommandName.FETCH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "refspecs": ("+main:main",),
                },
            ),
            (
                ShellGitCommandName.PUSH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "force": True,
                },
            ),
            (
                ShellGitCommandName.PUSH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "mirror": True,
                },
            ),
            (
                ShellGitCommandName.PUSH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "tags_all": True,
                },
            ),
            (
                ShellGitCommandName.FETCH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "upload_pack": "/tmp/upload-pack",
                },
            ),
            (
                ShellGitCommandName.PUSH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "receive_pack": "/tmp/receive-pack",
                },
            ),
            (
                ShellGitCommandName.FETCH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "askpass": "/tmp/askpass",
                },
            ),
            (
                ShellGitCommandName.FETCH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                    "ssh_command": "ssh -oProxyCommand=x",
                },
            ),
            (
                ShellGitCommandName.SUBMODULE_UPDATE,
                {"init": True, "recursive": True},
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            policy = _policy(
                root,
                allowed_commands=_REMOTE_COMMANDS,
                allow_submodule_update=True,
            )

            for command, options in cases:
                with self.subTest(command=command, options=options):
                    error = await _policy_error(
                        policy,
                        _request(command=command, options=options),
                    )
                    self.assertIn(
                        error.error_code,
                        (
                            ShellGitExecutionErrorCode.INVALID_OPTION,
                            ShellGitExecutionErrorCode.SUBMODULE_DENIED,
                        ),
                    )

    async def test_unsafe_remote_config_is_denied_before_network(
        self,
    ) -> None:
        cases = (
            (
                "duplicate_url",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\turl = {url}\n',
            ),
            (
                "trailing_comment_duplicate_url",
                ShellGitCommandName.FETCH,
                '[remote "origin"] # trusted remote\n\turl = {url}\n',
            ),
            (
                "dotted_trailing_comment_duplicate_url",
                ShellGitCommandName.FETCH,
                "[remote.origin];trusted remote\n\turl = {url}\n",
            ),
            (
                "dotted_duplicate_url",
                ShellGitCommandName.FETCH,
                "[remote.origin]\n\turl = {url}\n",
            ),
            (
                "inline_comment_dotted_duplicate_url",
                ShellGitCommandName.FETCH,
                (
                    "[core]\n\tbare = false\n"
                    "[remote.origin] # managed by test\n"
                    "\turl = {url}\n"
                ),
            ),
            (
                "pushurl",
                ShellGitCommandName.PUSH,
                '[remote "origin"]\n\tpushurl = {url}\n',
            ),
            (
                "trailing_comment_pushurl",
                ShellGitCommandName.PUSH,
                '[remote "origin"] # trusted remote\n\tpushurl = {url}\n',
            ),
            (
                "dotted_trailing_comment_pushurl",
                ShellGitCommandName.PUSH,
                "[remote.origin];trusted remote\n\tpushurl = {url}\n",
            ),
            (
                "dotted_pushurl",
                ShellGitCommandName.PUSH,
                "[remote.origin]\n\tpushurl = {url}\n",
            ),
            (
                "inline_comment_dotted_pushurl",
                ShellGitCommandName.PUSH,
                (
                    "[core]\n\tbare = false\n"
                    "[remote.origin] # managed by test\n"
                    "\tpushurl = {url}\n"
                ),
            ),
            (
                "mirror",
                ShellGitCommandName.PUSH,
                '[remote "origin"]\n\tmirror = true\n',
            ),
            (
                "bare_mirror",
                ShellGitCommandName.PUSH,
                '[remote "origin"]\n\tmirror\n',
            ),
            (
                "dotted_bare_mirror",
                ShellGitCommandName.PUSH,
                "[remote.origin]\n\tmirror\n",
            ),
            (
                "prune",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\tprune = true\n',
            ),
            (
                "bare_prune",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\tprune\n',
            ),
            (
                "unknown_bool_prune",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\tprune = maybe\n',
            ),
            (
                "tagopt",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\ttagOpt = --tags\n',
            ),
            (
                "push_tagopt_no_tags",
                ShellGitCommandName.PUSH,
                '[remote "origin"]\n\ttagOpt = --no-tags\n',
            ),
            (
                "uploadpack",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\tuploadpack = /tmp/upload-pack\n',
            ),
            (
                "remote_vcs",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\tvcs = unsafe-helper\n',
            ),
            (
                "dotted_remote_vcs",
                ShellGitCommandName.FETCH,
                "[remote.origin]\n\tvcs = unsafe-helper\n",
            ),
            (
                "server_option",
                ShellGitCommandName.FETCH,
                '[remote "origin"]\n\tserverOption = object-format=sha256\n',
            ),
            (
                "dotted_server_option",
                ShellGitCommandName.FETCH,
                "[remote.origin]\n\tserverOption = object-format=sha256\n",
            ),
            (
                "receivepack",
                ShellGitCommandName.PUSH,
                '[remote "origin"]\n\treceivepack = /tmp/receive-pack\n',
            ),
            (
                "fetch_prune",
                ShellGitCommandName.FETCH,
                "[fetch]\n\tprune = true\n",
            ),
            (
                "bare_fetch_prune",
                ShellGitCommandName.FETCH,
                "[fetch]\n\tprune\n",
            ),
            (
                "pull_rebase",
                ShellGitCommandName.PULL,
                "[pull]\n\trebase = true\n",
            ),
            (
                "bare_pull_rebase",
                ShellGitCommandName.PULL,
                "[pull]\n\trebase\n",
            ),
            (
                "push_follow_tags",
                ShellGitCommandName.PUSH,
                "[push]\n\tfollowTags = true\n",
            ),
            (
                "bare_push_follow_tags",
                ShellGitCommandName.PUSH,
                "[push]\n\tfollowTags\n",
            ),
            (
                "push_option",
                ShellGitCommandName.PUSH,
                "[push]\n\tpushOption = ci.skip\n",
            ),
            (
                "bare_push_option",
                ShellGitCommandName.PUSH,
                "[push]\n\tpushOption\n",
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            safe_url = _file_url(root / "other.git")
            base_config = (repo / ".git" / "config").read_text()
            policy = _policy(
                root,
                capabilities=("remote", "worktree", "history"),
                allowed_commands=_REMOTE_COMMANDS,
            )

            for name, command, config in cases:
                with self.subTest(name=name):
                    (repo / ".git" / "config").write_text(
                        base_config + config.format(url=safe_url)
                    )
                    error = await _policy_error(
                        policy,
                        _request(
                            command=command,
                            options=_remote_command_options(command),
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
                    )

    async def test_git_config_parser_edges_remain_bounded(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            remote_url = _file_url(root / "path-with-gpg" / "remote.git")
            (repo / ".git" / "config").write_text(
                "[core]\n"
                + "\trepositoryformatversion = 0\n"
                + "\tbare = false\n"
                + "\n"
                + "# managed by test\n"
                + "\t=value without key\n"
                + '\tcustom "a\\"b" # quoted comment\n'
                + "[]\n"
                + "\tignored = true\n"
                + '[http "https://github.com/a\\]b"]\n'
                + "\tuserAgent = avalan-test\n"
                + _remote_config(remote_url)
            )
            spec = await _policy(root, allowed_commands=("fetch",)).normalize(
                _request(
                    command=ShellGitCommandName.FETCH,
                    options={
                        "remote": "origin",
                        "ref_type": "branch",
                        "ref_name": "main",
                    },
                ),
            )

        self.assertEqual(spec.command, "git.fetch")

    async def test_git_config_dangerous_keys_are_denied(self) -> None:
        cases = (
            ("gpg_section", "[gpg]\n\tprogram = /tmp/gpg\n"),
            ("commit_gpgsign", "[commit]\n\tgpgSign = true\n"),
            ("user_signing_key", "[user]\n\tsigningKey = ABC123\n"),
            ("core_hooks_path", "[core]\n\thooksPath = .git/hooks\n"),
            (
                "extensions_worktree_config",
                "[extensions]\n\tworktreeConfig = true\n",
            ),
            ("credential_helper", "[credential]\n\thelper = store\n"),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            base_config = (repo / ".git" / "config").read_text()
            policy = _policy(root, allowed_commands=("fetch",))

            for name, config in cases:
                with self.subTest(name=name):
                    (repo / ".git" / "config").write_text(base_config + config)
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.FETCH,
                            options={
                                "remote": "origin",
                                "ref_type": "branch",
                                "ref_name": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
                    )

    async def test_read_commands_ignore_signing_only_git_config(self) -> None:
        signing_configs = (
            "[gpg]\n\tformat = ssh\n",
            "[commit]\n\tgpgSign = true\n",
            "[tag]\n\tgpgSign = true\n",
            "[user]\n\tsigningKey = /Users/example/.ssh/id.pub\n",
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / "tracked.txt").write_text("needle\n")
            base_config = (repo / ".git" / "config").read_text()
            policy = _policy(
                root,
                capabilities=("read",),
                allowed_commands=("grep",),
            )

            for config in signing_configs:
                with self.subTest(config=config):
                    (repo / ".git" / "config").write_text(base_config + config)
                    spec = await policy.normalize(
                        ShellGitCommandRequest(
                            tool_name="shell.git_grep",
                            command=ShellGitCommandName.GREP,
                            capability_required=ShellGitCapability.READ,
                            options={"pattern": "needle"},
                            pathspecs=("tracked.txt",),
                        )
                    )
                    self.assertEqual(spec.command, "git.grep")
                    self.assertIn("--no-textconv", spec.argv)

            (repo / ".git" / "config").write_text(
                base_config + "[core]\n\tfsmonitor = .git/watch\n"
            )
            error = await _policy_error(
                policy,
                ShellGitCommandRequest(
                    tool_name="shell.git_grep",
                    command=ShellGitCommandName.GREP,
                    capability_required=ShellGitCapability.READ,
                    options={"pattern": "needle"},
                    pathspecs=("tracked.txt",),
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        )

    async def test_read_command_skips_remote_repository_state(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            spec = await _policy(
                root,
                capabilities=("read",),
                allowed_commands=("status",),
            ).normalize(
                ShellGitCommandRequest(
                    tool_name="shell.git_status",
                    command=ShellGitCommandName.STATUS,
                    capability_required=ShellGitCapability.READ,
                    options={},
                ),
            )

        self.assertEqual(spec.command, "git.status")
        self.assertNotIn("git_remote_url", spec.metadata)

    async def test_malformed_git_config_sections_fail_closed(self) -> None:
        cases = (
            "[core] trailing\n\trepositoryformatversion = 0\n",
            '[http "https://github.com"\n\tuserAgent = avalan-test\n',
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            policy = _policy(root, allowed_commands=("fetch",))

            for config in cases:
                with self.subTest(config=config):
                    (repo / ".git" / "config").write_text(config)
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.FETCH,
                            options={
                                "remote": "origin",
                                "ref_type": "branch",
                                "ref_name": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
                    )

    async def test_http_credentials_config_is_denied_before_network(
        self,
    ) -> None:
        cases = (
            (
                "global_http_extra_header",
                "[http]\n\textraHeader = Authorization: Bearer secret-token\n",
            ),
            (
                "global_http_extra_header_trailing_comment",
                (
                    "[http] # managed by test\n"
                    "\textraHeader = Authorization: Bearer secret-token "
                    "# token\n"
                ),
            ),
            (
                "url_scoped_http_extra_header",
                (
                    '[http "https://github.com"]\n'
                    "\textraHeader = Authorization: Bearer secret-token\n"
                ),
            ),
            (
                "dotted_url_scoped_http_extra_header",
                (
                    "[http.https://github.com]\n"
                    "\textraHeader = Authorization: Bearer secret-token\n"
                ),
            ),
            (
                "global_http_cookie_file",
                "[http]\n\tcookieFile = /tmp/git-cookies\n",
            ),
            (
                "global_http_lowercase_cookiefile",
                "[http]\n\tcookiefile = /tmp/git-cookies\n",
            ),
            (
                "url_scoped_http_cookie_file",
                (
                    '[http "https://github.com"]\n'
                    "\tcookieFile = /tmp/git-cookies\n"
                ),
            ),
            (
                "global_http_save_cookies",
                "[http]\n\tsaveCookies = true\n",
            ),
            (
                "global_http_bare_save_cookies",
                "[http]\n\tsaveCookies\n",
            ),
            (
                "url_scoped_http_save_cookies",
                '[http "https://github.com"]\n\tsaveCookies = true\n',
            ),
            (
                "url_scoped_http_subsection_credentials",
                (
                    '[http "https://token@github.com"]\n'
                    "\tuserAgent = avalan-test\n"
                ),
            ),
            (
                "dotted_url_scoped_http_subsection_credentials",
                "[http.https://token@github.com]\n\tuserAgent = avalan-test\n",
            ),
            (
                "url_scoped_malformed_http_subsection_credentials",
                (
                    '[http "https://token@[github.com"]\n'
                    "\tuserAgent = avalan-test\n"
                ),
            ),
            (
                "url_scoped_path_at_marker",
                (
                    '[http "https://github.com/path@v1"]\n'
                    "\tuserAgent = avalan-test\n"
                ),
            ),
            (
                "global_http_ssl_cert",
                "[http]\n\tsslCert = /tmp/client.pem\n",
            ),
            (
                "url_scoped_http_ssl_cert",
                '[http "https://github.com"]\n\tsslCert = /tmp/client.pem\n',
            ),
            (
                "global_http_ssl_key",
                "[http]\n\tsslKey = /tmp/client.key\n",
            ),
            (
                "url_scoped_http_ssl_key",
                '[http "https://github.com"]\n\tsslKey = /tmp/client.key\n',
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            base_config = (repo / ".git" / "config").read_text()
            policy = _policy(
                root,
                allowed_commands=("fetch",),
            )

            for name, config in cases:
                with self.subTest(name=name):
                    (repo / ".git" / "config").write_text(base_config + config)
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.FETCH,
                            options={
                                "remote": "origin",
                                "ref_type": "branch",
                                "ref_name": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
                    )

    async def test_http_extra_header_is_denied_with_credentials_enabled(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            (repo / ".git" / "config").write_text(
                (repo / ".git" / "config").read_text()
                + "[http]\n"
                + "\textraHeader = Authorization: Bearer secret-token\n"
            )
            error = await _policy_error(
                _policy(
                    root,
                    allowed_commands=("fetch",),
                    allow_remote_credentials=True,
                ),
                _request(
                    command=ShellGitCommandName.FETCH,
                    options={
                        "remote": "origin",
                        "ref_type": "branch",
                        "ref_name": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
        )

    async def test_http_tls_config_is_denied_before_network(self) -> None:
        cases = (
            (
                "global_ssl_verify_false",
                "[http]\n\tsslVerify = false\n",
            ),
            (
                "global_ssl_verify_no",
                "[http]\n\tsslVerify = no\n",
            ),
            (
                "global_ssl_verify_off",
                "[http]\n\tsslVerify = off\n",
            ),
            (
                "global_ssl_verify_zero",
                "[http]\n\tsslVerify = 0\n",
            ),
            (
                "global_ssl_verify_trailing_comment",
                (
                    "[http] # managed by test\n"
                    "\tsslVerify = false # disable TLS verification\n"
                ),
            ),
            (
                "url_scoped_ssl_verify_false",
                '[http "https://github.com"]\n\tsslVerify = false\n',
            ),
            (
                "dotted_url_scoped_ssl_verify_false",
                "[http.https://github.com]\n\tsslVerify = false\n",
            ),
            (
                "bare_ssl_verify",
                "[http]\n\tsslVerify\n",
            ),
            (
                "ssl_ca_info",
                "[http]\n\tsslCAInfo = /tmp/ca.pem\n",
            ),
            (
                "ssl_ca_path",
                "[http]\n\tsslCAPath = /tmp/certs\n",
            ),
            (
                "ssl_cipher_list",
                "[http]\n\tsslCipherList = DEFAULT:@SECLEVEL=0\n",
            ),
            (
                "ssl_version",
                "[http]\n\tsslVersion = tlsv1\n",
            ),
            (
                "schannel_revoke_disabled",
                "[http]\n\tschannelCheckRevoke = false\n",
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            base_config = (repo / ".git" / "config").read_text()
            policy = _policy(
                root,
                allowed_commands=("fetch",),
            )

            for name, config in cases:
                with self.subTest(name=name):
                    (repo / ".git" / "config").write_text(base_config + config)
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.FETCH,
                            options={
                                "remote": "origin",
                                "ref_type": "branch",
                                "ref_name": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
                    )

    async def test_existing_remote_url_must_be_allowlisted(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(
                root / "repo",
                remote_url="https://evil.example/acme/repo.git",
            )
            error = await _policy_error(
                _policy(
                    root,
                    allowed_commands=("fetch",),
                    allowed_remote_protocols=("https",),
                    allowed_remote_hosts=("github.com",),
                ),
                _request(
                    command=ShellGitCommandName.FETCH,
                    options={
                        "remote": "origin",
                        "ref_type": "branch",
                        "ref_name": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
        )

    async def test_legacy_dotted_remote_list_audits_remote_url(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            remote_url = _file_url(root / "remote.git")
            upstream_url = _file_url(root / "upstream.git")
            (repo / ".git" / "config").write_text(
                (repo / ".git" / "config").read_text()
                + "[remote.origin]\n"
                + f"\turl = {remote_url}\n"
                + '[remote "upstream"]\n'
                + f"\turl = {upstream_url}\n"
            )
            spec = await _policy(
                root,
                allowed_commands=("remote-list",),
            ).normalize(
                _request(command=ShellGitCommandName.REMOTE_LIST),
            )

        self.assertEqual(spec.metadata["git_remote_protocol"], "file")
        self.assertEqual(spec.metadata["git_remote_host"], "localhost")
        self.assertEqual(
            spec.metadata["git_remote_urls"],
            (
                "file://localhost/[redacted]",
                "file://localhost/[redacted]",
            ),
        )

    async def test_legacy_dotted_remote_config_must_be_allowlisted(
        self,
    ) -> None:
        cases = (
            (
                ShellGitCommandName.REMOTE_LIST,
                {},
            ),
            (
                ShellGitCommandName.FETCH,
                {
                    "remote": "origin",
                    "ref_type": "branch",
                    "ref_name": "main",
                },
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").write_text(
                (repo / ".git" / "config").read_text()
                + "[remote.origin]\n"
                + "\turl = https://evil.example/acme/repo.git\n"
            )
            policy = _policy(
                root,
                allowed_commands=("remote-list", "fetch"),
                allowed_remote_protocols=("https",),
                allowed_remote_hosts=("github.com",),
            )

            for command, options in cases:
                with self.subTest(command=command):
                    error = await _policy_error(
                        policy,
                        _request(command=command, options=options),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
                    )

    async def test_submodule_effective_config_fails_closed(
        self,
    ) -> None:
        cases = (
            (
                "repo_config_url_override",
                "",
                '[submodule "deps/lib"]\n'
                + "\turl = https://evil.example/acme/lib.git\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "trailing_comment_repo_config_url_override",
                "",
                '[submodule "deps/lib"] # vendored lib\n'
                + "\turl = https://evil.example/acme/lib.git\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "dotted_trailing_comment_repo_config_url_override",
                "",
                "[submodule.foo];vendored lib\n"
                + "\turl = https://evil.example/acme/lib.git\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "dotted_repo_config_url_override",
                "",
                "[submodule.foo]\n"
                + "\turl = https://evil.example/acme/lib.git\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "inline_comment_dotted_repo_config_url_override",
                "",
                "[submodule.foo] # managed by test\n"
                + "\turl = https://evil.example/acme/lib.git\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "dotted_repo_config_path_url_override",
                "",
                "[submodule.deps/lib]\n"
                + "\turl = https://evil.example/acme/lib.git\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "repo_config_update_strategy",
                _gitmodules_config("https://github.com/acme/lib.git"),
                '[submodule "deps/lib"]\n\tupdate = !echo unsafe\n',
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "trailing_comment_repo_config_update_strategy",
                _gitmodules_config("https://github.com/acme/lib.git"),
                '[submodule "deps/lib"] # vendored lib\n'
                + "\tupdate = !echo unsafe\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "dotted_trailing_comment_repo_config_update_strategy",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule.foo];vendored lib\n\tupdate = !echo unsafe\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "dotted_repo_config_update_strategy",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule.foo]\n\tupdate = !echo unsafe\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "inline_comment_dotted_repo_config_update_strategy",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule.foo] # managed by test\n"
                + "\tupdate = !echo unsafe\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "dotted_repo_config_path_update_strategy",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule.deps/lib]\n\tupdate = !echo unsafe\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "gitmodules_update_strategy",
                _gitmodules_config(
                    "https://github.com/acme/lib.git",
                    extra="\tupdate = merge\n",
                ),
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "trailing_comment_gitmodules_update_strategy",
                '[submodule "deps/lib"] # vendored lib\n'
                + "\tpath = deps/lib\n"
                + "\turl = https://github.com/acme/lib.git\n"
                + "\tupdate = merge\n",
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "dotted_gitmodules_update_strategy",
                "[submodule.foo]\n"
                + "\tpath = deps/lib\n"
                + "\turl = https://github.com/acme/lib.git\n"
                + "\tupdate = merge\n",
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "gitmodules_global_recurse_true",
                _gitmodules_config("https://github.com/acme/lib.git")
                + "[submodule]\n\trecurse = true\n",
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "gitmodules_global_recurse_bare",
                _gitmodules_config("https://github.com/acme/lib.git")
                + "[submodule]\n\trecurse\n",
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "gitmodules_global_recurse_unknown",
                _gitmodules_config("https://github.com/acme/lib.git")
                + "[submodule]\n\trecurse = maybe\n",
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "gitmodules_global_recurse_one",
                _gitmodules_config("https://github.com/acme/lib.git")
                + "[submodule]\n\trecurse = 1\n",
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "gitmodules_subsection_recurse_true",
                _gitmodules_config(
                    "https://github.com/acme/lib.git",
                    extra="\trecurse = true\n",
                ),
                "",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "repo_config_global_recurse_true",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule]\n\trecurse = true\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "repo_config_global_recurse_bare",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule]\n\trecurse\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "repo_config_global_recurse_unknown",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule]\n\trecurse = maybe\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "repo_config_global_recurse_yes",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule]\n\trecurse = yes\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "repo_config_global_recurse_on",
                _gitmodules_config("https://github.com/acme/lib.git"),
                "[submodule]\n\trecurse = on\n",
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "repo_config_subsection_recurse_true",
                _gitmodules_config("https://github.com/acme/lib.git"),
                '[submodule "deps/lib"]\n\trecurse = true\n',
                ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
            ),
            (
                "missing_validated_url",
                "",
                "",
                ShellGitExecutionErrorCode.SUBMODULE_DENIED,
            ),
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            base_config = (repo / ".git" / "config").read_text()
            policy = _policy(
                root,
                allowed_commands=("submodule-update",),
                allowed_remote_protocols=("https",),
                allowed_remote_hosts=("github.com",),
                allow_submodule_update=True,
            )

            for name, gitmodules_config, repo_config, error_code in cases:
                with self.subTest(name=name):
                    (repo / ".gitmodules").write_text(gitmodules_config)
                    (repo / ".git" / "config").write_text(
                        base_config + repo_config
                    )
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.SUBMODULE_UPDATE,
                            options={"init": True},
                            pathspecs=("deps/lib",),
                        ),
                    )
                    self.assertEqual(error.error_code, error_code)

    async def test_submodule_recurse_explicit_false_values_are_allowed(
        self,
    ) -> None:
        values = ("false", "no", "off", "0")
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            base_config = (repo / ".git" / "config").read_text()
            submodule_url = _file_url(root / "remote.git")
            policy = _policy(
                root,
                allowed_commands=("submodule-update",),
                allow_submodule_update=True,
            )

            for value in values:
                with self.subTest(source="gitmodules", value=value):
                    (repo / ".gitmodules").write_text(
                        "[core]\n\tbare = false\n"
                        + _gitmodules_config(submodule_url)
                        + f"[submodule]\n\trecurse = {value}\n"
                    )
                    (repo / ".git" / "config").write_text(base_config)
                    spec = await policy.normalize(
                        _request(
                            command=ShellGitCommandName.SUBMODULE_UPDATE,
                            options={"init": True},
                            pathspecs=("deps/lib",),
                        ),
                    )
                    self.assertEqual(
                        spec.metadata["git_submodule_urls"],
                        ("file://localhost/[redacted]",),
                    )

                with self.subTest(source="repo_config", value=value):
                    (repo / ".gitmodules").write_text(
                        _gitmodules_config(submodule_url)
                    )
                    (repo / ".git" / "config").write_text(
                        base_config + f"[submodule]\n\trecurse = {value}\n"
                    )
                    spec = await policy.normalize(
                        _request(
                            command=ShellGitCommandName.SUBMODULE_UPDATE,
                            options={"init": True},
                            pathspecs=("deps/lib",),
                        ),
                    )
                    self.assertEqual(
                        spec.metadata["git_submodule_urls"],
                        ("file://localhost/[redacted]",),
                    )

    async def test_submodule_update_audits_submodule_urls(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            submodule_url = _file_url(root / "remote.git")
            (repo / ".gitmodules").write_text(
                '[submodule "deps/lib"]\n'
                + "\tpath = deps/lib\n"
                + f"\turl = {submodule_url}\n"
            )
            spec = await _policy(
                root,
                allowed_commands=("submodule-update",),
                allow_submodule_update=True,
            ).normalize(
                _request(
                    command=ShellGitCommandName.SUBMODULE_UPDATE,
                    options={"init": True},
                    pathspecs=("deps/lib",),
                ),
            )

        self.assertEqual(spec.metadata["git_remote_protocol"], "file")
        self.assertEqual(spec.metadata["git_remote_host"], "localhost")
        self.assertEqual(
            spec.metadata["git_submodule_protocols"],
            ("file",),
        )
        self.assertEqual(
            spec.metadata["git_submodule_hosts"],
            ("localhost",),
        )
        self.assertEqual(
            spec.metadata["git_submodule_urls"],
            ("file://localhost/[redacted]",),
        )
        self.assertEqual(
            spec.metadata["git_remote_urls"],
            ("file://localhost/[redacted]",),
        )

    async def test_duplicate_submodule_url_keys_are_kept_distinct(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            first_url = _file_url(root / "remote.git")
            second_url = _file_url(root / "other.git")
            (repo / ".gitmodules").write_text(
                '[submodule "deps/lib"]\n'
                + "\tpath = deps/lib\n"
                + f"\turl = {first_url}\n"
                + '[submodule "deps/lib"]\n'
                + "\tpath = deps/lib\n"
                + f"\turl = {second_url}\n"
            )
            spec = await _policy(
                root,
                allowed_commands=("submodule-update",),
                allow_submodule_update=True,
            ).normalize(
                _request(
                    command=ShellGitCommandName.SUBMODULE_UPDATE,
                    options={"init": True},
                    pathspecs=("deps/lib",),
                ),
            )

        self.assertEqual(
            spec.metadata["git_remote_urls"],
            (
                "file://localhost/[redacted]",
                "file://localhost/[redacted]",
            ),
        )

    async def test_legacy_dotted_submodule_urls_are_audited(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            submodule_url = _file_url(root / "remote.git")
            (repo / ".gitmodules").write_text(
                "[submodule.deps/lib]\n"
                + "\tpath = deps/lib\n"
                + f"\turl = {submodule_url}\n"
            )
            spec = await _policy(
                root,
                allowed_commands=("submodule-update",),
                allow_submodule_update=True,
            ).normalize(
                _request(
                    command=ShellGitCommandName.SUBMODULE_UPDATE,
                    options={"init": True},
                    pathspecs=("deps/lib",),
                ),
            )

        self.assertEqual(
            spec.metadata["git_submodule_urls"],
            ("file://localhost/[redacted]",),
        )

    async def test_clone_destination_confinement_and_overwrite(self) -> None:
        cases = (
            "$HOME/repo-copy",
            "../outside",
            "/tmp/outside",
            "-repo-copy",
            "repo-copy/.git/config",
            ".hidden",
            "id_rsa",
            "existing",
            "missing/repo-copy",
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            (root / "existing").mkdir()
            (root / "link").symlink_to(root.parent, target_is_directory=True)
            policy = _policy(root, cwd=".", allowed_commands=("clone",))

            for destination in cases:
                with self.subTest(destination=destination):
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.CLONE,
                            options={
                                "url": _file_url(root / "remote.git"),
                                "destination": destination,
                                "branch": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

            pathspec_error = await _policy_error(
                policy,
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _file_url(root / "remote.git"),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                    pathspecs=("other-copy",),
                ),
            )
            symlink_error = await _policy_error(
                policy,
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _file_url(root / "remote.git"),
                        "destination": "link/repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            pathspec_error.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertEqual(
            symlink_error.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )

    async def test_default_clone_destination_denies_hidden_path(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _file_url(root / "remote.git"),
                        "destination": ".repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )

    async def test_allow_hidden_clone_destination_is_in_argv(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            policy = GitExecutionPolicy(
                settings=ShellToolSettings(
                    allow_hidden=True,
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd=".",
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_protocols=("file",),
                        allowed_remote_hosts=("localhost",),
                    ),
                ),
                executable_lookup=_fake_executable,
            )

            spec = await policy.normalize(
                _request(
                    command=ShellGitCommandName.CLONE,
                    options={
                        "url": _file_url(root / "remote.git"),
                        "destination": ".repo-copy",
                        "branch": "main",
                    },
                )
            )

        self.assertIn(".repo-copy", spec.argv)
        self.assertEqual(spec.argv[-1], ".repo-copy")

    async def test_allow_hidden_clone_destination_denies_sensitive_paths(
        self,
    ) -> None:
        cases = (
            ".git/config",
            ".ssh/repo-copy",
            ".env-copy",
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            policy = GitExecutionPolicy(
                settings=ShellToolSettings(
                    allow_hidden=True,
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd=".",
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_protocols=("file",),
                        allowed_remote_hosts=("localhost",),
                    ),
                ),
                executable_lookup=_fake_executable,
            )

            for destination in cases:
                with self.subTest(destination=destination):
                    error = await _policy_error(
                        policy,
                        _request(
                            command=ShellGitCommandName.CLONE,
                            options={
                                "url": _file_url(root / "remote.git"),
                                "destination": destination,
                                "branch": "main",
                            },
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

    async def test_clone_cwd_must_stay_inside_workspace(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            unsafe_error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    cwd="$HOME/repo",
                    options={
                        "url": _file_url(root / "remote.git"),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )
            outside_error = await _policy_error(
                _policy(root, cwd=".", allowed_commands=("clone",)),
                _request(
                    command=ShellGitCommandName.CLONE,
                    cwd=str(root.parent),
                    options={
                        "url": _file_url(root / "remote.git"),
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                ),
            )

        self.assertEqual(
            unsafe_error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )
        self.assertEqual(
            outside_error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

    async def test_remote_argv_and_audit_metadata_are_bounded(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote_url = _file_url(root / "remote.git")
            _write_minimal_bare_git_repo(root / "remote.git")
            _write_minimal_git_repo(root / "repo", remote_url=remote_url)
            policy = _policy(
                root,
                capabilities=("remote", "worktree", "history"),
                allowed_commands=_REMOTE_COMMANDS,
                allow_submodule_update=True,
            )

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.FETCH,
                        options={
                            "remote": "origin",
                            "ref_type": "branch",
                            "ref_name": "main",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "fetch",
                        "--no-tags",
                        "--no-prune",
                        "--no-recurse-submodules",
                        "--no-write-fetch-head",
                        "origin",
                        "refs/heads/main:refs/remotes/origin/main",
                    ),
                    ("refs/heads/main",),
                    False,
                ),
                (
                    _request(
                        command=ShellGitCommandName.FETCH,
                        options={
                            "remote": "origin",
                            "ref_type": "tag",
                            "ref_name": "v1.0",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "fetch",
                        "--no-tags",
                        "--no-prune",
                        "--no-recurse-submodules",
                        "--no-write-fetch-head",
                        "origin",
                        "refs/tags/v1.0:refs/tags/v1.0",
                    ),
                    ("refs/tags/v1.0",),
                    False,
                ),
                (
                    _request(
                        command=ShellGitCommandName.PULL,
                        options={"remote": "origin", "branch": "main"},
                    ),
                    (
                        *_git_prefix(),
                        "pull",
                        "--ff-only",
                        "--no-verify",
                        "--no-tags",
                        "--no-prune",
                        "--no-recurse-submodules",
                        "origin",
                        "main",
                    ),
                    ("refs/heads/main",),
                    False,
                ),
                (
                    _request(
                        command=ShellGitCommandName.PUSH,
                        options={
                            "remote": "origin",
                            "ref_type": "branch",
                            "ref_name": "main",
                        },
                    ),
                    (
                        *_git_prefix(),
                        "push",
                        "--no-verify",
                        "--porcelain",
                        "origin",
                        "refs/heads/main:refs/heads/main",
                    ),
                    ("refs/heads/main",),
                    True,
                ),
            )

            for request, argv, refs, mutates_remote in cases:
                with self.subTest(command=request.command):
                    spec = await policy.normalize(request)
                    self.assertEqual(spec.argv, argv)
                    self.assertEqual(
                        spec.metadata["git_remote_host"],
                        "localhost",
                    )
                    self.assertEqual(
                        spec.metadata["git_remote_protocol"],
                        "file",
                    )
                    self.assertEqual(spec.metadata["git_selected_refs"], refs)
                    self.assertEqual(
                        spec.metadata["git_remote_state_may_mutate"],
                        mutates_remote,
                    )
                    self.assertNotIn(str(root), str(spec.display_argv))
                    self.assertIn("GIT_TERMINAL_PROMPT", spec.env)
                    self.assertEqual(spec.env["GIT_TERMINAL_PROMPT"], "0")
                    self.assertEqual(spec.env["GIT_ASKPASS"], "/nonexistent")
                    self.assertEqual(spec.env["GIT_SSH"], "/nonexistent")
                    self.assertNotIn("GIT_SSH_COMMAND", spec.env)
                    self.assertEqual(spec.env["GIT_ALLOW_PROTOCOL"], "file")

    def test_remote_audit_metadata_handles_unparseable_urls(self) -> None:
        settings = ShellGitToolSettings(
            allowed_remote_protocols=("https",),
            allowed_remote_hosts=("github.com",),
        )
        metadata = git_remote_audit_metadata(
            _request(
                command=ShellGitCommandName.CLONE,
                options={
                    "url": "//[github.com]/repo",
                    "destination": "repo-copy",
                    "branch": "main",
                },
            ),
            settings=settings,
        )
        missing_scheme_metadata = git_remote_audit_metadata(
            _request(
                command=ShellGitCommandName.CLONE,
                options={
                    "url": "github.com/acme/repo",
                    "destination": "repo-copy",
                    "branch": "main",
                },
            ),
            settings=settings,
        )

        self.assertIsNone(metadata["git_remote_protocol"])
        self.assertIsNone(metadata["git_remote_host"])
        self.assertIsNone(missing_scheme_metadata["git_remote_protocol"])
        self.assertIsNone(missing_scheme_metadata["git_remote_host"])

    async def test_remote_ref_forms_fail_closed(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(
                root / "repo",
                remote_url=_file_url(root / "remote.git"),
            )
            error = await _policy_error(
                _policy(root, allowed_commands=("fetch",)),
                _request(
                    command=ShellGitCommandName.FETCH,
                    options={
                        "remote": "origin",
                        "ref_type": "branch",
                        "ref_name": "../main",
                    },
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REVISION_DENIED,
        )


class GitRemoteRedactionPhase7Test(IsolatedAsyncioTestCase):
    async def test_remote_urls_are_redacted_across_result_surfaces(
        self,
    ) -> None:
        secret_url = "https://token@github.com/acme/repo.git"
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            executor = _FakeGitExecutor(
                stdout=f"remote {secret_url}\n",
                stderr=f"diagnostic {secret_url}\n",
                metadata={"url": secret_url},
            )
            toolset = ShellToolSet(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd=".",
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_protocols=("https",),
                        allowed_remote_hosts=("github.com",),
                        allow_remote_credentials=True,
                    )
                ),
                executor=executor,
            ).with_enabled_tools(["shell.git_clone"])

            result = await _call_tool(
                toolset,
                "git_clone",
                url=secret_url,
                destination="repo-copy",
                branch="main",
            )

        surfaces = (
            str(result),
            result.git_result.stdout_snippet,
            result.git_result.stderr_snippet,
            str(result.git_result.display_argv),
            str(result.git_result.audit_metadata),
        )
        for surface in surfaces:
            with self.subTest(surface=surface):
                self.assertNotIn("token", surface)
                self.assertNotIn("repo.git", surface)
        self.assertIn("https://github.com/[redacted]", str(result))


@skipIf(_GIT_BINARY is None, "git executable is required for smoke tests")
class GitRemoteSmokePhase7Test(IsolatedAsyncioTestCase):
    async def test_fetch_pull_push_clone_with_local_bare_remote(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            seed, remote = _write_seeded_bare_remote(root, _GIT_BINARY)

            fetch_repo = root / "fetcher"
            _git(root, _GIT_BINARY, "init", "fetcher")
            _git(fetch_repo, _GIT_BINARY, "checkout", "-b", "main")
            _git(
                fetch_repo,
                _GIT_BINARY,
                "remote",
                "add",
                "origin",
                _file_url(remote),
            )
            fetch_toolset = _real_remote_toolset(
                root,
                fetch_repo,
                _GIT_BINARY,
                allowed_commands=("fetch",),
            )
            fetch = await _call_tool(
                fetch_toolset,
                "git_fetch",
                ref_name="main",
            )
            self.assertEqual(
                fetch.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )

            pull_repo = root / "puller"
            _git(root, _GIT_BINARY, "clone", _file_url(remote), "puller")
            _git(
                pull_repo,
                _GIT_BINARY,
                "checkout",
                "-b",
                "main",
                "origin/main",
            )
            (seed / "tracked.txt").write_text("updated\n")
            _git(seed, _GIT_BINARY, "add", "tracked.txt")
            _git(seed, _GIT_BINARY, "commit", "-m", "update")
            _git(seed, _GIT_BINARY, "push", "origin", "main")
            pull_toolset = _real_remote_toolset(
                root,
                pull_repo,
                _GIT_BINARY,
                allowed_commands=("pull",),
            )
            pull = await _call_tool(pull_toolset, "git_pull", branch="main")
            self.assertEqual(
                pull.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertEqual(
                (pull_repo / "tracked.txt").read_text(),
                "updated\n",
            )

            push_repo = root / "pusher"
            _git(root, _GIT_BINARY, "clone", _file_url(remote), "pusher")
            _git(
                push_repo,
                _GIT_BINARY,
                "checkout",
                "-b",
                "main",
                "origin/main",
            )
            (push_repo / "pushed.txt").write_text("pushed\n")
            _git(push_repo, _GIT_BINARY, "add", "pushed.txt")
            _git(push_repo, _GIT_BINARY, "commit", "-m", "push update")
            push_toolset = _real_remote_toolset(
                root,
                push_repo,
                _GIT_BINARY,
                allowed_commands=("push",),
            )
            push = await _call_tool(push_toolset, "git_push", ref_name="main")
            self.assertEqual(
                push.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )

            clone_toolset = _real_remote_toolset(
                root,
                root,
                _GIT_BINARY,
                allowed_commands=("clone",),
            )
            clone = await _call_tool(
                clone_toolset,
                "git_clone",
                url=_file_url(remote),
                destination="cloned",
                branch="main",
            )
            self.assertEqual(
                clone.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertTrue((root / "cloned" / "tracked.txt").is_file())

    async def test_push_denies_non_bare_file_target_before_execution(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            remote = _write_real_git_repo(root, _GIT_BINARY, "remote-repo")
            push_repo = root / "pusher"
            _git(root, _GIT_BINARY, "clone", _file_url(remote), "pusher")
            (push_repo / "pushed.txt").write_text("pushed\n")
            _git(push_repo, _GIT_BINARY, "add", "pushed.txt")
            _git(push_repo, _GIT_BINARY, "commit", "-m", "push update")
            push_toolset = _real_remote_toolset(
                root,
                push_repo,
                _GIT_BINARY,
                allowed_commands=("push",),
            )
            push = await _call_tool(push_toolset, "git_push", ref_name="main")

            self.assertEqual(
                push.git_result.status,
                ShellGitExecutionStatus.POLICY_DENIED,
            )
            self.assertEqual(
                push.git_result.error_code,
                ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            )

    async def test_push_denies_bare_file_target_with_commondir(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _, remote = _write_seeded_bare_remote(root, _GIT_BINARY)
            push_repo = root / "pusher"
            _git(root, _GIT_BINARY, "clone", _file_url(remote), "pusher")
            _git(
                push_repo,
                _GIT_BINARY,
                "checkout",
                "-b",
                "main",
                "origin/main",
            )
            (push_repo / "pushed.txt").write_text("pushed\n")
            _git(push_repo, _GIT_BINARY, "add", "pushed.txt")
            _git(push_repo, _GIT_BINARY, "commit", "-m", "push update")
            _git(root, _GIT_BINARY, "init", "--bare", "common.git")
            marker = root / "common-hook-ran"
            hook = root / "common.git" / "hooks" / "pre-receive"
            hook.write_text(f'#!/bin/sh\nprintf ran > "{marker}"\nexit 1\n')
            hook.chmod(0o755)
            (remote / "commondir").write_text("../common.git\n")
            push_toolset = _real_remote_toolset(
                root,
                push_repo,
                _GIT_BINARY,
                allowed_commands=("push",),
            )
            push = await _call_tool(push_toolset, "git_push", ref_name="main")

            self.assertEqual(
                push.git_result.status,
                ShellGitExecutionStatus.POLICY_DENIED,
            )
            self.assertEqual(
                push.git_result.error_code,
                ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            )
            self.assertFalse(marker.exists())

    async def test_push_denies_active_target_hook_before_execution(
        self,
    ) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _, remote = _write_seeded_bare_remote(root, _GIT_BINARY)
            push_repo = root / "pusher"
            _git(root, _GIT_BINARY, "clone", _file_url(remote), "pusher")
            _git(
                push_repo,
                _GIT_BINARY,
                "checkout",
                "-b",
                "main",
                "origin/main",
            )
            (push_repo / "pushed.txt").write_text("pushed\n")
            _git(push_repo, _GIT_BINARY, "add", "pushed.txt")
            _git(push_repo, _GIT_BINARY, "commit", "-m", "push update")
            marker = root / "target-hook-ran"
            hook = remote / "hooks" / "pre-receive"
            hook.write_text(f'#!/bin/sh\nprintf ran > "{marker}"\nexit 1\n')
            hook.chmod(0o755)
            push_toolset = _real_remote_toolset(
                root,
                push_repo,
                _GIT_BINARY,
                allowed_commands=("push",),
            )
            push = await _call_tool(push_toolset, "git_push", ref_name="main")

            self.assertEqual(
                push.git_result.status,
                ShellGitExecutionStatus.POLICY_DENIED,
            )
            self.assertEqual(
                push.git_result.error_code,
                ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
            )
            self.assertFalse(marker.exists())

    async def test_push_allows_sample_and_inactive_target_hooks(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _, remote = _write_seeded_bare_remote(root, _GIT_BINARY)
            push_repo = root / "pusher"
            _git(root, _GIT_BINARY, "clone", _file_url(remote), "pusher")
            _git(
                push_repo,
                _GIT_BINARY,
                "checkout",
                "-b",
                "main",
                "origin/main",
            )
            marker = root / "target-hook-ran"
            sample_hook = remote / "hooks" / "pre-receive.sample"
            sample_hook.write_text(
                f'#!/bin/sh\nprintf sample > "{marker}"\nexit 1\n',
            )
            sample_hook.chmod(0o755)
            inactive_hook = remote / "hooks" / "post-receive"
            inactive_hook.write_text(
                f'#!/bin/sh\nprintf inactive > "{marker}"\nexit 1\n',
            )
            inactive_hook.chmod(0o644)
            (push_repo / "pushed.txt").write_text("pushed\n")
            _git(push_repo, _GIT_BINARY, "add", "pushed.txt")
            _git(push_repo, _GIT_BINARY, "commit", "-m", "push update")
            push_toolset = _real_remote_toolset(
                root,
                push_repo,
                _GIT_BINARY,
                allowed_commands=("push",),
            )
            push = await _call_tool(push_toolset, "git_push", ref_name="main")

            self.assertEqual(
                push.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertFalse(marker.exists())

    async def test_remote_management_with_local_file_urls(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY, "repo")
            _git(root, _GIT_BINARY, "init", "--bare", "remote.git")
            _git(root, _GIT_BINARY, "init", "--bare", "other.git")
            toolset = _real_remote_toolset(
                root,
                repo,
                _GIT_BINARY,
                allowed_commands=(
                    "remote-add",
                    "remote-list",
                    "remote-set-url",
                    "remote-rename",
                    "remote-remove",
                ),
            )

            add = await _call_tool(
                toolset,
                "git_remote_add",
                name="origin",
                url=_file_url(root / "remote.git"),
            )
            listed = await _call_tool(toolset, "git_remote_list")
            set_url = await _call_tool(
                toolset,
                "git_remote_set_url",
                name="origin",
                url=_file_url(root / "other.git"),
            )
            rename = await _call_tool(
                toolset,
                "git_remote_rename",
                old_name="origin",
                new_name="upstream",
            )
            remove = await _call_tool(
                toolset,
                "git_remote_remove",
                name="upstream",
            )

        for result in (add, listed, set_url, rename, remove):
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
        self.assertIn(
            "file://localhost/[redacted]",
            listed.git_result.stdout_snippet,
        )
        self.assertNotIn("remote.git", listed.git_result.stdout_snippet)


def _schema_names(toolset: ShellToolSet) -> tuple[str, ...]:
    schemas = toolset.json_schemas()
    return tuple(schema["function"]["name"] for schema in schemas or ())


def _schema_field_names(schema: object) -> set[str]:
    names: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            names.update(str(name) for name in properties)
        for value in schema.values():
            names.update(_schema_field_names(value))
    elif isinstance(schema, list):
        for item in schema:
            names.update(_schema_field_names(item))
    return names


def _policy(
    workspace_root: Path,
    *,
    cwd: str = "repo",
    capabilities: tuple[str, ...] = ("remote",),
    allowed_commands: tuple[str, ...] = _REMOTE_COMMANDS,
    allowed_remote_protocols: tuple[str, ...] = ("file",),
    allowed_remote_hosts: tuple[str, ...] = ("localhost",),
    allow_remote_credentials: bool = False,
    allow_submodule_update: bool = False,
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=str(workspace_root),
                cwd=cwd,
                capabilities=capabilities,
                allowed_commands=allowed_commands,
                allowed_remote_protocols=allowed_remote_protocols,
                allowed_remote_hosts=allowed_remote_hosts,
                allow_remote_credentials=allow_remote_credentials,
                allow_submodule_update=allow_submodule_update,
            )
        ),
        executable_lookup=_fake_executable,
    )


def _request(
    *,
    command: ShellGitCommandName = ShellGitCommandName.FETCH,
    options: dict[str, object] | None = None,
    pathspecs: tuple[str, ...] = (),
    cwd: str | None = None,
) -> ShellGitCommandRequest:
    return ShellGitCommandRequest(
        tool_name=f"shell.git_{command.value.replace('-', '_')}",
        command=command,
        capability_required=ShellGitCapability.REMOTE,
        options={} if options is None else options,
        pathspecs=pathspecs,
        cwd=cwd,
    )


def _remote_command_options(
    command: ShellGitCommandName,
) -> dict[str, object]:
    if command is ShellGitCommandName.PULL:
        return {"remote": "origin", "branch": "main"}
    if command is ShellGitCommandName.PUSH:
        return {
            "remote": "origin",
            "ref_type": "branch",
            "ref_name": "main",
        }
    return {
        "remote": "origin",
        "ref_type": "branch",
        "ref_name": "main",
    }


def _push_request() -> ShellGitCommandRequest:
    return _request(
        command=ShellGitCommandName.PUSH,
        options=_remote_command_options(ShellGitCommandName.PUSH),
    )


def _write_push_repo_request(
    root: Path,
    remote: Path,
) -> ShellGitCommandRequest:
    _write_minimal_git_repo(root / "repo", remote_url=_file_url(remote))
    return _push_request()


async def _push_policy_error(
    root: Path,
    remote: Path,
) -> ShellGitPolicyDenied:
    return await _policy_error(
        _policy(root, allowed_commands=("push",)),
        _write_push_repo_request(root, remote),
    )


def _gitmodules_config(url: str, *, extra: str = "") -> str:
    return (
        '[submodule "deps/lib"]\n'
        + "\tpath = deps/lib\n"
        + f"\turl = {url}\n"
        + extra
    )


def _remote_config(url: str) -> str:
    return (
        '[remote "origin"]\n'
        + f"\turl = {url}\n"
        + "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
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


def _git_prefix() -> tuple[str, str, str]:
    return "git", "--no-pager", "--no-optional-locks"


def _write_minimal_git_repo(
    repo: Path,
    *,
    remote_url: str | None = None,
) -> Path:
    git_dir = repo / ".git"
    (git_dir / "objects" / "info").mkdir(parents=True)
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "refs" / "heads" / "main").write_text(
        "1111111111111111111111111111111111111111\n"
    )
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    config = "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    if remote_url is not None:
        config += _remote_config(remote_url)
    (git_dir / "config").write_text(config)
    return repo


def _write_minimal_bare_git_repo(repo: Path) -> Path:
    (repo / "objects" / "info").mkdir(parents=True)
    (repo / "objects" / "pack").mkdir()
    (repo / "refs" / "heads").mkdir(parents=True)
    (repo / "refs" / "tags").mkdir(parents=True)
    (repo / "hooks").mkdir()
    (repo / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = true\n"
    )
    return repo


def _file_url(path: Path) -> str:
    return f"file://localhost{path.resolve().as_posix()}"


def _hostless_file_url(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


def _real_remote_toolset(
    root: Path,
    cwd: Path,
    git_binary: str,
    *,
    allowed_commands: tuple[str, ...],
) -> ShellToolSet:
    return ShellToolSet(
        settings=ShellToolSettings(
            executable_search_paths=(str(Path(git_binary).parent),),
            git=ShellGitToolSettings(
                workspace_root=str(root),
                cwd="." if cwd == root else cwd.relative_to(root).as_posix(),
                capabilities=("remote", "worktree", "history"),
                allowed_commands=allowed_commands,
                allowed_remote_protocols=("file",),
                allowed_remote_hosts=("localhost",),
            ),
        ),
    ).with_enabled_tools(
        [
            f"shell.git_{command.replace('-', '_')}"
            for command in allowed_commands
        ]
    )


def _write_seeded_bare_remote(
    root: Path,
    git_binary: str,
) -> tuple[Path, Path]:
    seed = _write_real_git_repo(root, git_binary, "seed")
    remote = root / "remote.git"
    _git(root, git_binary, "init", "--bare", "remote.git")
    _git(seed, git_binary, "remote", "add", "origin", _file_url(remote))
    _git(seed, git_binary, "push", "origin", "main")
    return seed, remote


def _write_real_git_repo(root: Path, git_binary: str, name: str) -> Path:
    repo = root / name
    _git(root, git_binary, "init", name)
    _git(repo, git_binary, "checkout", "-b", "main")
    _git(repo, git_binary, "config", "user.name", "Avalan Test")
    _git(repo, git_binary, "config", "user.email", "avalan@example.test")
    (repo / "tracked.txt").write_text("base\n")
    _git(repo, git_binary, "add", "tracked.txt")
    _git(repo, git_binary, "commit", "-m", "initial")
    return repo


def _git(cwd: Path, git_binary: str, *args: str) -> None:
    isolation_root = _git_isolation_root(cwd)
    env = _git_env(isolation_root)
    result = run(
        (git_binary, *args),
        cwd=cwd,
        env=env,
        check=False,
        stdout=DEVNULL,
        stderr=DEVNULL,
        text=True,
    )
    assert isinstance(result, CompletedProcess)
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed")


def _git_isolation_root(cwd: Path) -> Path:
    current = cwd.resolve()
    for candidate in (current, *current.parents):
        if candidate.name.startswith("tmp"):
            return candidate
    return current


def _git_env(isolation_root: Path) -> dict[str, str]:
    home = isolation_root / "home"
    paths = (
        home,
        isolation_root / "templates",
        isolation_root / "templates" / "hooks",
        isolation_root / "tmp",
        isolation_root / "xdg-cache",
        isolation_root / "xdg-config",
        isolation_root / "xdg-data",
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "PATH": environ.get("PATH", ""),
        "GIT_ASKPASS": "",
        "GIT_CONFIG_GLOBAL": devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": devnull,
        "GIT_ALLOW_PROTOCOL": "file:https",
        "GIT_EDITOR": "true",
        "GIT_PAGER": "cat",
        "GIT_TEMPLATE_DIR": str(isolation_root / "templates"),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "Avalan Test",
        "GIT_AUTHOR_EMAIL": "avalan@example.test",
        "GIT_COMMITTER_NAME": "Avalan Test",
        "GIT_COMMITTER_EMAIL": "avalan@example.test",
        "HOME": str(home),
        "SSH_ASKPASS": "",
        "TEMP": str(isolation_root / "tmp"),
        "TMP": str(isolation_root / "tmp"),
        "TMPDIR": str(isolation_root / "tmp"),
        "XDG_CACHE_HOME": str(isolation_root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(isolation_root / "xdg-config"),
        "XDG_DATA_HOME": str(isolation_root / "xdg-data"),
    }


class _FakeGitExecutor:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        metadata: dict[str, object] | None = None,
        status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
        exit_code: int | None = 0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._metadata = {} if metadata is None else metadata
        self._status = status
        self._exit_code = exit_code

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        assert stream is None
        return ExecutionResult(
            backend=spec.backend,
            tool_name=spec.tool_name,
            command=spec.command,
            argv=spec.argv,
            display_argv=spec.display_argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=self._status,
            exit_code=self._exit_code,
            stdout=self._stdout,
            stderr=self._stderr,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            stdout_bytes=len(self._stdout.encode("utf-8")),
            stderr_bytes=len(self._stderr.encode("utf-8")),
            stdout_truncated=False,
            stderr_truncated=False,
            timed_out=self._status is ShellExecutionStatus.TIMEOUT,
            duration_ms=3,
            error_message=None,
            metadata={**spec.metadata, **self._metadata},
        )


async def _call_tool(
    toolset: ShellToolSet,
    command_id: str,
    **kwargs: object,
) -> ShellGitFormattedResult:
    tool = _tool_by_name(toolset, command_id)
    result = await tool(**kwargs, context=ToolCallContext())
    assert isinstance(result, ShellGitFormattedResult)
    return result


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> _GitToolCallable:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            return cast(_GitToolCallable, tool)
    raise AssertionError(f"tool not found: {command_id}")
