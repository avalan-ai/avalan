from ast import AST, Call, Import, ImportFrom, parse, walk
from asyncio import CancelledError
from hashlib import sha1
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.entities import ToolCallContext
from avalan.tool.shell import (
    GitExecutionPolicy,
    ShellGitCapability,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitExecutionErrorCode,
    ShellGitExecutionStatus,
    ShellGitFormattedResult,
    ShellGitToolSettings,
    ShellOutputKind,
    ShellToolSettings,
)
from avalan.tool.shell.entities import (
    ExecutionResult,
    ExecutionSpec,
    ShellExecutionStatus,
)
from avalan.tool.shell.git import ShellGitPolicyDenied
from avalan.tool.shell.git_policy import (
    _display_path,
    _redacted_metadata,
    git_remote_audit_metadata,
    redact_git_text,
)
from avalan.tool.shell.git_policy import (
    _read_bytes as _git_policy_read_bytes,
)
from avalan.tool.shell.tools import GitStatusTool, _git_policy_denied_result


class GitExecutionPolicyRepositoryTest(IsolatedAsyncioTestCase):
    async def test_safe_repository_discovery_normalizes_repo_and_cwd(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / "src").mkdir()
            policy = _policy(root)

            spec = await policy.normalize(_status_request(cwd="repo/src"))

        self.assertEqual(spec.cwd, str(repo.resolve()))
        self.assertEqual(spec.display_cwd, "repo/src")
        self.assertEqual(spec.metadata["git_repo_root"], "repo")
        self.assertEqual(spec.metadata["git_effective_cwd"], "repo/src")
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
            ),
        )

    async def test_configured_search_paths_discover_git_executable(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            git = bin_dir / "git"
            git.write_text("#!/bin/sh\nexit 0\n")
            git.chmod(0o700)

            spec = await GitExecutionPolicy(
                settings=_settings(
                    root,
                    executable_search_paths=(str(bin_dir),),
                )
            ).normalize(_status_request())

        self.assertEqual(spec.executable, str(git))

    async def test_configured_git_executable_path_skips_lookup(self) -> None:
        async def fail_lookup(_search_paths: tuple[str, ...]) -> str | None:
            raise AssertionError("lookup should not be called")

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")

            spec = await GitExecutionPolicy(
                settings=_settings(
                    root,
                    git_executable_path="/usr/bin/git",
                ),
                executable_lookup=fail_lookup,
            ).normalize(_status_request())

        self.assertEqual(spec.executable, "/usr/bin/git")

    async def test_read_commands_do_not_require_supported_index_form(
        self,
    ) -> None:
        index_cases: tuple[tuple[str, bytes | None], ...] = (
            ("absent", None),
            ("huge", b"x" * (8 * 1024 * 1024 + 1)),
            ("malformed", b"not an index"),
            ("v4", _git_index_v4_data()),
        )

        for name, index_data in index_cases:
            with self.subTest(index=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    repo = _write_minimal_git_repo(root / "repo")
                    if index_data is not None:
                        (repo / ".git" / "index").write_bytes(index_data)
                    policy = _policy(
                        root,
                        allowed_commands=("status", "log"),
                    )

                    status_spec = await policy.normalize(_status_request())
                    log_spec = await policy.normalize(
                        _request(command=ShellGitCommandName.LOG)
                    )

                self.assertEqual(status_spec.command, "git.status")
                self.assertEqual(log_spec.command, "git.log")

    async def test_read_commands_do_not_read_index(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "index").write_bytes(_git_index_v4_data())
            policy = _policy(
                root,
                allowed_commands=("status", "log"),
            )

            with patch(
                "avalan.tool.shell.git_policy._read_bytes",
                side_effect=_fail_on_index_read,
            ):
                status_spec = await policy.normalize(_status_request())
                log_spec = await policy.normalize(
                    _request(command=ShellGitCommandName.LOG)
                )

        self.assertEqual(status_spec.command, "git.status")
        self.assertEqual(log_spec.command, "git.log")

    def test_display_path_uses_outside_workspace_placeholder(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            outside = root.parent / f"{root.name}-outside"

            display_path = _display_path(root, outside)

        self.assertEqual(display_path, "[outside-workspace]")

    async def test_non_repository_cwd_returns_repo_not_found(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            (root / "plain").mkdir()

            error = await _policy_error(_policy(root, cwd="plain"))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_NOT_FOUND,
        )

    async def test_discovery_does_not_climb_outside_workspace(self) -> None:
        with TemporaryDirectory() as outer:
            outer_root = Path(outer)
            _write_minimal_git_repo(outer_root)
            workspace = outer_root / "workspace"
            workspace.mkdir()

            error = await _policy_error(_policy(workspace))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_NOT_FOUND,
        )

    async def test_workspace_and_cwd_boundaries_fail_closed(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(_policy(root / "missing"))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(_policy(root, cwd="~repo"))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(_policy(root, cwd="../outside"))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

    async def test_authorization_failures_are_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            bad_tool_request = ShellGitCommandRequest(
                tool_name="shell.git_wrong",
                command=ShellGitCommandName.STATUS,
                capability_required=ShellGitCapability.READ,
                options={},
            )

            tool_error = await _policy_error(
                _policy(root),
                request=bad_tool_request,
            )
            command_error = await _policy_error(
                _policy(root, allowed_commands=("log",)),
            )
            capability_error = await _policy_error(
                _policy(root, capabilities=("remote",)),
            )
            submodule_error = await _policy_error(
                _policy(
                    root,
                    capabilities=("remote",),
                    allowed_commands=("submodule-update",),
                    allowed_remote_hosts=("github.com",),
                ),
                request=_request(
                    command=ShellGitCommandName.SUBMODULE_UPDATE,
                    capability=ShellGitCapability.REMOTE,
                ),
            )

        self.assertEqual(
            tool_error.error_code,
            ShellGitExecutionErrorCode.COMMAND_DISABLED,
        )
        self.assertEqual(
            command_error.error_code,
            ShellGitExecutionErrorCode.COMMAND_DISABLED,
        )
        self.assertEqual(
            capability_error.error_code,
            ShellGitExecutionErrorCode.CAPABILITY_REQUIRED,
        )
        self.assertEqual(
            submodule_error.error_code,
            ShellGitExecutionErrorCode.SUBMODULE_DENIED,
        )

    async def test_repository_forms_fail_closed(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            bare = root / "bare.git"
            (bare / "objects").mkdir(parents=True)
            (bare / "refs").mkdir()
            (bare / "HEAD").write_text("ref: refs/heads/main\n")

            error = await _policy_error(_policy(root, cwd="bare.git"))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.BARE_REPO_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").write_text("gitdir: /outside/repo.git\n")

            error = await _policy_error(_policy(root))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            metadata = root / "metadata"
            metadata.mkdir()
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").symlink_to(
                metadata,
                target_is_directory=True,
            )

            error = await _policy_error(_policy(root))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "commondir").write_text("../common.git\n")

            error = await _policy_error(_policy(root))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            alternates = repo / ".git" / "objects" / "info" / "alternates"
            alternates.write_text("/outside/objects\n")

            error = await _policy_error(_policy(root))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.ALTERNATE_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            alternates = repo / ".git" / "objects" / "info" / "alternates"
            alternates.write_text("/outside/objects\n")

            error = await _policy_error(
                _policy(root, allow_alternates=True),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.ALTERNATE_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").write_text("[core]\n\tbare = true\n")

            error = await _policy_error(_policy(root))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.BARE_REPO_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "commondir").write_text("/outside/common\n")

            error = await _policy_error(
                _policy(root, allow_linked_worktrees=True),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.REPO_BOUNDARY_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            alternates = repo / ".git" / "objects" / "info" / "alternates"
            alternates.write_text("\n# comment\n..\n")

            spec = await _policy(root, allow_alternates=True).normalize(
                _status_request(),
            )

        self.assertEqual(spec.metadata["git_repo_root"], "repo")

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").unlink()

            spec = await _policy(root).normalize(_status_request())

        self.assertEqual(spec.metadata["git_repo_root"], "repo")

    async def test_submodule_recursion_fails_closed(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(
                    root,
                    capabilities=("remote",),
                    allowed_commands=("submodule-update",),
                    allow_submodule_update=True,
                    allowed_remote_hosts=("github.com",),
                ),
                request=_request(
                    command=ShellGitCommandName.SUBMODULE_UPDATE,
                    capability=ShellGitCapability.REMOTE,
                    options={"recursive": True},
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.SUBMODULE_DENIED,
        )

    async def test_unsafe_pathspec_forms_fail_closed(self) -> None:
        unsafe_pathspecs = (
            "/absolute",
            "../outside",
            "safe/../../outside",
            "bad\x00path",
            "bad\npath",
            ":(top)file",
            "$HOME/file",
            "*.py",
            "-option-looking",
            "rev:path",
            "windows\\path",
            ".git/config",
            "del\x7fpath",
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            for pathspec in unsafe_pathspecs:
                with self.subTest(pathspec=pathspec):
                    error = await _policy_error(
                        policy,
                        request=_status_request(pathspecs=(pathspec,)),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

            outside_repo = root / "outside"
            outside_repo.mkdir()
            (root / "repo" / "link").symlink_to(outside_repo)
            error = await _policy_error(
                policy,
                request=_status_request(pathspecs=("link/file",)),
            )
            self.assertEqual(
                error.error_code,
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            )

    async def test_unsafe_revisions_and_options_fail_closed(self) -> None:
        unsafe_revisions = (
            "-HEAD",
            "HEAD:README.md",
            "src/main.py",
            "../HEAD",
            "feature/name",
            "stash@{0}",
            "bad\x01ref",
        )
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root, allowed_commands=("log",))
            status_policy = _policy(root)

            for revision in unsafe_revisions:
                with self.subTest(revision=revision):
                    error = await _policy_error(
                        policy,
                        request=_request(
                            command=ShellGitCommandName.LOG,
                            options={"revision": revision},
                        ),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.REVISION_DENIED,
                    )

            non_string_error = await _policy_error(
                policy,
                request=_request(
                    command=ShellGitCommandName.LOG,
                    options={"revision": 123},
                ),
            )
            none_revision_spec = await policy.normalize(
                _request(
                    command=ShellGitCommandName.LOG,
                    options={"revision": None},
                ),
            )
            self.assertEqual(
                non_string_error.error_code,
                ShellGitExecutionErrorCode.REVISION_DENIED,
            )
            self.assertNotIn("None", none_revision_spec.argv)

            option_cases: tuple[dict[str, object], ...] = (
                {"mode": "short", "pager": "less"},
                {"mode": "short", "extra": "--git-dir=/tmp/repo"},
                {"mode": "short", "extra": "-c"},
                {"mode": "short", "extra": "core.fsmonitor=true"},
                {"mode": "short", "extra": "!shell"},
                {"mode": "short", "extra": ["--git-dir=/tmp/repo"]},
                {"mode": "short", "editor": "vi"},
                {"mode": "short", "external_diff": "tool"},
                {"mode": "short", "filter": "secret"},
                {"mode": "short", "hook": "pre-commit"},
                {"mode": "short", "signing": True},
                {"mode": "short", "prompt": True},
                {"mode": "short", "credential": "helper"},
                {"mode": "short", "ssh_command": "ssh -i key"},
                {"mode": "short", "askpass": "helper"},
            )
            for options in option_cases:
                with self.subTest(options=options):
                    error = await _policy_error(
                        status_policy,
                        request=_status_request(options=options),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.INVALID_OPTION,
                    )

            clone_error = await _policy_error(
                _policy(
                    root,
                    capabilities=("remote",),
                    allowed_commands=("clone",),
                    allowed_remote_hosts=("github.com",),
                ),
                request=_request(
                    command=ShellGitCommandName.CLONE,
                    capability=ShellGitCapability.REMOTE,
                    options={
                        "url": "https://github.com/acme/repo",
                        "destination": "repo-copy",
                        "branch": "main",
                        "revision": "../unsafe",
                    },
                ),
            )
            self.assertEqual(
                clone_error.error_code,
                ShellGitExecutionErrorCode.INVALID_OPTION,
            )

    async def test_remote_policy_failures_are_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            cases = (
                (
                    _policy(
                        root,
                        capabilities=("remote",),
                        allowed_commands=("fetch",),
                    ),
                    _request(
                        command=ShellGitCommandName.FETCH,
                        capability=ShellGitCapability.REMOTE,
                        options={
                            "remote": "origin",
                            "ref_type": "branch",
                            "ref_name": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
                ),
                (
                    _policy(
                        root,
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_hosts=("github.com",),
                    ),
                    _request(
                        command=ShellGitCommandName.CLONE,
                        capability=ShellGitCapability.REMOTE,
                        options={
                            "url": "not-a-url",
                            "destination": "repo-copy",
                            "branch": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
                ),
                (
                    _policy(
                        root,
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_hosts=("github.com",),
                    ),
                    _request(
                        command=ShellGitCommandName.CLONE,
                        capability=ShellGitCapability.REMOTE,
                        options={
                            "url": "git://github.com/acme/repo",
                            "destination": "repo-copy",
                            "branch": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
                ),
                (
                    _policy(
                        root,
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_hosts=("github.com",),
                    ),
                    _request(
                        command=ShellGitCommandName.CLONE,
                        capability=ShellGitCapability.REMOTE,
                        options={
                            "url": "ssh://github.com/acme/repo",
                            "destination": "repo-copy",
                            "branch": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
                ),
                (
                    _policy(
                        root,
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_hosts=("github.com",),
                    ),
                    _request(
                        command=ShellGitCommandName.CLONE,
                        capability=ShellGitCapability.REMOTE,
                        options={
                            "url": "https://evil.com/acme/repo",
                            "destination": "repo-copy",
                            "branch": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.REMOTE_HOST_DENIED,
                ),
                (
                    _policy(
                        root,
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_hosts=("github.com",),
                    ),
                    _request(
                        command=ShellGitCommandName.CLONE,
                        capability=ShellGitCapability.REMOTE,
                        options={
                            "url": "https://token@github.com/acme/repo",
                            "destination": "repo-copy",
                            "branch": "main",
                        },
                    ),
                    ShellGitExecutionErrorCode.CREDENTIAL_DENIED,
                ),
            )

            for policy, request, error_code in cases:
                with self.subTest(error_code=error_code):
                    error = await _policy_error(policy, request=request)
                    self.assertEqual(error.error_code, error_code)

    async def test_default_policy_denies_credentialed_clone_url(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            error = await _policy_error(
                _policy(
                    root,
                    capabilities=("remote",),
                    allowed_commands=("clone",),
                    allowed_remote_hosts=("github.com",),
                ),
                request=_request(
                    command=ShellGitCommandName.CLONE,
                    capability=ShellGitCapability.REMOTE,
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

    async def test_allow_explicit_policy_allows_credentialed_clone_url(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            policy = GitExecutionPolicy(
                settings=ShellToolSettings(
                    git=ShellGitToolSettings(
                        workspace_root=str(root),
                        cwd=".",
                        capabilities=("remote",),
                        allowed_commands=("clone",),
                        allowed_remote_protocols=("https",),
                        allowed_remote_hosts=("github.com",),
                        credential_policy="allow_explicit",
                    )
                ),
                executable_lookup=_found_git,
            )

            spec = await policy.normalize(
                _request(
                    command=ShellGitCommandName.CLONE,
                    capability=ShellGitCapability.REMOTE,
                    options={
                        "url": "https://token@github.com/acme/repo.git",
                        "destination": "repo-copy",
                        "branch": "main",
                    },
                )
            )

        self.assertIn("https://token@github.com/acme/repo.git", spec.argv)
        self.assertEqual(
            spec.metadata["git_credential_mode"],
            "allow_explicit",
        )
        self.assertEqual(spec.metadata["git_remote_protocol"], "https")
        self.assertEqual(spec.metadata["git_remote_host"], "github.com")

    async def test_unsafe_config_attributes_and_environment_are_neutralized(
        self,
    ) -> None:
        unsafe_configs = (
            "[core]\n\tpager = less\n",
            "[core]\n\tfsmonitor = .git/watch\n",
            "[core]\n\thooksPath = .git/hooks\n",
            "[core]\n\teditor = vi\n",
            "[core]\n\tworktree = /outside\n",
            "[credential]\n\thelper = store\n",
            '[filter "secret"]\n\tclean = secret-clean\n',
            "[diff]\n\texternal = secret-diff\n",
            '[diff "secret"]\n\ttextconv = secret-textconv\n',
            "[log]\n\tshowSignature = true\n",
        )
        for config in unsafe_configs:
            with self.subTest(config=config):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    repo = _write_minimal_git_repo(root / "repo")
                    (repo / ".git" / "config").write_text(config)

                    error = await _policy_error(_policy(root))

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
                )

        for attributes in (
            "*.txt filter=secret\n",
            "*.txt diff=secret\n",
            "*.txt textconv=secret\n",
        ):
            with self.subTest(attributes=attributes):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    repo = _write_minimal_git_repo(root / "repo")
                    (repo / ".gitattributes").write_text(attributes)

                    error = await _policy_error(_policy(root))

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
                )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            info_attributes = repo / ".git" / "info" / "attributes"
            info_attributes.parent.mkdir()
            info_attributes.write_text("*.txt filter=secret\n")

            error = await _policy_error(_policy(root))

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            sample_hook = repo / ".git" / "hooks" / "pre-commit.sample"
            sample_hook.parent.mkdir()
            sample_hook.write_text("#!/bin/sh\nexit 1\n")

            spec = await _policy(root).normalize(_status_request())

        self.assertEqual(spec.command, "git.status")

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            spec = await _policy(root).normalize(_status_request())

        self.assertEqual(spec.env["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(spec.env["GIT_PAGER"], "/nonexistent")
        self.assertEqual(spec.env["GIT_EDITOR"], "/nonexistent")
        self.assertEqual(spec.env["GIT_ASKPASS"], "/nonexistent")
        self.assertEqual(spec.env["SSH_ASKPASS"], "/nonexistent")
        self.assertEqual(spec.env["GIT_SSH"], "/nonexistent")
        self.assertEqual(spec.env["GIT_EXTERNAL_DIFF"], "/nonexistent")
        self.assertEqual(spec.env["HOME"], "/nonexistent")
        self.assertEqual(spec.env["GIT_OPTIONAL_LOCKS"], "0")
        self.assertNotIn("GIT_SSH_COMMAND", spec.env)

    async def test_config_include_surfaces_fail_closed(self) -> None:
        cases = (
            (
                "include",
                "[include]\n\tpath = dangerous.conf\n",
            ),
            (
                "include_if",
                '[includeIf "gitdir:repo/"]\n\tpath = dangerous.conf\n',
            ),
        )
        for name, config in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    repo = _write_minimal_git_repo(root / "repo")
                    dangerous = repo / ".git" / "dangerous.conf"
                    dangerous.write_text("[core]\n\tfsmonitor = .git/watch\n")
                    (repo / ".git" / "config").write_text(config)

                    error = await _policy_error(_policy(root))

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
                )

    async def test_status_options_are_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            option_cases: tuple[dict[str, object], ...] = (
                {"mode": "invalid", "include_branch": True},
                {"mode": "short", "include_branch": "yes"},
                {"mode": "short", "include_branch": False, "ignored": True},
                {
                    "mode": ["--git-dir=/tmp/repo"],
                    "include_branch": True,
                },
            )
            for options in option_cases:
                with self.subTest(options=options):
                    error = await _policy_error(
                        policy,
                        request=_status_request(options=options),
                    )
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.INVALID_OPTION,
                    )

            spec = await policy.normalize(
                _status_request(
                    options={
                        "mode": "short",
                        "include_branch": False,
                    },
                )
            )

        self.assertEqual(
            spec.argv,
            (
                "git",
                "--no-pager",
                "--no-optional-locks",
                "status",
                "--short",
                "--untracked-files=all",
                "--ignore-submodules=all",
            ),
        )

    def test_git_metadata_redaction_handles_collections(self) -> None:
        secret_url = "https://token@github.com/acme/private.git"
        settings = ShellGitToolSettings()

        redacted = _redacted_metadata(
            {
                "tuple_url": (secret_url,),
                "list_url": [secret_url],
            },
            settings,
        )

        self.assertEqual(
            redacted,
            {
                "tuple_url": ("https://github.com/[redacted]",),
                "list_url": ["https://github.com/[redacted]"],
            },
        )

    def test_git_text_redacts_remote_url_paths_queries_and_fragments(
        self,
    ) -> None:
        cases = (
            (
                (
                    "clone https://github.com/acme/private.git"
                    "?jwt=abc123&ref=hidden"
                ),
                "clone https://github.com/[redacted]",
            ),
            (
                "remote https://github.com?jwt=abc123",
                "remote https://github.com/[redacted]",
            ),
            (
                "remote https://github.com#private-fragment",
                "remote https://github.com/[redacted]",
            ),
            (
                (
                    "clone https://alice:hunter2@example.com/private.git"
                    "?token=abc123"
                ),
                "clone https://example.com/[redacted]",
            ),
            (
                "clone file:///Users/mariano/private.git",
                "clone file:///[redacted]",
            ),
            (
                "clone file:///Users/mariano/private.git?token=abc123#hidden",
                "clone file:///[redacted]",
            ),
            (
                (
                    "clone file://localhost/Users/mariano/private.git"
                    "?token=abc123#hidden"
                ),
                "clone file://localhost/[redacted]",
            ),
        )
        settings = ShellGitToolSettings()

        for text, expected in cases:
            with self.subTest(text=text):
                redacted = redact_git_text(text, settings)

                self.assertEqual(redacted, expected)
                self.assertNotIn("private.git", redacted)
                self.assertNotIn("jwt=abc123", redacted)
                self.assertNotIn("abc123", redacted)
                self.assertNotIn("hidden", redacted)
                self.assertNotIn("private-fragment", redacted)
                self.assertNotIn("alice", redacted)
                self.assertNotIn("hunter2", redacted)
                self.assertNotIn("/Users/mariano", redacted)
                self.assertNotIn("mariano", redacted)

    def test_git_text_redacts_author_email_fields_when_enabled(
        self,
    ) -> None:
        text = (
            "0123456789abcdef0123456789abcdef01234567\t"
            "Alice Example\talice@example.test\t"
            "2026-07-04T00:00:00+00:00\tAdd file\n"
            "Author: Alice Example <alice@example.test>\n"
        )
        settings = ShellGitToolSettings(
            redact_remote_urls=False,
            redact_credentials=False,
            redact_author_emails=True,
        )

        redacted = redact_git_text(text, settings)

        self.assertIn("Alice Example\t[redacted]\t", redacted)
        self.assertIn("Author: Alice Example <[redacted]>", redacted)
        self.assertNotIn("alice@example.test", redacted)

    def test_git_text_redacts_blame_porcelain_email_fields_when_enabled(
        self,
    ) -> None:
        text = (
            "0123456789abcdef0123456789abcdef01234567 1 1 1\n"
            "author Alice Example\n"
            "author-mail <alice@example.test>\n"
            "committer Bob Example\n"
            "committer-mail <bob@example.test>\n"
            "summary author-mail <carol@example.test>\n"
        )
        settings = ShellGitToolSettings(
            redact_remote_urls=False,
            redact_credentials=False,
            redact_author_emails=True,
        )

        redacted = redact_git_text(text, settings)

        self.assertIn("author-mail <[redacted]>", redacted)
        self.assertIn("committer-mail <[redacted]>", redacted)
        self.assertIn("summary author-mail <carol@example.test>", redacted)
        self.assertNotIn("alice@example.test", redacted)
        self.assertNotIn("bob@example.test", redacted)

    def test_git_text_keeps_author_email_fields_when_disabled(self) -> None:
        text = (
            "0123456789abcdef0123456789abcdef01234567\t"
            "Alice Example\talice@example.test\t"
            "2026-07-04T00:00:00+00:00\tAdd file\n"
            "Author: Alice Example <alice@example.test>\n"
            "author-mail <alice@example.test>\n"
            "committer-mail <bob@example.test>\n"
        )
        settings = ShellGitToolSettings(
            redact_remote_urls=False,
            redact_credentials=False,
            redact_author_emails=False,
        )

        self.assertEqual(redact_git_text(text, settings), text)

    def test_git_remote_audit_redacts_hostless_file_url_metadata(
        self,
    ) -> None:
        raw_url = "file:///Users/alice/private.git?token=abc123#hidden"
        settings = ShellGitToolSettings(
            allowed_remote_protocols=("file",),
            allowed_remote_hosts=("localhost",),
        )

        metadata = git_remote_audit_metadata(
            _request(
                command=ShellGitCommandName.CLONE,
                capability=ShellGitCapability.REMOTE,
                options={
                    "url": raw_url,
                    "destination": "repo-copy",
                    "branch": "main",
                },
            ),
            settings=settings,
        )

        self.assertEqual(metadata["git_remote_protocol"], "file")
        self.assertIsNone(metadata["git_remote_host"])
        self.assertEqual(metadata["git_remote_url"], "file:///[redacted]")
        self.assertEqual(metadata["git_remote_urls"], ("file:///[redacted]",))
        self.assertNotIn(raw_url, str(metadata))
        self.assertNotIn("/Users/alice", str(metadata))
        self.assertNotIn("private.git", str(metadata))
        self.assertNotIn("token=abc123", str(metadata))
        self.assertNotIn("abc123", str(metadata))
        self.assertNotIn("hidden", str(metadata))

    def test_policy_denied_result_redacts_hostless_file_url_metadata(
        self,
    ) -> None:
        raw_url = "file:///Users/alice/private.git?token=abc123#hidden"
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                cwd=".",
                capabilities=("remote",),
                allowed_commands=("clone",),
                allowed_remote_protocols=("file",),
                allowed_remote_hosts=("localhost",),
            ),
        )
        request = _request(
            command=ShellGitCommandName.CLONE,
            capability=ShellGitCapability.REMOTE,
            options={
                "url": raw_url,
                "destination": "repo-copy",
                "branch": "main",
            },
        )

        result = _git_policy_denied_result(
            request,
            ShellGitPolicyDenied(
                ShellGitExecutionErrorCode.REMOTE_PROTOCOL_DENIED,
                "remote URL protocol is unsupported",
            ),
            settings=settings,
        )
        metadata = result.audit_metadata

        self.assertEqual(result.status, ShellGitExecutionStatus.POLICY_DENIED)
        self.assertEqual(
            metadata["git_request_options"],
            {
                "url": "file:///[redacted]",
                "destination": "repo-copy",
                "branch": "main",
            },
        )
        self.assertEqual(
            metadata["request_options"],
            metadata["git_request_options"],
        )
        self.assertEqual(metadata["git_remote_protocol"], "file")
        self.assertIsNone(metadata["git_remote_host"])
        self.assertEqual(metadata["git_remote_url"], "file:///[redacted]")
        self.assertEqual(metadata["git_remote_urls"], ("file:///[redacted]",))
        self.assertNotIn(raw_url, str(metadata))
        self.assertNotIn("/Users/alice", str(metadata))
        self.assertNotIn("private.git", str(metadata))
        self.assertNotIn("token=abc123", str(metadata))
        self.assertNotIn("abc123", str(metadata))
        self.assertNotIn("hidden", str(metadata))

    async def test_boundaries_are_enforced(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            settings = _settings(
                root,
                default_timeout_seconds=5.0,
                max_timeout_seconds=7.0,
                max_stdout_bytes=11,
                max_stderr_bytes=13,
            )
            spec = await GitExecutionPolicy(
                settings=settings,
                executable_lookup=_found_git,
            ).normalize(
                _status_request(
                    timeout_seconds=999.0,
                    max_stdout_bytes=99,
                    max_stderr_bytes=99,
                )
            )

            self.assertEqual(spec.timeout_seconds, 7.0)
            self.assertEqual(spec.max_stdout_bytes, 11)
            self.assertEqual(spec.max_stderr_bytes, 13)

            cases = (
                (
                    _settings(root, max_arguments=2),
                    _status_request(),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _settings(root, max_argument_bytes=3),
                    _status_request(),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _settings(root, max_command_bytes=5),
                    _status_request(),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _settings(root, git_max_pathspecs=1),
                    _status_request(pathspecs=("a", "b")),
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                ),
                (
                    _settings(root, git_max_pathspec_bytes=3),
                    _status_request(pathspecs=("abcd",)),
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                ),
            )
            for settings, request, error_code in cases:
                with self.subTest(error_code=error_code, settings=settings):
                    error = await _policy_error(
                        GitExecutionPolicy(
                            settings=settings,
                            executable_lookup=_found_git,
                        ),
                        request=request,
                    )
                    self.assertEqual(error.error_code, error_code)


class GitWrapperResultTest(IsolatedAsyncioTestCase):
    async def test_missing_git_returns_stable_formatted_result(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            tool = _status_tool(root, executor=_UnavailableExecutor())

            result = await tool(context=ToolCallContext())

        self.assertIsInstance(result, ShellGitFormattedResult)
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

    async def test_timeout_cancellation_nonzero_and_truncation_are_stable(
        self,
    ) -> None:
        cases = (
            (
                ShellExecutionStatus.TIMEOUT,
                ShellGitExecutionStatus.TIMEOUT,
                ShellGitExecutionErrorCode.TIMEOUT,
                True,
                False,
                False,
                None,
            ),
            (
                ShellExecutionStatus.CANCELLED,
                ShellGitExecutionStatus.CANCELLED,
                None,
                False,
                True,
                False,
                None,
            ),
            (
                ShellExecutionStatus.TOO_LARGE,
                ShellGitExecutionStatus.FAILED,
                ShellGitExecutionErrorCode.OUTPUT_TRUNCATED,
                False,
                False,
                False,
                None,
            ),
            (
                ShellExecutionStatus.POLICY_DENIED,
                ShellGitExecutionStatus.POLICY_DENIED,
                ShellGitExecutionErrorCode.COMMAND_DISABLED,
                False,
                False,
                False,
                None,
            ),
            (
                ShellExecutionStatus.SPAWN_FAILED,
                ShellGitExecutionStatus.COMMAND_UNAVAILABLE,
                ShellGitExecutionErrorCode.COMMAND_UNAVAILABLE,
                False,
                False,
                False,
                None,
            ),
            (
                ShellExecutionStatus.NONZERO_EXIT,
                ShellGitExecutionStatus.FAILED,
                ShellGitExecutionErrorCode.NONZERO_EXIT,
                False,
                False,
                False,
                "https://user:pass@example.com/repo failed",
            ),
            (
                ShellExecutionStatus.COMPLETED,
                ShellGitExecutionStatus.FAILED,
                ShellGitExecutionErrorCode.OUTPUT_TRUNCATED,
                False,
                False,
                True,
                None,
            ),
        )
        for (
            shell_status,
            git_status,
            git_error,
            timed_out,
            cancelled,
            truncated,
            error_message,
        ) in cases:
            with self.subTest(shell_status=shell_status):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    tool = _status_tool(
                        root,
                        executor=_ResultExecutor(
                            shell_status,
                            timed_out=timed_out,
                            cancelled=cancelled,
                            truncated=truncated,
                            error_message=error_message,
                        ),
                        found=True,
                    )

                    result = await tool(context=ToolCallContext())

                self.assertIsInstance(result, ShellGitFormattedResult)
                assert isinstance(result, ShellGitFormattedResult)
                self.assertEqual(result.git_result.status, git_status)
                self.assertEqual(result.git_result.error_code, git_error)
                self.assertIn(f"status: {git_status.value}", result)
                self.assertNotIn("https://user:pass@example.com/repo", result)

        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            tool = _status_tool(
                root,
                executor=_CancellingExecutor(),
                found=True,
            )

            result = await tool(context=ToolCallContext())

        self.assertIsInstance(result, ShellGitFormattedResult)
        assert isinstance(result, ShellGitFormattedResult)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.CANCELLED,
        )
        self.assertIsNone(result.git_result.error_code)
        self.assertIn("cancelled: true", result)

    async def test_unexpected_executor_errors_propagate(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            tool = _status_tool(
                root,
                executor=_UnexpectedExecutor(),
                found=True,
            )

            with self.assertRaisesRegex(
                AssertionError,
                "executor should not be called",
            ):
                await tool(context=ToolCallContext())

    async def test_policy_denial_returns_stable_formatted_result(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            tool = _status_tool(
                root,
                executor=_UnexpectedExecutor(),
                found=True,
            )

            result = await tool(
                paths=("../outside",),
                context=ToolCallContext(),
            )

        self.assertIsInstance(result, ShellGitFormattedResult)
        assert isinstance(result, ShellGitFormattedResult)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertIn("execution_mode: policy", result)

    async def test_post_index_change_hook_denied_with_optional_locks_disabled(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            marker = root / "hook-ran"
            hook = repo / ".git" / "hooks" / "post-index-change"
            hook.parent.mkdir()
            hook.write_text(f"#!/bin/sh\nprintf ran > {marker}\n")
            hook.chmod(0o755)
            tool = _status_tool(
                root,
                executor=_UnexpectedExecutor(),
                found=True,
            )

            result = await tool(context=ToolCallContext())

        self.assertIsInstance(result, ShellGitFormattedResult)
        assert isinstance(result, ShellGitFormattedResult)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )
        self.assertFalse(marker.exists())

    async def test_non_directory_hooks_path_is_policy_denied(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "hooks").write_text("not a directory\n")
            tool = _status_tool(
                root,
                executor=_UnexpectedExecutor(),
                found=True,
            )

            result = await tool(context=ToolCallContext())

        self.assertIsInstance(result, ShellGitFormattedResult)
        assert isinstance(result, ShellGitFormattedResult)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )
        self.assertIn("execution_mode: policy", result)


class GitRuntimeGuardrailTest(TestCase):
    def test_git_modules_do_not_use_subprocess_or_shell_evaluation(
        self,
    ) -> None:
        source_root = (
            Path(__file__).parents[3] / "src" / "avalan" / "tool" / "shell"
        )
        paths = (
            source_root / "git.py",
            source_root / "git_policy.py",
        )
        for path in paths:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                tree = parse(source)
                self.assertNotIn("subprocess", source)
                self.assertNotIn("create_subprocess_shell", source)
                self.assertNotIn("shell=True", source)
                self.assertNotIn("/bin/sh", source)
                self.assertNotIn("bash -c", source)
                self.assertNotIn("sh -c", source)
                self.assertFalse(_imports_subprocess(tree))
                self.assertFalse(_calls_shell_spawn(tree))


def _policy(
    workspace_root: Path,
    *,
    cwd: str = "repo",
    capabilities: tuple[str, ...] = ("read",),
    allowed_commands: tuple[str, ...] = ("status",),
    allow_alternates: bool = False,
    allow_linked_worktrees: bool = False,
    allow_optional_locks: bool = False,
    allow_submodule_update: bool = False,
    allowed_remote_protocols: tuple[str, ...] = ("https",),
    allowed_remote_hosts: tuple[str, ...] = (),
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=_settings(
            workspace_root,
            cwd=cwd,
            git_capabilities=capabilities,
            git_allowed_commands=allowed_commands,
            allow_alternates=allow_alternates,
            allow_linked_worktrees=allow_linked_worktrees,
            allow_optional_locks=allow_optional_locks,
            allow_submodule_update=allow_submodule_update,
            allowed_remote_protocols=allowed_remote_protocols,
            allowed_remote_hosts=allowed_remote_hosts,
        ),
        executable_lookup=_found_git,
    )


def _settings(
    workspace_root: Path,
    *,
    cwd: str = "repo",
    git_capabilities: tuple[str, ...] = ("read",),
    git_allowed_commands: tuple[str, ...] = ("status",),
    git_max_pathspecs: int = 64,
    git_max_pathspec_bytes: int = 4096,
    allow_alternates: bool = False,
    allow_linked_worktrees: bool = False,
    allow_optional_locks: bool = False,
    allow_submodule_update: bool = False,
    allowed_remote_protocols: tuple[str, ...] = ("https",),
    allowed_remote_hosts: tuple[str, ...] = (),
    default_timeout_seconds: float = 10.0,
    max_timeout_seconds: float = 60.0,
    max_stdout_bytes: int = 65536,
    max_stderr_bytes: int = 32768,
    max_arguments: int = 128,
    max_argument_bytes: int = 8192,
    max_command_bytes: int = 32768,
    executable_search_paths: tuple[str, ...] = (),
    git_executable_path: str | None = None,
) -> ShellToolSettings:
    return ShellToolSettings(
        max_arguments=max_arguments,
        max_argument_bytes=max_argument_bytes,
        max_command_bytes=max_command_bytes,
        executable_search_paths=executable_search_paths,
        git=ShellGitToolSettings(
            workspace_root=str(workspace_root),
            cwd=cwd,
            capabilities=git_capabilities,
            allowed_commands=git_allowed_commands,
            executable_path=git_executable_path,
            default_timeout_seconds=default_timeout_seconds,
            max_timeout_seconds=max_timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            max_pathspecs=git_max_pathspecs,
            max_pathspec_bytes=git_max_pathspec_bytes,
            allow_alternates=allow_alternates,
            allow_linked_worktrees=allow_linked_worktrees,
            allow_optional_locks=allow_optional_locks,
            allow_submodule_update=allow_submodule_update,
            allowed_remote_protocols=allowed_remote_protocols,
            allowed_remote_hosts=allowed_remote_hosts,
        ),
    )


def _request(
    *,
    command: ShellGitCommandName = ShellGitCommandName.STATUS,
    capability: ShellGitCapability = ShellGitCapability.READ,
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
        capability_required=capability,
        options={} if options is None else options,
        pathspecs=pathspecs,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _status_request(
    *,
    options: dict[str, object] | None = None,
    pathspecs: tuple[str, ...] = (),
    cwd: str | None = None,
    timeout_seconds: float | None = None,
    max_stdout_bytes: int | None = None,
    max_stderr_bytes: int | None = None,
) -> ShellGitCommandRequest:
    return _request(
        options=(
            {"mode": "porcelain_v2", "include_branch": True}
            if options is None
            else options
        ),
        pathspecs=pathspecs,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


async def _policy_error(
    policy: GitExecutionPolicy,
    *,
    request: ShellGitCommandRequest | None = None,
) -> ShellGitPolicyDenied:
    try:
        active_request = _status_request() if request is None else request
        await policy.normalize(active_request)
    except ShellGitPolicyDenied as error:
        return error
    raise AssertionError("Git policy should have denied the request")


async def _found_git(search_paths: tuple[str, ...]) -> str | None:
    return "/trusted/bin/git"


async def _missing_git(search_paths: tuple[str, ...]) -> str | None:
    return None


async def _fail_on_index_read(path: Path) -> bytes:
    if path.name == "index":
        raise AssertionError("read-only Git commands must not read index")
    return await _git_policy_read_bytes(path)


def _write_minimal_git_repo(repo: Path) -> Path:
    git_dir = repo / ".git"
    (git_dir / "objects" / "info").mkdir(parents=True)
    (git_dir / "refs").mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    )
    return repo


def _git_index_v4_data() -> bytes:
    body = b"DIRC" + (4).to_bytes(4, "big") + (0).to_bytes(4, "big")
    return body + sha1(body).digest()


def _status_tool(
    workspace_root: Path,
    *,
    executor: Any,
    found: bool = False,
    allow_optional_locks: bool = False,
) -> GitStatusTool:
    settings = _settings(
        workspace_root,
        allow_optional_locks=allow_optional_locks,
    )
    lookup = _found_git if found else _missing_git
    return GitStatusTool(settings=settings).bind_execution(
        git_policy=GitExecutionPolicy(
            settings=settings,
            executable_lookup=lookup,
        ),
        executor=executor,
    )


class _ResultExecutor:
    def __init__(
        self,
        status: ShellExecutionStatus,
        *,
        timed_out: bool = False,
        cancelled: bool = False,
        truncated: bool = False,
        error_message: str | None = None,
    ) -> None:
        self._status = status
        self._timed_out = timed_out
        self._cancelled = cancelled
        self._truncated = truncated
        self._error_message = error_message

    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        stdout = "https://user:pass@example.com/repo\n"
        return ExecutionResult(
            backend=spec.backend,
            tool_name=spec.tool_name,
            command=spec.command,
            argv=spec.argv,
            display_argv=spec.display_argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=self._status,
            exit_code=(
                0 if self._status is ShellExecutionStatus.COMPLETED else 1
            ),
            stdout=stdout,
            stderr="",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            stdout_bytes=len(stdout.encode("utf-8")),
            stderr_bytes=0,
            stdout_truncated=self._truncated,
            stderr_truncated=False,
            timed_out=self._timed_out,
            cancelled=self._cancelled,
            duration_ms=12,
            error_message=self._error_message,
            metadata=spec.metadata,
        )


class _CancellingExecutor:
    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        raise CancelledError


class _UnexpectedExecutor:
    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        raise AssertionError("executor should not be called")


class _UnavailableExecutor:
    async def execute(self, spec: ExecutionSpec) -> ExecutionResult:
        return ExecutionResult(
            backend=spec.backend,
            tool_name=spec.tool_name,
            command=spec.command,
            argv=spec.argv,
            display_argv=spec.display_argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=ShellExecutionStatus.COMMAND_UNAVAILABLE,
            exit_code=None,
            stdout="",
            stderr="",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            stdout_bytes=0,
            stderr_bytes=0,
            metadata=spec.metadata,
        )


def _imports_subprocess(tree: AST) -> bool:
    for node in walk(tree):
        if isinstance(node, Import):
            if any(alias.name == "subprocess" for alias in node.names):
                return True
        if isinstance(node, ImportFrom) and node.module == "subprocess":
            return True
    return False


def _calls_shell_spawn(tree: AST) -> bool:
    return any(
        isinstance(node, Call)
        and getattr(node.func, "attr", getattr(node.func, "id", ""))
        == "create_subprocess_shell"
        for node in walk(tree)
    )


if __name__ == "__main__":
    main()
