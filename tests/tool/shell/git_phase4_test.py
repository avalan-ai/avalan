from collections.abc import Awaitable, Callable
from os import devnull, environ
from pathlib import Path
from shutil import which
from subprocess import CalledProcessError, run
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
    SHELL_GIT_COMMAND_CAPABILITIES,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitExecutionErrorCode,
    ShellGitExecutionStatus,
    ShellGitPolicyDenied,
)
from avalan.tool.shell.git_policy import (
    GitExecutionPolicy,
    _git_index_data_has_exact_file_pathspec,
)
from avalan.tool.shell.settings import ShellGitToolSettings, ShellToolSettings
from avalan.tool.shell.toolset import ShellToolSet

_GIT_BINARY = which("git")
_PHASE4_COMMANDS = (
    "diff",
    "show",
    "blame",
    "grep",
    "stash-list",
    "stash-show",
)
_PHASE4_TOOL_NAMES = [
    "shell.git_diff",
    "shell.git_show",
    "shell.git_blame",
    "shell.git_grep",
    "shell.git_stash_list",
    "shell.git_stash_show",
]
_GitToolCallable = Callable[..., Awaitable[str]]


class GitContentPolicyPhase4Test(IsolatedAsyncioTestCase):
    async def test_content_commands_build_fixed_argv(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={
                            "mode": "range",
                            "base_revision": "HEAD~1",
                            "head_revision": "HEAD",
                        },
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "diff",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-color",
                        "--no-renames",
                        "--patch",
                        "HEAD~1^{commit}",
                        "HEAD^{commit}",
                        "--",
                        "src/app.py",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.SHOW,
                        options={"revision": "HEAD", "mode": "patch"},
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "show",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-color",
                        "--no-decorate",
                        "--no-show-signature",
                        "--no-renames",
                        "--date=iso-strict",
                        "--format=%H%x09%an%x09%ae%x09%ad%x09%s",
                        "--patch",
                        "HEAD^{commit}",
                        "--",
                        "src/app.py",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.BLAME,
                        options={"start_line": 1, "end_line": 2},
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "blame",
                        "--no-textconv",
                        "--date=iso-strict",
                        "--line-porcelain",
                        "--no-progress",
                        "--no-ignore-revs-file",
                        "--no-color-lines",
                        "--no-color-by-age",
                        "-L",
                        "1,2",
                        "--",
                        "src/app.py",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.GREP,
                        options={
                            "pattern": "needle",
                            "case": "insensitive",
                            "max_matches": 3,
                        },
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "grep",
                        "--index",
                        "--no-recurse-submodules",
                        "--no-textconv",
                        "--fixed-strings",
                        "--line-number",
                        "--full-name",
                        "--no-color",
                        "--max-count=3",
                        "--ignore-case",
                        "-e",
                        "needle",
                        "--",
                        "src/app.py",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_LIST,
                        options={"max_count": 2},
                    ),
                    (
                        *_git_prefix(),
                        "stash",
                        "list",
                        "--format=%gd%x09%gs",
                        "--max-count=2",
                    ),
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_SHOW,
                        options={"stash": "stash@{1}", "mode": "patch"},
                        pathspecs=("src/app.py",),
                    ),
                    (
                        *_git_prefix(),
                        "diff",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-color",
                        "--no-renames",
                        "--patch",
                        "stash@{1}^1",
                        "stash@{1}",
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
                    self.assertEqual(spec.stdout_media_type, "text/plain")
                    self.assertEqual(
                        spec.metadata["git_display_argv"], expected_argv
                    )

    async def test_diff_range_operands_are_forced_to_commits(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            spec = await _policy(root).normalize(
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={
                        "mode": "range",
                        "base_revision": "README.md",
                        "head_revision": "HEAD",
                    },
                    pathspecs=("src/app.py",),
                )
            )

        self.assertIn("README.md^{commit}", spec.argv)
        self.assertIn("HEAD^{commit}", spec.argv)
        self.assertNotIn("README.md", spec.argv)
        self.assertEqual(
            spec.argv[-4:],
            ("README.md^{commit}", "HEAD^{commit}", "--", "src/app.py"),
        )

    async def test_content_modes_build_conservative_argv(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "worktree"},
                        pathspecs=("src/app.py",),
                    ),
                    ("diff", "--patch"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "staged"},
                        pathspecs=("src/app.py",),
                    ),
                    ("diff", "--cached", "--patch"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "stat"},
                        pathspecs=("src/app.py",),
                    ),
                    ("diff", "--stat"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "name_only"},
                        pathspecs=("src/app.py",),
                    ),
                    ("diff", "--name-only"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.SHOW,
                        options={"revision": "HEAD", "mode": "summary"},
                    ),
                    ("show", "--no-patch"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.SHOW,
                        options={"revision": "HEAD", "mode": "stat"},
                        pathspecs=("src/app.py",),
                    ),
                    ("show", "--stat"),
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_SHOW,
                        options={"stash": "stash@{0}", "mode": "stat"},
                        pathspecs=("src/app.py",),
                    ),
                    ("diff", "--stat"),
                ),
            )

            for request, expected_markers in cases:
                with self.subTest(
                    command=request.command.value,
                    options=request.options,
                ):
                    spec = await policy.normalize(request)
                    for marker in expected_markers:
                        self.assertIn(marker, spec.argv)
                    self.assertNotIn("--ext-diff", spec.argv)
                    self.assertNotIn("--textconv", spec.argv)
                    self.assertNotIn("--color", spec.argv)

    async def test_git_blame_allows_explicit_existing_file_scope(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")

            spec = await _policy(root).normalize(
                _request(
                    command=ShellGitCommandName.BLAME,
                    pathspecs=("src/app.py",),
                )
            )

        self.assertIn("blame", spec.argv)
        self.assertEqual(spec.argv[-2:], ("--", "src/app.py"))

    async def test_content_unsafe_forms_fail_closed(self) -> None:
        cases = (
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree", "external_diff": True},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree", "textconv": True},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "binary"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree", "binary": True},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree", "output": "diff.patch"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={
                        "mode": "worktree",
                        "base_revision": "HEAD",
                    },
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree", "no_index": True},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "range", "base_revision": "HEAD"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.BLAME,
                    pathspecs=(),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.BLAME,
                    pathspecs=("README.md", "src/app.py"),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.BLAME,
                    options={"start_line": 0},
                    pathspecs=("README.md",),
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"revision": "HEAD", "format": "%H"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"mode": "summary"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"revision": "HEAD", "object": "blob"},
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"revision": "HEAD:README.md"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"revision": "HEAD@{1}"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={
                        "mode": "range",
                        "base_revision": "HEAD@{1}",
                        "head_revision": "HEAD",
                    },
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={
                        "mode": "range",
                        "base_revision": "src/app.py",
                        "head_revision": "HEAD",
                    },
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree"},
                    pathspecs=(":(top)README.md",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree"},
                    pathspecs=("--output=diff.patch",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"revision": "https://github.com/acme/repo"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.GREP,
                    options={
                        "pattern": "needle",
                        "open_files_in_pager": True,
                    },
                ),
                ShellGitExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_SHOW,
                    options={"stash": "stash@{999}", "mode": "patch"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_SHOW,
                    options={"stash": "HEAD@{1}", "mode": "patch"},
                ),
                ShellGitExecutionErrorCode.REVISION_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.STASH_LIST,
                    options={"max_count": 1},
                    pathspecs=("README.md",),
                ),
                ShellGitExecutionErrorCode.PATHSPEC_DENIED,
            ),
            (
                _request(
                    command=ShellGitCommandName.FETCH,
                    options={"url": "https://github.com/acme/repo.git"},
                ),
                ShellGitExecutionErrorCode.COMMAND_DISABLED,
            ),
            (
                _request_with_unsafe_option_key(),
                ShellGitExecutionErrorCode.INVALID_OPTION,
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

    async def test_repo_wide_content_commands_require_pathspecs(
        self,
    ) -> None:
        cases = (
            _request(
                command=ShellGitCommandName.DIFF,
                options={"mode": "worktree"},
            ),
            _request(
                command=ShellGitCommandName.DIFF,
                options={
                    "mode": "range",
                    "base_revision": "HEAD~1",
                    "head_revision": "HEAD",
                },
            ),
            _request(
                command=ShellGitCommandName.GREP,
                options={"pattern": "needle"},
            ),
            _request(
                command=ShellGitCommandName.BLAME,
            ),
            _request(
                command=ShellGitCommandName.SHOW,
                options={"revision": "HEAD", "mode": "patch"},
            ),
            _request(
                command=ShellGitCommandName.STASH_SHOW,
                options={"stash": "stash@{0}", "mode": "patch"},
            ),
        )

        for request in cases:
            with self.subTest(
                command=request.command.value,
                options=request.options,
            ):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    repo = _write_minimal_git_repo(root / "repo")
                    (repo / ".env").write_text("TOKEN=secret\n")
                    (repo / "credentials").write_text("secret\n")

                    error = await _policy_error(_policy(root), request)

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                )

    async def test_summary_content_modes_allow_optional_directory_scopes(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "stat"},
                    ),
                    ("diff", "--stat"),
                    False,
                ),
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "name_only"},
                    ),
                    ("diff", "--name-only"),
                    False,
                ),
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "stat"},
                        pathspecs=("src",),
                    ),
                    ("diff", "--stat"),
                    True,
                ),
                (
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "name_only"},
                        pathspecs=("src",),
                    ),
                    ("diff", "--name-only"),
                    True,
                ),
                (
                    _request(
                        command=ShellGitCommandName.SHOW,
                        options={"revision": "HEAD", "mode": "stat"},
                    ),
                    ("show", "--stat"),
                    False,
                ),
                (
                    _request(
                        command=ShellGitCommandName.SHOW,
                        options={"revision": "HEAD", "mode": "stat"},
                        pathspecs=("src",),
                    ),
                    ("show", "--stat"),
                    True,
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_SHOW,
                        options={"stash": "stash@{0}", "mode": "stat"},
                    ),
                    ("diff", "--stat"),
                    False,
                ),
                (
                    _request(
                        command=ShellGitCommandName.STASH_SHOW,
                        options={"stash": "stash@{0}", "mode": "stat"},
                        pathspecs=("src",),
                    ),
                    ("diff", "--stat"),
                    True,
                ),
            )

            for request, expected_markers, includes_pathspec in cases:
                with self.subTest(
                    command=request.command.value,
                    options=request.options,
                    pathspecs=request.pathspecs,
                ):
                    spec = await policy.normalize(request)

                    for marker in expected_markers:
                        self.assertIn(marker, spec.argv)
                    if includes_pathspec:
                        self.assertEqual(spec.argv[-2:], ("--", "src"))
                    else:
                        self.assertNotIn("--", spec.argv)

    async def test_content_pathspecs_reject_hidden_and_sensitive_paths(
        self,
    ) -> None:
        cases = (
            _request(
                command=ShellGitCommandName.DIFF,
                options={"mode": "worktree"},
                pathspecs=(".hidden.txt",),
            ),
            _request(
                command=ShellGitCommandName.DIFF,
                options={"mode": "worktree"},
                pathspecs=(".env",),
            ),
            _request(
                command=ShellGitCommandName.GREP,
                options={"pattern": "needle"},
                pathspecs=("src/.hidden.txt",),
            ),
            _request(
                command=ShellGitCommandName.GREP,
                options={"pattern": "needle"},
                pathspecs=("credentials",),
            ),
            _request(
                command=ShellGitCommandName.BLAME,
                pathspecs=(".hidden.txt",),
            ),
            _request(
                command=ShellGitCommandName.BLAME,
                pathspecs=(".env",),
            ),
            _request(
                command=ShellGitCommandName.SHOW,
                options={"revision": "HEAD", "mode": "patch"},
                pathspecs=(".hidden.txt",),
            ),
            _request(
                command=ShellGitCommandName.SHOW,
                options={"revision": "HEAD", "mode": "patch"},
                pathspecs=(".env",),
            ),
            _request(
                command=ShellGitCommandName.STASH_SHOW,
                options={"stash": "stash@{0}", "mode": "patch"},
                pathspecs=(".hidden.txt",),
            ),
            _request(
                command=ShellGitCommandName.STASH_SHOW,
                options={"stash": "stash@{0}", "mode": "patch"},
                pathspecs=("credentials",),
            ),
        )

        for request in cases:
            with self.subTest(
                command=request.command.value,
                pathspecs=request.pathspecs,
            ):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    error = await _policy_error(_policy(root), request)
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

    async def test_content_pathspecs_reject_directory_scopes(self) -> None:
        cases = (
            _request(
                command=ShellGitCommandName.DIFF,
                options={"mode": "worktree"},
                pathspecs=(".",),
            ),
            _request(
                command=ShellGitCommandName.DIFF,
                options={"mode": "worktree"},
                pathspecs=("src",),
            ),
            _request(
                command=ShellGitCommandName.GREP,
                options={"pattern": "needle"},
                pathspecs=(".",),
            ),
            _request(
                command=ShellGitCommandName.GREP,
                options={"pattern": "needle"},
                pathspecs=("src",),
            ),
            _request(
                command=ShellGitCommandName.BLAME,
                pathspecs=(".",),
            ),
            _request(
                command=ShellGitCommandName.BLAME,
                pathspecs=("src",),
            ),
            _request(
                command=ShellGitCommandName.SHOW,
                options={"revision": "HEAD", "mode": "patch"},
                pathspecs=(".",),
            ),
            _request(
                command=ShellGitCommandName.SHOW,
                options={"revision": "HEAD", "mode": "patch"},
                pathspecs=("src",),
            ),
            _request(
                command=ShellGitCommandName.STASH_SHOW,
                options={"stash": "stash@{0}", "mode": "patch"},
                pathspecs=(".",),
            ),
            _request(
                command=ShellGitCommandName.STASH_SHOW,
                options={"stash": "stash@{0}", "mode": "patch"},
                pathspecs=("src",),
            ),
        )

        for request in cases:
            with self.subTest(command=request.command.value):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    repo = _write_minimal_git_repo(root / "repo")
                    (repo / "src" / ".env").write_text("TOKEN=secret\n")

                    error = await _policy_error(_policy(root), request)

                self.assertEqual(
                    error.error_code,
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                )

    async def test_content_pathspecs_reject_missing_file_scopes(self) -> None:
        cases = (
            _request(
                command=ShellGitCommandName.DIFF,
                options={"mode": "worktree"},
                pathspecs=("src/missing.py",),
            ),
            _request(
                command=ShellGitCommandName.DIFF,
                options={"mode": "staged"},
                pathspecs=("src/missing.py",),
            ),
            _request(
                command=ShellGitCommandName.GREP,
                options={"pattern": "needle"},
                pathspecs=("src/missing.py",),
            ),
            _request(
                command=ShellGitCommandName.BLAME,
                pathspecs=("src/missing.py",),
            ),
            _request(
                command=ShellGitCommandName.SHOW,
                options={"revision": "HEAD", "mode": "patch"},
                pathspecs=("src/missing.py",),
            ),
            _request(
                command=ShellGitCommandName.STASH_SHOW,
                options={"stash": "stash@{0}", "mode": "patch"},
                pathspecs=("src/missing.py",),
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
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                )

    async def test_patch_modes_deny_unprovable_missing_file_pathspecs(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            policy = _policy(root)

            cases = (
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree"},
                    pathspecs=("src/missing.py",),
                ),
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "staged"},
                    pathspecs=("src/missing.py",),
                ),
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={
                        "mode": "range",
                        "base_revision": "HEAD~1",
                        "head_revision": "HEAD",
                    },
                    pathspecs=("src/missing.py",),
                ),
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"revision": "HEAD", "mode": "patch"},
                    pathspecs=("src/missing.py",),
                ),
                _request(
                    command=ShellGitCommandName.STASH_SHOW,
                    options={"stash": "stash@{0}", "mode": "patch"},
                    pathspecs=("src/missing.py",),
                ),
            )

            for request in cases:
                with self.subTest(
                    command=request.command.value,
                    options=request.options,
                ):
                    error = await _policy_error(policy, request)
                    self.assertEqual(
                        error.error_code,
                        ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                    )

    def test_git_index_data_file_pathspec_matching(self) -> None:
        valid = _git_index_data(_git_index_entry("src/app.py"))
        exact_padding = _git_index_data(_git_index_entry("a"))
        symlink = _git_index_data(_git_index_entry("src/link", mode=0o120000))
        extended = _git_index_data(
            _git_index_entry("src/app.py", extended=True)
        )
        gitlink = _git_index_data(
            _git_index_entry("src/app.py", mode=0o160000)
        )
        split = _git_index_data(
            _git_index_entry("src/app.py"),
            checksum=b"\x00" * 20,
            extensions=((b"link", b"\x00" * 20),),
        )
        sparse = _git_index_data(
            _git_index_entry("src/app.py"),
            checksum=b"\x00" * 20,
            extensions=((b"sdir", b""),),
        )
        duplicate_link = _git_index_data(
            _git_index_entry("src/app.py"),
            checksum=b"\x00" * 20,
            extensions=((b"link", b""), (b"link", b"")),
        )
        unknown_mandatory = _git_index_data(
            _git_index_entry("src/app.py"),
            checksum=b"\x00" * 20,
            extensions=((b"abcd", b""),),
        )
        bad_extension_header = (
            _git_index_data(_git_index_entry("src/app.py"))
            + b"ABCD"
            + b"\x00" * 20
        )
        bad_extension_size = (
            _git_index_data(_git_index_entry("src/app.py"))
            + b"ABCD"
            + (1).to_bytes(4, "big")
            + b"\x00" * 20
        )
        bad_extension_end = (
            _git_index_data(_git_index_entry("src/app.py")) + b"\x00"
        )
        unsupported = b"DIRC" + (4).to_bytes(4, "big") + (0).to_bytes(4, "big")
        truncated = _git_index_data(b"\x00" * 10)
        version2_extended = _git_index_data(
            _git_index_entry("src/app.py", extended=True),
            version=2,
        )
        truncated_extended = _git_index_data(
            _git_index_truncated_extended_entry()
        )
        unterminated = _git_index_data(
            _git_index_entry("src/app.py", nul=False, pad=False)
        )
        unpadded = _git_index_data(_git_index_entry("src/app.py", pad=False))

        self.assertTrue(
            _git_index_data_has_exact_file_pathspec(valid, "src/app.py")
        )
        self.assertTrue(
            _git_index_data_has_exact_file_pathspec(exact_padding, "a")
        )
        self.assertTrue(
            _git_index_data_has_exact_file_pathspec(symlink, "src/link")
        )
        self.assertTrue(
            _git_index_data_has_exact_file_pathspec(extended, "src/app.py")
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(b"", "src/app.py")
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(unsupported, "src/app.py")
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(truncated, "src/app.py")
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                truncated_extended,
                "src/app.py",
            )
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                unterminated,
                "src/app.py",
            )
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(unpadded, "src/app.py")
        )
        self.assertFalse(_git_index_data_has_exact_file_pathspec(valid, "src"))
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(gitlink, "src/app.py")
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(split, "src/app.py")
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(sparse, "src/app.py")
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                duplicate_link,
                "src/app.py",
            )
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                unknown_mandatory,
                "src/app.py",
            )
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                bad_extension_header,
                "src/app.py",
            )
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                bad_extension_size,
                "src/app.py",
            )
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                bad_extension_end,
                "src/app.py",
            )
        )
        self.assertFalse(
            _git_index_data_has_exact_file_pathspec(
                version2_extended,
                "src/app.py",
            )
        )

    async def test_gitattributes_filters_are_denied(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".gitattributes").write_text("*.txt filter=secret\n")

            error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.GREP,
                    options={"pattern": "needle"},
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.EXTERNAL_PROCESS_DENIED,
        )

    async def test_content_denies_unsafe_repository_index_path(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "index").mkdir()

            error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree"},
                    pathspecs=("src/app.py",),
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        )
        self.assertEqual(str(error), "Git repository index is unsafe")

    async def test_content_denies_malformed_repository_index(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "index").write_bytes(b"not an index")

            error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree"},
                    pathspecs=("src/app.py",),
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        )
        self.assertEqual(
            str(error),
            "Git repository index format is unsupported",
        )

    async def test_git_show_rejects_log_show_signature_config(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").write_text(
                "[core]\n"
                "\trepositoryformatversion = 0\n"
                "\tbare = false\n"
                "[log]\n"
                "\tshowSignature = true\n"
            )

            error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.SHOW,
                    options={"revision": "HEAD"},
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        )

    async def test_git_blame_rejects_ignore_revs_file_config(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_minimal_git_repo(root / "repo")
            (repo / ".git" / "config").write_text(
                "[core]\n"
                "\trepositoryformatversion = 0\n"
                "\tbare = false\n"
                "[blame]\n"
                "\tignoreRevsFile = /etc/passwd\n"
            )

            error = await _policy_error(
                _policy(root),
                _request(
                    command=ShellGitCommandName.BLAME,
                    pathspecs=("src/app.py",),
                ),
            )

        self.assertEqual(
            error.error_code,
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        )

    async def test_content_boundaries_are_enforced(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            spec = await _policy(
                root,
                default_timeout_seconds=5,
                max_timeout_seconds=7,
                max_stdout_bytes=4096,
                max_stderr_bytes=13,
                max_diff_bytes=17,
            ).normalize(
                _request(
                    command=ShellGitCommandName.DIFF,
                    options={"mode": "worktree"},
                    pathspecs=("src/app.py",),
                    timeout_seconds=99,
                    max_stdout_bytes=99,
                    max_stderr_bytes=99,
                )
            )

            self.assertEqual(spec.timeout_seconds, 7)
            self.assertEqual(spec.max_stdout_bytes, 17)
            self.assertEqual(spec.max_stderr_bytes, 13)

            grep_spec = await _policy(
                root,
                max_stdout_bytes=23,
                max_diff_bytes=5,
            ).normalize(
                _request(
                    command=ShellGitCommandName.GREP,
                    options={"pattern": "needle", "max_matches": 2},
                    pathspecs=("src/app.py",),
                    max_stdout_bytes=99,
                )
            )

            self.assertEqual(grep_spec.max_stdout_bytes, 23)
            self.assertIn("--max-count=2", grep_spec.argv)

            cases = (
                (
                    _policy(root, max_grep_matches=2),
                    _request(
                        command=ShellGitCommandName.GREP,
                        options={"pattern": "needle", "max_matches": 3},
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _policy(root, max_grep_matches=2),
                    _request(
                        command=ShellGitCommandName.BLAME,
                        options={"start_line": 1, "end_line": 3},
                        pathspecs=("src/app.py",),
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _policy(root),
                    _request(
                        command=ShellGitCommandName.BLAME,
                        options={"end_line": 1},
                        pathspecs=("src/app.py",),
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _policy(root),
                    _request(
                        command=ShellGitCommandName.BLAME,
                        options={"start_line": 3, "end_line": 2},
                        pathspecs=("src/app.py",),
                    ),
                    ShellGitExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    _policy(root, max_pathspecs=1),
                    _request(
                        command=ShellGitCommandName.DIFF,
                        options={"mode": "worktree"},
                        pathspecs=("README.md", "src/app.py"),
                    ),
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                ),
                (
                    _policy(root, max_pathspec_bytes=3),
                    _request(
                        command=ShellGitCommandName.GREP,
                        options={"pattern": "needle"},
                        pathspecs=("README.md",),
                    ),
                    ShellGitExecutionErrorCode.PATHSPEC_DENIED,
                ),
            )

            for policy, request, error_code in cases:
                with self.subTest(
                    command=request.command.value,
                    options=request.options,
                    pathspecs=request.pathspecs,
                ):
                    error = await _policy_error(policy, request)
                    self.assertEqual(error.error_code, error_code)


class GitContentResultPhase4Test(IsolatedAsyncioTestCase):
    async def test_content_tools_return_stable_formatted_successes(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                stdout="content\n",
                status=ShellExecutionStatus.COMPLETED,
                exit_code=0,
            )
            toolset = _fake_toolset(root, executor)

            for command_id, kwargs in _PHASE4_CALLS.items():
                with self.subTest(command_id=command_id):
                    result = await _call_tool(toolset, command_id, **kwargs)
                    self.assertEqual(
                        result.git_result.status,
                        ShellGitExecutionStatus.SUCCESS,
                    )
                    self.assertEqual(result.git_result.error_code, None)
                    self.assertIn("status: success", result)
                    self.assertIn("stdout:\ncontent", result)

        self.assertEqual(executor.calls, len(_PHASE4_CALLS))

    async def test_content_policy_denials_are_formatted_results(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor()
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_show",
                revision="HEAD:README.md",
            )

        self.assertEqual(executor.calls, 0)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.REVISION_DENIED,
        )
        self.assertIn("status: policy_denied", result)
        self.assertIn("error_code: revision_denied", result)

    async def test_non_grep_no_match_status_is_failed_result(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                status=ShellExecutionStatus.NO_MATCHES,
                exit_code=1,
            )
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_diff",
                **_PHASE4_CALLS["git_diff"],
            )

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.NONZERO_EXIT,
        )
        self.assertIn("status: failed", result)
        self.assertIn("error_code: nonzero_exit", result)

    async def test_content_truncation_results_are_stable(self) -> None:
        cases = (
            ("git_diff", _PHASE4_CALLS["git_diff"]),
            ("git_show", _PHASE4_CALLS["git_show"]),
            ("git_blame", _PHASE4_CALLS["git_blame"]),
            ("git_grep", _PHASE4_CALLS["git_grep"]),
            ("git_stash_show", _PHASE4_CALLS["git_stash_show"]),
        )

        for command_id, kwargs in cases:
            with self.subTest(command_id=command_id):
                with TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    _write_minimal_git_repo(root / "repo")
                    executor = _FakeGitExecutor(
                        stdout="x" * 128,
                        status=ShellExecutionStatus.COMPLETED,
                        exit_code=0,
                        stdout_truncated=True,
                    )
                    toolset = _fake_toolset(root, executor)

                    result = await _call_tool(
                        toolset,
                        command_id,
                        **kwargs,
                    )

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

    async def test_content_timeout_result_is_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                status=ShellExecutionStatus.TIMEOUT,
                exit_code=None,
                timed_out=True,
                error_message="process timed out",
            )
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_grep",
                pattern="needle",
                paths=("src/app.py",),
            )

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

    async def test_git_grep_no_match_result_is_stable(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                status=ShellExecutionStatus.NO_MATCHES,
                exit_code=1,
            )
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_grep",
                pattern="absent",
                paths=("src/app.py",),
            )

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        self.assertEqual(result.git_result.exit_code, 1)
        self.assertEqual(result.git_result.error_code, None)
        self.assertIn("status: success", result)
        self.assertIn("exit_code: 1", result)

    async def test_git_grep_real_errors_still_fail(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            _write_minimal_git_repo(root / "repo")
            executor = _FakeGitExecutor(
                stderr="fatal: grep failed\n",
                status=ShellExecutionStatus.NONZERO_EXIT,
                exit_code=2,
            )
            toolset = _fake_toolset(root, executor)

            result = await _call_tool(
                toolset,
                "git_grep",
                pattern="needle",
                paths=("src/app.py",),
            )

        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.NONZERO_EXIT,
        )
        self.assertIn("status: failed", result)
        self.assertIn("error_code: nonzero_exit", result)


@skipIf(_GIT_BINARY is None, "git executable is not available")
class GitContentSmokePhase4Test(IsolatedAsyncioTestCase):
    async def test_git_diff_allows_deleted_file_pathspec(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(
                root,
                _GIT_BINARY,
                "https://github.com/acme/public.git",
            )
            (repo / "src" / "app.py").unlink()
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            diff = await _call_tool(
                toolset,
                "git_diff",
                paths=("src/app.py",),
            )

        self.assertEqual(
            diff.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        self.assertIn("deleted file mode", diff.git_result.stdout_snippet)

    async def test_git_diff_denies_deleted_file_with_index_v4(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(
                root,
                _GIT_BINARY,
                "https://github.com/acme/public.git",
            )
            _git(repo, _GIT_BINARY, "update-index", "--index-version", "4")
            (repo / "src" / "app.py").unlink()
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            diff = await _call_tool(
                toolset,
                "git_diff",
                paths=("src/app.py",),
            )

        self.assertEqual(
            diff.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            diff.git_result.error_code,
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        )
        self.assertNotEqual(
            diff.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertEqual(
            diff.git_result.error_message,
            "Git repository index format is unsupported",
        )

    async def test_git_diff_denies_deleted_file_with_split_index(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(
                root,
                _GIT_BINARY,
                "https://github.com/acme/public.git",
            )
            try:
                _git(repo, _GIT_BINARY, "update-index", "--split-index")
            except CalledProcessError as error:
                self.skipTest(
                    "git update-index --split-index is unsupported: "
                    f"{error.stderr}"
                )
            (repo / "src" / "app.py").unlink()
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            diff = await _call_tool(
                toolset,
                "git_diff",
                paths=("src/app.py",),
            )

        self.assertEqual(
            diff.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            diff.git_result.error_code,
            ShellGitExecutionErrorCode.UNSAFE_GIT_CONFIG,
        )
        self.assertNotEqual(
            diff.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )
        self.assertEqual(
            diff.git_result.error_message,
            "Git repository index format is unsupported",
        )

    async def test_git_diff_denies_deleted_directory_pathspec(self) -> None:
        assert _GIT_BINARY is not None
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(
                root,
                _GIT_BINARY,
                "https://github.com/acme/public.git",
            )
            (repo / "src" / "extra.py").write_text("extra\n")
            _git(repo, _GIT_BINARY, "add", "src/extra.py")
            _git(repo, _GIT_BINARY, "commit", "-m", "extra content")
            (repo / "src" / "app.py").unlink()
            (repo / "src" / "extra.py").unlink()
            (repo / "src").rmdir()
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            diff = await _call_tool(
                toolset,
                "git_diff",
                paths=("src",),
            )

        self.assertEqual(
            diff.git_result.status,
            ShellGitExecutionStatus.POLICY_DENIED,
        )
        self.assertEqual(
            diff.git_result.error_code,
            ShellGitExecutionErrorCode.PATHSPEC_DENIED,
        )

    async def test_content_tools_execute_against_temporary_repo(self) -> None:
        assert _GIT_BINARY is not None
        secret_url = "https://token@github.com/acme/private.git"
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            repo = _write_real_git_repo(root, _GIT_BINARY, secret_url)
            (repo / "src" / "app.py").write_text(
                "needle changed\nneedle extra\n"
            )
            toolset = _real_git_toolset(root, repo, _GIT_BINARY)

            diff = await _call_tool(
                toolset,
                "git_diff",
                paths=("src/app.py",),
            )
            show = await _call_tool(toolset, "git_show", revision="HEAD")
            blame = await _call_tool(
                toolset,
                "git_blame",
                path="README.md",
                start_line=1,
                end_line=1,
            )
            grep = await _call_tool(
                toolset,
                "git_grep",
                pattern="needle",
                paths=("src/app.py",),
                max_matches=2,
            )
            redacted = await _call_tool(
                toolset,
                "git_grep",
                pattern=secret_url,
                paths=("secrets.txt",),
                max_matches=1,
            )
            _git(repo, _GIT_BINARY, "stash", "push", "-m", "save")
            stash_list = await _call_tool(
                toolset,
                "git_stash_list",
                max_count=2,
            )
            stash_show = await _call_tool(
                toolset,
                "git_stash_show",
                stash="stash@{0}",
                mode="patch",
                paths=("src/app.py",),
            )
            no_match = await _call_tool(
                toolset,
                "git_grep",
                pattern="absent",
                paths=("src/app.py",),
                max_matches=1,
            )
            bad_ref = await _call_tool(
                toolset,
                "git_show",
                revision="BADREF",
            )
            path_like_range = await _call_tool(
                toolset,
                "git_diff",
                mode="range",
                base_revision="README.md",
                head_revision="HEAD",
                paths=("src/app.py",),
            )
            truncated = await _call_tool(
                _real_git_toolset(
                    root,
                    repo,
                    _GIT_BINARY,
                    max_stdout_bytes=20,
                    max_diff_bytes=20,
                ),
                "git_stash_show",
                stash="stash@{0}",
                mode="patch",
                paths=("src/app.py",),
            )

        for result in (
            diff,
            show,
            blame,
            grep,
            redacted,
            stash_list,
            stash_show,
        ):
            self.assertEqual(
                result.git_result.status,
                ShellGitExecutionStatus.SUCCESS,
            )
            self.assertIn("status: success", result)

        self.assertIn("needle changed", diff.git_result.stdout_snippet)
        self.assertIn("content setup", show.git_result.stdout_snippet)
        self.assertIn("README.md", blame.git_result.stdout_snippet)
        self.assertIn("src/app.py", grep.git_result.stdout_snippet)
        self.assertIn("save", stash_list.git_result.stdout_snippet)
        self.assertIn("needle changed", stash_show.git_result.stdout_snippet)
        self.assertEqual(
            no_match.git_result.status,
            ShellGitExecutionStatus.SUCCESS,
        )
        self.assertEqual(no_match.git_result.exit_code, 1)
        self.assertEqual(no_match.git_result.error_code, None)
        self.assertEqual(no_match.git_result.stdout_snippet, "")

        redacted_argv = " ".join(redacted.git_result.display_argv)
        self.assertNotIn("token", redacted_argv)
        self.assertIn("https://github.com/[redacted]", redacted_argv)
        self.assertNotIn("token", redacted.git_result.stdout_snippet)

        self.assertEqual(
            bad_ref.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            bad_ref.git_result.error_code,
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        )
        self.assertEqual(
            path_like_range.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            path_like_range.git_result.error_code,
            ShellGitExecutionErrorCode.REVISION_NOT_FOUND,
        )
        self.assertIn(
            "README.md^{commit}",
            path_like_range.git_result.display_argv,
        )
        self.assertEqual(
            truncated.git_result.status,
            ShellGitExecutionStatus.FAILED,
        )
        self.assertEqual(
            truncated.git_result.error_code,
            ShellGitExecutionErrorCode.OUTPUT_TRUNCATED,
        )
        self.assertTrue(truncated.git_result.stdout_truncated)


def _git_prefix() -> tuple[str, str, str]:
    return ("git", "--no-pager", "--no-optional-locks")


def _git_index_data(
    *entries: bytes,
    version: int = 3,
    extensions: tuple[tuple[bytes, bytes], ...] = (),
    checksum: bytes = b"",
) -> bytes:
    return (
        b"DIRC"
        + version.to_bytes(4, "big")
        + len(entries).to_bytes(4, "big")
        + b"".join(entries)
        + b"".join(
            signature + len(data).to_bytes(4, "big") + data
            for signature, data in extensions
        )
        + checksum
    )


def _git_index_entry(
    path: str,
    *,
    mode: int = 0o100644,
    extended: bool = False,
    nul: bool = True,
    pad: bool = True,
) -> bytes:
    path_bytes = path.encode("utf-8", "surrogateescape")
    flags = len(path_bytes)
    if extended:
        flags |= 0x4000
    header = bytearray(62)
    header[24:28] = mode.to_bytes(4, "big")
    header[60:62] = flags.to_bytes(2, "big")
    entry = bytes(header)
    if extended:
        entry += b"\x00\x00"
    entry += path_bytes
    if nul:
        entry += b"\x00"
    if pad:
        remainder = len(entry) % 8
        if remainder:
            entry += b"\x00" * (8 - remainder)
    return entry


def _git_index_truncated_extended_entry() -> bytes:
    entry = bytearray(62)
    entry[60:62] = (0x4000).to_bytes(2, "big")
    return bytes(entry)


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


def _request_with_unsafe_option_key() -> ShellGitCommandRequest:
    return ShellGitCommandRequest(
        tool_name="shell.git_diff",
        command=ShellGitCommandName.DIFF,
        capability_required=SHELL_GIT_COMMAND_CAPABILITIES[
            ShellGitCommandName.DIFF
        ],
        options={1: True},  # type: ignore[dict-item]
    )


def _policy(
    workspace_root: Path,
    *,
    default_timeout_seconds: float = 10.0,
    max_timeout_seconds: float = 60.0,
    max_stdout_bytes: int = 65536,
    max_stderr_bytes: int = 32768,
    max_diff_bytes: int = 131072,
    max_grep_matches: int = 1000,
    max_log_count: int = 50,
    max_pathspecs: int = 64,
    max_pathspec_bytes: int = 4096,
) -> GitExecutionPolicy:
    return GitExecutionPolicy(
        settings=ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root=str(workspace_root),
                cwd="repo",
                allowed_commands=_PHASE4_COMMANDS,
                default_timeout_seconds=default_timeout_seconds,
                max_timeout_seconds=max_timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
                max_diff_bytes=max_diff_bytes,
                max_grep_matches=max_grep_matches,
                max_log_count=max_log_count,
                max_pathspecs=max_pathspecs,
                max_pathspec_bytes=max_pathspec_bytes,
            )
        ),
        executable_lookup=_fake_executable,
    )


async def _fake_executable(search_paths: tuple[str, ...]) -> str | None:
    assert isinstance(search_paths, tuple)
    return "/usr/bin/git"


async def _missing_executable(search_paths: tuple[str, ...]) -> str | None:
    assert isinstance(search_paths, tuple)
    return "/definitely/missing/git"


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
    (repo / "src" / "app.py").write_text("needle\n")
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
                allowed_commands=_PHASE4_COMMANDS,
            ),
        ),
        executor=executor,
    ).with_enabled_tools(_PHASE4_TOOL_NAMES)


def _real_git_toolset(
    root: Path,
    repo: Path,
    git_binary: str,
    *,
    max_stdout_bytes: int = 65536,
    max_diff_bytes: int = 131072,
) -> ShellToolSet:
    return ShellToolSet(
        settings=ShellToolSettings(
            executable_search_paths=(str(Path(git_binary).parent),),
            git=ShellGitToolSettings(
                workspace_root=str(root),
                cwd=repo.name,
                allowed_commands=_PHASE4_COMMANDS,
                max_stdout_bytes=max_stdout_bytes,
                max_diff_bytes=max_diff_bytes,
            ),
        )
    ).with_enabled_tools(_PHASE4_TOOL_NAMES)


def _write_real_git_repo(
    root: Path,
    git_binary: str,
    secret_url: str,
) -> Path:
    repo = root / "repo"
    _git(root, git_binary, "init", "repo")
    _git(repo, git_binary, "checkout", "-b", "main")
    _git(repo, git_binary, "config", "user.name", "Avalan Test")
    _git(repo, git_binary, "config", "user.email", "avalan@example.test")
    (repo / "README.md").write_text("hello\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("needle original\n")
    (repo / "secrets.txt").write_text(f"remote={secret_url}\n")
    _git(repo, git_binary, "add", "README.md", "src/app.py", "secrets.txt")
    _git(repo, git_binary, "commit", "-m", "content setup")
    return repo


def _git(cwd: Path, git_binary: str, *args: str) -> None:
    isolation_root = _git_isolation_root(cwd)
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


_PHASE4_CALLS: dict[str, dict[str, object]] = {
    "git_diff": {"mode": "worktree", "paths": ("src/app.py",)},
    "git_show": {
        "revision": "HEAD",
        "mode": "patch",
        "paths": ("src/app.py",),
    },
    "git_blame": {
        "path": "src/app.py",
        "start_line": 1,
        "end_line": 1,
    },
    "git_grep": {
        "pattern": "needle",
        "paths": ("src/app.py",),
        "case": "sensitive",
        "max_matches": 2,
    },
    "git_stash_list": {"max_count": 1},
    "git_stash_show": {
        "stash": "stash@{0}",
        "mode": "stat",
        "paths": ("src/app.py",),
    },
}
