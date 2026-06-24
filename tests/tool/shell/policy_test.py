from math import nan
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, patch

from avalan.tool.shell.commands.find import filter_output as filter_find_output
from avalan.tool.shell.commands.helpers import (
    _media_path_argument,
    path_matches_sensitive_denylist,
)
from avalan.tool.shell.commands.rg import _rg_policy_deny_globs
from avalan.tool.shell.entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    ExecutionSpec,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellOutputKind,
    ShellPolicyDenied,
)
from avalan.tool.shell.filesystem import ShellPathMetadata
from avalan.tool.shell.policy import (
    ExecutionPolicy,
    _has_hidden_component,
    _is_virtual_filesystem_path,
    _normalized_paths,
    _normalized_workspace,
    _option_fragments,
    _path_component_candidates,
)
from avalan.tool.shell.registry import ShellCommandDefinition
from avalan.tool.shell.settings import ShellToolSettings


class ExecutionPolicyTest(IsolatedAsyncioTestCase):
    async def test_safe_minimal_request_normalizes_without_execution(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/rg")
        policy = ExecutionPolicy(resolver=resolver)

        spec = await policy.normalize(_request())

        self.assertIsInstance(spec, ExecutionSpec)
        self.assertEqual(spec.backend, "local")
        self.assertEqual(spec.command, "rg")
        self.assertEqual(spec.executable, "/usr/bin/rg")
        self.assertEqual(
            spec.argv,
            (
                "rg",
                "--no-config",
                "--color=never",
                "--no-heading",
                "--line-number",
                "--column",
                "--max-columns",
                "1000",
                "--max-columns-preview",
                *_glob_args(_rg_policy_deny_globs(ShellToolSettings())),
                "-e",
                "needle",
                "--",
                ".",
            ),
        )
        self.assertEqual(
            spec.display_argv,
            (
                "rg",
                "--no-config",
                "--color=never",
                "--no-heading",
                "--line-number",
                "--column",
                "--max-columns",
                "1000",
                "--max-columns-preview",
                "-e",
                "needle",
                "--",
                ".",
            ),
        )
        self.assertEqual(Path(spec.cwd), Path.cwd().resolve())
        self.assertEqual(spec.display_cwd, ".")
        self.assertEqual(spec.env, _expected_default_env())
        self.assertIsNone(spec.stdin)
        self.assertEqual(spec.timeout_seconds, 10.0)
        self.assertEqual(spec.max_stdout_bytes, 65536)
        self.assertEqual(spec.max_stderr_bytes, 32768)
        self.assertEqual(resolver.calls, ("rg",))

    async def test_missing_executable_stays_visible_on_spec(self) -> None:
        spec = await ExecutionPolicy().normalize(_request())

        self.assertIsNone(spec.executable)

    async def test_builds_deterministic_child_environment(self) -> None:
        settings = ShellToolSettings(tesseract_thread_limit=3)

        spec = await ExecutionPolicy(settings=settings).normalize(_request())

        self.assertEqual(spec.env, _expected_default_env(omp="3"))

    async def test_allowlisted_parent_environment_is_opt_in_and_scrubbed(
        self,
    ) -> None:
        settings = ShellToolSettings(
            environment_allowlist=(
                "SAFE_NAME",
                "EMPTY_NAME",
                "PATH",
                "LS_COLORS",
                "SERVICE_TOKEN",
                "MISSING_NAME",
            ),
        )

        with patch.dict(
            "avalan.tool.shell.policy.environ",
            {
                "SAFE_NAME": "safe",
                "EMPTY_NAME": "",
                "PATH": "/tmp/workspace/bin",
                "LS_COLORS": "danger",
                "SERVICE_TOKEN": "secret",
                "UNLISTED_NAME": "leak",
            },
            clear=True,
        ):
            spec = await ExecutionPolicy(settings=settings).normalize(
                _request()
            )

        self.assertEqual(spec.env["SAFE_NAME"], "safe")
        self.assertNotIn("EMPTY_NAME", spec.env)
        self.assertNotIn("PATH", spec.env)
        self.assertNotIn("LS_COLORS", spec.env)
        self.assertNotIn("SERVICE_TOKEN", spec.env)
        self.assertNotIn("MISSING_NAME", spec.env)
        self.assertNotIn("UNLISTED_NAME", spec.env)

    async def test_trusted_environment_injection_is_scrubbed_and_bounded(
        self,
    ) -> None:
        settings = ShellToolSettings(
            environment={
                "CUSTOM_NAME": "custom",
                "LC_ALL": "C.UTF-8",
                "HOME": "/tmp/home",
                "API_KEY": "secret",
                "JQ_LIBRARY_PATH": "/tmp/jq",
                "OMP_THREAD_LIMIT": "9",
            },
            tesseract_thread_limit=2,
        )

        spec = await ExecutionPolicy(settings=settings).normalize(_request())

        self.assertEqual(spec.env["CUSTOM_NAME"], "custom")
        self.assertEqual(spec.env["LC_ALL"], "C")
        self.assertEqual(spec.env["HOME"], "/nonexistent")
        self.assertEqual(spec.env["XDG_CONFIG_HOME"], "/nonexistent")
        self.assertEqual(spec.env["XDG_CACHE_HOME"], "/nonexistent")
        self.assertEqual(spec.env["OMP_THREAD_LIMIT"], "2")
        self.assertNotIn("API_KEY", spec.env)
        self.assertNotIn("JQ_LIBRARY_PATH", spec.env)

    async def test_budget_hints_are_clamped_and_recorded(self) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                default_timeout_seconds=1.0,
                max_timeout_seconds=5.0,
                max_stdout_bytes=10,
                max_stderr_bytes=20,
            )
        )

        spec = await policy.normalize(
            _request(
                timeout_seconds=8.0,
                max_stdout_bytes=11,
                max_stderr_bytes=21,
                metadata={"source": "test"},
            )
        )

        self.assertEqual(spec.timeout_seconds, 5.0)
        self.assertEqual(spec.max_stdout_bytes, 10)
        self.assertEqual(spec.max_stderr_bytes, 20)
        self.assertEqual(spec.metadata["source"], "test")
        self.assertEqual(
            spec.metadata["budget_clamps"],
            {
                "timeout_seconds": {"requested": 8.0, "applied": 5.0},
                "max_stdout_bytes": {"requested": 11, "applied": 10},
                "max_stderr_bytes": {"requested": 21, "applied": 20},
            },
        )

    async def test_budget_hints_at_limits_are_not_clamped(self) -> None:
        resolver = _CountingResolver("/usr/bin/rg")
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                default_timeout_seconds=1.0,
                max_timeout_seconds=5.0,
                max_stdout_bytes=10,
                max_stderr_bytes=20,
                max_arguments=128,
                max_argument_bytes=64,
                max_command_bytes=4096,
            ),
            resolver=resolver,
        )

        spec = await policy.normalize(
            _request(
                options={"pattern": "n"},
                timeout_seconds=5.0,
                max_stdout_bytes=10,
                max_stderr_bytes=20,
            )
        )

        self.assertEqual(spec.timeout_seconds, 5.0)
        self.assertEqual(spec.max_stdout_bytes, 10)
        self.assertEqual(spec.max_stderr_bytes, 20)
        self.assertNotIn("budget_clamps", spec.metadata)
        self.assertEqual(resolver.calls, ("rg",))

    async def test_supported_option_shapes_are_counted(self) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                max_arguments=128,
                max_argument_bytes=64,
                max_command_bytes=4096,
            )
        )

        spec = await policy.normalize(
            _request(
                options={
                    "pattern": "needle",
                    "fixed_strings": False,
                    "context_lines": 1,
                    "max_matches_per_file": 2,
                    "globs": ("*.py",),
                },
                cwd="subdir",
            )
        )

        self.assertEqual(Path(spec.cwd), Path.cwd().resolve() / "subdir")
        self.assertEqual(spec.display_cwd, "subdir")

    async def test_cwd_normalizes_relative_to_workspace_root(self) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                workspace_root="src",
                cwd="avalan",
            )
        )

        spec = await policy.normalize(_request(cwd="tool"))

        self.assertEqual(Path(spec.cwd), Path.cwd().resolve() / "src" / "tool")
        self.assertEqual(spec.display_cwd, "tool")

    async def test_request_paths_are_normalized_against_effective_cwd(
        self,
    ) -> None:
        path = PathOperand(
            name="input",
            path="policy.py",
            kind="text_file",
            access="read",
        )
        settings = ShellToolSettings(workspace_root="src", cwd="avalan/tool")
        workspace = await _normalized_workspace(settings, None)

        normalized_paths = await _normalized_paths(
            (path,),
            workspace=workspace,
            settings=settings,
        )

        self.assertEqual(
            normalized_paths[0].path,
            Path.cwd().resolve() / "src" / "avalan" / "tool" / "policy.py",
        )
        self.assertEqual(
            normalized_paths[0].display_path,
            "avalan/tool/policy.py",
        )

    async def test_absolute_paths_can_be_trusted_only_inside_workspace(
        self,
    ) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(allow_absolute_paths=True)
        )

        spec = await policy.normalize(_request(cwd=str(Path.cwd())))

        self.assertEqual(Path(spec.cwd), Path.cwd().resolve())
        self.assertEqual(spec.display_cwd, ".")

    async def test_zero_byte_budget_hints_are_denied(self) -> None:
        for kwargs in (
            {"max_stdout_bytes": 0},
            {"max_stderr_bytes": 0},
        ):
            with self.subTest(kwargs=kwargs):
                await self._assert_denied(
                    _request(**kwargs),
                    ShellExecutionErrorCode.INVALID_OPTION,
                )

    def test_request_rejects_malformed_budget_hints(self) -> None:
        for kwargs in (
            {"timeout_seconds": 0},
            {"timeout_seconds": -1},
            {"timeout_seconds": True},
            {"timeout_seconds": nan},
            {"timeout_seconds": "1"},
            {"max_stdout_bytes": -1},
            {"max_stdout_bytes": True},
            {"max_stdout_bytes": nan},
            {"max_stdout_bytes": "1"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    _request(**kwargs)

    async def test_policy_revalidates_corrupted_budget_hints(self) -> None:
        cases = (
            ("timeout_seconds", True),
            ("timeout_seconds", nan),
            ("max_stdout_bytes", True),
            ("max_stdout_bytes", nan),
        )

        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                request = _request()
                object.__setattr__(request, field_name, value)
                await self._assert_denied(
                    request,
                    ShellExecutionErrorCode.INVALID_OPTION,
                )

    async def test_denies_stdin_write_and_shell_evaluation(self) -> None:
        write_path = PathOperand(
            name="output",
            path="out.txt",
            kind="file",
            access="write",
        )
        cases = (
            (
                _request(stdin=b"input"),
                ShellExecutionErrorCode.STDIN_DENIED,
            ),
            (
                _request(paths=(write_path,)),
                ShellExecutionErrorCode.WRITE_DENIED,
            ),
            (
                _request(options={"shell": True}),
                ShellExecutionErrorCode.SHELL_DENIED,
            ),
            (
                _request(options={"shell_eval": True}),
                ShellExecutionErrorCode.SHELL_DENIED,
            ),
            (
                _request(options={"use_shell": True}),
                ShellExecutionErrorCode.SHELL_DENIED,
            ),
        )

        for request, error_code in cases:
            with self.subTest(error_code=error_code):
                await self._assert_denied(request, error_code)

    async def test_denies_disallowed_command_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/cat")
        policy = ExecutionPolicy(
            settings=ShellToolSettings(allowed_commands=("rg",)),
            resolver=resolver,
        )

        await self._assert_denied(
            _request(tool_name="shell.cat", command="cat"),
            ShellExecutionErrorCode.DENIED_COMMAND,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ())

    async def test_denies_tool_name_command_mismatch_before_resolver(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/cat")
        policy = ExecutionPolicy(resolver=resolver)
        cases = (
            _request(tool_name="shell.rg", command="cat"),
            _request(tool_name="tools.cat", command="cat"),
        )

        for request in cases:
            with self.subTest(tool_name=request.tool_name):
                await self._assert_denied(
                    request,
                    ShellExecutionErrorCode.DENIED_COMMAND,
                    policy=policy,
                )

        self.assertEqual(resolver.calls, ())

    async def test_denies_invalid_cwd_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/rg")
        policy = ExecutionPolicy(resolver=resolver)
        cases = (
            ("", ShellExecutionErrorCode.INVALID_CWD),
            ("\x00", ShellExecutionErrorCode.INVALID_CWD),
            ("~", ShellExecutionErrorCode.INVALID_CWD),
            ("$HOME", ShellExecutionErrorCode.INVALID_CWD),
            ("../outside", ShellExecutionErrorCode.TRAVERSAL),
            (str(Path.cwd()), ShellExecutionErrorCode.INVALID_CWD),
        )

        for cwd, error_code in cases:
            with self.subTest(cwd=cwd, error_code=error_code):
                request = _request()
                object.__setattr__(request, "cwd", cwd)
                await self._assert_denied(request, error_code, policy=policy)
        self.assertEqual(resolver.calls, ())

    async def test_denies_hidden_and_sensitive_cwd_before_resolver(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".hidden").mkdir()
            (root / ".git").mkdir()
            resolver = _CountingResolver("/usr/bin/rg")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
                resolver=resolver,
            )

            for cwd, error_code in (
                (".hidden", ShellExecutionErrorCode.HIDDEN_PATH),
                (".git", ShellExecutionErrorCode.HIDDEN_PATH),
            ):
                with self.subTest(cwd=cwd):
                    await self._assert_denied(
                        _request(cwd=cwd),
                        error_code,
                        policy=policy,
                    )

            sensitive_policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                ),
                resolver=resolver,
            )
            await self._assert_denied(
                _request(cwd=".git"),
                ShellExecutionErrorCode.SENSITIVE_PATH,
                policy=sensitive_policy,
            )

        self.assertEqual(resolver.calls, ())

    async def test_denies_symlink_cwd_before_resolver(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "target").mkdir()
            link = root / "link"
            try:
                link.symlink_to(root / "target", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symlink unavailable: {error}")

            resolver = _CountingResolver("/usr/bin/rg")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
                resolver=resolver,
            )

            await self._assert_denied(
                _request(cwd="link"),
                ShellExecutionErrorCode.SYMLINK,
                policy=policy,
            )

        self.assertEqual(resolver.calls, ())

    async def test_denies_invalid_paths_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/rg")
        policy = ExecutionPolicy(resolver=resolver)
        cases = (
            ("", ShellExecutionErrorCode.DENIED_PATH),
            ("\x00", ShellExecutionErrorCode.DENIED_PATH),
            ("~/.profile", ShellExecutionErrorCode.DENIED_PATH),
            ("$HOME/.profile", ShellExecutionErrorCode.DENIED_PATH),
            ("../outside.txt", ShellExecutionErrorCode.TRAVERSAL),
            ("-", ShellExecutionErrorCode.DENIED_PATH),
            (
                str(Path.cwd().resolve() / "input.txt"),
                ShellExecutionErrorCode.DENIED_PATH,
            ),
        )

        for value, error_code in cases:
            with self.subTest(value=value, error_code=error_code):
                path = PathOperand(
                    name="input",
                    path="placeholder",
                    kind="text_file",
                    access="read",
                )
                object.__setattr__(path, "path", value)
                await self._assert_denied(
                    _request(paths=(path,)),
                    error_code,
                    policy=policy,
                )
        self.assertEqual(resolver.calls, ())

    async def test_dash_prefixed_paths_are_preserved(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "-filename.txt").write_text("value", encoding="utf-8")
            path = PathOperand(
                name="input",
                path="-filename.txt",
                kind="text_file",
                access="read",
            )

            spec = await ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            ).normalize(_request(paths=(path,)))

        self.assertEqual(spec.display_cwd, ".")
        self.assertEqual(spec.argv[-1], "-filename.txt")

    async def test_rg_builds_full_argv_and_omits_injected_globs_from_display(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "-needle.txt").write_text("needle", encoding="utf-8")
            settings = ShellToolSettings(
                workspace_root=str(root),
                max_rg_context_lines=3,
                max_rg_matches_per_file=7,
            )
            request = _request(
                options={
                    "pattern": "-needle",
                    "case": "smart",
                    "fixed_strings": True,
                    "context_lines": 2,
                    "max_matches_per_file": 5,
                    "globs": ("*.txt", "!vendor/**"),
                },
                paths=(_path("-needle.txt"),),
            )

            spec = await ExecutionPolicy(
                settings=settings,
                resolver=_CountingResolver("/usr/bin/rg"),
            ).normalize(request)

        user_glob_args = ("--glob", "*.txt", "--glob", "!vendor/**")
        injected_glob_args = _glob_args(_rg_policy_deny_globs(settings))
        expected_prefix = (
            "rg",
            "--no-config",
            "--color=never",
            "--no-heading",
            "--line-number",
            "--column",
            "--max-columns",
            "1000",
            "--max-columns-preview",
            "--smart-case",
            "--fixed-strings",
            "--context",
            "2",
            "--max-count",
            "5",
            *user_glob_args,
        )

        self.assertEqual(
            spec.argv,
            (
                *expected_prefix,
                *injected_glob_args,
                "-e",
                "-needle",
                "--",
                "-needle.txt",
            ),
        )
        self.assertEqual(
            spec.display_argv,
            (
                *expected_prefix,
                "-e",
                "-needle",
                "--",
                "-needle.txt",
            ),
        )
        self.assertEqual(
            spec.metadata["exit_code_statuses"], {1: "no_matches"}
        )
        self.assertNotIn("!.env", spec.display_argv)

    async def test_rg_builds_case_and_optional_branches(self) -> None:
        insensitive = await ExecutionPolicy().normalize(
            _request(options={"pattern": "needle", "case": "insensitive"})
        )
        sensitive = await ExecutionPolicy().normalize(
            _request(
                options={
                    "pattern": "needle",
                    "fixed_strings": False,
                    "context_lines": 0,
                    "max_matches_per_file": None,
                    "globs": (),
                }
            )
        )

        self.assertIn("--ignore-case", insensitive.argv)
        self.assertNotIn("--smart-case", insensitive.argv)
        self.assertNotIn("--ignore-case", sensitive.argv)
        self.assertNotIn("--fixed-strings", sensitive.argv)
        self.assertNotIn("--context", sensitive.argv)
        self.assertNotIn("--max-count", sensitive.argv)

    async def test_rg_builds_native_context_depth_and_size_limits(
        self,
    ) -> None:
        settings = ShellToolSettings(max_rg_context_lines=3)

        spec = await ExecutionPolicy(
            settings=settings,
            resolver=_CountingResolver("/usr/bin/rg"),
        ).normalize(
            _request(
                options={
                    "pattern": "needle",
                    "context_lines": 0,
                    "before_context": 0,
                    "after_context": 2,
                    "max_depth": 0,
                    "max_filesize_bytes": 2048,
                }
            )
        )

        expected_display_argv = (
            "rg",
            "--no-config",
            "--color=never",
            "--no-heading",
            "--line-number",
            "--column",
            "--max-columns",
            "1000",
            "--max-columns-preview",
            "--before-context",
            "0",
            "--after-context",
            "2",
            "--max-depth",
            "0",
            "--max-filesize",
            "2048",
            "-e",
            "needle",
            "--",
            ".",
        )

        self.assertEqual(spec.display_argv, expected_display_argv)
        self.assertEqual(
            spec.argv,
            (
                *expected_display_argv[:-4],
                *_glob_args(_rg_policy_deny_globs(settings)),
                *expected_display_argv[-4:],
            ),
        )

    async def test_rg_allows_zero_directional_context(self) -> None:
        spec = await ExecutionPolicy(
            resolver=_CountingResolver("/usr/bin/rg"),
        ).normalize(
            _request(
                options={
                    "pattern": "needle",
                    "before_context": 0,
                    "after_context": 0,
                }
            )
        )

        self.assertIn("--before-context", spec.display_argv)
        self.assertIn("--after-context", spec.display_argv)
        self.assertEqual(
            spec.display_argv[spec.display_argv.index("--before-context") + 1],
            "0",
        )
        self.assertEqual(
            spec.display_argv[spec.display_argv.index("--after-context") + 1],
            "0",
        )

    async def test_rg_allows_combined_and_directional_context(self) -> None:
        spec = await ExecutionPolicy(
            resolver=_CountingResolver("/usr/bin/rg"),
        ).normalize(
            _request(
                options={
                    "pattern": "needle",
                    "context_lines": 2,
                    "before_context": 0,
                    "after_context": 1,
                }
            )
        )

        self.assertIn("--context", spec.display_argv)
        self.assertIn("--before-context", spec.display_argv)
        self.assertIn("--after-context", spec.display_argv)
        self.assertEqual(
            spec.display_argv[spec.display_argv.index("--context") + 1],
            "2",
        )
        self.assertEqual(
            spec.display_argv[spec.display_argv.index("--before-context") + 1],
            "0",
        )
        self.assertEqual(
            spec.display_argv[spec.display_argv.index("--after-context") + 1],
            "1",
        )

    async def test_rg_rejects_invalid_options_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/rg")
        policy = ExecutionPolicy(resolver=resolver)
        cases = (
            ({}, ShellExecutionErrorCode.INVALID_OPTION),
            ({"pattern": ""}, ShellExecutionErrorCode.INVALID_OPTION),
            ({"pattern": " "}, ShellExecutionErrorCode.INVALID_OPTION),
            (
                {"pattern": "needle", "case": "upper"},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "fixed_strings": 1},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "context_lines": True},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "context_lines": 11},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "before_context": True},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "before_context": -1},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "after_context": -1},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "max_matches_per_file": 0},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "max_matches_per_file": 1001},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "max_depth": True},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "max_depth": -1},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "max_filesize_bytes": True},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "max_filesize_bytes": 0},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                {"pattern": "needle", "globs": "*.py"},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            ({"hidden": True}, ShellExecutionErrorCode.INVALID_OPTION),
            ({"unknown": "value"}, ShellExecutionErrorCode.INVALID_OPTION),
        )

        for options, error_code in cases:
            with self.subTest(options=options):
                await self._assert_denied(
                    _request(options=options),
                    error_code,
                    policy=policy,
                )
        self.assertEqual(resolver.calls, ())

    async def test_rg_option_policy_denials_include_option_name(
        self,
    ) -> None:
        policy = ExecutionPolicy(resolver=_CountingResolver("/usr/bin/rg"))
        cases = (
            (
                {"pattern": "needle", "pattern_typo": "value"},
                "unknown rg option: pattern_typo",
            ),
            (
                {"pattern": "needle", "hidden": True},
                "unsupported rg option: hidden",
            ),
        )

        for options, message in cases:
            with self.subTest(options=options):
                await self._assert_denied(
                    _request(options=options),
                    ShellExecutionErrorCode.INVALID_OPTION,
                    policy=policy,
                    message=message,
                )

    async def test_rg_rejects_invalid_globs_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/rg")
        cases = (
            (
                ShellToolSettings(max_glob_count=1),
                {"pattern": "needle", "globs": ("*.py", "*.txt")},
                ShellExecutionErrorCode.GLOB_TOO_LARGE,
            ),
            (
                ShellToolSettings(max_glob_bytes_per_glob=4),
                {"pattern": "needle", "globs": ("abcde",)},
                ShellExecutionErrorCode.GLOB_TOO_LARGE,
            ),
            (
                ShellToolSettings(max_total_glob_bytes=8),
                {"pattern": "needle", "globs": ("abcd", "efghi")},
                ShellExecutionErrorCode.GLOB_TOO_LARGE,
            ),
            (
                ShellToolSettings(),
                {"pattern": "needle", "globs": (1,)},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                ShellToolSettings(),
                {"pattern": "needle", "globs": ("/tmp/*.py",)},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                ShellToolSettings(),
                {"pattern": "needle", "globs": ("../*.py",)},
                ShellExecutionErrorCode.TRAVERSAL,
            ),
            (
                ShellToolSettings(),
                {"pattern": "needle", "globs": (".*",)},
                ShellExecutionErrorCode.HIDDEN_PATH,
            ),
            (
                ShellToolSettings(),
                {"pattern": "needle", "globs": ("**/*.pem",)},
                ShellExecutionErrorCode.SENSITIVE_PATH,
            ),
            (
                ShellToolSettings(),
                {"pattern": "needle", "globs": ("**/**/file.txt",)},
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
        )

        for settings, options, error_code in cases:
            with self.subTest(options=options):
                await self._assert_denied(
                    _request(options=options),
                    error_code,
                    policy=ExecutionPolicy(
                        settings=settings,
                        resolver=resolver,
                    ),
                )
        self.assertEqual(resolver.calls, ())

    async def test_rg_glob_boundaries_and_argv_growth_are_bounded(
        self,
    ) -> None:
        settings = ShellToolSettings(
            max_glob_count=2,
            max_glob_bytes_per_glob=4,
            max_total_glob_bytes=8,
        )

        spec = await ExecutionPolicy(settings=settings).normalize(
            _request(options={"pattern": "n", "globs": ("a.py", "b.py")})
        )

        self.assertIn("--glob", spec.argv)
        self.assertLessEqual(
            len(spec.argv),
            13
            + len(("a.py", "b.py")) * 2
            + len(_rg_policy_deny_globs(settings)) * 2,
        )

    async def test_rg_max_path_and_glob_normalization_is_bounded(
        self,
    ) -> None:
        path_count = 4
        globs = ("*.py", "*.txt", "*.md")
        paths = tuple(_path(f"file{index}.txt") for index in range(path_count))
        settings = ShellToolSettings(
            max_path_count=path_count,
            max_glob_count=len(globs),
            max_glob_bytes_per_glob=8,
            max_total_glob_bytes=sum(len(glob) for glob in globs),
            max_arguments=128,
            max_argument_bytes=64,
            max_command_bytes=4096,
        )
        resolver = _CountingResolver("/usr/bin/rg")

        with patch(
            "avalan.tool.shell.policy.inspect_path",
            new=AsyncMock(side_effect=FileNotFoundError),
        ) as inspect:
            spec = await ExecutionPolicy(
                settings=settings,
                resolver=resolver,
            ).normalize(
                _request(
                    options={"pattern": "n", "globs": globs},
                    paths=paths,
                )
            )

        self.assertEqual(
            spec.display_argv[-path_count:], tuple(path.path for path in paths)
        )
        self.assertEqual(inspect.await_count, path_count)
        self.assertEqual(resolver.calls, ("rg",))
        self.assertEqual(
            spec.argv.count("--glob"),
            len(globs) + len(_rg_policy_deny_globs(settings)),
        )
        self.assertLessEqual(
            len(spec.argv),
            13
            + len(globs) * 2
            + len(_rg_policy_deny_globs(settings)) * 2
            + path_count,
        )

    async def test_policy_boundary_denial_does_not_reach_executor(
        self,
    ) -> None:
        executor = _RecordingExecutor()

        with self.assertRaises(ShellPolicyDenied):
            spec = await ExecutionPolicy().normalize(
                _request(tool_name="shell.cat", command="cat")
            )
            await executor.execute(spec)

        self.assertEqual(executor.calls, ())

    async def test_rg_argv_paths_stay_relative_to_effective_cwd(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "subdir").mkdir()
            outside_cwd = root / "outside.txt"
            outside_cwd.write_text("needle", encoding="utf-8")

            spec = await ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    cwd="subdir",
                    allow_absolute_paths=True,
                )
            ).normalize(
                _request(
                    paths=(_path(str(outside_cwd), kind="text_file"),),
                )
            )

        self.assertEqual(spec.argv[-1], "../outside.txt")
        self.assertEqual(spec.display_argv[-1], "outside.txt")
        self.assertNotIn(str(root.resolve()), spec.argv)
        self.assertNotIn(str(root.resolve()), spec.display_argv)

    async def test_rg_rejects_unsupported_path_kinds(self) -> None:
        await self._assert_denied(
            _request(paths=(_path("data.json", kind="json_file"),)),
            ShellExecutionErrorCode.INVALID_OPTION,
        )

    async def test_core_commands_build_portable_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "file.txt").write_text("first\nsecond\n", encoding="utf-8")
            (root / "-dash.txt").write_text("dash\n", encoding="utf-8")
            (root / "directory").mkdir()
            (root / "!").mkdir()
            (root / "(").mkdir()
            settings = ShellToolSettings(
                workspace_root=str(root),
                max_head_lines=5,
                max_tail_lines=90,
            )
            policy = ExecutionPolicy(settings=settings)

            head = await policy.normalize(
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"lines": 5},
                    paths=(_path("file.txt"),),
                )
            )
            tail = await policy.normalize(
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    paths=(_path("-dash.txt"),),
                )
            )
            cat = await policy.normalize(
                _request(
                    tool_name="shell.cat",
                    command="cat",
                    paths=(_path("file.txt"),),
                )
            )
            file_type = await policy.normalize(
                _request(
                    tool_name="shell.file",
                    command="file",
                    options={"brief": True, "mime_type": True},
                    paths=(_path("file.txt", kind="file"),),
                )
            )
            listing = await policy.normalize(
                _request(
                    tool_name="shell.ls",
                    command="ls",
                    paths=(_path("directory", kind="directory"),),
                )
            )
            found_file = await policy.normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={
                        "max_depth": 2,
                        "entry_type": "file",
                        "name": "file.txt",
                    },
                    paths=(_path("directory", kind="directory"),),
                )
            )
            found_directory = await policy.normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={
                        "max_depth": 0,
                        "entry_type": "directory",
                    },
                )
            )
            found_bang = await policy.normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    paths=(_path("!", kind="directory"),),
                )
            )
            found_paren = await policy.normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    paths=(_path("(", kind="directory"),),
                )
            )
            wc = await policy.normalize(
                _request(
                    tool_name="shell.wc",
                    command="wc",
                    options={
                        "lines": True,
                        "words": True,
                        "count_bytes": True,
                    },
                    paths=(_path("file.txt"), _path("-dash.txt")),
                )
            )

        self.assertEqual(head.argv, ("head", "-n", "5", "--", "file.txt"))
        self.assertEqual(
            head.display_argv,
            ("head", "-n", "5", "--", "file.txt"),
        )
        self.assertEqual(tail.argv, ("tail", "-n", "80", "--", "-dash.txt"))
        self.assertEqual(cat.argv, ("cat", "--", "file.txt"))
        self.assertEqual(
            file_type.argv,
            ("file", "--brief", "--mime-type", "--", "file.txt"),
        )
        self.assertEqual(listing.argv, ("ls", "-1p", "--", "directory"))
        self.assertEqual(
            found_file.display_argv,
            (
                "find",
                "directory",
                "-maxdepth",
                "2",
                "-type",
                "f",
                "-name",
                "file.txt",
                "-print",
            ),
        )
        self.assertIn("-prune", found_file.argv)
        self.assertIn("-name", found_file.argv)
        self.assertIn(".*", found_file.argv)
        self.assertIn(".ssh", found_file.argv)
        self.assertNotIn(".ssh", found_file.display_argv)
        self.assertEqual(
            found_file.argv[-5:],
            ("-type", "f", "-name", "file.txt", "-print"),
        )
        self.assertEqual(
            found_directory.display_argv,
            ("find", ".", "-maxdepth", "0", "-type", "d", "-print"),
        )
        self.assertEqual(found_bang.argv[1], "./!")
        self.assertEqual(found_bang.display_argv[1], "./!")
        self.assertEqual(found_paren.argv[1], "./(")
        self.assertEqual(found_paren.display_argv[1], "./(")
        self.assertEqual(
            wc.argv,
            ("wc", "-l", "-w", "-c", "--", "file.txt", "-dash.txt"),
        )
        self.assertEqual(len(wc.argv), 5 + 2)

    async def test_head_tail_build_native_byte_and_start_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "file.txt").write_text("first\nsecond\n", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    max_head_lines=5,
                    max_tail_lines=5,
                )
            )

            head_bytes = await policy.normalize(
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"byte_count": 6, "lines": 80},
                    paths=(_path("file.txt"),),
                )
            )
            tail_start_line = await policy.normalize(
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_line": 2, "lines": 80},
                    paths=(_path("file.txt"),),
                )
            )
            tail_bytes = await policy.normalize(
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"byte_count": 7, "lines": 80},
                    paths=(_path("file.txt"),),
                )
            )
            tail_start_byte = await policy.normalize(
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_byte": 7, "lines": 80},
                    paths=(_path("file.txt"),),
                )
            )

        self.assertEqual(
            head_bytes.argv, ("head", "-c", "6", "--", "file.txt")
        )
        self.assertEqual(
            head_bytes.display_argv,
            ("head", "-c", "6", "--", "file.txt"),
        )
        self.assertEqual(
            tail_start_line.argv,
            ("tail", "-n", "+2", "--", "file.txt"),
        )
        self.assertEqual(
            tail_bytes.argv, ("tail", "-c", "7", "--", "file.txt")
        )
        self.assertEqual(
            tail_start_byte.argv,
            ("tail", "-c", "+7", "--", "file.txt"),
        )

    async def test_head_tail_native_byte_modes_respect_stdout_budget(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "large.txt").write_text("1234567890", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    max_stdout_bytes=5,
                )
            )

            safe_suffix = await policy.normalize(
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_byte": 6},
                    paths=(_path("large.txt"),),
                )
            )
            denied_cases = (
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"byte_count": 6},
                    paths=(_path("large.txt"),),
                ),
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"byte_count": 6},
                    paths=(_path("large.txt"),),
                ),
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_line": 1},
                    paths=(_path("large.txt"),),
                ),
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_byte": 1},
                    paths=(_path("large.txt"),),
                ),
            )

            for request in denied_cases:
                with self.subTest(command=request.command):
                    await self._assert_denied(
                        request,
                        ShellExecutionErrorCode.INVALID_OPTION,
                        policy=policy,
                    )

        self.assertEqual(
            safe_suffix.argv,
            ("tail", "-c", "+6", "--", "large.txt"),
        )

    async def test_tail_from_start_requires_available_file_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=temporary_directory)
            )

            await self._assert_denied(
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_byte": 1},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )

    async def test_find_builds_native_min_depth_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "directory").mkdir()
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            found_file = await policy.normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={
                        "max_depth": 2,
                        "min_depth": 1,
                        "entry_type": "file",
                    },
                    paths=(_path("directory", kind="directory"),),
                )
            )
            omitted_min_depth = await policy.normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"min_depth": None},
                )
            )

        self.assertEqual(
            found_file.display_argv,
            (
                "find",
                "directory",
                "-mindepth",
                "1",
                "-maxdepth",
                "2",
                "-type",
                "f",
                "-print",
            ),
        )
        self.assertIn("-mindepth", found_file.argv)
        self.assertEqual(
            omitted_min_depth.display_argv,
            ("find", ".", "-maxdepth", "3", "-print"),
        )

    async def test_ls_hidden_flag_is_trusted_configuration_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "directory").mkdir()
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                )
            )

            default_path = await policy.normalize(
                _request(tool_name="shell.ls", command="ls")
            )
            directory = await policy.normalize(
                _request(
                    tool_name="shell.ls",
                    command="ls",
                    paths=(_path("directory", kind="directory"),),
                )
            )

        self.assertEqual(default_path.argv, ("ls", "-1p", "-A", "--", "."))
        self.assertEqual(
            directory.argv, ("ls", "-1p", "-A", "--", "directory")
        )

    async def test_find_hidden_behavior_is_trusted_configuration_only(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".hidden").mkdir()
            (root / "-dash").mkdir()

            default_find = await ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            ).normalize(_request(tool_name="shell.find", command="find"))
            hidden_find = await ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                ),
            ).normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={
                        "max_depth": 1,
                        "entry_type": "any",
                        "name": ".hidden",
                    },
                    paths=(_path(".hidden", kind="directory"),),
                )
            )
            dash_find = await ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            ).normalize(
                _request(
                    tool_name="shell.find",
                    command="find",
                    paths=(_path("-dash", kind="directory"),),
                )
            )

        self.assertEqual(
            default_find.display_argv,
            ("find", ".", "-maxdepth", "3", "-print"),
        )
        self.assertIn(".*", default_find.argv)
        self.assertNotIn(".*", hidden_find.argv)
        self.assertIn("-iname", hidden_find.argv)
        self.assertIn(".git", hidden_find.argv)
        self.assertIn(".env*", hidden_find.argv)
        self.assertIn("credentials", hidden_find.argv)
        self.assertIn("*.key", hidden_find.argv)
        self.assertIn("*.pem", hidden_find.argv)
        self.assertIn("id_ed25519", hidden_find.argv)
        self.assertEqual(hidden_find.display_argv[1], ".hidden")
        self.assertEqual(dash_find.argv[1], "./-dash")
        self.assertEqual(dash_find.display_argv[1], "./-dash")

    def test_find_output_filter_redacts_denied_paths(self) -> None:
        filtered = filter_find_output("./visible.txt\n./.env\n./.ssh/key\n")

        self.assertIn("./visible.txt", filtered)
        self.assertEqual(filtered.count("[redacted_path]"), 2)
        self.assertNotIn(".ssh", filtered)

    async def test_wc_defaults_to_line_count_when_flags_are_false(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "file.txt").write_text("first\nsecond\n", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            default_flags = await policy.normalize(
                _request(
                    tool_name="shell.wc",
                    command="wc",
                    paths=(_path("file.txt"),),
                )
            )
            all_false = await policy.normalize(
                _request(
                    tool_name="shell.wc",
                    command="wc",
                    options={
                        "lines": False,
                        "words": False,
                        "count_bytes": False,
                    },
                    paths=(_path("file.txt"),),
                )
            )

        self.assertEqual(default_flags.argv, ("wc", "-l", "--", "file.txt"))
        self.assertEqual(all_false.argv, ("wc", "-l", "--", "file.txt"))

    async def test_core_commands_reject_invalid_options_before_resolver(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/core")
        policy = ExecutionPolicy(
            settings=ShellToolSettings(max_tail_lines=2),
            resolver=resolver,
        )
        cases = (
            (
                _request(tool_name="shell.head", command="head"),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"lines": 0},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"lines": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"byte_count": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"byte_count": 0},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"byte_count": 6, "lines": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"lines": 3},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_line": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_line": 0},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"byte_count": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"byte_count": 0},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_byte": -1},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_line": 2, "byte_count": 10},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"byte_count": 10, "start_byte": 2},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tail",
                    command="tail",
                    options={"start_line": 2, "lines": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.cat",
                    command="cat",
                    options={"number": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.file",
                    command="file",
                    options={"mime": True},
                    paths=(_path("missing.txt", kind="file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.file",
                    command="file",
                    options={"brief": 1},
                    paths=(_path("missing.txt", kind="file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(tool_name="shell.file", command="file"),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"expression": "-delete"},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"max_depth": True},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"max_depth": 11},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"min_depth": True},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"min_depth": -1},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"min_depth": 11},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"max_depth": 3, "min_depth": 2},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"max_depth": 1, "min_depth": 2},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"entry_type": "symlink"},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"name": 1},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"name": ""},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"name": "nested/file.txt"},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"name": "file*"},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"name": "bad\nname"},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"name": "a" * 256},
                ),
                ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            ),
            (
                _request(
                    tool_name="shell.find",
                    command="find",
                    options={"name": ".hidden"},
                ),
                ShellExecutionErrorCode.HIDDEN_PATH,
            ),
            (
                _request(
                    tool_name="shell.ls",
                    command="ls",
                    options={"show_hidden": True},
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.ls",
                    command="ls",
                    paths=(
                        _path("first.txt", kind="file"),
                        _path("second.txt", kind="file"),
                    ),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.wc",
                    command="wc",
                    options={"lines": 1},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.wc",
                    command="wc",
                    options={"characters": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(tool_name="shell.wc", command="wc"),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
        )

        for request, error_code in cases:
            with self.subTest(request=request):
                await self._assert_denied(
                    request,
                    error_code,
                    policy=policy,
                )
        await self._assert_denied(
            _request(
                tool_name="shell.find",
                command="find",
                options={"name": ".env"},
            ),
            ShellExecutionErrorCode.SENSITIVE_PATH,
            policy=ExecutionPolicy(
                settings=ShellToolSettings(allow_hidden=True),
                resolver=resolver,
            ),
        )
        self.assertEqual(resolver.calls, ())

    async def test_core_commands_reject_path_mismatches(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "directory").mkdir()
            (root / "file.txt").write_text("value", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            await self._assert_denied(
                _request(
                    tool_name="shell.head",
                    command="head",
                    paths=(_path("directory", kind="directory"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
                policy=policy,
            )
            await self._assert_denied(
                _request(
                    tool_name="shell.cat",
                    command="cat",
                    paths=(_path("directory"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )
            await self._assert_denied(
                _request(
                    tool_name="shell.ls",
                    command="ls",
                    paths=(_path("file.txt", kind="text_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
                policy=policy,
            )
            await self._assert_denied(
                _request(
                    tool_name="shell.file",
                    command="file",
                    paths=(_path("directory", kind="file"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )
            await self._assert_denied(
                _request(
                    tool_name="shell.find",
                    command="find",
                    paths=(_path("file.txt", kind="text_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
                policy=policy,
            )
            await self._assert_denied(
                _request(
                    tool_name="shell.wc",
                    command="wc",
                    options={
                        "lines": False,
                        "words": False,
                        "count_bytes": True,
                    },
                    paths=(_path("directory", kind="file"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )

    async def test_ls_listing_preflight_rejects_unavailable_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            await self._assert_denied(
                _request(
                    tool_name="shell.ls",
                    command="ls",
                    paths=(_path("missing.txt", kind="file"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )
            await self._assert_denied(
                _request(
                    tool_name="shell.find",
                    command="find",
                    paths=(_path("missing.txt", kind="file"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )

    async def test_ls_listing_preflight_rejects_metadata_failures(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            file_path = root / "file.txt"
            file_path.write_text("value", encoding="utf-8")
            file_metadata = ShellPathMetadata(
                path=file_path,
                resolved_path=file_path,
                mode=0,
                size=5,
                is_file=True,
                is_directory=False,
                is_symlink=False,
                is_special_file=False,
            )
            special_metadata = ShellPathMetadata(
                path=file_path,
                resolved_path=file_path,
                mode=0,
                size=0,
                is_file=False,
                is_directory=False,
                is_symlink=False,
                is_special_file=True,
            )
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            with patch(
                "avalan.tool.shell.policy.inspect_path",
                new=AsyncMock(
                    side_effect=(file_metadata, PermissionError("denied")),
                ),
            ):
                await self._assert_denied(
                    _request(
                        tool_name="shell.ls",
                        command="ls",
                        paths=(_path("file.txt", kind="file"),),
                    ),
                    ShellExecutionErrorCode.DENIED_PATH,
                    policy=policy,
                )

            with patch(
                "avalan.tool.shell.policy.inspect_path",
                new=AsyncMock(side_effect=(file_metadata, special_metadata)),
            ):
                await self._assert_denied(
                    _request(
                        tool_name="shell.ls",
                        command="ls",
                        paths=(_path("file.txt", kind="file"),),
                    ),
                    ShellExecutionErrorCode.DENIED_PATH,
                    policy=policy,
                )

    async def test_denies_paths_outside_workspace_without_host_leak(
        self,
    ) -> None:
        workspace_root = Path.cwd().resolve() / "src"
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                workspace_root=str(workspace_root),
                allow_absolute_paths=True,
            )
        )
        path = PathOperand(
            name="input",
            path=str(Path.cwd().resolve() / "pyproject.toml"),
            kind="text_file",
            access="read",
        )

        with self.assertRaises(ShellPolicyDenied) as context:
            await policy.normalize(_request(paths=(path,)))

        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.DENIED_PATH,
        )
        self.assertNotIn(str(Path.cwd().resolve()), str(context.exception))

    async def test_denies_path_count_and_deep_traversal_before_resolver(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/rg")
        policy = ExecutionPolicy(
            settings=ShellToolSettings(max_path_count=1),
            resolver=resolver,
        )
        first_path = _path("first.txt")
        second_path = _path("second.txt")

        await self._assert_denied(
            _request(paths=(first_path, second_path)),
            ShellExecutionErrorCode.DENIED_PATH,
            policy=policy,
        )
        await self._assert_denied(
            _request(paths=(_path("a/" * 128 + "../blocked.txt"),)),
            ShellExecutionErrorCode.TRAVERSAL,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ())

    async def test_denies_hidden_paths_unless_trusted(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".hidden.txt").write_text("hidden", encoding="utf-8")
            request = _request(paths=(_path(".hidden.txt"),))

            await self._assert_denied(
                request,
                ShellExecutionErrorCode.HIDDEN_PATH,
                policy=ExecutionPolicy(
                    settings=ShellToolSettings(workspace_root=str(root)),
                ),
            )
            spec = await ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                ),
            ).normalize(request)

        self.assertEqual(Path(spec.cwd), root.resolve())

    async def test_denies_sensitive_paths_without_leaking_names(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                )
            )

            with self.assertRaises(ShellPolicyDenied) as context:
                await policy.normalize(_request(paths=(_path(".env"),)))

        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.SENSITIVE_PATH,
        )
        self.assertNotIn(".env", str(context.exception))
        self.assertNotIn("TOKEN", str(context.exception))

    async def test_denies_sensitive_nested_patterns(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "secrets"
            nested.mkdir()
            (nested / "deploy.pem").write_text("secret", encoding="utf-8")
            (nested / "id_ed25519").write_text("secret", encoding="utf-8")
            (root / "id_ecdsa").write_text("secret", encoding="utf-8")
            (root / "id_dsa").write_text("secret", encoding="utf-8")
            (root / "id_report.txt").write_text("safe", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            for path in (
                "secrets/deploy.pem",
                "secrets/id_ed25519",
                "id_ecdsa",
                "id_dsa",
            ):
                with self.subTest(path=path):
                    await self._assert_denied(
                        _request(paths=(_path(path),)),
                        ShellExecutionErrorCode.SENSITIVE_PATH,
                        policy=policy,
                    )

            spec = await policy.normalize(
                _request(paths=(_path("id_report.txt"),))
            )

        self.assertEqual(spec.display_argv[-1], "id_report.txt")

    async def test_denies_sensitive_directory_descendants(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            credential_store = root / "credentials"
            key_store = root / "archive.key"
            credential_store.mkdir()
            key_store.mkdir()
            (credential_store / "token.txt").write_text(
                "secret", encoding="utf-8"
            )
            (key_store / "public.txt").write_text("secret", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            for path in (
                "credentials/token.txt",
                "archive.key/public.txt",
            ):
                with self.subTest(path=path):
                    await self._assert_denied(
                        _request(paths=(_path(path),)),
                        ShellExecutionErrorCode.SENSITIVE_PATH,
                        policy=policy,
                    )

    async def test_denies_sensitive_env_files_when_hidden_is_trusted(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "app"
            nested.mkdir()
            (root / ".envrc").write_text("TOKEN=secret", encoding="utf-8")
            (nested / ".env.local").write_text(
                "TOKEN=secret", encoding="utf-8"
            )
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                ),
            )

            for path in (".envrc", "app/.env.local"):
                with self.subTest(path=path):
                    await self._assert_denied(
                        _request(paths=(_path(path),)),
                        ShellExecutionErrorCode.SENSITIVE_PATH,
                        policy=policy,
                    )

    async def test_denies_case_variant_sensitive_paths(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "ID_ED25519").write_text("secret", encoding="utf-8")
            (root / ".GIT").mkdir()
            (root / ".GIT" / "config").write_text(
                "secret",
                encoding="utf-8",
            )
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                ),
            )

            for path in ("ID_ED25519", ".GIT/config"):
                with self.subTest(path=path):
                    await self._assert_denied(
                        _request(paths=(_path(path),)),
                        ShellExecutionErrorCode.SENSITIVE_PATH,
                        policy=policy,
                    )

    async def test_denies_symlink_path_components_by_default(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target"
            target.mkdir()
            (target / "visible.txt").write_text("value", encoding="utf-8")
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symlink unavailable: {error}")

            await self._assert_denied(
                _request(paths=(_path("link/visible.txt"),)),
                ShellExecutionErrorCode.SYMLINK,
                policy=ExecutionPolicy(
                    settings=ShellToolSettings(workspace_root=str(root)),
                ),
            )
            spec = await ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_symlinks=True,
                ),
            ).normalize(_request(paths=(_path("link/visible.txt"),)))

        self.assertEqual(spec.display_cwd, ".")

    async def test_denies_external_symlink_before_following_target(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace_directory:
            with TemporaryDirectory() as external_directory:
                root = Path(workspace_directory)
                external = Path(external_directory)
                (external / "visible.txt").write_text(
                    "value",
                    encoding="utf-8",
                )
                link = root / "link"
                try:
                    link.symlink_to(external, target_is_directory=True)
                except OSError as error:
                    self.skipTest(f"symlink unavailable: {error}")

                policy = ExecutionPolicy(
                    settings=ShellToolSettings(workspace_root=str(root)),
                )

                with self.assertRaises(ShellPolicyDenied) as context:
                    await policy.normalize(
                        _request(paths=(_path("link/visible.txt"),))
                    )

                self.assertEqual(
                    context.exception.error_code,
                    ShellExecutionErrorCode.SYMLINK,
                )
                self.assertNotIn(str(external), str(context.exception))

    async def test_trusted_external_symlink_remains_workspace_denied(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace_directory:
            with TemporaryDirectory() as external_directory:
                root = Path(workspace_directory)
                external = Path(external_directory)
                (external / "visible.txt").write_text(
                    "value",
                    encoding="utf-8",
                )
                link = root / "link"
                try:
                    link.symlink_to(external, target_is_directory=True)
                except OSError as error:
                    self.skipTest(f"symlink unavailable: {error}")

                policy = ExecutionPolicy(
                    settings=ShellToolSettings(
                        workspace_root=str(root),
                        allow_symlinks=True,
                    ),
                )

                with self.assertRaises(ShellPolicyDenied) as context:
                    await policy.normalize(
                        _request(paths=(_path("link/visible.txt"),))
                    )

                self.assertEqual(
                    context.exception.error_code,
                    ShellExecutionErrorCode.DENIED_PATH,
                )
                self.assertNotIn(str(external), str(context.exception))

    async def test_denies_special_files_and_unavailable_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "fifo"
            metadata = ShellPathMetadata(
                path=path,
                resolved_path=path,
                mode=0,
                size=0,
                is_file=False,
                is_directory=False,
                is_symlink=False,
                is_special_file=True,
            )

            with patch(
                "avalan.tool.shell.policy.inspect_path",
                new=AsyncMock(return_value=metadata),
            ):
                await self._assert_denied(
                    _request(paths=(_path("fifo"),)),
                    ShellExecutionErrorCode.SPECIAL_FILE,
                    policy=ExecutionPolicy(
                        settings=ShellToolSettings(workspace_root=str(root)),
                    ),
                )

            with patch(
                "avalan.tool.shell.policy.inspect_path",
                new=AsyncMock(side_effect=PermissionError("denied")),
            ):
                await self._assert_denied(
                    _request(paths=(_path("locked.txt"),)),
                    ShellExecutionErrorCode.DENIED_PATH,
                    policy=ExecutionPolicy(
                        settings=ShellToolSettings(workspace_root=str(root)),
                    ),
                )

    async def test_denies_virtual_filesystem_paths(self) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                workspace_root="/",
                allow_absolute_paths=True,
                allow_hidden=True,
            )
        )

        await self._assert_denied(
            _request(paths=(_path("/dev/null"),)),
            ShellExecutionErrorCode.SPECIAL_FILE,
            policy=policy,
        )

    async def test_denies_media_command_when_media_tools_disabled(
        self,
    ) -> None:
        for command in ("pdfinfo", "pdftotext"):
            with self.subTest(command=command):
                await self._assert_denied(
                    _request(tool_name=f"shell.{command}", command=command),
                    ShellExecutionErrorCode.DENIED_COMMAND,
                )

    async def test_media_commands_build_constrained_argv(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        resolver = _CountingResolver("/usr/bin/media")
        settings = ShellToolSettings(
            workspace_root=str(fixture_root),
            allow_media_tools=True,
            allowed_tesseract_languages=("eng", "spa"),
            default_pdf_timeout_seconds=7.0,
            max_pdf_timeout_seconds=9.0,
            default_ocr_timeout_seconds=11.0,
            max_ocr_timeout_seconds=13.0,
        )
        policy = ExecutionPolicy(settings=settings, resolver=resolver)

        info = await policy.normalize(
            _request(
                tool_name="shell.pdfinfo",
                command="pdfinfo",
                options={
                    "first_page": 1,
                    "last_page": 2,
                    "boxes": True,
                    "iso_dates": True,
                },
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        info_default = await policy.normalize(
            _request(
                tool_name="shell.pdfinfo",
                command="pdfinfo",
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        info_boxes_default = await policy.normalize(
            _request(
                tool_name="shell.pdfinfo",
                command="pdfinfo",
                options={"boxes": True},
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        info_last_only = await policy.normalize(
            _request(
                tool_name="shell.pdfinfo",
                command="pdfinfo",
                options={"last_page": 1},
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        info_first_only = await policy.normalize(
            _request(
                tool_name="shell.pdfinfo",
                command="pdfinfo",
                options={"first_page": 2},
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        text = await policy.normalize(
            _request(
                tool_name="shell.pdftotext",
                command="pdftotext",
                options={
                    "first_page": 2,
                    "layout": True,
                    "no_page_breaks": True,
                },
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        raster = await policy.normalize(
            _request(
                tool_name="shell.pdftoppm",
                command="pdftoppm",
                options={
                    "first_page": 1,
                    "last_page": 2,
                    "dpi": 72,
                    "grayscale": True,
                },
                paths=(_path("media/small.pdf", kind="pdf_file"),),
                timeout_seconds=99.0,
            )
        )
        ocr = await policy.normalize(
            _request(
                tool_name="shell.tesseract",
                command="tesseract",
                options={
                    "languages": ("eng", "spa"),
                    "psm": 6,
                    "oem": 1,
                    "dpi": 300,
                },
                paths=(_path("ocr/small.pgm", kind="image_file"),),
            )
        )

        self.assertEqual(
            info.argv,
            (
                "pdfinfo",
                "-f",
                "1",
                "-l",
                "2",
                "-box",
                "-isodates",
                "media/small.pdf",
            ),
        )
        self.assertEqual(info.resource_class, "heavy")
        self.assertEqual(info.timeout_seconds, 7.0)
        self.assertEqual(info.metadata["page_range"], {"first": 1, "last": 2})
        self.assertEqual(info_default.argv, ("pdfinfo", "media/small.pdf"))
        self.assertNotIn("page_range", info_default.metadata)
        self.assertEqual(
            info_boxes_default.argv,
            ("pdfinfo", "-f", "1", "-l", "1", "-box", "media/small.pdf"),
        )
        self.assertEqual(
            info_last_only.argv,
            ("pdfinfo", "-f", "1", "-l", "1", "media/small.pdf"),
        )
        self.assertEqual(
            info_first_only.argv,
            ("pdfinfo", "-f", "2", "-l", "2", "media/small.pdf"),
        )
        self.assertEqual(
            text.argv,
            (
                "pdftotext",
                "-f",
                "2",
                "-l",
                "2",
                "-layout",
                "-nopgbrk",
                "media/small.pdf",
                "-",
            ),
        )
        self.assertEqual(text.resource_class, "heavy")
        self.assertEqual(text.timeout_seconds, 7.0)
        self.assertEqual(text.metadata["page_range"], {"first": 2, "last": 2})
        self.assertEqual(
            raster.argv[:10],
            (
                "pdftoppm",
                "-f",
                "1",
                "-l",
                "2",
                "-r",
                "72",
                "-gray",
                "-png",
                "media/small.pdf",
            ),
        )
        self.assertEqual(
            raster.argv[10],
            GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
        )
        self.assertEqual(raster.display_argv[10], "GENERATED_PREFIX")
        self.assertIsNotNone(raster.output_plan)
        self.assertEqual(raster.output_plan.prefix_name, "page")
        self.assertEqual(raster.output_kind, ShellOutputKind.GENERATED_FILES)
        self.assertEqual(raster.stdout_media_type, "application/json")
        self.assertEqual(raster.timeout_seconds, 9.0)
        self.assertEqual(raster.metadata["dpi"], 72)
        self.assertEqual(
            raster.metadata["generated_output_display_prefix"],
            "GENERATED_PREFIX",
        )
        self.assertEqual(
            ocr.argv,
            (
                "tesseract",
                "ocr/small.pgm",
                "stdout",
                "-l",
                "eng+spa",
                "--psm",
                "6",
                "--oem",
                "1",
                "--dpi",
                "300",
            ),
        )
        self.assertEqual(ocr.timeout_seconds, 11.0)
        self.assertEqual(ocr.metadata["ocr_languages"], ("eng", "spa"))
        self.assertEqual(ocr.metadata["ocr_thread_limit"], 1)
        self.assertEqual(
            resolver.calls,
            (
                "pdfinfo",
                "pdfinfo",
                "pdfinfo",
                "pdfinfo",
                "pdfinfo",
                "pdftotext",
                "pdftoppm",
                "tesseract",
            ),
        )

    async def test_media_paths_without_double_dash_are_disambiguated(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "-dash.pdf").write_bytes(b"%PDF-1.4\n")
            (root / "-dash.pgm").write_bytes(b"P2\n1 1\n255\n0\n")
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
            )
            policy = ExecutionPolicy(settings=settings)

            pdf = await policy.normalize(
                _request(
                    tool_name="shell.pdftotext",
                    command="pdftotext",
                    paths=(_path("-dash.pdf", kind="pdf_file"),),
                )
            )
            info = await policy.normalize(
                _request(
                    tool_name="shell.pdfinfo",
                    command="pdfinfo",
                    options={"boxes": True},
                    paths=(_path("-dash.pdf", kind="pdf_file"),),
                )
            )
            ocr = await policy.normalize(
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    paths=(_path("-dash.pgm", kind="image_file"),),
                )
            )

        self.assertEqual(pdf.argv[-2], "./-dash.pdf")
        self.assertEqual(pdf.display_argv[-2], "./-dash.pdf")
        self.assertEqual(info.argv[-1], "./-dash.pdf")
        self.assertEqual(info.metadata["page_range"], {"first": 1, "last": 1})
        self.assertEqual(ocr.argv[1], "./-dash.pgm")

    async def test_denies_binary_text_inputs_before_resolver(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures" / "filesystem"
        resolver = _CountingResolver("/usr/bin/cat")
        binary_path = _path("binary.bin")
        invalid_utf8_path = _path("invalid-utf8.bin")
        cases = (
            ("cat", {}, binary_path),
            ("head", {}, binary_path),
            ("tail", {}, binary_path),
            ("wc", {"lines": True}, binary_path),
            ("wc", {"words": True}, binary_path),
            ("awk", {}, binary_path),
            ("sed", {"line_ranges": ("1",)}, binary_path),
            ("cat", {}, invalid_utf8_path),
        )

        for command, options, path in cases:
            with self.subTest(command=command, options=options, path=path):
                await self._assert_denied(
                    _request(
                        tool_name=f"shell.{command}",
                        command=command,
                        options=options,
                        paths=(path,),
                    ),
                    ShellExecutionErrorCode.BINARY_CONTENT,
                    policy=ExecutionPolicy(
                        settings=ShellToolSettings(
                            workspace_root=str(fixture_root),
                        ),
                        resolver=resolver,
                    ),
                )

        self.assertEqual(resolver.calls, ())

    async def test_full_file_size_applies_only_to_cat_and_wc_text_modes(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            large_path = root / "large.txt"
            large_path.write_text("abcdef", encoding="utf-8")
            settings = ShellToolSettings(
                workspace_root=str(root),
                max_full_file_bytes=5,
            )
            denied_cases = (
                ("cat", {}, ShellExecutionErrorCode.TOO_LARGE),
                ("wc", {"lines": True}, ShellExecutionErrorCode.TOO_LARGE),
                ("wc", {"words": True}, ShellExecutionErrorCode.TOO_LARGE),
                ("wc", {}, ShellExecutionErrorCode.TOO_LARGE),
            )

            for command, options, error_code in denied_cases:
                with self.subTest(command=command, options=options):
                    await self._assert_denied(
                        _request(
                            tool_name=f"shell.{command}",
                            command=command,
                            options=options,
                            paths=(_path("large.txt"),),
                        ),
                        error_code,
                        policy=ExecutionPolicy(settings=settings),
                    )

            resolver = _CountingResolver("/usr/bin/head")
            for command, options in (
                ("head", {}),
                ("tail", {}),
                ("wc", {"lines": False, "words": False, "count_bytes": True}),
            ):
                with self.subTest(command=command, options=options):
                    spec = await ExecutionPolicy(
                        settings=settings,
                        resolver=resolver,
                    ).normalize(
                        _request(
                            tool_name=f"shell.{command}",
                            command=command,
                            options=options,
                            paths=(_path("large.txt"),),
                        )
                    )
                    self.assertEqual(spec.command, command)

        self.assertEqual(resolver.calls, ("head", "tail", "wc"))

    async def test_head_allows_two_hundred_lines_by_default(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "file.txt").write_text("first\nsecond\n", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root))
            )

            head = await policy.normalize(
                _request(
                    tool_name="shell.head",
                    command="head",
                    options={"lines": 200},
                    paths=(_path("file.txt"),),
                )
            )

        self.assertEqual(head.argv, ("head", "-n", "200", "--", "file.txt"))
        self.assertEqual(
            head.display_argv,
            ("head", "-n", "200", "--", "file.txt"),
        )

    async def test_wc_count_bytes_allows_binary_large_files(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures" / "filesystem"
        resolver = _CountingResolver("/usr/bin/wc")

        spec = await ExecutionPolicy(
            settings=ShellToolSettings(
                workspace_root=str(fixture_root),
                max_full_file_bytes=1,
            ),
            resolver=resolver,
        ).normalize(
            _request(
                tool_name="shell.wc",
                command="wc",
                options={"lines": False, "words": False, "count_bytes": True},
                paths=(_path("binary.bin"),),
            )
        )

        self.assertEqual(spec.command, "wc")
        self.assertEqual(resolver.calls, ("wc",))

    async def test_file_allows_binary_regular_files(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures" / "filesystem"
        resolver = _CountingResolver("/usr/bin/file")

        spec = await ExecutionPolicy(
            settings=ShellToolSettings(workspace_root=str(fixture_root)),
            resolver=resolver,
        ).normalize(
            _request(
                tool_name="shell.file",
                command="file",
                paths=(_path("binary.bin", kind="file"),),
            )
        )

        self.assertEqual(spec.command, "file")
        self.assertEqual(spec.argv, ("file", "--", "binary.bin"))
        self.assertEqual(resolver.calls, ("file",))

    async def test_content_inputs_must_be_available_regular_files(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "directory").mkdir()
            file_path = root / "visible.txt"
            file_path.write_text("text", encoding="utf-8")
            metadata = ShellPathMetadata(
                path=file_path,
                resolved_path=file_path,
                mode=0,
                size=4,
                is_file=True,
                is_directory=False,
                is_symlink=False,
                is_special_file=False,
            )
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )

            await self._assert_denied(
                _request(
                    tool_name="shell.cat",
                    command="cat",
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )
            await self._assert_denied(
                _request(
                    tool_name="shell.cat",
                    command="cat",
                    paths=(_path("directory"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
                policy=policy,
            )
            with patch(
                "avalan.tool.shell.policy.inspect_path",
                new=AsyncMock(
                    side_effect=(
                        metadata,
                        PermissionError("denied"),
                    )
                ),
            ):
                await self._assert_denied(
                    _request(
                        tool_name="shell.cat",
                        command="cat",
                        paths=(_path("visible.txt"),),
                    ),
                    ShellExecutionErrorCode.DENIED_PATH,
                    policy=policy,
                )

    async def test_awk_builds_constrained_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "table.tsv").write_text(
                "name\tstatus\none\tactive\n",
                encoding="utf-8",
            )
            (root / "-dash.tsv").write_text("two\tactive\n", encoding="utf-8")
            (root / "a=b.tsv").write_text("three\tactive\n", encoding="utf-8")
            settings = ShellToolSettings(
                workspace_root=str(root),
                max_filter_selectors=8,
            )

            projected = await ExecutionPolicy(settings=settings).normalize(
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={
                        "fields": (2, 1),
                        "field_separator": "tab",
                        "output_separator": "|",
                        "pattern": "active",
                        "start_line": 2,
                        "end_line": 4,
                    },
                    paths=(
                        _path("table.tsv"),
                        _path("-dash.tsv"),
                        _path("a=b.tsv"),
                    ),
                )
            )
            default_program = await ExecutionPolicy(
                settings=settings
            ).normalize(
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    paths=(_path("table.tsv"),),
                )
            )
            explicit_default_program = await ExecutionPolicy(
                settings=settings
            ).normalize(
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"output_separator": " "},
                    paths=(_path("table.tsv"),),
                )
            )

        self.assertEqual(
            projected.argv,
            (
                "awk",
                "-F",
                "\\t",
                "-v",
                "OFS=|",
                "-v",
                "pat=active",
                "NR >= 2 && NR <= 4 && $0 ~ pat { print $2, $1 }",
                "table.tsv",
                "./-dash.tsv",
                "./a=b.tsv",
            ),
        )
        self.assertEqual(
            projected.display_argv[-2:], ("./-dash.tsv", "./a=b.tsv")
        )
        self.assertEqual(
            default_program.argv,
            ("awk", "-v", "OFS= ", "{ print $0 }", "table.tsv"),
        )
        self.assertEqual(
            explicit_default_program.argv,
            ("awk", "-v", "OFS= ", "{ print $0 }", "table.tsv"),
        )

    async def test_sed_builds_print_only_selectors(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "lines.txt").write_text(
                "one\ntwo\nthree\n", encoding="utf-8"
            )
            (root / "-dash.txt").write_text("dash\n", encoding="utf-8")

            spec = await ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            ).normalize(
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={
                        "line_ranges": ("1", "2,3"),
                        "patterns": ("two/three", "a.*b["),
                    },
                    paths=(_path("lines.txt"), _path("-dash.txt")),
                )
            )

        self.assertEqual(
            spec.argv,
            (
                "sed",
                "-n",
                "-e",
                "1p",
                "-e",
                "2,3p",
                "-e",
                "/two\\/three/p",
                "-e",
                "/a\\.\\*b\\[/p",
                "lines.txt",
                "./-dash.txt",
            ),
        )
        self.assertEqual(spec.display_argv, spec.argv)

    async def test_sed_builds_native_line_window_selectors(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "lines.txt").write_text(
                "one\ntwo\nthree\nfour\n", encoding="utf-8"
            )
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )
            cases = (
                (
                    "start-only",
                    {"start_line": 2},
                    ("sed", "-n", "-e", "2,$p", "lines.txt"),
                ),
                (
                    "end-only",
                    {"end_line": 3},
                    ("sed", "-n", "-e", "1,3p", "lines.txt"),
                ),
                (
                    "bounded",
                    {"start_line": 2, "end_line": 3},
                    ("sed", "-n", "-e", "2,3p", "lines.txt"),
                ),
                (
                    "combined",
                    {
                        "line_ranges": ("1",),
                        "patterns": ("four",),
                        "start_line": 2,
                        "end_line": 3,
                    },
                    (
                        "sed",
                        "-n",
                        "-e",
                        "1p",
                        "-e",
                        "2,3p",
                        "-e",
                        "/four/p",
                        "lines.txt",
                    ),
                ),
            )

            for name, options, argv in cases:
                with self.subTest(name=name):
                    spec = await policy.normalize(
                        _request(
                            tool_name="shell.sed",
                            command="sed",
                            options=options,
                            paths=(_path("lines.txt"),),
                        )
                    )
                    self.assertEqual(spec.argv, argv)
                    self.assertEqual(spec.display_argv, argv)

    async def test_jq_builds_flags_and_output_contract(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(workspace_root=str(fixture_root))
        resolver = _CountingResolver("/usr/bin/jq")
        policy = ExecutionPolicy(settings=settings, resolver=resolver)

        json_spec = await policy.normalize(
            _request(
                tool_name="shell.jq",
                command="jq",
                options={
                    "filter": "-.items[]",
                    "compact": True,
                    "slurp": True,
                    "sort_keys": True,
                },
                paths=(_path("json/valid.json", kind="json_file"),),
            )
        )
        text_spec = await policy.normalize(
            _request(
                tool_name="shell.jq",
                command="jq",
                options={"filter": ".name", "raw_output": True},
                paths=(_path("json/valid.json", kind="json_file"),),
            )
        )

        self.assertEqual(
            json_spec.argv,
            (
                "jq",
                "--compact-output",
                "--slurp",
                "--sort-keys",
                "--",
                "-.items[]",
                "json/valid.json",
            ),
        )
        self.assertEqual(json_spec.output_kind, ShellOutputKind.JSON)
        self.assertEqual(json_spec.stdout_media_type, "application/json")
        self.assertEqual(
            text_spec.argv,
            ("jq", "--raw-output", "--", ".name", "json/valid.json"),
        )
        self.assertEqual(text_spec.output_kind, ShellOutputKind.TEXT)
        self.assertEqual(text_spec.stdout_media_type, "text/plain")
        self.assertEqual(resolver.calls, ("jq", "jq"))

    async def test_filter_commands_reject_invalid_options_before_resolver(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/filter")
        settings = ShellToolSettings(
            max_awk_fields=2,
            max_awk_separator_bytes=2,
            max_filter_pattern_bytes=4,
            max_filter_program_bytes=8,
            max_filter_selectors=1,
            max_jq_filter_bytes=4,
        )
        cases = (
            (
                _request(tool_name="shell.awk", command="awk"),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"fields": "1"},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"fields": ()},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"fields": (0,)},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"fields": (True,)},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"field_separator": "regex"},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"output_separator": ""},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"output_separator": "\n"},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"pattern": " "},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"pattern": "abcde"},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"start_line": 3, "end_line": 2},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    options={"fields": (1, 2)},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"line_ranges": "1"},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"line_ranges": ("0",)},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"line_ranges": ("2,1",)},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"start_line": True},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"start_line": 0},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"end_line": -1},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"start_line": 3, "end_line": 2},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"patterns": ("a",), "start_line": 1},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"patterns": ("a;b",)},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"patterns": ("abcde",)},
                    paths=(_path("missing.txt"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
            ),
            (
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"line_ranges": ("1",)},
                    paths=(_path("-", kind="text_file"),),
                ),
                ShellExecutionErrorCode.DENIED_PATH,
            ),
            (
                _request(tool_name="shell.jq", command="jq"),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.jq",
                    command="jq",
                    options={"filter": ".", "raw_output": 1},
                    paths=(_path("missing.json", kind="json_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.jq",
                    command="jq",
                    options={"filter": " "},
                    paths=(_path("missing.json", kind="json_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.jq",
                    command="jq",
                    options={"filter": "abcde"},
                    paths=(_path("missing.json", kind="json_file"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
            ),
        )

        for request, error_code in cases:
            with self.subTest(request=request):
                await self._assert_denied(
                    request,
                    error_code,
                    policy=ExecutionPolicy(
                        settings=settings, resolver=resolver
                    ),
                )
        self.assertEqual(resolver.calls, ())

    async def test_jq_lexical_validator_blocks_unsafe_features(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        resolver = _CountingResolver("/usr/bin/jq")
        policy = ExecutionPolicy(
            settings=ShellToolSettings(workspace_root=str(fixture_root)),
            resolver=resolver,
        )

        allowed = await policy.normalize(
            _request(
                tool_name="shell.jq",
                command="jq",
                options={
                    "filter": '.env | "import env input $ENV" # import env'
                },
                paths=(_path("json/valid.json", kind="json_file"),),
            )
        )

        self.assertEqual(allowed.command, "jq")
        for jq_filter in (
            'import "module" as m; .',
            'include "module";',
            'module {name:"x"};',
            "env.PATH",
            "$ENV.PATH",
            "input",
            "inputs",
            '"unterminated',
            ".\x01",
        ):
            with self.subTest(jq_filter=jq_filter):
                await self._assert_denied(
                    _request(
                        tool_name="shell.jq",
                        command="jq",
                        options={"filter": jq_filter},
                        paths=(_path("json/valid.json", kind="json_file"),),
                    ),
                    (
                        ShellExecutionErrorCode.UNSUPPORTED_JQ_FEATURE
                        if jq_filter
                        not in (
                            '"unterminated',
                            ".\x01",
                        )
                        else ShellExecutionErrorCode.UNSAFE_FILTER
                    ),
                    policy=policy,
                )
        self.assertEqual(resolver.calls, ("jq",))

    async def test_filter_input_caps_are_totaled_before_resolver(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "first.txt").write_text("abc", encoding="utf-8")
            (root / "second.txt").write_text("def", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    max_text_filter_input_bytes=5,
                ),
                resolver=_CountingResolver("/usr/bin/awk"),
            )

            await self._assert_denied(
                _request(
                    tool_name="shell.awk",
                    command="awk",
                    paths=(_path("first.txt"), _path("second.txt")),
                ),
                ShellExecutionErrorCode.TOO_LARGE,
                policy=policy,
            )

    async def test_sed_selector_count_diagnostic_includes_limit(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "input.txt").write_text("abc\n", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    max_filter_selectors=1,
                )
            )

            await self._assert_denied(
                _request(
                    tool_name="shell.sed",
                    command="sed",
                    options={"line_ranges": ("1", "2")},
                    paths=(_path("input.txt"),),
                ),
                ShellExecutionErrorCode.UNSAFE_FILTER,
                policy=policy,
                message="sed selector count is too large (2 > 1)",
            )

    async def test_filter_validator_boundary_branches(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "input.txt").write_text("abc\n", encoding="utf-8")
            (root / "input.json").write_text(
                '{"env": "safe"}\n', encoding="utf-8"
            )
            wide_program_settings = ShellToolSettings(
                workspace_root=str(root),
                max_filter_program_bytes=128,
            )
            cases = (
                (
                    ExecutionPolicy(settings=wide_program_settings),
                    _request(
                        tool_name="shell.awk",
                        command="awk",
                        options={"output_separator": []},
                        paths=(_path("input.txt"),),
                    ),
                    ShellExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    ExecutionPolicy(
                        settings=ShellToolSettings(
                            workspace_root=str(root),
                            max_awk_separator_bytes=2,
                        )
                    ),
                    _request(
                        tool_name="shell.awk",
                        command="awk",
                        options={"output_separator": "abc"},
                        paths=(_path("input.txt"),),
                    ),
                    ShellExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    ExecutionPolicy(settings=wide_program_settings),
                    _request(
                        tool_name="shell.awk",
                        command="awk",
                        options={"pattern": []},
                        paths=(_path("input.txt"),),
                    ),
                    ShellExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    ExecutionPolicy(
                        settings=ShellToolSettings(
                            workspace_root=str(root),
                            max_filter_program_bytes=128,
                            max_filter_selectors=1,
                        )
                    ),
                    _request(
                        tool_name="shell.awk",
                        command="awk",
                        options={"fields": (1, 2)},
                        paths=(_path("input.txt"),),
                    ),
                    ShellExecutionErrorCode.UNSAFE_FILTER,
                ),
                (
                    ExecutionPolicy(
                        settings=ShellToolSettings(
                            workspace_root=str(root),
                            max_filter_selectors=1,
                        )
                    ),
                    _request(
                        tool_name="shell.sed",
                        command="sed",
                        options={"line_ranges": ("1", "2")},
                        paths=(_path("input.txt"),),
                    ),
                    ShellExecutionErrorCode.UNSAFE_FILTER,
                ),
                (
                    ExecutionPolicy(settings=wide_program_settings),
                    _request(
                        tool_name="shell.sed",
                        command="sed",
                        options={"line_ranges": ([],)},
                        paths=(_path("input.txt"),),
                    ),
                    ShellExecutionErrorCode.INVALID_OPTION,
                ),
                (
                    ExecutionPolicy(
                        settings=ShellToolSettings(
                            workspace_root=str(root),
                            max_filter_pattern_bytes=2,
                        )
                    ),
                    _request(
                        tool_name="shell.sed",
                        command="sed",
                        options={"line_ranges": ("123",)},
                        paths=(_path("input.txt"),),
                    ),
                    ShellExecutionErrorCode.UNSAFE_FILTER,
                ),
            )

            for policy, request, error_code in cases:
                with self.subTest(request=request):
                    await self._assert_denied(
                        request,
                        error_code,
                        policy=policy,
                    )

            jq_policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
            )
            for jq_filter in (
                "$value",
                '"escaped \\" value"',
                ". \n env",
            ):
                with self.subTest(jq_filter=jq_filter):
                    spec = await jq_policy.normalize(
                        _request(
                            tool_name="shell.jq",
                            command="jq",
                            options={"filter": jq_filter},
                            paths=(_path("input.json", kind="json_file"),),
                        )
                    )
                    self.assertEqual(spec.command, "jq")

    async def test_jq_validates_json_files_before_resolver(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        resolver = _CountingResolver("/usr/bin/jq")
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                workspace_root=str(fixture_root),
                max_json_input_bytes=65,
            ),
            resolver=resolver,
        )

        spec = await policy.normalize(
            _request(
                tool_name="shell.jq",
                command="jq",
                options={"filter": "."},
                paths=(_path("json/valid.json", kind="json_file"),),
            )
        )

        self.assertEqual(spec.command, "jq")
        self.assertEqual(resolver.calls, ("jq",))
        await self._assert_denied(
            _request(
                tool_name="shell.jq",
                command="jq",
                options={"filter": "."},
                paths=(_path("filesystem/binary.bin", kind="json_file"),),
            ),
            ShellExecutionErrorCode.BINARY_CONTENT,
            policy=policy,
        )
        await self._assert_denied(
            _request(
                tool_name="shell.jq",
                command="jq",
                options={"filter": "."},
                paths=(
                    _path(
                        "json/oversized-template.json",
                        kind="json_file",
                    ),
                ),
            ),
            ShellExecutionErrorCode.TOO_LARGE,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ("jq",))

    async def test_media_inputs_use_signatures_and_size_caps(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            workspace_root=str(fixture_root),
            allow_media_tools=True,
            max_pdf_input_bytes=512,
            max_ocr_input_bytes=128,
            max_ocr_pixels=9,
            max_raster_long_edge_pixels=8,
        )
        resolver = _CountingResolver("/usr/bin/pdftotext")

        pdf_spec = await ExecutionPolicy(
            settings=settings,
            resolver=resolver,
        ).normalize(
            _request(
                tool_name="shell.pdftotext",
                command="pdftotext",
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        pdf_info_spec = await ExecutionPolicy(
            settings=settings,
            resolver=resolver,
        ).normalize(
            _request(
                tool_name="shell.pdfinfo",
                command="pdfinfo",
                paths=(_path("media/small.pdf", kind="pdf_file"),),
            )
        )
        image_spec = await ExecutionPolicy(
            settings=settings,
            resolver=resolver,
        ).normalize(
            _request(
                tool_name="shell.tesseract",
                command="tesseract",
                paths=(_path("ocr/small.pgm", kind="image_file"),),
            )
        )

        self.assertEqual(pdf_spec.command, "pdftotext")
        self.assertEqual(pdf_info_spec.command, "pdfinfo")
        self.assertEqual(image_spec.command, "tesseract")
        await self._assert_denied(
            _request(
                tool_name="shell.pdftotext",
                command="pdftotext",
                paths=(_path("ocr/small.pgm", kind="pdf_file"),),
            ),
            ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
            policy=ExecutionPolicy(settings=settings),
        )
        await self._assert_denied(
            _request(
                tool_name="shell.tesseract",
                command="tesseract",
                paths=(
                    _path("ocr/unsupported-signature.dat", kind="image_file"),
                ),
            ),
            ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
            policy=ExecutionPolicy(settings=settings),
        )
        await self._assert_denied(
            _request(
                tool_name="shell.pdftotext",
                command="pdftotext",
                paths=(
                    _path("media/oversized-template.pdf", kind="pdf_file"),
                ),
            ),
            ShellExecutionErrorCode.TOO_LARGE,
            policy=ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(fixture_root),
                    allow_media_tools=True,
                    max_pdf_input_bytes=8,
                )
            ),
        )
        await self._assert_denied(
            _request(
                tool_name="shell.tesseract",
                command="tesseract",
                paths=(_path("ocr/small.pgm", kind="image_file"),),
            ),
            ShellExecutionErrorCode.TOO_LARGE,
            policy=ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(fixture_root),
                    allow_media_tools=True,
                    max_ocr_input_bytes=8,
                )
            ),
        )
        await self._assert_denied(
            _request(
                tool_name="shell.tesseract",
                command="tesseract",
                paths=(_path("ocr/small.pgm", kind="image_file"),),
            ),
            ShellExecutionErrorCode.TOO_LARGE,
            policy=ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(fixture_root),
                    allow_media_tools=True,
                    max_ocr_pixels=8,
                    max_raster_long_edge_pixels=8,
                )
            ),
        )

    async def test_media_commands_reject_invalid_options_before_resolver(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        resolver = _CountingResolver("/usr/bin/media")
        settings = ShellToolSettings(
            workspace_root=str(fixture_root),
            allow_media_tools=True,
            allowed_tesseract_languages=("eng", "spa"),
            max_pdf_text_pages=2,
            max_pdf_raster_pages=2,
            max_pdf_raster_dpi=100,
            max_output_files=1,
            max_ocr_languages=1,
            max_tesseract_dpi=300,
        )
        cases = (
            (
                _request(
                    tool_name="shell.pdfinfo",
                    command="pdfinfo",
                    options={"js": True},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.pdfinfo",
                    command="pdfinfo",
                    options={"boxes": 1},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.pdfinfo",
                    command="pdfinfo",
                    options={"first_page": 3, "last_page": 2},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_PAGE_RANGE,
            ),
            (
                _request(
                    tool_name="shell.pdfinfo",
                    command="pdfinfo",
                    options={"last_page": 3},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.PDF_PAGE_CAP_EXCEEDED,
            ),
            (
                _request(
                    tool_name="shell.pdftotext",
                    command="pdftotext",
                    options={"output_path": "out.txt"},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.pdftotext",
                    command="pdftotext",
                    options={"first_page": 3, "last_page": 2},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_PAGE_RANGE,
            ),
            (
                _request(
                    tool_name="shell.pdftotext",
                    command="pdftotext",
                    options={"first_page": 1, "last_page": 3},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.PDF_PAGE_CAP_EXCEEDED,
            ),
            (
                _request(
                    tool_name="shell.pdftoppm",
                    command="pdftoppm",
                    options={"dpi": 101},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.RASTER_DPI_CAP_EXCEEDED,
            ),
            (
                _request(
                    tool_name="shell.pdftoppm",
                    command="pdftoppm",
                    options={"dpi": "72"},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.pdftoppm",
                    command="pdftoppm",
                    options={"dpi": 0},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.pdftoppm",
                    command="pdftoppm",
                    options={"first_page": 1, "last_page": 2},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            ),
            (
                _request(
                    tool_name="shell.pdftoppm",
                    command="pdftoppm",
                    options={"prefix": "chosen"},
                    paths=(_path("media/small.pdf", kind="pdf_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"languages": ("fra",)},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.UNSUPPORTED_OCR_LANGUAGE,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"languages": "eng"},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.UNSUPPORTED_OCR_LANGUAGE,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"languages": ("eng", "spa")},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.UNSUPPORTED_OCR_LANGUAGE,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"psm": 14},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OCR_MODE,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"oem": True},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OCR_MODE,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"config": "quiet"},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"tessdata_path": "tessdata"},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                _request(
                    tool_name="shell.tesseract",
                    command="tesseract",
                    options={"output_path": "chosen"},
                    paths=(_path("ocr/small.pgm", kind="image_file"),),
                ),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
        )

        for request, error_code in cases:
            with self.subTest(request=request):
                await self._assert_denied(
                    request,
                    error_code,
                    policy=ExecutionPolicy(
                        settings=settings,
                        resolver=resolver,
                    ),
                )
        self.assertEqual(resolver.calls, ())

    async def test_denies_argument_budget_violations(self) -> None:
        cases = (
            (
                ShellToolSettings(max_arguments=2),
                _request(options={"a": "b"}),
                ShellExecutionErrorCode.TOO_MANY_ARGUMENTS,
            ),
            (
                ShellToolSettings(max_argument_bytes=2),
                _request(options={"p": "abc"}),
                ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            ),
            (
                ShellToolSettings(
                    max_arguments=3,
                    max_argument_bytes=10,
                    max_command_bytes=4,
                ),
                _request(options={"a": "b"}),
                ShellExecutionErrorCode.COMMAND_TOO_LARGE,
            ),
        )

        for settings, request, error_code in cases:
            with self.subTest(error_code=error_code):
                await self._assert_denied(
                    request,
                    error_code,
                    policy=ExecutionPolicy(settings=settings),
                )

    async def test_argument_budget_boundaries_are_exact_and_bounded(
        self,
    ) -> None:
        request = _request(options={"pattern": "n"})
        resolver = _CountingResolver("/usr/bin/rg")
        spec = await ExecutionPolicy(
            settings=ShellToolSettings(
                max_arguments=128,
                max_argument_bytes=64,
                max_command_bytes=4096,
            ),
            resolver=resolver,
        ).normalize(request)

        self.assertEqual(spec.command, "rg")
        self.assertEqual(resolver.calls, ("rg",))

        for settings, denied_request, error_code in (
            (
                ShellToolSettings(max_arguments=2),
                request,
                ShellExecutionErrorCode.TOO_MANY_ARGUMENTS,
            ),
            (
                ShellToolSettings(max_argument_bytes=7),
                _request(options={"pattern": "12345678"}),
                ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            ),
            (
                ShellToolSettings(max_command_bytes=11),
                request,
                ShellExecutionErrorCode.COMMAND_TOO_LARGE,
            ),
        ):
            denied_resolver = _CountingResolver("/usr/bin/rg")
            with self.subTest(error_code=error_code):
                await self._assert_denied(
                    denied_request,
                    error_code,
                    policy=ExecutionPolicy(
                        settings=settings,
                        resolver=denied_resolver,
                    ),
                )
            self.assertEqual(denied_resolver.calls, ())

    async def test_normalized_argv_budget_boundaries_are_enforced(
        self,
    ) -> None:
        request = _request(options={"pattern": "n"})
        baseline = await ExecutionPolicy(
            settings=ShellToolSettings(
                max_arguments=128,
                max_argument_bytes=64,
                max_command_bytes=4096,
            ),
            resolver=_CountingResolver("/usr/bin/rg"),
        ).normalize(request)
        max_argument_bytes = max(
            len(argument.encode("utf-8")) for argument in baseline.argv
        )
        command_bytes = _command_bytes(baseline.argv)
        exact_resolver = _CountingResolver("/usr/bin/rg")

        exact = await ExecutionPolicy(
            settings=ShellToolSettings(
                max_arguments=len(baseline.display_argv),
                max_argument_bytes=max_argument_bytes,
                max_command_bytes=command_bytes,
            ),
            resolver=exact_resolver,
        ).normalize(request)

        self.assertEqual(exact.argv, baseline.argv)
        self.assertEqual(exact_resolver.calls, ("rg",))

        for settings, error_code in (
            (
                ShellToolSettings(
                    max_arguments=len(baseline.display_argv) - 1
                ),
                ShellExecutionErrorCode.TOO_MANY_ARGUMENTS,
            ),
            (
                ShellToolSettings(max_argument_bytes=max_argument_bytes - 1),
                ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            ),
            (
                ShellToolSettings(max_command_bytes=command_bytes - 1),
                ShellExecutionErrorCode.COMMAND_TOO_LARGE,
            ),
        ):
            denied_resolver = _CountingResolver("/usr/bin/rg")
            with self.subTest(error_code=error_code):
                await self._assert_denied(
                    request,
                    error_code,
                    policy=ExecutionPolicy(
                        settings=settings,
                        resolver=denied_resolver,
                    ),
                )
            self.assertEqual(denied_resolver.calls, ())

    async def test_rg_internal_deny_globs_do_not_consume_argv_count_budget(
        self,
    ) -> None:
        request = _request(
            options={
                "pattern": "stream",
                "case": "insensitive",
                "context_lines": 2,
                "before_context": 0,
                "after_context": 0,
                "max_matches_per_file": 5,
                "max_depth": 6,
                "max_filesize_bytes": 9999999,
                "globs": (),
            }
        )
        baseline = await ExecutionPolicy(
            resolver=_CountingResolver("/usr/bin/rg"),
        ).normalize(request)

        self.assertGreater(len(baseline.argv), len(baseline.display_argv))

        exact = await ExecutionPolicy(
            settings=ShellToolSettings(
                max_arguments=len(baseline.display_argv),
                max_argument_bytes=64,
                max_command_bytes=4096,
            ),
            resolver=_CountingResolver("/usr/bin/rg"),
        ).normalize(request)

        self.assertEqual(exact.argv, baseline.argv)

        await self._assert_denied(
            request,
            ShellExecutionErrorCode.TOO_MANY_ARGUMENTS,
            policy=ExecutionPolicy(
                settings=ShellToolSettings(
                    max_arguments=len(baseline.display_argv) - 1,
                    max_argument_bytes=64,
                    max_command_bytes=4096,
                ),
                resolver=_CountingResolver("/usr/bin/rg"),
            ),
        )

    async def test_display_argv_does_not_drive_process_argument_budgets(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "file.txt").write_text("content\n", encoding="utf-8")

            spec = await ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    cwd="nested",
                    max_argument_bytes=8,
                    max_command_bytes=36,
                ),
                resolver=_CountingResolver("/usr/bin/head"),
            ).normalize(
                _request(
                    tool_name="shell.head",
                    command="head",
                    paths=(_path("file.txt", kind="file"),),
                )
            )

        self.assertEqual(spec.argv, ("head", "-n", "80", "--", "file.txt"))
        self.assertEqual(
            spec.display_argv,
            ("head", "-n", "80", "--", "nested/file.txt"),
        )

    async def test_denies_malformed_options_and_metadata(self) -> None:
        cases = (
            _request(options={1: "value"}),  # type: ignore[dict-item]
            _request(options={"pattern": ""}),
            _request(options={"pattern": b"value"}),
            _request(options={"ratio": nan}),
            _request(metadata={1: "value"}),  # type: ignore[dict-item]
            _request(
                timeout_seconds=61.0,
                metadata={"budget_clamps": "reserved"},
            ),
        )

        for request in cases:
            with self.subTest(request=request):
                await self._assert_denied(
                    request,
                    ShellExecutionErrorCode.INVALID_OPTION,
                )

    async def _assert_denied(
        self,
        request: ShellCommandRequest,
        error_code: ShellExecutionErrorCode,
        *,
        policy: ExecutionPolicy | None = None,
        message: str | None = None,
    ) -> None:
        active_policy = policy or ExecutionPolicy()
        with self.assertRaises(ShellPolicyDenied) as context:
            await active_policy.normalize(request)
        self.assertEqual(context.exception.error_code, error_code)
        if message is not None:
            self.assertEqual(str(context.exception), message)


class _CountingResolver:
    def __init__(self, result: str | None) -> None:
        self._result = result
        self._calls: list[str] = []

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        self._calls.append(command.logical_id)
        return self._result


class _RecordingExecutor:
    def __init__(self) -> None:
        self._calls: list[ExecutionSpec] = []

    @property
    def calls(self) -> tuple[ExecutionSpec, ...]:
        return tuple(self._calls)

    async def execute(self, spec: ExecutionSpec) -> None:
        self._calls.append(spec)


def _request(
    *,
    tool_name: str = "shell.rg",
    command: str = "rg",
    options: dict[object, object] | None = None,
    paths: tuple[PathOperand, ...] = (),
    cwd: str | None = None,
    stdin: bytes | None = None,
    timeout_seconds: object | None = None,
    max_stdout_bytes: object | None = None,
    max_stderr_bytes: object | None = None,
    metadata: dict[object, object] | None = None,
) -> ShellCommandRequest:
    default_options: dict[object, object]
    default_options = {"pattern": "needle"} if command == "rg" else {}
    return ShellCommandRequest(
        tool_name=tool_name,
        command=command,
        options=default_options if options is None else options,  # type: ignore[arg-type]
        paths=paths,
        cwd=cwd,
        stdin=stdin,
        timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
        max_stdout_bytes=max_stdout_bytes,  # type: ignore[arg-type]
        max_stderr_bytes=max_stderr_bytes,  # type: ignore[arg-type]
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
    )


def _path(value: str, *, kind: str = "text_file") -> PathOperand:
    return PathOperand(
        name="input",
        path=value,
        kind=kind,  # type: ignore[arg-type]
        access="read",
    )


def _expected_default_env(*, omp: str = "1") -> dict[str, str]:
    return {
        "LC_ALL": "C",
        "LANG": "C",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "CLICOLOR": "0",
        "CLICOLOR_FORCE": "0",
        "HOME": "/nonexistent",
        "XDG_CONFIG_HOME": "/nonexistent",
        "XDG_CACHE_HOME": "/nonexistent",
        "OMP_THREAD_LIMIT": omp,
    }


def _glob_args(globs: tuple[str, ...]) -> tuple[str, ...]:
    args: list[str] = []
    for glob in globs:
        args.extend(("--glob", glob))
    return tuple(args)


def _command_bytes(argv: tuple[str, ...]) -> int:
    encoded = tuple(argument.encode("utf-8") for argument in argv)
    return sum(len(argument) for argument in encoded) + max(len(argv) - 1, 0)


class PolicyPathMatcherTest(IsolatedAsyncioTestCase):
    def test_hidden_component_detection_uses_path_components(self) -> None:
        self.assertTrue(_has_hidden_component(".hidden"))
        self.assertTrue(_has_hidden_component("nested/.hidden/file.txt"))
        self.assertFalse(_has_hidden_component("visible/file.txt"))
        self.assertFalse(_has_hidden_component("."))

    def test_sensitive_denylist_uses_canonical_relative_paths(self) -> None:
        denied_paths = (
            ".git",
            ".git/config",
            ".GIT/config",
            ".env",
            ".env.local",
            ".envrc",
            "nested/.env",
            "nested/.env.local",
            "nested/.envrc",
            "deploy.pem",
            "nested/deploy.pem",
            "deploy.pem/private.txt",
            "nested/deploy.pem/private.txt",
            "id_rsa",
            "nested/id_rsa",
            "id_rsa/public.txt",
            "nested/id_rsa/public.txt",
            "id_dsa",
            "nested/id_dsa",
            "id_dsa/public.txt",
            "nested/id_dsa/public.txt",
            "id_ecdsa",
            "nested/id_ecdsa",
            "id_ecdsa/public.txt",
            "nested/id_ecdsa/public.txt",
            "id_ed25519",
            "ID_ED25519",
            "nested/id_ed25519",
            "id_ed25519/public.txt",
            "nested/id_ed25519/public.txt",
            ".ssh/config",
            "home/.ssh/config",
            ".docker/config.json",
            "home/.docker/config.json",
            ".config/gh/hosts.yml",
            "home/.config/gcloud/application_default_credentials.json",
            "credentials",
            "nested/credentials",
            "credentials/token.txt",
            "nested/credentials/token.txt",
        )
        allowed_paths = (
            "git/config",
            "env",
            "notes.key.txt",
            "id_report.txt",
            "nested/credential-notes.txt",
        )

        for path in denied_paths:
            with self.subTest(path=path):
                self.assertTrue(path_matches_sensitive_denylist(path))
        for path in allowed_paths:
            with self.subTest(path=path):
                self.assertFalse(path_matches_sensitive_denylist(path))
        with self.assertRaises(AssertionError):
            path_matches_sensitive_denylist(1)  # type: ignore[arg-type]

    def test_virtual_filesystem_detection_uses_absolute_roots(self) -> None:
        self.assertTrue(_is_virtual_filesystem_path(Path("/dev/null")))
        self.assertTrue(_is_virtual_filesystem_path(Path("/proc/self")))
        self.assertTrue(_is_virtual_filesystem_path(Path("/sys/kernel")))
        self.assertFalse(_is_virtual_filesystem_path(Path("/tmp/dev/null")))
        self.assertFalse(_is_virtual_filesystem_path(Path("dev/null")))

    def test_component_candidates_are_empty_for_external_paths(self) -> None:
        self.assertEqual(
            _path_component_candidates(Path("/workspace"), Path("/other")),
            (),
        )

    def test_option_fragments_accept_finite_float_values(self) -> None:
        self.assertEqual(_option_fragments(1.5, "ratio"), ("1.5",))

    def test_option_fragments_allow_whitespace_only_strings(self) -> None:
        self.assertEqual(
            _option_fragments(" ", "options.output_separator"),
            (" ",),
        )

    def test_option_fragments_reject_empty_strings(self) -> None:
        with self.assertRaises(ShellPolicyDenied) as context:
            _option_fragments("", "options.output_separator")

        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.INVALID_OPTION,
        )

    def test_media_path_argument_rejects_stdin_sentinel(self) -> None:
        with self.assertRaises(ShellPolicyDenied) as context:
            _media_path_argument(Path.cwd(), Path.cwd() / "-")

        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.DENIED_PATH,
        )


if __name__ == "__main__":
    main()
