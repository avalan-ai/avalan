from ast import AST, Call, Constant, Import, ImportFrom, keyword, parse, walk
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

from avalan.tool.shell.commands.rg import _rg_policy_deny_globs
from avalan.tool.shell.entities import (
    ExecutionResult,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
)
from avalan.tool.shell.formatting import format_shell_result
from avalan.tool.shell.settings import ShellToolSettings

SHELL_SOURCE_ROOT = (
    Path(__file__).parents[3] / "src" / "avalan" / "tool" / "shell"
)
COMMAND_SOURCE_ROOT = SHELL_SOURCE_ROOT / "commands"
SHELL_COMMANDS = (
    "rg",
    "head",
    "tail",
    "ls",
    "cat",
    "nl",
    "pgrep",
    "file",
    "find",
    "wc",
    "awk",
    "sed",
    "jq",
    "pdfinfo",
    "pdftotext",
    "pdftoppm",
    "reportlab",
    "pdfplumber",
    "pypdf",
    "tesseract",
)
SHELL_EVALUATION_STRINGS = (
    "/bin/sh",
    "/bin/bash",
    "bash -c",
    "sh -c",
    "shell=True",
)
SYNC_SUBPROCESS_CALLS = {
    "call",
    "check_call",
    "check_output",
    "getoutput",
    "getstatusoutput",
    "Popen",
    "run",
}
SYNC_FILE_CALLS = {
    "glob",
    "iterdir",
    "listdir",
    "open",
    "read_bytes",
    "read_text",
    "rglob",
    "scandir",
    "stat",
    "walk",
    "write_bytes",
    "write_text",
}
DIRECT_IO_ALLOWED_MODULES = {
    SHELL_SOURCE_ROOT / "filesystem.py",
    SHELL_SOURCE_ROOT / "python_pdf.py",
}
DIRECT_SPAWN_ALLOWED_MODULES = {
    SHELL_SOURCE_ROOT / "process.py",
}
OUTPUT_FILTER_ALLOWED_MODULES = {
    COMMAND_SOURCE_ROOT / "find.py",
    COMMAND_SOURCE_ROOT / "ls.py",
    COMMAND_SOURCE_ROOT / "rg.py",
}


class ShellGuardrailTest(TestCase):
    def test_command_source_files_match_locked_command_ids(self) -> None:
        command_paths = {
            source_path.stem
            for source_path in COMMAND_SOURCE_ROOT.glob("*.py")
            if source_path.name
            not in {"__init__.py", "base.py", "helpers.py", "python_pdf.py"}
        }

        self.assertEqual(command_paths, set(SHELL_COMMANDS))

    def test_shell_runtime_never_uses_shell_evaluation(self) -> None:
        for source_path, tree in _shell_trees():
            shell_spawn_names = _async_subprocess_call_names(
                tree,
                "create_subprocess_shell",
            )
            exec_spawn_names = _async_subprocess_call_names(
                tree,
                "create_subprocess_exec",
            )
            with self.subTest(source_path=source_path):
                self.assertFalse(
                    _imports_module(tree, "subprocess"),
                    f"{source_path} imports sync subprocess",
                )
                self.assertFalse(
                    _calls_any_name(tree, shell_spawn_names),
                    f"{source_path} uses shell subprocess creation",
                )
                if source_path not in DIRECT_SPAWN_ALLOWED_MODULES:
                    self.assertFalse(
                        _calls_any_name(tree, exec_spawn_names),
                        f"{source_path} directly creates subprocesses",
                    )
                for call in _calls(tree):
                    self.assertFalse(
                        _has_shell_true_keyword(call),
                        f"{source_path} passes shell=True",
                    )
                    call_name = _call_name(call)
                    self.assertNotIn(
                        call_name,
                        SYNC_SUBPROCESS_CALLS,
                        f"{source_path} calls sync subprocess API",
                    )
                    if call_name == "system" and _call_base_name(call) == "os":
                        self.fail(f"{source_path} calls os.system")
                for value in _string_constants(tree):
                    self.assertFalse(
                        any(
                            shell_text in value
                            for shell_text in SHELL_EVALUATION_STRINGS
                        ),
                        f"{source_path} contains shell evaluation text",
                    )

    def test_shell_runtime_guardrail_detects_aliased_async_spawn(
        self,
    ) -> None:
        tree = parse(
            "from asyncio import create_subprocess_exec as spawn_exec\n"
            "from asyncio import create_subprocess_shell as spawn_shell\n"
            "from asyncio.subprocess import "
            "create_subprocess_exec as spawn_exec_submodule\n"
            "async def run():\n"
            "    await spawn_exec('tool')\n"
            "    await spawn_exec_submodule('tool')\n"
            "    await spawn_shell('tool')\n"
        )

        self.assertTrue(
            _calls_any_name(
                tree,
                _async_subprocess_call_names(
                    tree,
                    "create_subprocess_exec",
                ),
            )
        )
        self.assertTrue(
            _calls_any_name(
                tree,
                _async_subprocess_call_names(
                    tree,
                    "create_subprocess_shell",
                ),
            )
        )

    def test_shell_runtime_uses_async_filesystem_boundary(self) -> None:
        for source_path, tree in _shell_trees():
            if source_path in DIRECT_IO_ALLOWED_MODULES:
                continue
            with self.subTest(source_path=source_path):
                for call in _calls(tree):
                    self.assertNotIn(
                        _call_name(call),
                        SYNC_FILE_CALLS,
                        f"{source_path} calls sync filesystem API",
                    )

    def test_command_modules_do_not_reimplement_shell_commands(self) -> None:
        command_paths = tuple(COMMAND_SOURCE_ROOT.glob("*.py"))
        for source_path, tree in _parsed_trees(command_paths):
            if source_path.name in {"__init__.py", "base.py", "helpers.py"}:
                continue
            with self.subTest(source_path=source_path):
                self.assertFalse(
                    _imports_any(
                        tree,
                        {
                            "csv",
                            "glob",
                            "json",
                            "os",
                            "shutil",
                            "subprocess",
                        },
                    ),
                    f"{source_path} imports command implementation helpers",
                )
                for call in _calls(tree):
                    self.assertNotIn(
                        _call_name(call),
                        SYNC_FILE_CALLS | SYNC_SUBPROCESS_CALLS,
                        f"{source_path} executes command behavior directly",
                    )

    def test_command_modules_only_filter_display_output_by_exception(
        self,
    ) -> None:
        for source_path in COMMAND_SOURCE_ROOT.glob("*.py"):
            if source_path.name in {"__init__.py", "base.py", "helpers.py"}:
                continue
            source = source_path.read_text()
            with self.subTest(source_path=source_path):
                if source_path in OUTPUT_FILTER_ALLOWED_MODULES:
                    self.assertIn("output_filter=filter_output", source)
                else:
                    self.assertNotIn("output_filter=", source)

    def test_formatter_redacts_model_visible_sentinels(self) -> None:
        secret_env = "SHELL_SENTINEL_ENV_VALUE"
        denied_file = "sentinel-secret.pem"
        private_key = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "sentinel-private-key\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        home = Path("/Users/sentinel-user")
        host_path = "/private/var/sentinel workspace/key.txt"
        formatted = _format_with_fake_home(
            home,
            _result(
                display_argv=(
                    "rg",
                    secret_env,
                    "--",
                    denied_file,
                    host_path,
                ),
                display_cwd=str(home / "project"),
                stdout=(
                    f"\x1b[31m{secret_env}\x1b[0m\n"
                    f"{denied_file}:1:1:{private_key}\n"
                    "safe.txt:1:1:safe\u202etext"
                ),
                stderr=f"failed at {host_path}\x07",
                error_message=f"denied {secret_env} {host_path}",
                stdout_bytes=240,
                stderr_bytes=64,
            ),
            settings=ShellToolSettings(environment={"TOKEN": secret_env}),
        )

        for leaked in (
            secret_env,
            denied_file,
            "sentinel-private-key",
            "PRIVATE KEY",
            str(home),
            "sentinel-user",
            host_path,
            "sentinel workspace",
            "\x1b",
            "\x07",
            "\u202e",
        ):
            self.assertNotIn(leaked, formatted)
        self.assertIn("[redacted]", formatted)
        self.assertIn("[redacted_path]", formatted)

    def test_policy_injected_rg_deny_globs_never_reach_display_output(
        self,
    ) -> None:
        settings = ShellToolSettings()
        display_argv = ("rg", "--no-config", "-e", "needle", "--", ".")
        formatted = format_shell_result(
            _result(
                argv=(
                    "rg",
                    "--no-config",
                    *_glob_args(_rg_policy_deny_globs(settings)),
                    "-e",
                    "needle",
                    "--",
                    ".",
                ),
                display_argv=display_argv,
                stdout="safe.txt:1:1:needle",
                stdout_bytes=20,
            ),
            settings=settings,
        )

        for glob in _rg_policy_deny_globs(settings):
            self.assertNotIn(glob, formatted)
        self.assertIn("command: rg --no-config -e needle -- .", formatted)

    def test_redaction_output_growth_is_bounded_by_captured_output(
        self,
    ) -> None:
        secret = "SHELL_SENTINEL_SECRET_VALUE"
        host_path = "/tmp/sentinel-project/output.log"
        stdout = "\n".join(
            f"{index}:{secret}:{host_path}:safe" for index in range(600)
        )

        formatted = format_shell_result(
            _result(stdout=stdout, stdout_bytes=len(stdout.encode())),
            settings=ShellToolSettings(environment={"TOKEN": secret}),
        )

        self.assertNotIn(secret, formatted)
        self.assertNotIn(host_path, formatted)
        self.assertGreaterEqual(formatted.count("[redacted]"), 600)
        self.assertLessEqual(len(formatted), len(stdout) + 2400)

    def test_post_filter_output_growth_is_bounded_by_captured_output(
        self,
    ) -> None:
        safe_lines = tuple(
            f"safe-{index}.txt:1:1:needle" for index in range(800)
        )
        denied_lines = tuple(
            f".ssh/id_ed25519-{index}-secret context" for index in range(200)
        )
        stdout = "\n".join((*safe_lines, *denied_lines))

        formatted = format_shell_result(
            _result(stdout=stdout, stdout_bytes=len(stdout.encode()))
        )

        self.assertEqual(formatted.count("[redacted_path]"), 200)
        self.assertNotIn("secret context", formatted)
        self.assertLessEqual(len(formatted), len(stdout) + 2400)


def _shell_trees() -> tuple[tuple[Path, AST], ...]:
    return _parsed_trees(SHELL_SOURCE_ROOT.rglob("*.py"))


def _parsed_trees(paths: object) -> tuple[tuple[Path, AST], ...]:
    return tuple(
        (path, parse(path.read_text(), filename=str(path)))
        for path in sorted(paths)
        if "__pycache__" not in path.parts
    )


def _calls(tree: AST) -> tuple[Call, ...]:
    return tuple(node for node in walk(tree) if isinstance(node, Call))


def _call_name(call: Call) -> str:
    function = call.func
    return getattr(function, "attr", getattr(function, "id", ""))


def _call_base_name(call: Call) -> str:
    function = call.func
    value = getattr(function, "value", None)
    return getattr(value, "id", "")


def _calls_name(tree: AST, name: str) -> bool:
    return any(_call_name(call) == name for call in _calls(tree))


def _calls_any_name(tree: AST, names: set[str]) -> bool:
    return any(_call_name(call) in names for call in _calls(tree))


def _async_subprocess_call_names(tree: AST, imported_name: str) -> set[str]:
    names = {imported_name}
    for node in walk(tree):
        if not isinstance(node, ImportFrom):
            continue
        if node.module not in {"asyncio", "asyncio.subprocess"}:
            continue
        for alias in node.names:
            if alias.name == imported_name:
                names.add(alias.asname or alias.name)
    return names


def _has_shell_true_keyword(call: Call) -> bool:
    return any(
        item.arg == "shell"
        and isinstance(item.value, Constant)
        and item.value.value is True
        for item in call.keywords
        if isinstance(item, keyword)
    )


def _imports_module(tree: AST, module_name: str) -> bool:
    return _imports_any(tree, {module_name})


def _imports_any(tree: AST, module_names: set[str]) -> bool:
    for node in walk(tree):
        if isinstance(node, Import):
            if any(alias.name in module_names for alias in node.names):
                return True
        if isinstance(node, ImportFrom) and node.module in module_names:
            return True
    return False


def _string_constants(tree: AST) -> tuple[str, ...]:
    return tuple(
        node.value
        for node in walk(tree)
        if isinstance(node, Constant) and isinstance(node.value, str)
    )


def _glob_args(globs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(argument for glob in globs for argument in ("--glob", glob))


def _format_with_fake_home(
    home: Path,
    result: ExecutionResult,
    *,
    settings: ShellToolSettings,
) -> str:
    with patch("avalan.tool.shell.formatting.Path.home", return_value=home):
        return format_shell_result(result, settings=settings)


def _result(
    *,
    backend: str = "local",
    tool_name: str = "shell.rg",
    command: str = "rg",
    argv: tuple[str, ...] = ("rg", "needle", "--", "."),
    display_argv: tuple[str, ...] = ("rg", "needle", "--", "."),
    cwd: str = "/workspace",
    display_cwd: str = ".",
    status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
    exit_code: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    stdout_media_type: str = "text/plain",
    output_kind: ShellOutputKind = ShellOutputKind.TEXT,
    stdout_bytes: int = 0,
    stderr_bytes: int = 0,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
    duration_ms: int = 0,
    error_message: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        backend=backend,
        tool_name=tool_name,
        command=command,
        argv=argv,
        display_argv=display_argv,
        cwd=cwd,
        display_cwd=display_cwd,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        generated_files=(),
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
        duration_ms=duration_ms,
        error_code=ShellExecutionErrorCode(status.value),
        error_message=error_message,
    )


if __name__ == "__main__":
    main()
