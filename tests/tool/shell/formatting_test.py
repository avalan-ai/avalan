from pathlib import Path
from tempfile import gettempdir
from unittest import TestCase, main
from unittest.mock import patch

from avalan.tool.shell.entities import (
    ExecutionResult,
    GeneratedFile,
    ShellCompositionResult,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellOutputKind,
)
from avalan.tool.shell.formatting import (
    format_shell_composition_result,
    format_shell_result,
)
from avalan.tool.shell.settings import ShellToolSettings


class ShellFormattingTest(TestCase):
    def test_completed_result_formats_stable_envelope(self) -> None:
        self.assertEqual(
            format_shell_result(
                _result(
                    status=ShellExecutionStatus.COMPLETED,
                    exit_code=0,
                    stdout="match",
                    stdout_bytes=5,
                    duration_ms=12,
                )
            ),
            "\n".join(
                (
                    "tool: shell.rg",
                    "status: completed",
                    "command: rg needle -- .",
                    "cwd: .",
                    "exit_code: 0",
                    "error_code: completed",
                    "error_message: null",
                    "output_kind: text",
                    "stdout_media_type: text/plain",
                    "timed_out: false",
                    "duration_ms: 12",
                    "stdout_bytes: 5",
                    "stderr_bytes: 0",
                    "stdout_truncated: false",
                    "stderr_truncated: false",
                    "",
                    "stdout:",
                    "match",
                )
            ),
        )

    def test_composition_result_formats_aggregate_without_stage_stdout(
        self,
    ) -> None:
        formatted = format_shell_composition_result(
            ShellCompositionResult(
                mode="pipeline",
                status=ShellExecutionStatus.COMPLETED,
                stdout="final stdout\n",
                stderr="[read:cat]\nwarning\n",
                steps=(
                    ShellExecutionStepResult(
                        id="read",
                        command="cat",
                        status=ShellExecutionStatus.COMPLETED,
                        exit_code=0,
                        stdout="INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK",
                        stderr="warning\n",
                        stdout_bytes=35,
                        stderr_bytes=8,
                        stdout_truncated=False,
                        stderr_truncated=False,
                        duration_ms=3,
                        metadata={"private": "PRIVATE_METADATA"},
                    ),
                    ShellExecutionStepResult(
                        id="count",
                        command="wc",
                        status=ShellExecutionStatus.COMPLETED,
                        exit_code=0,
                        stdout="final stdout\n",
                        stderr="",
                        stdout_bytes=13,
                        stderr_bytes=0,
                        stdout_truncated=False,
                        stderr_truncated=False,
                        duration_ms=4,
                    ),
                ),
                stdout_bytes=13,
                stderr_bytes=19,
                duration_ms=9,
                metadata={"private": "PRIVATE_RESULT_METADATA"},
            )
        )

        self.assertIn("tool: shell.pipeline", formatted)
        self.assertIn("status: completed", formatted)
        self.assertIn("stage_chain: cat | wc", formatted)
        self.assertIn("  status: completed", formatted)
        self.assertIn("\nstdout:\nfinal stdout\n", formatted)
        self.assertIn("\nstderr:\n[read:cat]\nwarning\n", formatted)
        self.assertNotIn("INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK", formatted)
        self.assertNotIn("PRIVATE_METADATA", formatted)
        self.assertNotIn("PRIVATE_RESULT_METADATA", formatted)

    def test_no_output_keeps_stdout_block(self) -> None:
        self.assertEqual(
            format_shell_result(
                _result(status=ShellExecutionStatus.COMPLETED)
            ),
            "\n".join(
                (
                    "tool: shell.rg",
                    "status: completed",
                    "command: rg needle -- .",
                    "cwd: .",
                    "exit_code: 0",
                    "error_code: completed",
                    "error_message: null",
                    "output_kind: text",
                    "stdout_media_type: text/plain",
                    "timed_out: false",
                    "duration_ms: 0",
                    "stdout_bytes: 0",
                    "stderr_bytes: 0",
                    "stdout_truncated: false",
                    "stderr_truncated: false",
                    "",
                    "stdout:",
                    "",
                )
            ),
        )

    def test_stderr_only_completed_result_includes_stderr(self) -> None:
        formatted = format_shell_result(
            _result(
                status=ShellExecutionStatus.COMPLETED,
                stderr="warning",
                stderr_bytes=7,
            )
        )

        self.assertIn("\nstderr:\nwarning", formatted)
        self.assertIn("status: completed", formatted)

    def test_error_statuses_include_stderr_block_when_empty(self) -> None:
        for status in (
            ShellExecutionStatus.NO_MATCHES,
            ShellExecutionStatus.NONZERO_EXIT,
            ShellExecutionStatus.TIMEOUT,
            ShellExecutionStatus.COMMAND_UNAVAILABLE,
            ShellExecutionStatus.SPAWN_FAILED,
        ):
            with self.subTest(status=status):
                formatted = format_shell_result(
                    _result(status=status, exit_code=1)
                )

                self.assertIn(f"status: {status.value}", formatted)
                self.assertIn("\nstderr:\n", formatted)

    def test_nonzero_result_formats_error_fields(self) -> None:
        formatted = format_shell_result(
            _result(
                status=ShellExecutionStatus.NONZERO_EXIT,
                exit_code=2,
                stderr="failed",
                stderr_bytes=6,
                error_message="command exited with status 2",
            )
        )

        self.assertIn("exit_code: 2", formatted)
        self.assertIn("error_code: nonzero_exit", formatted)
        self.assertIn(
            "error_message: command exited with status 2",
            formatted,
        )
        self.assertIn("\nstderr:\nfailed", formatted)

    def test_timeout_and_truncation_flags_are_rendered(self) -> None:
        formatted = format_shell_result(
            _result(
                status=ShellExecutionStatus.TIMEOUT,
                exit_code=None,
                stdout="partial",
                stderr="still running",
                stdout_bytes=7,
                stderr_bytes=13,
                stdout_truncated=True,
                stderr_truncated=True,
                timed_out=True,
                error_message="command timed out",
            )
        )

        self.assertIn("timed_out: true", formatted)
        self.assertIn("stdout_truncated: true", formatted)
        self.assertIn("stderr_truncated: true", formatted)
        self.assertIn("\nstdout:\npartial", formatted)
        self.assertIn("\nstderr:\nstill running", formatted)

    def test_json_and_ocr_text_outputs_preserve_media_type(self) -> None:
        json_output = format_shell_result(
            _result(
                command="jq",
                tool_name="shell.jq",
                display_argv=("jq", ".name", "data.json"),
                stdout='{"name":"Ada"}',
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.JSON,
            )
        )
        ocr_output = format_shell_result(
            _result(
                command="tesseract",
                tool_name="shell.tesseract",
                display_argv=("tesseract", "image.png", "stdout"),
                stdout="OCR text",
                stdout_media_type="text/plain",
            )
        )

        self.assertIn("output_kind: json", json_output)
        self.assertIn("stdout_media_type: application/json", json_output)
        self.assertIn("tool: shell.tesseract", ocr_output)
        self.assertIn("\nstdout:\nOCR text", ocr_output)

    def test_generated_files_format_bounded_metadata(self) -> None:
        generated_file = GeneratedFile(
            display_path="outputs/page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=123,
            sha256="a" * 64,
            page=1,
            width=320,
            height=240,
            content_base64="aW1hZ2U=",
            truncated=False,
        )
        formatted = format_shell_result(
            _result(
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                display_argv=("pdftoppm", "-png", "document.pdf"),
                output_kind=ShellOutputKind.GENERATED_FILES,
                generated_files=(generated_file,),
            )
        )

        self.assertIn("output_kind: generated_files", formatted)
        self.assertIn("\ngenerated_files:\n", formatted)
        self.assertIn("- display_path: outputs/page-1.png", formatted)
        self.assertIn("  media_type: image/png", formatted)
        self.assertIn("  suffix: .png", formatted)
        self.assertIn("  bytes: 123", formatted)
        self.assertIn(f"  sha256: {'a' * 64}", formatted)
        self.assertIn("  page: 1", formatted)
        self.assertIn("  width: 320", formatted)
        self.assertIn("  height: 240", formatted)
        self.assertIn("  truncated: false", formatted)
        self.assertIn("  content_base64: aW1hZ2U=", formatted)

    def test_empty_generated_files_format_as_empty_list(self) -> None:
        formatted = format_shell_result(
            _result(output_kind=ShellOutputKind.GENERATED_FILES)
        )

        self.assertTrue(formatted.endswith("\ngenerated_files:\n[]"))

    def test_all_statuses_have_stable_formatted_output(self) -> None:
        for status in ShellExecutionStatus:
            with self.subTest(status=status):
                formatted = format_shell_result(
                    _result(
                        status=status,
                        exit_code=(
                            None
                            if status
                            in {
                                ShellExecutionStatus.COMMAND_UNAVAILABLE,
                                ShellExecutionStatus.SPAWN_FAILED,
                                ShellExecutionStatus.TIMEOUT,
                                ShellExecutionStatus.TOOL_ERROR,
                            }
                            else 0
                        ),
                    )
                )

                self.assertIn(f"status: {status.value}", formatted)
                self.assertIn(
                    "error_code:"
                    f" {ShellExecutionErrorCode(status.value).value}",
                    formatted,
                )

    def test_core_redaction_covers_env_values_and_private_key_blocks(
        self,
    ) -> None:
        home = str(Path.home())
        tempdir = gettempdir()
        private_key = (
            "-----BEGIN PRIVATE KEY-----\ntopsecret\n-----END PRIVATE KEY-----"
        )
        formatted = format_shell_result(
            _result(
                stdout=(
                    f"token=topsecret {private_key} home={home} "
                    f"tmp={tempdir} explicit=manual-secret"
                ),
                stdout_bytes=200,
            ),
            settings=ShellToolSettings(environment={"TOKEN": "topsecret"}),
            redaction_values=("manual-secret",),
        )

        self.assertNotIn("topsecret", formatted)
        self.assertNotIn("PRIVATE KEY", formatted)
        self.assertNotIn(home, formatted)
        self.assertNotIn(tempdir, formatted)
        self.assertNotIn("manual-secret", formatted)
        self.assertIn("[redacted]", formatted)
        self.assertIn("[redacted_path]", formatted)

    def test_redaction_strips_terminal_and_unicode_controls(self) -> None:
        formatted = format_shell_result(
            _result(
                stdout="\x1b[31mred\x1b[0m\r\nsafe\u202ehidden\x07\ttext",
                stderr="\x1b]0;title\x07stderr",
                stdout_bytes=32,
                stderr_bytes=16,
            )
        )

        self.assertIn("\nstdout:\nred\nsafehidden\ttext", formatted)
        self.assertIn("\nstderr:\nstderr", formatted)
        self.assertNotIn("\x1b", formatted)
        self.assertNotIn("\r", formatted)
        self.assertNotIn("\u202e", formatted)
        self.assertNotIn("\x07", formatted)

    def test_command_redaction_replaces_control_arguments(self) -> None:
        formatted = format_shell_result(
            _result(
                tool_name="shell.nl",
                command="nl",
                argv=("nl", "-d", "\x01\x02", "--", "visible.txt"),
                display_argv=("nl", "-d", "\x01\x02", "--", "visible.txt"),
            )
        )

        self.assertIn("command: nl -d '[redacted]' -- visible.txt", formatted)
        self.assertNotIn("\x01\x02", formatted)

    def test_redaction_covers_secret_like_assignments(self) -> None:
        formatted = format_shell_result(
            _result(
                stdout=(
                    "API_TOKEN=abc123 password: hunter2 "
                    "credential_value = keep api_key=value "
                    "monkey=banana"
                ),
                stdout_bytes=64,
            )
        )

        self.assertNotIn("abc123", formatted)
        self.assertNotIn("hunter2", formatted)
        self.assertNotIn("keep", formatted)
        self.assertNotIn("api_key=value", formatted)
        self.assertIn("API_TOKEN=[redacted]", formatted)
        self.assertIn("password: [redacted]", formatted)
        self.assertIn("credential_value = [redacted]", formatted)
        self.assertIn("api_key=[redacted]", formatted)
        self.assertIn("monkey=banana", formatted)

    def test_redaction_covers_quoted_secret_like_assignments(self) -> None:
        formatted = format_shell_result(
            _result(
                stdout=(
                    'API_TOKEN="alpha beta" '
                    "PASSWORD='gamma delta' "
                    'credential_value="epsilon \\"zeta\\" eta" '
                    'monkey="banana split"'
                ),
                stdout_bytes=128,
            )
        )

        for leaked_value in (
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
        ):
            self.assertNotIn(leaked_value, formatted)
        self.assertIn("API_TOKEN=[redacted]", formatted)
        self.assertIn("PASSWORD=[redacted]", formatted)
        self.assertIn("credential_value=[redacted]", formatted)
        self.assertIn('monkey="banana split"', formatted)

    def test_redaction_covers_quoted_key_secret_like_members(self) -> None:
        formatted = format_shell_result(
            _result(
                command="jq",
                tool_name="shell.jq",
                display_argv=("jq", ".", "data.json"),
                stdout=(
                    '{"api_token":"abc123","password": 12345,'
                    "'credential_value': 'gamma delta',"
                    '"api_key":99999,"safe":"visible"}'
                ),
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.JSON,
                stdout_bytes=128,
            )
        )

        for leaked_value in ("abc123", "12345", "gamma", "delta", "99999"):
            self.assertNotIn(leaked_value, formatted)
        self.assertIn('"api_token":"[redacted]"', formatted)
        self.assertIn('"password": "[redacted]"', formatted)
        self.assertIn("'credential_value': '[redacted]'", formatted)
        self.assertIn('"api_key":"[redacted]"', formatted)
        self.assertIn('"safe":"visible"', formatted)

    def test_rg_output_filters_denied_path_lines(self) -> None:
        formatted = format_shell_result(
            _result(
                command="rg",
                stdout=(
                    "safe.txt:1:1:ok\n"
                    ".env:1:1:TOKEN=secret\n"
                    "nested/.ssh/config:2:1:Host example\n"
                    "!*.pem:policy glob leaked"
                ),
                stdout_bytes=120,
            )
        )

        self.assertIn("safe.txt:1:1:ok", formatted)
        self.assertNotIn(".env", formatted)
        self.assertNotIn(".ssh", formatted)
        self.assertNotIn("!*.pem", formatted)
        self.assertGreaterEqual(formatted.count("[redacted_path]"), 3)

    def test_rg_output_suppresses_denied_line_content(self) -> None:
        formatted = format_shell_result(
            _result(
                command="rg",
                stdout=(
                    "safe.txt:1:1:keep ordinary match\n"
                    ".env:2:1:leaked-private-value\n"
                    "nested/.ssh/config-3-sensitive context"
                ),
                stdout_bytes=120,
            )
        )

        self.assertIn("safe.txt:1:1:keep ordinary match", formatted)
        self.assertNotIn("leaked-private-value", formatted)
        self.assertNotIn("sensitive context", formatted)
        self.assertEqual(formatted.count("[redacted_path]"), 2)

    def test_rg_output_filters_denied_context_lines(self) -> None:
        formatted = format_shell_result(
            _result(
                command="rg",
                stdout=(
                    "safe-1.txt-2-safe context\n"
                    "credentials-1-secret context\n"
                    "nested/id_ed25519-2-key context\n"
                    ".netrc-3-login context"
                ),
                stdout_bytes=120,
            )
        )

        self.assertIn("safe-1.txt-2-safe context", formatted)
        self.assertNotIn("credentials-1-secret", formatted)
        self.assertNotIn("id_ed25519-2-key", formatted)
        self.assertNotIn(".netrc-3-login", formatted)
        self.assertEqual(formatted.count("[redacted_path]"), 3)

    def test_ls_output_filters_denied_paths_with_directory_suffix(
        self,
    ) -> None:
        formatted = format_shell_result(
            _result(
                command="ls",
                display_argv=("ls", "-1p", "--", "."),
                stdout="safe.txt\n.git/\ncredentials\nvisible/",
                stdout_bytes=40,
            )
        )

        self.assertIn("safe.txt", formatted)
        self.assertIn("visible/", formatted)
        self.assertNotIn(".git", formatted)
        self.assertNotIn("credentials", formatted)
        self.assertEqual(formatted.count("[redacted_path]"), 2)

    def test_ls_output_filters_entire_denied_entries_with_spaces(
        self,
    ) -> None:
        formatted = format_shell_result(
            _result(
                command="ls",
                display_argv=("ls", "-1p", "--", "."),
                stdout="safe.txt\n.env backup/\nvisible backup/\n",
                stdout_bytes=38,
            )
        )

        self.assertIn("safe.txt", formatted)
        self.assertIn("visible backup/", formatted)
        self.assertNotIn(".env", formatted)
        self.assertNotIn("env backup", formatted)
        self.assertEqual(formatted.count("[redacted_path]"), 1)

    def test_ls_output_filtering_length_is_bounded(self) -> None:
        safe_entries = [f"safe-{index}.txt" for index in range(200)]
        denied_entries = [f".env generated {index}/" for index in range(25)]
        stdout = "\n".join((*safe_entries, *denied_entries))

        formatted = format_shell_result(
            _result(
                command="ls",
                display_argv=("ls", "-1p", "--", "."),
                stdout=stdout,
                stdout_bytes=len(stdout.encode()),
            )
        )

        self.assertEqual(formatted.count("[redacted_path]"), 25)
        self.assertNotIn(".env generated", formatted)
        self.assertLess(len(formatted), len(stdout) + 800)

    def test_unregistered_command_uses_generic_output_redaction(self) -> None:
        formatted = format_shell_result(
            _result(
                command="custom",
                tool_name="shell.custom",
                display_argv=("custom",),
                stdout="visible\n.env:1:1:match=value",
                stdout_bytes=28,
            )
        )

        self.assertIn("tool: shell.custom", formatted)
        self.assertIn("\nstdout:\nvisible\n", formatted)
        self.assertNotIn(".env", formatted)
        self.assertIn("[redacted_path]:1:1:match=value", formatted)

    def test_sensitive_relative_paths_are_redacted_for_all_commands(
        self,
    ) -> None:
        formatted = format_shell_result(
            _result(
                command="cat",
                tool_name="shell.cat",
                display_argv=("cat", "visible.txt"),
                stdout=(
                    "safe docs/readme.md .env nested/.ssh/config "
                    "id_ed25519 !*.pem"
                ),
                stderr="reported credentials and deploy.key",
                stdout_bytes=72,
                stderr_bytes=35,
            )
        )

        self.assertIn("safe docs/readme.md", formatted)
        for leaked_value in (
            ".env",
            ".ssh",
            "id_ed25519",
            "!*.pem",
            "credentials",
            "deploy.key",
        ):
            self.assertNotIn(leaked_value, formatted)
        self.assertGreaterEqual(formatted.count("[redacted_path]"), 6)

    def test_non_sensitive_relative_paths_are_not_redacted(self) -> None:
        formatted = format_shell_result(
            _result(
                command="cat",
                tool_name="shell.cat",
                display_argv=("cat", "visible.txt"),
                stdout=(
                    "docs/readme.md ./notes.txt image/png "
                    "https://example.test/path"
                ),
                stdout_bytes=68,
            )
        )

        self.assertIn("docs/readme.md", formatted)
        self.assertIn("./notes.txt", formatted)
        self.assertIn("stdout_media_type: text/plain", formatted)
        self.assertIn("image/png", formatted)
        self.assertIn("https://example.test/path", formatted)

    def test_generated_output_private_paths_are_redacted(self) -> None:
        tempdir = gettempdir()
        generated_name = "avalan-shell-0123456789abcdef0123456789abcdef"
        generated_file = GeneratedFile(
            display_path=f"{tempdir}/{generated_name}/page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=123,
            sha256="a" * 64,
            content_base64="binary\x00data",
        )
        formatted = format_shell_result(
            _result(
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                display_argv=("pdftoppm", f"{tempdir}/{generated_name}"),
                stdout=f"wrote {tempdir}/{generated_name}/page-1.png",
                output_kind=ShellOutputKind.GENERATED_FILES,
                generated_files=(generated_file,),
                stdout_bytes=80,
            )
        )

        self.assertNotIn(tempdir, formatted)
        self.assertNotIn(generated_name, formatted)
        self.assertNotIn("\x00", formatted)
        self.assertIn("[redacted_path]", formatted)

    def test_arbitrary_absolute_host_paths_are_redacted(self) -> None:
        formatted = format_shell_result(
            _result(
                stdout=(
                    "see /etc/passwd and /opt/app/secrets.env "
                    "but keep text/plain and https://example.test/path"
                ),
                stderr="failed at /usr/local/bin/tool",
                stderr_bytes=29,
                stdout_bytes=94,
            )
        )

        self.assertNotIn("/etc/passwd", formatted)
        self.assertNotIn("/opt/app/secrets.env", formatted)
        self.assertNotIn("/usr/local/bin/tool", formatted)
        self.assertIn("[redacted_path]", formatted)
        self.assertIn("stdout_media_type: text/plain", formatted)
        self.assertIn("https://example.test/path", formatted)

    def test_absolute_host_paths_with_spaces_are_fully_redacted(
        self,
    ) -> None:
        home_path = Path.home() / "Secret Project" / "token.txt"
        other_path = "/opt/Visible Project/report.txt"
        formatted = format_shell_result(
            _result(
                stdout=f"{home_path}: denied\n{other_path}: denied",
                stdout_bytes=128,
            )
        )

        for leaked_value in (
            str(home_path),
            str(other_path),
            "Secret Project",
            "Visible Project",
            "token.txt",
            "report.txt",
        ):
            self.assertNotIn(leaked_value, formatted)
        self.assertIn("[redacted_path]: denied", formatted)

    def test_redaction_covers_standalone_host_username(self) -> None:
        with patch(
            "avalan.tool.shell.formatting.Path.home",
            return_value=Path("/Users/aliceuser"),
        ):
            formatted = format_shell_result(
                _result(
                    stdout=(
                        "owner=aliceuser prefixaliceuser "
                        "aliceuser_suffix aliceuser-name"
                    ),
                    stderr="permission denied for aliceuser",
                    stdout_bytes=64,
                    stderr_bytes=31,
                )
            )

        self.assertNotIn("owner=aliceuser", formatted)
        self.assertNotIn("for aliceuser", formatted)
        self.assertIn("owner=[redacted]", formatted)
        self.assertIn("for [redacted]", formatted)
        self.assertIn("prefixaliceuser", formatted)
        self.assertIn("aliceuser_suffix", formatted)
        self.assertIn("aliceuser-name", formatted)

    def test_redaction_time_is_linear_in_number_of_configured_values(
        self,
    ) -> None:
        values = tuple(f"secret-{index:03d}" for index in range(100))
        formatted = format_shell_result(
            _result(stdout=" ".join(values), stdout_bytes=1200),
            redaction_values=values,
        )

        self.assertEqual(formatted.count("[redacted]"), len(values))
        self.assertLess(len(formatted), 4000)

    def test_redaction_values_rejects_scalar_string(self) -> None:
        with self.assertRaises(AssertionError):
            format_shell_result(_result(), redaction_values="manual-secret")

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            format_shell_result(object())
        with self.assertRaises(AssertionError):
            format_shell_result(_result(), settings=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            format_shell_result(_result(), redaction_values=(1,))  # type: ignore[list-item]


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
    generated_files: tuple[GeneratedFile, ...] = (),
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
        generated_files=generated_files,
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
