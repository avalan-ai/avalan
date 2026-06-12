from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from avalan.tool.shell.entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    SHELL_STATUS_ERROR_CODES,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    GeneratedOutputPlan,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolError,
)
from avalan.tool.shell.policy import ExecutionPolicy


def _create_execution_spec(**kwargs: object) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(**kwargs)  # type: ignore[arg-type]


class ShellEntitiesTest(TestCase):
    def test_status_values_are_locked(self) -> None:
        self.assertEqual(
            {status.value for status in ShellExecutionStatus},
            {
                "completed",
                "no_matches",
                "nonzero_exit",
                "policy_denied",
                "command_unavailable",
                "spawn_failed",
                "timeout",
                "cancelled",
                "binary_skipped",
                "too_large",
                "tool_error",
            },
        )

    def test_error_code_values_cover_statuses_and_policy_reasons(
        self,
    ) -> None:
        self.assertLessEqual(
            {status.value for status in ShellExecutionStatus},
            {code.value for code in ShellExecutionErrorCode},
        )
        self.assertEqual(
            {code.value for code in ShellExecutionErrorCode}
            - {status.value for status in ShellExecutionStatus},
            {
                "denied_command",
                "denied_path",
                "traversal",
                "hidden_path",
                "sensitive_path",
                "symlink",
                "special_file",
                "binary_content",
                "too_many_arguments",
                "argument_too_large",
                "command_too_large",
                "glob_too_large",
                "invalid_option",
                "invalid_cwd",
                "write_denied",
                "shell_denied",
                "stdin_denied",
                "executable_unavailable",
                "unsafe_filter",
                "unsupported_jq_feature",
                "invalid_page_range",
                "pdf_page_cap_exceeded",
                "raster_dpi_cap_exceeded",
                "generated_output_cap_exceeded",
                "unsupported_ocr_language",
                "invalid_ocr_mode",
                "unsupported_media_signature",
            },
        )

    def test_status_error_code_registry_is_stable(self) -> None:
        self.assertEqual(
            SHELL_STATUS_ERROR_CODES,
            {
                status: ShellExecutionErrorCode(status.value)
                for status in ShellExecutionStatus
            },
        )

    def test_output_kind_values_are_locked(self) -> None:
        self.assertEqual(
            {kind.value for kind in ShellOutputKind},
            {"text", "json", "generated_files"},
        )

    def test_path_operand_accepts_expected_fields(self) -> None:
        operand = PathOperand(
            name="input",
            path="src/main.py",
            kind="text_file",
            access="read",
        )

        self.assertEqual(operand.name, "input")
        self.assertTrue(operand.required)

    def test_path_operand_rejects_invalid_fields(self) -> None:
        for kwargs in (
            {"name": "", "path": "file", "kind": "file", "access": "read"},
            {"name": "input", "path": "", "kind": "file", "access": "read"},
            {
                "name": "input",
                "path": "file",
                "kind": "invalid",
                "access": "read",
            },
            {
                "name": "input",
                "path": "file",
                "kind": "file",
                "access": "invalid",
            },
            {
                "name": "input",
                "path": "file",
                "kind": "file",
                "access": "read",
                "required": 1,
            },
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    PathOperand(**kwargs)  # type: ignore[arg-type]

    def test_shell_command_request_copies_mutable_inputs(self) -> None:
        options = {"line_numbers": True}
        metadata = {"source": "test"}
        operand = PathOperand(
            name="input",
            path="file.txt",
            kind="text_file",
            access="read",
        )
        request = ShellCommandRequest(
            tool_name="shell.rg",
            command="rg",
            options=options,
            paths=(operand,),
            cwd=".",
            metadata=metadata,
        )

        options["line_numbers"] = False
        metadata["source"] = "changed"

        self.assertEqual(request.options, {"line_numbers": True})
        self.assertEqual(request.paths, (operand,))
        self.assertEqual(request.metadata, {"source": "test"})

    def test_shell_command_request_rejects_invalid_fields(self) -> None:
        valid = {
            "tool_name": "shell.rg",
            "command": "rg",
            "options": {},
            "paths": (),
            "cwd": None,
        }
        for kwargs in (
            {"tool_name": ""},
            {"command": ""},
            {
                "options": [],
            },
            {
                "paths": [],
            },
            {
                "paths": (object(),),
            },
            {
                "cwd": "",
            },
            {
                "stdin": "input",
            },
            {
                "timeout_seconds": 0,
            },
            {
                "max_stdout_bytes": True,
            },
            {
                "metadata": [],
            },
        ):
            with self.subTest(kwargs=kwargs):
                request_kwargs = dict(valid)
                request_kwargs.update(kwargs)
                with self.assertRaises(AssertionError):
                    ShellCommandRequest(
                        **request_kwargs,
                    )  # type: ignore[arg-type]

    def test_execution_spec_copies_mutable_inputs(self) -> None:
        env = {"LC_ALL": "C"}
        metadata = {"clamped": True}
        spec = _create_execution_spec(
            backend="local",
            tool_name="shell.rg",
            command="rg",
            executable="/usr/bin/rg",
            argv=("rg", "needle"),
            display_argv=("rg", "needle"),
            cwd="/workspace",
            display_cwd=".",
            env=env,
            stdin=None,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            resource_class="standard",
            output_plan=None,
            timeout_seconds=1.0,
            max_stdout_bytes=10,
            max_stderr_bytes=11,
            metadata=metadata,
        )

        env["LC_ALL"] = "changed"
        metadata["clamped"] = False

        self.assertEqual(spec.env, {"LC_ALL": "C"})
        self.assertEqual(spec.metadata, {"clamped": True})

    def test_execution_spec_preserves_display_only_fields(self) -> None:
        spec = _create_execution_spec(
            backend="local",
            tool_name="shell.rg",
            command="rg",
            executable="/usr/bin/rg",
            argv=("/usr/bin/rg", "--glob", "!/private/workspace/.git/**"),
            display_argv=("rg",),
            cwd="/private/workspace",
            display_cwd=".",
            env={},
            stdin=None,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            resource_class="standard",
            output_plan=None,
            timeout_seconds=1.0,
            max_stdout_bytes=10,
            max_stderr_bytes=10,
        )

        self.assertNotEqual(spec.argv, spec.display_argv)
        self.assertNotEqual(spec.cwd, spec.display_cwd)

    def test_execution_spec_rejects_direct_construction(self) -> None:
        valid = {
            "backend": "local",
            "tool_name": "shell.rg",
            "command": "rg",
            "executable": "/usr/bin/rg",
            "argv": ("rg",),
            "display_argv": ("rg",),
            "cwd": "/workspace",
            "display_cwd": ".",
            "env": {},
            "stdin": None,
            "stdout_media_type": "text/plain",
            "output_kind": ShellOutputKind.TEXT,
            "resource_class": "standard",
            "output_plan": None,
            "timeout_seconds": 1.0,
            "max_stdout_bytes": 10,
            "max_stderr_bytes": 10,
        }

        with self.assertRaises(TypeError):
            ExecutionSpec(**valid)  # type: ignore[call-arg]
        with self.assertRaises(AssertionError):
            ExecutionSpec(_policy_owned=object(), **valid)
        self.assertFalse(hasattr(ExecutionSpec, "create"))
        self.assertFalse(hasattr(ExecutionSpec, "_create"))

    def test_execution_spec_rejects_invalid_fields(self) -> None:
        valid = {
            "backend": "local",
            "tool_name": "shell.rg",
            "command": "rg",
            "executable": "/usr/bin/rg",
            "argv": ("rg",),
            "display_argv": ("rg",),
            "cwd": "/workspace",
            "display_cwd": ".",
            "env": {},
            "stdin": None,
            "stdout_media_type": "text/plain",
            "output_kind": ShellOutputKind.TEXT,
            "resource_class": "standard",
            "output_plan": None,
            "timeout_seconds": 1.0,
            "max_stdout_bytes": 10,
            "max_stderr_bytes": 10,
        }
        invalid_values = {
            "backend": "remote",
            "tool_name": "",
            "command": "",
            "executable": "",
            "argv": ["rg"],
            "display_argv": ["rg"],
            "cwd": "",
            "display_cwd": "",
            "env": {"": "value"},
            "stdin": "input",
            "stdout_media_type": "",
            "output_kind": "text",
            "resource_class": "unknown",
            "output_plan": object(),
            "timeout_seconds": 0,
            "max_stdout_bytes": -1,
            "max_stderr_bytes": True,
            "metadata": [],
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(valid)
                kwargs[field_name] = value
                with self.assertRaises(AssertionError):
                    _create_execution_spec(**kwargs)

    def test_execution_spec_is_frozen(self) -> None:
        spec = _create_execution_spec(
            backend="local",
            tool_name="shell.rg",
            command="rg",
            executable="/usr/bin/rg",
            argv=("rg",),
            display_argv=("rg",),
            cwd="/workspace",
            display_cwd=".",
            env={},
            stdin=None,
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            resource_class="standard",
            output_plan=None,
            timeout_seconds=1.0,
            max_stdout_bytes=10,
            max_stderr_bytes=10,
        )

        with self.assertRaises(FrozenInstanceError):
            spec.cwd = "/other"

    def test_generated_output_plan_copies_mutable_inputs(self) -> None:
        suffixes = [".png"]
        media_types = {".png": "image/png"}
        plan = GeneratedOutputPlan(
            prefix_name="page",
            display_prefix="page",
            allowed_suffixes=tuple(suffixes),
            suffix_media_types=media_types,
            max_files=1,
            max_file_bytes=2,
            max_total_bytes=3,
            max_inline_bytes=4,
        )

        suffixes.append(".jpg")
        media_types[".png"] = "application/octet-stream"

        self.assertEqual(plan.allowed_suffixes, (".png",))
        self.assertEqual(plan.suffix_media_types, {".png": "image/png"})

    def test_generated_output_plan_is_frozen(self) -> None:
        plan = GeneratedOutputPlan(
            prefix_name="page",
            display_prefix="page",
            allowed_suffixes=(".png",),
            suffix_media_types={".png": "image/png"},
            max_files=1,
            max_file_bytes=2,
            max_total_bytes=3,
            max_inline_bytes=4,
        )

        with self.assertRaises(FrozenInstanceError):
            plan.prefix_name = "other"

    def test_generated_output_plan_rejects_invalid_fields(self) -> None:
        valid = {
            "prefix_name": "page",
            "display_prefix": "page",
            "allowed_suffixes": (".png",),
            "suffix_media_types": {".png": "image/png"},
            "max_files": 1,
            "max_file_bytes": 2,
            "max_total_bytes": 3,
            "max_inline_bytes": 4,
        }
        invalid_values = {
            "prefix_name": "../page",
            "display_prefix": "",
            "allowed_suffixes": [".png"],
            "suffix_media_types": {".png": "not a media type"},
            "max_files": -1,
            "max_file_bytes": True,
            "max_total_bytes": -1,
            "max_inline_bytes": -1,
            "max_raster_long_edge_pixels": -1,
            "max_raster_pixels": True,
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(valid)
                kwargs[field_name] = value
                with self.assertRaises(AssertionError):
                    GeneratedOutputPlan(**kwargs)  # type: ignore[arg-type]

    def test_execution_spec_rejects_generated_output_without_placeholder(
        self,
    ) -> None:
        plan = GeneratedOutputPlan(
            prefix_name="page",
            display_prefix="GENERATED_PREFIX",
            allowed_suffixes=(".png",),
            suffix_media_types={".png": "image/png"},
            max_files=1,
            max_file_bytes=2,
            max_total_bytes=3,
            max_inline_bytes=4,
        )

        with self.assertRaises(AssertionError):
            _create_execution_spec(
                backend="local",
                tool_name="shell.pdftoppm",
                command="pdftoppm",
                executable="/trusted/bin/pdftoppm",
                argv=("pdftoppm", "page"),
                display_argv=("pdftoppm", plan.display_prefix),
                cwd="/workspace",
                display_cwd=".",
                env={},
                stdin=None,
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.GENERATED_FILES,
                resource_class="heavy",
                output_plan=plan,
                timeout_seconds=1.0,
                max_stdout_bytes=1,
                max_stderr_bytes=1,
            )

        spec = _create_execution_spec(
            backend="local",
            tool_name="shell.pdftoppm",
            command="pdftoppm",
            executable="/trusted/bin/pdftoppm",
            argv=("pdftoppm", GENERATED_OUTPUT_PREFIX_PLACEHOLDER),
            display_argv=("pdftoppm", plan.display_prefix),
            cwd="/workspace",
            display_cwd=".",
            env={},
            stdin=None,
            stdout_media_type="application/json",
            output_kind=ShellOutputKind.GENERATED_FILES,
            resource_class="heavy",
            output_plan=plan,
            timeout_seconds=1.0,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
        )

        self.assertEqual(spec.output_plan, plan)

    def test_generated_file_defaults_and_metadata_copy(self) -> None:
        metadata = {"page": 1}
        generated_file = GeneratedFile(
            display_path="page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=10,
            metadata=metadata,
        )

        metadata["page"] = 2

        self.assertIsNone(generated_file.sha256)
        self.assertIsNone(generated_file.page)
        self.assertIsNone(generated_file.width)
        self.assertIsNone(generated_file.height)
        self.assertIsNone(generated_file.content_base64)
        self.assertFalse(generated_file.truncated)
        self.assertEqual(generated_file.metadata, {"page": 1})

    def test_generated_file_accepts_optional_metadata_fields(self) -> None:
        generated_file = GeneratedFile(
            display_path="page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=10,
            sha256="a" * 64,
            page=1,
            width=2,
            height=3,
            content_base64="YWJj",
            truncated=True,
        )

        self.assertEqual(generated_file.sha256, "a" * 64)
        self.assertEqual(generated_file.content_base64, "YWJj")
        self.assertTrue(generated_file.truncated)

    def test_generated_file_is_frozen(self) -> None:
        generated_file = GeneratedFile(
            display_path="page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=10,
        )

        with self.assertRaises(FrozenInstanceError):
            generated_file.bytes = 20

    def test_generated_file_rejects_invalid_fields(self) -> None:
        valid = {
            "display_path": "page-1.png",
            "media_type": "image/png",
            "suffix": ".png",
            "bytes": 10,
        }
        invalid_values = {
            "display_path": "",
            "media_type": "not a media type",
            "suffix": "../png",
            "bytes": -1,
            "sha256": "A" * 64,
            "page": -1,
            "width": True,
            "height": -1,
            "content_base64": "",
            "truncated": 1,
            "metadata": [],
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(valid)
                kwargs[field_name] = value
                with self.assertRaises(AssertionError):
                    GeneratedFile(**kwargs)  # type: ignore[arg-type]

    def test_execution_result_defaults_and_metadata_copy(self) -> None:
        metadata = {"duration_source": "clock"}
        result = ExecutionResult(
            backend="local",
            tool_name="shell.rg",
            command="rg",
            argv=("/usr/bin/rg", "needle"),
            display_argv=("rg", "needle"),
            cwd="/workspace",
            display_cwd=".",
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="match\n",
            stderr="",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            metadata=metadata,
        )

        metadata["duration_source"] = "changed"

        self.assertEqual(result.generated_files, ())
        self.assertEqual(result.stdout_bytes, 0)
        self.assertEqual(result.stderr_bytes, 0)
        self.assertFalse(result.stdout_truncated)
        self.assertFalse(result.stderr_truncated)
        self.assertFalse(result.timed_out)
        self.assertFalse(result.cancelled)
        self.assertEqual(result.duration_ms, 0)
        self.assertIsNone(result.error_code)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.metadata, {"duration_source": "clock"})

    def test_execution_result_accepts_generated_files_and_errors(
        self,
    ) -> None:
        generated_file = GeneratedFile(
            display_path="page-1.png",
            media_type="image/png",
            suffix=".png",
            bytes=10,
        )
        result = ExecutionResult(
            backend="local",
            tool_name="shell.pdftoppm",
            command="pdftoppm",
            argv=("pdftoppm",),
            display_argv=("pdftoppm",),
            cwd="/workspace",
            display_cwd=".",
            status=ShellExecutionStatus.TIMEOUT,
            exit_code=None,
            stdout="partial",
            stderr="timed out",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.GENERATED_FILES,
            generated_files=(generated_file,),
            stdout_bytes=7,
            stderr_bytes=9,
            stdout_truncated=True,
            stderr_truncated=True,
            timed_out=True,
            cancelled=True,
            duration_ms=1000,
            error_code=ShellExecutionErrorCode.TIMEOUT,
            error_message="timed out",
        )

        self.assertEqual(result.generated_files, (generated_file,))
        self.assertTrue(result.timed_out)
        self.assertTrue(result.cancelled)

    def test_execution_result_rejects_invalid_fields(self) -> None:
        valid = {
            "backend": "local",
            "tool_name": "shell.rg",
            "command": "rg",
            "argv": ("rg",),
            "display_argv": ("rg",),
            "cwd": "/workspace",
            "display_cwd": ".",
            "status": ShellExecutionStatus.COMPLETED,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "stdout_media_type": "text/plain",
            "output_kind": ShellOutputKind.TEXT,
        }
        invalid_values = {
            "backend": "",
            "tool_name": "",
            "command": "",
            "argv": ["rg"],
            "display_argv": ["rg"],
            "cwd": "",
            "display_cwd": "",
            "status": "completed",
            "exit_code": True,
            "stdout": b"",
            "stderr": b"",
            "stdout_media_type": "plain",
            "output_kind": "text",
            "generated_files": [object()],
            "stdout_bytes": -1,
            "stderr_bytes": True,
            "stdout_truncated": 1,
            "stderr_truncated": 0,
            "timed_out": 1,
            "cancelled": 0,
            "duration_ms": -1,
            "error_code": "timeout",
            "error_message": "",
            "metadata": [],
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(valid)
                kwargs[field_name] = value
                with self.assertRaises(AssertionError):
                    ExecutionResult(**kwargs)  # type: ignore[arg-type]

    def test_execution_result_is_frozen(self) -> None:
        result = ExecutionResult(
            backend="local",
            tool_name="shell.rg",
            command="rg",
            argv=("rg",),
            display_argv=("rg",),
            cwd="/workspace",
            display_cwd=".",
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="",
            stderr="",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
        )

        with self.assertRaises(FrozenInstanceError):
            result.stdout = "changed"

    def test_shell_exceptions_validate_error_codes(self) -> None:
        denied = ShellPolicyDenied(
            ShellExecutionErrorCode.DENIED_COMMAND,
            "command is disabled",
        )

        self.assertIsInstance(denied, ShellToolError)
        self.assertEqual(
            denied.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )
        self.assertEqual(str(denied), "command is disabled")

        with self.assertRaises(AssertionError):
            ShellPolicyDenied(  # type: ignore[arg-type]
                "denied_command",
                "command is disabled",
            )
        with self.assertRaises(AssertionError):
            ShellToolError(ShellExecutionErrorCode.TOOL_ERROR, "")


if __name__ == "__main__":
    main()
