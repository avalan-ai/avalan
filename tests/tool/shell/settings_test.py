from dataclasses import fields
from types import MappingProxyType
from unittest import TestCase, main

from avalan.container import ContainerProfileSelection
from avalan.isolation import SandboxProfileSelection
from avalan.tool.shell.registry import SHELL_COMMAND_IDS
from avalan.tool.shell.settings import ShellToolSettings


class ShellToolSettingsTest(TestCase):
    def test_defaults_lock_complete_contract(self) -> None:
        settings = ShellToolSettings()
        expected_defaults = {
            "backend": "local",
            "execution_mode": "local",
            "workspace_root": ".",
            "cwd": ".",
            "materialized_input_files_dir": "avalan-input-files",
            "input_file_manifest_enabled": True,
            "input_file_manifest_message": (
                "Attached files available to tools:"
            ),
            "input_file_manifest_path_message": (
                "Use these path values as tool arguments."
            ),
            "default_timeout_seconds": 10.0,
            "max_timeout_seconds": 60.0,
            "max_stdout_bytes": 65536,
            "max_stderr_bytes": 32768,
            "max_stdin_bytes": 0,
            "max_pipeline_stages": 8,
            "max_pipeline_bytes": 1048576,
            "max_intermediate_bytes": 1048576,
            "max_arguments": 128,
            "max_argument_bytes": 8192,
            "max_command_bytes": 32768,
            "max_path_count": 128,
            "max_glob_count": 32,
            "max_glob_bytes_per_glob": 2048,
            "max_total_glob_bytes": 8192,
            "max_full_file_bytes": 1048576,
            "max_rg_columns": 1000,
            "max_rg_context_lines": 10,
            "max_rg_matches_per_file": 1000,
            "max_head_lines": 500,
            "max_tail_lines": 500,
            "max_text_filter_input_bytes": 1048576,
            "max_filter_program_bytes": 8192,
            "max_filter_pattern_bytes": 2048,
            "max_filter_selectors": 32,
            "max_awk_fields": 64,
            "max_awk_separator_bytes": 16,
            "max_json_input_bytes": 5242880,
            "max_jq_filter_bytes": 4096,
            "max_pdf_input_bytes": 104857600,
            "max_pdf_text_pages": 50,
            "max_pdf_raster_pages": 8,
            "max_pdf_raster_dpi": 600,
            "max_raster_long_edge_pixels": 2048,
            "max_raster_pixels": 40000000,
            "max_output_files": 8,
            "max_output_file_bytes": 10485760,
            "max_total_output_file_bytes": 52428800,
            "max_inline_output_file_bytes": 2097152,
            "max_ocr_input_bytes": 26214400,
            "max_ocr_pixels": 20000000,
            "max_ocr_languages": 4,
            "max_tesseract_dpi": 600,
            "stream_read_chunk_bytes": 8192,
            "max_concurrent_processes": 4,
            "max_concurrent_heavy_processes": 1,
            "default_pdf_timeout_seconds": 30.0,
            "max_pdf_timeout_seconds": 120.0,
            "default_ocr_timeout_seconds": 60.0,
            "max_ocr_timeout_seconds": 300.0,
            "tesseract_thread_limit": 1,
            "allow_pipelines": False,
            "allow_media_tools": False,
            "allow_write": False,
            "allow_shell": False,
            "allow_absolute_paths": False,
            "allow_symlinks": False,
            "allow_hidden": False,
        }

        for field_name, value in expected_defaults.items():
            with self.subTest(field_name=field_name):
                self.assertEqual(getattr(settings, field_name), value)
        self.assertEqual(settings.allowed_commands, SHELL_COMMAND_IDS)
        self.assertEqual(settings.allowed_pdf_raster_formats, ("png",))
        self.assertEqual(settings.allowed_tesseract_output_formats, ("txt",))
        self.assertEqual(settings.allowed_tesseract_languages, ("eng",))
        self.assertIsInstance(settings.environment, MappingProxyType)
        self.assertIsInstance(settings.executable_paths, MappingProxyType)

    def test_cli_scalar_fields_expose_only_safe_scalar_settings(self) -> None:
        field_names = {field.name for field in fields(ShellToolSettings)}
        scalar_fields = set(ShellToolSettings.CLI_SCALAR_FIELDS)

        self.assertLess(scalar_fields, field_names)
        self.assertIn("allow_media_tools", scalar_fields)
        self.assertIn("materialized_input_files_dir", scalar_fields)
        self.assertIn("input_file_manifest_enabled", scalar_fields)
        self.assertIn("input_file_manifest_message", scalar_fields)
        self.assertIn("input_file_manifest_path_message", scalar_fields)
        self.assertIn("allow_pipelines", scalar_fields)
        self.assertIn("max_stdout_bytes", scalar_fields)
        self.assertIn("max_pipeline_stages", scalar_fields)
        self.assertIn("max_pipeline_bytes", scalar_fields)
        self.assertIn("max_intermediate_bytes", scalar_fields)
        self.assertNotIn("allowed_commands", scalar_fields)
        self.assertNotIn("environment", scalar_fields)
        self.assertNotIn("environment_allowlist", scalar_fields)
        self.assertNotIn("executable_paths", scalar_fields)
        self.assertNotIn("executable_search_paths", scalar_fields)
        self.assertNotIn("container", scalar_fields)
        self.assertNotIn("sandbox", scalar_fields)
        self.assertNotIn("allow_write", scalar_fields)
        self.assertNotIn("allow_shell", scalar_fields)

    def test_execution_mode_and_backend_alias_are_synchronized(self) -> None:
        canonical = ShellToolSettings(execution_mode="sandbox")
        legacy = ShellToolSettings(backend="container")

        self.assertEqual(canonical.execution_mode, "sandbox")
        self.assertEqual(canonical.backend, "sandbox")
        self.assertEqual(legacy.execution_mode, "container")
        self.assertEqual(legacy.backend, "container")
        with self.assertRaises(AssertionError):
            ShellToolSettings(
                execution_mode="sandbox",
                backend="container",
            )

    def test_isolated_modes_and_profile_selection_are_trusted_settings(
        self,
    ) -> None:
        container_selection = ContainerProfileSelection(
            profile="workspace-readonly",
            required=True,
        )
        sandbox_selection = SandboxProfileSelection(
            profile="workspace-readonly",
            required=True,
        )
        container_settings = ShellToolSettings(
            execution_mode="container",
            container=container_selection,
        )
        sandbox_settings = ShellToolSettings(
            execution_mode="sandbox",
            sandbox=sandbox_selection,
        )

        self.assertEqual(container_settings.execution_mode, "container")
        self.assertIs(container_settings.container, container_selection)
        self.assertEqual(sandbox_settings.execution_mode, "sandbox")
        self.assertIs(sandbox_settings.sandbox, sandbox_selection)

    def test_mixed_mode_policy_is_rejected(self) -> None:
        container_selection = ContainerProfileSelection(
            profile="workspace-readonly",
            required=True,
        )
        sandbox_selection = SandboxProfileSelection(
            profile="workspace-readonly",
            required=True,
        )

        with self.assertRaises(AssertionError):
            ShellToolSettings(
                execution_mode="container",
                sandbox=sandbox_selection,
            )
        with self.assertRaises(AssertionError):
            ShellToolSettings(
                execution_mode="sandbox",
                container=container_selection,
            )
        with self.assertRaises(AssertionError):
            ShellToolSettings(container=container_selection)
        with self.assertRaises(AssertionError):
            ShellToolSettings(sandbox=sandbox_selection)

    def test_mutable_inputs_are_copied(self) -> None:
        allowed_commands = ["rg"]
        environment = {"LC_ALL": "C"}
        environment_allowlist = ["PATH"]
        executable_paths = {"rg": "/usr/bin/rg"}
        executable_search_paths = ["/usr/bin"]
        settings = ShellToolSettings(
            allowed_commands=allowed_commands,
            environment=environment,
            environment_allowlist=environment_allowlist,
            executable_paths=executable_paths,
            executable_search_paths=executable_search_paths,
        )

        allowed_commands.append("cat")
        environment["LC_ALL"] = "changed"
        environment_allowlist.append("HOME")
        executable_paths["rg"] = "/bin/rg"
        executable_search_paths.append("/bin")

        self.assertEqual(settings.allowed_commands, ("rg",))
        self.assertEqual(settings.environment, {"LC_ALL": "C"})
        self.assertEqual(settings.environment_allowlist, ("PATH",))
        self.assertEqual(settings.executable_paths, {"rg": "/usr/bin/rg"})
        self.assertEqual(settings.executable_search_paths, ("/usr/bin",))

    def test_numeric_settings_reject_bool_as_int(self) -> None:
        for field_name in _numeric_field_names():
            with self.subTest(field_name=field_name):
                with self.assertRaises(AssertionError):
                    ShellToolSettings(**{field_name: True})

    def test_boolean_settings_reject_ints(self) -> None:
        for field_name in _boolean_field_names():
            with self.subTest(field_name=field_name):
                with self.assertRaises(AssertionError):
                    ShellToolSettings(**{field_name: 1})

    def test_rejects_reserved_write_and_shell_settings(self) -> None:
        with self.assertRaises(AssertionError):
            ShellToolSettings(allow_write=True)
        with self.assertRaises(AssertionError):
            ShellToolSettings(allow_shell=True)

    def test_pipeline_settings_accept_positive_boundaries(self) -> None:
        settings = ShellToolSettings(
            allow_pipelines=True,
            max_pipeline_stages=1,
            max_pipeline_bytes=1,
            max_intermediate_bytes=1,
        )

        self.assertTrue(settings.allow_pipelines)
        self.assertEqual(settings.max_pipeline_stages, 1)
        self.assertEqual(settings.max_pipeline_bytes, 1)
        self.assertEqual(settings.max_intermediate_bytes, 1)

    def test_pipeline_integer_settings_reject_zero_and_negative(self) -> None:
        for field_name in (
            "max_pipeline_stages",
            "max_pipeline_bytes",
            "max_intermediate_bytes",
        ):
            for value in (0, -1):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(AssertionError):
                        ShellToolSettings(**{field_name: value})

    def test_rejects_invalid_scalar_settings(self) -> None:
        invalid_kwargs = (
            {"backend": "remote"},
            {"execution_mode": "remote"},
            {"workspace_root": ""},
            {"cwd": ""},
            {"materialized_input_files_dir": ""},
            {"materialized_input_files_dir": "/tmp/inputs"},
            {"materialized_input_files_dir": "../inputs"},
            {"input_file_manifest_message": ""},
            {"input_file_manifest_path_message": ""},
            {"max_stdout_bytes": 0},
            {"max_stdin_bytes": 1},
            {"default_timeout_seconds": 61.0},
            {"default_pdf_timeout_seconds": 121.0},
            {"default_ocr_timeout_seconds": 301.0},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ShellToolSettings(**kwargs)  # type: ignore[arg-type]

    def test_rejects_invalid_trusted_lists(self) -> None:
        invalid_kwargs = (
            {"allowed_commands": ()},
            {"allowed_commands": ("unknown",)},
            {"allowed_pdf_raster_formats": ()},
            {"allowed_pdf_raster_formats": ("jpg",)},
            {"allowed_tesseract_output_formats": ()},
            {"allowed_tesseract_output_formats": ("pdf",)},
            {"allowed_tesseract_languages": ()},
            {"allowed_tesseract_languages": ("eng+spa",)},
            {"allowed_tesseract_languages": ("-c",)},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ShellToolSettings(**kwargs)  # type: ignore[arg-type]

    def test_rejects_invalid_environment_settings(self) -> None:
        invalid_kwargs = (
            {"environment": {"1BAD": "value"}},
            {"environment": {"GOOD": ""}},
            {"environment_allowlist": ("",)},
            {"environment_allowlist": ("1BAD",)},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ShellToolSettings(**kwargs)  # type: ignore[arg-type]

    def test_rejects_invalid_executable_path_settings(self) -> None:
        invalid_kwargs = (
            {"executable_paths": []},
            {"executable_paths": {"rg": "relative"}},
            {"executable_search_paths": "/usr/bin"},
            {"executable_search_paths": ("relative",)},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ShellToolSettings(**kwargs)  # type: ignore[arg-type]


def _numeric_field_names() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(ShellToolSettings)
        if field.name.startswith(("default_", "max_"))
        or field.name
        in (
            "stream_read_chunk_bytes",
            "tesseract_thread_limit",
        )
    )


def _boolean_field_names() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(ShellToolSettings)
        if field.name.startswith("allow_")
        or field.name == "input_file_manifest_enabled"
    )


if __name__ == "__main__":
    main()
