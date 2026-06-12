from dataclasses import fields
from types import MappingProxyType
from unittest import TestCase, main

from avalan.tool.shell.registry import SHELL_COMMAND_IDS
from avalan.tool.shell.settings import ShellToolSettings


class ShellToolSettingsTest(TestCase):
    def test_defaults_lock_complete_contract(self) -> None:
        settings = ShellToolSettings()

        self.assertEqual(settings.backend, "local")
        self.assertEqual(settings.workspace_root, ".")
        self.assertEqual(settings.cwd, ".")
        self.assertEqual(settings.max_stdin_bytes, 0)
        self.assertFalse(settings.allow_media_tools)
        self.assertFalse(settings.allow_write)
        self.assertFalse(settings.allow_shell)
        self.assertEqual(settings.allowed_commands, SHELL_COMMAND_IDS)
        self.assertEqual(settings.allowed_pdf_raster_formats, ("png",))
        self.assertEqual(settings.allowed_tesseract_output_formats, ("txt",))
        self.assertEqual(settings.allowed_tesseract_languages, ("eng",))
        self.assertIsInstance(settings.environment, MappingProxyType)
        self.assertIsInstance(settings.executable_paths, MappingProxyType)

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

    def test_rejects_invalid_scalar_settings(self) -> None:
        invalid_kwargs = (
            {"backend": "remote"},
            {"workspace_root": ""},
            {"cwd": ""},
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
    )


if __name__ == "__main__":
    main()
