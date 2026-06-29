import io
import unittest
from argparse import Namespace
from tempfile import NamedTemporaryFile

from avalan.cli.commands import agent as agent_cmds
from avalan.tool.browser import BrowserToolSettings
from avalan.tool.database import DatabaseToolSettings
from avalan.tool.shell import ShellToolSettings


class GetToolSettingsTestCase(unittest.TestCase):
    def test_values(self):
        args = Namespace(
            tool_browser_engine="webkit",
            tool_browser_debug=True,
            tool_browser_search=False,
            tool_browser_search_context=7,
        )
        settings = agent_cmds.get_tool_settings(
            args, prefix="browser", settings_cls=BrowserToolSettings
        )
        self.assertIsInstance(settings, BrowserToolSettings)
        self.assertEqual(settings.engine, "webkit")
        self.assertTrue(settings.debug)
        self.assertFalse(settings.search)
        self.assertEqual(settings.search_context, 7)

    def test_defaults(self):
        args = Namespace(
            tool_browser_engine=None,
            tool_browser_debug=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
        )
        settings = agent_cmds.get_tool_settings(
            args, prefix="browser", settings_cls=BrowserToolSettings
        )
        self.assertIsNone(settings)

    def test_debug_source_opened(self):
        with NamedTemporaryFile("w+") as tmp:
            args = Namespace(tool_browser_debug_source=tmp.name)
            settings = agent_cmds.get_tool_settings(
                args, prefix="browser", settings_cls=BrowserToolSettings
            )
            self.assertIsInstance(settings.debug_source, io.TextIOBase)
            self.assertFalse(settings.debug_source.closed)
            settings.debug_source.close()

    def test_from_dict_mapping(self):
        cfg = {"engine": "chromium", "debug": True}
        settings = agent_cmds._tool_settings_from_mapping(
            cfg, settings_cls=BrowserToolSettings, open_files=False
        )
        self.assertEqual(settings.engine, "chromium")
        self.assertTrue(settings.debug)

    def test_prefix_fallback_to_field_name(self):
        cfg = {"engine": "chromium"}
        settings = agent_cmds._tool_settings_from_mapping(
            cfg,
            prefix="browser",
            settings_cls=BrowserToolSettings,
            open_files=False,
        )
        self.assertEqual(settings.engine, "chromium")

    def test_database_settings(self):
        args = Namespace(tool_database_dsn="sqlite:///db.sqlite")
        settings = agent_cmds.get_tool_settings(
            args, prefix="database", settings_cls=DatabaseToolSettings
        )
        self.assertIsInstance(settings, DatabaseToolSettings)
        self.assertEqual(settings.dsn, "sqlite:///db.sqlite")

    def test_database_allowed_commands_single_cli_value(self):
        args = Namespace(
            tool_database_dsn="sqlite:///db.sqlite",
            tool_database_allowed_commands="select",
        )
        settings = agent_cmds.get_tool_settings(
            args, prefix="database", settings_cls=DatabaseToolSettings
        )

        self.assertIsInstance(settings, DatabaseToolSettings)
        self.assertEqual(settings.allowed_commands, ["select"])

    def test_database_allowed_commands_preserves_list(self):
        args = Namespace(
            tool_database_dsn="sqlite:///db.sqlite",
            tool_database_allowed_commands=["select"],
        )
        settings = agent_cmds.get_tool_settings(
            args, prefix="database", settings_cls=DatabaseToolSettings
        )

        self.assertIsInstance(settings, DatabaseToolSettings)
        self.assertEqual(settings.allowed_commands, ["select"])

    def test_database_allowed_commands_repeated_values(self):
        args = Namespace(
            tool_database_dsn="sqlite:///db.sqlite",
            tool_database_allowed_commands=["select", "insert"],
        )
        settings = agent_cmds.get_tool_settings(
            args, prefix="database", settings_cls=DatabaseToolSettings
        )

        self.assertIsInstance(settings, DatabaseToolSettings)
        self.assertEqual(settings.allowed_commands, ["select", "insert"])

    def test_database_allowed_commands_mapping_list_value(self):
        settings = agent_cmds._tool_settings_from_mapping(
            {
                "dsn": "sqlite:///db.sqlite",
                "allowed_commands": ["select", "update"],
            },
            prefix="database",
            settings_cls=DatabaseToolSettings,
            open_files=False,
        )

        self.assertIsInstance(settings, DatabaseToolSettings)
        self.assertEqual(settings.allowed_commands, ["select", "update"])

    def test_database_allowed_commands_mapping_scalar_value(self):
        settings = agent_cmds._tool_settings_from_mapping(
            {
                "dsn": "sqlite:///db.sqlite",
                "allowed_commands": "select",
            },
            prefix="database",
            settings_cls=DatabaseToolSettings,
            open_files=False,
        )

        self.assertIsInstance(settings, DatabaseToolSettings)
        self.assertEqual(settings.allowed_commands, ["select"])

    def test_database_allowed_commands_rejects_invalid_values(self):
        cases = (
            {"command": "select"},
            b"select",
            [""],
        )

        for allowed_commands in cases:
            with self.subTest(allowed_commands=allowed_commands):
                with self.assertRaises(AssertionError):
                    agent_cmds.get_tool_settings(
                        Namespace(
                            tool_database_dsn="sqlite:///db.sqlite",
                            tool_database_allowed_commands=allowed_commands,
                        ),
                        prefix="database",
                        settings_cls=DatabaseToolSettings,
                    )

    def test_shell_settings(self):
        args = Namespace(
            tool_shell_allow_media_tools=True,
            tool_shell_allow_pipelines=True,
            tool_shell_max_pipeline_stages=3,
            tool_shell_max_pipeline_bytes=1024,
            tool_shell_max_intermediate_bytes=512,
            tool_shell_pipeline_transport="native",
        )
        settings = agent_cmds.get_tool_settings(
            args, prefix="shell", settings_cls=ShellToolSettings
        )
        self.assertIsInstance(settings, ShellToolSettings)
        self.assertTrue(settings.allow_media_tools)
        self.assertTrue(settings.allow_pipelines)
        self.assertEqual(settings.max_pipeline_stages, 3)
        self.assertEqual(settings.max_pipeline_bytes, 1024)
        self.assertEqual(settings.max_intermediate_bytes, 512)
        self.assertEqual(settings.pipeline_transport, "native")

    def test_shell_pipeline_explicit_fields_are_tracked(self):
        args = Namespace(
            tool_shell_allow_pipelines=True,
            tool_shell_max_pipeline_stages=3,
            tool_shell_max_pipeline_bytes=None,
            tool_shell_max_intermediate_bytes=512,
            tool_shell_pipeline_transport="native",
        )

        explicit_fields = (
            agent_cmds._tool_settings_explicit_fields_from_mapping(
                args,
                prefix="shell",
                settings_cls=ShellToolSettings,
            )
        )

        self.assertEqual(
            explicit_fields,
            frozenset(
                {
                    "allow_pipelines",
                    "max_pipeline_stages",
                    "max_intermediate_bytes",
                    "pipeline_transport",
                }
            ),
        )

    def test_shell_pipeline_settings_reject_invalid_values(self):
        cases = (
            Namespace(tool_shell_max_pipeline_stages=True),
            Namespace(tool_shell_max_pipeline_stages=0),
            Namespace(tool_shell_max_pipeline_bytes=True),
            Namespace(tool_shell_max_pipeline_bytes=0),
            Namespace(tool_shell_max_intermediate_bytes=True),
            Namespace(tool_shell_max_intermediate_bytes=0),
            Namespace(tool_shell_pipeline_transport="shell"),
        )

        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(AssertionError):
                    agent_cmds.get_tool_settings(
                        args,
                        prefix="shell",
                        settings_cls=ShellToolSettings,
                    )

    def test_shell_settings_accepts_manifest_options(self):
        args = Namespace(
            tool_shell_input_file_manifest_enabled=False,
            tool_shell_input_file_manifest_message="Use attached paths:",
            tool_shell_input_file_manifest_path_message="Pass them to tools.",
        )

        settings = agent_cmds.get_tool_settings(
            args, prefix="shell", settings_cls=ShellToolSettings
        )

        self.assertIsInstance(settings, ShellToolSettings)
        self.assertFalse(settings.input_file_manifest_enabled)
        self.assertEqual(
            settings.input_file_manifest_message,
            "Use attached paths:",
        )
        self.assertEqual(
            settings.input_file_manifest_path_message,
            "Pass them to tools.",
        )

    def test_shell_settings_executable_resolution(self):
        args = Namespace(
            tool_shell_executable_search_paths=["/usr/bin", "/bin"],
            tool_shell_executable_paths=[
                ("rg", "/usr/bin/rg"),
                ("cat", "/bin/cat"),
            ],
        )

        settings = agent_cmds.get_tool_settings(
            args, prefix="shell", settings_cls=ShellToolSettings
        )

        self.assertIsInstance(settings, ShellToolSettings)
        self.assertEqual(
            settings.executable_search_paths,
            ("/usr/bin", "/bin"),
        )
        self.assertEqual(
            settings.executable_paths,
            {"rg": "/usr/bin/rg", "cat": "/bin/cat"},
        )

    def test_shell_settings_rejects_unknown_executable_mapping(self):
        args = Namespace(
            tool_shell_executable_paths=[("unknown", "/usr/bin/unknown")],
        )

        with self.assertRaises(AssertionError):
            agent_cmds.get_tool_settings(
                args, prefix="shell", settings_cls=ShellToolSettings
            )

    def test_shell_settings_rejects_relative_executable_search_path(self):
        args = Namespace(tool_shell_executable_search_paths=["bin"])

        with self.assertRaises(AssertionError):
            agent_cmds.get_tool_settings(
                args, prefix="shell", settings_cls=ShellToolSettings
            )

    def test_shell_executable_path_coercion_preserves_non_cli_values(self):
        mapping = {"rg": "/usr/bin/rg"}

        self.assertIs(
            agent_cmds._coerce_shell_tool_setting_value(
                "executable_paths", mapping
            ),
            mapping,
        )
        self.assertEqual(
            agent_cmds._coerce_shell_tool_setting_value("cwd", "."),
            ".",
        )
        self.assertEqual(
            agent_cmds._coerce_shell_tool_setting_value(
                "executable_paths", "rg=/usr/bin/rg"
            ),
            "rg=/usr/bin/rg",
        )
        self.assertFalse(agent_cmds._is_tuple_pair_sequence("rg"))

    def test_agent_enabled_tools_preserves_omitted_and_explicit_selection(
        self,
    ):
        self.assertIsNone(
            agent_cmds._agent_enabled_tools(
                Namespace(tool=None, tools=None),
            )
        )
        self.assertEqual(
            agent_cmds._agent_enabled_tools(
                Namespace(tool=["shell.rg"], tools=None),
            ),
            ["shell.rg"],
        )
        self.assertEqual(
            agent_cmds._agent_enabled_tools(
                Namespace(tool=None, tools=["math"]),
            ),
            ["math"],
        )

    def test_uses_ds4_backend_requires_engine_agent(self):
        orchestrator = Namespace(engine_agent=None, engine=Namespace())

        self.assertFalse(agent_cmds._uses_ds4_backend(orchestrator))

    def test_uses_ds4_backend_detects_uri_backend(self):
        orchestrator = Namespace(
            engine_agent=Namespace(
                engine_uri=Namespace(
                    params={"backend": agent_cmds.Backend.DS4}
                )
            ),
            engine=Namespace(model_type="transformers"),
        )

        self.assertTrue(agent_cmds._uses_ds4_backend(orchestrator))
