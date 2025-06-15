import io
import unittest
from argparse import Namespace
from tempfile import NamedTemporaryFile

from avalan.cli.commands import agent as agent_cmds
from avalan.tool.browser import BrowserToolSettings


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
