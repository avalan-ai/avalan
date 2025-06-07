import unittest
from argparse import Namespace
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
