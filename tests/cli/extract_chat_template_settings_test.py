import unittest

from avalan.cli.__main__ import CLI


class ExtractChatTemplateSettingsTestCase(unittest.TestCase):
    def test_extracts_options(self):
        argv = [
            "--foo",
            "--run-chat-enable-thinking",
            "--bar",
            "--run-chat-other",
        ]
        new_argv, opts = CLI._extract_chat_template_settings(argv)
        self.assertEqual(new_argv, ["--foo", "--bar"])
        self.assertEqual(opts, {"enable_thinking": True, "other": True})
