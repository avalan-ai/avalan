import importlib
import sys
from unittest import TestCase


class CompatImportErrorTestCase(TestCase):
    def _reload_module(self):
        if "avalan.compat" in sys.modules:
            del sys.modules["avalan.compat"]
        return importlib.import_module("avalan.compat")

    def test_override_from_typing_extensions(self):
        """Test that override is imported from typing_extensions."""
        compat = self._reload_module()
        self.assertEqual(compat.override.__module__, "typing_extensions")

        def func():
            return 1

        # The override decorator should return the function unchanged
        decorated = compat.override(func)
        self.assertEqual(decorated(), 1)
