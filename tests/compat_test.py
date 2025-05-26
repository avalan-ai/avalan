import importlib
import sys
import typing
from unittest import TestCase


class CompatImportErrorTestCase(TestCase):
    def _reload_module(self):
        if "avalan.compat" in sys.modules:
            del sys.modules["avalan.compat"]
        return importlib.import_module("avalan.compat")

    def test_override_fallback_when_missing(self):
        if hasattr(typing, "override"):
            delattr(typing, "override")
        compat = self._reload_module()
        self.assertEqual(compat.override.__module__, "avalan.compat")
        def func():
            return 1
        self.assertIs(compat.override(func), func)

    def test_override_uses_typing_override_when_available(self):
        def sentinel(func):
            func.sentinel = True
            return func
        typing.override = sentinel
        try:
            compat = self._reload_module()
            self.assertIs(compat.override, sentinel)
        finally:
            delattr(typing, "override")
            self._reload_module()
