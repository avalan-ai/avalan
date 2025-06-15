from __future__ import annotations

import builtins
import importlib
import sys
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import avalan.secrets as secrets


class SecretsBaseTest(TestCase):
    def test_base_methods_raise(self) -> None:
        class Dummy(secrets.Secrets):
            def read(self, key: str) -> str | None:  # type: ignore[override]
                return super().read(key)

            def write(self, key: str, secret: str) -> None:  # type: ignore[override]
                super().write(key, secret)

            def delete(self, key: str) -> None:  # type: ignore[override]
                super().delete(key)

        dummy = Dummy()
        with self.assertRaises(NotImplementedError):
            dummy.read("k")
        with self.assertRaises(NotImplementedError):
            dummy.write("k", "v")
        with self.assertRaises(NotImplementedError):
            dummy.delete("k")


class KeyringSecretsInitTest(TestCase):
    def test_init_uses_get_keyring_and_delete_handles_errors(self) -> None:
        ring = MagicMock()
        with patch.object(secrets, "get_keyring", return_value=ring):
            sec = secrets.KeyringSecrets()
        self.assertIs(sec._ring, ring)

        ring.get_password.return_value = "val"
        self.assertEqual(sec.read("key"), "val")
        ring.get_password.assert_called_once_with("avalan", "key")

        sec.write("key", "secret")
        ring.set_password.assert_called_once_with("avalan", "key", "secret")

        ring.delete_password.side_effect = Exception()
        sec.delete("key")  # should not raise
        ring.delete_password.assert_called_once_with("avalan", "key")


class ImportFallbackTest(TestCase):
    def test_import_without_keyring(self) -> None:
        real_import = builtins.__import__
        saved = sys.modules.pop("keyring", None)

        def fake_import(
            name: str, globals=None, locals=None, fromlist=(), level=0
        ):
            if name.startswith("keyring"):
                raise ModuleNotFoundError
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            mod = importlib.reload(secrets)
            self.assertIsNone(mod.get_keyring)
            self.assertIs(mod.KeyringBackend, object)

        if saved is not None:
            sys.modules["keyring"] = saved
        importlib.reload(secrets)


if __name__ == "__main__":
    main()
