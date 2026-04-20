from __future__ import annotations

from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from avalan.secrets import keyring as keyring_module
from avalan.secrets.keyring import KeyringSecrets


class KeyringModuleTest(TestCase):
    def test_methods(self) -> None:
        ring = MagicMock()
        with patch("avalan.secrets.keyring._get_keyring", return_value=ring):
            sec = KeyringSecrets()
        ring.get_password.return_value = "val"
        self.assertEqual(sec.read("k"), "val")
        ring.get_password.assert_called_once_with("avalan", "k")

        sec.write("k", "v")
        ring.set_password.assert_called_once_with("avalan", "k", "v")

        sec.delete("k")
        ring.delete_password.assert_called_once_with("avalan", "k")

    def test_delete_without_password_delete_error_guard(self) -> None:
        ring = MagicMock()
        secrets = KeyringSecrets(ring)

        with patch.object(keyring_module, "_password_delete_error", None):
            secrets.delete("k")

        ring.delete_password.assert_called_once_with("avalan", "k")


if __name__ == "__main__":
    main()
