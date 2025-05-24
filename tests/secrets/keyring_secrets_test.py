from avalan.secrets import KeyringSecrets
from unittest import TestCase, main
from unittest.mock import MagicMock, patch


class KeyringSecretsTestCase(TestCase):
    def test_init_with_get_keyring(self):
        ring = MagicMock()
        with patch("avalan.secrets.get_keyring", return_value=ring):
            secrets = KeyringSecrets()
        self.assertIs(secrets._ring, ring)

    def test_read_write_delete(self):
        ring = MagicMock()
        ring.get_password.return_value = "val"
        secrets = KeyringSecrets(ring)

        self.assertEqual(secrets.read("k"), "val")
        ring.get_password.assert_called_once_with("avalan", "k")

        secrets.write("k", "v")
        ring.set_password.assert_called_once_with("avalan", "k", "v")

        ring.delete_password.side_effect = Exception()
        secrets.delete("k")
        ring.delete_password.assert_called_once_with("avalan", "k")

    def test_read_without_backend(self):
        with patch("avalan.secrets.get_keyring", None):
            secrets = KeyringSecrets()
        with self.assertRaises(AssertionError):
            secrets.read("k")


if __name__ == "__main__":
    main()
