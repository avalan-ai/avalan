from unittest import TestCase, main
from unittest.mock import MagicMock

from avalan.secrets import KeyringSecrets
from avalan.secrets import keyring as keyring_module


class KeyringSecretsTestCase(TestCase):
    def test_read_write_delete(self):
        ring = MagicMock()
        ring.get_password.return_value = "val"
        secrets = KeyringSecrets(ring)

        self.assertEqual(secrets.read("k"), "val")
        ring.get_password.assert_called_once_with("avalan", "k")

        secrets.write("k", "v")
        ring.set_password.assert_called_once_with("avalan", "k", "v")

        if keyring_module._password_delete_error is None:
            ring.delete_password.side_effect = RuntimeError()
            with self.assertRaises(RuntimeError):
                secrets.delete("k")
        else:
            ring.delete_password.side_effect = (
                keyring_module._password_delete_error()
            )
            secrets.delete("k")

        ring.delete_password.assert_called_once_with("avalan", "k")


if __name__ == "__main__":
    main()
