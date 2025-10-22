from unittest import TestCase, main
from unittest.mock import MagicMock

from avalan.secrets import KeyringSecrets


class KeyringSecretsTestCase(TestCase):
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


if __name__ == "__main__":
    main()
