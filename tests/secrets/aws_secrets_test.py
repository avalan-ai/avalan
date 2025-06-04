from avalan.secrets.aws import AwsSecrets
from unittest import TestCase, main
from unittest.mock import MagicMock


class AwsSecretsTestCase(TestCase):
    def test_read_write_delete(self):
        client = MagicMock()
        client.get_secret_value.return_value = {"SecretString": "val"}
        secrets = AwsSecrets(client)

        self.assertEqual(secrets.read("k"), "val")
        client.get_secret_value.assert_called_once_with(SecretId="k")

        secrets.write("k", "v")
        client.put_secret_value.assert_called_once_with(
            SecretId="k", SecretString="v"
        )

        secrets.delete("k")
        client.delete_secret.assert_called_once_with(
            SecretId="k", ForceDeleteWithoutRecovery=True
        )


if __name__ == "__main__":
    main()
