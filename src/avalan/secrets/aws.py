from boto3 import client
from . import Secrets


class AwsSecrets(Secrets):
    _SERVICE = "secretsmanager"

    def __init__(self, aws_client: object | None = None):
        self._client = aws_client or client(self._SERVICE)

    def read(self, key: str) -> str | None:
        response = self._client.get_secret_value(SecretId=key)
        return response.get("SecretString")

    def write(self, key: str, secret: str) -> None:
        self._client.put_secret_value(SecretId=key, SecretString=secret)

    def delete(self, key: str) -> None:
        self._client.delete_secret(
            SecretId=key,
            ForceDeleteWithoutRecovery=True,
        )
