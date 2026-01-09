from . import Secrets

from typing import Any

from boto3 import client


class AwsSecrets(Secrets):
    """Secrets backend using AWS Secrets Manager."""

    _SERVICE = "secretsmanager"

    def __init__(self, aws_client: Any | None = None) -> None:
        self._client: Any = aws_client or client(self._SERVICE)

    def read(self, key: str) -> str | None:
        """Return secret stored under *key*."""
        response: dict[str, Any] = self._client.get_secret_value(SecretId=key)
        return response.get("SecretString")

    def write(self, key: str, secret: str) -> None:
        """Store *secret* under *key*."""
        self._client.put_secret_value(SecretId=key, SecretString=secret)

    def delete(self, key: str) -> None:
        """Remove secret associated with *key*."""
        self._client.delete_secret(
            SecretId=key,
            ForceDeleteWithoutRecovery=True,
        )
