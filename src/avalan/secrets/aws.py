from . import Secrets

from typing import Protocol, cast

from boto3 import client


class AwsSecretsManagerClient(Protocol):
    def get_secret_value(self, *, SecretId: str) -> dict[str, str]:
        """Get a secret from AWS Secrets Manager."""

    def put_secret_value(self, *, SecretId: str, SecretString: str) -> None:
        """Store a secret in AWS Secrets Manager."""

    def delete_secret(
        self, *, SecretId: str, ForceDeleteWithoutRecovery: bool
    ) -> None:
        """Delete a secret from AWS Secrets Manager."""


class AwsSecrets(Secrets):
    _SERVICE = "secretsmanager"

    def __init__(self, aws_client: AwsSecretsManagerClient | None = None):
        self._client = (
            aws_client
            if aws_client is not None
            else cast(AwsSecretsManagerClient, client(self._SERVICE))
        )

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
