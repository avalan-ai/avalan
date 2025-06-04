from . import Secrets
from keyring import get_keyring
from keyring.backend import KeyringBackend


class KeyringSecrets(Secrets):
    _SERVICE = "avalan"

    def __init__(self, ring: KeyringBackend | None = None):
        self._ring = get_keyring()

    def read(self, key: str) -> str | None:
        return self._ring.get_password(self._SERVICE, key)

    def write(self, key: str, secret: str) -> None:
        self._ring.set_password(self._SERVICE, key, secret)

    def delete(self, key: str) -> None:
        self._ring.delete_password(self._SERVICE, key)
