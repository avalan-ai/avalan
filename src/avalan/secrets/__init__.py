"""Secret storage utilities."""

from typing import Optional

try:
    from keyring import get_keyring
    from keyring.backend import KeyringBackend
except Exception:  # Module may not be installed
    get_keyring = None  # type: ignore
    KeyringBackend = object  # type: ignore


class KeyringSecrets:
    """Wrapper around :mod:`keyring` to store and retrieve secrets."""

    _SERVICE = "avalan"

    def __init__(self, ring: KeyringBackend | None = None):
        if ring is None:
            if get_keyring is None:
                raise RuntimeError("keyring package is required for secret storage")
            ring = get_keyring()
        self._ring = ring

    def read(self, key: str) -> Optional[str]:
        """Retrieve secret associated with ``key``."""
        return self._ring.get_password(self._SERVICE, key)

    def write(self, key: str, secret: str) -> None:
        """Store ``secret`` under ``key``."""
        self._ring.set_password(self._SERVICE, key, secret)

    def delete(self, key: str) -> None:
        """Remove secret stored under ``key`` if present."""
        try:
            self._ring.delete_password(self._SERVICE, key)
        except Exception:
            pass

