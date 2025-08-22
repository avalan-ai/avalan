from . import Secrets

try:
    from keyring import get_keyring
    from keyring.backend import KeyringBackend
except Exception:  # pragma: no cover - optional dependency
    get_keyring = None  # type: ignore[assignment]
    KeyringBackend = object  # type: ignore[assignment]


class KeyringSecrets(Secrets):
    """Secrets backend backed by the system keyring."""

    _SERVICE = "avalan"

    def __init__(self, ring: KeyringBackend | None = None) -> None:
        if ring is None and get_keyring:
            ring = get_keyring()
        self._ring = ring

    def read(self, key: str) -> str | None:
        """Return secret stored under *key*."""
        assert self._ring, "keyring package not installed"
        return self._ring.get_password(self._SERVICE, key)

    def write(self, key: str, secret: str) -> None:
        """Store *secret* under *key*."""
        assert self._ring, "keyring package not installed"
        self._ring.set_password(self._SERVICE, key, secret)

    def delete(self, key: str) -> None:
        """Remove secret associated with *key*."""
        assert self._ring, "keyring package not installed"
        try:
            self._ring.delete_password(self._SERVICE, key)
        except Exception:
            pass
