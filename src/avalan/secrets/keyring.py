from . import Secrets

from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from keyring.backend import KeyringBackend
else:

    class KeyringBackend(Protocol):
        def get_password(self, service: str, key: str) -> str | None:
            """Get secret value from keyring."""

        def set_password(self, service: str, key: str, secret: str) -> None:
            """Set secret value in keyring."""

        def delete_password(self, service: str, key: str) -> None:
            """Delete secret value from keyring."""


_get_keyring: Callable[[], KeyringBackend] | None
_password_delete_error: type[Exception] | None
try:
    from keyring import get_keyring as _imported_get_keyring
    from keyring.errors import (
        PasswordDeleteError as _imported_password_delete_error,
    )
except Exception:  # pragma: no cover - optional dependency
    _get_keyring = None
    _password_delete_error = None
else:
    _get_keyring = _imported_get_keyring
    _password_delete_error = _imported_password_delete_error


class KeyringSecrets(Secrets):
    """Secrets backend backed by the system keyring."""

    _SERVICE = "avalan"

    def __init__(self, ring: KeyringBackend | None = None) -> None:
        if ring is None and _get_keyring is not None:
            ring = _get_keyring()
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
        if _password_delete_error is None:
            self._ring.delete_password(self._SERVICE, key)
            return
        try:
            self._ring.delete_password(self._SERVICE, key)
        except _password_delete_error:
            return
