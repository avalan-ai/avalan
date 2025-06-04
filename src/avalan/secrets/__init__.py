from abc import ABC, abstractmethod


class Secrets(ABC):
    @abstractmethod
    def read(self, key: str) -> str | None:
        raise NotImplementedError()

    @abstractmethod
    def write(self, key: str, secret: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError()


try:
    from keyring import get_keyring
    from keyring.backend import KeyringBackend
except Exception:  # pragma Module may not be installed
    get_keyring = None  # type: ignore
    KeyringBackend = object  # type: ignore


class KeyringSecrets:
    _SERVICE = "avalan"

    def __init__(self, ring: KeyringBackend | None = None):
        if ring is None and get_keyring:
            ring = get_keyring()
        self._ring = ring

    def read(self, key: str) -> str | None:
        assert self._ring, "keyring package not installed"
        return self._ring.get_password(self._SERVICE, key)

    def write(self, key: str, secret: str) -> None:
        assert self._ring, "keyring package not installed"
        self._ring.set_password(self._SERVICE, key, secret)

    def delete(self, key: str) -> None:
        assert self._ring, "keyring package not installed"
        try:
            self._ring.delete_password(self._SERVICE, key)
        except Exception:
            pass
