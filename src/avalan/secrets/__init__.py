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


from .keyring import KeyringSecrets as KeyringSecrets  # noqa: E402
