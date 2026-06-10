from importlib import import_module
from typing import Any


class LazyExternal:
    __test__ = False

    def __init__(self, module_name: str, name: str) -> None:
        self._module_name = module_name
        self._name = name

    def __call__(self, *args: object, **kwargs: object) -> Any:
        target = self._target()
        return target(*args, **kwargs)

    def for_model(self, *args: object, **kwargs: object) -> Any:
        return self._target().for_model(*args, **kwargs)

    def from_config(self, *args: object, **kwargs: object) -> Any:
        return self._target().from_config(*args, **kwargs)

    def from_pretrained(self, *args: object, **kwargs: object) -> Any:
        return self._target().from_pretrained(*args, **kwargs)

    def get_config_dict(self, *args: object, **kwargs: object) -> Any:
        return self._target().get_config_dict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "_is_coroutine" or (
            name.startswith("__") and name.endswith("__")
        ):
            raise AttributeError(name)
        return getattr(self._target(), name)

    def _target(self) -> Any:
        return getattr(import_module(self._module_name), self._name)
