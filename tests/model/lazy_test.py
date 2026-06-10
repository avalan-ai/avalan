from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from avalan.model.lazy import LazyExternal


class _Target:
    marker = "attribute"

    def __call__(self, *args: object, **kwargs: object) -> object:
        return ("call", args, kwargs)

    def for_model(self, *args: object, **kwargs: object) -> object:
        return ("for_model", args, kwargs)

    def from_config(self, *args: object, **kwargs: object) -> object:
        return ("from_config", args, kwargs)

    def from_pretrained(self, *args: object, **kwargs: object) -> object:
        return ("from_pretrained", args, kwargs)

    def get_config_dict(self, *args: object, **kwargs: object) -> object:
        return ("get_config_dict", args, kwargs)


class LazyExternalTestCase(TestCase):
    def test_forwards_supported_operations_to_imported_target(self) -> None:
        target = _Target()
        module = SimpleNamespace(Target=target)

        with patch("avalan.model.lazy.import_module", return_value=module):
            lazy = LazyExternal("external.module", "Target")

            self.assertEqual(
                lazy(1, named=True),
                ("call", (1,), {"named": True}),
            )
            self.assertEqual(
                lazy.for_model("id"),
                ("for_model", ("id",), {}),
            )
            self.assertEqual(
                lazy.from_config("cfg"),
                ("from_config", ("cfg",), {}),
            )
            self.assertEqual(
                lazy.from_pretrained("model"),
                ("from_pretrained", ("model",), {}),
            )
            self.assertEqual(
                lazy.get_config_dict("model"),
                ("get_config_dict", ("model",), {}),
            )
            self.assertEqual(lazy.marker, "attribute")


if __name__ == "__main__":
    main()
