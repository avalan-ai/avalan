from avalan.model.engine import Engine
from unittest import TestCase


class ModelTypePropertyTestCase(TestCase):
    def test_subclasses_use_engine_property(self) -> None:
        seen = set()
        stack = [Engine]
        subclasses = []
        while stack:
            parent = stack.pop()
            for sub in parent.__subclasses__():
                if sub not in seen:
                    seen.add(sub)
                    stack.append(sub)
                    subclasses.append(sub)
        for cls in subclasses:
            with self.subTest(model=cls):
                self.assertNotIn("model_type", cls.__dict__)

    def test_model_type_is_class_name(self) -> None:
        seen = set()
        stack = [Engine]
        classes = [Engine]
        while stack:
            parent = stack.pop()
            for sub in parent.__subclasses__():
                if sub not in seen:
                    seen.add(sub)
                    stack.append(sub)
                    classes.append(sub)

        for cls in classes:
            with self.subTest(model=cls):
                original = getattr(cls, "__abstractmethods__", None)
                if original:
                    cls.__abstractmethods__ = frozenset()
                instance = object.__new__(cls)
                result = instance.model_type
                if original is not None:
                    cls.__abstractmethods__ = original
                self.assertEqual(result, cls.__name__)
