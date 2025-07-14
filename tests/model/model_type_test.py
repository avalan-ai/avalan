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
