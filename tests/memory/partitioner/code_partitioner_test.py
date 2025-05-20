from avalan.memory.partitioner import Encoding
from avalan.memory.partitioner.code import CodePartitioner, LanguageName
from logging import Logger
from unittest import main, TestCase
from unittest.mock import MagicMock


class CodePartitionerTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources : list[tuple[LanguageName,str,int,Encoding,str]] = [
            ("python", "ai.avalan.tests.memory.partitioner", 256, "utf-8", """
#!shebang
# License blah blah (Apache 2.0)
"This is a module docstring."

from datetime import datetime


a = 1

'''This
is
not
a
multiline
comment.'''

b = 2

class Test:
    "This is a class docstring."

    'This is bogus.'

    def test(self):
        "This is a function docstring."

        "Please, no."

        return 1

    def test_2(self, name: str, age: int) -> list[str]:
        return [
            "Hello ",
            name
        ]

    def test_3(
        cls,
        first,
        *,
        extra: datetime,
        second: bool = True,
        third: Node | None = None,
        **kwargs,
        fourth: float = 3.14159
    ):
        pass

c = 3
            """,
            )
        ]

    def test_instantiation_with_load_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        for language_name, namespace, max_chars, encoding, source \
            in self.sources:
            with self.subTest():
                partitioner = CodePartitioner(logger=logger_mock)
                self.assertIsInstance(partitioner, CodePartitioner)

                partitions, functions = partitioner.partition(
                    language_name,
                    source,
                    encoding,
                    max_chars=max_chars,
                    namespace=namespace
                )

                self.assertEqual(len(partitions), 6)
                self.assertEqual(
                    [len(p.data) for p in partitions],
                    [159, 13, 157, 123, 215, 7]
                )
                self.assertEqual(partitions[0].encoding, encoding)
                self.assertIsNone(partitions[0].function_id)
                self.assertEqual(partitions[1].encoding, encoding)
                self.assertIsNone(partitions[1].function_id)
                self.assertEqual(partitions[2].encoding, encoding)
                self.assertEqual(
                    partitions[2].function_id,
                    f"{namespace}.Test.test_2"
                )
                self.assertEqual(partitions[3].encoding, encoding)
                self.assertEqual(
                    partitions[3].function_id,
                    f"{namespace}.Test.test_3"
                )
                self.assertEqual(partitions[4].encoding, encoding)
                self.assertEqual(
                    partitions[4].function_id,
                    f"{namespace}.Test.test_3"
                )
                self.assertEqual(partitions[5].encoding, encoding)
                self.assertIsNone(partitions[5].function_id)

                self.assertEqual(len(functions), 3)
                self.assertEqual(
                    functions[0].id,
                    f"{namespace}.Test.test"
                )
                self.assertEqual(functions[0].namespace, namespace)
                self.assertEqual(functions[0].class_name, "Test")
                self.assertEqual(functions[0].name, "test")
                self.assertEqual(len(functions[0].parameters), 1)
                self.assertEqual(
                    functions[0].parameters[0].parameter_type,
                    "identifier"
                )
                self.assertIsNone(functions[0].parameters[0].type)
                self.assertEqual(functions[0].parameters[0].name, "self")
                self.assertIsNone(functions[0].return_type)

                self.assertEqual(functions[1].id, f"{namespace}.Test.test_2")
                self.assertEqual(functions[1].namespace, namespace)
                self.assertEqual(functions[1].class_name, "Test")
                self.assertEqual(functions[1].name, "test_2")
                self.assertEqual(len(functions[1].parameters), 3)
                self.assertEqual(
                    functions[1].parameters[0].parameter_type, "identifier")
                self.assertIsNone(functions[1].parameters[0].type)
                self.assertEqual(functions[1].parameters[0].name, "self")
                self.assertEqual(
                    functions[1].parameters[1].parameter_type,
                    "typed_parameter"
                )
                self.assertEqual(functions[1].parameters[1].type, "str")
                self.assertEqual(functions[1].parameters[1].name, "name")
                self.assertEqual(
                    functions[1].parameters[2].parameter_type,
                    "typed_parameter"
                )
                self.assertEqual(functions[1].parameters[2].type, "int")
                self.assertEqual(functions[1].parameters[2].name, "age")
                self.assertEqual(functions[1].return_type, "list[str]")

                self.assertEqual(functions[2].id, f"{namespace}.Test.test_3")
                self.assertEqual(functions[2].namespace, namespace)
                self.assertEqual(functions[2].class_name, "Test")
                self.assertEqual(functions[2].name, "test_3")
                self.assertEqual(len(functions[2].parameters), 8)
                self.assertEqual(
                    functions[2].parameters[0].parameter_type, "identifier")
                self.assertIsNone(functions[2].parameters[0].type)
                self.assertEqual(functions[2].parameters[0].name, "cls")
                self.assertEqual(
                    functions[2].parameters[1].parameter_type, "identifier")
                self.assertIsNone(functions[2].parameters[1].type)
                self.assertEqual(functions[2].parameters[1].name, "first")
                self.assertEqual(
                    functions[2].parameters[2].parameter_type,
                    "keyword_separator"
                )
                self.assertIsNone(functions[2].parameters[2].type)
                self.assertEqual(functions[2].parameters[2].name, "*")
                self.assertEqual(
                    functions[2].parameters[3].parameter_type,
                    "typed_parameter"
                )
                self.assertEqual(functions[2].parameters[3].type, "datetime")
                self.assertEqual(functions[2].parameters[3].name, "extra")
                self.assertEqual(
                    functions[2].parameters[4].parameter_type,
                    "typed_default_parameter"
                )
                self.assertEqual(functions[2].parameters[4].type, "bool")
                self.assertEqual(functions[2].parameters[4].name, "second")
                self.assertEqual(
                    functions[2].parameters[5].parameter_type,
                    "typed_default_parameter"
                )
                self.assertEqual(
                    functions[2].parameters[5].type,
                    "Node | None"
                )
                self.assertEqual(functions[2].parameters[5].name, "third")
                self.assertEqual(
                    functions[2].parameters[6].parameter_type,
                    "dictionary_splat_pattern"
                )
                self.assertIsNone(functions[2].parameters[6].type)
                self.assertEqual(functions[2].parameters[6].name, "**kwargs")
                self.assertEqual(
                    functions[2].parameters[7].parameter_type,
                    "typed_default_parameter"
                )
                self.assertEqual(functions[2].parameters[7].type, "float")
                self.assertEqual(functions[2].parameters[7].name, "fourth")
                self.assertIsNone(functions[2].return_type)

if __name__ == '__main__':
    main()
