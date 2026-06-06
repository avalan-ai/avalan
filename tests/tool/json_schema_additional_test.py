from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypedDict
from unittest import TestCase
from unittest.mock import patch

from avalan.tool.json_schema import (
    _annotation_schema,
    _enum_schema,
    _json_type,
    _literal_schema,
    _mapping_schema,
    _parse_docstring_sections,
    _tuple_schema,
    _union_schema,
    _unique_schemas,
    get_json_schema,
)


class JsonSchemaUtilitiesAdditionalTestCase(TestCase):
    def test_parse_docstring_sections_with_none_and_empty(self) -> None:
        self.assertEqual(_parse_docstring_sections(None), {})
        self.assertEqual(_parse_docstring_sections(""), {})

    def test_literal_schema_without_values_or_mixed_types(self) -> None:
        with patch("avalan.tool.json_schema.get_args", return_value=()):
            self.assertEqual(
                _literal_schema(Literal["value"]), {"type": "object"}
            )

        mixed_literal = _literal_schema(Literal["x", 1])
        self.assertEqual(mixed_literal, {"enum": ["x", 1]})

    def test_get_json_schema_maps_unhandled_types_to_object(self) -> None:
        def transform(
            payload: dict[str, int],
            items: tuple[str, ...],
            note: object,
        ) -> list[str | int]:
            """Transform data.

            Args:
                payload: Payload mapping.
                items: Collection of items.
                note: Generic note.
            """
            return [*items]

        schema = get_json_schema(transform)
        properties = schema["function"]["parameters"]["properties"]
        self.assertEqual(properties["payload"]["type"], "object")
        self.assertEqual(
            properties["payload"]["additionalProperties"],
            {"type": "integer"},
        )
        self.assertEqual(properties["items"]["type"], "array")
        self.assertEqual(properties["items"]["items"], {"type": "string"})
        self.assertEqual(properties["note"]["type"], "object")
        self.assertEqual(schema["function"]["return"]["type"], "array")
        self.assertEqual(
            schema["function"]["return"]["items"]["anyOf"],
            [{"type": "string"}, {"type": "integer"}],
        )

    def test_json_type_covers_builtin_and_unknown_annotations(self) -> None:
        class CustomType:
            pass

        self.assertEqual(_annotation_schema(float), {"type": "number"})
        self.assertEqual(_json_type(dict), "object")
        self.assertEqual(_json_type(list), "array")
        self.assertEqual(_json_type(CustomType), "object")
        self.assertEqual(_json_type(str | int), "object")
        self.assertEqual(_json_type(str | None), "string")

    def test_annotation_schema_handles_nested_arrays_and_mappings(
        self,
    ) -> None:
        schema = _annotation_schema(dict[str, list[int | None]])
        sequence_schema = _annotation_schema(Sequence[str])

        self.assertEqual(schema["type"], "object")
        self.assertEqual(
            schema["additionalProperties"],
            {
                "type": "array",
                "items": {"type": ["integer", "null"]},
            },
        )
        self.assertEqual(
            sequence_schema,
            {"type": "array", "items": {"type": "string"}},
        )

    def test_annotation_schema_handles_fixed_tuples(self) -> None:
        schema = _annotation_schema(tuple[str, int])

        self.assertEqual(
            schema,
            {
                "type": "array",
                "prefixItems": [
                    {"type": "string"},
                    {"type": "integer"},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
        )

    def test_annotation_schema_handles_dataclasses(self) -> None:
        @dataclass(frozen=True)
        class Payload:
            name: str
            tags: list[str]
            retries: int = 0

        schema = _annotation_schema(Payload)

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["name", "tags"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["name"], {"type": "string"})
        self.assertEqual(
            schema["properties"]["tags"],
            {"type": "array", "items": {"type": "string"}},
        )

    def test_annotation_schema_handles_typed_dicts(self) -> None:
        class Payload(TypedDict):
            name: str
            metadata: dict[str, int]

        schema = _annotation_schema(Payload)

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["metadata", "name"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["name"], {"type": "string"})
        self.assertEqual(
            schema["properties"]["metadata"],
            {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
        )

    def test_enum_schema_handles_string_and_mixed_value_enums(self) -> None:
        class Empty(Enum):
            pass

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        class Mixed(Enum):
            NAME = "name"
            COUNT = 3

        self.assertEqual(
            _enum_schema(Color),
            {"type": "string", "enum": ["red", "blue"]},
        )
        self.assertEqual(_annotation_schema(Color), _enum_schema(Color))
        self.assertEqual(_enum_schema(Empty), {"type": "object"})
        self.assertEqual(_enum_schema(Mixed), {"enum": ["name", 3]})
        self.assertIsNone(_enum_schema(str))

    def test_schema_helpers_handle_empty_and_duplicate_inputs(self) -> None:
        self.assertEqual(_union_schema(()), {"type": "object"})
        self.assertEqual(
            _union_schema((Literal["x", 1], type(None))),
            {"anyOf": [{"enum": ["x", 1]}, {"type": "null"}]},
        )
        self.assertEqual(_tuple_schema(()), {"type": "array"})
        self.assertEqual(_mapping_schema(()), {"type": "object"})
        self.assertEqual(
            _unique_schemas(
                [
                    {"type": "string"},
                    {"type": "string"},
                    {"type": "integer"},
                ]
            ),
            [{"type": "string"}, {"type": "integer"}],
        )

    def test_get_json_schema_closes_parameters_and_excludes_context(
        self,
    ) -> None:
        class NonJsonDefault:
            pass

        class Mode(Enum):
            FAST = "fast"

        def configure(
            name: str,
            context: object,
            mode: Mode = Mode.FAST,
            enabled: bool = True,
            payload: object = NonJsonDefault(),
        ) -> None:
            """Configure a tool.

            Args:
                name: Configuration name.
                enabled: Whether configuration is active.
            """

        schema = get_json_schema(configure)
        parameters = schema["function"]["parameters"]
        properties = parameters["properties"]

        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(
            set(properties), {"name", "mode", "enabled", "payload"}
        )
        self.assertEqual(parameters["required"], ["name"])
        self.assertEqual(properties["mode"]["default"], "fast")
        self.assertEqual(properties["enabled"]["default"], True)
        self.assertNotIn("default", properties["payload"])
        self.assertEqual(schema["function"]["return"]["type"], "null")
