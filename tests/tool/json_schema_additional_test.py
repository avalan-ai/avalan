from typing import Literal
from unittest import TestCase
from unittest.mock import patch

from avalan.tool.json_schema import _literal_schema, _parse_docstring_sections, get_json_schema


class JsonSchemaUtilitiesAdditionalTestCase(TestCase):
    def test_parse_docstring_sections_with_none_and_empty(self) -> None:
        self.assertEqual(_parse_docstring_sections(None), {})
        self.assertEqual(_parse_docstring_sections(""), {})

    def test_literal_schema_without_values_or_mixed_types(self) -> None:
        with patch("avalan.tool.json_schema.get_args", return_value=()):
            self.assertEqual(_literal_schema(Literal["value"]), {"type": "object"})

        mixed_literal = _literal_schema(Literal["x", 1])
        self.assertEqual(mixed_literal, {"type": "object"})

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
        self.assertEqual(properties["items"]["type"], "array")
        self.assertEqual(properties["note"]["type"], "object")
        self.assertEqual(schema["function"]["return"]["type"], "array")
