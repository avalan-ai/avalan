from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.flow import (
    FLOW_INPUT_KEY,
    FlowDefinition,
    FlowDefinitionLoader,
    FlowInputType,
    FlowLoadError,
    FlowLoadIssueCategory,
    FlowNodeDefinition,
    FlowNodeRegistry,
    build_flow,
    flow_input_binding,
    load_flow_definition,
    load_flow_definition_result,
    loads_flow_definition,
    loads_flow_definition_result,
)
from avalan.flow import loader as flow_loader
from avalan.flow.node import Node

VALID_FLOW = """
[flow]
name = "simple"
version = "1"
entrypoint = "start"
output_node = "finish"

[flow.input]
name = "payload"
type = "object"

[flow.output]
name = "result"
type = "json"

[nodes.start]
type = "input"
input = "payload"

[nodes.finish]
type = "echo"
input = "start"

[[edges]]
source = "start"
target = "finish"
"""


class FlowDefinitionLoaderTestCase(IsolatedAsyncioTestCase):
    async def test_loads_valid_flow_and_builds_executable_graph(self) -> None:
        result = loads_flow_definition_result(VALID_FLOW)

        self.assertTrue(result.ok)
        assert result.definition is not None
        assert result.flow is not None
        self.assertEqual(result.definition.name, "simple")
        self.assertEqual(result.definition.input.type, FlowInputType.OBJECT)

        output = await result.flow.execute_async(
            initial_node=result.definition.entrypoint,
            initial_inputs=flow_input_binding(
                result.definition.input,
                {"answer": "ok"},
            ),
        )

        self.assertEqual(output, {"answer": "ok"})

    async def test_load_accepts_top_level_input_and_custom_registry(
        self,
    ) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(
                definition.name,
                func=lambda inputs: inputs[FLOW_INPUT_KEY] + "!",
            )

        loader = FlowDefinitionLoader(FlowNodeRegistry({"excited": factory}))
        result = loader.loads_result("""
            [flow]
            name = "custom"
            entrypoint = "start"
            output_node = "start"

            [input]
            name = "value"
            type = "string"

            [output]
            name = "result"
            type = "text"

            [variables]
            owner = "tests"

            [nodes.start]
            type = "excited"
            """)

        self.assertTrue(result.ok)
        assert result.definition is not None
        assert result.flow is not None
        self.assertEqual(result.definition.variables["owner"], "tests")
        self.assertEqual(
            await result.flow.execute_async(
                initial_node="start",
                initial_inputs=flow_input_binding(
                    result.definition.input,
                    "ready",
                ),
            ),
            "ready!",
        )

    async def test_load_accepts_nested_node_config(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "constant"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "constant"

            [nodes.start.config]
            value = "ok"
            """)

        self.assertTrue(result.ok)
        assert result.flow is not None
        self.assertEqual(await result.flow.execute_async(), "ok")

    def test_load_and_wrapper_helpers_read_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "flow.toml"
            path.write_text(VALID_FLOW, encoding="utf-8")

            definition = load_flow_definition(path)
            result = load_flow_definition_result(path)
            loaded = loads_flow_definition(VALID_FLOW)

        self.assertIsInstance(definition, FlowDefinition)
        self.assertIsInstance(loaded, FlowDefinition)
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.flow)

    def test_load_wrapper_raises_structured_error_for_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "flow.toml"
            path.write_text("[flow]\nname = 'missing'", encoding="utf-8")

            with self.assertRaises(FlowLoadError) as context:
                FlowDefinitionLoader().load(path)

        self.assertEqual(
            context.exception.issues[0].code, "flow.missing_section"
        )

    def test_loads_wrapper_raises_structured_error(self) -> None:
        with self.assertRaises(FlowLoadError) as context:
            loads_flow_definition("[flow]\nname = 'missing'")

        self.assertEqual(
            context.exception.issues[0].code, "flow.missing_section"
        )
        self.assertIn("flow.missing_section", str(context.exception))

    def test_malformed_toml_uses_safe_generic_path(self) -> None:
        result = loads_flow_definition_result(
            "[flow\nsecret = 'private'",
            source_path="/private/customer.toml",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.malformed_toml")
        self.assertEqual(result.issues[0].path, "toml")
        self.assertEqual(
            result.issues[0].category, FlowLoadIssueCategory.PARSE
        )
        self.assertNotIn("private", str(result.issues[0].as_dict()))
        self.assertNotIn("customer", str(result.issues[0].as_dict()))

    def test_missing_sections_and_invalid_shapes_are_aggregated(self) -> None:
        result = loads_flow_definition_result("""
            flow = "invalid"
            nodes = "invalid"
            edges = "invalid"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            [
                "flow.invalid_section",
                "flow.invalid_section",
                "flow.invalid_section",
            ],
        )

    def test_rejects_invalid_child_sections_and_missing_fields(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"
                input = "invalid"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_section",
            ),
            (
                """
                [flow]
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.missing_field",
            ),
            (
                """
                [flow]
                name = 3
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                version = 3
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_rejects_duplicate_and_unsupported_sections(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "unsupported"
            entrypoint = "start"
            output_node = "start"

            [flow.input]
            name = "value"
            type = "string"

            [input]
            name = "value"
            type = "string"

            [nodes.start]
            type = "echo"

            [cli]
            runner = "python private.py"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["flow.unsupported_section", "flow.duplicate_section"],
        )
        self.assertNotIn("private.py", str(result.issues))

    def test_rejects_unsupported_fields_and_values(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "invalid"
            entrypoint = "start"
            output_node = "start"

            [flow.input]
            name = "document"
            type = "file"
            delivery = "direct"
            memory = false

            [flow.output]
            name = "result"
            type = "unknown"

            [nodes.start]
            type = "agent"
            ref = "../private/agent.toml"
            user_prompt_ref = "private.txt"
            response_format_ref = "format.json"
            """)

        self.assertFalse(result.ok)
        codes = [issue.code for issue in result.issues]
        self.assertIn("flow.unsupported_field", codes)
        self.assertIn("flow.invalid_enum", codes)
        self.assertIn("flow.path_escape", codes)
        self.assertIn("flow.unsupported_node_type", codes)
        self.assertNotIn("private", str(result.issues))

    def test_rejects_invalid_input_and_output_shapes(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "document"
                mime_types = "application/pdf"

                [nodes.start]
                type = "echo"
                """,
                ("flow.missing_field", "flow.invalid_type"),
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "document"
                type = "file"
                mime_types = [3]

                [nodes.start]
                type = "echo"
                """,
                ("flow.invalid_type",),
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.input]
                name = "payload"
                type = "object"
                schema = "invalid"

                [nodes.start]
                type = "echo"
                """,
                ("flow.invalid_type",),
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [flow.output]
                name = "result"

                [nodes.start]
                type = "echo"
                """,
                ("flow.missing_field",),
            ),
        )

        for source, expected_codes in cases:
            with self.subTest(expected_codes=expected_codes):
                result = loads_flow_definition_result(source)
                codes = [issue.code for issue in result.issues]

                self.assertFalse(result.ok)
                for code in expected_codes:
                    self.assertIn(code, codes)

    def test_rejects_unknown_and_untrusted_node_types(self) -> None:
        for node_type, code in (
            ("unknown", "flow.unknown_node_type"),
            ("callable", "flow.untrusted_callable"),
        ):
            with self.subTest(node_type=node_type):
                result = loads_flow_definition_result(f"""
                    [flow]
                    name = "invalid"
                    entrypoint = "start"
                    output_node = "start"

                    [nodes.start]
                    type = "{node_type}"
                    ref = "package.module:function"
                    """)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])
                self.assertNotIn("package.module", str(result.issues))

    def test_rejects_invalid_node_and_edge_shapes(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.""]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes]
                start = "invalid"
                """,
                "flow.invalid_section",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                input = "payload"
                """,
                "flow.missing_field",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                config = "invalid"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                ref = "safe.toml"
                """,
                "flow.untrusted_callable",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"

                [edges]
                start = "finish"
                """,
                "flow.invalid_section",
            ),
            (
                """
                edges = ["invalid"]

                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"
                """,
                "flow.invalid_type",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "agent"
                ref = "https://example.invalid/private.toml"
                """,
                "flow.path_escape",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "agent"
                ref = "/private/agent.toml"
                """,
                "flow.path_escape",
            ),
            (
                """
                [flow]
                name = "invalid"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"

                [[edges]]
                source = "start"
                """,
                "flow.missing_field",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_rejects_bad_edge_references_and_cycles(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "bad_source"
                entrypoint = "start"
                output_node = "end"

                [nodes.start]
                type = "echo"

                [nodes.end]
                type = "echo"

                [[edges]]
                source = "missing"
                target = "end"
                """,
                "flow.bad_reference",
            ),
            (
                """
                [flow]
                name = "bad_edge"
                entrypoint = "start"
                output_node = "end"

                [nodes.start]
                type = "echo"

                [[edges]]
                source = "start"
                target = "missing"
                """,
                "flow.bad_reference",
            ),
            (
                """
                [flow]
                name = "cycle"
                entrypoint = "a"
                output_node = "b"

                [nodes.a]
                type = "echo"

                [nodes.b]
                type = "echo"

                [[edges]]
                source = "a"
                target = "b"

                [[edges]]
                source = "b"
                target = "a"
                """,
                "flow.invalid_entrypoint",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_rejects_unknown_entrypoint_and_output_node(self) -> None:
        result = loads_flow_definition_result("""
            [flow]
            name = "unknown_refs"
            entrypoint = "missing_start"
            output_node = "missing_end"

            [nodes.start]
            type = "echo"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["flow.unknown_entrypoint", "flow.unknown_output_node"],
        )

    def test_rejects_multiple_and_non_terminal_outputs(self) -> None:
        cases = (
            (
                """
                [flow]
                name = "multiple"
                entrypoint = "start"
                output_node = "left"

                [nodes.start]
                type = "echo"

                [nodes.left]
                type = "echo"

                [nodes.right]
                type = "echo"

                [[edges]]
                source = "start"
                target = "left"

                [[edges]]
                source = "start"
                target = "right"
                """,
                "flow.multiple_outputs",
            ),
            (
                """
                [flow]
                name = "non_terminal"
                entrypoint = "start"
                output_node = "start"

                [nodes.start]
                type = "echo"

                [nodes.end]
                type = "echo"

                [[edges]]
                source = "start"
                target = "end"
                """,
                "flow.invalid_output_node",
            ),
        )

        for source, code in cases:
            with self.subTest(code=code):
                result = loads_flow_definition_result(source)

                self.assertFalse(result.ok)
                self.assertIn(code, [issue.code for issue in result.issues])

    def test_invalid_node_factory_returns_stable_issue(self) -> None:
        def factory(_: FlowNodeDefinition) -> Node:
            raise AssertionError("private factory failure")

        loader = FlowDefinitionLoader(FlowNodeRegistry({"broken": factory}))

        result = loader.loads_result("""
            [flow]
            name = "broken"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "broken"
            """)

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.invalid_node")
        self.assertNotIn("private factory failure", str(result.issues[0]))

    def test_private_helpers_cover_toml_impossible_shapes(self) -> None:
        issues: list[flow_loader.FlowLoadIssue] = []
        raw = {
            1: "ignored",
            "flow": {
                "name": "raw",
                "entrypoint": "start",
                "output_node": "start",
            },
            "nodes": {"start": {"type": "echo"}},
        }

        result = flow_loader._build_result(  # type: ignore[attr-defined]
            raw,  # type: ignore[arg-type]
            registry=FlowNodeRegistry(),
            source_path=None,
        )
        tuple_value = flow_loader._string_tuple(  # type: ignore[attr-defined]
            {"mime_types": None},
            "flow.input.mime_types",
            "mime_types",
            issues,
        )
        metadata = flow_loader._metadata(  # type: ignore[attr-defined]
            {1: "bad"},  # type: ignore[dict-item]
            "metadata",
            issues,
        )

        self.assertFalse(result.ok)
        self.assertEqual(tuple_value, ())
        self.assertIsNone(metadata)
        self.assertIn("flow.invalid_type", [issue.code for issue in issues])


class FlowBuildTestCase(TestCase):
    def test_build_flow_from_definition(self) -> None:
        definition = FlowDefinition(
            name="manual",
            entrypoint="start",
            output_node="start",
            nodes=(
                FlowNodeDefinition(
                    name="start",
                    type="constant",
                    config={"value": "ok"},
                ),
            ),
        )

        flow = build_flow(definition)

        self.assertEqual(flow.execute(), "ok")


if __name__ == "__main__":
    main()
