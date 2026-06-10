from asyncio import run as asyncio_run
from pathlib import Path
from unittest import TestCase, main

from avalan.flow import (
    FlowEdgeKind,
    FlowGraphEdgeClassification,
    FlowGraphNodeClassification,
    compile_flow_file,
)

DOC_ROOT = Path(__file__).parents[2] / "docs"
EXAMPLE_ROOT = DOC_ROOT / "examples" / "flows"
FLOW_AUTHORING_DOC = DOC_ROOT / "FLOW_AUTHORING.md"


class FlowAuthoringExamplesTest(TestCase):
    def test_graph_authoring_docs_cover_executable_contract(self) -> None:
        docs = FLOW_AUTHORING_DOC.read_text(encoding="utf-8")
        normalized = " ".join(docs.split())

        for phrase in (
            "## Graph-Authored Strict Flows",
            "`[graph]`",
            'source = "inline"',
            'source = "file"',
            "exactly match declared strict node names",
            "Decorative nodes and edges",
            "`[graph.edges.<edge_id>]`",
            "Labels, classes, styles, shapes",
            "canonical strict TOML",
            "never runs Mermaid directly",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)

    def test_graph_authoring_examples_compile_to_strict_edges(self) -> None:
        for filename in (
            "graph_inline.flow.toml",
            "graph_file.flow.toml",
        ):
            with self.subTest(example=filename):
                result = asyncio_run(
                    compile_flow_file(EXAMPLE_ROOT / filename)
                )

                self.assertTrue(result.ok, result.public_diagnostics)
                self.assertTrue(result.authoring_graph)
                assert result.definition is not None
                assert result.canonical_source is not None
                self.assertEqual(
                    [
                        (
                            edge.source,
                            edge.target,
                            edge.kind,
                            edge.label,
                        )
                        for edge in result.definition.edges
                    ],
                    [
                        (
                            "start",
                            "pick",
                            FlowEdgeKind.SUCCESS,
                            "profile_ready",
                        )
                    ],
                )
                self.assertIn("[[edges]]", result.canonical_source)
                self.assertNotIn("[graph]", result.canonical_source)
                self.assertNotIn("flowchart", result.canonical_source)
                self.assertNotIn(
                    "diagram label ignored",
                    result.canonical_source,
                )

    def test_graph_authoring_example_keeps_visual_metadata_inert(
        self,
    ) -> None:
        result = asyncio_run(
            compile_flow_file(EXAMPLE_ROOT / "graph_inline.flow.toml")
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        assert result.graph_inspection is not None
        node_classes = {
            node.id: node.classification
            for node in result.graph_inspection.nodes
        }
        edge_classes = {
            (edge.source, edge.target): edge.classification
            for edge in result.graph_inspection.edges
        }

        self.assertEqual(
            node_classes,
            {
                "start": FlowGraphNodeClassification.ACTUAL,
                "pick": FlowGraphNodeClassification.ACTUAL,
                "note": FlowGraphNodeClassification.DECORATIVE,
            },
        )
        self.assertEqual(
            edge_classes,
            {
                ("start", "pick"): FlowGraphEdgeClassification.EXECUTABLE,
                ("start", "note"): FlowGraphEdgeClassification.DECORATIVE,
                ("note", "pick"): FlowGraphEdgeClassification.DECORATIVE,
            },
        )
        assert result.definition is not None
        self.assertEqual(len(result.definition.edges), 1)
        self.assertNotEqual(
            result.definition.edges[0].label,
            "diagram label ignored",
        )


if __name__ == "__main__":
    main()
