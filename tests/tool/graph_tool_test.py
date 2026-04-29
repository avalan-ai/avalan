from base64 import b64decode
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main, skipIf
from unittest.mock import patch

from avalan.entities import ToolCallContext
from avalan.tool.graph import (
    HAS_GRAPH_DEPENDENCIES,
    BarChartTool,
    GraphOutputFormat,
    GraphToolSet,
    GraphToolSettings,
    HistogramTool,
    LineChartTool,
    PieChartTool,
    ScatterPlotTool,
)


def _decode_data_uri(result: dict[str, object]) -> bytes:
    data_uri = result["data_uri"]
    assert isinstance(data_uri, str)
    _, encoded = data_uri.split(",", 1)
    return b64decode(encoded)


class GraphToolSetTestCase(TestCase):
    def test_json_schemas_include_graph_tools(self) -> None:
        toolset = GraphToolSet(namespace="graph")
        schemas = toolset.json_schemas()

        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            [
                "graph.pie",
                "graph.bar",
                "graph.line",
                "graph.scatter",
                "graph.histogram",
            ],
        )
        self.assertIn(
            "series",
            schemas[1]["function"]["parameters"]["properties"],
        )


class GraphToolValidationTestCase(IsolatedAsyncioTestCase):
    async def test_missing_matplotlib_raises_import_error(self) -> None:
        tool = PieChartTool()
        with patch("avalan.tool.graph.HAS_GRAPH_DEPENDENCIES", False):
            with self.assertRaisesRegex(
                ImportError,
                "Graph tools require optional dependencies",
            ):
                await tool(
                    ["A"],
                    [1],
                    context=ToolCallContext(),
                )

    async def test_invalid_output_format(self) -> None:
        tool = HistogramTool()
        with self.assertRaisesRegex(
            AssertionError,
            "output_format must be one of",
        ):
            await tool(
                [1, 2, 3],
                context=ToolCallContext(),
                output_format=cast(GraphOutputFormat, "jpg"),
            )

    async def test_empty_rendered_output_raises(self) -> None:
        class EmptyFigure:
            def savefig(self, buffer: Any, *, format: str, dpi: int) -> None:
                _ = buffer, format, dpi

        tool = PieChartTool()
        with self.assertRaisesRegex(
            AssertionError,
            "graph rendering produced empty output",
        ):
            tool._render(
                EmptyFigure(),
                chart_type="pie",
                output_format="png",
                title=None,
                width=1,
                height=1,
                dpi=100,
                series_names=["values"],
                point_count=1,
            )

    async def test_file_setting_rejects_directory_path(self) -> None:
        with TemporaryDirectory() as directory:
            tool = HistogramTool(GraphToolSettings(file=directory))
            with self.assertRaisesRegex(
                AssertionError, "graph file must be a file path"
            ):
                await tool([1, 2, 3], context=ToolCallContext())

    async def test_file_setting_rejects_file_parent(self) -> None:
        with TemporaryDirectory() as directory:
            parent = f"{directory}/parent"
            with open(parent, "w", encoding="utf-8") as file:
                file.write("not a directory")
            tool = HistogramTool(GraphToolSettings(file=f"{parent}/chart.png"))

            with self.assertRaisesRegex(
                AssertionError, "graph file parent must be a directory"
            ):
                await tool([1, 2, 3], context=ToolCallContext())

    async def test_pie_rejects_invalid_inputs(self) -> None:
        tool = PieChartTool()
        with self.assertRaisesRegex(
            AssertionError, "labels and values must have the same length"
        ):
            await tool(["A"], [1, 2], context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "values must contain a positive total"
        ):
            await tool(["A"], [0], context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError,
            "values entries must be greater than or equal to 0",
        ):
            await tool(["A"], [-1], context=ToolCallContext())

    async def test_pie_rejects_invalid_colors(self) -> None:
        tool = PieChartTool()
        with self.assertRaisesRegex(
            AssertionError, "colors length must match labels length"
        ):
            await tool(
                ["A", "B"],
                [1, 2],
                context=ToolCallContext(),
                colors=["red"],
            )

    async def test_bar_rejects_invalid_inputs(self) -> None:
        tool = BarChartTool()
        with self.assertRaisesRegex(
            AssertionError, "provide either values or series"
        ):
            await tool(
                ["A"],
                values=[1],
                series={"x": [1]},
                context=ToolCallContext(),
            )

        with self.assertRaisesRegex(
            AssertionError, "series must not be empty"
        ):
            await tool(["A"], series={}, context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "series names must be non-empty strings"
        ):
            await tool(["A"], series={"": [1]}, context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "series\\[x\\] length must match categories length"
        ):
            await tool(["A"], series={"x": [1, 2]}, context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "orientation must be vertical or horizontal"
        ):
            await tool(
                ["A"],
                values=[1],
                orientation=cast(Literal["vertical", "horizontal"], "bad"),
                context=ToolCallContext(),
            )

    async def test_line_rejects_invalid_inputs(self) -> None:
        tool = LineChartTool()
        with self.assertRaisesRegex(
            AssertionError, "x_labels entries must be non-empty strings"
        ):
            await tool([""], values=[1], context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError,
            "series\\[sales\\] length must match x_labels length",
        ):
            await tool(
                ["Q1"],
                series={"sales": [1, 2]},
                context=ToolCallContext(),
            )

    async def test_scatter_rejects_invalid_inputs(self) -> None:
        tool = ScatterPlotTool()
        with self.assertRaisesRegex(
            AssertionError, "x and y must have the same length"
        ):
            await tool([1], [1, 2], context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "labels length must match x and y length"
        ):
            await tool([1], [2], labels=["A", "B"], context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "point_size must be greater than zero"
        ):
            await tool([1], [2], point_size=0, context=ToolCallContext())

    async def test_histogram_rejects_invalid_inputs(self) -> None:
        tool = HistogramTool()
        with self.assertRaisesRegex(AssertionError, "bins must be greater"):
            await tool([1, 2], bins=0, context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "values entries must be numbers"
        ):
            await tool(
                cast(list[float], ["bad"]),
                context=ToolCallContext(),
            )

        with self.assertRaisesRegex(
            AssertionError, "values entries must be numbers"
        ):
            await tool(
                cast(list[float], [True]),
                context=ToolCallContext(),
            )

        with self.assertRaisesRegex(
            AssertionError, "values entries must be finite numbers"
        ):
            await tool(
                [float("inf")],
                context=ToolCallContext(),
            )

        with self.assertRaisesRegex(
            AssertionError, "width must be greater than zero"
        ):
            await tool([1], width=0, context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "height must be greater than zero"
        ):
            await tool([1], height=0, context=ToolCallContext())

        with self.assertRaisesRegex(
            AssertionError, "dpi must be greater than zero"
        ):
            await tool([1], dpi=0, context=ToolCallContext())


@skipIf(not HAS_GRAPH_DEPENDENCIES, "matplotlib not installed")
class GraphToolRenderingTestCase(IsolatedAsyncioTestCase):
    async def test_pie_chart_png(self) -> None:
        result = await PieChartTool()(
            ["A", "B"],
            [1, 2],
            title="Share",
            colors=["#1f77b4", "#ff7f0e"],
            show_percentages=False,
            context=ToolCallContext(),
        )

        self.assertEqual(result["chart_type"], "pie")
        self.assertEqual(result["mime_type"], "image/png")
        self.assertEqual(result["points"], 2)
        self.assertTrue(_decode_data_uri(result).startswith(b"\x89PNG\r\n"))

    async def test_bar_chart_variants(self) -> None:
        categories = [f"C{i}" for i in range(9)]
        vertical = await BarChartTool()(
            categories,
            values=[float(i) for i in range(9)],
            title="Vertical",
            x_label="Category",
            y_label="Value",
            grid=True,
            context=ToolCallContext(),
        )

        self.assertEqual(vertical["chart_type"], "bar")
        self.assertEqual(vertical["points"], 9)
        self.assertTrue(_decode_data_uri(vertical).startswith(b"\x89PNG\r\n"))

        stacked = await BarChartTool()(
            ["Q1", "Q2"],
            series={"sales": [3, 4], "cost": [1, 2]},
            stacked=True,
            output_format="svg",
            context=ToolCallContext(),
        )
        svg = _decode_data_uri(stacked)
        self.assertEqual(stacked["mime_type"], "image/svg+xml")
        self.assertIn(b"<svg", svg)

        horizontal = await BarChartTool()(
            ["Q1", "Q2"],
            series={"sales": [3, 4], "cost": [1, 2]},
            orientation="horizontal",
            context=ToolCallContext(),
        )
        self.assertEqual(horizontal["points"], 4)

        horizontal_stacked = await BarChartTool()(
            ["Q1", "Q2"],
            series={"sales": [3, 4], "cost": [1, 2]},
            orientation="horizontal",
            stacked=True,
            show_legend=False,
            context=ToolCallContext(),
        )
        self.assertEqual(horizontal_stacked["points"], 4)

    async def test_line_chart_png(self) -> None:
        result = await LineChartTool()(
            [f"2024-{month:02d}" for month in range(1, 10)],
            series={
                "sales": [float(month) for month in range(1, 10)],
                "cost": [float(month) / 2 for month in range(1, 10)],
            },
            title="Sales",
            x_label="Month",
            y_label="Revenue",
            markers=False,
            context=ToolCallContext(),
        )

        self.assertEqual(result["chart_type"], "line")
        self.assertEqual(result["series"], ["sales", "cost"])
        self.assertTrue(_decode_data_uri(result).startswith(b"\x89PNG\r\n"))

    async def test_scatter_plot_png(self) -> None:
        result = await ScatterPlotTool()(
            [1, 2, 3],
            [3, 2, 5],
            labels=["A", "B", "C"],
            title="Scatter",
            x_label="X",
            y_label="Y",
            color="green",
            show_legend=True,
            context=ToolCallContext(),
        )

        self.assertEqual(result["chart_type"], "scatter")
        self.assertEqual(result["points"], 3)
        self.assertTrue(_decode_data_uri(result).startswith(b"\x89PNG\r\n"))

    async def test_histogram_png(self) -> None:
        result = await HistogramTool()(
            [1, 1, 2, 3, 5, 8],
            bins=3,
            title="Histogram",
            x_label="Value",
            y_label="Count",
            color="purple",
            cumulative=True,
            context=ToolCallContext(),
        )

        self.assertEqual(result["chart_type"], "histogram")
        self.assertEqual(result["points"], 6)
        self.assertTrue(_decode_data_uri(result).startswith(b"\x89PNG\r\n"))

    async def test_file_setting_saves_rendered_graph(self) -> None:
        with TemporaryDirectory() as directory:
            path = f"{directory}/chart.png"
            result = await BarChartTool(GraphToolSettings(file=path))(
                ["Jan", "Feb"],
                values=[10, 12],
                context=ToolCallContext(),
            )

            self.assertEqual(result["file"], str(Path(path).resolve()))
            with open(path, "rb") as file:
                data = file.read()
            self.assertEqual(data, _decode_data_uri(result))
            self.assertTrue(data.startswith(b"\x89PNG\r\n"))


if __name__ == "__main__":
    main()
