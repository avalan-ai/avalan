from ..compat import override
from ..entities import ToolCallContext
from . import Tool, ToolSet

from base64 import b64encode
from contextlib import AsyncExitStack
from dataclasses import dataclass
from io import BytesIO
from math import isfinite
from os import environ
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Literal, final

environ.setdefault("MPLCONFIGDIR", f"{gettempdir()}/avalan-matplotlib")
environ.setdefault("XDG_CACHE_HOME", f"{gettempdir()}/avalan-cache")

_Figure: Any | None

try:
    import matplotlib as _matplotlib
    from matplotlib.figure import Figure as _MatplotlibFigure

    _matplotlib.use("Agg", force=True)
    _matplotlib.rcParams["svg.fonttype"] = "none"
    _Figure = _MatplotlibFigure
    HAS_GRAPH_DEPENDENCIES = True
except ImportError:
    _Figure = None
    HAS_GRAPH_DEPENDENCIES = False


GraphOutputFormat = Literal["png", "svg", "pdf"]

GRAPH_MIME_TYPES: dict[GraphOutputFormat, str] = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
}


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class GraphToolSettings(dict[str, object]):
    file: str | None = None

    def __post_init__(self) -> None:
        self["file"] = self.file


class GraphRenderer:
    _DEFAULT_WIDTH = 6.4
    _DEFAULT_HEIGHT = 4.8
    _DEFAULT_DPI = 100
    _settings: GraphToolSettings

    def _new_axes(self, width: float, height: float) -> tuple[Any, Any]:
        self._assert_matplotlib_available()
        self._assert_dimensions(width, height)
        assert _Figure is not None
        figure = _Figure(figsize=(width, height), layout="constrained")
        axes = figure.add_subplot(1, 1, 1)
        return figure, axes

    def _render(
        self,
        figure: Any,
        *,
        chart_type: str,
        output_format: GraphOutputFormat,
        title: str | None,
        width: float,
        height: float,
        dpi: int,
        series_names: list[str],
        point_count: int,
    ) -> dict[str, object]:
        assert (
            output_format in GRAPH_MIME_TYPES
        ), "output_format must be one of: png, svg, pdf"
        assert dpi > 0, "dpi must be greater than zero"

        buffer = BytesIO()
        figure.savefig(buffer, format=output_format, dpi=dpi)
        data = buffer.getvalue()
        assert data, "graph rendering produced empty output"

        mime_type = GRAPH_MIME_TYPES[output_format]
        encoded = b64encode(data).decode("ascii")
        result: dict[str, object] = {
            "chart_type": chart_type,
            "format": output_format,
            "mime_type": mime_type,
            "encoding": "base64_data_uri",
            "data_uri": f"data:{mime_type};base64,{encoded}",
            "title": title,
            "width": width,
            "height": height,
            "dpi": dpi,
            "series": series_names,
            "points": point_count,
        }
        file_path = self._save_file(data)
        if file_path:
            result["file"] = file_path
        return result

    def _apply_axes_labels(
        self,
        axes: Any,
        *,
        title: str | None,
        x_label: str | None = None,
        y_label: str | None = None,
        grid: bool = False,
    ) -> None:
        if title:
            axes.set_title(title)
        if x_label:
            axes.set_xlabel(x_label)
        if y_label:
            axes.set_ylabel(y_label)
        if grid:
            axes.grid(True, alpha=0.3)

    def _series_from_values(
        self,
        values: list[float] | None,
        series: dict[str, list[float]] | None,
        *,
        series_name: str,
    ) -> dict[str, list[float]]:
        assert (values is None) != (
            series is None
        ), "provide either values or series"
        if values is not None:
            return {series_name: self._numbers(values, "values")}

        assert series is not None
        assert series, "series must not be empty"
        cleaned: dict[str, list[float]] = {}
        for name, raw_values in series.items():
            assert (
                isinstance(name, str) and name.strip()
            ), "series names must be non-empty strings"
            cleaned[name] = self._numbers(raw_values, f"series[{name}]")
        return cleaned

    def _labels(self, values: list[str], name: str) -> list[str]:
        assert values, f"{name} must not be empty"
        labels = []
        for value in values:
            assert (
                isinstance(value, str) and value.strip()
            ), f"{name} entries must be non-empty strings"
            labels.append(value)
        return labels

    def _numbers(
        self,
        values: list[float],
        name: str,
        *,
        minimum: float | None = None,
    ) -> list[float]:
        assert values, f"{name} must not be empty"
        numbers = []
        for value in values:
            assert not isinstance(
                value, bool
            ), f"{name} entries must be numbers"
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise AssertionError(
                    f"{name} entries must be numbers"
                ) from exc
            assert isfinite(number), f"{name} entries must be finite numbers"
            if minimum is not None:
                assert (
                    number >= minimum
                ), f"{name} entries must be greater than or equal to {minimum}"
            numbers.append(number)
        return numbers

    def _assert_series_lengths(
        self,
        series: dict[str, list[float]],
        *,
        expected: int,
        axis_name: str,
    ) -> None:
        for name, values in series.items():
            assert (
                len(values) == expected
            ), f"series[{name}] length must match {axis_name} length"

    @staticmethod
    def _assert_dimensions(width: float, height: float) -> None:
        assert width > 0, "width must be greater than zero"
        assert height > 0, "height must be greater than zero"

    @staticmethod
    def _assert_matplotlib_available() -> None:
        if not HAS_GRAPH_DEPENDENCIES:
            raise ImportError(
                "Graph tools require optional dependencies. "
                "Install them with: pip install avalan[tool]"
            )

    def _save_file(self, data: bytes) -> str | None:
        file = self._settings.file
        if not file:
            return None

        path = Path(file).expanduser()
        parent = path.parent
        assert (
            not parent.exists() or parent.is_dir()
        ), "graph file parent must be a directory"
        parent.mkdir(parents=True, exist_ok=True)
        assert (
            not path.exists() or path.is_file()
        ), "graph file must be a file path"
        path.write_bytes(data)
        return str(path.resolve())


class PieChartTool(Tool, GraphRenderer):
    """Create a pie chart from labels and values.

    Args:
        labels: Slice labels in display order.
        values: Non-negative slice values matching the labels.
        title: Optional chart title.
        output_format: Output image format: png, svg, or pdf.
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Raster dots per inch used for PNG output.
        colors: Optional slice colors matching the labels.
        show_percentages: Include percentage labels on slices.

    Returns:
        Graph image metadata with a base64 data URI.
    """

    def __init__(self, settings: GraphToolSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or GraphToolSettings()
        self.__name__ = "pie"

    async def __call__(
        self,
        labels: list[str],
        values: list[float],
        *,
        context: ToolCallContext,
        title: str | None = None,
        output_format: GraphOutputFormat = "png",
        width: float = GraphRenderer._DEFAULT_WIDTH,
        height: float = GraphRenderer._DEFAULT_HEIGHT,
        dpi: int = GraphRenderer._DEFAULT_DPI,
        colors: list[str] | None = None,
        show_percentages: bool = True,
    ) -> dict[str, object]:
        _ = context
        labels_clean = self._labels(labels, "labels")
        values_clean = self._numbers(values, "values", minimum=0)
        assert len(labels_clean) == len(
            values_clean
        ), "labels and values must have the same length"
        assert sum(values_clean) > 0, "values must contain a positive total"
        if colors is not None:
            colors = self._labels(colors, "colors")
            assert len(colors) == len(
                labels_clean
            ), "colors length must match labels length"

        figure, axes = self._new_axes(width, height)
        if title:
            axes.set_title(title)
        axes.pie(
            values_clean,
            labels=labels_clean,
            colors=colors,
            autopct="%1.1f%%" if show_percentages else None,
        )
        axes.axis("equal")

        return self._render(
            figure,
            chart_type="pie",
            output_format=output_format,
            title=title,
            width=width,
            height=height,
            dpi=dpi,
            series_names=["values"],
            point_count=len(values_clean),
        )


class BarChartTool(Tool, GraphRenderer):
    """Create a bar chart from categories and numeric series.

    Args:
        categories: Category labels in display order.
        values: Single numeric series matching the categories.
        series: Named numeric series matching the categories.
        series_name: Name used when values provides a single series.
        title: Optional chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.
        output_format: Output image format: png, svg, or pdf.
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Raster dots per inch used for PNG output.
        orientation: Bar orientation: vertical or horizontal.
        stacked: Stack named series instead of grouping them.
        show_legend: Show a legend for named series.
        grid: Show a light value-axis grid.

    Returns:
        Graph image metadata with a base64 data URI.
    """

    def __init__(self, settings: GraphToolSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or GraphToolSettings()
        self.__name__ = "bar"

    async def __call__(
        self,
        categories: list[str],
        *,
        context: ToolCallContext,
        values: list[float] | None = None,
        series: dict[str, list[float]] | None = None,
        series_name: str = "values",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        output_format: GraphOutputFormat = "png",
        width: float = GraphRenderer._DEFAULT_WIDTH,
        height: float = GraphRenderer._DEFAULT_HEIGHT,
        dpi: int = GraphRenderer._DEFAULT_DPI,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        stacked: bool = False,
        show_legend: bool = True,
        grid: bool = False,
    ) -> dict[str, object]:
        _ = context
        categories_clean = self._labels(categories, "categories")
        series_clean = self._series_from_values(
            values, series, series_name=series_name
        )
        self._assert_series_lengths(
            series_clean,
            expected=len(categories_clean),
            axis_name="categories",
        )
        assert orientation in (
            "vertical",
            "horizontal",
        ), "orientation must be vertical or horizontal"

        figure, axes = self._new_axes(width, height)
        positions = list(range(len(categories_clean)))
        series_items = list(series_clean.items())

        if orientation == "vertical":
            self._draw_vertical_bars(axes, positions, series_items, stacked)
            axes.set_xticks(positions)
            axes.set_xticklabels(categories_clean)
            if len(categories_clean) > 8:
                axes.tick_params(axis="x", labelrotation=45)
        else:
            self._draw_horizontal_bars(axes, positions, series_items, stacked)
            axes.set_yticks(positions)
            axes.set_yticklabels(categories_clean)

        self._apply_axes_labels(
            axes,
            title=title,
            x_label=x_label,
            y_label=y_label,
            grid=grid,
        )
        if show_legend and (len(series_items) > 1 or series_name != "values"):
            axes.legend()

        return self._render(
            figure,
            chart_type="bar",
            output_format=output_format,
            title=title,
            width=width,
            height=height,
            dpi=dpi,
            series_names=[name for name, _ in series_items],
            point_count=len(categories_clean) * len(series_items),
        )

    @staticmethod
    def _draw_vertical_bars(
        axes: Any,
        positions: list[int],
        series_items: list[tuple[str, list[float]]],
        stacked: bool,
    ) -> None:
        if stacked:
            bottoms = [0.0] * len(positions)
            for name, values in series_items:
                axes.bar(positions, values, bottom=bottoms, label=name)
                bottoms = [
                    bottom + value for bottom, value in zip(bottoms, values)
                ]
            return

        bar_width = 0.8 / len(series_items)
        start = -0.4 + bar_width / 2
        for index, (name, values) in enumerate(series_items):
            offsets = [
                position + start + index * bar_width for position in positions
            ]
            axes.bar(offsets, values, width=bar_width, label=name)

    @staticmethod
    def _draw_horizontal_bars(
        axes: Any,
        positions: list[int],
        series_items: list[tuple[str, list[float]]],
        stacked: bool,
    ) -> None:
        if stacked:
            lefts = [0.0] * len(positions)
            for name, values in series_items:
                axes.barh(positions, values, left=lefts, label=name)
                lefts = [left + value for left, value in zip(lefts, values)]
            return

        bar_height = 0.8 / len(series_items)
        start = -0.4 + bar_height / 2
        for index, (name, values) in enumerate(series_items):
            offsets = [
                position + start + index * bar_height for position in positions
            ]
            axes.barh(offsets, values, height=bar_height, label=name)


class LineChartTool(Tool, GraphRenderer):
    """Create a line chart from x labels and numeric series.

    Args:
        x_labels: X-axis labels in display order.
        values: Single numeric series matching the x labels.
        series: Named numeric series matching the x labels.
        series_name: Name used when values provides a single series.
        title: Optional chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.
        output_format: Output image format: png, svg, or pdf.
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Raster dots per inch used for PNG output.
        markers: Show circular markers at each data point.
        show_legend: Show a legend for named series.
        grid: Show a light axis grid.

    Returns:
        Graph image metadata with a base64 data URI.
    """

    def __init__(self, settings: GraphToolSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or GraphToolSettings()
        self.__name__ = "line"

    async def __call__(
        self,
        x_labels: list[str],
        *,
        context: ToolCallContext,
        values: list[float] | None = None,
        series: dict[str, list[float]] | None = None,
        series_name: str = "values",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        output_format: GraphOutputFormat = "png",
        width: float = GraphRenderer._DEFAULT_WIDTH,
        height: float = GraphRenderer._DEFAULT_HEIGHT,
        dpi: int = GraphRenderer._DEFAULT_DPI,
        markers: bool = True,
        show_legend: bool = True,
        grid: bool = True,
    ) -> dict[str, object]:
        _ = context
        x_labels_clean = self._labels(x_labels, "x_labels")
        series_clean = self._series_from_values(
            values, series, series_name=series_name
        )
        self._assert_series_lengths(
            series_clean,
            expected=len(x_labels_clean),
            axis_name="x_labels",
        )

        figure, axes = self._new_axes(width, height)
        marker = "o" if markers else None
        for name, y_values in series_clean.items():
            axes.plot(x_labels_clean, y_values, marker=marker, label=name)
        if len(x_labels_clean) > 8:
            axes.tick_params(axis="x", labelrotation=45)
        self._apply_axes_labels(
            axes,
            title=title,
            x_label=x_label,
            y_label=y_label,
            grid=grid,
        )
        if show_legend and (len(series_clean) > 1 or series_name != "values"):
            axes.legend()

        return self._render(
            figure,
            chart_type="line",
            output_format=output_format,
            title=title,
            width=width,
            height=height,
            dpi=dpi,
            series_names=list(series_clean.keys()),
            point_count=len(x_labels_clean) * len(series_clean),
        )


class ScatterPlotTool(Tool, GraphRenderer):
    """Create a scatter plot from numeric x and y values.

    Args:
        x: Numeric x-axis values.
        y: Numeric y-axis values matching x.
        title: Optional chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.
        output_format: Output image format: png, svg, or pdf.
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Raster dots per inch used for PNG output.
        labels: Optional point labels matching x and y.
        series_name: Name used in the legend.
        point_size: Marker area in points squared.
        color: Marker color.
        show_legend: Show a legend for the series.
        grid: Show a light axis grid.

    Returns:
        Graph image metadata with a base64 data URI.
    """

    def __init__(self, settings: GraphToolSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or GraphToolSettings()
        self.__name__ = "scatter"

    async def __call__(
        self,
        x: list[float],
        y: list[float],
        *,
        context: ToolCallContext,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        output_format: GraphOutputFormat = "png",
        width: float = GraphRenderer._DEFAULT_WIDTH,
        height: float = GraphRenderer._DEFAULT_HEIGHT,
        dpi: int = GraphRenderer._DEFAULT_DPI,
        labels: list[str] | None = None,
        series_name: str = "values",
        point_size: float = 36,
        color: str | None = None,
        show_legend: bool = False,
        grid: bool = True,
    ) -> dict[str, object]:
        _ = context
        x_values = self._numbers(x, "x")
        y_values = self._numbers(y, "y")
        assert len(x_values) == len(
            y_values
        ), "x and y must have the same length"
        if labels is not None:
            labels = self._labels(labels, "labels")
            assert len(labels) == len(
                x_values
            ), "labels length must match x and y length"
        assert point_size > 0, "point_size must be greater than zero"

        figure, axes = self._new_axes(width, height)
        axes.scatter(
            x_values,
            y_values,
            s=point_size,
            c=color,
            label=series_name,
        )
        if labels:
            for x_value, y_value, label in zip(x_values, y_values, labels):
                axes.annotate(label, (x_value, y_value))
        self._apply_axes_labels(
            axes,
            title=title,
            x_label=x_label,
            y_label=y_label,
            grid=grid,
        )
        if show_legend:
            axes.legend()

        return self._render(
            figure,
            chart_type="scatter",
            output_format=output_format,
            title=title,
            width=width,
            height=height,
            dpi=dpi,
            series_names=[series_name],
            point_count=len(x_values),
        )


class HistogramTool(Tool, GraphRenderer):
    """Create a histogram from numeric values.

    Args:
        values: Numeric values to bucket.
        bins: Number of histogram bins.
        title: Optional chart title.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.
        output_format: Output image format: png, svg, or pdf.
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Raster dots per inch used for PNG output.
        color: Bar fill color.
        cumulative: Accumulate bin counts from left to right.
        grid: Show a light count-axis grid.

    Returns:
        Graph image metadata with a base64 data URI.
    """

    def __init__(self, settings: GraphToolSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or GraphToolSettings()
        self.__name__ = "histogram"

    async def __call__(
        self,
        values: list[float],
        *,
        context: ToolCallContext,
        bins: int = 10,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        output_format: GraphOutputFormat = "png",
        width: float = GraphRenderer._DEFAULT_WIDTH,
        height: float = GraphRenderer._DEFAULT_HEIGHT,
        dpi: int = GraphRenderer._DEFAULT_DPI,
        color: str | None = None,
        cumulative: bool = False,
        grid: bool = True,
    ) -> dict[str, object]:
        _ = context
        values_clean = self._numbers(values, "values")
        assert bins > 0, "bins must be greater than zero"

        figure, axes = self._new_axes(width, height)
        axes.hist(
            values_clean,
            bins=bins,
            color=color,
            edgecolor="white",
            cumulative=cumulative,
        )
        self._apply_axes_labels(
            axes,
            title=title,
            x_label=x_label,
            y_label=y_label,
            grid=grid,
        )

        return self._render(
            figure,
            chart_type="histogram",
            output_format=output_format,
            title=title,
            width=width,
            height=height,
            dpi=dpi,
            series_names=["values"],
            point_count=len(values_clean),
        )


class GraphToolSet(ToolSet):
    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
        settings: GraphToolSettings | None = None,
    ) -> None:
        settings = settings or GraphToolSettings()
        tools = [
            PieChartTool(settings),
            BarChartTool(settings),
            LineChartTool(settings),
            ScatterPlotTool(settings),
            HistogramTool(settings),
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )
