import importlib
import inspect
import pkgutil

import avalan.tool as tool_pkg
from avalan.tool import Tool


def _iter_tool_classes() -> list[type[Tool]]:
    classes: list[type[Tool]] = []
    for module_info in pkgutil.walk_packages(
        tool_pkg.__path__, tool_pkg.__name__ + "."
    ):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Tool) and obj is not Tool:
                classes.append(obj)
    return classes


def test_tool_docstrings_follow_format() -> None:
    tool_classes = _iter_tool_classes()
    assert tool_classes, "No tool implementations discovered."

    for cls in tool_classes:
        if inspect.isabstract(cls) or cls.__name__ == "DatabaseTool":
            continue

        doc = inspect.getdoc(cls)
        assert doc, f"{cls.__name__} must define a docstring."

        lines = doc.splitlines()
        summary = lines[0].strip()

        assert summary, f"{cls.__name__} docstring must start with a summary."
        assert summary[
            0
        ].isupper(), (
            f"{cls.__name__} summary must start with an uppercase letter."
        )
        assert summary.endswith(
            "."
        ), f"{cls.__name__} summary must end with a period."
        assert (
            len(summary.split()) >= 3
        ), f"{cls.__name__} summary must be descriptive."

        assert (
            len(lines) > 2
        ), f"{cls.__name__} docstring must include Args/Returns sections."
        assert lines[1] == "", (
            f"{cls.__name__} docstring must include a blank line after the"
            " summary."
        )

        normalized_lines = [line.strip() for line in lines]
        assert (
            "Args:" in normalized_lines
        ), f"{cls.__name__} docstring must document arguments."
        assert "context:" not in normalized_lines, (
            f"{cls.__name__} docstring must omit the implicit context"
            " parameter."
        )
        assert (
            "Returns:" in normalized_lines
        ), f"{cls.__name__} docstring must document return values."
