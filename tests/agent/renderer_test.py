from importlib.util import module_from_spec, spec_from_file_location
from os import path
from sys import modules
from types import ModuleType
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch


class _SimpleTemplate:
    def __init__(self, source: str) -> None:
        self.source = source

    def render(self, **kwargs: str) -> str:
        output = self.source
        for key, value in kwargs.items():
            output = output.replace(f"{{{{{key}}}}}", str(value))
        return output


class _SimpleLoader:
    def __init__(self, searchpath: str) -> None:
        self.searchpath = searchpath

    def get_source(self, _env: object, template_id: str) -> str:
        with open(
            path.join(self.searchpath, template_id), "r", encoding="utf-8"
        ) as fh:
            return fh.read()


class _SimpleEnvironment:
    def __init__(self, loader: _SimpleLoader, **_kwargs: object) -> None:
        self.loader = loader

    def get_template(self, template_id: str) -> _SimpleTemplate:
        if template_id == "agent.md":
            tmpl = _SimpleTemplate("")

            def render(**kwargs: str) -> str:
                name = kwargs.get("name", "")
                return (
                    "You are a helpful assistant.\n\nYour name is"
                    f" {name}.\n\n\n\n"
                )

            tmpl.render = render  # type: ignore[attr-defined]
            return tmpl
        source = self.loader.get_source(self, template_id)
        return _SimpleTemplate(source)


_jinja = ModuleType("jinja2")
_jinja.Environment = _SimpleEnvironment
_jinja.FileSystemLoader = _SimpleLoader
_jinja.Template = _SimpleTemplate
modules.setdefault("jinja2", _jinja)


def _load_renderer() -> type:
    spec = spec_from_file_location(
        "avalan.agent.renderer",
        path.join("src", "avalan", "agent", "renderer.py"),
    )
    module = module_from_spec(spec)
    stubs = {
        "avalan": ModuleType("avalan"),
        "avalan.agent": ModuleType("avalan.agent"),
        "avalan.agent.engine": ModuleType("avalan.agent.engine"),
        "avalan.memory.manager": ModuleType("avalan.memory.manager"),
        "avalan.model.engine": ModuleType("avalan.model.engine"),
        "avalan.model.manager": ModuleType("avalan.model.manager"),
        "avalan.tool.manager": ModuleType("avalan.tool.manager"),
        "avalan.entities": ModuleType("avalan.entities"),
        "avalan.event.manager": ModuleType("avalan.event.manager"),
    }
    for name, mod in stubs.items():
        modules.setdefault(name, mod)
    stubs["avalan.agent.engine"].EngineAgent = object
    stubs["avalan.memory.manager"].MemoryManager = object
    stubs["avalan.model.engine"].Engine = object
    stubs["avalan.model.manager"].ModelManager = object
    stubs["avalan.tool.manager"].ToolManager = object
    stubs["avalan.entities"].EngineUri = object
    stubs["avalan.event.manager"].EventManager = object
    stubs["avalan.agent"].Role = object
    stubs["avalan.agent"].Specification = object
    spec.loader.exec_module(module)
    return module.Renderer


Renderer = _load_renderer()


class RendererTestCase(TestCase):
    def test_render_basic(self) -> None:
        renderer = Renderer()
        result = renderer("agent.md", name="Leo")
        expected = "You are a helpful assistant.\n\nYour name is Leo.\n\n\n"
        self.assertEqual(result, expected)

    def test_custom_template_path(self) -> None:
        with TemporaryDirectory() as tmp:
            path_greet = path.join(tmp, "greet.txt")
            with open(path_greet, "w", encoding="utf-8") as fh:
                fh.write("Hello {{name}}!")
            renderer = Renderer(tmp)
            self.assertEqual(renderer("greet.txt", name="Bob"), "Hello Bob!")

    def test_from_string(self) -> None:
        renderer = Renderer()
        tmpl = "Hi {{name}}"
        self.assertEqual(
            renderer.from_string(tmpl, {"name": "Ada"}), b"Hi Ada"
        )
        self.assertEqual(renderer.from_string(tmpl), tmpl)

    def test_template_caching(self) -> None:
        Renderer._templates.clear()
        renderer = Renderer()
        with patch.object(
            renderer._environment,
            "get_template",
            wraps=renderer._environment.get_template,
        ) as mock:
            renderer("agent.md", name="A")
            renderer("agent.md", name="B")
            self.assertEqual(mock.call_count, 1)

    def test_no_clean_spaces(self) -> None:
        renderer = Renderer(clean_spaces=False)
        result = renderer("agent.md", name="Leo")
        self.assertIn("Your name is Leo.", result)


if __name__ == "__main__":  # pragma: no cover - convenience
    main()
