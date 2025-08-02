from importlib.util import module_from_spec, spec_from_file_location
from os import path
from sys import modules
from types import ModuleType
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch


def load_renderer():
    spec = spec_from_file_location(
        "avalan.agent.renderer",
        path.join("src", "avalan", "agent", "renderer.py"),
    )

    stubs = {
        "avalan": ModuleType("avalan"),
        "avalan.agent": ModuleType("avalan.agent"),
        "avalan.agent.engine": ModuleType("avalan.agent.engine"),
        "avalan.memory.manager": ModuleType("avalan.memory.manager"),
        "avalan.model.engine": ModuleType("avalan.model.engine"),
        "avalan.tool.manager": ModuleType("avalan.tool.manager"),
    }

    for name, module in stubs.items():
        modules.setdefault(name, module)

    stubs["avalan.agent.engine"].EngineAgent = object
    stubs["avalan.memory.manager"].MemoryManager = object
    stubs["avalan.model.engine"].Engine = object
    stubs["avalan.tool.manager"].ToolManager = object
    stubs["avalan.agent"].Role = object
    stubs["avalan.agent"].Specification = object

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Renderer


class RendererTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Renderer = load_renderer()

    def test_render_basic(self):
        renderer = self.Renderer()
        result = renderer("agent.md", name="Leo")
        expected = "You are a helpful assistant.\n\nYour name is Leo.\n\n\n"
        self.assertEqual(result, expected)

    def test_custom_template_path(self):
        with TemporaryDirectory() as tmp:
            path_greet = path.join(tmp, "greet.txt")
            with open(path_greet, "w", encoding="utf-8") as fh:
                fh.write("Hello {{name}}!")
            renderer = self.Renderer(tmp)
            self.assertEqual(renderer("greet.txt", name="Bob"), "Hello Bob!")

    def test_from_string(self):
        renderer = self.Renderer()
        tmpl = "Hi {{name}}"
        self.assertEqual(
            renderer.from_string(tmpl, {"name": "Ada"}), b"Hi Ada"
        )
        self.assertEqual(renderer.from_string(tmpl), tmpl)

    def test_template_caching(self):
        # clear cache to observe calls
        self.Renderer._templates.clear()
        renderer = self.Renderer()
        with patch.object(
            renderer._environment,
            "get_template",
            wraps=renderer._environment.get_template,
        ) as mock:
            renderer("agent.md", name="A")
            renderer("agent.md", name="B")
            self.assertEqual(mock.call_count, 1)

    def test_no_clean_spaces(self):
        renderer = self.Renderer(clean_spaces=False)
        result = renderer("agent.md", name="Leo")
        self.assertTrue(
            result.strip().startswith("You are a helpful assistant.")
        )
        self.assertIn("Your name is Leo.", result)


if __name__ == "__main__":
    main()
