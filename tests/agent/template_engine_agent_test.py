from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from os import path
from sys import modules
from types import ModuleType
from typing import Any
from unittest import TestCase


class _SimpleTemplate:
    def __init__(self, source: str) -> None:
        self.source = source

    def render(self, **kwargs: Any) -> str:
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

            def render(**kwargs: Any) -> str:
                name = kwargs.get("name", "")
                return f"agent.md:{name}"

            tmpl.render = render  # type: ignore[attr-defined]
            return tmpl
        source = self.loader.get_source(self, template_id)
        return _SimpleTemplate(source)


_jinja = ModuleType("jinja2")
_jinja.Environment = _SimpleEnvironment
_jinja.FileSystemLoader = _SimpleLoader
_jinja.Template = _SimpleTemplate
modules.setdefault("jinja2", _jinja)


def _load_module():
    spec = spec_from_file_location(
        "avalan.agent.renderer",
        path.join("src", "avalan", "agent", "renderer.py"),
    )
    module = module_from_spec(spec)
    # stub packages
    avalan = ModuleType("avalan")
    agent_pkg = ModuleType("avalan.agent")
    event_pkg = ModuleType("avalan.event.manager")
    agent_engine_pkg = ModuleType("avalan.agent.engine")
    memory_pkg = ModuleType("avalan.memory.manager")
    model_pkg = ModuleType("avalan.model.manager")
    engine_pkg = ModuleType("avalan.model.engine")
    tool_pkg = ModuleType("avalan.tool.manager")
    entities_pkg = ModuleType("avalan.entities")
    modules.update(
        {
            "avalan": avalan,
            "avalan.agent": agent_pkg,
            "avalan.agent.engine": agent_engine_pkg,
            "avalan.event.manager": event_pkg,
            "avalan.memory.manager": memory_pkg,
            "avalan.model.manager": model_pkg,
            "avalan.model.engine": engine_pkg,
            "avalan.tool.manager": tool_pkg,
            "avalan.entities": entities_pkg,
        }
    )

    class EngineAgent:  # minimal base
        def __init__(
            self,
            _model: object,
            _memory: object,
            _tool: object,
            _event_manager: object,
            _model_manager: object,
            _engine_uri: object,
            *,
            name: str | None = None,
            id: Any | None = None,
        ) -> None:
            self._name = name

    class EventManager:  # pragma: no cover - stub
        pass

    class MemoryManager:  # pragma: no cover - stub
        pass

    class ModelManager:  # pragma: no cover - stub
        pass

    class Engine:  # pragma: no cover - stub
        pass

    class ToolManager:  # pragma: no cover - stub
        pass

    @dataclass(slots=True)
    class Role:
        persona: list[str]

    @dataclass(slots=True)
    class Goal:
        task: str
        instructions: list[str]

    @dataclass(slots=True)
    class GenerationSettings:
        template_vars: dict | None = None

    @dataclass(slots=True)
    class Specification:
        role: Role | str | list[str] | None = None
        goal: Goal | None = None
        system_prompt: str | None = None
        developer_prompt: str | None = None
        rules: list[str] | None = field(default_factory=list)
        settings: GenerationSettings | None = None
        template_id: str | None = None
        template_vars: dict | None = None

    @dataclass(slots=True)
    class EngineUri:  # pragma: no cover - stub
        pass

    agent_pkg.Role = Role
    agent_pkg.Goal = Goal
    agent_pkg.Specification = Specification
    entities_pkg.GenerationSettings = GenerationSettings
    entities_pkg.EngineUri = EngineUri
    agent_engine_pkg.EngineAgent = EngineAgent
    event_pkg.EventManager = EventManager
    memory_pkg.MemoryManager = MemoryManager
    model_pkg.ModelManager = ModelManager
    engine_pkg.Engine = Engine
    tool_pkg.ToolManager = ToolManager
    agent_pkg.EngineAgent = EngineAgent

    spec.loader.exec_module(module)
    return module


module = _load_module()
TemplateEngineAgent = module.TemplateEngineAgent
Renderer = module.Renderer
Role = modules["avalan.agent"].Role
Goal = modules["avalan.agent"].Goal
Specification = modules["avalan.agent"].Specification
GenerationSettings = modules["avalan.entities"].GenerationSettings


class DummyRenderer(Renderer):
    def __init__(self) -> None:  # type: ignore[override]
        super().__init__(clean_spaces=False)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, template_id: str, **kwargs: Any) -> str:  # type: ignore[override]
        self.calls.append((template_id, kwargs.copy()))
        return f"{template_id}:{len(kwargs)}"

    def from_string(
        self,
        template: str,
        template_vars: dict | None = None,
        encoding: str = "utf-8",
    ) -> str:  # type: ignore[override]
        return (
            template.format(**template_vars).encode(encoding)
            if template_vars
            else template
        )


class TemplateEngineAgentPrepareTestCase(TestCase):
    def setUp(self) -> None:
        self.renderer = DummyRenderer()
        self.agent = TemplateEngineAgent(
            model=object(),
            memory=modules["avalan.memory.manager"].MemoryManager(),
            tool=modules["avalan.tool.manager"].ToolManager(),
            event_manager=modules["avalan.event.manager"].EventManager(),
            model_manager=modules["avalan.model.manager"].ModelManager(),
            renderer=self.renderer,
            engine_uri=modules["avalan.entities"].EngineUri(),
            name="Bob",
        )

    def test_system_and_developer_prompt_short_circuit(self) -> None:
        spec = Specification(system_prompt="sys", developer_prompt="dev")
        result = self.agent._prepare_call(spec, "hi")
        self.assertEqual(result["system_prompt"], "sys")
        self.assertEqual(result["developer_prompt"], "dev")
        self.assertEqual(len(self.renderer.calls), 0)

    def test_template_vars_and_settings_merge(self) -> None:
        spec = Specification(
            role=Role(persona=["role {verb}"]),
            goal=Goal(task="do {verb}", instructions=["inst {verb}"]),
            rules=["rule {verb}"],
            template_vars={"verb": "run"},
            settings=GenerationSettings(template_vars={"verb": "walk"}),
            developer_prompt="dev",
        )
        result = self.agent._prepare_call(spec, "hi")
        call = self.renderer.calls[-1]
        self.assertEqual(call[0], "agent.md")
        self.assertEqual(call[1]["name"], "Bob")
        self.assertEqual(call[1]["roles"], [b"role walk"])
        self.assertEqual(call[1]["task"], b"do walk")
        self.assertEqual(call[1]["instructions"], [b"inst walk"])
        self.assertEqual(call[1]["rules"], [b"rule walk"])
        self.assertEqual(result["developer_prompt"], "dev")

    def test_role_variants(self) -> None:
        spec = Specification(
            role="single",
            template_vars={"x": "y"},
            goal=Goal(task="t {x}", instructions=["i {x}"]),
            rules=["r {x}"],
        )
        self.agent._prepare_call(spec, "hi")
        self.assertEqual(self.renderer.calls[-1][1]["roles"], ["single"])

        spec.role = ["a", "b"]
        self.agent._prepare_call(spec, "hi")
        self.assertEqual(self.renderer.calls[-1][1]["roles"], ["a", "b"])

        spec.role = Role(persona=["p {x}"])
        self.agent._prepare_call(spec, "hi")
        self.assertEqual(self.renderer.calls[-1][1]["roles"], [b"p y"])
