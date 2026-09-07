"""Resolve static execution references before a deployment is activated."""

from .deployment import (
    DeploymentFile,
    DeploymentFileRoot,
    ExecutionDeploymentError,
    deployment_path,
)

from collections.abc import Mapping
from hashlib import sha256
from json import JSONDecodeError
from json import loads as json_loads
from pathlib import Path
from tomllib import TOMLDecodeError
from tomllib import loads as toml_loads

from jinja2 import Environment, TemplateSyntaxError
from jinja2.meta import find_referenced_templates


class ExecutionClosure:
    """Discover files from supported task, agent and Flow syntax."""

    def __init__(self, application_root: Path, runtime_root: Path) -> None:
        self.roots = {
            DeploymentFileRoot.APPLICATION: application_root.resolve(
                strict=True
            ),
            DeploymentFileRoot.RUNTIME: runtime_root.resolve(strict=True),
        }
        self.files: dict[tuple[DeploymentFileRoot, str], DeploymentFile] = {}
        self._visited: set[tuple[DeploymentFileRoot, str, str]] = set()
        self.tools: set[str] = set()
        self.tool_filters: list[tuple[str, ...] | None] = []

    def read(self, root: DeploymentFileRoot, path: str) -> bytes:
        deployment_path(path)
        parent = self.roots[root]
        candidate = parent / path
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(parent) or candidate != resolved:
                raise ExecutionDeploymentError("file.root")
            if not resolved.is_file():
                raise ExecutionDeploymentError("file.type")
            content = resolved.read_bytes()
        except OSError:
            raise ExecutionDeploymentError("file.unavailable") from None
        value = DeploymentFile(
            root=root, path=path, sha256=sha256(content).hexdigest()
        )
        previous = self.files.get((root, path))
        if previous is not None and previous != value:
            raise ExecutionDeploymentError("file.changed")
        self.files[(root, path)] = value
        return content

    def collect(
        self, task_ref: str, *, file_delivery: bool = False
    ) -> tuple[DeploymentFile, ...]:
        self._configuration(
            DeploymentFileRoot.APPLICATION,
            task_ref,
            "task",
            file_delivery=file_delivery,
        )
        return tuple(self.files[key] for key in sorted(self.files))

    def _reference(self, parent: str, value: object) -> str:
        if (
            not isinstance(value, str)
            or not value
            or any(
                marker in value for marker in ("{{", "{%", "://", "\\", "\x00")
            )
        ):
            raise ExecutionDeploymentError("reference")
        relative = value.split("#", 1)[0]
        if not relative:
            return parent
        deployment_path(relative)
        return deployment_path((Path(parent).parent / relative).as_posix())

    def _configuration(
        self,
        root: DeploymentFileRoot,
        path: str,
        kind: str,
        *,
        file_delivery: bool = False,
    ) -> None:
        identity = (root, path, kind + (":files" if file_delivery else ""))
        if identity in self._visited:
            return
        self._visited.add(identity)
        try:
            raw = toml_loads(self.read(root, path).decode("utf-8"))
        except (TOMLDecodeError, UnicodeError):
            raise ExecutionDeploymentError("configuration") from None
        if kind == "task":
            execution = self._section(raw, "execution")
            target = execution.get("type")
            if target not in {"task", "agent", "flow"}:
                raise ExecutionDeploymentError("target")
            self._configuration(
                root,
                self._reference(path, execution.get("ref")),
                str(target),
                file_delivery=file_delivery
                or self._section(raw, "input").get("type")
                in {"file", "file_array"},
            )
            for name in ("input", "output"):
                self._schemas(root, path, self._section(raw, name))
        elif kind == "agent":
            agent = self._section(raw, "agent")
            enabled = self._section(raw, "tool").get("enable")
            if enabled is None:
                self.tool_filters.append(None)
            elif isinstance(enabled, str):
                self.tool_filters.append((enabled,))
            elif isinstance(enabled, list) and all(
                isinstance(value, str) for value in enabled
            ):
                self.tool_filters.append(tuple(enabled))
            else:
                raise ExecutionDeploymentError("agent.tools")

            # The standard orchestrator uses the bundled renderer; task file
            # input rendering also resolves user templates beside the agent.
            template = (
                "agent_json.md" if agent.get("type") == "json" else "agent.md"
            )
            self._template(
                DeploymentFileRoot.RUNTIME, "agent/templates/" + template
            )
            user = agent.get("user_template")
            if user is not None:
                if file_delivery:
                    self._template(
                        root,
                        self._reference(path, user),
                        base=Path(path).parent.as_posix(),
                    )
                self._template(
                    DeploymentFileRoot.RUNTIME,
                    self._reference("agent/templates/agent.md", user),
                )
            response = self._section(
                self._section(raw, "run"), "response_format"
            )
            self._schemas(root, path, response)
            self._schemas(root, path, self._section(response, "json_schema"))
            for name in (
                "system",
                "developer",
                "instructions",
                "user",
                "goal_instructions",
                "role",
                "task",
                "rules",
            ):
                inline = agent.get(name)
                if isinstance(inline, str):
                    self._inline(inline)
        elif kind == "flow":
            for name in ("input", "output"):
                self._schemas(root, path, self._section(raw, name))
            graph = self._section(raw, "graph")
            if graph.get("path") is not None:
                self.read(root, self._reference(path, graph["path"]))
            nodes = self._section(raw, "nodes")
            for node in nodes.values():
                if not isinstance(node, Mapping):
                    raise ExecutionDeploymentError("flow.nodes")
                target = node.get("type")
                if target == "agent":
                    self._configuration(
                        root,
                        self._reference(path, node.get("ref")),
                        "agent",
                        file_delivery=file_delivery,
                    )
                elif target == "subflow":
                    self._configuration(
                        root,
                        self._reference(path, node.get("ref")),
                        "flow",
                        file_delivery=file_delivery,
                    )
                elif target == "tool":
                    ref = node.get("ref")
                    if not isinstance(ref, str) or not ref:
                        raise ExecutionDeploymentError("flow.tool")
                    self.tools.add(ref)
                self._schemas(root, path, node)

    def _schemas(
        self,
        root: DeploymentFileRoot,
        path: str,
        section: Mapping[str, object],
    ) -> None:
        ref = section.get("schema_ref")
        if ref is not None:
            self._schema(root, self._reference(path, ref))
        schema = section.get("schema")
        if isinstance(schema, Mapping | list):
            self._schema_refs(root, path, schema)

    def _schema(self, root: DeploymentFileRoot, path: str) -> None:
        identity = (root, path, "schema")
        if identity in self._visited:
            return
        self._visited.add(identity)
        try:
            value = json_loads(self.read(root, path))
        except (JSONDecodeError, UnicodeError):
            raise ExecutionDeploymentError("schema") from None
        self._schema_refs(root, path, value)

    def _schema_refs(
        self, root: DeploymentFileRoot, path: str, value: object
    ) -> None:
        if isinstance(value, Mapping):
            reference = value.get("$ref")
            if reference is not None and (
                not isinstance(reference, str) or not reference.startswith("#")
            ):
                self._schema(root, self._reference(path, reference))
            for child in value.values():
                self._schema_refs(root, path, child)
        elif isinstance(value, list):
            for child in value:
                self._schema_refs(root, path, child)

    def _template(
        self, root: DeploymentFileRoot, path: str, *, base: str | None = None
    ) -> None:
        base = base if base is not None else Path(path).parent.as_posix()
        identity = (root, path, "template:" + base)
        if identity in self._visited:
            return
        self._visited.add(identity)
        try:
            ast = Environment().parse(self.read(root, path).decode("utf-8"))
            for reference in find_referenced_templates(ast):
                self._template(
                    root,
                    self._reference(
                        (Path(base) / "template").as_posix(), reference
                    ),
                    base=base,
                )
        except (TemplateSyntaxError, UnicodeError):
            raise ExecutionDeploymentError("template") from None

    def _inline(self, source: str) -> None:
        try:
            ast = Environment().parse(source)
            if tuple(find_referenced_templates(ast)):
                raise ExecutionDeploymentError("inline.include")
        except TemplateSyntaxError:
            raise ExecutionDeploymentError("template") from None

    @staticmethod
    def _section(
        value: Mapping[str, object], name: str
    ) -> Mapping[str, object]:
        section = value.get(name, {})
        if not isinstance(section, Mapping):
            raise ExecutionDeploymentError("configuration.section")
        return section
