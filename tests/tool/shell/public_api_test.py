from ast import (
    Attribute,
    Call,
    Constant,
    Import,
    ImportFrom,
    Name,
    parse,
    walk,
)
from asyncio import run as asyncio_run
from dataclasses import FrozenInstanceError
from importlib import import_module
from inspect import iscoroutinefunction, signature
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan import filesystem
from avalan.tool import ToolSet
from avalan.tool.shell import (
    SHELL_COMMAND_DEFINITIONS,
    SHELL_COMMAND_IDS,
    SHELL_COMMANDS,
    CommandExecutor,
    ExecutableLookup,
    ExecutableResolver,
    ExecutionPolicy,
    ExecutionSpec,
    GeneratedFile,
    GeneratedOutputPlan,
    LocalCommandExecutor,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellDependencyGroup,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
    enables_shell_tools,
    normalize_shell_enabled_tools,
    should_append_shell_toolset,
    unavailable_executable_lookup,
)
from avalan.tool.shell.formatting import format_shell_result


class ShellPublicApiTest(TestCase):
    def test_package_exports_public_contract(self) -> None:
        self.assertIs(
            ShellToolSettings,
            import_module("avalan.tool.shell").ShellToolSettings,
        )
        self.assertIs(
            ShellToolSet, import_module("avalan.tool.shell").ShellToolSet
        )
        self.assertIs(
            ExecutionPolicy, import_module("avalan.tool.shell").ExecutionPolicy
        )
        self.assertIs(
            LocalCommandExecutor,
            import_module("avalan.tool.shell").LocalCommandExecutor,
        )
        self.assertIs(
            CommandExecutor, import_module("avalan.tool.shell").CommandExecutor
        )
        self.assertIs(
            ExecutableResolver,
            import_module("avalan.tool.shell").ExecutableResolver,
        )
        self.assertIs(
            ExecutableLookup,
            import_module("avalan.tool.shell").ExecutableLookup,
        )
        self.assertIs(
            TrustedExecutableResolver,
            import_module("avalan.tool.shell").TrustedExecutableResolver,
        )
        self.assertIs(
            unavailable_executable_lookup,
            import_module("avalan.tool.shell").unavailable_executable_lookup,
        )
        self.assertIs(
            PathOperand, import_module("avalan.tool.shell").PathOperand
        )
        self.assertIs(
            ShellCommandRequest,
            import_module("avalan.tool.shell").ShellCommandRequest,
        )
        self.assertIs(
            ExecutionSpec, import_module("avalan.tool.shell").ExecutionSpec
        )
        self.assertIs(
            ShellPolicyDenied,
            import_module("avalan.tool.shell").ShellPolicyDenied,
        )
        self.assertIs(
            GeneratedFile, import_module("avalan.tool.shell").GeneratedFile
        )
        self.assertIs(
            GeneratedOutputPlan,
            import_module("avalan.tool.shell").GeneratedOutputPlan,
        )
        self.assertIs(
            ShellExecutionStatus,
            import_module("avalan.tool.shell").ShellExecutionStatus,
        )
        self.assertIs(
            ShellExecutionErrorCode,
            import_module("avalan.tool.shell").ShellExecutionErrorCode,
        )
        self.assertIs(
            ShellOutputKind, import_module("avalan.tool.shell").ShellOutputKind
        )
        self.assertIs(
            ShellCommandDefinition,
            import_module("avalan.tool.shell").ShellCommandDefinition,
        )
        self.assertIs(
            ShellDependencyGroup,
            import_module("avalan.tool.shell").ShellDependencyGroup,
        )
        self.assertIs(
            SHELL_COMMANDS, import_module("avalan.tool.shell").SHELL_COMMANDS
        )
        self.assertIs(
            SHELL_COMMAND_IDS,
            import_module("avalan.tool.shell").SHELL_COMMAND_IDS,
        )
        self.assertIs(
            SHELL_COMMAND_DEFINITIONS,
            import_module("avalan.tool.shell").SHELL_COMMAND_DEFINITIONS,
        )
        self.assertIs(
            normalize_shell_enabled_tools,
            import_module("avalan.tool.shell").normalize_shell_enabled_tools,
        )
        self.assertIs(
            enables_shell_tools,
            import_module("avalan.tool.shell").enables_shell_tools,
        )
        self.assertIs(
            should_append_shell_toolset,
            import_module("avalan.tool.shell").should_append_shell_toolset,
        )

    def test_package_does_not_declare_all(self) -> None:
        self.assertFalse(
            hasattr(import_module("avalan.tool.shell"), "__all__")
        )

    def test_expected_layout_modules_import(self) -> None:
        for module_name in (
            "avalan.tool.shell.entities",
            "avalan.tool.shell.executor",
            "avalan.tool.shell.formatting",
            "avalan.tool.shell.opt_in",
            "avalan.tool.shell.policy",
            "avalan.tool.shell.registry",
            "avalan.tool.shell.resolver",
            "avalan.tool.shell.settings",
            "avalan.tool.shell.toolset",
        ):
            self.assertEqual(import_module(module_name).__name__, module_name)

    def test_settings_lock_backend_default(self) -> None:
        settings = ShellToolSettings()

        self.assertEqual(settings.backend, "local")

    def test_settings_are_frozen(self) -> None:
        settings = ShellToolSettings()

        with self.assertRaises(FrozenInstanceError):
            settings.backend = "remote"

    def test_shell_toolset_is_namespaced_and_empty_until_tools_land(
        self,
    ) -> None:
        toolset = ShellToolSet()

        self.assertIsInstance(toolset, ToolSet)
        self.assertEqual(toolset.namespace, "shell")
        self.assertEqual(toolset.tools, [])
        self.assertEqual(toolset.json_schemas(), [])

    def test_shell_toolset_accepts_explicit_settings(self) -> None:
        settings = ShellToolSettings()
        toolset = ShellToolSet(settings=settings, namespace="custom")

        self.assertEqual(toolset.namespace, "custom")

    def test_policy_denied_preserves_stable_error_code(self) -> None:
        error = ShellPolicyDenied(
            ShellExecutionErrorCode.DENIED_COMMAND,
            "denied",
        )

        self.assertEqual(
            error.error_code, ShellExecutionErrorCode.DENIED_COMMAND
        )
        self.assertEqual(str(error), "denied")

    def test_formatter_is_inert_until_results_land(self) -> None:
        with self.assertRaises(NotImplementedError):
            format_shell_result(object())


class ShellAsyncContractTest(IsolatedAsyncioTestCase):
    def test_policy_normalize_signature_is_typed_async(self) -> None:
        normalize = ExecutionPolicy.normalize
        parameters = signature(normalize).parameters

        self.assertTrue(iscoroutinefunction(normalize))
        self.assertIs(
            parameters["request"].annotation,
            ShellCommandRequest,
        )
        self.assertIs(signature(normalize).return_annotation, ExecutionSpec)

    async def test_command_executor_protocol_stub_is_inert(self) -> None:
        class InertCommandExecutor(CommandExecutor):
            pass

        executor = InertCommandExecutor()

        with self.assertRaises(NotImplementedError):
            await executor.execute(object())

    async def test_executable_resolver_protocol_stub_is_inert(self) -> None:
        class InertExecutableResolver(ExecutableResolver):
            pass

        resolver = InertExecutableResolver()

        with self.assertRaises(NotImplementedError):
            await resolver.resolve(SHELL_COMMAND_DEFINITIONS["rg"])

    async def test_policy_normalize_is_async_and_inert(self) -> None:
        policy = ExecutionPolicy()
        request = ShellCommandRequest(
            tool_name="shell.rg",
            command="rg",
            options={},
            paths=(),
            cwd=None,
        )

        with self.assertRaises(NotImplementedError):
            await policy.normalize(request)

    async def test_local_executor_is_async_and_inert(self) -> None:
        executor = LocalCommandExecutor()

        with self.assertRaises(NotImplementedError):
            await executor.execute(object())


class ShellRuntimeStaticContractTest(TestCase):
    def test_shell_runtime_uses_async_boundaries(self) -> None:
        violations: list[str] = []
        for source_path in _shell_source_paths():
            tree = parse(source_path.read_text(encoding="utf-8"))
            for node in walk(tree):
                if _is_forbidden_import(node):
                    violations.append(
                        f"{source_path.name}: import at line {node.lineno}"
                    )
                if isinstance(node, Call):
                    violation = _forbidden_call_name(node)
                    if violation:
                        violations.append(
                            f"{source_path.name}: {violation} at line"
                            f" {node.lineno}"
                        )
                    if any(
                        keyword.arg == "shell"
                        and _is_true_literal(keyword.value)
                        for keyword in node.keywords
                    ):
                        violations.append(
                            f"{source_path.name}: shell=True at line"
                            f" {node.lineno}"
                        )

        self.assertEqual(violations, [])

    def test_policy_filesystem_helpers_are_async_and_bounded(self) -> None:
        self.assertTrue(iscoroutinefunction(filesystem.read_bytes_prefix))
        self.assertTrue(iscoroutinefunction(filesystem.stat_path))

        with self.assertRaises(AssertionError):
            asyncio_run(filesystem.read_bytes_prefix("file.bin", True))  # type: ignore[arg-type]


def _shell_source_paths() -> tuple[Path, ...]:
    shell_root = (
        Path(__file__).parents[3] / "src" / "avalan" / "tool" / "shell"
    )
    return tuple(sorted(shell_root.glob("*.py")))


def _is_forbidden_import(node: object) -> bool:
    if isinstance(node, Import):
        return any(
            alias.name in {"subprocess", "asyncio.to_thread"}
            or alias.name.startswith("subprocess.")
            for alias in node.names
        )
    if isinstance(node, ImportFrom):
        module_name = node.module or ""
        if module_name == "subprocess":
            return True
        return module_name == "asyncio" and any(
            alias.name in {"to_thread", "create_subprocess_shell"}
            for alias in node.names
        )
    return False


def _forbidden_call_name(node: Call) -> str | None:
    name = _call_name(node.func)
    forbidden_names = {
        "create_subprocess_shell",
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "Path.open",
        "Path.read_text",
        "Path.read_bytes",
        "Path.write_text",
        "Path.write_bytes",
        "Path.stat",
        "open",
        "to_thread",
    }
    if name in forbidden_names:
        return name
    if name.startswith("asyncio.") and name in {
        "asyncio.to_thread",
        "asyncio.create_subprocess_shell",
    }:
        return name
    return None


def _call_name(node: object) -> str:
    if isinstance(node, Name):
        return node.id
    if isinstance(node, Attribute):
        owner = _call_name(node.value)
        if owner:
            return f"{owner}.{node.attr}"
        return node.attr
    if isinstance(node, Call):
        return _call_name(node.func)
    return ""


def _is_true_literal(node: object) -> bool:
    return isinstance(node, Constant) and node.value is True


if __name__ == "__main__":
    main()
