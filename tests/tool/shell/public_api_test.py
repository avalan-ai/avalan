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
    SHELL_GIT_CAPABILITY_IDS,
    SHELL_GIT_COMMAND_CAPABILITIES,
    SHELL_GIT_COMMAND_IDS,
    SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS,
    SHELL_GIT_TOOL_COMMANDS,
    SHELL_GIT_TOOL_NAMES,
    SHELL_STATUS_ERROR_CODES,
    BackendBoundaryCompositionExecutor,
    CommandExecutor,
    CompositionExecutor,
    ExecutableLookup,
    ExecutableResolver,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    GeneratedOutputPlan,
    KillTool,
    LocalCommandExecutor,
    LocalCompositionExecutor,
    PathOperand,
    PsTool,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionMode,
    ShellCompositionRequest,
    ShellCompositionResult,
    ShellCompositionSpec,
    ShellDependencyGroup,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellExecutionStepSpec,
    ShellFormattedCompositionResult,
    ShellFormattedResult,
    ShellGitCapability,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitCommandResult,
    ShellGitExecutionErrorCode,
    ShellGitExecutionStatus,
    ShellGitFormattedResult,
    ShellGitToolSettings,
    ShellOutputKind,
    ShellPathMetadata,
    ShellPolicyDenied,
    ShellSandboxCommandExecutor,
    ShellStreamRef,
    ShellToolError,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
    enables_shell_tools,
    ensure_file_size_at_most,
    file_size,
    inspect_path,
    normalize_shell_enabled_tools,
    private_temp_directory,
    probe_image_dimensions,
    read_image_signature,
    read_pdf_signature,
    read_signature,
    resolve_policy_path,
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
        self.assertIs(KillTool, import_module("avalan.tool.shell").KillTool)
        self.assertIs(PsTool, import_module("avalan.tool.shell").PsTool)
        self.assertIs(
            ExecutionPolicy, import_module("avalan.tool.shell").ExecutionPolicy
        )
        self.assertIs(
            LocalCommandExecutor,
            import_module("avalan.tool.shell").LocalCommandExecutor,
        )
        self.assertIs(
            CompositionExecutor,
            import_module("avalan.tool.shell").CompositionExecutor,
        )
        self.assertIs(
            LocalCompositionExecutor,
            import_module("avalan.tool.shell").LocalCompositionExecutor,
        )
        self.assertIs(
            BackendBoundaryCompositionExecutor,
            import_module(
                "avalan.tool.shell"
            ).BackendBoundaryCompositionExecutor,
        )
        self.assertIs(
            ShellSandboxCommandExecutor,
            import_module("avalan.tool.shell").ShellSandboxCommandExecutor,
        )
        self.assertIs(
            ShellPathMetadata,
            import_module("avalan.tool.shell").ShellPathMetadata,
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
            resolve_policy_path,
            import_module("avalan.tool.shell").resolve_policy_path,
        )
        self.assertIs(
            inspect_path,
            import_module("avalan.tool.shell").inspect_path,
        )
        self.assertIs(file_size, import_module("avalan.tool.shell").file_size)
        self.assertIs(
            ensure_file_size_at_most,
            import_module("avalan.tool.shell").ensure_file_size_at_most,
        )
        self.assertIs(
            read_signature,
            import_module("avalan.tool.shell").read_signature,
        )
        self.assertIs(
            read_pdf_signature,
            import_module("avalan.tool.shell").read_pdf_signature,
        )
        self.assertIs(
            read_image_signature,
            import_module("avalan.tool.shell").read_image_signature,
        )
        self.assertIs(
            probe_image_dimensions,
            import_module("avalan.tool.shell").probe_image_dimensions,
        )
        self.assertIs(
            private_temp_directory,
            import_module("avalan.tool.shell").private_temp_directory,
        )
        self.assertIs(
            PathOperand, import_module("avalan.tool.shell").PathOperand
        )
        self.assertIs(
            ShellCommandRequest,
            import_module("avalan.tool.shell").ShellCommandRequest,
        )
        self.assertIs(
            ShellCommandStepRequest,
            import_module("avalan.tool.shell").ShellCommandStepRequest,
        )
        self.assertIs(
            ShellCompositionMode,
            import_module("avalan.tool.shell").ShellCompositionMode,
        )
        self.assertIs(
            ShellCompositionRequest,
            import_module("avalan.tool.shell").ShellCompositionRequest,
        )
        self.assertIs(
            ShellCompositionResult,
            import_module("avalan.tool.shell").ShellCompositionResult,
        )
        self.assertIs(
            ShellCompositionSpec,
            import_module("avalan.tool.shell").ShellCompositionSpec,
        )
        self.assertIs(
            ExecutionSpec, import_module("avalan.tool.shell").ExecutionSpec
        )
        self.assertIs(
            ExecutionResult,
            import_module("avalan.tool.shell").ExecutionResult,
        )
        self.assertIs(
            ShellPolicyDenied,
            import_module("avalan.tool.shell").ShellPolicyDenied,
        )
        self.assertIs(
            ShellToolError,
            import_module("avalan.tool.shell").ShellToolError,
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
            ShellExecutionStepResult,
            import_module("avalan.tool.shell").ShellExecutionStepResult,
        )
        self.assertIs(
            ShellExecutionStepSpec,
            import_module("avalan.tool.shell").ShellExecutionStepSpec,
        )
        self.assertIs(
            ShellOutputKind, import_module("avalan.tool.shell").ShellOutputKind
        )
        self.assertIs(
            ShellStreamRef,
            import_module("avalan.tool.shell").ShellStreamRef,
        )
        self.assertIs(
            ShellFormattedResult,
            import_module("avalan.tool.shell").ShellFormattedResult,
        )
        self.assertIs(
            ShellFormattedCompositionResult,
            import_module("avalan.tool.shell").ShellFormattedCompositionResult,
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
            SHELL_STATUS_ERROR_CODES,
            import_module("avalan.tool.shell").SHELL_STATUS_ERROR_CODES,
        )
        self.assertIs(
            SHELL_GIT_CAPABILITY_IDS,
            import_module("avalan.tool.shell").SHELL_GIT_CAPABILITY_IDS,
        )
        self.assertIs(
            SHELL_GIT_COMMAND_CAPABILITIES,
            import_module("avalan.tool.shell").SHELL_GIT_COMMAND_CAPABILITIES,
        )
        self.assertIs(
            SHELL_GIT_COMMAND_IDS,
            import_module("avalan.tool.shell").SHELL_GIT_COMMAND_IDS,
        )
        self.assertIs(
            SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS,
            import_module(
                "avalan.tool.shell"
            ).SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS,
        )
        self.assertIs(
            SHELL_GIT_TOOL_COMMANDS,
            import_module("avalan.tool.shell").SHELL_GIT_TOOL_COMMANDS,
        )
        self.assertIs(
            SHELL_GIT_TOOL_NAMES,
            import_module("avalan.tool.shell").SHELL_GIT_TOOL_NAMES,
        )
        self.assertIs(
            ShellGitCapability,
            import_module("avalan.tool.shell").ShellGitCapability,
        )
        self.assertIs(
            ShellGitCommandName,
            import_module("avalan.tool.shell").ShellGitCommandName,
        )
        self.assertIs(
            ShellGitCommandRequest,
            import_module("avalan.tool.shell").ShellGitCommandRequest,
        )
        self.assertIs(
            ShellGitCommandResult,
            import_module("avalan.tool.shell").ShellGitCommandResult,
        )
        self.assertIs(
            ShellGitExecutionErrorCode,
            import_module("avalan.tool.shell").ShellGitExecutionErrorCode,
        )
        self.assertIs(
            ShellGitExecutionStatus,
            import_module("avalan.tool.shell").ShellGitExecutionStatus,
        )
        self.assertIs(
            ShellGitFormattedResult,
            import_module("avalan.tool.shell").ShellGitFormattedResult,
        )
        self.assertIs(
            ShellGitToolSettings,
            import_module("avalan.tool.shell").ShellGitToolSettings,
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
            "avalan.tool.shell.commands",
            "avalan.tool.shell.commands.awk",
            "avalan.tool.shell.commands.base",
            "avalan.tool.shell.commands.cat",
            "avalan.tool.shell.commands.file",
            "avalan.tool.shell.commands.find",
            "avalan.tool.shell.commands.head",
            "avalan.tool.shell.commands.helpers",
            "avalan.tool.shell.commands.jq",
            "avalan.tool.shell.commands.kill",
            "avalan.tool.shell.commands.ls",
            "avalan.tool.shell.commands.nl",
            "avalan.tool.shell.commands.pdfinfo",
            "avalan.tool.shell.commands.pgrep",
            "avalan.tool.shell.commands.ps",
            "avalan.tool.shell.commands.pdfplumber",
            "avalan.tool.shell.commands.pdftoppm",
            "avalan.tool.shell.commands.pdftotext",
            "avalan.tool.shell.commands.pypdf",
            "avalan.tool.shell.commands.python_pdf",
            "avalan.tool.shell.commands.rg",
            "avalan.tool.shell.commands.reportlab",
            "avalan.tool.shell.commands.sed",
            "avalan.tool.shell.commands.tail",
            "avalan.tool.shell.commands.tesseract",
            "avalan.tool.shell.commands.wc",
            "avalan.tool.shell.composition_executor",
            "avalan.tool.shell.entities",
            "avalan.tool.shell.executor",
            "avalan.tool.shell.filesystem",
            "avalan.tool.shell.formatting",
            "avalan.tool.shell.git",
            "avalan.tool.shell.kill",
            "avalan.tool.shell.opt_in",
            "avalan.tool.shell.pgrep",
            "avalan.tool.shell.ps",
            "avalan.tool.shell.policy",
            "avalan.tool.shell.python_pdf",
            "avalan.tool.shell.registry",
            "avalan.tool.shell.resolver",
            "avalan.tool.shell.sandbox",
            "avalan.tool.shell.settings",
            "avalan.tool.shell.tools",
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

    def test_ps_view_appends_to_legacy_positional_contract(self) -> None:
        parameters = signature(PsTool.__call__).parameters

        self.assertEqual(
            tuple(parameters)[:7],
            (
                "self",
                "pids",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
                "view",
            ),
        )
        self.assertEqual(parameters["view"].default, "summary")

    def test_shell_toolset_is_namespaced_with_available_tools(
        self,
    ) -> None:
        toolset = ShellToolSet()

        self.assertIsInstance(toolset, ToolSet)
        self.assertEqual(toolset.namespace, "shell")
        self.assertEqual(
            [tool.__name__ for tool in toolset.tools],
            list(SHELL_COMMAND_IDS),
        )
        self.assertEqual(
            [schema["function"]["name"] for schema in toolset.json_schemas()],
            [f"shell.{command_id}" for command_id in SHELL_COMMAND_IDS],
        )

    def test_shell_pipeline_is_not_registered_by_default(self) -> None:
        toolset = ShellToolSet()
        tool_names = tuple(
            getattr(tool, "__name__", "") for tool in toolset.tools
        )
        schema_names = tuple(
            schema["function"]["name"] for schema in toolset.json_schemas()
        )

        self.assertNotIn("pipeline", SHELL_COMMAND_IDS)
        self.assertNotIn("pipeline", SHELL_COMMAND_DEFINITIONS)
        self.assertNotIn("pipeline", tool_names)
        self.assertNotIn("shell.pipeline", schema_names)

    def test_shell_toolset_accepts_explicit_settings(self) -> None:
        settings = ShellToolSettings()
        toolset = ShellToolSet(settings=settings)

        self.assertEqual(toolset.namespace, "shell")
        self.assertIs(toolset._settings, settings)

    def test_policy_denied_preserves_stable_error_code(self) -> None:
        error = ShellPolicyDenied(
            ShellExecutionErrorCode.DENIED_COMMAND,
            "denied",
        )

        self.assertEqual(
            error.error_code, ShellExecutionErrorCode.DENIED_COMMAND
        )
        self.assertEqual(str(error), "denied")

    def test_formatter_rejects_non_result_objects(self) -> None:
        with self.assertRaises(AssertionError):
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

    async def test_composition_executor_protocol_stub_is_inert(self) -> None:
        class InertCompositionExecutor(CompositionExecutor):
            pass

        executor = InertCompositionExecutor()

        with self.assertRaises(NotImplementedError):
            await executor.execute_composition(object())

    async def test_executable_resolver_protocol_stub_is_inert(self) -> None:
        class InertExecutableResolver(ExecutableResolver):
            pass

        resolver = InertExecutableResolver()

        with self.assertRaises(NotImplementedError):
            await resolver.resolve(SHELL_COMMAND_DEFINITIONS["rg"])

    async def test_policy_normalize_is_async_and_returns_spec(self) -> None:
        policy = ExecutionPolicy()
        request = ShellCommandRequest(
            tool_name="shell.rg",
            command="rg",
            options={"pattern": "needle"},
            paths=(),
            cwd=None,
        )

        spec = await policy.normalize(request)

        self.assertIsInstance(spec, ExecutionSpec)
        self.assertEqual(spec.tool_name, "shell.rg")
        self.assertEqual(spec.command, "rg")
        self.assertIsNone(spec.executable)

    async def test_policy_denies_media_commands_by_default(self) -> None:
        policy = ExecutionPolicy()
        request = ShellCommandRequest(
            tool_name="shell.pdftotext",
            command="pdftotext",
            options={},
            paths=(),
            cwd=None,
        )

        with self.assertRaises(ShellPolicyDenied) as context:
            await policy.normalize(request)
        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )

    async def test_policy_denies_unknown_commands(self) -> None:
        policy = ExecutionPolicy()
        request = ShellCommandRequest(
            tool_name="shell.unknown",
            command="unknown",
            options={},
            paths=(),
            cwd=None,
        )

        with self.assertRaises(ShellPolicyDenied) as context:
            await policy.normalize(request)
        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )

    async def test_policy_denies_commands_outside_settings_allowlist(
        self,
    ) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(allowed_commands=("rg",))
        )
        request = ShellCommandRequest(
            tool_name="shell.cat",
            command="cat",
            options={},
            paths=(),
            cwd=None,
        )

        with self.assertRaises(ShellPolicyDenied) as context:
            await policy.normalize(request)
        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )

    async def test_policy_allowlist_still_applies_to_enabled_media_tools(
        self,
    ) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                allow_media_tools=True,
                allowed_commands=("rg",),
            )
        )
        request = ShellCommandRequest(
            tool_name="shell.pdftotext",
            command="pdftotext",
            options={},
            paths=(),
            cwd=None,
        )

        with self.assertRaises(ShellPolicyDenied) as context:
            await policy.normalize(request)
        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )

    async def test_policy_denies_write_path_requests(self) -> None:
        policy = ExecutionPolicy(
            settings=ShellToolSettings(allow_media_tools=True)
        )
        request = ShellCommandRequest(
            tool_name="shell.rg",
            command="rg",
            options={"pattern": "needle"},
            paths=(
                PathOperand(
                    name="output",
                    path="out.txt",
                    kind="file",
                    access="write",
                ),
            ),
            cwd=None,
        )

        with self.assertRaises(ShellPolicyDenied) as context:
            await policy.normalize(request)
        self.assertEqual(
            context.exception.error_code,
            ShellExecutionErrorCode.WRITE_DENIED,
        )

    async def test_policy_allows_enabled_media_commands_to_normalize(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        policy = ExecutionPolicy(
            settings=ShellToolSettings(
                workspace_root=str(fixture_root),
                allow_media_tools=True,
            )
        )
        request = ShellCommandRequest(
            tool_name="shell.pdftotext",
            command="pdftotext",
            options={},
            paths=(
                PathOperand(
                    name="input",
                    path="media/small.pdf",
                    kind="pdf_file",
                    access="read",
                ),
            ),
            cwd=None,
        )

        spec = await policy.normalize(request)

        self.assertEqual(spec.command, "pdftotext")
        self.assertEqual(spec.resource_class, "heavy")

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
    return tuple(sorted(shell_root.rglob("*.py")))


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
