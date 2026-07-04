from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import (
    ToolCallContext,
    ToolManagerSettings,
    ToolNameResolutionStatus,
)
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    SHELL_GIT_COMMAND_IDS,
    SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS,
    ShellGitCommandName,
    ShellGitCommandRequest,
    ShellGitExecutionErrorCode,
    ShellGitExecutionStatus,
    ShellGitFormattedResult,
    ShellGitToolSettings,
    ShellToolSet,
    ShellToolSettings,
)
from avalan.tool.shell.tools import _ShellGitCommandTool

_READ_GIT_SCHEMA_NAMES = tuple(
    f"shell.git_{command_id.replace('-', '_')}"
    for command_id in SHELL_GIT_DEFAULT_ALLOWED_COMMAND_IDS
)
_REMOTE_MANAGEMENT_COMMAND_IDS = (
    ShellGitCommandName.REMOTE_LIST.value,
    ShellGitCommandName.REMOTE_ADD.value,
    ShellGitCommandName.REMOTE_SET_URL.value,
    ShellGitCommandName.REMOTE_REMOVE.value,
    ShellGitCommandName.REMOTE_RENAME.value,
)
_FORBIDDEN_SCHEMA_FIELD_NAMES = {
    "alias",
    "aliases",
    "args",
    "argv",
    "flags",
    "options",
    "shell",
    "subcommand",
}


class ShellGitToolSetPhase1Test(TestCase):
    def test_concrete_git_enablement_exposes_only_status(self) -> None:
        toolset = ShellToolSet().with_enabled_tools(["shell.git_status"])

        self.assertEqual(_schema_names(toolset), ("shell.git_status",))

    def test_shell_namespace_exposes_read_only_git_tools(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(capabilities=("read",))
        )

        for enable_tools in (["shell"], ["shell.*"]):
            with self.subTest(enable_tools=enable_tools):
                toolset = ShellToolSet(settings=settings).with_enabled_tools(
                    enable_tools
                )

                self.assertEqual(
                    _git_schema_names(toolset),
                    _READ_GIT_SCHEMA_NAMES,
                )

    def test_git_tools_absent_without_explicit_enablement(self) -> None:
        toolset = ShellToolSet()

        self.assertEqual(_git_schema_names(toolset), ())

    def test_git_tools_absent_when_not_allowed_by_settings(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(allowed_commands=())
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.git_status"]
        )

        self.assertEqual(_schema_names(toolset), ())

    def test_read_only_settings_hide_mutating_and_remote_tools(self) -> None:
        toolset = ShellToolSet().with_enabled_tools(["shell.*"])
        names = set(_schema_names(toolset))

        self.assertNotIn("shell.git_add", names)
        self.assertNotIn("shell.git_commit", names)
        self.assertNotIn("shell.git_fetch", names)
        self.assertEqual(_git_schema_names(toolset), _READ_GIT_SCHEMA_NAMES)

    def test_wildcard_enablement_obeys_git_allowed_commands(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(allowed_commands=("status",))
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )

        self.assertEqual(_git_schema_names(toolset), ("shell.git_status",))

    def test_capabilities_participate_in_git_exposure(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("worktree",),
                allowed_commands=("status", "add"),
            )
        )
        toolset = ShellToolSet(settings=settings).with_enabled_tools(
            ["shell.*"]
        )

        self.assertIn("shell.git_add", _schema_names(toolset))
        self.assertNotIn("shell.git_status", _schema_names(toolset))

    def test_remote_tools_require_remote_policy_allowlist(self) -> None:
        denied = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=("fetch",),
            )
        )
        allowed = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=("fetch",),
                allowed_remote_hosts=("github.com",),
            )
        )
        allowed_management = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=_REMOTE_MANAGEMENT_COMMAND_IDS,
                allowed_remote_hosts=("github.com",),
            )
        )

        self.assertNotIn(
            "shell.git_fetch",
            _schema_names(
                ShellToolSet(settings=denied).with_enabled_tools(["shell.*"])
            ),
        )
        self.assertIn(
            "shell.git_fetch",
            _schema_names(
                ShellToolSet(settings=allowed).with_enabled_tools(["shell.*"])
            ),
        )
        management_names = set(
            _schema_names(
                ShellToolSet(settings=allowed_management).with_enabled_tools(
                    ["shell.*"]
                )
            )
        )
        for command_id in _REMOTE_MANAGEMENT_COMMAND_IDS:
            with self.subTest(command_id=command_id):
                self.assertIn(
                    f"shell.git_{command_id.replace('-', '_')}",
                    management_names,
                )
        self.assertNotIn("shell.git_fetch", management_names)

    def test_submodule_update_requires_explicit_submodule_setting(
        self,
    ) -> None:
        denied = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=("submodule-update",),
                allowed_remote_hosts=("github.com",),
            )
        )
        allowed = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("remote",),
                allowed_commands=("submodule-update",),
                allowed_remote_hosts=("github.com",),
                allow_submodule_update=True,
            )
        )

        self.assertNotIn(
            "shell.git_submodule_update",
            _schema_names(
                ShellToolSet(settings=denied).with_enabled_tools(["shell.*"])
            ),
        )
        self.assertIn(
            "shell.git_submodule_update",
            _schema_names(
                ShellToolSet(settings=allowed).with_enabled_tools(["shell.*"])
            ),
        )


class ShellGitSettingsPhase1Test(TestCase):
    def test_remote_enabled_spec_example_is_accepted(self) -> None:
        settings = ShellToolSettings(
            git={
                "capabilities": ["read", "remote"],
                "allowed_commands": [
                    "status",
                    "fetch",
                    "push",
                    *_REMOTE_MANAGEMENT_COMMAND_IDS,
                ],
                "allowed_remote_protocols": ["https"],
                "allowed_remote_hosts": ["github.com"],
                "allow_remote_credentials": True,
                "redact_remote_urls": True,
            }
        )

        git_settings = settings.git
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(
            git_settings.allowed_commands,
            (
                ShellGitCommandName.STATUS.value,
                ShellGitCommandName.FETCH.value,
                ShellGitCommandName.PUSH.value,
                *_REMOTE_MANAGEMENT_COMMAND_IDS,
            ),
        )
        self.assertEqual(git_settings.credential_policy, "allow_explicit")
        self.assertTrue(git_settings.allow_remote_credentials)

    def test_remote_capability_is_not_allowed_command_alias(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "git.allowed_commands contains unsupported value: 'remote'",
        ):
            ShellGitToolSettings(allowed_commands=("remote",))

    def test_git_settings_accept_mapping_and_copy_sequences(self) -> None:
        capabilities = ["read"]
        allowed_commands = ["status"]
        allowed_hosts = ["GitHub.com"]
        settings = ShellToolSettings(
            git={
                "capabilities": capabilities,
                "allowed_commands": allowed_commands,
                "allowed_remote_hosts": allowed_hosts,
            }
        )

        capabilities.append("remote")
        allowed_commands.append("fetch")
        allowed_hosts.append("example.com")

        git_settings = settings.git
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.capabilities, ("read",))
        self.assertEqual(git_settings.allowed_commands, ("status",))
        self.assertEqual(git_settings.allowed_remote_hosts, ("github.com",))

    def test_invalid_git_settings_fail_with_stable_diagnostics(self) -> None:
        cases = (
            (
                {"capabilities": ("write",)},
                "git.capabilities contains unsupported value",
            ),
            (
                {"allowed_commands": ("init",)},
                "git.allowed_commands contains unsupported value",
            ),
            ({"max_diff_bytes": 0}, "git.max_diff_bytes must be positive"),
            (
                {"allowed_remote_hosts": ("*.github.com",)},
                "git.allowed_remote_hosts contains unsafe value",
            ),
            (
                {"credential_policy": "askpass"},
                "git.credential_policy must be deny or allow_explicit",
            ),
        )

        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(AssertionError, message):
                    ShellGitToolSettings(**kwargs)


class ShellGitToolManagerPhase1Test(TestCase):
    def test_sdk_read_only_profile_exposes_selected_git_tools(self) -> None:
        manager = ToolManager.create_instance(
            available_toolsets=[
                ShellToolSet(
                    settings=ShellToolSettings(
                        git=ShellGitToolSettings(
                            capabilities=("read",),
                            allowed_commands=("status", "diff", "log"),
                        )
                    )
                )
            ],
            enable_tools=[
                "shell.git_status",
                "shell.git_diff",
                "shell.git_log",
            ],
            settings=ToolManagerSettings(),
        )

        self.assertIsNotNone(manager.describe_tool("shell.git_status"))
        self.assertIsNotNone(manager.describe_tool("shell.git_diff"))
        self.assertIsNotNone(manager.describe_tool("shell.git_log"))
        self.assertIsNone(manager.describe_tool("shell.git_commit"))

    def test_tool_manager_provider_name_round_trips_git_tool(self) -> None:
        manager = ToolManager.create_instance(
            available_toolsets=[ShellToolSet()],
            enable_tools=["shell.git_status"],
            settings=ToolManagerSettings(),
        )

        self.assertEqual(_manager_tool_names(manager), ("shell.git_status",))
        provider_name = manager.provider_tool_name("shell.git_status")

        self.assertNotEqual(provider_name, "shell.git_status")
        self.assertEqual(
            manager.canonical_tool_name(provider_name),
            "shell.git_status",
        )
        self.assertEqual(
            _manager_schema_names(manager),
            ("shell.git_status",),
        )

    def test_shell_rg_enablement_does_not_advertise_git_status(self) -> None:
        manager = ToolManager.create_instance(
            available_toolsets=[ShellToolSet()],
            enable_tools=["shell.rg"],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("shell.git_status")

        self.assertEqual(_manager_tool_names(manager), ("shell.rg",))
        self.assertIs(resolution.status, ToolNameResolutionStatus.UNKNOWN)

    def test_explicit_disallowed_git_tool_resolves_disabled(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(allowed_commands=())
        )
        manager = ToolManager.create_instance(
            available_toolsets=[ShellToolSet(settings=settings)],
            enable_tools=["shell.git_status"],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("shell.git_status")

        self.assertEqual(_manager_tool_names(manager), ())
        self.assertIs(resolution.status, ToolNameResolutionStatus.DISABLED)

    def test_git_schemas_do_not_expose_raw_passthrough_fields(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("read", "worktree", "history", "remote"),
                allowed_commands=SHELL_GIT_COMMAND_IDS,
                allowed_remote_hosts=("github.com",),
                allow_submodule_update=True,
            )
        )
        manager = ToolManager.create_instance(
            available_toolsets=[ShellToolSet(settings=settings)],
            enable_tools=["shell.*"],
            settings=ToolManagerSettings(),
        )
        schemas = [
            schema
            for schema in manager.json_schemas() or ()
            if schema["function"]["name"].startswith("shell.git_")
        ]

        self.assertTrue(schemas)
        for schema in schemas:
            with self.subTest(name=schema["function"]["name"]):
                field_names = _schema_field_names(
                    schema["function"]["parameters"]
                )
                self.assertFalse(field_names & _FORBIDDEN_SCHEMA_FIELD_NAMES)

        status_schema = manager.describe_tool("shell.git_status")
        assert status_schema is not None
        assert status_schema.parameter_schema is not None
        self.assertEqual(
            status_schema.parameter_schema["properties"]["mode"]["enum"],
            ["porcelain_v2", "short"],
        )
        required_path_schemas = ("shell.git_blame", "shell.git_grep")
        for tool_name in required_path_schemas:
            with self.subTest(required_paths=tool_name):
                tool_schema = manager.describe_tool(tool_name)
                assert tool_schema is not None
                assert tool_schema.parameter_schema is not None
                self.assertIn(
                    "paths" if tool_name != "shell.git_blame" else "path",
                    tool_schema.parameter_schema["required"],
                )

        show_schema = manager.describe_tool("shell.git_show")
        assert show_schema is not None
        assert show_schema.parameter_schema is not None
        self.assertNotIn("paths", show_schema.parameter_schema["required"])

        for tool_name in ("shell.git_diff", "shell.git_stash_show"):
            with self.subTest(optional_paths=tool_name):
                tool_schema = manager.describe_tool(tool_name)
                assert tool_schema is not None
                assert tool_schema.parameter_schema is not None
                self.assertNotIn(
                    "paths",
                    tool_schema.parameter_schema["required"],
                )

    def test_worktree_reset_schema_excludes_history_modes(self) -> None:
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                capabilities=("worktree",),
                allowed_commands=("reset",),
            )
        )
        manager = ToolManager.create_instance(
            available_toolsets=[ShellToolSet(settings=settings)],
            enable_tools=["shell.git_reset"],
            settings=ToolManagerSettings(),
        )

        reset_schema = manager.describe_tool("shell.git_reset")
        assert reset_schema is not None
        assert reset_schema.parameter_schema is not None
        properties = reset_schema.parameter_schema["properties"]

        self.assertNotIn("mode", properties)
        self.assertNotIn("revision", properties)
        self.assertNotIn("mixed", dumps(reset_schema.parameter_schema))


class ShellGitWrapperPhase1Test(IsolatedAsyncioTestCase):
    async def test_git_wrapper_returns_stable_result_without_git_lookup(
        self,
    ) -> None:
        with TemporaryDirectory() as workspace:
            _write_minimal_git_repo(Path(workspace) / "repo")
            settings = ShellToolSettings(
                git=ShellGitToolSettings(
                    workspace_root=workspace,
                    cwd="repo",
                )
            )
            toolset = ShellToolSet(settings=settings).with_enabled_tools(
                ["shell.git_status"]
            )
            tool = cast(Any, toolset.tools[0])

            result = await tool(context=ToolCallContext())

        self.assertIsInstance(result, ShellGitFormattedResult)
        assert isinstance(result, ShellGitFormattedResult)
        self.assertEqual(
            result.git_result.status,
            ShellGitExecutionStatus.COMMAND_UNAVAILABLE,
        )
        self.assertEqual(
            result.git_result.error_code,
            ShellGitExecutionErrorCode.COMMAND_UNAVAILABLE,
        )
        self.assertIn("tool: shell.git_status", result)
        self.assertIn("error_code: command_unavailable", result)

    def test_git_wrapper_builds_typed_request(self) -> None:
        toolset = ShellToolSet().with_enabled_tools(["shell.git_status"])
        tool = cast(Any, toolset.tools[0])
        request = tool._build_request(paths=("src",))

        self.assertIsInstance(request, ShellGitCommandRequest)
        self.assertEqual(request.command, ShellGitCommandName.STATUS)
        self.assertEqual(request.pathspecs, ("src",))

    async def test_all_git_wrappers_build_requests_and_return_stable_results(
        self,
    ) -> None:
        tools = _all_git_tools_by_name()

        self.assertEqual(set(tools), set(_GIT_TOOL_ARGUMENTS))
        for name, kwargs in _GIT_TOOL_ARGUMENTS.items():
            with self.subTest(name=name):
                tool = cast(Any, tools[name])
                request = tool._build_request(**kwargs)
                result = await tool(**kwargs, context=ToolCallContext())

                self.assertIsInstance(request, ShellGitCommandRequest)
                self.assertIsInstance(result, ShellGitFormattedResult)
                assert isinstance(result, ShellGitFormattedResult)
                self.assertEqual(request.tool_name, f"shell.{name}")
                self.assertEqual(request.cwd, "repo")
                self.assertEqual(request.timeout_seconds, 1.0)
                self.assertEqual(request.max_stdout_bytes, 32)
                self.assertEqual(request.max_stderr_bytes, 33)
                self.assertEqual(
                    result.git_result.tool_name,
                    request.tool_name,
                )
                self.assertEqual(result.git_result.command, request.command)
                self.assertEqual(
                    result.git_result.status,
                    ShellGitExecutionStatus.POLICY_DENIED,
                )
                expected_error_code = ShellGitExecutionErrorCode.REPO_NOT_FOUND
                self.assertEqual(
                    result.git_result.error_code,
                    expected_error_code,
                )
                self.assertIn("execution_mode: policy", result)
                self.assertIn("status: policy_denied", result)

    async def test_base_git_wrapper_call_is_abstract(self) -> None:
        tool = _BaseGitCallProbeTool(settings=ShellToolSettings())

        with self.assertRaises(NotImplementedError):
            await tool(context=ToolCallContext())


class _BaseGitCallProbeTool(_ShellGitCommandTool):
    def __init__(self, *, settings: ShellToolSettings) -> None:
        super().__init__(command=ShellGitCommandName.STATUS, settings=settings)

    async def __call__(self, *, context: ToolCallContext) -> str:
        base_call = cast(Any, _ShellGitCommandTool.__call__)
        return cast(str, await base_call(self, context=context))


def _schema_names(toolset: ShellToolSet) -> tuple[str, ...]:
    schemas = toolset.json_schemas()
    return tuple(schema["function"]["name"] for schema in schemas or ())


def _git_schema_names(toolset: ShellToolSet) -> tuple[str, ...]:
    return tuple(
        name
        for name in _schema_names(toolset)
        if name.startswith("shell.git_")
    )


def _manager_tool_names(manager: ToolManager) -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in manager.list_tools())


def _manager_schema_names(manager: ToolManager) -> tuple[str, ...]:
    return tuple(
        schema["function"]["name"] for schema in manager.json_schemas() or ()
    )


def _schema_field_names(schema: object) -> set[str]:
    names: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            names.update(str(name) for name in properties)
        for value in schema.values():
            names.update(_schema_field_names(value))
    elif isinstance(schema, list):
        for item in schema:
            names.update(_schema_field_names(item))
    return names


def _all_git_tools_by_name() -> dict[str, object]:
    settings = ShellToolSettings(
        git=ShellGitToolSettings(
            capabilities=("read", "worktree", "history", "remote"),
            allowed_commands=SHELL_GIT_COMMAND_IDS,
            allowed_remote_hosts=("github.com",),
            allow_submodule_update=True,
        )
    )
    toolset = ShellToolSet(settings=settings).with_enabled_tools(["shell.*"])
    return {
        getattr(tool, "__name__"): tool
        for tool in toolset.tools
        if getattr(tool, "__name__", "").startswith("git_")
    }


def _git_kwargs(**kwargs: object) -> dict[str, object]:
    return {
        **kwargs,
        "cwd": "repo",
        "timeout_seconds": 1.0,
        "max_stdout_bytes": 32,
        "max_stderr_bytes": 33,
    }


def _write_minimal_git_repo(repo: Path) -> None:
    git_dir = repo / ".git"
    (git_dir / "objects" / "info").mkdir(parents=True)
    (git_dir / "refs").mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
    )


_GIT_TOOL_ARGUMENTS: dict[str, dict[str, object]] = {
    "git_status": _git_kwargs(
        mode="short",
        paths=("src",),
        include_branch=False,
    ),
    "git_rev_parse": _git_kwargs(fact="repo_root"),
    "git_branch": _git_kwargs(mode="list", contains="HEAD"),
    "git_tag": _git_kwargs(mode="show", name="v1.0", max_count=2),
    "git_describe": _git_kwargs(
        target="HEAD",
        mode="always",
        max_candidates=3,
    ),
    "git_ls_files": _git_kwargs(mode="others", paths=("src",)),
    "git_log": _git_kwargs(
        max_count=2,
        revision="HEAD~2..HEAD",
        paths=("src",),
        format="oneline",
    ),
    "git_diff": _git_kwargs(
        mode="range",
        base_revision="HEAD~1",
        head_revision="HEAD",
        paths=("src",),
    ),
    "git_show": _git_kwargs(
        revision="HEAD",
        mode="patch",
        paths=("src/app.py",),
    ),
    "git_blame": _git_kwargs(
        path="src/app.py",
        start_line=1,
        end_line=2,
    ),
    "git_grep": _git_kwargs(
        pattern="needle",
        paths=("src",),
        case="insensitive",
        max_matches=3,
    ),
    "git_stash_list": _git_kwargs(max_count=2),
    "git_stash_show": _git_kwargs(
        stash="stash@{1}",
        mode="patch",
        paths=("src/app.py",),
    ),
    "git_add": _git_kwargs(paths=("src/app.py",), mode="intent_to_add"),
    "git_restore": _git_kwargs(
        paths=("src/app.py",),
        source_revision="HEAD",
        staged=True,
        worktree=False,
    ),
    "git_checkout": _git_kwargs(
        target="feature",
        paths=("src/app.py",),
    ),
    "git_switch": _git_kwargs(branch="feature"),
    "git_reset": _git_kwargs(paths=("src/app.py",)),
    "git_rm": _git_kwargs(
        paths=("src/app.py",),
        cached=True,
    ),
    "git_mv": _git_kwargs(source="old.py", destination="new.py"),
    "git_stash_push": _git_kwargs(
        message="save",
        paths=("src/app.py",),
        include_untracked=True,
    ),
    "git_stash_apply": _git_kwargs(
        stash="stash@{1}",
        paths=("src/app.py",),
    ),
    "git_commit": _git_kwargs(message="message"),
    "git_branch_create": _git_kwargs(name="feature", start_point="HEAD"),
    "git_branch_delete": _git_kwargs(name="old", confirm_name="old"),
    "git_branch_rename": _git_kwargs(
        old_name="old",
        new_name="new",
        confirm_old_name="old",
    ),
    "git_tag_create": _git_kwargs(
        name="v1.0",
        target="HEAD",
        message="tag",
    ),
    "git_tag_delete": _git_kwargs(name="v0.1", confirm_name="v0.1"),
    "git_merge": _git_kwargs(
        revision="feature",
        confirm_revision="feature",
        mode="no_ff",
    ),
    "git_rebase": _git_kwargs(
        upstream="main",
        confirm_upstream="main",
        branch="feature",
    ),
    "git_cherry_pick": _git_kwargs(
        revision="abc123",
        confirm_revision="abc123",
    ),
    "git_revert": _git_kwargs(
        revision="abc123",
        confirm_revision="abc123",
    ),
    "git_clean": _git_kwargs(
        paths=("build.txt",),
        dry_run=False,
        confirm_paths=("build.txt",),
    ),
    "git_stash_pop": _git_kwargs(
        stash="stash@{1}",
        confirm_stash="stash@{1}",
        index=True,
    ),
    "git_stash_drop": _git_kwargs(
        stash="stash@{1}",
        confirm_stash="stash@{1}",
    ),
    "git_fetch": _git_kwargs(
        remote="origin",
        ref_type="branch",
        ref_name="main",
    ),
    "git_pull": _git_kwargs(remote="origin", branch="main"),
    "git_push": _git_kwargs(
        remote="origin",
        ref_type="branch",
        ref_name="main",
    ),
    "git_clone": _git_kwargs(
        url="https://github.com/example/repo.git",
        destination="repo-copy",
    ),
    "git_remote_list": _git_kwargs(),
    "git_remote_add": _git_kwargs(
        name="origin",
        url="https://github.com/example/repo.git",
    ),
    "git_remote_set_url": _git_kwargs(
        name="origin",
        url="https://github.com/example/repo.git",
    ),
    "git_remote_remove": _git_kwargs(name="origin"),
    "git_remote_rename": _git_kwargs(
        old_name="origin",
        new_name="upstream",
    ),
    "git_submodule_update": _git_kwargs(
        paths=("vendor/lib",),
        init=True,
    ),
}


if __name__ == "__main__":
    main()
