from collections.abc import Awaitable, Callable
from unittest import IsolatedAsyncioTestCase, main

from avalan.tool.shell.registry import (
    SHELL_COMMAND_DEFINITIONS,
    ShellCommandDefinition,
)
from avalan.tool.shell.resolver import (
    TrustedExecutableResolver,
    unavailable_executable_lookup,
)
from avalan.tool.shell.toolset import ShellToolSet


class TrustedExecutableResolverTest(IsolatedAsyncioTestCase):
    async def test_explicit_path_resolves_without_lookup(self) -> None:
        calls: list[str] = []

        async def lookup(
            command: ShellCommandDefinition,
            search_paths: tuple[str, ...],
        ) -> str | None:
            calls.append(command.logical_id)
            return None

        resolver = TrustedExecutableResolver(
            executable_paths={"rg": "/usr/bin/rg"},
            lookup=lookup,
        )

        self.assertEqual(await resolver.resolve_command("rg"), "/usr/bin/rg")
        self.assertEqual(calls, [])

    async def test_missing_command_is_cached(self) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []

        async def lookup(
            command: ShellCommandDefinition,
            search_paths: tuple[str, ...],
        ) -> str | None:
            calls.append((command.logical_id, search_paths))
            return None

        resolver = TrustedExecutableResolver(
            executable_search_paths=("/usr/bin",),
            lookup=lookup,
        )

        self.assertIsNone(await resolver.resolve_command("rg"))
        self.assertIsNone(await resolver.resolve_command("rg"))
        self.assertEqual(calls, [("rg", ("/usr/bin",))])

    async def test_resolved_command_is_cached_until_clear(self) -> None:
        calls = 0

        async def lookup(
            command: ShellCommandDefinition,
            search_paths: tuple[str, ...],
        ) -> str | None:
            nonlocal calls
            calls += 1
            return f"/tools/{command.executable_name}"

        resolver = TrustedExecutableResolver(lookup=lookup)

        self.assertEqual(await resolver.resolve_command("jq"), "/tools/jq")
        self.assertEqual(await resolver.resolve_command("jq"), "/tools/jq")
        self.assertEqual(calls, 1)

        resolver.clear_cache()

        self.assertEqual(await resolver.resolve_command("jq"), "/tools/jq")
        self.assertEqual(calls, 2)

    async def test_default_lookup_reports_unavailable(self) -> None:
        self.assertIsNone(
            await unavailable_executable_lookup(
                SHELL_COMMAND_DEFINITIONS["rg"],
                (),
            )
        )

    async def test_rejects_invalid_configuration_and_inputs(self) -> None:
        invalid_kwargs = (
            {"executable_paths": []},
            {"executable_paths": {"rg": "relative"}},
            {"executable_paths": {"unknown": "/bin/unknown"}},
            {"executable_search_paths": "/bin"},
            {"executable_search_paths": ("relative",)},
            {"lookup": None},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    TrustedExecutableResolver(**kwargs)  # type: ignore[arg-type]

        resolver = TrustedExecutableResolver()
        with self.assertRaises(AssertionError):
            await resolver.resolve("rg")  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await resolver.resolve_command("")
        with self.assertRaises(AssertionError):
            await resolver.resolve_command("unknown")
        with self.assertRaises(AssertionError):
            await unavailable_executable_lookup("rg", ())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await unavailable_executable_lookup(
                SHELL_COMMAND_DEFINITIONS["rg"],
                ("relative",),
            )

    async def test_rejects_empty_lookup_result(self) -> None:
        async def lookup(
            command: ShellCommandDefinition,
            search_paths: tuple[str, ...],
        ) -> str | None:
            return ""

        resolver = TrustedExecutableResolver(lookup=lookup)

        with self.assertRaises(AssertionError):
            await resolver.resolve_command("rg")

    async def test_binary_availability_does_not_change_tool_schemas(
        self,
    ) -> None:
        missing_one = TrustedExecutableResolver()
        missing_all = TrustedExecutableResolver()
        resolved = TrustedExecutableResolver(
            lookup=_resolved_lookup("/usr/bin/"),
        )
        before = ShellToolSet().json_schemas()

        self.assertIsNone(await missing_one.resolve_command("rg"))
        for command_id in SHELL_COMMAND_DEFINITIONS:
            self.assertIsNone(await missing_all.resolve_command(command_id))
        self.assertEqual(await resolved.resolve_command("rg"), "/usr/bin/rg")

        self.assertEqual(ShellToolSet().json_schemas(), before)


def _resolved_lookup(
    prefix: str,
) -> Callable[
    [ShellCommandDefinition, tuple[str, ...]],
    Awaitable[str | None],
]:
    async def lookup(
        command: ShellCommandDefinition,
        search_paths: tuple[str, ...],
    ) -> str | None:
        return f"{prefix}{command.executable_name}"

    return lookup


if __name__ == "__main__":
    main()
