"""Prove portable continuation imports need no optional service stack."""

from subprocess import run
from sys import executable
from textwrap import dedent


def test_portable_continuation_cold_import_has_no_service_dependencies() -> (
    None
):
    code = dedent("""
        import importlib.abc
        import sys

        blocked = (
            "a2a",
            "alembic",
            "asyncpg",
            "fastapi",
            "mcp",
            "psycopg",
            "psycopg2",
            "sqlalchemy",
        )

        class BlockOptionalServices(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == name or fullname.startswith(f"{name}.")
                    for name in blocked
                ):
                    raise AssertionError(
                        f"unexpected optional service import: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, BlockOptionalServices())
        sys.path.insert(0, "src")

        import avalan
        import avalan.interaction
        from avalan.interaction import PortableContinuation

        loaded = sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in blocked
            )
        )
        assert not loaded, loaded
        assert PortableContinuation.__module__.endswith(".continuation")
        print("portable-cold-import-ok")
        """)

    result = run(
        [executable, "-c", code],
        capture_output=True,
        check=False,
        cwd=".",
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "portable-cold-import-ok"
