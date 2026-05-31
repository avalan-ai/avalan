from pathlib import Path
from subprocess import run
from sys import executable


def test_task_import_does_not_load_optional_postgresql_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import builtins
blocked = {"asyncpg", "pgvector", "psycopg", "psycopg_pool", "sqlalchemy"}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"blocked import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import avalan.task
"""

    result = run(
        [executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
