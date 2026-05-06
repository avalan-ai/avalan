from os import environ, pathsep
from pathlib import Path
from subprocess import run
from sys import executable
from textwrap import dedent
from unittest import TestCase, main


class LoaderDependencyTestCase(TestCase):
    def test_agent_command_import_does_not_require_pgvector(self) -> None:
        root_path = Path(__file__).resolve().parents[2]
        source_path = root_path / "src"
        environment = environ.copy()
        python_path = str(source_path)
        if environment.get("PYTHONPATH"):
            python_path = f"{python_path}{pathsep}{environment['PYTHONPATH']}"
        environment["PYTHONPATH"] = python_path

        script = dedent(
            """
            import importlib.abc
            import sys


            class BlockPgvector(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if (
                        fullname == "pgvector"
                        or fullname.startswith("pgvector.")
                    ):
                        raise ModuleNotFoundError("No module named 'pgvector'")
                    return None


            sys.meta_path.insert(0, BlockPgvector())
            import avalan.cli.commands.agent
            """
        )

        result = run(
            [executable, "-c", script],
            capture_output=True,
            check=False,
            cwd=root_path,
            env=environment,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    main()
