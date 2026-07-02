from ast import (
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    Attribute,
    BinOp,
    BitOr,
    Call,
    ClassDef,
    FunctionDef,
    Import,
    ImportFrom,
    Name,
    NodeVisitor,
    parse,
)
from pathlib import Path
from re import search
from unittest import TestCase, main

_ROOT = Path(__file__).resolve().parents[2]
_SKILL_SOURCE = _ROOT / "src" / "avalan" / "skill"
_RUNTIME_MODULES = (
    "_async.py",
    "manifest.py",
    "matcher.py",
    "observability.py",
    "reader.py",
    "registry.py",
    "resolver.py",
    "runtime.py",
)
_PATH_FILESYSTEM_METHODS = {
    "exists",
    "glob",
    "is_dir",
    "is_file",
    "iterdir",
    "lstat",
    "open",
    "read_bytes",
    "read_text",
    "readlink",
    "resolve",
    "rglob",
    "stat",
    "write_bytes",
    "write_text",
}
_BLOCKING_BUILTIN_FILESYSTEM_CALLS = {"open"}
_BLOCKING_OS_FILESYSTEM_CALLS = {
    "listdir",
    "lstat",
    "open",
    "readlink",
    "scandir",
    "stat",
}
_PATH_RECEIVER_NAMES = {
    "candidate",
    "directory",
    "entry",
    "package_root",
    "path",
    "resolved",
    "resolved_package",
    "resource_path",
    "root",
    "root_path",
}
_ALLOWED_BLOCKING_FILESYSTEM_CALLS = {
    ("resolver.py", "SkillAsyncFileSystem.resolve_path", "resolve"),
    ("resolver.py", "SkillAsyncFileSystem.stat_path", "stat"),
    ("resolver.py", "SkillAsyncFileSystem.lstat_path", "lstat"),
    ("resolver.py", "_read_bytes", "open"),
    ("resolver.py", "_bounded_list_directory", "iterdir"),
}


class SkillRuntimeImportBoundaryPhase14Test(TestCase):
    def test_runtime_modules_use_async_filesystem_boundary(self) -> None:
        violations: list[str] = []
        for module_name in _RUNTIME_MODULES:
            path = _SKILL_SOURCE / module_name
            visitor = _BlockingFilesystemVisitor(path.name)
            visitor.visit(parse(path.read_text(encoding="utf-8")))
            violations.extend(visitor.violations)

        self.assertEqual(violations, [])

    def test_runtime_modules_do_not_define_retry_loops(self) -> None:
        retry_mentions: list[str] = []
        for module_name in _RUNTIME_MODULES:
            path = _SKILL_SOURCE / module_name
            text = path.read_text(encoding="utf-8")
            if search(r"\b(retry|retries|backoff)\b", text):
                retry_mentions.append(path.name)

        self.assertEqual(retry_mentions, [])

    def test_static_guard_catches_path_aliases_and_assigned_variables(
        self,
    ) -> None:
        cases = (
            (
                "assigned_path",
                (
                    "from pathlib import Path\n"
                    "def load():\n"
                    "    target = Path('x')\n"
                    "    return target.read_bytes()\n"
                ),
                "fixture.py:4:load:read_bytes",
            ),
            (
                "aliased_constructor",
                (
                    "from pathlib import Path as P\n"
                    "def load():\n"
                    "    return P('x').read_bytes()\n"
                ),
                "fixture.py:3:load:read_bytes",
            ),
            (
                "pathlib_import_alias",
                (
                    "import pathlib as pl\n"
                    "def load():\n"
                    "    target = pl.Path('x')\n"
                    "    return target.open()\n"
                ),
                "fixture.py:4:load:open",
            ),
        )

        for name, source, expected in cases:
            with self.subTest(name=name):
                visitor = _BlockingFilesystemVisitor("fixture.py")
                visitor.visit(parse(source))
                self.assertIn(expected, visitor.violations)

    def test_static_guard_catches_path_derived_receivers(self) -> None:
        cases = (
            (
                "binop_constructor_receiver",
                (
                    "from pathlib import Path\n"
                    "def load():\n"
                    "    return (Path('x') / 'y').read_bytes()\n"
                ),
                "fixture.py:3:load:read_bytes",
            ),
            (
                "joinpath_receiver",
                (
                    "from pathlib import Path\n"
                    "def load():\n"
                    "    return Path('x').joinpath('y').read_text()\n"
                ),
                "fixture.py:3:load:read_text",
            ),
            (
                "assigned_binop_from_typed_root",
                (
                    "from pathlib import Path\n"
                    "def load(root: Path):\n"
                    "    target = root / 'x'\n"
                    "    return target.open()\n"
                ),
                "fixture.py:4:load:open",
            ),
        )

        for name, source, expected in cases:
            with self.subTest(name=name):
                visitor = _BlockingFilesystemVisitor("fixture.py")
                visitor.visit(parse(source))
                self.assertIn(expected, visitor.violations)

    def test_static_guard_catches_os_aliases(self) -> None:
        cases = (
            (
                "os_module_alias",
                (
                    "import os as operating_system\n"
                    "def load():\n"
                    "    return operating_system.listdir('.')\n"
                ),
                "fixture.py:3:load:operating_system.listdir",
            ),
            (
                "os_function_alias",
                (
                    "from os import scandir as scan\n"
                    "def load():\n"
                    "    return scan('.')\n"
                ),
                "fixture.py:3:load:scan",
            ),
        )

        for name, source, expected in cases:
            with self.subTest(name=name):
                visitor = _BlockingFilesystemVisitor("fixture.py")
                visitor.visit(parse(source))
                self.assertIn(expected, visitor.violations)

    def test_static_guard_catches_pep604_path_annotations(self) -> None:
        cases = (
            (
                "path_union",
                (
                    "from pathlib import Path\n"
                    "def load(source: Path | None):\n"
                    "    if source is None:\n"
                    "        return None\n"
                    "    return source.read_text()\n"
                ),
                "fixture.py:5:load:read_text",
            ),
            (
                "aliased_path_union",
                (
                    "from pathlib import Path as P\n"
                    "def load(source: P | None):\n"
                    "    return source.open()\n"
                ),
                "fixture.py:3:load:open",
            ),
        )

        for name, source, expected in cases:
            with self.subTest(name=name):
                visitor = _BlockingFilesystemVisitor("fixture.py")
                visitor.visit(parse(source))
                self.assertIn(expected, visitor.violations)


class _BlockingFilesystemVisitor(NodeVisitor):
    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._scope: list[str] = []
        self._os_function_aliases: set[str] = set()
        self._os_modules: set[str] = {"os"}
        self._path_aliases: set[str] = set()
        self._path_modules: set[str] = {"pathlib"}
        self._path_variable_scopes: list[set[str]] = [set()]
        self.violations: list[str] = []

    def visit_Import(self, node: Import) -> None:
        for import_alias in node.names:
            if import_alias.name == "os":
                self._os_modules.add(import_alias.asname or "os")
            if import_alias.name == "pathlib":
                self._path_modules.add(import_alias.asname or "pathlib")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        if node.module == "os":
            for import_alias in node.names:
                if import_alias.name in _BLOCKING_OS_FILESYSTEM_CALLS:
                    self._os_function_aliases.add(
                        import_alias.asname or import_alias.name
                    )
        if node.module == "pathlib":
            for import_alias in node.names:
                if import_alias.name == "Path":
                    self._path_aliases.add(import_alias.asname or "Path")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ClassDef) -> None:
        self._scope.append(node.name)
        self._path_variable_scopes.append(set())
        self.generic_visit(node)
        self._path_variable_scopes.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Assign(self, node: Assign) -> None:
        if self._is_path_expression(node.value):
            for target in node.targets:
                self._mark_path_variable(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if node.value is not None and self._is_path_expression(node.value):
            self._mark_path_variable(node.target)
        self.generic_visit(node)

    def visit_Call(self, node: Call) -> None:
        if self._is_builtin_filesystem_call(node):
            self._record_violation(node, "builtin.open")
        if self._is_os_filesystem_call(node):
            self._record_violation(node, self._call_name(node))
        if isinstance(node.func, Attribute):
            call_name = node.func.attr
            if (
                call_name in _PATH_FILESYSTEM_METHODS
                and self._is_direct_path_receiver(node.func.value)
            ):
                scope = ".".join(self._scope)
                allowed = (
                    self._module_name,
                    scope,
                    call_name,
                ) in _ALLOWED_BLOCKING_FILESYSTEM_CALLS
                if not allowed:
                    self._record_violation(node, call_name)
        self.generic_visit(node)

    def _visit_function(self, node: FunctionDef | AsyncFunctionDef) -> None:
        self._scope.append(node.name)
        self._path_variable_scopes.append(set())
        self._mark_path_arguments(node)
        self.generic_visit(node)
        self._path_variable_scopes.pop()
        self._scope.pop()

    def _record_violation(self, node: Call, call_name: str) -> None:
        scope = ".".join(self._scope)
        self.violations.append(
            f"{self._module_name}:{node.lineno}:{scope}:{call_name}"
        )

    def _is_builtin_filesystem_call(self, node: Call) -> bool:
        if not isinstance(node.func, Name):
            return False
        return node.func.id in _BLOCKING_BUILTIN_FILESYSTEM_CALLS

    def _is_os_filesystem_call(self, node: Call) -> bool:
        if isinstance(node.func, Name):
            return node.func.id in self._os_function_aliases
        if not isinstance(node.func, Attribute):
            return False
        if not isinstance(node.func.value, Name):
            return False
        return (
            node.func.value.id in self._os_modules
            and node.func.attr in _BLOCKING_OS_FILESYSTEM_CALLS
        )

    def _is_direct_path_receiver(self, node: object) -> bool:
        return self._is_path_expression(node)

    def _is_path_expression(self, node: object) -> bool:
        if isinstance(node, Name):
            return (
                self._is_path_variable(node.id)
                or node.id in _PATH_RECEIVER_NAMES
                or node.id.endswith("_path")
            )
        if isinstance(node, Call):
            return self._is_path_constructor(
                node
            ) or self._is_path_joinpath_call(node)
        if isinstance(node, BinOp):
            return self._is_path_expression(
                node.left
            ) or self._is_path_expression(node.right)
        if isinstance(node, Attribute):
            return self._is_path_expression(node.value)
        return False

    def _is_path_constructor(self, node: object) -> bool:
        if not isinstance(node, Call):
            return False
        if isinstance(node.func, Name):
            return node.func.id in self._path_aliases
        if not isinstance(node.func, Attribute):
            return False
        if node.func.attr != "Path":
            return False
        if not isinstance(node.func.value, Name):
            return False
        return node.func.value.id in self._path_modules

    def _is_path_joinpath_call(self, node: Call) -> bool:
        if not isinstance(node.func, Attribute):
            return False
        return node.func.attr == "joinpath" and self._is_path_expression(
            node.func.value
        )

    def _mark_path_variable(self, node: object) -> None:
        if isinstance(node, Name):
            self._path_variable_scopes[-1].add(node.id)

    def _is_path_variable(self, name: str) -> bool:
        return any(
            name in scope for scope in reversed(self._path_variable_scopes)
        )

    def _mark_path_arguments(
        self,
        node: FunctionDef | AsyncFunctionDef,
    ) -> None:
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        for argument in arguments:
            if self._annotation_is_path(argument.annotation):
                self._path_variable_scopes[-1].add(argument.arg)

    def _annotation_is_path(self, annotation: object) -> bool:
        if isinstance(annotation, BinOp) and isinstance(annotation.op, BitOr):
            return self._annotation_is_path(
                annotation.left
            ) or self._annotation_is_path(annotation.right)
        if isinstance(annotation, Name):
            return annotation.id in self._path_aliases
        if not isinstance(annotation, Attribute):
            return False
        if annotation.attr != "Path":
            return False
        if not isinstance(annotation.value, Name):
            return False
        return annotation.value.id in self._path_modules

    def _call_name(self, node: Call) -> str:
        if isinstance(node.func, Name):
            return node.func.id
        if isinstance(node.func, Attribute):
            if isinstance(node.func.value, Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return "unknown"


if __name__ == "__main__":
    main()
