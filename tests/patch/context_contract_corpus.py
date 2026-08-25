"""Share inherited patch context cases across concrete runtimes."""

from dataclasses import dataclass
from pathlib import Path

from avalan.patch.domain import OperationType, PatchStatus


@dataclass(frozen=True, slots=True)
class ContextCorpusCase:
    """Bind one context-neutral semantic contract case."""

    case_id: str
    source_contract: str
    category: str
    operation: OperationType
    arguments: dict[str, object]
    initial_files: tuple[tuple[str, bytes], ...]
    expected_files: tuple[tuple[str, bytes], ...]
    expected_status: PatchStatus
    replace_root: bool = False
    inspection_only: bool = False
    expected_error: bool = False


SHARED_CONTEXT_CORPUS = (
    ContextCorpusCase(
        "P4-nested-exact-match",
        "phase_4_contract_test.py",
        "semantic",
        OperationType.EDIT,
        {
            "path": "nested/note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        },
        (("nested/note.txt", b"before\n"),),
        (("nested/note.txt", b"before\n"),),
        PatchStatus.COMMITTED,
        inspection_only=True,
    ),
    ContextCorpusCase(
        "P4-missing-match-fault",
        "phase_4_contract_test.py",
        "fault",
        OperationType.EDIT,
        {
            "path": "note.txt",
            "edits": [{"old_text": "absent", "new_text": "after"}],
        },
        (("note.txt", b"before\n"),),
        (("note.txt", b"before\n"),),
        PatchStatus.REJECTED,
        expected_error=True,
    ),
    ContextCorpusCase(
        "P4-root-replacement-race",
        "phase_4_contract_test.py",
        "race",
        OperationType.EDIT,
        {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        },
        (("note.txt", b"before\n"),),
        (("note.txt", b"before\n"),),
        PatchStatus.STALE,
        replace_root=True,
    ),
    ContextCorpusCase(
        "P7-operation-matrix",
        "phase_7_contract_test.py",
        "operation_matrix",
        OperationType.APPLY,
        {
            "patch": "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Add File: created.txt",
                    "+created",
                    "*** Delete File: deleted.txt",
                    "*** Update File: source.txt",
                    "*** Move to: moved.txt",
                    "*** Update File: note.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** End Patch",
                )
            )
        },
        (
            ("deleted.txt", b"deleted\n"),
            ("note.txt", b"before\n"),
            ("source.txt", b"source\n"),
        ),
        (
            ("created.txt", b"created\n"),
            ("moved.txt", b"source\n"),
            ("note.txt", b"after\n"),
        ),
        PatchStatus.COMMITTED,
    ),
    ContextCorpusCase(
        "P7-text-representation",
        "phase_7_contract_test.py",
        "representation",
        OperationType.APPLY,
        {
            "patch": "\n".join(
                (
                    "*** Begin Patch v1",
                    "*** Update File: bom.txt",
                    "@@",
                    "-before",
                    "+after",
                    "*** Update File: none.txt",
                    "@@",
                    "-before",
                    "\\ No newline at end of file",
                    "+after",
                    "\\ No newline at end of file",
                    "*** End of File",
                    "*** End Patch",
                )
            )
        },
        (
            ("bom.txt", b"\xef\xbb\xbfbefore\r\n"),
            ("none.txt", b"before"),
        ),
        (
            ("bom.txt", b"\xef\xbb\xbfafter\r\n"),
            ("none.txt", b"after"),
        ),
        PatchStatus.COMMITTED,
    ),
    ContextCorpusCase(
        "P9-closed-model-projection",
        "phase_9_contract_test.py",
        "projection",
        OperationType.EDIT,
        {
            "path": "note.txt",
            "edits": [{"old_text": "before", "new_text": "after"}],
        },
        (("note.txt", b"before\n"),),
        (("note.txt", b"after\n"),),
        PatchStatus.COMMITTED,
    ),
)


def write_context_corpus_tree(
    root: Path,
    files: tuple[tuple[str, bytes], ...],
) -> None:
    """Materialize one bounded shared-corpus tree."""
    for path, value in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)


def read_context_corpus_tree(root: Path) -> tuple[tuple[str, bytes], ...]:
    """Return the complete regular-file tree in stable logical order."""
    return tuple(
        sorted(
            (path.relative_to(root).as_posix(), path.read_bytes())
            for path in root.rglob("*")
            if path.is_file()
        )
    )
