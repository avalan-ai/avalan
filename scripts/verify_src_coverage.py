#!/usr/bin/env python
"""Verify exact statement coverage for the complete source inventory."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from pathlib import Path, PurePosixPath
from re import IGNORECASE
from re import compile as compile_regex
from sys import stderr
from typing import cast

from coverage import Coverage
from coverage.exceptions import CoverageException
from input_contract_json import StrictJsonError, strict_json_path

_EXCLUSION_PATTERN = compile_regex(
    r"#\s*(?:pragma\s*:?\s*no\s*cover|coverage\s*:?\s*ignore)",
    IGNORECASE,
)
_PYTEST_COVERAGE_OPTION_PATTERN = compile_regex(
    r"--(?:no-)?cov(?:[-=\s\"']|$)",
    IGNORECASE,
)
_DYNAMIC_CODE_PATTERN = compile_regex(r"\b(?:exec|compile)\s*\(")


class CoverageVerificationError(RuntimeError):
    """Report an incomplete or internally inconsistent coverage report."""


@dataclass(frozen=True, kw_only=True, slots=True)
class CoverageSummary:
    """Store exact statement counts for one coverage scope."""

    covered_lines: int
    excluded_lines: int
    missing_lines: int
    num_statements: int


@dataclass(frozen=True, kw_only=True, slots=True)
class CoverageVerification:
    """Store the verified source inventory and aggregate counts."""

    files: tuple[str, ...]
    summary: CoverageSummary


@dataclass(frozen=True, kw_only=True, slots=True)
class ExclusionDirective:
    """Store one stable source exclusion identity and location."""

    identity: str
    path: str
    line: int
    text: str
    occurrence: int


@dataclass(frozen=True, kw_only=True, slots=True)
class ExclusionSnapshot:
    """Store one pinned exclusion snapshot."""

    directives: tuple[ExclusionDirective, ...]
    report_lines: dict[str, tuple[int, ...]]
    digest: str


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def fixture_root() -> Path:
    """Return the tracked structured-input fixture directory."""
    return repository_root() / "tests" / "fixtures" / "input"


def default_report_path() -> Path:
    """Return the default coverage report path."""
    return repository_root() / "coverage.json"


def default_exclusion_baseline_path() -> Path:
    """Return the tracked source-exclusion baseline path."""
    return fixture_root() / "coverage_exclusions.json"


def default_exclusion_current_path() -> Path:
    """Return the tracked current source-exclusion snapshot path."""
    return fixture_root() / "coverage_exclusions_current.json"


def default_exclusion_relocation_path() -> Path:
    """Return the compact baseline-to-current delta ledger path."""
    return fixture_root() / "coverage_exclusion_relocations_current.json"


def verify_src_coverage(
    report_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    exclusion_baseline_path: Path | None = None,
    exclusion_current_path: Path | None = None,
    exclusion_relocation_path: Path | None = None,
) -> CoverageVerification:
    """Verify exact total, per-file, and inventory source coverage."""
    root = (repo_root or repository_root()).resolve()
    report = report_path or root / "coverage.json"
    source_root = (root / "src").resolve()
    if not source_root.is_dir():
        raise CoverageVerificationError(
            f"source root is not a directory: {source_root}"
        )
    verify_report_freshness(report, root)
    verify_no_dynamic_coverage_tricks(root)
    fixtures = root / "tests" / "fixtures" / "input"
    baseline = read_exclusion_snapshot(
        exclusion_baseline_path or fixtures / "coverage_exclusions.json",
        root,
        digest_field="baseline_sha256",
        label="baseline",
    )
    current = read_exclusion_snapshot(
        exclusion_current_path
        or fixtures / "coverage_exclusions_current.json",
        root,
        digest_field="snapshot_sha256",
        label="current",
    )
    verify_exclusion_delta(
        exclusion_relocation_path
        or fixtures / "coverage_exclusion_relocations_current.json",
        baseline,
        current,
    )
    verify_observed_exclusions(current, root)

    try:
        payload = _mapping(
            strict_json_path(report),
            "coverage report",
        )
    except StrictJsonError as exc:
        raise CoverageVerificationError(str(exc)) from exc
    raw_files = _mapping(payload.get("files"), "coverage files")
    raw_totals = _mapping(payload.get("totals"), "coverage totals")
    allowed_files, required_files, empty_files = _source_inventory(
        root, source_root
    )
    analyzer = Coverage(config_file=False, data_file=None)
    measured: dict[str, CoverageSummary] = {}
    for raw_name, raw_file in raw_files.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise CoverageVerificationError(
                "coverage file names must be non-empty strings"
            )
        normalized = _normalize_source_path(raw_name, root, source_root)
        if normalized in measured:
            raise CoverageVerificationError(
                f"duplicate normalized coverage file: {normalized}"
            )
        entry = _mapping(raw_file, f"coverage file {normalized}")
        summary = _summary(
            _mapping(entry.get("summary"), f"coverage file {normalized}"),
            f"coverage file {normalized}",
        )
        executed = _executed_lines(
            _list(entry.get("executed_lines"), "executed_lines"),
            normalized,
            summary,
        )
        excluded = _line_numbers(
            _list(entry.get("excluded_lines"), "excluded_lines"),
            normalized,
            "excluded",
        )
        missing = _line_numbers(
            _list(entry.get("missing_lines"), "missing_lines"),
            normalized,
            "missing",
        )
        source_path = root / normalized
        try:
            _, statements, parser_excluded, _, _ = analyzer.analysis2(
                str(source_path)
            )
        except (CoverageException, OSError, UnicodeError) as exc:
            raise CoverageVerificationError(
                f"cannot analyze source statements for {normalized}: {exc}"
            ) from exc
        statement_set = set(statements)
        executed_statements = set(executed) & statement_set
        if (
            len(executed_statements) != summary.covered_lines
            or len(missing) != summary.missing_lines
            or len(excluded) != summary.excluded_lines
            or len(statement_set) != summary.num_statements
        ):
            raise CoverageVerificationError(
                f"coverage line counts are inconsistent for {normalized}"
            )
        if excluded != current.report_lines.get(normalized, ()):
            raise CoverageVerificationError(
                f"excluded-line evidence changed for {normalized}"
            )
        if tuple(parser_excluded) != excluded:
            raise CoverageVerificationError(
                f"coverage parser exclusions differ for {normalized}"
            )
        if (
            set(executed) & set(missing)
            or set(excluded) & set(missing)
            or executed_statements & set(excluded)
            or executed_statements | set(missing) != statement_set
        ):
            raise CoverageVerificationError(
                f"line evidence is inconsistent for {normalized}"
            )
        source_lines = len(
            source_path.read_text(encoding="utf-8").splitlines()
        )
        if any(line > source_lines for line in (*executed, *missing)):
            raise CoverageVerificationError(
                f"coverage evidence is outside {normalized}"
            )
        if normalized in empty_files and summary != CoverageSummary(
            covered_lines=0,
            excluded_lines=0,
            missing_lines=0,
            num_statements=0,
        ):
            raise CoverageVerificationError(
                f"empty source module has invented statements: {normalized}"
            )
        measured[normalized] = summary

    missing_files = sorted(required_files - set(measured))
    unexpected_files = sorted(set(measured) - allowed_files)
    if missing_files or unexpected_files:
        raise CoverageVerificationError(
            "coverage source inventory mismatch: "
            f"missing={missing_files}, unexpected={unexpected_files}"
        )
    totals = _summary(raw_totals, "coverage totals")
    calculated = CoverageSummary(
        covered_lines=sum(item.covered_lines for item in measured.values()),
        excluded_lines=sum(item.excluded_lines for item in measured.values()),
        missing_lines=sum(item.missing_lines for item in measured.values()),
        num_statements=sum(item.num_statements for item in measured.values()),
    )
    if totals != calculated:
        raise CoverageVerificationError(
            "coverage totals are inconsistent with file summaries"
        )
    under_covered = sorted(
        name
        for name, item in measured.items()
        if item.missing_lines or item.covered_lines != item.num_statements
    )
    if (
        totals.missing_lines
        or totals.covered_lines != totals.num_statements
        or under_covered
    ):
        raise CoverageVerificationError(
            "source statement coverage is not exact: "
            f"total={totals}, files={under_covered}"
        )
    return CoverageVerification(
        files=tuple(sorted(measured)),
        summary=totals,
    )


def verify_report_freshness(report: Path, root: Path) -> None:
    """Bind coverage evidence to the current source, tests, and gate code."""
    if not report.is_file():
        raise CoverageVerificationError(
            f"coverage report does not exist: {report}"
        )
    inputs = [
        path
        for directory in ("src", "tests", "scripts")
        for path in (root / directory).rglob("*")
        if path.is_file()
        and (
            path.suffix in {".py", ".json"}
            or path.name in {"Makefile", "pyproject.toml"}
        )
    ]
    inputs.extend(
        path
        for path in (root / "Makefile", root / "pyproject.toml")
        if path.is_file()
    )
    newest_input = max(
        (path.stat().st_mtime_ns for path in inputs),
        default=0,
    )
    if report.stat().st_mtime_ns < newest_input:
        raise CoverageVerificationError(
            "coverage report predates current source, tests, or gate scripts"
        )


def verify_no_dynamic_coverage_tricks(
    root: Path,
    manifest_path: Path | None = None,
) -> None:
    """Reject dynamic-code calls in active acceptance test files."""
    path = (
        manifest_path
        or root / "tests" / "fixtures" / "input" / "acceptance_manifest.json"
    )
    if not path.exists() and manifest_path is None:
        return
    try:
        payload = _mapping(
            strict_json_path(path),
            "acceptance manifest",
        )
    except StrictJsonError as exc:
        raise CoverageVerificationError(str(exc)) from exc
    raw_nodes = _list(payload.get("nodes"), "acceptance nodes")
    files: set[str] = set()
    for raw in raw_nodes:
        node = _mapping(raw, "acceptance node")
        if node.get("lifecycle") != "active":
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or "::" not in node_id:
            raise CoverageVerificationError(
                "active acceptance node has an invalid pytest ID"
            )
        relative = node_id.split("::", 1)[0]
        candidate = (root / relative).resolve()
        tests_root = (root / "tests").resolve()
        if not candidate.is_relative_to(tests_root) or not candidate.is_file():
            raise CoverageVerificationError(
                f"active acceptance test does not exist: {relative}"
            )
        files.add(relative)
    if not files:
        raise CoverageVerificationError(
            "acceptance manifest has no active test files"
        )
    for relative in sorted(files):
        match = _DYNAMIC_CODE_PATTERN.search(
            (root / relative).read_text(encoding="utf-8")
        )
        if match is not None:
            raise CoverageVerificationError(
                "active acceptance tests contain a prohibited coverage trick "
                f"using dynamic code: {relative}:{match.group(0)}"
            )


def read_exclusion_snapshot(
    path: Path,
    root: Path,
    *,
    digest_field: str,
    label: str,
) -> ExclusionSnapshot:
    """Read and validate one complete exclusion snapshot."""
    try:
        payload = _mapping(strict_json_path(path), f"{label} snapshot")
    except StrictJsonError as exc:
        raise CoverageVerificationError(str(exc)) from exc
    _exact_keys(
        payload,
        {
            "schema_version",
            "source_root",
            "exclusions",
            "report_excluded_lines",
            "coverage_configuration",
            digest_field,
        },
        f"{label} snapshot",
    )
    if payload.get("schema_version") != 1:
        raise CoverageVerificationError(
            "coverage exclusion schema_version must be 1"
        )
    if payload.get("source_root") != "src":
        raise CoverageVerificationError(
            "coverage exclusion source_root must be src"
        )
    if payload.get("coverage_configuration") != "none":
        raise CoverageVerificationError(
            "feature-specific coverage configuration is prohibited"
        )
    source_root = (root / "src").resolve()
    raw_directives = _list(payload.get("exclusions"), "coverage exclusions")
    directives: list[ExclusionDirective] = []
    occurrences: dict[tuple[str, str], int] = {}
    for raw in raw_directives:
        entry = _mapping(raw, "coverage exclusion")
        _exact_keys(entry, {"path", "line", "text"}, "coverage exclusion")
        path_value = _nonempty_string(entry.get("path"), "exclusion path")
        line = _positive_int(entry.get("line"), "exclusion line")
        text = _nonempty_string(entry.get("text"), "exclusion text")
        if _EXCLUSION_PATTERN.search(text) is None:
            raise CoverageVerificationError(
                "coverage exclusion text lacks an exclusion directive"
            )
        normalized = _normalize_source_path(path_value, root, source_root)
        key = (normalized, text)
        occurrence = occurrences.get(key, 0) + 1
        occurrences[key] = occurrence
        directives.append(
            ExclusionDirective(
                identity=_directive_identity(normalized, text, occurrence),
                path=normalized,
                line=line,
                text=text,
                occurrence=occurrence,
            )
        )
    locations = [(item.path, item.line, item.text) for item in directives]
    if locations != sorted(set(locations)):
        raise CoverageVerificationError(
            f"coverage exclusion {label} snapshot is unsorted or duplicated"
        )
    raw_report = _mapping(
        payload.get("report_excluded_lines"),
        "coverage report exclusions",
    )
    report_lines: dict[str, tuple[int, ...]] = {}
    for raw_path, raw_lines in raw_report.items():
        if not isinstance(raw_path, str):
            raise CoverageVerificationError(
                "coverage report exclusion path must be a string"
            )
        normalized = _normalize_source_path(raw_path, root, source_root)
        if normalized in report_lines or not (root / normalized).is_file():
            raise CoverageVerificationError(
                f"invalid coverage report exclusion path: {normalized}"
            )
        lines = _line_numbers(
            _list(raw_lines, "coverage report exclusion lines"),
            normalized,
            "excluded",
        )
        if not lines:
            raise CoverageVerificationError(
                f"empty exclusion report entry: {normalized}"
            )
        report_lines[normalized] = lines
    digest_payload = {
        "coverage_configuration": payload["coverage_configuration"],
        "exclusions": payload["exclusions"],
        "report_excluded_lines": payload["report_excluded_lines"],
        "source_root": payload["source_root"],
    }
    digest = _digest(digest_payload)
    if payload.get(digest_field) != digest:
        raise CoverageVerificationError(
            f"coverage exclusion {label} snapshot digest is invalid"
        )
    return ExclusionSnapshot(
        directives=tuple(directives),
        report_lines=report_lines,
        digest=digest,
    )


def exclusion_delta(
    baseline: ExclusionSnapshot,
    current: ExclusionSnapshot,
) -> dict[str, object]:
    """Return compact counts and digests for the derived snapshot delta."""
    before = {item.identity: item for item in baseline.directives}
    after = {item.identity: item for item in current.directives}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    relocated = sorted(
        identity
        for identity in set(before) & set(after)
        if before[identity].line != after[identity].line
    )
    directive_details = {
        "added": [_directive_record(after[identity]) for identity in added],
        "removed": [
            _directive_record(before[identity]) for identity in removed
        ],
        "relocated": [
            {
                "identity": identity,
                "path": after[identity].path,
                "from_line": before[identity].line,
                "to_line": after[identity].line,
            }
            for identity in relocated
        ],
    }
    baseline_lines = {
        (path, line)
        for path, lines in baseline.report_lines.items()
        for line in lines
    }
    current_lines = {
        (path, line)
        for path, lines in current.report_lines.items()
        for line in lines
    }
    added_lines = sorted(current_lines - baseline_lines)
    removed_lines = sorted(baseline_lines - current_lines)
    report_details = {
        "added": [{"path": path, "line": line} for path, line in added_lines],
        "removed": [
            {"path": path, "line": line} for path, line in removed_lines
        ],
    }
    return {
        "directives": {
            "added": len(added),
            "removed": len(removed),
            "relocated": len(relocated),
            "sha256": _digest(directive_details),
        },
        "report_lines": {
            "added": len(added_lines),
            "removed": len(removed_lines),
            "sha256": _digest(report_details),
        },
    }


def verify_exclusion_delta(
    path: Path,
    baseline: ExclusionSnapshot,
    current: ExclusionSnapshot,
) -> None:
    """Validate the single compact baseline-to-current delta ledger."""
    try:
        payload = _mapping(strict_json_path(path), "exclusion delta")
    except StrictJsonError as exc:
        raise CoverageVerificationError(str(exc)) from exc
    _exact_keys(
        payload,
        {
            "schema_version",
            "baseline_snapshot_sha256",
            "current_snapshot_sha256",
            "directives",
            "report_lines",
            "review",
            "ledger_sha256",
        },
        "exclusion delta",
    )
    if payload.get("schema_version") != 2:
        raise CoverageVerificationError(
            "coverage exclusion delta schema_version must be 2"
        )
    if (
        payload.get("baseline_snapshot_sha256") != baseline.digest
        or payload.get("current_snapshot_sha256") != current.digest
    ):
        raise CoverageVerificationError(
            "coverage exclusion delta does not bind both snapshots"
        )
    derived = exclusion_delta(baseline, current)
    if (
        payload.get("directives") != derived["directives"]
        or payload.get("report_lines") != derived["report_lines"]
    ):
        raise CoverageVerificationError(
            "coverage exclusion delta differs from the derived change set"
        )
    review = _mapping(payload.get("review"), "exclusion delta review")
    _exact_keys(review, {"reviewed_by", "rationale"}, "delta review")
    _nonempty_string(review.get("reviewed_by"), "delta reviewer")
    _nonempty_string(review.get("rationale"), "delta rationale")
    canonical = {
        key: value for key, value in payload.items() if key != "ledger_sha256"
    }
    if payload.get("ledger_sha256") != _digest(canonical):
        raise CoverageVerificationError(
            "coverage exclusion delta ledger digest is invalid"
        )


def verify_observed_exclusions(
    current: ExclusionSnapshot,
    root: Path,
) -> None:
    """Verify current source directives and parser exclusions."""
    source_root = (root / "src").resolve()
    observed: list[tuple[str, int, str]] = []
    parser_lines: dict[str, tuple[int, ...]] = {}
    analyzer = Coverage(config_file=False, data_file=None)
    for source_path in sorted(source_root.rglob("*.py")):
        normalized = source_path.resolve().relative_to(root).as_posix()
        for line_number, text in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if _EXCLUSION_PATTERN.search(text):
                observed.append((normalized, line_number, text))
        try:
            _, _, excluded, _, _ = analyzer.analysis2(str(source_path))
        except (CoverageException, OSError, UnicodeError) as exc:
            raise CoverageVerificationError(
                f"cannot analyze source exclusions for {normalized}: {exc}"
            ) from exc
        if excluded:
            parser_lines[normalized] = tuple(excluded)
    expected = [
        (item.path, item.line, item.text) for item in current.directives
    ]
    if observed != expected:
        raise CoverageVerificationError(
            "source coverage directives differ from the current snapshot"
        )
    if parser_lines != current.report_lines:
        raise CoverageVerificationError(
            "coverage parser exclusions differ from the current snapshot"
        )
    _verify_coverage_configuration(root)


def _source_inventory(
    root: Path,
    source_root: Path,
) -> tuple[set[str], set[str], set[str]]:
    allowed: set[str] = set()
    required: set[str] = set()
    empty: set[str] = set()
    for path in source_root.rglob("*.py"):
        if path.is_symlink() and not path.resolve().is_relative_to(
            source_root
        ):
            raise CoverageVerificationError(
                f"source path escapes source root: {path}"
            )
        if not path.is_file():
            continue
        normalized = path.resolve().relative_to(root).as_posix()
        allowed.add(normalized)
        if path.read_bytes().strip():
            required.add(normalized)
        else:
            empty.add(normalized)
    return allowed, required, empty


def _verify_coverage_configuration(root: Path) -> None:
    if (root / ".coveragerc").exists():
        raise CoverageVerificationError(
            "unfrozen coverage configuration is prohibited: .coveragerc"
        )
    for path in (
        root / "setup.cfg",
        root / "tox.ini",
        root / "pyproject.toml",
        root / "pytest.ini",
    ):
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        lowered = content.lower()
        if (
            "[coverage" in lowered
            or "[tool.coverage" in lowered
            or _PYTEST_COVERAGE_OPTION_PATTERN.search(content)
        ):
            raise CoverageVerificationError(
                f"unfrozen coverage configuration is prohibited: {path.name}"
            )


def _normalize_source_path(
    raw_name: str,
    root: Path,
    source_root: Path,
) -> str:
    posix = PurePosixPath(raw_name.replace("\\", "/"))
    candidate = Path(*posix.parts)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        relative = resolved.relative_to(source_root)
    except ValueError as exc:
        raise CoverageVerificationError(
            f"coverage path is outside src/: {raw_name}"
        ) from exc
    return (Path("src") / relative).as_posix()


def _summary(raw: dict[str, object], label: str) -> CoverageSummary:
    values = {
        field: _nonnegative_int(raw.get(field), f"{label} {field}")
        for field in (
            "covered_lines",
            "excluded_lines",
            "missing_lines",
            "num_statements",
        )
    }
    summary = CoverageSummary(**values)
    if summary.covered_lines + summary.missing_lines != summary.num_statements:
        raise CoverageVerificationError(
            f"{label} statement counts are inconsistent"
        )
    return summary


def _executed_lines(
    raw: list[object],
    name: str,
    summary: CoverageSummary,
) -> tuple[int, ...]:
    if raw == [0] and summary == CoverageSummary(
        covered_lines=0,
        excluded_lines=0,
        missing_lines=0,
        num_statements=0,
    ):
        return ()
    return _line_numbers(raw, name, "executed")


def _line_numbers(
    raw: list[object],
    name: str,
    label: str,
) -> tuple[int, ...]:
    values = [
        _positive_int(value, f"{label} line number for {name}")
        for value in raw
    ]
    if values != sorted(set(values)):
        raise CoverageVerificationError(
            f"{label} line numbers are unsorted or duplicated for {name}"
        )
    return tuple(values)


def _directive_identity(path: str, text: str, occurrence: int) -> str:
    return sha256(f"{path}\0{text}\0{occurrence}".encode()).hexdigest()


def _directive_record(item: ExclusionDirective) -> dict[str, object]:
    return {
        "identity": item.identity,
        "path": item.path,
        "line": item.line,
        "text": item.text,
        "occurrence": item.occurrence,
    }


def _digest(value: object) -> str:
    return sha256(
        dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CoverageVerificationError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise CoverageVerificationError(f"{label} must be a list")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoverageVerificationError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise CoverageVerificationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CoverageVerificationError(
            f"{label} must be a non-negative integer"
        )
    return value


def _exact_keys(
    value: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise CoverageVerificationError(
            f"{label} has invalid keys: {sorted(set(value) ^ expected)}"
        )


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Verify exact statement coverage for every source module."
    )
    parser.add_argument("--report", type=Path, default=default_report_path())
    parser.add_argument(
        "--exclusion-baseline",
        type=Path,
        default=default_exclusion_baseline_path(),
    )
    parser.add_argument(
        "--exclusion-current",
        type=Path,
        default=default_exclusion_current_path(),
    )
    parser.add_argument(
        "--exclusion-relocations",
        type=Path,
        default=default_exclusion_relocation_path(),
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    return parser.parse_args()


def main() -> int:
    """Run exact source-coverage verification."""
    args = _parse_args()
    try:
        result = verify_src_coverage(
            args.report,
            repo_root=args.repo_root,
            exclusion_baseline_path=args.exclusion_baseline,
            exclusion_current_path=args.exclusion_current,
            exclusion_relocation_path=args.exclusion_relocations,
        )
    except (CoverageVerificationError, StrictJsonError) as exc:
        print(f"exact source coverage failed: {exc}", file=stderr)
        return 1
    print(
        "exact source coverage passed: "
        f"files={len(result.files)} statements={result.summary.num_statements}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
