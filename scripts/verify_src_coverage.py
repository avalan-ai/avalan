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
_EXPECTED_EXCLUSION_BASELINE_SHA256 = (
    "111cfc4a1b6b5c85a550d458aa6906d73360c7f939b0c9e01e4dcbaa3587041c"
)


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


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def default_report_path() -> Path:
    """Return the default coverage report path."""
    return repository_root() / "coverage.json"


def default_exclusion_baseline_path() -> Path:
    """Return the tracked source-exclusion baseline path."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "input"
        / "coverage_exclusions.json"
    )


def verify_src_coverage(
    report_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    exclusion_baseline_path: Path | None = None,
) -> CoverageVerification:
    """Verify exact total, per-file, and inventory source coverage."""
    root = (repo_root or repository_root()).resolve()
    report = report_path or root / "coverage.json"
    try:
        raw_payload = strict_json_path(report)
    except StrictJsonError as exc:
        raise CoverageVerificationError(str(exc)) from exc
    if not isinstance(raw_payload, dict):
        raise CoverageVerificationError("coverage report must be an object")
    payload = cast(dict[str, object], raw_payload)
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict):
        raise CoverageVerificationError("coverage files must be an object")
    raw_totals = payload.get("totals")
    if not isinstance(raw_totals, dict):
        raise CoverageVerificationError("coverage totals must be an object")

    source_root = (root / "src").resolve()
    if not source_root.is_dir():
        raise CoverageVerificationError(
            f"source root is not a directory: {source_root}"
        )
    allowed_files, required_files, empty_files = _source_inventory(
        root,
        source_root,
    )
    expected_excluded_lines = _verify_exclusion_baseline(
        exclusion_baseline_path
        or root / "tests" / "fixtures" / "input" / "coverage_exclusions.json",
        root,
        source_root,
    )
    source_analyzer = Coverage(config_file=False, data_file=None)
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
        if not isinstance(raw_file, dict):
            raise CoverageVerificationError(
                f"coverage file entry must be an object: {normalized}"
            )
        raw_summary = raw_file.get("summary")
        raw_executed = raw_file.get("executed_lines")
        raw_excluded = raw_file.get("excluded_lines")
        raw_missing = raw_file.get("missing_lines")
        if (
            not isinstance(raw_summary, dict)
            or not isinstance(raw_executed, list)
            or not isinstance(raw_excluded, list)
            or not isinstance(raw_missing, list)
        ):
            raise CoverageVerificationError(
                "coverage file lacks summary, executed_lines, "
                f"excluded_lines, or missing_lines: {normalized}"
            )
        summary = _summary(raw_summary, f"coverage file {normalized}")
        executed_lines = _executed_line_numbers(
            raw_executed,
            normalized,
            summary,
        )
        missing_lines = _missing_line_numbers(raw_missing, normalized)
        excluded_lines = _line_numbers(
            raw_excluded,
            normalized,
            label="excluded",
        )
        excluded_line_set = set(excluded_lines)
        source_path = root / normalized
        try:
            _, statements, parser_excluded, _, _ = source_analyzer.analysis2(
                str(source_path)
            )
        except (CoverageException, OSError, UnicodeError) as exc:
            raise CoverageVerificationError(
                f"cannot analyze source statements for {normalized}: {exc}"
            ) from exc
        statement_set = set(statements)
        executed_statement_set = set(executed_lines) & statement_set
        if len(executed_statement_set) != summary.covered_lines:
            raise CoverageVerificationError(
                f"executed-line count is inconsistent for {normalized}"
            )
        if len(missing_lines) != summary.missing_lines:
            raise CoverageVerificationError(
                f"missing-line count is inconsistent for {normalized}"
            )
        if len(excluded_lines) != summary.excluded_lines:
            raise CoverageVerificationError(
                f"excluded-line count is inconsistent for {normalized}"
            )
        if excluded_lines != expected_excluded_lines.get(normalized, ()):
            raise CoverageVerificationError(
                f"excluded-line evidence changed for {normalized}"
            )
        if tuple(parser_excluded) != excluded_lines:
            raise CoverageVerificationError(
                f"coverage parser exclusions differ for {normalized}"
            )
        if len(statement_set) != summary.num_statements:
            raise CoverageVerificationError(
                f"source statement inventory changed for {normalized}"
            )
        source_line_count = len(
            source_path.read_text(encoding="utf-8").splitlines()
        )
        if any(
            line > source_line_count
            for line in (*executed_lines, *missing_lines)
        ):
            raise CoverageVerificationError(
                f"coverage evidence is outside {normalized}"
            )
        if set(executed_lines) & set(missing_lines):
            raise CoverageVerificationError(
                f"executed and missing lines overlap for {normalized}"
            )
        if excluded_line_set & set(missing_lines):
            raise CoverageVerificationError(
                f"excluded and missing lines overlap for {normalized}"
            )
        if (
            executed_statement_set & excluded_line_set
            or executed_statement_set & set(missing_lines)
            or executed_statement_set | set(missing_lines) != statement_set
        ):
            raise CoverageVerificationError(
                f"line evidence is inconsistent for {normalized}"
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
            "coverage totals are inconsistent with file summaries: "
            f"reported={totals}, calculated={calculated}"
        )
    under_covered = sorted(
        name
        for name, item in measured.items()
        if item.missing_lines != 0 or item.covered_lines != item.num_statements
    )
    if (
        totals.missing_lines != 0
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


def _verify_exclusion_baseline(
    path: Path,
    root: Path,
    source_root: Path,
) -> dict[str, tuple[int, ...]]:
    try:
        raw = strict_json_path(path)
    except StrictJsonError as exc:
        raise CoverageVerificationError(str(exc)) from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "source_root",
        "exclusions",
        "report_excluded_lines",
        "coverage_configuration",
        "baseline_sha256",
    }:
        raise CoverageVerificationError(
            "coverage exclusion baseline has invalid shape"
        )
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
    ):
        raise CoverageVerificationError(
            "coverage exclusion schema_version must be the integer 1"
        )
    if raw.get("source_root") != "src":
        raise CoverageVerificationError(
            "coverage exclusion source_root must be src"
        )
    if raw.get("coverage_configuration") != "none":
        raise CoverageVerificationError(
            "feature-specific coverage configuration is prohibited"
        )
    expected_raw = raw.get("exclusions")
    if not isinstance(expected_raw, list):
        raise CoverageVerificationError("coverage exclusions must be a list")
    expected: list[tuple[str, int, str]] = []
    for entry in expected_raw:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "line",
            "text",
        }:
            raise CoverageVerificationError(
                "coverage exclusion entry has invalid shape"
            )
        raw_path = entry.get("path")
        line = entry.get("line")
        text = entry.get("text")
        if (
            not isinstance(raw_path, str)
            or type(line) is not int
            or line <= 0
            or not isinstance(text, str)
            or not _EXCLUSION_PATTERN.search(text)
        ):
            raise CoverageVerificationError(
                "coverage exclusion entry has invalid fields"
            )
        normalized = _normalize_source_path(raw_path, root, source_root)
        expected.append((normalized, line, text))
    if len(expected) != len(set(expected)):
        raise CoverageVerificationError(
            "coverage exclusion baseline contains duplicates"
        )
    observed: list[tuple[str, int, str]] = []
    for source_path in sorted(source_root.rglob("*.py")):
        normalized = source_path.resolve().relative_to(root).as_posix()
        for line_number, text in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if _EXCLUSION_PATTERN.search(text):
                observed.append((normalized, line_number, text))
    if expected != observed:
        raise CoverageVerificationError(
            "source coverage exclusions differ from the frozen baseline"
        )
    raw_report_lines = raw.get("report_excluded_lines")
    if not isinstance(raw_report_lines, dict):
        raise CoverageVerificationError(
            "coverage report exclusion baseline must be an object"
        )
    report_lines: dict[str, tuple[int, ...]] = {}
    for raw_path, raw_lines in raw_report_lines.items():
        if not isinstance(raw_path, str) or not isinstance(raw_lines, list):
            raise CoverageVerificationError(
                "coverage report exclusion entry has invalid fields"
            )
        normalized = _normalize_source_path(raw_path, root, source_root)
        if normalized in report_lines:
            raise CoverageVerificationError(
                f"duplicate normalized exclusion report path: {normalized}"
            )
        if not normalized.endswith(".py") or not (root / normalized).is_file():
            raise CoverageVerificationError(
                f"coverage exclusion report path does not exist: {normalized}"
            )
        lines = _line_numbers(raw_lines, normalized, label="excluded")
        if not lines:
            raise CoverageVerificationError(
                f"empty exclusion report entry is prohibited: {normalized}"
            )
        report_lines[normalized] = lines
    digest_payload = {
        "coverage_configuration": raw["coverage_configuration"],
        "exclusions": raw["exclusions"],
        "report_excluded_lines": raw["report_excluded_lines"],
        "source_root": raw["source_root"],
    }
    calculated_digest = sha256(
        dumps(
            digest_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if (
        raw.get("baseline_sha256") != calculated_digest
        or calculated_digest != _EXPECTED_EXCLUSION_BASELINE_SHA256
    ):
        raise CoverageVerificationError(
            "coverage exclusion baseline digest changed without verifier"
            " review"
        )
    direct_configuration = root / ".coveragerc"
    if direct_configuration.exists():
        raise CoverageVerificationError(
            "unfrozen coverage configuration is prohibited: .coveragerc"
        )
    for config in (
        root / "setup.cfg",
        root / "tox.ini",
        root / "pyproject.toml",
        root / "pytest.ini",
    ):
        if not config.exists():
            continue
        configuration = config.read_text(encoding="utf-8")
        lowered = configuration.lower()
        if (
            "[coverage" in lowered
            or "[tool.coverage" in lowered
            or _PYTEST_COVERAGE_OPTION_PATTERN.search(configuration)
        ):
            raise CoverageVerificationError(
                f"unfrozen coverage configuration is prohibited: {config.name}"
            )
    return report_lines


def _normalize_source_path(
    raw_name: str,
    root: Path,
    source_root: Path,
) -> str:
    if "\\" in raw_name:
        raw_name = raw_name.replace("\\", "/")
    posix = PurePosixPath(raw_name)
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


def _summary(raw: dict[object, object], label: str) -> CoverageSummary:
    values: dict[str, int] = {}
    for field in (
        "covered_lines",
        "excluded_lines",
        "missing_lines",
        "num_statements",
    ):
        value = raw.get(field)
        if type(value) is not int or value < 0:
            raise CoverageVerificationError(
                f"{label} {field} must be a non-negative integer"
            )
        values[field] = value
    summary = CoverageSummary(**values)
    if summary.covered_lines + summary.missing_lines != summary.num_statements:
        raise CoverageVerificationError(
            f"{label} statement counts are inconsistent"
        )
    return summary


def _missing_line_numbers(raw: list[object], name: str) -> tuple[int, ...]:
    return _line_numbers(raw, name, label="missing")


def _executed_line_numbers(
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
    return _line_numbers(raw, name, label="executed")


def _line_numbers(
    raw: list[object],
    name: str,
    *,
    label: str,
) -> tuple[int, ...]:
    values: list[int] = []
    for value in raw:
        if type(value) is not int or value <= 0:
            raise CoverageVerificationError(
                f"{label} line numbers must be positive integers: {name}"
            )
        values.append(value)
    if len(values) != len(set(values)):
        raise CoverageVerificationError(
            f"duplicate {label} line numbers for {name}"
        )
    if values != sorted(values):
        raise CoverageVerificationError(
            f"{label} line numbers are not sorted for {name}"
        )
    return tuple(values)


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Verify exact statement coverage for every source module."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=default_report_path(),
    )
    parser.add_argument(
        "--exclusion-baseline",
        type=Path,
        default=default_exclusion_baseline_path(),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repository_root(),
    )
    return parser.parse_args()


def main() -> int:
    """Run exact source-coverage verification."""
    args = _parse_args()
    try:
        result = verify_src_coverage(
            args.report,
            repo_root=args.repo_root,
            exclusion_baseline_path=args.exclusion_baseline,
        )
    except CoverageVerificationError as exc:
        print(f"exact source coverage failed: {exc}", file=stderr)
        return 1
    print(
        "exact source coverage passed: "
        f"files={len(result.files)} statements={result.summary.num_statements}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
