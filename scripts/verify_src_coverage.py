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
_EXPECTED_EXCLUSION_PHASE1_SHA256 = (
    "4b480b52db122eee22a7139ce5a3eaa7d8e436e0cef1122fe84af00bf0b90575"
)
_EXPECTED_EXCLUSION_PHASE1_RELOCATION_SHA256 = (
    "efdc3890ced956313276da34397c282390f7955b98f3ddc801271a6cd10ad134"
)
_EXPECTED_EXCLUSION_PHASE2_SHA256 = (
    "226f1035786d8328f0e1fdea6070507b5d11a267f63910cd6c9fef3db8d2f63c"
)
_EXPECTED_EXCLUSION_PHASE2_RELOCATION_SHA256 = (
    "9c621baca6bb91d6fc18f82f4d0765e3f48f7881f1e63e668e56d16b0ebd129c"
)
_EXPECTED_EXCLUSION_PRIOR_SHA256 = (
    "eed04bcca2fbe223d9fe418df5a4b5f6d3bf2e10f12a3fcbe4320eb3feb44799"
)
_EXPECTED_EXCLUSION_PRIOR_RELOCATION_SHA256 = (
    "e580686aa03ca866cfd27f1794413643f191e8e32750f7b1ec64b5cba9a85405"
)
_EXPECTED_EXCLUSION_CURRENT_SHA256 = (
    "2ac7bbe973d9fbf38c6afb1ce1210f9a43d9781807d7078b3e57519008d1187d"
)
_EXPECTED_EXCLUSION_RELOCATION_SHA256 = (
    "8000a76131ea2fd0d7747adbc3057084fc0c4217c62694f62c02a547622c6c5f"
)
_EXPECTED_EXCLUSION_PHASE1_REVIEWER = "/root/interaction_round4_gates"
_EXPECTED_EXCLUSION_PHASE2_REVIEWER = "/root/phase2_coverage_metadata"
_EXPECTED_EXCLUSION_PRIOR_REVIEWER = (
    "/root/phase3_closure_audit/turn3_toolmanager_readonly"
)
_EXPECTED_EXCLUSION_REVIEWER = "/root/a2a_contract_review"


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
class _ExclusionDirective:
    """Store one stable source exclusion identity and location."""

    identity: str
    path: str
    line: int
    text: str
    occurrence: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _ExclusionSnapshot:
    """Store one pinned exclusion snapshot."""

    directives: tuple[_ExclusionDirective, ...]
    report_lines: dict[str, tuple[int, ...]]
    digest: str


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


def default_exclusion_current_path() -> Path:
    """Return the tracked current source-exclusion snapshot path."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "input"
        / "coverage_exclusions_current.json"
    )


def default_exclusion_relocation_path() -> Path:
    """Return the tracked source-exclusion relocation ledger path."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "input"
        / "coverage_exclusion_relocations_current.json"
    )


def default_exclusion_prior_path() -> Path:
    """Return the preceding source-exclusion snapshot path."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "input"
        / "coverage_exclusions_phase3.json"
    )


def default_exclusion_prior_relocation_path() -> Path:
    """Return the preceding source-exclusion relocation ledger path."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "input"
        / "coverage_exclusion_relocations_phase3.json"
    )


def verify_src_coverage(
    report_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    exclusion_baseline_path: Path | None = None,
    exclusion_prior_path: Path | None = None,
    exclusion_prior_relocation_path: Path | None = None,
    exclusion_current_path: Path | None = None,
    exclusion_relocation_path: Path | None = None,
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
    fixtures = root / "tests" / "fixtures" / "input"
    baseline_path = (
        exclusion_baseline_path or fixtures / "coverage_exclusions.json"
    )
    current_path = (
        exclusion_current_path or fixtures / "coverage_exclusions_current.json"
    )
    relocation_path = (
        exclusion_relocation_path
        or fixtures / "coverage_exclusion_relocations_current.json"
    )
    if (exclusion_prior_path is None) != (
        exclusion_prior_relocation_path is None
    ):
        raise CoverageVerificationError(
            "coverage exclusion prior snapshot and ledger must be paired"
        )
    explicit_prior = (
        exclusion_prior_path is not None
        and exclusion_prior_relocation_path is not None
    )
    implicit_complete_history = (
        exclusion_baseline_path is None
        and exclusion_current_path is None
        and exclusion_relocation_path is None
    )
    if explicit_prior or implicit_complete_history:
        expected_excluded_lines = _verify_exclusion_history_chain(
            baseline_path,
            exclusion_prior_path
            or fixtures / "coverage_exclusions_phase3.json",
            exclusion_prior_relocation_path
            or fixtures / "coverage_exclusion_relocations_phase3.json",
            current_path,
            relocation_path,
            root,
            source_root,
        )
    else:
        expected_excluded_lines = _verify_exclusion_history(
            baseline_path,
            current_path,
            relocation_path,
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


def _verify_exclusion_history(
    baseline_path: Path,
    current_path: Path,
    relocation_path: Path,
    root: Path,
    source_root: Path,
) -> dict[str, tuple[int, ...]]:
    baseline = _read_exclusion_snapshot(
        baseline_path,
        root,
        source_root,
        digest_field="baseline_sha256",
        expected_digest=_EXPECTED_EXCLUSION_BASELINE_SHA256,
        label="baseline",
    )
    current = _read_exclusion_snapshot(
        current_path,
        root,
        source_root,
        digest_field="snapshot_sha256",
        expected_digest=_EXPECTED_EXCLUSION_CURRENT_SHA256,
        label="current",
    )
    _verify_exclusion_relocations(
        relocation_path,
        baseline,
        current,
        root,
        source_root,
        expected_digest=_EXPECTED_EXCLUSION_RELOCATION_SHA256,
        expected_reviewer=_EXPECTED_EXCLUSION_REVIEWER,
        allow_report_additions=False,
    )
    _verify_observed_exclusions(current, root, source_root)
    return current.report_lines


def _verify_exclusion_history_chain(
    baseline_path: Path,
    prior_path: Path,
    prior_relocation_path: Path,
    current_path: Path,
    relocation_path: Path,
    root: Path,
    source_root: Path,
) -> dict[str, tuple[int, ...]]:
    """Verify every immutable exclusion-history link through current."""
    baseline = _read_exclusion_snapshot(
        baseline_path,
        root,
        source_root,
        digest_field="baseline_sha256",
        expected_digest=_EXPECTED_EXCLUSION_BASELINE_SHA256,
        label="baseline",
    )
    fixtures = baseline_path.parent
    phase1 = _read_exclusion_snapshot(
        fixtures / "coverage_exclusions_phase1.json",
        root,
        source_root,
        digest_field="snapshot_sha256",
        expected_digest=_EXPECTED_EXCLUSION_PHASE1_SHA256,
        label="phase-1",
    )
    _verify_exclusion_relocations(
        fixtures / "coverage_exclusion_relocations_phase1.json",
        baseline,
        phase1,
        root,
        source_root,
        expected_digest=_EXPECTED_EXCLUSION_PHASE1_RELOCATION_SHA256,
        expected_reviewer=_EXPECTED_EXCLUSION_PHASE1_REVIEWER,
        allow_report_additions=False,
    )
    phase2 = _read_exclusion_snapshot(
        fixtures / "coverage_exclusions_phase2.json",
        root,
        source_root,
        digest_field="snapshot_sha256",
        expected_digest=_EXPECTED_EXCLUSION_PHASE2_SHA256,
        label="phase-2",
    )
    _verify_exclusion_relocations(
        fixtures / "coverage_exclusion_relocations_phase2.json",
        phase1,
        phase2,
        root,
        source_root,
        expected_digest=_EXPECTED_EXCLUSION_PHASE2_RELOCATION_SHA256,
        expected_reviewer=_EXPECTED_EXCLUSION_PHASE2_REVIEWER,
        allow_report_additions=True,
    )
    prior = _read_exclusion_snapshot(
        prior_path,
        root,
        source_root,
        digest_field="snapshot_sha256",
        expected_digest=_EXPECTED_EXCLUSION_PRIOR_SHA256,
        label="prior",
    )
    _verify_exclusion_relocations(
        prior_relocation_path,
        phase2,
        prior,
        root,
        source_root,
        expected_digest=_EXPECTED_EXCLUSION_PRIOR_RELOCATION_SHA256,
        expected_reviewer=_EXPECTED_EXCLUSION_PRIOR_REVIEWER,
        allow_report_additions=True,
        allow_report_removals=True,
    )
    current = _read_exclusion_snapshot(
        current_path,
        root,
        source_root,
        digest_field="snapshot_sha256",
        expected_digest=_EXPECTED_EXCLUSION_CURRENT_SHA256,
        label="current",
    )
    _verify_exclusion_relocations(
        relocation_path,
        prior,
        current,
        root,
        source_root,
        expected_digest=_EXPECTED_EXCLUSION_RELOCATION_SHA256,
        expected_reviewer=_EXPECTED_EXCLUSION_REVIEWER,
        allow_report_additions=True,
        allow_report_removals=True,
        require_exact_report_deltas=True,
    )
    _verify_observed_exclusions(current, root, source_root)
    return current.report_lines


def _verify_observed_exclusions(
    current: _ExclusionSnapshot,
    root: Path,
    source_root: Path,
) -> None:
    """Verify source directives and configuration against a snapshot."""
    observed: list[tuple[str, int, str]] = []
    for source_path in sorted(source_root.rglob("*.py")):
        normalized = source_path.resolve().relative_to(root).as_posix()
        for line_number, text in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if _EXCLUSION_PATTERN.search(text):
                observed.append((normalized, line_number, text))
    expected = [
        (directive.path, directive.line, directive.text)
        for directive in current.directives
    ]
    if expected != observed:
        raise CoverageVerificationError(
            "source coverage exclusions differ from the reviewed current"
            " snapshot"
        )
    _verify_coverage_configuration(root)


def _read_exclusion_snapshot(
    path: Path,
    root: Path,
    source_root: Path,
    *,
    digest_field: str,
    expected_digest: str,
    label: str,
) -> _ExclusionSnapshot:
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
        digest_field,
    }:
        raise CoverageVerificationError(
            f"coverage exclusion {label} snapshot has invalid shape"
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
    directives: list[_ExclusionDirective] = []
    occurrences: dict[tuple[str, str], int] = {}
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
        occurrence_key = (normalized, text)
        occurrence = occurrences.get(occurrence_key, 0) + 1
        occurrences[occurrence_key] = occurrence
        directives.append(
            _ExclusionDirective(
                identity=_directive_identity(normalized, text, occurrence),
                path=normalized,
                line=line,
                text=text,
                occurrence=occurrence,
            )
        )
    locations = [
        (directive.path, directive.line, directive.text)
        for directive in directives
    ]
    if len(locations) != len(set(locations)):
        raise CoverageVerificationError(
            f"coverage exclusion {label} snapshot contains duplicates"
        )
    if locations != sorted(locations):
        raise CoverageVerificationError(
            f"coverage exclusion {label} snapshot is not sorted"
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
        raw.get(digest_field) != calculated_digest
        or calculated_digest != expected_digest
    ):
        raise CoverageVerificationError(
            f"coverage exclusion {label} snapshot digest changed without"
            " verifier"
            " review"
        )
    return _ExclusionSnapshot(
        directives=tuple(directives),
        report_lines=report_lines,
        digest=calculated_digest,
    )


def _verify_exclusion_relocations(
    path: Path,
    baseline: _ExclusionSnapshot,
    current: _ExclusionSnapshot,
    root: Path,
    source_root: Path,
    *,
    expected_digest: str,
    expected_reviewer: str,
    allow_report_additions: bool,
    allow_report_removals: bool = False,
    require_exact_report_deltas: bool = False,
) -> None:
    try:
        raw = strict_json_path(path)
    except StrictJsonError as exc:
        raise CoverageVerificationError(str(exc)) from exc
    base_keys = {
        "schema_version",
        "baseline_snapshot_sha256",
        "current_snapshot_sha256",
        "directive_count_before",
        "directive_count_after",
        "report_excluded_line_count_before",
        "report_excluded_line_count_after",
        "directive_relocations",
        "report_exclusion_relocations",
        "ledger_sha256",
    }
    if not isinstance(raw, dict):
        raise CoverageVerificationError(
            "coverage exclusion relocation ledger has invalid shape"
        )
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2, 3, 4}:
        raise CoverageVerificationError(
            "coverage exclusion relocation schema_version must be the"
            " integer 1, 2, 3, or 4"
        )
    expected_keys = base_keys
    if schema_version >= 2:
        expected_keys |= {"report_exclusion_additions"}
    if schema_version >= 3:
        expected_keys |= {"report_exclusion_removals"}
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise CoverageVerificationError(
            "coverage exclusion relocation ledger has invalid shape"
        )
    if schema_version >= 2 and not allow_report_additions:
        raise CoverageVerificationError(
            "coverage exclusion report additions are prohibited for this"
            " history link"
        )
    if schema_version >= 3 and not allow_report_removals:
        raise CoverageVerificationError(
            "coverage exclusion report removals are prohibited for this"
            " history link"
        )
    if require_exact_report_deltas and schema_version != 4:
        raise CoverageVerificationError(
            "current coverage exclusion relocation requires schema_version 4"
        )
    if (
        raw.get("baseline_snapshot_sha256") != baseline.digest
        or raw.get("current_snapshot_sha256") != current.digest
    ):
        raise CoverageVerificationError(
            "coverage exclusion relocation snapshot references changed"
        )
    directive_relocations = _directive_relocations(
        raw.get("directive_relocations"),
        root,
        source_root,
        expected_reviewer,
    )
    report_relocations = _report_exclusion_relocations(
        raw.get("report_exclusion_relocations"),
        root,
        source_root,
        expected_reviewer,
    )
    report_additions = _report_exclusion_additions(
        raw.get("report_exclusion_additions", []),
        root,
        source_root,
        expected_reviewer,
    )
    report_removals = _report_exclusion_removals(
        raw.get("report_exclusion_removals", []),
        root,
        source_root,
        expected_reviewer,
    )
    _verify_exclusion_counts(
        raw,
        baseline,
        current,
        report_additions,
        report_removals,
        exact_report_deltas=schema_version >= 4,
    )
    _verify_directive_relocations(
        baseline.directives,
        current.directives,
        directive_relocations,
    )
    _verify_report_relocations(
        baseline.report_lines,
        current.report_lines,
        report_relocations,
        report_additions,
        report_removals,
        exact_report_deltas=schema_version >= 4,
    )
    digest_payload = {
        key: raw[key] for key in expected_keys if key != "ledger_sha256"
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
        raw.get("ledger_sha256") != calculated_digest
        or calculated_digest != expected_digest
    ):
        raise CoverageVerificationError(
            "coverage exclusion relocation ledger digest changed without"
            " verifier review"
        )


def _verify_exclusion_counts(
    raw: dict[str, object],
    baseline: _ExclusionSnapshot,
    current: _ExclusionSnapshot,
    report_additions: dict[str, tuple[int, ...]],
    report_removals: dict[str, tuple[int, ...]],
    *,
    exact_report_deltas: bool,
) -> None:
    counts = {
        "directive_count_before": len(baseline.directives),
        "directive_count_after": len(current.directives),
        "report_excluded_line_count_before": sum(
            len(lines) for lines in baseline.report_lines.values()
        ),
        "report_excluded_line_count_after": sum(
            len(lines) for lines in current.report_lines.values()
        ),
    }
    for field, expected in counts.items():
        value = raw.get(field)
        if type(value) is not int or value != expected:
            raise CoverageVerificationError(
                f"coverage exclusion relocation {field} is inconsistent"
            )
    if counts["directive_count_before"] != counts["directive_count_after"]:
        raise CoverageVerificationError(
            "coverage exclusion relocation changed exclusion counts"
        )
    report_line_delta = (
        counts["report_excluded_line_count_after"]
        - counts["report_excluded_line_count_before"]
    )
    if exact_report_deltas:
        expected_delta = sum(
            len(lines) for lines in report_additions.values()
        ) - sum(len(lines) for lines in report_removals.values())
    else:
        reviewed_existing_growth = sum(
            max(
                0,
                len(current.report_lines[path])
                - len(baseline.report_lines[path]),
            )
            for path in set(baseline.report_lines) & set(current.report_lines)
        )
        expected_delta = (
            sum(len(lines) for lines in report_additions.values())
            + reviewed_existing_growth
            - sum(len(lines) for lines in report_removals.values())
        )
    if report_line_delta != expected_delta:
        raise CoverageVerificationError(
            "coverage exclusion relocation changed exclusion counts"
        )


def _directive_relocations(
    raw: object,
    root: Path,
    source_root: Path,
    expected_reviewer: str,
) -> dict[str, tuple[str, str, int, int, int]]:
    if not isinstance(raw, list):
        raise CoverageVerificationError(
            "coverage directive relocations must be a list"
        )
    relocations: dict[str, tuple[str, str, int, int, int]] = {}
    expected_keys = {
        "identity",
        "path",
        "text",
        "occurrence",
        "from_line",
        "to_line",
        "reviewed_by",
        "reason",
    }
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise CoverageVerificationError(
                "coverage directive relocation has invalid shape"
            )
        identity = entry.get("identity")
        raw_path = entry.get("path")
        text = entry.get("text")
        occurrence = entry.get("occurrence")
        from_line = entry.get("from_line")
        to_line = entry.get("to_line")
        if (
            not isinstance(identity, str)
            or not isinstance(raw_path, str)
            or not isinstance(text, str)
            or not _EXCLUSION_PATTERN.search(text)
            or type(occurrence) is not int
            or occurrence <= 0
            or type(from_line) is not int
            or from_line <= 0
            or type(to_line) is not int
            or to_line <= 0
            or from_line == to_line
        ):
            raise CoverageVerificationError(
                "coverage directive relocation has invalid fields"
            )
        _verify_relocation_review(entry, expected_reviewer)
        normalized = _normalize_source_path(raw_path, root, source_root)
        if identity != _directive_identity(normalized, text, occurrence):
            raise CoverageVerificationError(
                "coverage directive relocation identity changed"
            )
        if identity in relocations:
            raise CoverageVerificationError(
                "coverage directive relocation is duplicated"
            )
        relocations[identity] = (
            normalized,
            text,
            occurrence,
            from_line,
            to_line,
        )
    return relocations


def _report_exclusion_relocations(
    raw: object,
    root: Path,
    source_root: Path,
    expected_reviewer: str,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    if not isinstance(raw, list):
        raise CoverageVerificationError(
            "coverage report exclusion relocations must be a list"
        )
    relocations: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    expected_keys = {
        "path",
        "from_lines",
        "to_lines",
        "reviewed_by",
        "reason",
    }
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise CoverageVerificationError(
                "coverage report exclusion relocation has invalid shape"
            )
        raw_path = entry.get("path")
        from_raw = entry.get("from_lines")
        to_raw = entry.get("to_lines")
        if (
            not isinstance(raw_path, str)
            or not isinstance(from_raw, list)
            or not isinstance(to_raw, list)
        ):
            raise CoverageVerificationError(
                "coverage report exclusion relocation has invalid fields"
            )
        _verify_relocation_review(entry, expected_reviewer)
        normalized = _normalize_source_path(raw_path, root, source_root)
        from_lines = _line_numbers(
            from_raw,
            normalized,
            label="baseline excluded",
        )
        to_lines = _line_numbers(
            to_raw,
            normalized,
            label="current excluded",
        )
        if not from_lines or not to_lines:
            raise CoverageVerificationError(
                "coverage report exclusion relocation must preserve evidence"
            )
        if from_lines == to_lines:
            raise CoverageVerificationError(
                "coverage report exclusion relocation did not move"
            )
        if normalized in relocations:
            raise CoverageVerificationError(
                "coverage report exclusion relocation is duplicated"
            )
        relocations[normalized] = (from_lines, to_lines)
    return relocations


def _report_exclusion_additions(
    raw: object,
    root: Path,
    source_root: Path,
    expected_reviewer: str,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(raw, list):
        raise CoverageVerificationError(
            "coverage report exclusion additions must be a list"
        )
    additions: dict[str, tuple[int, ...]] = {}
    expected_keys = {
        "path",
        "lines",
        "reviewed_by",
        "reason",
    }
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise CoverageVerificationError(
                "coverage report exclusion addition has invalid shape"
            )
        raw_path = entry.get("path")
        raw_lines = entry.get("lines")
        if not isinstance(raw_path, str) or not isinstance(raw_lines, list):
            raise CoverageVerificationError(
                "coverage report exclusion addition has invalid fields"
            )
        _verify_relocation_review(entry, expected_reviewer)
        normalized = _normalize_source_path(raw_path, root, source_root)
        if not normalized.endswith(".py") or not (root / normalized).is_file():
            raise CoverageVerificationError(
                "coverage report exclusion addition path does not exist:"
                f" {normalized}"
            )
        lines = _line_numbers(
            raw_lines,
            normalized,
            label="added excluded",
        )
        if not lines:
            raise CoverageVerificationError(
                "empty coverage report exclusion addition is prohibited:"
                f" {normalized}"
            )
        if normalized in additions:
            raise CoverageVerificationError(
                "coverage report exclusion addition is duplicated"
            )
        additions[normalized] = lines
    return additions


def _report_exclusion_removals(
    raw: object,
    root: Path,
    source_root: Path,
    expected_reviewer: str,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(raw, list):
        raise CoverageVerificationError(
            "coverage report exclusion removals must be a list"
        )
    removals: dict[str, tuple[int, ...]] = {}
    expected_keys = {"path", "lines", "reviewed_by", "reason"}
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise CoverageVerificationError(
                "coverage report exclusion removal has invalid shape"
            )
        raw_path = entry.get("path")
        raw_lines = entry.get("lines")
        if not isinstance(raw_path, str) or not isinstance(raw_lines, list):
            raise CoverageVerificationError(
                "coverage report exclusion removal has invalid fields"
            )
        _verify_relocation_review(entry, expected_reviewer)
        normalized = _normalize_source_path(raw_path, root, source_root)
        lines = _line_numbers(
            raw_lines,
            normalized,
            label="removed excluded",
        )
        if not lines:
            raise CoverageVerificationError(
                "empty coverage report exclusion removal is prohibited:"
                f" {normalized}"
            )
        if normalized in removals:
            raise CoverageVerificationError(
                "coverage report exclusion removal is duplicated"
            )
        removals[normalized] = lines
    return removals


def _verify_relocation_review(
    entry: dict[str, object],
    expected_reviewer: str,
) -> None:
    reviewer = entry.get("reviewed_by")
    reason = entry.get("reason")
    if reviewer != expected_reviewer:
        raise CoverageVerificationError(
            "coverage exclusion relocation is unreviewed"
        )
    if (
        not isinstance(reason, str)
        or len(reason.strip()) < 30
        or reason.strip().lower() in {"pending", "placeholder", "tbd", "todo"}
    ):
        raise CoverageVerificationError(
            "coverage exclusion relocation lacks a concrete reason"
        )


def _verify_directive_relocations(
    baseline: tuple[_ExclusionDirective, ...],
    current: tuple[_ExclusionDirective, ...],
    relocations: dict[str, tuple[str, str, int, int, int]],
) -> None:
    baseline_by_id = {directive.identity: directive for directive in baseline}
    current_by_id = {directive.identity: directive for directive in current}
    if (
        len(baseline_by_id) != len(baseline)
        or len(current_by_id) != len(current)
        or set(baseline_by_id) != set(current_by_id)
    ):
        raise CoverageVerificationError(
            "coverage directive identity, text, or count changed"
        )
    changed = {
        identity
        for identity, directive in baseline_by_id.items()
        if directive.line != current_by_id[identity].line
    }
    if set(relocations) != changed:
        raise CoverageVerificationError(
            "coverage directive relocation is missing, extra, or unreviewed"
        )
    for identity in changed:
        before = baseline_by_id[identity]
        after = current_by_id[identity]
        expected = (
            before.path,
            before.text,
            before.occurrence,
            before.line,
            after.line,
        )
        if relocations[identity] != expected:
            raise CoverageVerificationError(
                "coverage directive relocation differs from snapshots"
            )


def _verify_report_relocations(
    baseline: dict[str, tuple[int, ...]],
    current: dict[str, tuple[int, ...]],
    relocations: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    additions: dict[str, tuple[int, ...]],
    removals: dict[str, tuple[int, ...]],
    *,
    exact_report_deltas: bool,
) -> None:
    removed = set(baseline) - set(current)
    added = set(current) - set(baseline)
    if removed:
        raise CoverageVerificationError(
            "coverage report exclusion path inventory removed entries"
        )
    shared = set(baseline) & set(current)
    if exact_report_deltas:
        expected_additions = {
            path: tuple(sorted(set(current[path]) - set(baseline[path])))
            for path in shared
            if set(current[path]) - set(baseline[path])
        }
        expected_additions.update({path: current[path] for path in added})
        if additions != expected_additions:
            raise CoverageVerificationError(
                "coverage report exclusion addition differs from snapshots"
            )
        expected_removals = {
            path: tuple(sorted(set(baseline[path]) - set(current[path])))
            for path in shared
            if set(baseline[path]) - set(current[path])
        }
        if removals != expected_removals:
            raise CoverageVerificationError(
                "coverage report exclusion removal differs from snapshots"
            )
    else:
        if set(additions) != added:
            raise CoverageVerificationError(
                "coverage report exclusion addition is missing, extra, or"
                " unreviewed"
            )
        expected_removal_paths = {
            path for path in shared if len(baseline[path]) > len(current[path])
        }
        if set(removals) != expected_removal_paths:
            raise CoverageVerificationError(
                "coverage report exclusion removal is missing, extra, or"
                " unreviewed"
            )
    changed = {
        path
        for path, lines in baseline.items()
        if path in current and lines != current[path]
    }
    if set(relocations) != changed:
        raise CoverageVerificationError(
            "coverage report exclusion relocation is missing, extra, or"
            " unreviewed"
        )
    for path in changed:
        if relocations[path] != (baseline[path], current[path]):
            raise CoverageVerificationError(
                "coverage report exclusion relocation differs from snapshots"
            )
        if not exact_report_deltas:
            removed_lines = removals.get(path, ())
            removed_count = max(0, len(baseline[path]) - len(current[path]))
            if removed_count != len(removed_lines) or not set(
                removed_lines
            ) <= set(baseline[path]):
                raise CoverageVerificationError(
                    "coverage report exclusion removal differs from snapshots"
                )
    if not exact_report_deltas:
        for path in added:
            if additions[path] != current[path]:
                raise CoverageVerificationError(
                    "coverage report exclusion addition differs from snapshots"
                )


def _directive_identity(path: str, text: str, occurrence: int) -> str:
    return sha256(f"{path}\0{text}\0{occurrence}".encode("utf-8")).hexdigest()


def _verify_coverage_configuration(root: Path) -> None:
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
        "--exclusion-current",
        type=Path,
        default=default_exclusion_current_path(),
    )
    parser.add_argument(
        "--exclusion-relocations",
        type=Path,
        default=default_exclusion_relocation_path(),
    )
    parser.add_argument(
        "--exclusion-prior",
        type=Path,
        default=default_exclusion_prior_path(),
    )
    parser.add_argument(
        "--exclusion-prior-relocations",
        type=Path,
        default=default_exclusion_prior_relocation_path(),
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
    default_history = (
        args.exclusion_baseline == default_exclusion_baseline_path()
        and args.exclusion_prior == default_exclusion_prior_path()
        and args.exclusion_prior_relocations
        == default_exclusion_prior_relocation_path()
        and args.exclusion_current == default_exclusion_current_path()
        and args.exclusion_relocations == default_exclusion_relocation_path()
    )
    history_arguments = (
        {}
        if default_history
        else {
            "exclusion_baseline_path": args.exclusion_baseline,
            "exclusion_prior_path": args.exclusion_prior,
            "exclusion_prior_relocation_path": (
                args.exclusion_prior_relocations
            ),
            "exclusion_current_path": args.exclusion_current,
            "exclusion_relocation_path": args.exclusion_relocations,
        }
    )
    try:
        result = verify_src_coverage(
            args.report,
            repo_root=args.repo_root,
            **history_arguments,
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
