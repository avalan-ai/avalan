"""Exercise compact exact source-coverage verification."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps
from os import utime
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "input"


def _load_verifier() -> ModuleType:
    """Return the source coverage verifier module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_compact_src_coverage_verifier"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_src_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _write(path: Path, value: object) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _freshen(report: Path, root: Path) -> None:
    """Make a synthetic report newer than every bound gate input."""
    inputs = [
        path
        for directory in ("src", "tests", "scripts")
        for path in (root / directory).rglob("*")
        if path.is_file()
    ]
    inputs.extend(
        path
        for path in (root / "Makefile", root / "pyproject.toml")
        if path.is_file()
    )
    newest = max(
        (path.stat().st_mtime_ns for path in inputs),
        default=report.stat().st_mtime_ns,
    )
    utime(report, ns=(newest + 1, newest + 1))


def _snapshot(
    *,
    directives: list[dict[str, object]] | None = None,
    report_lines: dict[str, list[int]] | None = None,
    digest_field: str,
) -> dict[str, object]:
    """Return one signed exclusion snapshot."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_root": "src",
        "exclusions": directives or [],
        "report_excluded_lines": report_lines or {},
        "coverage_configuration": "none",
    }
    digest_payload = {
        key: payload[key]
        for key in (
            "coverage_configuration",
            "exclusions",
            "report_excluded_lines",
            "source_root",
        )
    }
    payload[digest_field] = _VERIFIER._digest(digest_payload)
    return payload


def _coverage_report(
    *,
    executed: list[int] | None = None,
    missing: list[int] | None = None,
) -> dict[str, Any]:
    """Return exact coverage JSON for a one-statement source file."""
    executed_lines = [1] if executed is None else executed
    missing_lines = [] if missing is None else missing
    covered = 1 if executed_lines == [1] else 0
    summary = {
        "covered_lines": covered,
        "excluded_lines": 0,
        "missing_lines": len(missing_lines),
        "num_statements": 1,
    }
    return {
        "files": {
            "src/sample.py": {
                "executed_lines": executed_lines,
                "excluded_lines": [],
                "missing_lines": missing_lines,
                "summary": summary,
            }
        },
        "totals": deepcopy(summary),
    }


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal repository with compact exclusion evidence."""
    source = tmp_path / "src"
    fixtures = tmp_path / "tests" / "fixtures" / "input"
    source.mkdir(parents=True)
    fixtures.mkdir(parents=True)
    (source / "__init__.py").write_text("", encoding="utf-8")
    (source / "sample.py").write_text("value = 1\n", encoding="utf-8")
    baseline_path = fixtures / "coverage_exclusions.json"
    current_path = fixtures / "coverage_exclusions_current.json"
    _write(
        baseline_path,
        _snapshot(digest_field="baseline_sha256"),
    )
    _write(
        current_path,
        _snapshot(digest_field="snapshot_sha256"),
    )
    baseline = _VERIFIER.read_exclusion_snapshot(
        baseline_path,
        tmp_path,
        digest_field="baseline_sha256",
        label="baseline",
    )
    current = _VERIFIER.read_exclusion_snapshot(
        current_path,
        tmp_path,
        digest_field="snapshot_sha256",
        label="current",
    )
    delta = _VERIFIER.exclusion_delta(baseline, current)
    ledger: dict[str, object] = {
        "schema_version": 2,
        "baseline_snapshot_sha256": baseline.digest,
        "current_snapshot_sha256": current.digest,
        "directives": delta["directives"],
        "report_lines": delta["report_lines"],
        "review": {
            "reviewed_by": "test reviewer",
            "rationale": "Synthetic exact baseline-to-current review.",
        },
    }
    ledger["ledger_sha256"] = _VERIFIER._digest(ledger)
    _write(
        fixtures / "coverage_exclusion_relocations_current.json",
        ledger,
    )
    report = tmp_path / "coverage.json"
    _write(report, _coverage_report())
    return tmp_path, report


@pytest.mark.parametrize("empty_executed", (None, [], [0]))
def test_exact_coverage_accepts_empty_module_variants(
    tmp_path: Path,
    empty_executed: list[int] | None,
) -> None:
    """Accept exact source coverage while deriving empty module inventory."""
    root, report = _repo(tmp_path)
    payload = _coverage_report()
    if empty_executed is not None:
        payload["files"]["src/__init__.py"] = {
            "executed_lines": empty_executed,
            "excluded_lines": [],
            "missing_lines": [],
            "summary": {
                "covered_lines": 0,
                "excluded_lines": 0,
                "missing_lines": 0,
                "num_statements": 0,
            },
        }
        _write(report, payload)
        _freshen(report, root)

    result = _VERIFIER.verify_src_coverage(report, repo_root=root)

    expected = (
        ("src/sample.py",)
        if empty_executed is None
        else ("src/__init__.py", "src/sample.py")
    )
    assert result.files == expected
    assert result.summary.covered_lines == 1
    assert result.summary.missing_lines == 0


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("counts", "statement counts are inconsistent"),
        ("missing", "not exact"),
        ("excluded", "excluded-line evidence changed"),
        ("inventory", "source inventory mismatch"),
        ("outside", "outside src"),
        ("duplicate", "duplicate normalized"),
        ("totals", "totals are inconsistent"),
        ("json", "duplicate"),
        ("schema", "coverage files must be an object"),
    ),
)
def test_exact_coverage_rejects_invalid_reports(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    """Reject forged statements, exclusions, inventory, paths, and totals."""
    root, report = _repo(tmp_path)
    payload = _coverage_report()
    entry = payload["files"]["src/sample.py"]
    match case:
        case "counts":
            entry["summary"]["covered_lines"] = 0
        case "missing":
            entry.update(executed_lines=[], missing_lines=[1])
            entry["summary"].update(covered_lines=0, missing_lines=1)
        case "excluded":
            entry["excluded_lines"] = [1]
            entry["summary"]["excluded_lines"] = 1
        case "inventory":
            (root / "src" / "unreported.py").write_text(
                "value = 2\n", encoding="utf-8"
            )
        case "outside":
            payload["files"]["outside.py"] = deepcopy(entry)
        case "duplicate":
            payload["files"][str((root / "src" / "sample.py").resolve())] = (
                deepcopy(entry)
            )
        case "totals":
            payload["totals"]["covered_lines"] = 0
            payload["totals"]["missing_lines"] = 1
        case "schema":
            payload["files"] = []
    if case in {"missing", "excluded"}:
        payload["totals"] = payload["files"]["src/sample.py"]["summary"]
    if case == "json":
        report.write_text('{"files": {}, "files": {}}\n', encoding="utf-8")
    else:
        _write(report, payload)
    _freshen(report, root)

    errors = (
        _VERIFIER.CoverageVerificationError,
        _VERIFIER.StrictJsonError,
    )
    with pytest.raises(errors, match=match):
        _VERIFIER.verify_src_coverage(report, repo_root=root)


@pytest.mark.parametrize(
    "relative",
    (
        "src/sample.py",
        "tests/changed_test.py",
        "scripts/gate.py",
        "Makefile",
        "pyproject.toml",
    ),
)
def test_report_freshness_binds_every_gate_input(
    tmp_path: Path,
    relative: str,
) -> None:
    """Reject a report older than any source, test, script, or gate file."""
    root, report = _repo(tmp_path)
    changed = root / relative
    if not changed.exists():
        changed.parent.mkdir(parents=True, exist_ok=True)
        changed.write_text("# changed gate input\n", encoding="utf-8")
    newer = (
        max(
            report.stat().st_mtime_ns,
            changed.stat().st_mtime_ns,
        )
        + 1
    )
    utime(changed, ns=(newer, newer))

    with pytest.raises(
        _VERIFIER.CoverageVerificationError,
        match="predates current",
    ):
        _VERIFIER.verify_src_coverage(report, repo_root=root)

    _freshen(report, root)
    assert _VERIFIER.verify_src_coverage(report, repo_root=root).files == (
        "src/sample.py",
    )


@pytest.mark.parametrize("constructor", ("exec", "compile"))
def test_prohibited_dynamic_coverage_tricks_fail_closed(
    tmp_path: Path,
    constructor: str,
) -> None:
    """Reject dynamic-code calls in active acceptance tests."""
    root, report = _repo(tmp_path)
    active = root / "tests" / "active_test.py"
    active.write_text(
        f"def test_case() -> None:\n    {constructor}('value = 1')\n",
        encoding="utf-8",
    )
    _write(
        root / "tests" / "fixtures" / "input" / "acceptance_manifest.json",
        {
            "nodes": [
                {
                    "lifecycle": "active",
                    "node_id": "tests/active_test.py::test_case",
                }
            ]
        },
    )
    _freshen(report, root)

    with pytest.raises(
        _VERIFIER.CoverageVerificationError,
        match="prohibited coverage trick",
    ):
        _VERIFIER.verify_src_coverage(report, repo_root=root)

    active.write_text(
        "def test_case() -> None:\n    pass\n",
        encoding="utf-8",
    )
    _freshen(report, root)
    _VERIFIER.verify_src_coverage(report, repo_root=root)


@pytest.mark.parametrize(
    "ledger_case",
    ("valid", "schema", "binding", "delta", "review", "digest"),
)
def test_exact_coverage_allows_only_reviewed_exclusion_relocations(
    tmp_path: Path,
    ledger_case: str,
) -> None:
    """Validate a single direct baseline-to-current relocation ledger."""
    source = tmp_path / "src"
    fixtures = tmp_path / "tests" / "fixtures" / "input"
    source.mkdir(parents=True)
    fixtures.mkdir(parents=True)
    directive = "if False:  # pragma: no cover"
    (source / "sample.py").write_text(
        f"\n{directive}\n    raise AssertionError\n",
        encoding="utf-8",
    )
    baseline_path = fixtures / "coverage_exclusions.json"
    current_path = fixtures / "coverage_exclusions_current.json"
    _write(
        baseline_path,
        _snapshot(
            directives=[
                {"path": "src/sample.py", "line": 1, "text": directive}
            ],
            report_lines={"src/sample.py": [1, 2]},
            digest_field="baseline_sha256",
        ),
    )
    _write(
        current_path,
        _snapshot(
            directives=[
                {"path": "src/sample.py", "line": 2, "text": directive}
            ],
            report_lines={"src/sample.py": [2, 3]},
            digest_field="snapshot_sha256",
        ),
    )
    baseline = _VERIFIER.read_exclusion_snapshot(
        baseline_path,
        tmp_path,
        digest_field="baseline_sha256",
        label="baseline",
    )
    current = _VERIFIER.read_exclusion_snapshot(
        current_path,
        tmp_path,
        digest_field="snapshot_sha256",
        label="current",
    )
    delta = _VERIFIER.exclusion_delta(baseline, current)
    ledger: dict[str, Any] = {
        "schema_version": 2,
        "baseline_snapshot_sha256": baseline.digest,
        "current_snapshot_sha256": current.digest,
        "directives": delta["directives"],
        "report_lines": delta["report_lines"],
        "review": {
            "reviewed_by": "test reviewer",
            "rationale": "The exact directive moved with its source.",
        },
    }
    ledger["ledger_sha256"] = _VERIFIER._digest(ledger)
    ledger_path = fixtures / "coverage_exclusion_relocations_current.json"
    _write(ledger_path, ledger)

    _VERIFIER.verify_exclusion_delta(ledger_path, baseline, current)
    _VERIFIER.verify_observed_exclusions(current, tmp_path)
    assert delta["directives"]["relocated"] == 1
    assert delta["report_lines"]["added"] == 1
    assert delta["report_lines"]["removed"] == 1

    if ledger_case == "valid":
        return
    match ledger_case:
        case "schema":
            ledger["schema_version"] = 1
            match = "schema_version"
        case "binding":
            ledger["baseline_snapshot_sha256"] = "0" * 64
            match = "bind both snapshots"
        case "delta":
            ledger["directives"]["relocated"] = 0
            match = "derived change set"
        case "review":
            ledger["review"]["reviewed_by"] = ""
            match = "non-empty string"
        case "digest":
            match = "ledger digest"
    if ledger_case == "digest":
        ledger["ledger_sha256"] = "0" * 64
    else:
        ledger["ledger_sha256"] = _VERIFIER._digest(
            {
                key: value
                for key, value in ledger.items()
                if key != "ledger_sha256"
            }
        )
    _write(ledger_path, ledger)
    with pytest.raises(
        _VERIFIER.CoverageVerificationError,
        match=match,
    ):
        _VERIFIER.verify_exclusion_delta(ledger_path, baseline, current)


def test_repository_exclusion_evidence_matches_current_source() -> None:
    """Validate the real compact baseline/current/delta evidence."""
    baseline = _VERIFIER.read_exclusion_snapshot(
        _FIXTURES / "coverage_exclusions.json",
        _ROOT,
        digest_field="baseline_sha256",
        label="baseline",
    )
    current = _VERIFIER.read_exclusion_snapshot(
        _FIXTURES / "coverage_exclusions_current.json",
        _ROOT,
        digest_field="snapshot_sha256",
        label="current",
    )

    _VERIFIER.verify_exclusion_delta(
        _FIXTURES / "coverage_exclusion_relocations_current.json",
        baseline,
        current,
    )
    _VERIFIER.verify_observed_exclusions(current, _ROOT)

    assert len(baseline.directives) == 55
    assert len(current.directives) == 67
    assert sum(map(len, current.report_lines.values())) == 1844


def test_coverage_configuration_fails_closed(tmp_path: Path) -> None:
    """Reject hidden coverage configuration outside reviewed fixtures."""
    root, _ = _repo(tmp_path)
    (root / ".coveragerc").write_text("[run]\nomit = src/sample.py\n")

    with pytest.raises(
        _VERIFIER.CoverageVerificationError,
        match="configuration is prohibited",
    ):
        _VERIFIER._verify_coverage_configuration(root)
