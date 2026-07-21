"""Exercise exact source-coverage inventory verification."""

from copy import deepcopy
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_verifier() -> ModuleType:
    """Return the source-coverage verifier module."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    name = "_input_contract_src_coverage"
    spec = spec_from_file_location(
        name, _ROOT / "scripts" / "verify_src_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic JSON document."""
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _summary(
    *, covered: int, excluded: int, missing: int, statements: int
) -> dict[str, int]:
    """Return the exact summary fields consumed by the verifier."""
    return {
        "covered_lines": covered,
        "excluded_lines": excluded,
        "missing_lines": missing,
        "num_statements": statements,
    }


def _synthetic_repository(tmp_path: Path) -> tuple[Path, dict[str, Any], Path]:
    """Create a tiny source tree and exact raw coverage evidence."""
    source = tmp_path / "src" / "package"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "empty.py").write_text("", encoding="utf-8")
    sample = source / "sample.py"
    sample.write_text(
        "def choose(flag: bool) -> int:\n"
        "    if flag:  # pragma: no cover\n"
        "        return 1\n"
        "    return 2\n"
        "\n"
        "# non-statement padding\n",
        encoding="utf-8",
    )
    analyzer = _VERIFIER.Coverage(config_file=False, data_file=None)
    _, statements, excluded, _, _ = analyzer.analysis2(str(sample))
    normalized = "src/package/sample.py"
    executed = sorted(set(statements) | {excluded[0]})
    sample_entry = {
        "executed_lines": executed,
        "excluded_lines": excluded,
        "missing_lines": [],
        "summary": _summary(
            covered=len(statements),
            excluded=len(excluded),
            missing=0,
            statements=len(statements),
        ),
    }
    report = {
        "files": {normalized: sample_entry},
        "totals": deepcopy(sample_entry["summary"]),
    }
    exclusions = [
        {
            "path": normalized,
            "line": 2,
            "text": "    if flag:  # pragma: no cover",
        }
    ]
    digest_payload = {
        "coverage_configuration": "none",
        "exclusions": exclusions,
        "report_excluded_lines": {normalized: excluded},
        "source_root": "src",
    }
    baseline = {
        "schema_version": 1,
        **digest_payload,
        "baseline_sha256": (
            sha256(
                dumps(
                    digest_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        ),
    }
    baseline_path = tmp_path / "coverage_exclusions.json"
    _write_json(baseline_path, baseline)
    return sample, report, baseline_path


def _verify(
    root: Path,
    report: dict[str, Any],
    baseline: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Verify one report with its synthetic baseline digest pinned."""
    baseline_value = loads(baseline.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_EXCLUSION_BASELINE_SHA256",
        baseline_value["baseline_sha256"],
    )
    report_path = root / "coverage.json"
    _write_json(report_path, report)
    return _VERIFIER.verify_src_coverage(
        report_path,
        repo_root=root,
        exclusion_baseline_path=baseline,
    )


def test_exact_coverage_accepts_empty_module_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept normalized control-line overlap and both empty encodings."""
    _, report, baseline = _synthetic_repository(tmp_path)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "unrelated.md").write_text("sentinel\n", encoding="utf-8")
    original_read_text = Path.read_text
    observed_reads: list[Path] = []

    def read_non_markdown(
        target: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        observed_reads.append(target)
        assert target.suffix.lower() != ".md"
        return original_read_text(target, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_non_markdown)
    empty_summary = _summary(covered=0, excluded=0, missing=0, statements=0)
    for executed in ([], [0]):
        candidate = deepcopy(report)
        candidate["files"]["src/package/__init__.py"] = {
            "executed_lines": executed,
            "excluded_lines": [],
            "missing_lines": [],
            "summary": empty_summary,
        }
        candidate["files"]["src/empty.py"] = {
            "executed_lines": executed,
            "excluded_lines": [],
            "missing_lines": [],
            "summary": empty_summary,
        }
        result = _verify(tmp_path, candidate, baseline, monkeypatch)
        assert result.summary.missing_lines == 0
        assert result.summary.covered_lines == result.summary.num_statements
        overlap = set(
            candidate["files"]["src/package/sample.py"]["executed_lines"]
        ) & set(candidate["files"]["src/package/sample.py"]["excluded_lines"])
        assert overlap
    assert observed_reads


def test_exact_coverage_rejects_invalid_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject missing, duplicate, outside, partial, and forged evidence."""
    sample, report, baseline = _synthetic_repository(tmp_path)
    error = _VERIFIER.CoverageVerificationError

    malformed_path = tmp_path / "malformed-coverage.json"
    malformed_path.write_text('{"files":', encoding="utf-8")
    with pytest.raises(error):
        _VERIFIER.verify_src_coverage(
            malformed_path,
            repo_root=tmp_path,
            exclusion_baseline_path=baseline,
        )

    duplicate_key_path = tmp_path / "duplicate-key-coverage.json"
    duplicate_key_path.write_text(
        '{"files":{"src/package/sample.py":{},'
        '"src/package/sample.py":{}},"totals":{}}',
        encoding="utf-8",
    )
    with pytest.raises(error, match="duplicate"):
        _VERIFIER.verify_src_coverage(
            duplicate_key_path,
            repo_root=tmp_path,
            exclusion_baseline_path=baseline,
        )

    missing = deepcopy(report)
    missing["files"] = {}
    missing["totals"] = _summary(
        covered=0, excluded=0, missing=0, statements=0
    )
    with pytest.raises(error, match="inventory mismatch"):
        _verify(tmp_path, missing, baseline, monkeypatch)

    duplicate = deepcopy(report)
    duplicate["files"][str(sample.resolve())] = deepcopy(
        duplicate["files"]["src/package/sample.py"]
    )
    duplicate["totals"] = {
        key: value * 2 for key, value in duplicate["totals"].items()
    }
    with pytest.raises(error, match="duplicate normalized"):
        _verify(tmp_path, duplicate, baseline, monkeypatch)

    outside = deepcopy(report)
    outside["files"]["outside.py"] = outside["files"].pop(
        "src/package/sample.py"
    )
    with pytest.raises(error, match="outside src"):
        _verify(tmp_path, outside, baseline, monkeypatch)

    partial = deepcopy(report)
    entry = partial["files"]["src/package/sample.py"]
    statement = next(
        line
        for line in entry["executed_lines"]
        if line not in entry["excluded_lines"]
    )
    entry["executed_lines"].remove(statement)
    entry["missing_lines"] = [statement]
    entry["summary"]["covered_lines"] -= 1
    entry["summary"]["missing_lines"] = 1
    partial["totals"] = deepcopy(entry["summary"])
    with pytest.raises(error, match="not exact"):
        _verify(tmp_path, partial, baseline, monkeypatch)

    nominal = deepcopy(partial)
    nominal_entry = nominal["files"]["src/package/sample.py"]
    nominal_entry["summary"]["percent_covered"] = 99.999
    nominal["totals"]["percent_covered"] = 99.999
    with pytest.raises(error, match="not exact"):
        _verify(tmp_path, nominal, baseline, monkeypatch)

    inconsistent_totals = deepcopy(report)
    inconsistent_totals["totals"]["covered_lines"] -= 1
    inconsistent_totals["totals"]["missing_lines"] += 1
    with pytest.raises(error, match="totals are inconsistent"):
        _verify(tmp_path, inconsistent_totals, baseline, monkeypatch)

    forged = deepcopy(report)
    forged_entry = forged["files"]["src/package/sample.py"]
    statements = set(forged_entry["executed_lines"])
    excluded = set(forged_entry["excluded_lines"])
    non_statement = next(
        line
        for line in range(
            1, len(sample.read_text(encoding="utf-8").splitlines()) + 1
        )
        if line not in statements | excluded
    )
    forged_entry["executed_lines"].append(non_statement)
    forged_entry["executed_lines"].sort()
    forged_entry["summary"]["covered_lines"] += 1
    forged_entry["summary"]["num_statements"] += 1
    forged["totals"] = deepcopy(forged_entry["summary"])
    with pytest.raises(error, match="inconsistent|inventory changed"):
        _verify(tmp_path, forged, baseline, monkeypatch)

    repeated = deepcopy(report)
    repeated_entry = repeated["files"]["src/package/sample.py"]
    repeated_entry["executed_lines"].append(
        repeated_entry["executed_lines"][0]
    )
    with pytest.raises(error, match="duplicate executed"):
        _verify(tmp_path, repeated, baseline, monkeypatch)
