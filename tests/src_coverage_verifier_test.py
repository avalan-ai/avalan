"""Exercise exact source-coverage inventory verification."""

from copy import deepcopy
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NO_COVER_DIRECTIVE = "# pragma:" + " no cover"


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


def _canonical_digest(value: object) -> str:
    """Return the verifier's canonical JSON digest."""
    return sha256(
        dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _snapshot_digest_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return the digest-bearing fields shared by exclusion snapshots."""
    return {
        "coverage_configuration": snapshot["coverage_configuration"],
        "exclusions": snapshot["exclusions"],
        "report_excluded_lines": snapshot["report_excluded_lines"],
        "source_root": snapshot["source_root"],
    }


def _set_snapshot_digest(snapshot: dict[str, Any], field: str) -> None:
    """Refresh one synthetic exclusion snapshot digest."""
    snapshot[field] = _canonical_digest(_snapshot_digest_payload(snapshot))


def _set_ledger_digest(ledger: dict[str, Any]) -> None:
    """Refresh one synthetic exclusion relocation ledger digest."""
    payload = {
        key: value for key, value in ledger.items() if key != "ledger_sha256"
    }
    ledger["ledger_sha256"] = _canonical_digest(payload)


def _synthetic_repository(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], Path, Path, Path]:
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
    snapshot_payload = {
        "coverage_configuration": "none",
        "exclusions": exclusions,
        "report_excluded_lines": {normalized: excluded},
        "source_root": "src",
    }
    baseline = {
        "schema_version": 1,
        **deepcopy(snapshot_payload),
    }
    _set_snapshot_digest(baseline, "baseline_sha256")
    current = {
        "schema_version": 1,
        **deepcopy(snapshot_payload),
    }
    _set_snapshot_digest(current, "snapshot_sha256")
    ledger = {
        "schema_version": 1,
        "baseline_snapshot_sha256": baseline["baseline_sha256"],
        "current_snapshot_sha256": current["snapshot_sha256"],
        "directive_count_before": 1,
        "directive_count_after": 1,
        "report_excluded_line_count_before": len(excluded),
        "report_excluded_line_count_after": len(excluded),
        "directive_relocations": [],
        "report_exclusion_relocations": [],
    }
    _set_ledger_digest(ledger)
    baseline_path = tmp_path / "coverage_exclusions.json"
    current_path = tmp_path / "coverage_exclusions_phase1.json"
    relocation_path = tmp_path / "coverage_exclusion_relocations_phase1.json"
    _write_json(baseline_path, baseline)
    _write_json(current_path, current)
    _write_json(relocation_path, ledger)
    return sample, report, baseline_path, current_path, relocation_path


def _verify(
    root: Path,
    report: dict[str, Any],
    baseline: Path,
    current: Path,
    relocations: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Verify one report with its synthetic baseline digest pinned."""
    baseline_value = loads(baseline.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_EXCLUSION_BASELINE_SHA256",
        baseline_value["baseline_sha256"],
    )
    current_value = loads(current.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_EXCLUSION_CURRENT_SHA256",
        current_value["snapshot_sha256"],
    )
    relocation_value = loads(relocations.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        _VERIFIER,
        "_EXPECTED_EXCLUSION_RELOCATION_SHA256",
        relocation_value["ledger_sha256"],
    )
    report_path = root / "coverage.json"
    _write_json(report_path, report)
    return _VERIFIER.verify_src_coverage(
        report_path,
        repo_root=root,
        exclusion_baseline_path=baseline,
        exclusion_current_path=current,
        exclusion_relocation_path=relocations,
    )


def test_exact_coverage_accepts_empty_module_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept normalized control-line overlap and both empty encodings."""
    _, report, baseline, current, relocations = _synthetic_repository(tmp_path)
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
        result = _verify(
            tmp_path,
            candidate,
            baseline,
            current,
            relocations,
            monkeypatch,
        )
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
    sample, report, baseline, current, relocations = _synthetic_repository(
        tmp_path
    )
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
        _verify(
            tmp_path,
            missing,
            baseline,
            current,
            relocations,
            monkeypatch,
        )

    duplicate = deepcopy(report)
    duplicate["files"][str(sample.resolve())] = deepcopy(
        duplicate["files"]["src/package/sample.py"]
    )
    duplicate["totals"] = {
        key: value * 2 for key, value in duplicate["totals"].items()
    }
    with pytest.raises(error, match="duplicate normalized"):
        _verify(
            tmp_path,
            duplicate,
            baseline,
            current,
            relocations,
            monkeypatch,
        )

    outside = deepcopy(report)
    outside["files"]["outside.py"] = outside["files"].pop(
        "src/package/sample.py"
    )
    with pytest.raises(error, match="outside src"):
        _verify(
            tmp_path,
            outside,
            baseline,
            current,
            relocations,
            monkeypatch,
        )

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
        _verify(
            tmp_path,
            partial,
            baseline,
            current,
            relocations,
            monkeypatch,
        )

    nominal = deepcopy(partial)
    nominal_entry = nominal["files"]["src/package/sample.py"]
    nominal_entry["summary"]["percent_covered"] = 99.999
    nominal["totals"]["percent_covered"] = 99.999
    with pytest.raises(error, match="not exact"):
        _verify(
            tmp_path,
            nominal,
            baseline,
            current,
            relocations,
            monkeypatch,
        )

    inconsistent_totals = deepcopy(report)
    inconsistent_totals["totals"]["covered_lines"] -= 1
    inconsistent_totals["totals"]["missing_lines"] += 1
    with pytest.raises(error, match="totals are inconsistent"):
        _verify(
            tmp_path,
            inconsistent_totals,
            baseline,
            current,
            relocations,
            monkeypatch,
        )

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
        _verify(
            tmp_path,
            forged,
            baseline,
            current,
            relocations,
            monkeypatch,
        )

    repeated = deepcopy(report)
    repeated_entry = repeated["files"]["src/package/sample.py"]
    repeated_entry["executed_lines"].append(
        repeated_entry["executed_lines"][0]
    )
    with pytest.raises(error, match="duplicate executed"):
        _verify(
            tmp_path,
            repeated,
            baseline,
            current,
            relocations,
            monkeypatch,
        )


def test_exact_coverage_allows_only_reviewed_exclusion_relocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow reviewed moves and reject every directive inventory drift."""
    sample, _, baseline_path, current_path, relocation_path = (
        _synthetic_repository(tmp_path)
    )
    baseline = loads(baseline_path.read_text(encoding="utf-8"))
    current = loads(current_path.read_text(encoding="utf-8"))
    ledger = loads(relocation_path.read_text(encoding="utf-8"))
    sample.write_text(
        "def choose(flag: bool) -> int:\n"
        "\n"
        f"    if flag:  {_NO_COVER_DIRECTIVE}\n"
        "        return 1\n"
        "    return 2\n"
        "\n"
        "# non-statement padding\n",
        encoding="utf-8",
    )
    analyzer = _VERIFIER.Coverage(config_file=False, data_file=None)
    _, _, current_excluded, _, _ = analyzer.analysis2(str(sample))
    normalized = "src/package/sample.py"
    text = f"    if flag:  {_NO_COVER_DIRECTIVE}"
    identity = _VERIFIER._directive_identity(normalized, text, 1)
    current["exclusions"][0]["line"] = 3
    current["report_excluded_lines"][normalized] = current_excluded
    _set_snapshot_digest(current, "snapshot_sha256")
    ledger.update(
        {
            "current_snapshot_sha256": current["snapshot_sha256"],
            "directive_relocations": [
                {
                    "identity": identity,
                    "path": normalized,
                    "text": text,
                    "occurrence": 1,
                    "from_line": 2,
                    "to_line": 3,
                    "reviewed_by": _VERIFIER._EXPECTED_EXCLUSION_REVIEWER,
                    "reason": (
                        "A synthetic padding line moved this unchanged"
                        " exclusion."
                    ),
                }
            ],
            "report_exclusion_relocations": [
                {
                    "path": normalized,
                    "from_lines": baseline["report_excluded_lines"][
                        normalized
                    ],
                    "to_lines": current_excluded,
                    "reviewed_by": _VERIFIER._EXPECTED_EXCLUSION_REVIEWER,
                    "reason": (
                        "A synthetic padding line moved the unchanged parser"
                        " exclusions."
                    ),
                }
            ],
        }
    )
    _set_ledger_digest(ledger)
    valid_current = deepcopy(current)
    valid_ledger = deepcopy(ledger)

    def verify_case(
        candidate_current: dict[str, Any],
        candidate_ledger: dict[str, Any],
    ) -> dict[str, tuple[int, ...]]:
        _set_snapshot_digest(candidate_current, "snapshot_sha256")
        candidate_ledger["current_snapshot_sha256"] = candidate_current[
            "snapshot_sha256"
        ]
        candidate_ledger["directive_count_after"] = len(
            candidate_current["exclusions"]
        )
        candidate_ledger["report_excluded_line_count_after"] = sum(
            len(lines)
            for lines in candidate_current["report_excluded_lines"].values()
        )
        _set_ledger_digest(candidate_ledger)
        _write_json(current_path, candidate_current)
        _write_json(relocation_path, candidate_ledger)
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_BASELINE_SHA256",
            baseline["baseline_sha256"],
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_CURRENT_SHA256",
            candidate_current["snapshot_sha256"],
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_RELOCATION_SHA256",
            candidate_ledger["ledger_sha256"],
        )
        return cast(
            dict[str, tuple[int, ...]],
            _VERIFIER._verify_exclusion_history(
                baseline_path,
                current_path,
                relocation_path,
                tmp_path,
                tmp_path / "src",
            ),
        )

    assert verify_case(deepcopy(valid_current), deepcopy(valid_ledger))[
        normalized
    ] == tuple(current_excluded)

    for mutation, message in (
        ("added", "changed exclusion counts"),
        ("removed", "changed exclusion counts"),
        ("changed", "identity, text, or count changed"),
        ("duplicated", "snapshot contains duplicates"),
        ("unreviewed", "relocation is unreviewed"),
        ("unledgered", "missing, extra, or unreviewed"),
        ("unreviewed_report", "relocation is unreviewed"),
    ):
        candidate_current = deepcopy(valid_current)
        candidate_ledger = deepcopy(valid_ledger)
        if mutation == "added":
            candidate_current["exclusions"].append(
                {
                    "path": normalized,
                    "line": 5,
                    "text": f"    return 2  {_NO_COVER_DIRECTIVE}",
                }
            )
        elif mutation == "removed":
            candidate_current["exclusions"] = []
        elif mutation == "changed":
            candidate_current["exclusions"][0]["text"] += " - changed"
        elif mutation == "duplicated":
            candidate_current["exclusions"].append(
                deepcopy(candidate_current["exclusions"][0])
            )
        elif mutation == "unreviewed":
            candidate_ledger["directive_relocations"][0][
                "reviewed_by"
            ] = "pending"
        elif mutation == "unledgered":
            candidate_ledger["directive_relocations"] = []
        else:
            candidate_ledger["report_exclusion_relocations"][0][
                "reviewed_by"
            ] = "pending"
        with pytest.raises(
            _VERIFIER.CoverageVerificationError,
            match=message,
        ):
            verify_case(candidate_current, candidate_ledger)

    added_source = sample.parent / "protocol.py"
    added_source.write_text(
        "from typing import Protocol\n"
        "\n"
        "class AddedProtocol(Protocol):\n"
        "    def call(self) -> None:\n"
        "        ...\n",
        encoding="utf-8",
    )
    _, _, added_excluded, _, _ = analyzer.analysis2(str(added_source))
    added_normalized = "src/package/protocol.py"
    next_current = deepcopy(valid_current)
    next_current["report_excluded_lines"][added_normalized] = added_excluded
    _set_snapshot_digest(next_current, "snapshot_sha256")
    next_ledger = {
        "schema_version": 2,
        "baseline_snapshot_sha256": valid_current["snapshot_sha256"],
        "current_snapshot_sha256": next_current["snapshot_sha256"],
        "directive_count_before": len(valid_current["exclusions"]),
        "directive_count_after": len(next_current["exclusions"]),
        "report_excluded_line_count_before": sum(
            len(lines)
            for lines in valid_current["report_excluded_lines"].values()
        ),
        "report_excluded_line_count_after": sum(
            len(lines)
            for lines in next_current["report_excluded_lines"].values()
        ),
        "directive_relocations": [],
        "report_exclusion_relocations": [],
        "report_exclusion_additions": [
            {
                "path": added_normalized,
                "lines": added_excluded,
                "reviewed_by": _VERIFIER._EXPECTED_EXCLUSION_REVIEWER,
                "reason": (
                    "A synthetic protocol adds reviewed parser exclusion"
                    " evidence."
                ),
            }
        ],
    }
    _set_ledger_digest(next_ledger)
    next_current_path = tmp_path / "coverage_exclusions_phase2.json"
    next_ledger_path = tmp_path / "coverage_exclusion_relocations_phase2.json"

    def verify_addition(
        candidate_current: dict[str, Any],
        candidate_ledger: dict[str, Any],
    ) -> dict[str, tuple[int, ...]]:
        _set_snapshot_digest(candidate_current, "snapshot_sha256")
        candidate_ledger["current_snapshot_sha256"] = candidate_current[
            "snapshot_sha256"
        ]
        candidate_ledger["directive_count_after"] = len(
            candidate_current["exclusions"]
        )
        candidate_ledger["report_excluded_line_count_after"] = sum(
            len(lines)
            for lines in candidate_current["report_excluded_lines"].values()
        )
        _set_ledger_digest(candidate_ledger)
        _write_json(current_path, valid_current)
        _write_json(relocation_path, valid_ledger)
        _write_json(next_current_path, candidate_current)
        _write_json(next_ledger_path, candidate_ledger)
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_BASELINE_SHA256",
            baseline["baseline_sha256"],
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_PHASE1_SHA256",
            valid_current["snapshot_sha256"],
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_PHASE1_RELOCATION_SHA256",
            valid_ledger["ledger_sha256"],
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_PHASE1_REVIEWER",
            _VERIFIER._EXPECTED_EXCLUSION_REVIEWER,
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_CURRENT_SHA256",
            candidate_current["snapshot_sha256"],
        )
        monkeypatch.setattr(
            _VERIFIER,
            "_EXPECTED_EXCLUSION_RELOCATION_SHA256",
            candidate_ledger["ledger_sha256"],
        )
        return cast(
            dict[str, tuple[int, ...]],
            _VERIFIER._verify_exclusion_history_chain(
                baseline_path,
                current_path,
                relocation_path,
                next_current_path,
                next_ledger_path,
                tmp_path,
                tmp_path / "src",
            ),
        )

    assert verify_addition(
        deepcopy(next_current),
        deepcopy(next_ledger),
    )[
        added_normalized
    ] == tuple(added_excluded)

    for mutation, message in (
        ("unledgered", "changed exclusion counts"),
        ("unreviewed", "relocation is unreviewed"),
        ("duplicated", "addition is duplicated"),
        ("existing_path", "missing, extra, or unreviewed"),
        ("removed_path", "changed exclusion counts"),
    ):
        candidate_current = deepcopy(next_current)
        candidate_ledger = deepcopy(next_ledger)
        if mutation == "unledgered":
            candidate_ledger["report_exclusion_additions"] = []
        elif mutation == "unreviewed":
            candidate_ledger["report_exclusion_additions"][0][
                "reviewed_by"
            ] = "pending"
        elif mutation == "duplicated":
            candidate_ledger["report_exclusion_additions"].append(
                deepcopy(candidate_ledger["report_exclusion_additions"][0])
            )
        elif mutation == "existing_path":
            candidate_ledger["report_exclusion_additions"][0][
                "path"
            ] = normalized
        else:
            del candidate_current["report_excluded_lines"][normalized]
        with pytest.raises(
            _VERIFIER.CoverageVerificationError,
            match=message,
        ):
            verify_addition(candidate_current, candidate_ledger)

    verify_addition(deepcopy(next_current), deepcopy(next_ledger))
    rewritten_prior = deepcopy(valid_current)
    rewritten_prior["exclusions"][0]["line"] += 1
    _set_snapshot_digest(rewritten_prior, "snapshot_sha256")
    _write_json(current_path, rewritten_prior)
    with pytest.raises(
        _VERIFIER.CoverageVerificationError,
        match="prior snapshot digest changed",
    ):
        _VERIFIER._verify_exclusion_history_chain(
            baseline_path,
            current_path,
            relocation_path,
            next_current_path,
            next_ledger_path,
            tmp_path,
            tmp_path / "src",
        )
