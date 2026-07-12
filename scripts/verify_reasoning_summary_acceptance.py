#!/usr/bin/env python
"""Collect and execute the exact reasoning-summary acceptance inventory."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from json import JSONDecodeError
from os import environ
from pathlib import Path, PurePosixPath
from subprocess import CompletedProcess, TimeoutExpired, run
from sys import executable, stderr
from typing import cast

from reasoning_summary_json import (
    StrictJsonError,
    strict_json_loads,
)

_COLLECT_SENTINEL = "__REASONING_SUMMARY_COLLECT__"
_EXECUTE_SENTINEL = "__REASONING_SUMMARY_EXECUTE__"
_DISALLOWED_MARKERS = frozenset(("skip", "skipif", "xfail"))
_PROCESS_TIMEOUT_SECONDS = 300
_COLLECTION_PAYLOAD_KEYS = frozenset(
    (
        "exit_code",
        "items",
        "deselected",
        "collection_reports",
    )
)
_EXECUTION_PAYLOAD_KEYS = frozenset(
    (
        "exit_code",
        "items",
        "deselected",
        "collection_reports",
        "reports",
    )
)
_PROBE_DIAGNOSTIC_KEYS = frozenset(("probe_stdout", "probe_stderr"))
_COLLECTION_ITEM_KEYS = frozenset(("nodeid", "markers"))
_COLLECTION_REPORT_KEYS = frozenset(("nodeid", "outcome", "detail"))
_EXECUTION_REPORT_KEYS = frozenset(
    (
        "nodeid",
        "when",
        "outcome",
        "wasxfail",
        "detail",
    )
)

_COLLECT_DRIVER = f"""
from json import dumps
from sys import argv

from pytest import main


class Probe:
    def __init__(self):
        self.items = []
        self.deselected = []
        self.collection_reports = []

    def pytest_collection_finish(self, session):
        self.items = [
            {{
                "nodeid": item.nodeid,
                "markers": sorted(
                    marker.name for marker in item.iter_markers()
                ),
            }}
            for item in session.items
        ]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection_reports.append(
                {{
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "detail": str(report.longrepr),
                }}
            )


probe = Probe()
exit_code = main(
    [
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "anyio.pytest_plugin",
        *argv[1:],
    ],
    plugins=[probe],
)
print(
    "{_COLLECT_SENTINEL}"
    + dumps(
        {{
            "exit_code": int(exit_code),
            "items": probe.items,
            "deselected": probe.deselected,
            "collection_reports": probe.collection_reports,
        }},
        sort_keys=True,
    )
)
"""

_EXECUTE_DRIVER = f"""
from json import dumps
from sys import argv

from pytest import main


class Probe:
    def __init__(self):
        self.items = []
        self.deselected = []
        self.collection_reports = []
        self.reports = []

    def pytest_collection_finish(self, session):
        self.items = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection_reports.append(
                {{
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "detail": str(report.longrepr),
                }}
            )

    def pytest_runtest_logreport(self, report):
        self.reports.append(
            {{
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "wasxfail": str(getattr(report, "wasxfail", "")),
                "detail": (
                    str(report.longrepr)
                    if report.failed or report.skipped
                    else ""
                ),
            }}
        )


probe = Probe()
exit_code = main(
    [
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "anyio.pytest_plugin",
        *argv[1:],
    ],
    plugins=[probe],
)
print(
    "{_EXECUTE_SENTINEL}"
    + dumps(
        {{
            "exit_code": int(exit_code),
            "items": probe.items,
            "deselected": probe.deselected,
            "collection_reports": probe.collection_reports,
            "reports": probe.reports,
        }},
        sort_keys=True,
    )
)
"""


class AcceptanceVerificationError(RuntimeError):
    """Report an incomplete or non-passing acceptance inventory."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceManifest:
    """Store the exact active acceptance node IDs grouped by dimension."""

    path: Path
    active_phase: int
    dimensions: dict[str, tuple[str, ...]]

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return every manifest node ID in declared dimension order."""
        return tuple(
            node_id
            for node_ids in self.dimensions.values()
            for node_id in node_ids
        )


def default_manifest_path() -> Path:
    """Return the checked-in reasoning-summary acceptance manifest path."""
    return (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "reasoning_summary"
        / "acceptance_manifest.json"
    )


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load and strictly validate an acceptance manifest."""
    assert isinstance(path, Path)
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except (StrictJsonError, JSONDecodeError, OSError) as exc:
        raise AcceptanceVerificationError(
            f"cannot read acceptance manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AcceptanceVerificationError(
            "acceptance manifest must be an object"
        )
    if set(payload) != {
        "schema_version",
        "feature",
        "active_phase",
        "dimensions",
    }:
        raise AcceptanceVerificationError(
            "acceptance manifest must contain exactly schema_version, "
            "feature, active_phase, and dimensions"
        )
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise AcceptanceVerificationError(
            "acceptance manifest schema_version must be the integer 1"
        )
    if payload.get("feature") != "reasoning_summary":
        raise AcceptanceVerificationError(
            "acceptance manifest feature must be reasoning_summary"
        )
    active_phase = payload.get("active_phase")
    if type(active_phase) is not int or active_phase < 0 or active_phase > 9:
        raise AcceptanceVerificationError(
            "acceptance manifest active_phase must be an integer from 0 to 9"
        )
    raw_dimensions = payload.get("dimensions")
    if not isinstance(raw_dimensions, dict) or not raw_dimensions:
        raise AcceptanceVerificationError(
            "acceptance manifest dimensions must be a non-empty object"
        )
    dimensions: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for raw_dimension, raw_node_ids in raw_dimensions.items():
        if not isinstance(raw_dimension, str) or not raw_dimension.strip():
            raise AcceptanceVerificationError(
                "acceptance dimension names must be non-empty strings"
            )
        if not isinstance(raw_node_ids, list) or not raw_node_ids:
            raise AcceptanceVerificationError(
                f"acceptance dimension {raw_dimension!r} must not be empty"
            )
        node_ids: list[str] = []
        for raw_node_id in raw_node_ids:
            if (
                not isinstance(raw_node_id, str)
                or "::" not in raw_node_id
                or not raw_node_id.split("::", 1)[0].endswith(".py")
            ):
                raise AcceptanceVerificationError(
                    f"invalid acceptance node ID: {raw_node_id!r}"
                )
            if raw_node_id in seen:
                raise AcceptanceVerificationError(
                    f"duplicate acceptance node ID: {raw_node_id}"
                )
            seen.add(raw_node_id)
            node_ids.append(raw_node_id)
        dimensions[raw_dimension] = tuple(node_ids)
    return AcceptanceManifest(
        path=path,
        active_phase=active_phase,
        dimensions=dimensions,
    )


def verify_acceptance(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> AcceptanceManifest:
    """Collect and execute every exact node in the acceptance manifest."""
    path = manifest_path or default_manifest_path()
    root = repo_root or repository_root()
    manifest = load_manifest(path)
    _validate_execution_scope(manifest, root)
    expected = manifest.node_ids
    collect_payload = _run_probe(
        _COLLECT_DRIVER,
        _COLLECT_SENTINEL,
        expected,
        root,
    )
    _verify_collection(expected, collect_payload)
    execute_payload = _run_probe(
        _EXECUTE_DRIVER,
        _EXECUTE_SENTINEL,
        expected,
        root,
    )
    _verify_execution(expected, execute_payload)
    return manifest


def _run_probe(
    driver: str,
    sentinel: str,
    node_ids: tuple[str, ...],
    repo_root: Path,
) -> dict[str, object]:
    environment = dict(environ)
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_PLUGINS", None)
    environment.pop("PYTHONPATH", None)
    for key in tuple(environment):
        if key.startswith("COV_CORE_") or key == "COVERAGE_PROCESS_START":
            del environment[key]
    completed = run(
        [executable, "-c", driver, *node_ids],
        cwd=repo_root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=_PROCESS_TIMEOUT_SECONDS,
    )
    return _probe_payload(completed, sentinel)


def _probe_payload(
    completed: CompletedProcess[str],
    sentinel: str,
) -> dict[str, object]:
    returncode = completed.returncode
    if type(returncode) is not int or returncode != 0:
        output = completed.stdout + completed.stderr
        raise AcceptanceVerificationError(
            "acceptance probe process exited with code "
            f"{returncode}:\n{output}"
        )
    payload_lines = [
        line.removeprefix(sentinel)
        for line in completed.stdout.splitlines()
        if line.startswith(sentinel)
    ]
    if len(payload_lines) != 1:
        output = completed.stdout + completed.stderr
        raise AcceptanceVerificationError(
            f"acceptance probe did not return one result payload:\n{output}"
        )
    try:
        payload = strict_json_loads(payload_lines[0])
    except (StrictJsonError, JSONDecodeError) as exc:
        raise AcceptanceVerificationError(
            f"acceptance probe returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AcceptanceVerificationError(
            "acceptance probe payload must be an object"
        )
    if sentinel == _COLLECT_SENTINEL:
        expected_keys = _COLLECTION_PAYLOAD_KEYS
    elif sentinel == _EXECUTE_SENTINEL:
        expected_keys = _EXECUTION_PAYLOAD_KEYS
    else:
        raise AcceptanceVerificationError(
            f"unknown acceptance probe sentinel: {sentinel!r}"
        )
    _verify_exact_mapping_keys(payload, expected_keys, "probe payload")
    payload["probe_stdout"] = completed.stdout[-4000:]
    payload["probe_stderr"] = completed.stderr[-4000:]
    return cast(dict[str, object], payload)


def _verify_collection(
    expected: tuple[str, ...], payload: dict[str, object]
) -> None:
    _verify_exact_mapping_keys(
        payload,
        _COLLECTION_PAYLOAD_KEYS | _PROBE_DIAGNOSTIC_KEYS,
        "collection payload",
    )
    _verify_probe_diagnostics(payload)
    _verify_no_collection_failures(payload)
    exit_code = payload.get("exit_code")
    if type(exit_code) is not int or exit_code != 0:
        raise AcceptanceVerificationError(
            f"acceptance collection exited with code {exit_code}: "
            f"{payload.get('probe_stdout', '')}"
            f"{payload.get('probe_stderr', '')}"
        )
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise AcceptanceVerificationError("collection items must be a list")
    collected: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise AcceptanceVerificationError(
                "collected acceptance item must be an object"
            )
        _verify_exact_mapping_keys(
            raw_item,
            _COLLECTION_ITEM_KEYS,
            "collection item",
        )
        node_id = raw_item.get("nodeid")
        markers = raw_item.get("markers")
        if (
            not isinstance(node_id, str)
            or not isinstance(markers, list)
            or not all(isinstance(marker, str) for marker in markers)
        ):
            raise AcceptanceVerificationError(
                "collected acceptance item has invalid fields"
            )
        disallowed = sorted(
            marker
            for marker in markers
            if isinstance(marker, str) and marker in _DISALLOWED_MARKERS
        )
        if disallowed:
            raise AcceptanceVerificationError(
                f"{node_id} has disallowed markers: {disallowed}"
            )
        collected.append(node_id)
    _verify_exact_nodes(expected, tuple(collected), "collected")


def _verify_execution(
    expected: tuple[str, ...], payload: dict[str, object]
) -> None:
    _verify_exact_mapping_keys(
        payload,
        _EXECUTION_PAYLOAD_KEYS | _PROBE_DIAGNOSTIC_KEYS,
        "execution payload",
    )
    _verify_probe_diagnostics(payload)
    _verify_no_collection_failures(payload)
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not all(
        isinstance(item, str) for item in raw_items
    ):
        raise AcceptanceVerificationError("execution items must be strings")
    _verify_exact_nodes(
        expected, tuple(cast(list[str], raw_items)), "executed"
    )
    raw_reports = payload.get("reports")
    if not isinstance(raw_reports, list):
        raise AcceptanceVerificationError("execution reports must be a list")
    reports_by_node: dict[str, list[dict[str, object]]] = {
        node_id: [] for node_id in expected
    }
    for raw_report in raw_reports:
        if not isinstance(raw_report, dict):
            raise AcceptanceVerificationError(
                "acceptance execution report must be an object"
            )
        _verify_exact_mapping_keys(
            raw_report,
            _EXECUTION_REPORT_KEYS,
            "execution report",
        )
        node_id = raw_report.get("nodeid")
        when = raw_report.get("when")
        outcome = raw_report.get("outcome")
        wasxfail = raw_report.get("wasxfail")
        detail = raw_report.get("detail")
        if not isinstance(node_id, str) or node_id not in reports_by_node:
            raise AcceptanceVerificationError(
                f"unexpected acceptance execution report: {node_id!r}"
            )
        if not isinstance(when, str) or when not in {
            "setup",
            "call",
            "teardown",
        }:
            raise AcceptanceVerificationError(
                f"invalid acceptance execution phase: {when!r}"
            )
        if not isinstance(outcome, str) or outcome not in {
            "passed",
            "failed",
            "skipped",
        }:
            raise AcceptanceVerificationError(
                f"invalid acceptance execution outcome: {outcome!r}"
            )
        if not isinstance(wasxfail, str) or not isinstance(detail, str):
            raise AcceptanceVerificationError(
                "acceptance execution report text fields must be strings"
            )
        reports_by_node[node_id].append(cast(dict[str, object], raw_report))
    for node_id, reports in reports_by_node.items():
        phases = [
            cast(str, report["when"])
            for report in reports
            if isinstance(report.get("when"), str)
        ]
        expected_phases = {"setup", "call", "teardown"}
        phase_counts = {
            phase: phases.count(phase) for phase in expected_phases
        }
        if (
            len(reports) != 3
            or set(phases) != expected_phases
            or any(count != 1 for count in phase_counts.values())
        ):
            raise AcceptanceVerificationError(
                f"{node_id} was not exactly once fully executed; "
                f"phase counts: {phase_counts}, observed={phases}"
            )
        for report in reports:
            outcome = report.get("outcome")
            wasxfail = report.get("wasxfail")
            if wasxfail:
                raise AcceptanceVerificationError(
                    f"{node_id} produced an xfail/xpass outcome: {wasxfail}"
                )
            if outcome != "passed":
                detail = report.get("detail")
                raise AcceptanceVerificationError(
                    f"{node_id} {report.get('when')} outcome was "
                    f"{outcome}: {detail}"
                )
    exit_code = payload.get("exit_code")
    if type(exit_code) is not int or exit_code != 0:
        raise AcceptanceVerificationError(
            f"acceptance execution exited with code {exit_code}"
        )


def _verify_no_collection_failures(payload: dict[str, object]) -> None:
    deselected = payload.get("deselected")
    if not isinstance(deselected, list) or not all(
        isinstance(node_id, str) for node_id in deselected
    ):
        raise AcceptanceVerificationError(
            "deselected nodes must be a list of strings"
        )
    if deselected:
        raise AcceptanceVerificationError(
            f"acceptance nodes were deselected: {deselected}"
        )
    collection_reports = payload.get("collection_reports")
    if not isinstance(collection_reports, list):
        raise AcceptanceVerificationError("collection reports must be a list")
    for raw_report in collection_reports:
        if not isinstance(raw_report, dict):
            raise AcceptanceVerificationError(
                "collection report must be an object"
            )
        _verify_exact_mapping_keys(
            raw_report,
            _COLLECTION_REPORT_KEYS,
            "collection report",
        )
        node_id = raw_report.get("nodeid")
        outcome = raw_report.get("outcome")
        detail = raw_report.get("detail")
        if (
            not isinstance(node_id, str)
            or not isinstance(outcome, str)
            or outcome not in {"failed", "skipped"}
            or not isinstance(detail, str)
        ):
            raise AcceptanceVerificationError(
                "collection report fields are invalid"
            )
    if collection_reports:
        raise AcceptanceVerificationError(
            "acceptance collection was skipped or failed: "
            f"{collection_reports}"
        )


def _verify_exact_mapping_keys(
    payload: dict[str, object],
    expected_keys: frozenset[str],
    label: str,
) -> None:
    if set(payload) != expected_keys:
        raise AcceptanceVerificationError(
            f"{label} has invalid keys: expected={sorted(expected_keys)}, "
            f"observed={sorted(payload)}"
        )


def _verify_probe_diagnostics(payload: dict[str, object]) -> None:
    if not isinstance(payload.get("probe_stdout"), str) or not isinstance(
        payload.get("probe_stderr"), str
    ):
        raise AcceptanceVerificationError(
            "acceptance probe diagnostics must be strings"
        )


def _verify_exact_nodes(
    expected: tuple[str, ...], observed: tuple[str, ...], label: str
) -> None:
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    duplicates = sorted(
        node_id for node_id in set(observed) if observed.count(node_id) > 1
    )
    if missing or unexpected or duplicates or len(observed) != len(expected):
        raise AcceptanceVerificationError(
            f"acceptance nodes were not exactly {label}: "
            f"missing={missing}, unexpected={unexpected}, "
            f"duplicates={duplicates}"
        )


def _validate_execution_scope(
    manifest: AcceptanceManifest,
    repo_root: Path,
) -> None:
    root = repo_root.resolve()
    if not root.is_dir():
        raise AcceptanceVerificationError(
            f"acceptance repository root is not a directory: {repo_root}"
        )
    try:
        manifest.path.resolve().relative_to(root)
    except ValueError as exc:
        raise AcceptanceVerificationError(
            "acceptance manifest must be inside the repository root"
        ) from exc
    for node_id in manifest.node_ids:
        raw_path = node_id.split("::", 1)[0]
        node_path = PurePosixPath(raw_path)
        if (
            node_path.is_absolute()
            or ".." in node_path.parts
            or "\\" in raw_path
        ):
            raise AcceptanceVerificationError(
                f"acceptance node path escapes repository root: {raw_path}"
            )
        try:
            (root / Path(*node_path.parts)).resolve().relative_to(root)
        except ValueError as exc:
            raise AcceptanceVerificationError(
                f"acceptance node path escapes repository root: {raw_path}"
            ) from exc


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Collect and execute every exact reasoning-summary acceptance "
            "node without skips, xfails, or deselection."
        )
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line acceptance verification."""
    _parse_args()
    try:
        manifest = verify_acceptance()
    except (AcceptanceVerificationError, TimeoutExpired) as exc:
        print(f"reasoning-summary acceptance failed: {exc}", file=stderr)
        return 1
    print(
        "reasoning-summary acceptance passed: "
        f"phase={manifest.active_phase} nodes={len(manifest.node_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
