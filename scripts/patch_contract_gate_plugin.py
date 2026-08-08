"""Record one complete pytest outcome for the patch contract gate."""

from json import dumps
from os import environ
from pathlib import Path

from pytest import CollectReport, Config, Session

_FACTS_PATH_ENV = "AVALAN_PATCH_CONTRACT_PYTEST_FACTS_PATH"
_LEAK_WARNING_NAMES = frozenset(
    (
        "ResourceWarning",
        "PytestUnraisableExceptionWarning",
        "PytestUnhandledThreadExceptionWarning",
    )
)


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    """Write complete outcome facts when the patch gate requests them."""
    raw_path = environ.get(_FACTS_PATH_ENV)
    if raw_path is None:
        return
    path = _facts_path(raw_path)
    if path.exists():
        raise RuntimeError("patch pytest facts path already exists")
    stats = _terminal_stats(session.config)
    skipped = _reports(stats, "skipped")
    passed = _reports(stats, "passed")
    xfailed = tuple(
        report for report in skipped if getattr(report, "wasxfail", None)
    ) + _reports(stats, "xfailed")
    xpassed = tuple(
        report for report in passed if getattr(report, "wasxfail", None)
    ) + _reports(stats, "xpassed")
    ordinary_skipped = tuple(
        report for report in skipped if not getattr(report, "wasxfail", None)
    )
    collection_skipped = tuple(
        report
        for report in ordinary_skipped
        if isinstance(report, CollectReport)
    )
    warnings = _reports(stats, "warnings")
    facts = {
        "schema_version": 1,
        "collected": session.testscollected,
        "passed": (
            len(passed)
            - len(
                tuple(
                    report
                    for report in passed
                    if getattr(report, "wasxfail", None)
                )
            )
        ),
        "failed": len(_reports(stats, "failed")),
        "errors": len(_reports(stats, "error")),
        "skipped": len(ordinary_skipped),
        "collection_skipped": len(collection_skipped),
        "xfailed": len(xfailed),
        "xpassed": len(xpassed),
        "deselected": len(_reports(stats, "deselected")),
        "warnings": len(warnings),
        "leak_warnings": sum(_is_leak_warning(report) for report in warnings),
        "exitstatus": exitstatus,
    }
    path.write_text(
        dumps(facts, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _facts_path(raw_path: str) -> Path:
    """Return the one regular output path owned by the current checkout."""
    path = Path(raw_path).resolve()
    root = Path.cwd().resolve()
    if path.parent != root or path.name != ".patch-contract-pytest-facts.json":
        raise RuntimeError(
            "patch pytest facts path is outside the checkout root"
        )
    if path.is_symlink():
        raise RuntimeError("patch pytest facts path cannot be a symbolic link")
    return path


def _terminal_stats(config: Config) -> dict[str, list[object]]:
    """Return the terminal reporter's immutable outcome categories."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    raw_stats = getattr(reporter, "stats", None)
    if not isinstance(raw_stats, dict):
        raise RuntimeError("pytest terminal reporter facts are unavailable")
    result: dict[str, list[object]] = {}
    for key, value in raw_stats.items():
        if not isinstance(key, str) or not isinstance(value, list):
            raise RuntimeError("pytest terminal reporter facts are malformed")
        result[key] = value
    return result


def _reports(
    stats: dict[str, list[object]], category: str
) -> tuple[object, ...]:
    """Return one closed terminal-reporter category."""
    return tuple(stats.get(category, ()))


def _is_leak_warning(report: object) -> bool:
    """Return whether one warning records an unclosed runtime resource."""
    category = getattr(report, "category", None)
    name = getattr(category, "__name__", "")
    return isinstance(name, str) and name in _LEAK_WARNING_NAMES
