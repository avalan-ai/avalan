#!/usr/bin/env python
"""Verify lifecycle-aware structured-input type-contract fixtures."""

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from os import environ
from pathlib import Path, PurePosixPath
from subprocess import run
from sys import executable, stderr
from typing import cast

from input_contract_json import StrictJsonError, strict_json_path

_FEATURE = "structured_task_input"
_MAX_PHASE = 12
_EXPECTED_TYPE_LEDGER_SHA256 = (
    "3aade03590126acfd31df623804a768dd1f7b95c89743765b3c3a9aab86435f1"
)


class TypeContractVerificationError(RuntimeError):
    """Report an invalid or non-conforming type-contract fixture."""


@dataclass(frozen=True, kw_only=True, slots=True)
class TypeFixture:
    """Store one positive or negative static type fixture."""

    id: str
    kind: str
    lifecycle: str
    active_from_phase: int
    path: str
    expected_diagnostics: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class TypeContractManifest:
    """Store the validated static type fixture inventory."""

    current_phase: int
    fixtures: tuple[TypeFixture, ...]


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def default_manifest_path() -> Path:
    """Return the tracked type-contract manifest path."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "input"
        / "type_contract_manifest.json"
    )


def load_manifest(path: Path) -> TypeContractManifest:
    """Load and validate the static type fixture inventory."""
    try:
        raw = strict_json_path(path)
    except StrictJsonError as exc:
        raise TypeContractVerificationError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise TypeContractVerificationError(
            "type-contract manifest must be an object"
        )
    payload = cast(dict[str, object], raw)
    expected_keys = {
        "schema_version",
        "feature",
        "current_phase",
        "activation_history",
        "activation_snapshots",
        "replacements",
        "fixtures",
    }
    if set(payload) != expected_keys:
        raise TypeContractVerificationError(
            "type-contract manifest has invalid keys"
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise TypeContractVerificationError(
            "type-contract schema_version must be the integer 1"
        )
    if payload.get("feature") != _FEATURE:
        raise TypeContractVerificationError(
            f"type-contract feature must be {_FEATURE}"
        )
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise TypeContractVerificationError(
            "type-contract fixtures must be a non-empty list"
        )
    fixtures = tuple(
        _type_fixture(item, current_phase) for item in raw_fixtures
    )
    _require_unique((item.id for item in fixtures), "fixture ID")
    _require_unique((item.path for item in fixtures), "fixture path")
    _activation_history(
        payload.get("activation_history"),
        fixtures,
        current_phase,
    )
    _activation_snapshots(
        payload.get("activation_snapshots"),
        payload.get("replacements"),
        fixtures,
        current_phase,
    )
    return TypeContractManifest(
        current_phase=current_phase,
        fixtures=fixtures,
    )


def verify_input_types(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
    acceptance_manifest_path: Path | None = None,
) -> TypeContractManifest:
    """Run mypy against every active fixture through the requested gate."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    acceptance_path = acceptance_manifest_path or path.with_name(
        "acceptance_manifest.json"
    )
    acceptance_phase = _acceptance_current_phase(acceptance_path)
    if acceptance_phase != manifest.current_phase:
        raise TypeContractVerificationError(
            "type and acceptance manifests must implement the same phase"
        )
    if through_phase < 0 or through_phase > manifest.current_phase:
        raise TypeContractVerificationError(
            "through-phase must be implemented by the current manifest"
        )
    selected = tuple(
        fixture
        for fixture in manifest.fixtures
        if fixture.lifecycle == "active"
        and fixture.active_from_phase <= through_phase
    )
    if not selected:
        raise TypeContractVerificationError(
            "the selected type-contract inventory has no active fixtures"
        )
    environment = {
        key: value
        for key, value in environ.items()
        if key.upper() != "PYTHONPATH" and not key.upper().startswith("MYPY")
    }
    environment["MYPYPATH"] = str(root / "tests")
    for fixture in selected:
        path = _fixture_path(fixture.path, root)
        if not path.is_file():
            raise TypeContractVerificationError(
                f"active type fixture does not exist: {fixture.path}"
            )
        completed = run(
            [
                executable,
                "-m",
                "mypy",
                "--strict",
                "--show-error-codes",
                "--no-error-summary",
                fixture.path,
            ],
            cwd=root,
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )
        output = completed.stdout + completed.stderr
        if fixture.kind == "positive":
            if completed.returncode != 0:
                raise TypeContractVerificationError(
                    f"positive type fixture failed: {fixture.id}\n{output}"
                )
            continue
        if completed.returncode == 0:
            raise TypeContractVerificationError(
                f"negative type fixture unexpectedly passed: {fixture.id}"
            )
        observed = tuple(
            line.strip() for line in output.splitlines() if ": error:" in line
        )
        if observed != fixture.expected_diagnostics:
            raise TypeContractVerificationError(
                "negative type fixture diagnostics changed: "
                f"{fixture.id}, expected={fixture.expected_diagnostics}, "
                f"observed={observed}\n{output}"
            )
    return manifest


def _type_fixture(raw: object, current_phase: int) -> TypeFixture:
    if not isinstance(raw, dict):
        raise TypeContractVerificationError("type fixture must be an object")
    item = cast(dict[str, object], raw)
    expected_keys = {
        "id",
        "kind",
        "lifecycle",
        "active_from_phase",
        "path",
        "expected_diagnostics",
    }
    if set(item) != expected_keys:
        raise TypeContractVerificationError("type fixture has invalid keys")
    identifier = _string(item.get("id"), "fixture id")
    kind = _string(item.get("kind"), "fixture kind")
    if kind not in {"positive", "negative"}:
        raise TypeContractVerificationError(
            f"invalid type fixture kind: {kind}"
        )
    active_from_phase = _phase(
        item.get("active_from_phase"),
        "active_from_phase",
    )
    lifecycle = _string(item.get("lifecycle"), "fixture lifecycle")
    expected_lifecycle = (
        "active" if active_from_phase <= current_phase else "planned"
    )
    if lifecycle != expected_lifecycle:
        raise TypeContractVerificationError(
            f"type fixture lifecycle regression for {identifier}"
        )
    path = _string(item.get("path"), "fixture path")
    _validate_manifest_path(path)
    raw_diagnostics = item.get("expected_diagnostics")
    if not isinstance(raw_diagnostics, list) or not all(
        isinstance(value, str) and value for value in raw_diagnostics
    ):
        raise TypeContractVerificationError(
            "expected diagnostics must be a string list"
        )
    diagnostics = tuple(cast(list[str], raw_diagnostics))
    if kind == "positive" and diagnostics:
        raise TypeContractVerificationError(
            "positive type fixtures cannot expect diagnostics"
        )
    if kind == "negative" and not diagnostics:
        raise TypeContractVerificationError(
            "negative type fixtures must expect diagnostics"
        )
    return TypeFixture(
        id=identifier,
        kind=kind,
        lifecycle=lifecycle,
        active_from_phase=active_from_phase,
        path=path,
        expected_diagnostics=diagnostics,
    )


def _fixture_path(raw: str, root: Path) -> Path:
    posix = PurePosixPath(raw)
    if posix.is_absolute() or ".." in posix.parts or "\\" in raw:
        raise TypeContractVerificationError(
            f"type fixture escapes repository root: {raw}"
        )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TypeContractVerificationError(
            f"type fixture escapes repository root: {raw}"
        ) from exc
    return path


def _validate_manifest_path(raw: str) -> None:
    posix = PurePosixPath(raw)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or "\\" in raw
        or len(posix.parts) < 3
        or posix.parts[:2] != ("tests", "input_type_contracts")
        or not raw.endswith(".py")
    ):
        raise TypeContractVerificationError(
            f"type fixture path is outside its tracked directory: {raw}"
        )


def _activation_history(
    raw: object,
    fixtures: tuple[TypeFixture, ...],
    current_phase: int,
) -> None:
    if not isinstance(raw, list) or len(raw) != current_phase + 1:
        raise TypeContractVerificationError(
            "type activation history must contain every implemented phase"
        )
    observed: list[str] = []
    for expected_phase, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"phase", "fixture_ids"}:
            raise TypeContractVerificationError(
                "type activation history entry has invalid shape"
            )
        phase = _phase(item.get("phase"), "activation phase")
        if phase != expected_phase:
            raise TypeContractVerificationError(
                "type activation history phases must be contiguous"
            )
        raw_ids = item.get("fixture_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(value, str) and value for value in raw_ids
        ):
            raise TypeContractVerificationError(
                "type activation fixture_ids must be a string list"
            )
        fixture_ids = tuple(cast(list[str], raw_ids))
        _require_unique(fixture_ids, f"activation fixture ID at phase {phase}")
        expected_ids = tuple(
            fixture.id
            for fixture in fixtures
            if fixture.active_from_phase == phase
        )
        if set(fixture_ids) != set(expected_ids) or len(fixture_ids) != len(
            expected_ids
        ):
            raise TypeContractVerificationError(
                f"type activation history mismatch at phase {phase}"
            )
        observed.extend(fixture_ids)
    active_ids = tuple(
        fixture.id for fixture in fixtures if fixture.lifecycle == "active"
    )
    if set(observed) != set(active_ids) or len(observed) != len(active_ids):
        raise TypeContractVerificationError(
            "type activation history does not preserve active fixtures"
        )


def _acceptance_current_phase(path: Path) -> int:
    try:
        raw = strict_json_path(path)
    except StrictJsonError as exc:
        raise TypeContractVerificationError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise TypeContractVerificationError(
            "acceptance manifest must be an object"
        )
    return _phase(raw.get("current_phase"), "acceptance current_phase")


def _activation_snapshots(
    raw_snapshots: object,
    raw_replacements: object,
    fixtures: tuple[TypeFixture, ...],
    current_phase: int,
) -> None:
    if (
        not isinstance(raw_snapshots, list)
        or len(raw_snapshots) != current_phase + 1
    ):
        raise TypeContractVerificationError(
            "type activation snapshots must preserve every implemented phase"
        )
    if not isinstance(raw_replacements, list):
        raise TypeContractVerificationError("type replacements must be a list")
    fixture_by_id = {fixture.id: fixture for fixture in fixtures}
    replacements: dict[str, tuple[str, ...]] = {}
    replacement_phases: dict[str, int] = {}
    replacement_targets: set[str] = set()
    for raw in raw_replacements:
        if not isinstance(raw, dict) or set(raw) != {
            "phase",
            "old_fixture_id",
            "replacement_fixture_ids",
            "reviewed_by",
            "evidence",
        }:
            raise TypeContractVerificationError(
                "type replacement has invalid shape"
            )
        phase = _phase(raw.get("phase"), "type replacement phase")
        if phase > current_phase:
            raise TypeContractVerificationError(
                "type replacement phase is not implemented"
            )
        old_fixture_id = _string(raw.get("old_fixture_id"), "old fixture id")
        if old_fixture_id in replacements:
            raise TypeContractVerificationError(
                f"type fixture is replaced more than once: {old_fixture_id}"
            )
        replacement_ids = raw.get("replacement_fixture_ids")
        if (
            not isinstance(replacement_ids, list)
            or not replacement_ids
            or not all(
                isinstance(value, str) and value for value in replacement_ids
            )
        ):
            raise TypeContractVerificationError(
                "replacement fixture IDs must be a non-empty string list"
            )
        typed_replacement_ids = tuple(cast(list[str], replacement_ids))
        _require_unique(
            typed_replacement_ids,
            "replacement fixture ID",
        )
        for fixture_id in typed_replacement_ids:
            if fixture_id in replacement_targets:
                raise TypeContractVerificationError(
                    f"replacement type target is reused: {fixture_id}"
                )
            replacement_targets.add(fixture_id)
        _string(raw.get("reviewed_by"), "replacement reviewed_by")
        _string(raw.get("evidence"), "replacement evidence")
        replacements[old_fixture_id] = typed_replacement_ids
        replacement_phases[old_fixture_id] = phase
    ledger_digest = sha256(
        dumps(
            {
                "activation_snapshots": raw_snapshots,
                "replacements": raw_replacements,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if ledger_digest != _EXPECTED_TYPE_LEDGER_SHA256:
        raise TypeContractVerificationError(
            "type activation ledger changed without verifier review"
        )
    snapshots: list[tuple[str, ...]] = []
    for expected_phase, raw in enumerate(raw_snapshots):
        if not isinstance(raw, dict) or set(raw) != {
            "phase",
            "fixture_ids",
            "sha256",
        }:
            raise TypeContractVerificationError(
                "type activation snapshot has invalid shape"
            )
        if _phase(raw.get("phase"), "type snapshot phase") != expected_phase:
            raise TypeContractVerificationError(
                "type activation snapshot phases must be contiguous"
            )
        raw_ids = raw.get("fixture_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or not all(isinstance(value, str) and value for value in raw_ids)
        ):
            raise TypeContractVerificationError(
                "type snapshot fixture IDs must be a non-empty string list"
            )
        fixture_ids = tuple(cast(list[str], raw_ids))
        _require_unique(
            fixture_ids, f"type snapshot fixture ID at phase {expected_phase}"
        )
        digest = _string(raw.get("sha256"), "type snapshot SHA-256")
        calculated = sha256("\n".join(fixture_ids).encode("utf-8")).hexdigest()
        if digest != calculated:
            raise TypeContractVerificationError(
                f"type snapshot digest mismatch at phase {expected_phase}"
            )
        snapshots.append(fixture_ids)

    replacements_by_phase = {
        phase: {
            old_fixture_id
            for old_fixture_id, replacement_phase in replacement_phases.items()
            if replacement_phase == phase
        }
        for phase in range(current_phase + 1)
    }
    targets_by_phase = {
        phase: {
            target
            for old_fixture_id in replacements_by_phase[phase]
            for target in replacements[old_fixture_id]
        }
        for phase in range(current_phase + 1)
    }
    previous: set[str] = set()
    all_snapshot_ids: set[str] = set()
    for phase, snapshot in enumerate(snapshots):
        current = set(snapshot)
        added = current - previous
        removed = previous - current
        expected_removed = replacements_by_phase[phase]
        if removed != expected_removed:
            raise TypeContractVerificationError(
                "type snapshot removals lack exact reviewed tombstones at"
                f" phase {phase}: expected={sorted(expected_removed)},"
                f" observed={sorted(removed)}"
            )
        missing_targets = targets_by_phase[phase] - added
        if missing_targets:
            raise TypeContractVerificationError(
                "replacement type targets are not same-phase additions:"
                f" phase={phase}, missing={sorted(missing_targets)}"
            )
        expected_current_additions = {
            fixture.id
            for fixture in fixtures
            if fixture.lifecycle == "active"
            and fixture.active_from_phase == phase
        }
        missing_current = expected_current_additions - added
        if missing_current:
            raise TypeContractVerificationError(
                "active type fixtures are absent from their snapshot:"
                f" phase={phase}, missing={sorted(missing_current)}"
            )
        for fixture_id in added:
            fixture = fixture_by_id.get(fixture_id)
            if fixture is not None:
                if (
                    fixture.lifecycle != "active"
                    or fixture.active_from_phase != phase
                ):
                    raise TypeContractVerificationError(
                        "type fixture was added outside its activation phase:"
                        f" {fixture_id}"
                    )
            elif fixture_id not in replacements:
                raise TypeContractVerificationError(
                    "historical type fixture lacks a later tombstone:"
                    f" {fixture_id}"
                )
        previous = current
        all_snapshot_ids.update(current)

    historical_only = all_snapshot_ids - set(fixture_by_id)
    if not historical_only <= set(replacements):
        raise TypeContractVerificationError(
            "historical type fixtures lack reviewed tombstones"
        )
    for old_fixture_id, replacement_ids in replacements.items():
        phase = replacement_phases[old_fixture_id]
        if phase == 0 or old_fixture_id not in set(snapshots[phase - 1]):
            raise TypeContractVerificationError(
                "replacement type fixture was not active immediately before"
                f" its tombstone: {old_fixture_id}"
            )
        if old_fixture_id in set(snapshots[phase]):
            raise TypeContractVerificationError(
                f"replacement type fixture remains active: {old_fixture_id}"
            )
        for target_id in replacement_ids:
            target = fixture_by_id.get(target_id)
            if target is not None:
                if target.active_from_phase != phase:
                    raise TypeContractVerificationError(
                        "replacement type target activated in another phase:"
                        f" {target_id}"
                    )
                continue
            target_phase = replacement_phases.get(target_id)
            if target_phase is None or target_phase <= phase:
                raise TypeContractVerificationError(
                    "type replacement chain is cyclic or lacks a later"
                    f" tombstone: {target_id}"
                )

    active = tuple(
        fixture.id for fixture in fixtures if fixture.lifecycle == "active"
    )
    if snapshots[-1] != active:
        raise TypeContractVerificationError(
            "latest type activation snapshot differs from active inventory"
        )


def _phase(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > _MAX_PHASE:
        raise TypeContractVerificationError(
            f"{label} must be an integer from 0 to {_MAX_PHASE}"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeContractVerificationError(
            f"{label} must be a non-empty string"
        )
    return value


def _require_unique(values: Iterable[str], label: str) -> None:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise TypeContractVerificationError(f"duplicate {label}")


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Verify active structured-input static type contracts."
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    return parser.parse_args()


def main() -> int:
    """Run static type-contract verification from the command line."""
    args = _parse_args()
    try:
        manifest = verify_input_types(
            args.manifest,
            repo_root=args.repo_root,
            through_phase=args.through_phase,
        )
    except TypeContractVerificationError as exc:
        print(f"structured-input type contract failed: {exc}", file=stderr)
        return 1
    active = sum(
        fixture.lifecycle == "active"
        and fixture.active_from_phase <= args.through_phase
        for fixture in manifest.fixtures
    )
    print(
        "structured-input type contract passed: "
        f"through_phase={args.through_phase} fixtures={active}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
