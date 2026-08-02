#!/usr/bin/env python
"""Verify lifecycle-aware conversation static type contracts."""

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from os import environ, pathsep
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from subprocess import run
from sys import executable, stderr

from contract_gate import (
    ContractGateError,
    StrictJsonError,
    canonical_sha256,
    mapping,
    object_list,
    strict_json_path,
)

_FEATURE = "conversation_continuity"
_MAX_PHASE = 12
_PROHIBITED_SOURCE_PATTERN = compile_regex(
    r"(?:\b" + "A" + r"ny\b|#\s*type:\s*ignore|\b(?:exec|compile)\s*\(|"
    r"#\s*pragma:\s*no\s*cover)"
)
_PHASE0_TYPE_FIXTURE_PAYLOAD_SHA256 = (
    "68d1767472491fe804474e6d9e0532dbca7fa568bb9d64775b3b3d630cce54cb"
)
_PHASE0_TYPE_FIXTURE_INVENTORY = (
    (
        "phase0-contract-positive",
        "positive",
        0,
        "tests/conversation_type_contracts/phase0_positive.py",
    ),
    (
        "phase0-identity-interchange-negative",
        "negative",
        0,
        "tests/conversation_type_contracts/identity_interchange_negative.py",
    ),
    (
        "phase0-storage-axis-negative",
        "negative",
        0,
        "tests/conversation_type_contracts/storage_axis_negative.py",
    ),
)
_TYPE_FIXTURE_PAYLOAD_SHA256_BY_PHASE = {
    0: _PHASE0_TYPE_FIXTURE_PAYLOAD_SHA256,
    1: "3c956beb448e555794ee940c1abb6dac26ea3dbe3b613bfd06aaa808901d7feb",
    2: "332ca232bcd07b433eee48d0a6a79036f7b730fb5405e3dc6458a0717779de74",
    3: "203d278171cd5246cb2b6a018a039ce4871d2920cc65a0f9c88ca724046f9eb9",
    4: "cdfea565d3441c07a7fe02cebf3a61c6690c59a79a1b183d676ceaf752bbe0e5",
    5: "c290884561e74464e74d4ad44c113b97fd7c2094aa5b0b32fc11b0b16f97d719",
}
_TYPE_ACTIVATION_HISTORY_BY_PHASE = {
    0: "fdc3c82ff02fcbfb54491d748f5568a9fb4c4783ec846bdb11bd9e189c809491",
    1: "549af2d3dc75ed1b38f477410671db03a4a55764424f7da78e9b2974d2c0541c",
    2: "b0bc7dc13fa87f53d10523d5565c9d27dcd96004b67b696e38925a734090f2df",
    3: "660f4748cd74131df44512cffe4abd2c94b2f9156bc582abe2bd717a635cf2b9",
    4: "dd5b10c4b3aca6136f9a59f484c3637bc72326aed35237a25dc2c6adde0da899",
    5: "2dd076e03d294e393754c8986e2b28ab86058919b87ac345f818d1ac78e57fbd",
}
_TYPE_REPLACEMENT_HISTORY_BY_PHASE = {
    0: (
        0,
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    1: (
        0,
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    2: (
        1,
        "921a38443dedb906b3f8c0c68b0ac7cb435dd4180ad3d9a540ca0718dea734b5",
    ),
    3: (
        1,
        "921a38443dedb906b3f8c0c68b0ac7cb435dd4180ad3d9a540ca0718dea734b5",
    ),
    4: (
        1,
        "921a38443dedb906b3f8c0c68b0ac7cb435dd4180ad3d9a540ca0718dea734b5",
    ),
    5: (
        1,
        "921a38443dedb906b3f8c0c68b0ac7cb435dd4180ad3d9a540ca0718dea734b5",
    ),
}
_TYPE_SOURCE_SHA256_BY_PHASE = {
    0: {
        "tests/conversation_type_contracts/phase0_positive.py": (
            "1879433067170cd90c801a91c9f79d3a7a6ab8d84f041bf4df03e0d4b0c519b3"
        ),
        "tests/conversation_type_contracts/identity_interchange_negative.py": (
            "ee66184ddc28485cbce7520bef6b4769e16875deceac4f30a25b6872deb51344"
        ),
        "tests/conversation_type_contracts/storage_axis_negative.py": (
            "3ae41df1c8a31798bfce91808a1fcf295f8e3a7e6b0fba11c909be2906f72f5b"
        ),
    },
    1: {
        "tests/conversation_type_contracts/phase1_positive.py": (
            "1222219ebed59ee2d1f97526b16ccb5cdbcde31a07f20dc154247fc3ceee9826"
        ),
        (
            "tests/conversation_type_contracts/"
            + "phase1_identity_interchange_negative.py"
        ): "34a32980eb653003246ad67fa53bb773111af32dda8cf54fcb95b058869e1f42",
        "tests/conversation_type_contracts/phase1_mode_mixing_negative.py": (
            "413efb785be9096d57fe136bf2cf6c536550266fd7d847f9ef640dc90c13ac4c"
        ),
        "tests/conversation_type_contracts/phase1_sync_protocol_negative.py": (
            "5ec7e9ff3a113be71328b03edc2066d4b32d2a055387bcabf3cc78d085d87720"
        ),
        (
            "tests/conversation_type_contracts/"
            + "phase1_untyped_payload_negative.py"
        ): "5eb9f6bc20b1e22d7d55a6f2ab6d765e94c245ab40d6cf51519459d4fcf44290",
        (
            "tests/conversation_type_contracts/"
            + "phase1_mutable_checkpoint_negative.py"
        ): "8a69c01cae054bd2217839ffacadcac856a96db996191526cf094e69b28f18e0",
    },
    2: {
        "tests/conversation_type_contracts/phase2_positive.py": (
            "c08af99e2023fb8599a65661993f01436fd9fce20b797ee5916d446aff7edf61"
        ),
        (
            "tests/conversation_type_contracts/"
            + "phase2_sync_effects_negative.py"
        ): "c21e162a44b273a61531fbafadba9da419f715987bdf2bb0714707feb0211ef9",
    },
    3: {
        "tests/conversation_type_contracts/phase3_positive.py": (
            "b14d8feee7e9068cf26e94a2cf6fa1acfea2eb55a63a86f91861bbaf2a0146b9"
        ),
        (
            "tests/conversation_type_contracts/"
            + "phase3_sync_crypto_negative.py"
        ): "ab3f4ee06fbd0c80c387b825540a596d775b89b37cae6aa7a1289bc6602a7a99",
    },
    4: {
        "tests/conversation_type_contracts/phase4_positive.py": (
            "015ba4958cec9878c1cb8bf86e1d0533de4dcbbf3dd00c2b64ce14b839e15152"
        ),
        (
            "tests/conversation_type_contracts/"
            + "phase4_sync_sdk_negative.py"
        ): "15e2f7ca55f5212bddbde37e2b20868f83790e950639d0192f82d9aecc36d5d6",
    },
    5: {
        "tests/conversation_type_contracts/phase5_positive.py": (
            "2cefa80804de1cf7b6026fb9ab982c0446fd08d77b3c444d9f718db99e6bdc9f"
        ),
        (
            "tests/conversation_type_contracts/"
            + "phase5_sync_provider_negative.py"
        ): "25bdb4902a8fa7f5e46a8b57d266871d74e7a50b0c74170808aaca7fa44ed0ab",
    },
}


class ConversationTypeContractError(RuntimeError):
    """Report an invalid or non-conforming type fixture."""


@dataclass(frozen=True, kw_only=True, slots=True)
class TypeFixture:
    """Store one positive or intentionally rejected static fixture."""

    id: str
    kind: str
    lifecycle: str
    active_from_phase: int
    path: str
    source_sha256: str
    expected_diagnostics: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class TypeReplacement:
    """Store one reviewed append-only static-fixture replacement."""

    phase: int
    old_fixture_id: str
    replacement_fixture_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class TypeContractManifest:
    """Store the validated static fixture inventory."""

    current_phase: int
    fixtures: tuple[TypeFixture, ...]
    replacements: tuple[TypeReplacement, ...]


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def default_manifest_path() -> Path:
    """Return the tracked conversation type-contract manifest."""
    return (
        repository_root()
        / "tests"
        / "fixtures"
        / "conversation"
        / "type_contract_manifest.phase5.json"
    )


def load_manifest(path: Path) -> TypeContractManifest:
    """Load and validate the conversation static fixture inventory."""
    try:
        payload = mapping(strict_json_path(path), "type-contract manifest")
    except (ContractGateError, StrictJsonError) as exc:
        raise ConversationTypeContractError(str(exc)) from exc
    expected = {
        "schema_version",
        "feature",
        "current_phase",
        "activation_history",
        "replacements",
        "fixtures",
        "manifest_sha256",
    }
    if set(payload) != expected:
        raise ConversationTypeContractError(
            "type-contract manifest has invalid keys"
        )
    if payload.get("schema_version") != 1:
        raise ConversationTypeContractError(
            "type-contract schema_version must be 1"
        )
    if payload.get("feature") != _FEATURE:
        raise ConversationTypeContractError(
            f"type-contract feature must be {_FEATURE}"
        )
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    raw_fixtures = object_list(payload.get("fixtures"), "type fixtures")
    if not raw_fixtures:
        raise ConversationTypeContractError(
            "type fixture inventory must be non-empty"
        )
    fixtures = tuple(_type_fixture(raw, current_phase) for raw in raw_fixtures)
    _unique((fixture.id for fixture in fixtures), "type fixture ID")
    _unique((fixture.path for fixture in fixtures), "type fixture path")
    if {fixture.kind for fixture in fixtures} != {"positive", "negative"}:
        raise ConversationTypeContractError(
            "type fixtures need positive and negative cases"
        )
    replacements = _validate_replacements(
        payload.get("replacements"), fixtures, current_phase
    )
    activation_history = _validate_activation_history(
        payload.get("activation_history"), fixtures, current_phase
    )
    _validate_replacement_transitions(
        replacements,
        fixtures,
        activation_history,
    )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if payload.get("manifest_sha256") != canonical_sha256(canonical):
        raise ConversationTypeContractError(
            "type-contract manifest digest is invalid"
        )
    phase0_inventory = tuple(
        (
            fixture.id,
            fixture.kind,
            fixture.active_from_phase,
            fixture.path,
        )
        for fixture in fixtures
        if fixture.active_from_phase == 0
    )
    if phase0_inventory != _PHASE0_TYPE_FIXTURE_INVENTORY:
        raise ConversationTypeContractError(
            "type fixture inventory differs from the Phase 0 anchor"
        )
    _validate_type_fixture_phase_anchors(
        raw_fixtures,
        fixtures,
        current_phase,
    )
    return TypeContractManifest(
        current_phase=current_phase,
        fixtures=fixtures,
        replacements=replacements,
    )


def verify_conversation_types(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
) -> TypeContractManifest:
    """Run strict mypy against every selected active fixture."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if through_phase < 0 or through_phase > manifest.current_phase:
        raise ConversationTypeContractError(
            "through-phase must be implemented by the type manifest"
        )
    selected = tuple(
        fixture
        for fixture in manifest.fixtures
        if fixture.lifecycle == "active"
        and fixture.active_from_phase <= through_phase
    )
    if not selected:
        raise ConversationTypeContractError(
            "the selected type fixture inventory is empty"
        )
    validate_type_source_phase_anchors(manifest, root)
    environment = {
        key: value
        for key, value in environ.items()
        if key.upper() != "PYTHONPATH" and not key.upper().startswith("MYPY")
    }
    mypy_paths = (str(root / "tests"), str(root / "src"))
    environment["MYPYPATH"] = pathsep.join(mypy_paths)
    for fixture in selected:
        fixture_path = _fixture_path(fixture.path, root)
        if not fixture_path.is_file():
            raise ConversationTypeContractError(
                f"active type fixture is missing: {fixture.path}"
            )
        source = fixture_path.read_bytes()
        if sha256(source).hexdigest() != fixture.source_sha256:
            raise ConversationTypeContractError(
                f"type fixture source digest changed: {fixture.id}"
            )
        text = source.decode("utf-8")
        prohibited = _PROHIBITED_SOURCE_PATTERN.search(text)
        if prohibited is not None:
            raise ConversationTypeContractError(
                "type fixture contains a prohibited typing or coverage "
                f"escape: {fixture.id}"
            )
        completed = run(
            (
                executable,
                "-m",
                "mypy",
                "--strict",
                "--show-error-codes",
                "--no-error-summary",
                "--no-pretty",
                fixture.path,
            ),
            cwd=root,
            capture_output=True,
            check=False,
            env=environment,
            text=True,
        )
        output = completed.stdout + completed.stderr
        if fixture.kind == "positive":
            if completed.returncode != 0:
                raise ConversationTypeContractError(
                    f"positive type fixture failed: {fixture.id}\n{output}"
                )
            continue
        if completed.returncode == 0:
            raise ConversationTypeContractError(
                f"negative type fixture unexpectedly passed: {fixture.id}"
            )
        observed = tuple(
            line.strip() for line in output.splitlines() if ": error:" in line
        )
        if observed != fixture.expected_diagnostics:
            raise ConversationTypeContractError(
                "negative type fixture diagnostics changed: "
                f"{fixture.id}, expected={fixture.expected_diagnostics}, "
                f"observed={observed}\n{output}"
            )
    return manifest


def _type_fixture(raw: object, current_phase: int) -> TypeFixture:
    item = mapping(raw, "type fixture")
    expected = {
        "id",
        "kind",
        "lifecycle",
        "active_from_phase",
        "path",
        "source_sha256",
        "expected_diagnostics",
    }
    if set(item) != expected:
        raise ConversationTypeContractError("type fixture has invalid keys")
    identifier = _string(item.get("id"), "type fixture ID")
    kind = _string(item.get("kind"), "type fixture kind")
    if kind not in {"positive", "negative"}:
        raise ConversationTypeContractError(
            f"invalid type fixture kind: {kind}"
        )
    phase = _phase(item.get("active_from_phase"), "type fixture phase")
    lifecycle = _string(item.get("lifecycle"), "type fixture lifecycle")
    if lifecycle not in {"active", "planned", "replaced"} or (
        (phase > current_phase) != (lifecycle == "planned")
    ):
        raise ConversationTypeContractError(
            "type fixture lifecycle disagrees with activation"
        )
    fixture_path = _string(item.get("path"), "type fixture path")
    _validate_fixture_path(fixture_path)
    source_digest = _string(
        item.get("source_sha256"), "type fixture source digest"
    )
    if len(source_digest) != 64 or any(
        char not in "0123456789abcdef" for char in source_digest
    ):
        raise ConversationTypeContractError(
            "type fixture source digest must be lowercase SHA-256"
        )
    diagnostics = _string_list(
        item.get("expected_diagnostics"),
        "expected diagnostics",
        allow_empty=kind == "positive",
    )
    if (kind == "positive" and diagnostics) or (
        kind == "negative" and not diagnostics
    ):
        raise ConversationTypeContractError(
            "type fixture diagnostics disagree with fixture kind"
        )
    return TypeFixture(
        id=identifier,
        kind=kind,
        lifecycle=lifecycle,
        active_from_phase=phase,
        path=fixture_path,
        source_sha256=source_digest,
        expected_diagnostics=diagnostics,
    )


def _validate_activation_history(
    raw: object,
    fixtures: tuple[TypeFixture, ...],
    current_phase: int,
) -> tuple[tuple[str, ...], ...]:
    history = object_list(raw, "type activation history")
    _require_phase_anchor_keys(
        _TYPE_ACTIVATION_HISTORY_BY_PHASE,
        current_phase,
        "type activation history",
    )
    if len(history) != current_phase + 1:
        raise ConversationTypeContractError(
            "type activation history must preserve every implemented phase"
        )
    previous: set[str] = set()
    snapshots: list[tuple[str, ...]] = []
    for expected_phase, raw_entry in enumerate(history):
        entry = mapping(raw_entry, "type activation entry")
        if set(entry) != {"phase", "fixture_ids", "sha256"}:
            raise ConversationTypeContractError(
                "type activation entry has invalid keys"
            )
        if (
            _phase(entry.get("phase"), "type activation phase")
            != expected_phase
        ):
            raise ConversationTypeContractError(
                "type activation phases must be contiguous"
            )
        fixture_ids = _string_list(
            entry.get("fixture_ids"), "type activation fixture IDs"
        )
        _unique(fixture_ids, "type activation fixture ID")
        expected_ids = tuple(
            fixture.id
            for fixture in fixtures
            if fixture.lifecycle in {"active", "replaced"}
            and fixture.active_from_phase <= expected_phase
        )
        if fixture_ids != expected_ids or not previous <= set(fixture_ids):
            raise ConversationTypeContractError(
                "type activation history is not monotonic"
            )
        digest = sha256("\n".join(fixture_ids).encode("utf-8")).hexdigest()
        if entry.get("sha256") != digest:
            raise ConversationTypeContractError(
                "type activation history digest is invalid"
            )
        if digest != _TYPE_ACTIVATION_HISTORY_BY_PHASE[expected_phase]:
            raise ConversationTypeContractError(
                "type activation history differs from its immutable phase "
                f"anchor at phase {expected_phase}"
            )
        previous = set(fixture_ids)
        snapshots.append(fixture_ids)
    return tuple(snapshots)


def _validate_type_fixture_phase_anchors(
    raw_fixtures: list[object],
    fixtures: tuple[TypeFixture, ...],
    current_phase: int,
) -> None:
    """Validate independently anchored type payloads by activation phase."""
    _require_phase_anchor_keys(
        _TYPE_FIXTURE_PAYLOAD_SHA256_BY_PHASE,
        current_phase,
        "type fixture payload",
    )
    for phase in range(current_phase + 1):
        payload = [
            {
                key: value
                for key, value in mapping(raw, "type fixture").items()
                if key != "lifecycle"
            }
            for raw, fixture in zip(raw_fixtures, fixtures, strict=True)
            if fixture.active_from_phase == phase
        ]
        if (
            canonical_sha256(payload)
            != _TYPE_FIXTURE_PAYLOAD_SHA256_BY_PHASE[phase]
        ):
            raise ConversationTypeContractError(
                "type fixture payload differs from its independent phase "
                f"anchor at phase {phase}"
            )


def validate_type_source_phase_anchors(
    manifest: TypeContractManifest,
    root: Path,
) -> None:
    """Validate source digests through append-only per-phase anchors."""
    _require_phase_anchor_keys(
        _TYPE_SOURCE_SHA256_BY_PHASE,
        manifest.current_phase,
        "type source",
    )
    for phase in range(manifest.current_phase + 1):
        observed = {
            fixture.path: fixture.source_sha256
            for fixture in manifest.fixtures
            if fixture.lifecycle in {"active", "replaced"}
            and fixture.active_from_phase == phase
        }
        if observed != _TYPE_SOURCE_SHA256_BY_PHASE[phase]:
            raise ConversationTypeContractError(
                "type source inventory differs from its immutable phase "
                f"anchor at phase {phase}"
            )
        for relative, expected_sha256 in observed.items():
            source = _fixture_path(relative, root)
            if not source.is_file():
                raise ConversationTypeContractError(
                    f"active type fixture is missing: {relative}"
                )
            if sha256(source.read_bytes()).hexdigest() != expected_sha256:
                raise ConversationTypeContractError(
                    f"type fixture source digest changed: {relative}"
                )


def _require_phase_anchor_keys(
    anchors: Mapping[int, object],
    current_phase: int,
    label: str,
) -> None:
    """Require one append-only independent anchor per implemented phase."""
    if not set(range(current_phase + 1)) <= set(anchors):
        raise ConversationTypeContractError(
            f"{label} anchors must cover every implemented phase"
        )


def _validate_replacements(
    raw: object,
    fixtures: tuple[TypeFixture, ...],
    current_phase: int,
) -> tuple[TypeReplacement, ...]:
    replacements = object_list(raw, "type replacements")
    current_ids = {fixture.id for fixture in fixtures}
    parsed: list[TypeReplacement] = []
    old_ids: list[str] = []
    target_ids: list[str] = []
    phases: list[int] = []
    for raw_entry in replacements:
        entry = mapping(raw_entry, "type replacement")
        if set(entry) != {
            "phase",
            "old_fixture_id",
            "replacement_fixture_ids",
            "reviewed_by",
            "evidence",
        }:
            raise ConversationTypeContractError(
                "type replacement has invalid keys"
            )
        phase = _phase(entry.get("phase"), "type replacement phase")
        if phase > current_phase:
            raise ConversationTypeContractError(
                "future type replacement cannot alter current history"
            )
        old = _string(entry.get("old_fixture_id"), "old fixture ID")
        targets = _string_list(
            entry.get("replacement_fixture_ids"), "replacement fixture IDs"
        )
        if (
            old not in current_ids
            or not targets
            or not set(targets) <= current_ids
            or old in targets
        ):
            raise ConversationTypeContractError(
                "type replacement tombstone differs from current inventory"
            )
        old_ids.append(old)
        target_ids.extend(targets)
        phases.append(phase)
        parsed.append(
            TypeReplacement(
                phase=phase,
                old_fixture_id=old,
                replacement_fixture_ids=targets,
            )
        )
        _string(entry.get("reviewed_by"), "type replacement reviewer")
        _string(entry.get("evidence"), "type replacement evidence")
    _unique(old_ids, "replaced type fixture ID")
    _unique(target_ids, "replacement type fixture target")
    _validate_replacement_phase_anchors(
        replacements,
        tuple(phases),
        current_phase,
    )
    return tuple(parsed)


def _validate_replacement_transitions(
    replacements: tuple[TypeReplacement, ...],
    fixtures: tuple[TypeFixture, ...],
    activation_history: tuple[tuple[str, ...], ...],
) -> None:
    """Validate retained type tombstones against adjacent snapshots."""
    fixture_by_id = {fixture.id: fixture for fixture in fixtures}
    replacement_by_old = {
        replacement.old_fixture_id: replacement for replacement in replacements
    }
    replaced_ids = {
        fixture.id for fixture in fixtures if fixture.lifecycle == "replaced"
    }
    if replaced_ids != set(replacement_by_old):
        raise ConversationTypeContractError(
            "replaced type records and reviewed ledger entries differ"
        )
    for replacement in replacements:
        if replacement.phase == 0:
            raise ConversationTypeContractError(
                "type replacements require a preceding phase snapshot"
            )
        old = fixture_by_id[replacement.old_fixture_id]
        previous = set(activation_history[replacement.phase - 1])
        current = set(activation_history[replacement.phase])
        additions = current - previous
        if (
            old.lifecycle != "replaced"
            or old.active_from_phase >= replacement.phase
            or replacement.old_fixture_id not in previous
        ):
            raise ConversationTypeContractError(
                "type replacement old record is not a retained prior "
                "snapshot member"
            )
        for target_id in replacement.replacement_fixture_ids:
            target = fixture_by_id[target_id]
            if (
                target.active_from_phase != replacement.phase
                or target.lifecycle not in {"active", "replaced"}
                or target_id not in additions
                or target.kind != old.kind
            ):
                raise ConversationTypeContractError(
                    "type replacement targets must be new same-phase records "
                    "with preserved fixture kind"
                )


def _validate_replacement_phase_anchors(
    replacements: list[object],
    phases: tuple[int, ...],
    current_phase: int,
) -> None:
    """Validate cumulative append-only type replacement history."""
    _require_phase_anchor_keys(
        _TYPE_REPLACEMENT_HISTORY_BY_PHASE,
        current_phase,
        "type replacement history",
    )
    previous_count = 0
    for phase in range(current_phase + 1):
        count, expected_sha256 = _TYPE_REPLACEMENT_HISTORY_BY_PHASE[phase]
        if (
            count < previous_count
            or count > len(replacements)
            or any(value > phase for value in phases[:count])
            or any(value <= phase for value in phases[count:])
        ):
            raise ConversationTypeContractError(
                "type replacement history anchors are not append-only"
            )
        if canonical_sha256(replacements[:count]) != expected_sha256:
            raise ConversationTypeContractError(
                "type replacement history differs from its immutable phase "
                f"anchor at phase {phase}"
            )
        previous_count = count
    if previous_count != len(replacements):
        raise ConversationTypeContractError(
            "type replacement history has unanchored appended payload"
        )


def _fixture_path(raw: str, root: Path) -> Path:
    path = (root / Path(*PurePosixPath(raw).parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConversationTypeContractError(
            f"type fixture escapes repository root: {raw}"
        ) from exc
    return path


def _validate_fixture_path(raw: str) -> None:
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in raw
        or path.parts[:2] != ("tests", "conversation_type_contracts")
        or path.suffix != ".py"
    ):
        raise ConversationTypeContractError(
            f"type fixture is outside its tracked directory: {raw}"
        )


def _phase(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > _MAX_PHASE:
        raise ConversationTypeContractError(
            f"{label} must be an integer from 0 through {_MAX_PHASE}"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationTypeContractError(f"{label} must be non-empty")
    return value


def _string_list(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    values = object_list(value, label)
    if not allow_empty and not values:
        raise ConversationTypeContractError(f"{label} must be non-empty")
    return tuple(_string(item, label) for item in values)


def _unique(values: Iterable[str], label: str) -> None:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ConversationTypeContractError(f"duplicate {label}")


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Verify active conversation static type contracts."
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    return parser.parse_args()


def main() -> int:
    """Run conversation static type verification from the command line."""
    args = _parse_args()
    try:
        manifest = verify_conversation_types(
            args.manifest,
            repo_root=args.repo_root,
            through_phase=args.through_phase,
        )
    except (
        ContractGateError,
        ConversationTypeContractError,
        StrictJsonError,
    ) as exc:
        print(f"conversation type contract failed: {exc}", file=stderr)
        return 1
    active = sum(
        fixture.lifecycle == "active"
        and fixture.active_from_phase <= args.through_phase
        for fixture in manifest.fixtures
    )
    print(
        "conversation type contract passed: "
        f"through_phase={args.through_phase} fixtures={active}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
