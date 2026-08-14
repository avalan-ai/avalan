"""Serve the immutable sandbox mutation worker protocol.

This module intentionally imports neither the public SDK/tool layer nor any
workspace code.  The runtime loads it from a digest-verified private bundle.
"""

from base64 import b64decode
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest, digest
from json import dumps, loads
from os import environ, getpid
from pathlib import Path
from sys import stdin, stdout
from typing import NewType, TypedDict

from avalan.patch.domain import (
    AlgorithmDigest,
    ByteSize,
    Capability,
    FileMode,
    LogicalPath,
    MetadataProfile,
    PatchLineageId,
    PatchPlanId,
    ProposedBytes,
    SourceBytes,
)
from avalan.patch.planner import (
    Match,
    MatchKind,
    PlannedFile,
    PlannedLineage,
    TextSpan,
)
from avalan.patch.rooted_worker import (
    FileIdentity,
    RootedInspectionProfile,
    RootedMutationCommand,
    RootedMutationProfile,
    RootWitness,
    TargetErrorCode,
    TargetInspectionError,
    _commit_rooted,
    capture_rooted_root,
    inspect_rooted,
    probe_rooted_metadata,
    rooted_snapshot_payload,
)
from avalan.patch.sandbox_wire import canonical_sandbox_plan_bytes

_MESSAGE_VERSION = 2
_MAX_MESSAGE_BYTES = 1_048_576
_ExecutionPlanFingerprint = NewType("_ExecutionPlanFingerprint", str)


class _RuntimeMessage(TypedDict):
    """Encode one authenticated runtime message."""

    payload: Mapping[str, object]
    mac: str


class _RuntimeRequestPayload(TypedDict):
    """Store one validated host request."""

    version: int
    sequence: int
    kind: str
    receipt: str
    identity: dict[str, str]
    channel_id: str
    implementation_id: str
    body: Mapping[str, object]


class _RuntimeResponsePayload(TypedDict):
    """Store one authenticated worker response."""

    version: int
    sequence: int
    receipt: str
    identity: dict[str, str]
    channel_id: str
    implementation_id: str
    body: Mapping[str, object]
    error: str | None


class _RuntimeChildConfig(TypedDict):
    """Store immutable worker session configuration."""

    root: str
    namespace: str
    cwd: str | None
    maximum: int
    aggregate_maximum: int
    token: str
    receipt: str
    identity: dict[str, str]
    channel_id: str
    implementation_id: str
    implementation_digest: str
    source_digest: str
    implementation_root: str
    read_canary: str
    session_id: str
    execution_plan: _ExecutionPlanFingerprint
    backend: str
    workspace_view: str
    private_view: str
    context_lifetime: str
    protocol: str
    persistent_lease: str
    filesystem: str
    mount: str


def main() -> int:
    """Serve a strict sequence of authenticated runtime messages."""
    encoded = environ.get("AVALAN_SANDBOX_PATCH_SESSION")
    if encoded is None:
        return 2
    try:
        config_value = loads(b64decode(encoded, validate=True))
        config = _child_config(config_value)
        token = bytes.fromhex(config["token"])
        if (
            len(token) != 32
            or _implementation_digest(Path(config["implementation_root"]))
            != config["implementation_digest"]
            or _worker_source_digest(
                Path(config["implementation_root"]) / "avalan"
            )
            != config["source_digest"]
        ):
            return 2
        root = capture_rooted_root(Path(config["root"]))
    except (OSError, TypeError, ValueError, TargetInspectionError):
        return 2
    sequence = 0
    while True:
        line = stdin.buffer.readline(_MAX_MESSAGE_BYTES + 1)
        if not line or len(line) > _MAX_MESSAGE_BYTES:
            return 2
        try:
            request = _child_request(line, token, config, sequence + 1)
            sequence += 1
            body, should_close = _child_dispatch(
                request["kind"], request["body"], config, root, request, token
            )
            response = _child_response(request, body, None, token)
        except TargetInspectionError as exc:
            response = _child_response_from_line(line, exc.code, token)
            should_close = True
        except (OSError, TypeError, ValueError, KeyError):
            return 2
        stdout.buffer.write(
            dumps(response, separators=(",", ":")).encode() + b"\n"
        )
        stdout.buffer.flush()
        if should_close:
            return 0


def _implementation_digest(root: Path) -> str:
    """Return a stable digest over every regular implementation file."""
    if not root.is_dir():
        raise ValueError
    files = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise ValueError
    digest_value = sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest_value.update(len(relative).to_bytes(4, "big"))
        digest_value.update(relative)
        digest_value.update(len(payload).to_bytes(8, "big"))
        digest_value.update(payload)
    return digest_value.hexdigest()


def _worker_source_digest(source_package: Path) -> str:
    """Hash the exact immutable worker source imported by this child."""
    if not source_package.is_dir():
        raise ValueError
    files = tuple(
        path
        for path in sorted(source_package.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    )
    if not files:
        raise ValueError
    digest_value = sha256()
    for path in files:
        relative = (
            Path("avalan") / path.relative_to(source_package)
        ).as_posix()
        relative_bytes = relative.encode()
        payload = path.read_bytes()
        digest_value.update(len(relative_bytes).to_bytes(4, "big"))
        digest_value.update(relative_bytes)
        digest_value.update(len(payload).to_bytes(8, "big"))
        digest_value.update(payload)
    return digest_value.hexdigest()


def _root_payload(root: RootWitness) -> Mapping[str, object]:
    """Encode one non-path root witness."""
    return {
        "device": root.identity.device,
        "inode": root.identity.inode,
        "mount": root.mount_id,
        "filesystem": root.filesystem_id,
    }


def _root_from_payload(value: object) -> RootWitness:
    """Decode one exact root witness."""
    if not isinstance(value, dict) or set(value) != {
        "device",
        "inode",
        "mount",
        "filesystem",
    }:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    device = value["device"]
    inode = value["inode"]
    mount = value["mount"]
    filesystem = value["filesystem"]
    if (
        type(device) is not int
        or type(inode) is not int
        or not isinstance(mount, str)
        or not isinstance(filesystem, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return RootWitness(FileIdentity(device, inode), mount, filesystem)


def _child_dispatch(
    kind: str,
    body: Mapping[str, object],
    config: _RuntimeChildConfig,
    root: RootWitness,
    request: _RuntimeRequestPayload,
    token: bytes,
) -> tuple[Mapping[str, object], bool]:
    """Execute only closed operations against the initial selected root."""
    profile = RootedInspectionProfile(
        Path(config["root"]),
        LogicalPath(config["cwd"]) if config["cwd"] is not None else None,
        config["maximum"],
        config["aggregate_maximum"],
    )
    if kind == "witness":
        if body:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return {"root": _root_payload(root)}, False
    if kind == "canary":
        if body:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        try:
            Path(config["read_canary"]).read_bytes()
        except OSError:
            denied = True
        else:
            denied = False
        try:
            metadata_probe = probe_rooted_metadata(Path(config["namespace"]))
        except OSError as exc:
            raise TargetInspectionError(
                TargetErrorCode.CAPABILITY_UNAVAILABLE
            ) from exc
        return {
            "pid": getpid(),
            "outside_read_denied": denied,
            "metadata_probe": metadata_probe,
        }, False
    if kind == "inspect":
        if set(body) != {"paths", "root"}:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        paths = body["paths"]
        expected_root = _root_from_payload(body["root"])
        if (
            expected_root != root
            or not isinstance(paths, list)
            or any(not isinstance(path, str) for path in paths)
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        snapshots = inspect_rooted(
            profile, tuple(LogicalPath(path) for path in paths), root
        )
        return {
            "snapshots": [rooted_snapshot_payload(item) for item in snapshots]
        }, False
    if kind == "commit":
        command = _mutation_command(body, config, root)
        mutation = RootedMutationProfile(
            Path(config["root"]),
            profile.cwd,
            FileMode(0o644),
        )
        fence = _FenceChecker(request, config, token)
        report = _commit_rooted(command, mutation, root, fence.check)
        if report.journal is None:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return {
            "steps": [
                {
                    "id": item.identifier.value,
                    "lineage": item.lineage.value,
                    "state": item.state.value,
                }
                for item in report.journal.steps
            ],
            "artifacts": [
                {"id": item.identifier, "state": item.state.value}
                for item in report.journal.artifacts
            ],
            "postcondition": report.journal.postcondition.value,
        }, False
    if kind == "close" and not body:
        return {}, True
    raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)


@dataclass(slots=True)
class _FenceChecker:
    """Require a fresh authenticated owner-fence permit per effect."""

    request: _RuntimeRequestPayload
    config: _RuntimeChildConfig
    token: bytes
    effect: int = 0

    def check(self) -> None:
        """Block the next effect until the trusted host validates its fence."""
        self.effect += 1
        response = _child_response(
            self.request,
            {"control": "fence", "effect": self.effect},
            None,
            self.token,
        )
        stdout.buffer.write(
            dumps(response, separators=(",", ":")).encode() + b"\n"
        )
        stdout.buffer.flush()
        line = stdin.buffer.readline(_MAX_MESSAGE_BYTES + 1)
        if not line or len(line) > _MAX_MESSAGE_BYTES:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        permit = _child_request(
            line, self.token, self.config, self.request["sequence"]
        )
        body = permit["body"]
        if (
            permit["kind"] != "fence_permit"
            or set(body) != {"effect", "allowed"}
            or body["effect"] != self.effect
            or type(body["allowed"]) is not bool
            or not body["allowed"]
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)


def _mutation_command(
    value: Mapping[str, object],
    config: _RuntimeChildConfig,
    root: RootWitness,
) -> RootedMutationCommand:
    """Decode and validate the complete canonical mutation transaction."""
    if set(value) != {"schema", "plan", "command", "scope", "runtime"}:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    if value["schema"] != "sandbox-patch-command-v1":
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    plan = _mapping(
        value["plan"],
        {
            "id",
            "fingerprint",
            "canonical",
            "sealed_fingerprint",
            "sealed_canonical",
            "request",
            "subject",
            "context",
            "target",
            "cwd",
            "request_digest",
            "authorized_effects",
            "lineages",
            "final_files",
            "diff",
            "review",
        },
    )
    command = _mapping(
        value["command"], {"domain", "request", "fence", "footprint"}
    )
    scope = _mapping(value["scope"], {"target", "cwd", "root"})
    runtime = _mapping(
        value["runtime"],
        {
            "backend",
            "execution_plan",
            "workspace_view",
            "private_view",
            "channel",
            "protocol",
            "implementation",
            "implementation_digest",
            "receipt",
            "session",
            "context_lifetime",
            "persistent_lease",
            "filesystem",
            "mount",
        },
    )
    expected_runtime = {
        "backend": config["backend"],
        "execution_plan": config["execution_plan"],
        "workspace_view": config["workspace_view"],
        "private_view": config["private_view"],
        "channel": config["channel_id"],
        "protocol": config["protocol"],
        "implementation": config["implementation_id"],
        "implementation_digest": config["implementation_digest"],
        "receipt": config["receipt"],
        "session": config["session_id"],
        "context_lifetime": config["context_lifetime"],
        "persistent_lease": config["persistent_lease"],
        "filesystem": config["filesystem"],
        "mount": config["mount"],
    }
    request = _mapping(
        plan["request"],
        {
            "schema",
            "id",
            "execution",
            "operation",
            "input_digest",
            "paths",
        },
    )
    subject = _mapping(
        plan["subject"],
        {"principal", "tenant", "run", "session", "task", "agent"},
    )
    del subject
    target = _string_mapping(plan["target"])
    scope_target = _string_mapping(scope["target"])
    if (
        runtime != expected_runtime
        or target != config["identity"]
        or scope_target != config["identity"]
        or scope["root"] != _root_payload(root)
        or scope["cwd"] != plan["cwd"]
        or scope["cwd"] != config["cwd"]
        or plan["context"] != "sandbox"
        or command["domain"] != target["domain"]
        or command["request"] != request["id"]
        or type(command["fence"]) is not int
        or command["fence"] <= 0
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    footprint = command["footprint"]
    if (
        not isinstance(footprint, list)
        or not footprint
        or any(not isinstance(item, str) or not item for item in footprint)
        or footprint[0] != "workspace"
        or footprint[1:] != sorted(footprint[1:])
        or len(set(footprint)) != len(footprint)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    canonical = _b64(plan["canonical"])
    fingerprint = _b64(plan["fingerprint"])
    wire_fields = {
        key: item
        for key, item in plan.items()
        if key not in {"canonical", "fingerprint"}
    }
    sealed_canonical = _b64(plan["sealed_canonical"])
    sealed_fingerprint = _b64(plan["sealed_fingerprint"])
    if (
        len(fingerprint) != 32
        or canonical_sandbox_plan_bytes(wire_fields) != canonical
        or sha256(canonical).digest() != fingerprint
        or len(sealed_fingerprint) != 32
        or sha256(sealed_canonical).digest() != sealed_fingerprint
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    AlgorithmDigest("sha256", _string(plan["request_digest"]))
    AlgorithmDigest("sha256", _string(request["input_digest"]))
    if (
        request["schema"] != 1
        or request["operation"] not in {"edit", "apply"}
        or not isinstance(request["paths"], list)
        or not request["paths"]
        or any(not isinstance(item, str) for item in request["paths"])
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    diff = _mapping(plan["diff"], {"entries", "rendered", "digest"})
    rendered = _b64(diff["rendered"])
    diff_digest = AlgorithmDigest("sha256", _string(diff["digest"]))
    entries = diff["entries"]
    if diff_digest != AlgorithmDigest.from_bytes(rendered) or not isinstance(
        entries, list
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    tuple(_b64(item) for item in entries)
    review = _mapping(plan["review"], {"expiry", "diff_digest"})
    if (
        type(review["expiry"]) is not int
        or review["expiry"] <= 0
        or review["diff_digest"] != diff_digest.value
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    lineages_value = plan["lineages"]
    final_files_value = plan["final_files"]
    effects_value = plan["authorized_effects"]
    if (
        not isinstance(lineages_value, list)
        or not lineages_value
        or not isinstance(final_files_value, list)
        or not isinstance(effects_value, list)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    lineages = tuple(_lineage(item) for item in lineages_value)
    final_files = tuple(_planned_file(item) for item in final_files_value)
    try:
        effects = frozenset(
            Capability(_string(item)) for item in effects_value
        )
        plan_id = PatchPlanId(_string(plan["id"]))
    except ValueError as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc
    final_paths = {item.path for item in final_files}
    if (
        len(final_paths) != len(final_files)
        or len({item.lineage_id for item in lineages}) != len(lineages)
        or effects_value != sorted(set(_string_list(effects_value)))
        or any(
            not lineage.capabilities.issubset(effects) for lineage in lineages
        )
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return RootedMutationCommand(plan_id, lineages, effects)


def _lineage(value: object) -> PlannedLineage:
    """Decode every field of one immutable target lineage."""
    item = _mapping(
        value,
        {
            "id",
            "initial",
            "final",
            "source",
            "destination",
            "capabilities",
            "matches",
            "parents",
            "mounts",
            "locks",
            "atomicity",
            "steps",
            "staging",
            "diff",
            "parent_identities",
        },
    )
    capabilities = _string_list(item["capabilities"])
    matches_value = item["matches"]
    parent_identities_value = item["parent_identities"]
    if not isinstance(matches_value, list) or not isinstance(
        parent_identities_value, list
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    matches = tuple(_match(value) for value in matches_value)
    parent_identities: list[tuple[LogicalPath | None, tuple[int, int]]] = []
    for identity in parent_identities_value:
        if (
            not isinstance(identity, list)
            or len(identity) != 3
            or identity[0] is not None
            and not isinstance(identity[0], str)
            or type(identity[1]) is not int
            or type(identity[2]) is not int
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        parent_identities.append(
            (
                None if identity[0] is None else LogicalPath(identity[0]),
                (identity[1], identity[2]),
            )
        )
    try:
        return PlannedLineage(
            PatchLineageId(_string(item["id"])),
            _planned_file(item["initial"]),
            _planned_file(item["final"]),
            _optional_path(item["source"]),
            _optional_path(item["destination"]),
            frozenset(Capability(value) for value in capabilities),
            matches,
            tuple(
                LogicalPath(value) for value in _string_list(item["parents"])
            ),
            tuple(_string_list(item["mounts"])),
            tuple(LogicalPath(value) for value in _string_list(item["locks"])),
            _string(item["atomicity"]),
            tuple(_string_list(item["steps"])),
            _string(item["staging"]),
            _b64(item["diff"]),
            tuple(parent_identities),
        )
    except ValueError as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc


def _planned_file(value: object) -> PlannedFile:
    """Decode and integrity-check one complete regular-file fact."""
    item = _mapping(
        value,
        {
            "path",
            "present",
            "content_kind",
            "content",
            "metadata",
            "digest",
            "size",
            "identity",
            "protected_metadata",
        },
    )
    present = item["present"]
    size = item["size"]
    content: SourceBytes | ProposedBytes | None
    metadata: MetadataProfile | None
    digest_value: AlgorithmDigest | None
    if type(present) is not bool or type(size) is not int:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    if present:
        content_value = _b64(item["content"])
        kind = item["content_kind"]
        if kind == "source":
            content = SourceBytes(content_value)
        elif kind == "proposed":
            content = ProposedBytes(content_value)
        else:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        metadata_value = _mapping(item["metadata"], {"mode", "bom", "newline"})
        if type(metadata_value["bom"]) is not bool:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        metadata = MetadataProfile(
            FileMode(_integer(metadata_value["mode"])),
            metadata_value["bom"],
            _string(metadata_value["newline"]),
        )
        digest_value = AlgorithmDigest("sha256", _string(item["digest"]))
        if digest_value != content.digest() or size != len(content_value):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    else:
        if (
            any(
                item[name] is not None
                for name in ("content_kind", "content", "metadata", "digest")
            )
            or size != 0
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        content = None
        metadata = None
        digest_value = None
    identity_value = item["identity"]
    identity: tuple[int, int] | None
    if identity_value is None:
        identity = None
    elif (
        isinstance(identity_value, list)
        and len(identity_value) == 2
        and all(type(value) is int for value in identity_value)
    ):
        identity = (identity_value[0], identity_value[1])
    else:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    protected = item["protected_metadata"]
    return PlannedFile(
        LogicalPath(_string(item["path"])),
        present,
        content,
        metadata,
        digest_value,
        ByteSize(size),
        identity,
        (
            None
            if protected is None
            else AlgorithmDigest("sha256", _string(protected))
        ),
    )


def _match(value: object) -> Match:
    """Decode one exact source match span."""
    item = _mapping(
        value,
        {"kind", "logical_start", "logical_end", "byte_start", "byte_end"},
    )
    return Match(
        MatchKind(_string(item["kind"])),
        TextSpan(
            _integer(item["logical_start"]),
            _integer(item["logical_end"]),
            _integer(item["byte_start"]),
            _integer(item["byte_end"]),
        ),
    )


def _mapping(value: object, keys: set[str]) -> Mapping[str, object]:
    """Require one mapping with exactly the named fields."""
    if not isinstance(value, dict) or set(value) != keys:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return value


def _string_mapping(value: object) -> dict[str, str]:
    """Require a string-only mapping."""
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or not item
        for key, item in value.items()
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return dict(value)


def _string_list(value: object) -> list[str]:
    """Require a list containing only strings."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return value


def _string(value: object) -> str:
    """Require one nonempty string."""
    if not isinstance(value, str) or not value:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return value


def _integer(value: object) -> int:
    """Require one nonnegative integer."""
    if type(value) is not int or value < 0:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return value


def _b64(value: object) -> bytes:
    """Decode one canonical Base64 field."""
    try:
        return b64decode(_string(value), validate=True)
    except ValueError as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc


def _optional_path(value: object) -> LogicalPath | None:
    """Decode one optional logical path."""
    return None if value is None else LogicalPath(_string(value))


def _child_config(value: object) -> _RuntimeChildConfig:
    """Decode the exact immutable worker configuration shape."""
    keys = {
        "root",
        "namespace",
        "cwd",
        "maximum",
        "aggregate_maximum",
        "token",
        "receipt",
        "identity",
        "channel_id",
        "implementation_id",
        "implementation_digest",
        "source_digest",
        "implementation_root",
        "read_canary",
        "session_id",
        "execution_plan",
        "backend",
        "workspace_view",
        "private_view",
        "context_lifetime",
        "protocol",
        "persistent_lease",
        "filesystem",
        "mount",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError
    cwd_value = value["cwd"]
    if cwd_value is not None and not isinstance(cwd_value, str):
        raise ValueError
    try:
        return {
            "root": _string(value["root"]),
            "namespace": _string(value["namespace"]),
            "cwd": cwd_value,
            "maximum": _integer(value["maximum"]),
            "aggregate_maximum": _integer(value["aggregate_maximum"]),
            "token": _string(value["token"]),
            "receipt": _string(value["receipt"]),
            "identity": _string_mapping(value["identity"]),
            "channel_id": _string(value["channel_id"]),
            "implementation_id": _string(value["implementation_id"]),
            "implementation_digest": _string(value["implementation_digest"]),
            "source_digest": _string(value["source_digest"]),
            "implementation_root": _string(value["implementation_root"]),
            "read_canary": _string(value["read_canary"]),
            "session_id": _string(value["session_id"]),
            "execution_plan": _ExecutionPlanFingerprint(
                _string(value["execution_plan"])
            ),
            "backend": _string(value["backend"]),
            "workspace_view": _string(value["workspace_view"]),
            "private_view": _string(value["private_view"]),
            "context_lifetime": _string(value["context_lifetime"]),
            "protocol": _string(value["protocol"]),
            "persistent_lease": _string(value["persistent_lease"]),
            "filesystem": _string(value["filesystem"]),
            "mount": _string(value["mount"]),
        }
    except TargetInspectionError as exc:
        raise ValueError from exc


def _child_request(
    line: bytes,
    token: bytes,
    config: _RuntimeChildConfig,
    expected_sequence: int,
) -> _RuntimeRequestPayload:
    """Authenticate a versioned complete message before using its body."""
    value = loads(line)
    if (
        not isinstance(value, dict)
        or set(value) != {"payload", "mac"}
        or not isinstance(value["payload"], dict)
        or not isinstance(value["mac"], str)
    ):
        raise ValueError
    payload = value["payload"]
    raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    if not compare_digest(value["mac"], digest(token, raw, "sha256").hex()):
        raise ValueError
    request = _runtime_request_from_payload(payload)
    if (
        request["version"] != _MESSAGE_VERSION
        or request["sequence"] != expected_sequence
        or request["receipt"] != config["receipt"]
        or request["identity"] != config["identity"]
        or request["channel_id"] != config["channel_id"]
        or request["implementation_id"] != config["implementation_id"]
    ):
        raise ValueError
    return request


def _runtime_request_from_payload(value: object) -> _RuntimeRequestPayload:
    """Decode a complete closed request shape."""
    keys = {
        "version",
        "sequence",
        "kind",
        "receipt",
        "identity",
        "channel_id",
        "implementation_id",
        "body",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError
    body = value["body"]
    if not isinstance(body, dict):
        raise ValueError
    try:
        return {
            "version": _integer(value["version"]),
            "sequence": _integer(value["sequence"]),
            "kind": _string(value["kind"]),
            "receipt": _string(value["receipt"]),
            "identity": _string_mapping(value["identity"]),
            "channel_id": _string(value["channel_id"]),
            "implementation_id": _string(value["implementation_id"]),
            "body": body,
        }
    except TargetInspectionError as exc:
        raise ValueError from exc


def _child_response(
    request: _RuntimeRequestPayload,
    body: Mapping[str, object],
    error: TargetErrorCode | None,
    token: bytes,
) -> Mapping[str, object]:
    """Return an authenticated response bound to the exact request."""
    payload: _RuntimeResponsePayload = {
        "version": request["version"],
        "sequence": request["sequence"],
        "receipt": request["receipt"],
        "identity": request["identity"],
        "channel_id": request["channel_id"],
        "implementation_id": request["implementation_id"],
        "body": body,
        "error": None if error is None else error.value,
    }
    raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return {"payload": payload, "mac": digest(token, raw, "sha256").hex()}


def _child_response_from_line(
    line: bytes, error: TargetErrorCode, token: bytes
) -> Mapping[str, object]:
    """Return a failure only for an authenticated request shape."""
    value = loads(line)
    if not isinstance(value, dict) or not isinstance(
        value.get("payload"), dict
    ):
        raise ValueError
    return _child_response(
        _runtime_request_from_payload(value["payload"]), {}, error, token
    )
