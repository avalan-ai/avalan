"""Exercise rooted, incapable local patch inspection."""

from asyncio import CancelledError, create_task, run, sleep
from base64 import b64encode
from copy import copy, deepcopy
from dataclasses import replace
from hmac import digest
from inspect import getsource
from io import BytesIO
from json import dumps
from os import O_RDONLY, close, fstat, mkfifo, stat_result
from os import open as open_fd
from os import read as read_fd
from pathlib import Path
from pickle import dumps as pickle_dumps
from runpy import run_path
from socket import socketpair
from subprocess import run as run_process
from sys import executable
from types import SimpleNamespace
from typing import Never, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import avalan.patch as patch_package
import avalan.patch.target as target_module
from avalan.patch.domain import (
    ByteSize,
    Capability,
    ContextKind,
    DurationTicks,
    FileMode,
    LogicalPath,
    MetadataProfile,
    PatchApprovalId,
    PatchContextId,
    PatchDomainId,
    PatchLimits,
    PatchProtocolId,
    PatchTargetId,
    PatchWorkspaceId,
    SourceBytes,
)
from avalan.patch.parser import (
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.planner import plan
from avalan.patch.target import (
    AliasMode,
    CommitUnavailable,
    FileIdentity,
    ForeignWriterGuarantee,
    InspectionBatch,
    InspectionRequest,
    LocalInspectionTarget,
    LocalPlatformProfile,
    LocalScopeResolver,
    LocalTargetProfile,
    MetadataClassification,
    ParentWitness,
    ProbeState,
    ScopeSelection,
    TargetErrorCode,
    TargetIdentity,
    TargetIncapableReason,
    TargetInspectionError,
    TargetPrimitive,
    TargetSnapshot,
    TrustedLocalRoot,
    WorkerIsolationPolicy,
)

_TEST_RUNTIME_AUTHORITY_SIGNER = Ed25519PrivateKey.generate()
_TEST_RUNTIME_AUTHORITY_VERIFIER = Ed25519PublicKey.from_public_bytes(
    _TEST_RUNTIME_AUTHORITY_SIGNER.public_key().public_bytes(
        Encoding.Raw,
        PublicFormat.Raw,
    )
)
_TEST_RUNTIME_AUTHORITY_VERIFIER_BYTES = (
    _TEST_RUNTIME_AUTHORITY_VERIFIER.public_bytes(
        Encoding.Raw,
        PublicFormat.Raw,
    )
)
_PRODUCTION_WORKER_BOOTSTRAP = target_module._WORKER_BOOTSTRAP


def _test_worker_bootstrap() -> str:
    """Return a test-runtime worker bootstrap with its public verifier only."""
    return _PRODUCTION_WORKER_BOOTSTRAP.replace(
        "from avalan.patch.target import _worker_main\n"
        "raise SystemExit(_worker_main())",
        "import avalan.patch.target as target_module\n"
        "target_module._RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES = "
        + repr(_TEST_RUNTIME_AUTHORITY_VERIFIER_BYTES)
        + "\nraise SystemExit(target_module._worker_main())",
    )


@pytest.fixture(autouse=True)
def _test_runtime_authority_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject the test deployment's public verifier into the trusted host."""
    monkeypatch.setattr(
        target_module,
        "_RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES",
        _TEST_RUNTIME_AUTHORITY_VERIFIER_BYTES,
    )
    monkeypatch.setattr(
        target_module,
        "_WORKER_BOOTSTRAP",
        _test_worker_bootstrap(),
    )


def _test_runtime_authority(
    root: Path,
) -> target_module._RuntimeTargetAuthority:
    """Return the test deployment capability for one exact configured root."""
    return target_module._RuntimeTargetAuthority(
        _TEST_RUNTIME_AUTHORITY_SIGNER.sign(
            target_module._runtime_target_authority_message(root)
        )
    )


def _limits(snapshot_bytes: int = 10_000) -> PatchLimits:
    """Return finite policy limits detached from test workspace contents."""
    return PatchLimits(
        ByteSize(10_000),
        ByteSize(20),
        ByteSize(512),
        ByteSize(20),
        ByteSize(20),
        ByteSize(snapshot_bytes),
        ByteSize(10_000),
        ByteSize(10_000),
        DurationTicks(1),
        DurationTicks(1),
        DurationTicks(1),
    )


def _profile(
    root: Path,
    *,
    cwd: LogicalPath | None = None,
    maximum: int = 256,
    alias_mode: AliasMode = AliasMode.CASE_SENSITIVE,
    normalization: str = "NFC",
    hidden: bool = False,
    policy: str = "policy-a",
    platform: LocalPlatformProfile = LocalPlatformProfile.POSIX,
    worker_policy: WorkerIsolationPolicy | None = None,
    aggregate_maximum: int = 10_000,
    authority: target_module._RuntimeTargetAuthority | None = None,
) -> LocalTargetProfile:
    """Return one trusted local configuration with no model input fields."""
    runtime_authority = authority or _test_runtime_authority(root)
    trusted_root = TrustedLocalRoot(
        root,
        _runtime_authority=runtime_authority,
    )
    witness = target_module._capture_root_witness(trusted_root)
    identity = TargetIdentity(
        PatchContextId("context_" + "a" * 16),
        PatchWorkspaceId("workspace_" + "a" * 16),
        PatchDomainId("domain_" + "a" * 16),
        PatchTargetId("target_" + "a" * 16),
        PatchProtocolId("protocol_" + "a" * 16),
        witness.filesystem_id,
        witness.mount_id,
        policy,
        "workspace-lease-a",
        PatchApprovalId("approval_" + "a" * 16),
    )
    return LocalTargetProfile(
        identity,
        trusted_root,
        cwd,
        _limits(aggregate_maximum),
        ByteSize(maximum),
        _runtime_authority=runtime_authority,
        alias_mode=alias_mode,
        unicode_normalization=normalization,
        hidden_paths_allowed=hidden,
        platform=platform,
        worker_policy=worker_policy or WorkerIsolationPolicy(),
    )


async def _request(
    target: LocalInspectionTarget, *paths: str
) -> InspectionRequest:
    """Resolve one fixed trusted scope before submitting logical paths."""
    scope = await LocalScopeResolver(target.profile).resolve(
        ScopeSelection(ContextKind.LOCAL)
    )
    return InspectionRequest(scope, tuple(LogicalPath(path) for path in paths))


def test_patch_phase_4_requirements(tmp_path: Path) -> None:
    """Inspect regular files, parent witnesses, and handshake facts."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.txt").write_bytes(b"\xef\xbb\xbfalpha\r\n")
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> int:
        request = await _request(target, "docs/note.txt", "missing.txt")
        handshake = await target.handshake(request.scope)
        batch = await target.inspect(request)
        assert handshake.supports_inspection()
        assert handshake.advertised_operations() == frozenset(
            (
                Capability.OBSERVE_MUTATION_PRECONDITIONS,
                Capability.READ_FOR_MUTATION,
            )
        )
        assert (
            TargetIncapableReason.COMMIT_DEFERRED
            in handshake.incapable_reasons
        )
        present, absent = batch.snapshots
        assert present.present and present.bytes_value is not None
        assert present.bytes_value._value == b"\xef\xbb\xbfalpha\r\n"
        assert present.metadata is not None and present.metadata.has_utf8_bom
        assert present.parent.path == LogicalPath("docs")
        assert not absent.present and absent.parent.path is None
        workspace = batch.planner_workspace()
        assert tuple(item.path for item in workspace.files) == (
            LogicalPath("docs/note.txt"),
        )
        canonical = PatchRequestParser(PatchInputLimits()).parse(
            RawPatchIngress(
                RawProviderProfile("phase4-provider"),
                RawToolCallId("phase4-call"),
                RawPatchInputKind.EDIT_JSON,
                RawPatchInputState.COMPLETE,
                dumps(
                    {
                        "path": "docs/note.txt",
                        "edits": [
                            {"old_text": "alpha\n", "new_text": "beta\n"}
                        ],
                    },
                    separators=(",", ":"),
                ).encode(),
            )
        )
        assert plan(canonical, workspace).lineages[0].final.present
        assert await target.commit(request) == CommitUnavailable()
        return len(batch.snapshots)

    assert run(execute()) == 2


def test_patch_phase_4_constructor_and_protocol_boundaries(
    tmp_path: Path,
) -> None:
    """Reject malformed immutable scope, handshake, snapshot, and request."""
    profile = _profile(tmp_path)
    authority = _test_runtime_authority(tmp_path)
    identity = profile.identity
    assert not hasattr(patch_package, "TrustedLocalRoot")
    assert not hasattr(patch_package, "LocalTargetProfile")
    assert not hasattr(patch_package, "LocalScopeResolver")
    with pytest.raises(TargetInspectionError):
        TargetIdentity(
            identity.context_id,
            identity.workspace_id,
            identity.domain_id,
            identity.target_id,
            identity.protocol_id,
            "",
            "mount",
            "policy",
            "lease",
            PatchApprovalId("approval_" + "a" * 16),
        )
    with pytest.raises(TypeError):
        getattr(TrustedLocalRoot, "__call__")(Path("relative"))
    with pytest.raises(TargetInspectionError):
        TrustedLocalRoot(
            Path("relative"),
            _runtime_authority=cast(
                target_module._RuntimeTargetAuthority, object()
            ),
        )
    different_root = tmp_path / "different-root"
    different_root.mkdir()
    with pytest.raises(TargetInspectionError):
        TrustedLocalRoot(
            different_root,
            _runtime_authority=authority,
        )
    with pytest.raises(TargetInspectionError):
        LocalTargetProfile(
            identity,
            TrustedLocalRoot(tmp_path, _runtime_authority=authority),
            None,
            _limits(),
            ByteSize(1),
            _runtime_authority=authority,
            unicode_normalization="NFKC",
        )
    with pytest.raises(TargetInspectionError):
        target_module.ResolvedMutationScope(
            ContextKind.LOCAL,
            identity,
            None,
            _limits(),
            frozenset(),
            frozenset(),
        )
    with pytest.raises(TargetInspectionError):
        target_module.TargetHandshake(
            identity, frozenset(), (TargetIncapableReason.COMMIT_DEFERRED,) * 2
        )
    with pytest.raises(TargetInspectionError):
        FileIdentity(-1, 0)
    parent = ParentWitness(None, FileIdentity(1, 2), "mount")
    with pytest.raises(TargetInspectionError):
        TargetSnapshot(
            LogicalPath("bad.txt"), True, None, None, None, 0, parent
        )
    with pytest.raises(TargetInspectionError):
        TargetSnapshot(
            LogicalPath("bad.txt"), False, None, None, None, 1, parent
        )

    async def execute() -> LogicalPath:
        request = await _request(LocalInspectionTarget(profile), "missing.txt")
        with pytest.raises(TargetInspectionError):
            InspectionRequest(request.scope, ())
        with pytest.raises(TargetInspectionError):
            InspectionRequest(request.scope, (LogicalPath("a.txt"),) * 2)
        return request.paths[0]

    assert run(execute()) == LogicalPath("missing.txt")


def test_patch_phase_4_workspace_import_cannot_mint_a_root_authority(
    tmp_path: Path,
) -> None:
    """Reject forged capabilities before arbitrary-root inspection begins."""
    probe = run_process(
        (
            executable,
            "-c",
            (
                "import copy,inspect,pickle,sys\nfrom pathlib import"
                " Path\nfrom cryptography.hazmat.primitives.asymmetric.ed25519"
                " import Ed25519PrivateKey\nimport avalan.patch.target as"
                " target\nattempts = []\ntry:\n   "
                " attempts.append(target._RuntimeTargetAuthority(b'0' *"
                " 64))\nexcept BaseException:\n    pass\ntry:\n   "
                " authority_type = target._RuntimeTargetAuthority\n   "
                " attempts.append(object.__new__(authority_type))\nexcept"
                " BaseException:\n    pass\nfor value in tuple(attempts):\n   "
                " for operation in (copy.copy, copy.deepcopy, pickle.dumps):\n"
                "        try:\n            operation(value)\n        except"
                " (TypeError, pickle.PicklingError):\n            pass\n      "
                "  else:\n            raise SystemExit(10)\ntry:\n    class"
                " Forged(target._RuntimeTargetAuthority):\n       "
                " pass\nexcept TypeError:\n    pass\nelse:\n    raise"
                " SystemExit(11)\nsigner ="
                " Ed25519PrivateKey.generate()\nforged ="
                " target._RuntimeTargetAuthority(\n   "
                " signer.sign(target._runtime_target_authority_message(\n     "
                "   Path('/private/tmp')\n    ))\n)\npublic_values ="
                " dict(inspect.getmembers(target))\nverifier_name ="
                " '_RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES'\nassert"
                " verifier_name in public_values\nattempts.append(forged)\nfor"
                " authority in attempts:\n    try:\n       "
                " target.TrustedLocalRoot(\n            Path('/private/tmp'),"
                " _runtime_authority=authority\n        )\n    except"
                " target.TargetInspectionError:\n        pass\n    else:\n    "
                "    raise SystemExit(12)\nraise SystemExit(0)\n"
            ),
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0


def test_patch_phase_4_worker_rechecks_parent_authority_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a root whose parent verifier was replaced by workspace code."""
    attacker = Ed25519PrivateKey.generate()
    attacker_verifier = attacker.public_key().public_bytes(
        Encoding.Raw,
        PublicFormat.Raw,
    )
    monkeypatch.setattr(
        target_module,
        "_RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES",
        attacker_verifier,
    )
    monkeypatch.setattr(
        target_module,
        "_WORKER_BOOTSTRAP",
        _PRODUCTION_WORKER_BOOTSTRAP,
    )
    authority = target_module._RuntimeTargetAuthority(
        attacker.sign(
            target_module._runtime_target_authority_message(tmp_path)
        )
    )
    target = LocalInspectionTarget(_profile(tmp_path, authority=authority))

    async def resolve() -> None:
        """Attempt a root observation through the fresh isolated worker."""
        with pytest.raises(TargetInspectionError) as rejected:
            await LocalScopeResolver(target.profile).resolve(
                ScopeSelection(ContextKind.LOCAL)
            )
        assert rejected.value.code is TargetErrorCode.WORKER_UNAVAILABLE

    run(resolve())


def test_patch_phase_4_cwd_alias_and_parent_handle_paths(
    tmp_path: Path,
) -> None:
    """Inspect through a retained trusted cwd and Unicode alias projection."""
    (tmp_path / "nested" / "child").mkdir(parents=True)
    (tmp_path / "nested" / "child" / "file.txt").write_text("value")
    target = LocalInspectionTarget(
        _profile(tmp_path, cwd=LogicalPath("nested/child"))
    )
    normalized = LocalInspectionTarget(_profile(tmp_path, normalization="NFD"))

    async def execute() -> TargetErrorCode:
        request = await _request(target, "file.txt")
        assert (await target.inspect(request)).snapshots[0].present
        request = await _request(normalized, "caf\u00e9.txt", "cafe\u0301.txt")
        with pytest.raises(TargetInspectionError) as error:
            await normalized.inspect(request)
        assert error.value.code is TargetErrorCode.ALIAS_DENIED
        return error.value.code

    assert run(execute()) is TargetErrorCode.ALIAS_DENIED


def test_patch_phase_4_releases_intermediate_parent_handles(
    tmp_path: Path,
) -> None:
    """Inspect a deep path through retained directory handles."""
    (tmp_path / "one" / "two").mkdir(parents=True)
    (tmp_path / "one" / "two" / "note.txt").write_text("note")
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> LogicalPath:
        request = await _request(target, "one/two/note.txt")
        assert (await target.inspect(request)).snapshots[0].present
        return request.paths[0]

    assert run(execute()) == LogicalPath("one/two/note.txt")


def test_patch_phase_4_worker_race_and_mount_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed when a leaf or mount witness changes."""
    leaf = tmp_path / "note.txt"
    leaf.write_text("note")
    descriptor = target_module._open_directory(tmp_path)
    profile = _profile(tmp_path)
    parent = ParentWitness(None, FileIdentity(0, 0), "mount")
    try:
        with pytest.raises(TargetInspectionError) as mount:
            target_module._snapshot_leaf(
                descriptor,
                LogicalPath("note.txt"),
                "note.txt",
                parent,
                profile,
            )
        assert mount.value.code is TargetErrorCode.MOUNT_DENIED
        original = fstat

        def stale(descriptor_value: int) -> stat_result:
            """Return one altered descriptor identity after leaf open."""
            value = list(original(descriptor_value))
            value[1] += 1
            return stat_result(value)

        real_parent = ParentWitness(
            None,
            FileIdentity(leaf.stat().st_dev, tmp_path.stat().st_ino),
            "mount",
        )
        monkeypatch.setattr("avalan.patch.target.fstat", stale)
        with pytest.raises(TargetInspectionError) as stale_error:
            target_module._snapshot_leaf(
                descriptor,
                LogicalPath("note.txt"),
                "note.txt",
                real_parent,
                profile,
            )
        assert stale_error.value.code is TargetErrorCode.WITNESS_STALE
    finally:
        close(descriptor)


def test_patch_phase_4_closes_failed_root_and_cwd_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise cleanup when a configured root or cwd becomes invalid."""
    original_fstat = fstat

    def not_directory(descriptor: int) -> stat_result:
        """Make a valid opened root fail the immediate directory proof."""
        value = list(original_fstat(descriptor))
        value[0] = 0
        return stat_result(value)

    monkeypatch.setattr("avalan.patch.target.fstat", not_directory)
    with pytest.raises(TargetInspectionError) as root_error:
        target_module._open_directory(tmp_path)
    assert root_error.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    monkeypatch.setattr("avalan.patch.target.fstat", original_fstat)
    (tmp_path / "first").mkdir()
    root_fd = target_module._open_directory(tmp_path)
    try:
        with pytest.raises(TargetInspectionError) as cwd_error:
            target_module._open_cwd(root_fd, LogicalPath("first/missing"))
        assert cwd_error.value.code is TargetErrorCode.PATH_DENIED
        with pytest.raises(TargetInspectionError) as first_missing:
            target_module._open_cwd(root_fd, LogicalPath("missing"))
        assert first_missing.value.code is TargetErrorCode.PATH_DENIED
    finally:
        close(root_fd)

    def unavailable(*args: object, **kwargs: object) -> Never:
        """Make final leaf status report a closed target failure."""
        del args, kwargs
        raise OSError()

    root_fd = target_module._open_directory(tmp_path)
    try:
        parent = ParentWitness(
            None,
            FileIdentity(tmp_path.stat().st_dev, tmp_path.stat().st_ino),
            "mount",
        )
        monkeypatch.setattr("avalan.patch.target.stat_at", unavailable)
        with pytest.raises(TargetInspectionError) as leaf_error:
            target_module._snapshot_leaf(
                root_fd,
                LogicalPath("missing.txt"),
                "missing.txt",
                parent,
                _profile(tmp_path),
            )
        assert leaf_error.value.code is TargetErrorCode.PATH_DENIED
    finally:
        close(root_fd)


def test_patch_phase_4_helper_failures_close_descriptors(
    tmp_path: Path,
) -> None:
    """Cover target helper failures without ever invoking a write primitive."""
    target = LocalInspectionTarget(_profile(tmp_path))

    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(TargetInspectionError) as open_error:
        target_module._open_directory(tmp_path / "missing")
    assert open_error.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    root_fd = target_module._open_directory(root)
    try:
        with pytest.raises(TargetInspectionError) as child_error:
            target_module._open_child_directory(root_fd, "missing")
        assert child_error.value.code is TargetErrorCode.PATH_DENIED
        (root / "file.txt").write_text("value")
        with pytest.raises(TargetInspectionError) as child_file:
            target_module._open_child_directory(root_fd, "file.txt")
        assert child_file.value.code is TargetErrorCode.SPECIAL_FILE_DENIED
        with pytest.raises(TargetInspectionError) as regular:
            target_module._open_regular(root_fd, "missing")
        assert regular.value.code is TargetErrorCode.LINK_DENIED
    finally:
        close(root_fd)
    descriptor = open_fd(root / "file.txt", O_RDONLY)
    try:
        with pytest.raises(TargetInspectionError) as oversized:
            target_module._read_bounded(descriptor, 1)
        assert oversized.value.code is TargetErrorCode.LIMIT_EXCEEDED
    finally:
        close(descriptor)
    root_fd = target_module._open_directory(root)
    try:
        root_status = fstat(root_fd)
        assert target_module._root_mount_id(
            root_fd, root_status
        ) == target_module._root_mount_id(root_fd, root_status)
    finally:
        close(root_fd)
    assert (
        target_module.FileIdentity(1, 2).opaque()
        == target_module.FileIdentity(1, 2).opaque()
    )
    assert target.profile.identity.policy_revision == "policy-a"


def test_patch_phase_4_denies_sensitive_paths_before_open(
    tmp_path: Path,
) -> None:
    """Reject a sensitive lexical spelling without disclosing presence."""
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> TargetErrorCode:
        request = await _request(target, ".git/config")
        with pytest.raises(TargetInspectionError) as error:
            await target.inspect(request)
        assert error.value.code is TargetErrorCode.PATH_DENIED
        return error.value.code

    assert run(execute()) is TargetErrorCode.PATH_DENIED


def test_patch_phase_4_denies_obscured_paths_before_open(
    tmp_path: Path,
) -> None:
    """Reject obscured lexical spellings without disclosing file presence."""
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> int:
        for path in (
            ".hidden/note.txt",
            "space /note.txt",
            "note\u202e.txt",
            "name:stream",
            "$HOME/note.txt",
        ):
            request = await _request(target, path)
            with pytest.raises(TargetInspectionError) as error:
                await target.inspect(request)
            assert error.value.code is TargetErrorCode.PATH_DENIED
        return len(
            (
                ".hidden/note.txt",
                "space /note.txt",
                "note\u202e.txt",
                "name:stream",
                "$HOME/note.txt",
            )
        )

    assert run(execute()) == 5


def test_patch_phase_4_rejects_links_special_files_and_hardlinks(
    tmp_path: Path,
) -> None:
    """Reject links, directories, and hard-linked leaves."""
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "canary.txt").write_text("outside")
    (tmp_path / "link").symlink_to(
        tmp_path / "outside", target_is_directory=True
    )
    (tmp_path / "leaf-link").symlink_to(tmp_path / "outside" / "canary.txt")
    (tmp_path / "directory").mkdir()
    (tmp_path / "regular.txt").write_text("regular")
    (tmp_path / "hard.txt").hardlink_to(tmp_path / "regular.txt")
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> str:
        for path, code in (
            ("link/canary.txt", TargetErrorCode.LINK_DENIED),
            ("leaf-link", TargetErrorCode.LINK_DENIED),
            ("directory", TargetErrorCode.SPECIAL_FILE_DENIED),
            ("hard.txt", TargetErrorCode.HARDLINK_DENIED),
        ):
            request = await _request(target, path)
            with pytest.raises(TargetInspectionError) as error:
                await target.inspect(request)
            assert error.value.code is code
        assert (tmp_path / "outside" / "canary.txt").read_text() == "outside"
        return (tmp_path / "outside" / "canary.txt").read_text()

    assert run(execute()) == "outside"


def test_patch_phase_4_alias_bounds_and_duplicate_identity_fail_closed(
    tmp_path: Path,
) -> None:
    """Reject case aliases, oversized snapshots, and duplicate identity."""
    (tmp_path / "Name.txt").write_text("content")
    (tmp_path / "small.txt").write_text("x" * 20)
    case_target = LocalInspectionTarget(
        _profile(tmp_path, alias_mode=AliasMode.CASE_INSENSITIVE)
    )
    bounded_target = LocalInspectionTarget(_profile(tmp_path, maximum=3))

    async def execute() -> TargetErrorCode:
        request = await _request(case_target, "Name.txt", "name.txt")
        with pytest.raises(TargetInspectionError) as error:
            await case_target.inspect(request)
        assert error.value.code is TargetErrorCode.ALIAS_DENIED
        request = await _request(bounded_target, "small.txt")
        with pytest.raises(TargetInspectionError) as error:
            await bounded_target.inspect(request)
        assert error.value.code is TargetErrorCode.LIMIT_EXCEEDED
        parent = ParentWitness(None, FileIdentity(1, 1), "mount")
        item = TargetSnapshot(
            LogicalPath("a.txt"), False, None, None, None, 0, parent
        )
        with pytest.raises(TargetInspectionError) as duplicate:
            InspectionBatch((item, item))
        assert duplicate.value.code is TargetErrorCode.ALIAS_DENIED
        return duplicate.value.code

    assert run(execute()) is TargetErrorCode.ALIAS_DENIED


def test_patch_phase_4_rejects_fifo_swap_and_aggregate_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a post-stat FIFO and aggregate snapshots without blocking."""
    leaf = tmp_path / "swap.txt"
    leaf.write_text("safe")
    profile = _profile(tmp_path, maximum=10, aggregate_maximum=5)
    witness = target_module._capture_root_witness(profile.root)
    original_barrier = target_module._inspection_barrier

    def replace_with_fifo(stage: str) -> None:
        """Replace the prechecked regular leaf without opening a writer."""
        if stage == "leaf" and leaf.exists():
            leaf.unlink()
            mkfifo(leaf)

    monkeypatch.setattr(
        target_module, "_inspection_barrier", replace_with_fifo
    )
    with pytest.raises(TargetInspectionError) as special:
        target_module._inspect_many(
            profile, (LogicalPath("swap.txt"),), witness
        )
    assert special.value.code is TargetErrorCode.SPECIAL_FILE_DENIED
    monkeypatch.setattr(target_module, "_inspection_barrier", original_barrier)
    leaf.unlink()
    (tmp_path / "one.txt").write_text("abc")
    (tmp_path / "two.txt").write_text("def")
    read_sizes: list[int] = []
    original_read = read_fd

    def counted_read(descriptor: int, size: int) -> bytes:
        """Record each bounded read without changing its source bytes."""
        read_sizes.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(target_module, "read_fd", counted_read)
    with pytest.raises(TargetInspectionError) as aggregate:
        target_module._inspect_many(
            profile,
            (LogicalPath("one.txt"), LogicalPath("two.txt")),
            witness,
        )
    assert aggregate.value.code is TargetErrorCode.LIMIT_EXCEEDED
    assert read_sizes == [3]


def test_patch_phase_4_rejects_same_device_mount_topology_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a retained child whose mount topology changes without st_dev."""
    child = tmp_path / "child"
    child.mkdir()
    (child / "note.txt").write_text("safe")
    profile = _profile(tmp_path)
    witness = target_module._capture_root_witness(profile.root)
    child_inode = child.stat().st_ino
    original_topology = target_module._mount_topology

    def rebound_topology(descriptor: int) -> target_module._MountTopology:
        """Emulate a same-filesystem bind or nullfs mount transition."""
        topology = original_topology(descriptor)
        if fstat(descriptor).st_ino == child_inode:
            return target_module._MountTopology(
                "changed-mount-topology", topology.filesystem_id
            )
        return topology

    monkeypatch.setattr(target_module, "_mount_topology", rebound_topology)
    with pytest.raises(TargetInspectionError) as changed:
        target_module._inspect_many(
            profile, (LogicalPath("child/note.txt"),), witness
        )
    assert changed.value.code is TargetErrorCode.WITNESS_STALE


def test_patch_phase_4_scope_witness_and_async_heartbeat(
    tmp_path: Path,
) -> None:
    """Fail stale scope replacement while loop work advances."""
    (tmp_path / "note.txt").write_text("note")
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> bool:
        request = await _request(target, "note.txt")
        heartbeat = create_task(sleep(0, result="advanced"))
        batch = await target.inspect(request)
        assert batch.snapshots[0].present
        assert await heartbeat == "advanced"
        other = LocalInspectionTarget(_profile(tmp_path, policy="policy-b"))
        with pytest.raises(TargetInspectionError) as error:
            await other.handshake(request.scope)
        assert error.value.code is TargetErrorCode.WITNESS_STALE
        with pytest.raises(TargetInspectionError) as wrong_context:
            await LocalScopeResolver(target.profile).resolve(
                ScopeSelection(ContextKind.SANDBOX)
            )
        assert (
            wrong_context.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
        )
        return batch.snapshots[0].present

    assert run(execute()) is True


def test_patch_phase_4_worker_channel_isolated_and_unforgeable(
    tmp_path: Path,
) -> None:
    """Reject ambient worker authority and forged channel scope access."""
    for arguments in (
        {"inherited_descriptor_count": 1},
        {"credential_count": 1},
        {"network_enabled": True},
        {"workspace_imports_enabled": True},
    ):
        with pytest.raises(TargetInspectionError) as rejected:
            WorkerIsolationPolicy(**arguments)
        assert rejected.value.code is TargetErrorCode.ISOLATION_DENIED
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> TargetErrorCode:
        scope = await LocalScopeResolver(target.profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        forged = replace(scope, _worker_authorization=None)
        with pytest.raises(TargetInspectionError) as rejected:
            await target.handshake(forged)
        assert rejected.value.code is TargetErrorCode.WITNESS_STALE
        return rejected.value.code

    assert run(execute()) is TargetErrorCode.WITNESS_STALE
    assert "environ" not in getsource(target_module._inspect_many)


def test_patch_phase_4_denies_worker_when_seatbelt_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed before launching any unsandboxed worker fallback."""
    target = LocalInspectionTarget(_profile(tmp_path))
    launched = False

    async def unavailable() -> bool:
        """Report the deterministic unavailable enforcement result."""
        return False

    async def unexpected_launch(*args: object, **kwargs: object) -> Never:
        """Reject any attempt to substitute an unsandboxed worker process."""
        nonlocal launched
        del args, kwargs
        launched = True
        raise AssertionError("worker launch must not occur")

    monkeypatch.setattr(
        target_module, "_seatbelt_worker_available", unavailable
    )
    monkeypatch.setattr(
        target_module, "create_subprocess_exec", unexpected_launch
    )

    async def execute() -> TargetErrorCode:
        with pytest.raises(TargetInspectionError) as unavailable_error:
            await LocalScopeResolver(target.profile).resolve(
                ScopeSelection(ContextKind.LOCAL)
            )
        return unavailable_error.value.code

    assert run(execute()) is TargetErrorCode.WORKER_UNAVAILABLE
    assert not launched


def test_patch_phase_4_generates_network_denied_seatbelt_worker_policy(
    tmp_path: Path,
) -> None:
    """Generate Avalan's concrete no-network Seatbelt worker policy."""
    profile = _profile(tmp_path)
    policy = target_module._worker_seatbelt_profile(
        profile,
        (executable, "-I", "-m", "avalan.patch.target"),
        "a" * 64,
    )
    assert "(deny network*)" in policy
    assert "(allow network" not in policy
    assert "(deny process-fork)" in policy
    assert str(tmp_path) in policy


def test_patch_phase_4_platform_handshake_and_read_only_probes(
    tmp_path: Path,
) -> None:
    """Expose exact POSIX facts and fail closed for unsupported profiles."""
    target = LocalInspectionTarget(_profile(tmp_path))
    unsupported = LocalInspectionTarget(
        _profile(tmp_path, platform=LocalPlatformProfile.UNSUPPORTED)
    )

    async def execute() -> int:
        scope = await LocalScopeResolver(target.profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        handshake = await target.handshake(scope)
        assert handshake.supports_inspection()
        assert handshake.platform is LocalPlatformProfile.POSIX
        assert handshake.worker is scope.worker
        assert (
            handshake.foreign_writer_guarantee
            is ForeignWriterGuarantee.REVALIDATE_BEFORE_COMMIT
        )
        assert {probe.state for probe in handshake.probes} == {
            ProbeState.UNAVAILABLE
        }
        assert {probe.primitive for probe in handshake.probes} == {
            TargetPrimitive.BOUNDED_WRITE,
            TargetPrimitive.REPLACE_PUBLICATION,
            TargetPrimitive.NOREPLACE_CREATE_MOVE,
            TargetPrimitive.DIRECTORY_ENTRY_DELETE,
            TargetPrimitive.METADATA_PRESERVATION,
            TargetPrimitive.SAME_FILESYSTEM_MOVE,
            TargetPrimitive.STAGING,
            TargetPrimitive.STRUCTURAL_VERIFICATION,
        }
        unsupported_scope = await LocalScopeResolver(
            unsupported.profile
        ).resolve(ScopeSelection(ContextKind.LOCAL))
        unsupported_handshake = await unsupported.handshake(unsupported_scope)
        assert not unsupported_handshake.supports_inspection()
        with pytest.raises(TargetInspectionError) as rejected:
            await unsupported.inspect(
                InspectionRequest(
                    unsupported_scope, (LogicalPath("note.txt"),)
                )
            )
        assert rejected.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
        return len(handshake.probes)

    assert run(execute()) == 8


def test_patch_phase_4_classifies_executable_and_privileged_metadata(
    tmp_path: Path,
) -> None:
    """Classify executable modes and reject privileged metadata profiles."""
    executable = tmp_path / "executable.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o700)
    privileged = tmp_path / "privileged.sh"
    privileged.write_text("#!/bin/sh\n")
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> MetadataClassification:
        request = await _request(target, "executable.sh")
        snapshot = (await target.inspect(request)).snapshots[0]
        assert snapshot.security_metadata is MetadataClassification.EXECUTABLE
        assert (
            target_module._classify_metadata(0o6755)
            is MetadataClassification.PRIVILEGED
        )
        return snapshot.security_metadata

    assert run(execute()) is MetadataClassification.EXECUTABLE


def test_patch_phase_4_rebinds_only_ephemeral_worker_witness(
    tmp_path: Path,
) -> None:
    """Accept fenced worker rebinding but reject a replaced trusted root."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.txt").write_text("note")
    target = LocalInspectionTarget(_profile(root))
    resolver = LocalScopeResolver(target.profile)

    async def execute() -> TargetErrorCode:
        scope = await resolver.resolve(ScopeSelection(ContextKind.LOCAL))
        rebound = await resolver.rebind_ephemeral(scope)
        assert rebound.identity == scope.identity
        assert rebound.root_witness == scope.root_witness
        assert rebound.worker is not None and scope.worker is not None
        assert (
            rebound.worker.worker_instance_id
            != scope.worker.worker_instance_id
        )
        assert rebound.worker.fence_id == scope.worker.fence_id
        assert (await target.handshake(rebound)).worker == rebound.worker
        root.rename(tmp_path / "retired-root")
        root.mkdir()
        with pytest.raises(TargetInspectionError) as rejected:
            await target.inspect(
                InspectionRequest(scope, (LogicalPath("note.txt"),))
            )
        assert rejected.value.code is TargetErrorCode.WITNESS_STALE
        return rejected.value.code

    assert run(execute()) is TargetErrorCode.WITNESS_STALE


def test_patch_phase_4_barriers_deny_ancestor_leaf_and_mount_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep outside canaries unreachable across deterministic race barriers."""
    root = tmp_path / "root"
    inside = root / "inside"
    outside = tmp_path / "outside"
    inside.mkdir(parents=True)
    outside.mkdir()
    leaf = inside / "note.txt"
    leaf.write_text("inside")
    canary = outside / "canary.txt"
    canary.write_text("outside")
    profile = _profile(root)
    witness = target_module._capture_root_witness(profile.root)
    original_barrier = target_module._inspection_barrier

    def execute() -> int:
        paths = (LogicalPath("inside/note.txt"),)

        def replace_ancestor(stage: str) -> None:
            """Swap a traversed directory for an outside-root symlink."""
            if stage == "component" and inside.exists():
                inside.rename(root / "retired-inside")
                inside.symlink_to(outside, target_is_directory=True)

        monkeypatch.setattr(
            target_module, "_inspection_barrier", replace_ancestor
        )
        with pytest.raises(TargetInspectionError):
            target_module._inspect_many(profile, paths, witness)
        assert canary.read_text() == "outside"
        inside.unlink()
        (root / "retired-inside").rename(inside)
        monkeypatch.setattr(
            target_module, "_inspection_barrier", original_barrier
        )

        def replace_leaf(stage: str) -> None:
            """Swap a validated leaf for an outside-root symlink."""
            if stage == "leaf" and leaf.exists():
                leaf.unlink()
                leaf.symlink_to(canary)

        monkeypatch.setattr(target_module, "_inspection_barrier", replace_leaf)
        with pytest.raises(TargetInspectionError):
            target_module._inspect_many(profile, paths, witness)
        assert canary.read_text() == "outside"
        monkeypatch.setattr(
            target_module, "_inspection_barrier", original_barrier
        )
        leaf.unlink()
        leaf.write_text("inside")
        monkeypatch.setattr(
            target_module, "_filesystem_id", lambda descriptor: "other"
        )
        with pytest.raises(TargetInspectionError) as mount:
            target_module._inspect_many(profile, paths, witness)
        assert mount.value.code is TargetErrorCode.WITNESS_STALE
        return len(canary.read_bytes())

    assert execute() == len(b"outside")


def test_patch_phase_4_rejects_malformed_worker_and_metadata_witnesses(
    tmp_path: Path,
) -> None:
    """Reject forged workers, inspection probes, and privileged snapshots."""
    with pytest.raises(TargetInspectionError) as primitive:
        target_module.PrimitiveProbe(
            TargetPrimitive.BOUNDED_READ, ProbeState.UNAVAILABLE
        )
    assert primitive.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as worker:
        target_module.EphemeralWorkerWitness("", "worker", "fence")
    assert worker.value.code is TargetErrorCode.ISOLATION_DENIED
    parent = ParentWitness(None, FileIdentity(1, 1), "mount")
    with pytest.raises(TargetInspectionError) as metadata:
        TargetSnapshot(
            LogicalPath("note.txt"),
            False,
            None,
            None,
            None,
            0,
            parent,
            MetadataClassification.PRIVILEGED,
        )
    assert metadata.value.code is TargetErrorCode.METADATA_DENIED
    target = LocalInspectionTarget(_profile(tmp_path))

    async def execute() -> TargetErrorCode:
        scope = await LocalScopeResolver(target.profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        with pytest.raises(TargetInspectionError) as rejected:
            await LocalScopeResolver(target.profile).rebind_ephemeral(
                replace(scope, _worker_authorization=None)
            )
        assert rejected.value.code is TargetErrorCode.WITNESS_STALE
        return rejected.value.code

    assert run(execute()) is TargetErrorCode.WITNESS_STALE


def test_patch_phase_4_detects_descriptor_and_cwd_witness_swaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject post-open directory and retained-cwd identity replacement."""
    child = tmp_path / "child"
    child.mkdir()
    (child / "note.txt").write_text("note")
    root_fd = target_module._open_directory(tmp_path)
    original_fstat = fstat

    def changed(descriptor: int) -> stat_result:
        """Return an identity that cannot match the no-follow precheck."""
        value = list(original_fstat(descriptor))
        value[1] += 1
        return stat_result(value)

    try:
        monkeypatch.setattr("avalan.patch.target.fstat", changed)
        with pytest.raises(TargetInspectionError) as child_error:
            target_module._open_child_directory(root_fd, "child")
        assert child_error.value.code is TargetErrorCode.WITNESS_STALE
        monkeypatch.setattr("avalan.patch.target.fstat", original_fstat)
        cwd_fd, identity = target_module._open_cwd(
            root_fd, LogicalPath("child")
        )
        try:
            monkeypatch.setattr("avalan.patch.target.fstat", changed)
            with pytest.raises(TargetInspectionError) as cwd_error:
                target_module._inspect_path(
                    cwd_fd,
                    identity,
                    target_module._root_mount_id(
                        root_fd, original_fstat(root_fd)
                    ),
                    target_module._filesystem_id(cwd_fd),
                    target_module._root_mount_id(
                        cwd_fd, original_fstat(cwd_fd)
                    ),
                    LogicalPath("note.txt"),
                    _profile(tmp_path),
                )
            assert cwd_error.value.code is TargetErrorCode.WITNESS_STALE
        finally:
            monkeypatch.setattr("avalan.patch.target.fstat", original_fstat)
            close(cwd_fd)
    finally:
        close(root_fd)


def test_patch_phase_4_rejects_root_and_cwd_mount_witness_swaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject identity changes after root open and before cwd inspection."""
    original_fstat = fstat
    calls = 0

    def stale_root(descriptor: int) -> stat_result:
        """Alter only the second root descriptor observation."""
        nonlocal calls
        calls += 1
        value = list(original_fstat(descriptor))
        if calls == 2:
            value[1] += 1
        return stat_result(value)

    monkeypatch.setattr("avalan.patch.target.fstat", stale_root)
    with pytest.raises(TargetInspectionError) as root_error:
        target_module._open_directory(tmp_path)
    assert root_error.value.code is TargetErrorCode.WITNESS_STALE
    monkeypatch.setattr("avalan.patch.target.fstat", original_fstat)
    profile = _profile(tmp_path)
    witness = target_module._capture_root_witness(profile.root)
    original_cwd = target_module._open_cwd
    watched: set[int] = set()

    def opened_cwd(
        root_fd: int,
        cwd: LogicalPath | None,
        expected_filesystem_id: str | None = None,
        expected_mount_id: str | None = None,
    ) -> tuple[int, FileIdentity]:
        """Mark only the returned retained cwd descriptor as cross-mount."""
        descriptor, identity = original_cwd(
            root_fd, cwd, expected_filesystem_id, expected_mount_id
        )
        watched.add(descriptor)
        return descriptor, identity

    def cross_mount(descriptor: int) -> stat_result:
        """Make the inspected cwd descriptor report another device."""
        value = list(original_fstat(descriptor))
        if descriptor in watched:
            value[2] += 1
        return stat_result(value)

    monkeypatch.setattr(target_module, "_open_cwd", opened_cwd)
    monkeypatch.setattr("avalan.patch.target.fstat", cross_mount)
    with pytest.raises(TargetInspectionError) as mount_error:
        target_module._inspect_many(
            profile, (LogicalPath("missing.txt"),), witness
        )
    assert mount_error.value.code is TargetErrorCode.MOUNT_DENIED


def test_patch_phase_4_hostile_ambient_authority_stays_unobserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignore hostile ambient, descriptor, network, and import artifacts."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.txt").write_text("safe")
    secret = tmp_path / "ambient-secret.txt"
    secret.write_text("ambient")
    hook = root / "sitecustomize.py"
    hook.write_text("raise RuntimeError('workspace hook executed')")
    monkeypatch.setenv("PATCH_WORKER_SECRET", "ambient")
    monkeypatch.chdir(root)
    descriptor = open_fd(secret, O_RDONLY)
    reader, writer = socketpair()
    reader.settimeout(0.01)
    target = LocalInspectionTarget(_profile(root))

    async def execute() -> bytes:
        request = await _request(target, "note.txt")
        snapshot = (await target.inspect(request)).snapshots[0]
        assert snapshot.bytes_value is not None
        return snapshot.bytes_value._value

    try:
        assert run(execute()) == b"safe"
        with pytest.raises(TimeoutError):
            reader.recv(1)
        assert "PATCH_WORKER_SECRET" not in getsource(
            target_module._inspect_many
        )
        assert target.profile.worker_policy.inherited_descriptor_count == 0
        assert target.profile.worker_policy.credential_count == 0
        assert not target.profile.worker_policy.network_enabled
        assert not target.profile.worker_policy.workspace_imports_enabled
    finally:
        reader.close()
        writer.close()
        close(descriptor)


def test_patch_phase_4_workspace_process_cannot_reach_worker_channel(
    tmp_path: Path,
) -> None:
    """Keep the authenticated worker bearer out of workspace process inputs."""
    root = tmp_path / "root"
    root.mkdir()
    target = LocalInspectionTarget(_profile(root))

    async def execute() -> bytes:
        scope = await LocalScopeResolver(target.profile).resolve(
            ScopeSelection(ContextKind.LOCAL)
        )
        with pytest.raises(TypeError):
            pickle_dumps(scope)
        return scope.worker.channel_id.encode() if scope.worker else b""

    channel = run(execute())
    probe = run_process(
        (
            executable,
            "-c",
            (
                "import os,sys;sys.exit("
                "int(os.getenv('PATCH_WORKER_CHANNEL') is not None))"
            ),
        ),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert channel == b"local-inspection-channel-v1"
    assert probe.returncode == 0


def test_patch_phase_4_root_barrier_replacement_never_reaches_canary(
    tmp_path: Path,
) -> None:
    """Fail closed when a captured root becomes an outside-root symlink."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary.txt"
    canary.write_text("outside")
    target = LocalInspectionTarget(_profile(root))

    async def execute() -> TargetErrorCode:
        request = await _request(target, "note.txt")
        root.rename(tmp_path / "retired-root")
        root.symlink_to(outside, target_is_directory=True)
        with pytest.raises(TargetInspectionError) as rejected:
            await target.inspect(request)
        assert canary.read_text() == "outside"
        return rejected.value.code

    assert run(execute()) is TargetErrorCode.WITNESS_STALE


def test_patch_phase_4_worker_wire_rejects_malformed_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed for malformed worker payloads before any target read."""
    snapshot_values: tuple[object, ...] = (
        None,
        [],
        {"path": 1, "present": False, "parent": {}},
        {"path": "note.txt", "present": "yes", "parent": {}},
    )
    for snapshot_value in snapshot_values:
        with pytest.raises(TargetInspectionError) as snapshot:
            target_module._snapshot_from_worker(snapshot_value)
        assert snapshot.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    parent_values: tuple[object, ...] = (
        None,
        [],
        {"path": 1, "identity": [1, 2], "mount_id": "mount"},
        {"path": None, "identity": [1], "mount_id": "mount"},
        {"path": None, "identity": [1, 2], "mount_id": 1},
    )
    for parent_value in parent_values:
        with pytest.raises(TargetInspectionError) as parent:
            target_module._parent_from_worker(parent_value)
        assert parent.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    worker_profile_payloads: tuple[dict[str, object], ...] = (
        {},
        {
            "root": "relative",
            "cwd": None,
            "maximum": 1,
            "aggregate_maximum": 1,
        },
        {
            "root": "/private/tmp",
            "cwd": 1,
            "maximum": 1,
            "aggregate_maximum": 1,
        },
        {
            "root": "/private/tmp",
            "cwd": None,
            "maximum": 0,
            "aggregate_maximum": 1,
        },
        {
            "root": "/private/tmp",
            "cwd": None,
            "maximum": 1,
            "aggregate_maximum": 0,
        },
    )
    for worker_profile_payload in worker_profile_payloads:
        with pytest.raises(TargetInspectionError) as profile:
            target_module._worker_profile(worker_profile_payload)
        assert profile.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as non_mapping_profile:
        target_module._worker_profile(None)
    assert non_mapping_profile.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    with pytest.raises(TargetInspectionError) as malformed_response:
        target_module._worker_response_payload({"error": 1})
    assert malformed_response.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    worker_root_values: tuple[object, ...] = (
        None,
        [],
        {},
        {"identity": [1], "mount_id": "mount"},
    )
    for worker_root_value in worker_root_values:
        with pytest.raises(TargetInspectionError) as root:
            target_module._worker_root(worker_root_value)
        assert root.value.code is TargetErrorCode.WITNESS_STALE
    assert target_module._worker_response({"operation": "unknown"}) == {
        "error": TargetErrorCode.WORKER_UNAVAILABLE.value
    }

    class Stream:
        """Provide a bytes buffer matching the worker stdio projection."""

        def __init__(self, value: bytes = b"") -> None:
            """Initialize one in-memory worker stream."""
            self.buffer = BytesIO(value)

    token = b"a" * 32
    monkeypatch.setattr(target_module, "environ", {})
    assert target_module._worker_main() == 2
    monkeypatch.setattr(
        target_module,
        "environ",
        {target_module._WORKER_TOKEN_ENV: "not-hex"},
    )
    assert target_module._worker_main() == 2
    monkeypatch.setattr(target_module, "stdin", Stream(b"not-json"))
    assert target_module._worker_main() == 2
    payload: dict[str, object] = {
        "operation": "unsupported",
        "root": "/private/tmp",
        "cwd": None,
        "maximum": 1,
        "aggregate_maximum": 1,
        "paths": [],
        "expected_root": None,
    }
    raw_payload = dumps(payload, separators=(",", ":")).encode()
    request = dumps(
        {
            "payload": payload,
            "mac": digest(token, raw_payload, "sha256").hex(),
        },
        separators=(",", ":"),
    ).encode()
    output = Stream()
    monkeypatch.setattr(
        target_module,
        "environ",
        {target_module._WORKER_TOKEN_ENV: token.hex()},
    )
    monkeypatch.setattr(target_module, "stdin", Stream(request))
    monkeypatch.setattr(target_module, "stdout", output)
    assert target_module._worker_main() == 0
    assert output.buffer.getvalue()


def test_patch_phase_4_worker_request_rejects_transport_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject worker transport failures without an unsandboxed fallback."""
    profile = _profile(tmp_path)

    class Process:
        """Model one deterministic worker subprocess result."""

        def __init__(
            self,
            response: bytes,
            returncode: int = 0,
            cancelled: bool = False,
        ) -> None:
            """Initialize one controlled process interaction."""
            self.response = response
            self.returncode = returncode
            self.cancelled = cancelled
            self.terminated = False

        async def communicate(self, _request: bytes) -> tuple[bytes, bytes]:
            """Return one buffered worker response or caller cancellation."""
            if self.cancelled:
                raise CancelledError
            return self.response, b"private diagnostic"

        def terminate(self) -> None:
            """Record cancellation settlement without a real subprocess."""
            self.terminated = True

        async def wait(self) -> None:
            """Settle the controlled cancellation path."""

    async def available() -> bool:
        """Keep all cases on the sealed-worker branch."""
        return True

    monkeypatch.setattr(target_module, "_seatbelt_worker_available", available)

    def envelope(payload: object, mac: str | None = None) -> bytes:
        """Encode one worker response using the current private channel key."""
        raw = dumps(payload, separators=(",", ":")).encode()
        return dumps(
            {
                "payload": payload,
                "mac": (
                    mac
                    or digest(
                        profile._worker_authorization.token, raw, "sha256"
                    ).hex()
                ),
            },
            separators=(",", ":"),
        ).encode()

    async def request_with(
        process: Process,
    ) -> target_module._WorkerResponsePayload:
        """Run one request through a controlled sealed process result."""

        async def launch(*args: object, **kwargs: object) -> Process:
            """Return the selected controlled process."""
            del args, kwargs
            return process

        monkeypatch.setattr(target_module, "create_subprocess_exec", launch)
        return await target_module._worker_request(
            profile, "witness", (), None
        )

    assert run(request_with(Process(envelope({"snapshots": []})))) == {
        "snapshots": []
    }
    for process in (
        Process(b"", returncode=1),
        Process(b"not-json"),
        Process(dumps([]).encode()),
        Process(dumps({"payload": [], "mac": 1}).encode()),
        Process(dumps({"payload": {}, "mac": 1}).encode()),
        Process(envelope({"snapshots": []}, mac="wrong")),
        Process(envelope({"error": "unknown"})),
        Process(envelope({"error": TargetErrorCode.PATH_DENIED.value})),
    ):
        with pytest.raises(TargetInspectionError):
            run(request_with(process))

    async def unavailable_launch(*args: object, **kwargs: object) -> Never:
        """Raise the launch error that must remain capability-unavailable."""
        del args, kwargs
        raise OSError("unavailable")

    monkeypatch.setattr(
        target_module, "create_subprocess_exec", unavailable_launch
    )
    with pytest.raises(TargetInspectionError) as unavailable:
        run(target_module._worker_request(profile, "witness", (), None))
    assert unavailable.value.code is TargetErrorCode.WORKER_UNAVAILABLE
    cancelled = Process(b"", cancelled=True)
    with pytest.raises(CancelledError):
        run(request_with(cancelled))
    assert cancelled.terminated


def test_patch_phase_4_helper_failure_branches_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise bounded representation, read, and topology failure branches."""
    seatbelt_values = ("", "bad\x00", "bad\n", "bad\r")
    for seatbelt_value in seatbelt_values:
        with pytest.raises(TargetInspectionError):
            target_module._seatbelt_string(seatbelt_value)
    assert target_module._seatbelt_read_data("/").endswith('"/"))')
    representation_values = (
        b"\xef\xbb\xbf\xef\xbb\xbfx",
        b"\xff",
        b"nul\x00",
        b"bare\r",
        b"lf\ncrlf\r\n",
    )
    for representation_value in representation_values:
        with pytest.raises(TargetInspectionError) as representation:
            target_module._snapshot_representation(representation_value)
        assert representation.value.code is TargetErrorCode.METADATA_DENIED
    source = tmp_path / "source.txt"
    source.write_text("abc")
    descriptor = open_fd(source, O_RDONLY)
    try:
        with pytest.raises(TargetInspectionError) as limit:
            target_module._read_bounded(descriptor, 2)
        assert limit.value.code is TargetErrorCode.LIMIT_EXCEEDED
    finally:
        close(descriptor)
    descriptor = open_fd(source, O_RDONLY)
    try:
        monkeypatch.setattr(target_module, "read_fd", lambda fd, size: b"")
        with pytest.raises(TargetInspectionError) as stale:
            target_module._read_bounded(descriptor, 3)
        assert stale.value.code is TargetErrorCode.WITNESS_STALE
    finally:
        close(descriptor)
    monkeypatch.setattr(target_module, "platform", "unsupported")
    with pytest.raises(TargetInspectionError) as unsupported:
        target_module._mount_topology(-1)
    assert unsupported.value.code is TargetErrorCode.MOUNT_DENIED
    with pytest.raises(TargetInspectionError):
        target_module._root_mount_id(-1, object())


def test_patch_phase_4_authority_and_worker_wire_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every local capability and worker-wire rejection branch."""
    with pytest.raises(TargetInspectionError):
        target_module._RuntimeTargetAuthority(b"short")
    authority = _test_runtime_authority(tmp_path)
    with pytest.raises(TypeError):
        copy(authority)
    with pytest.raises(TypeError):
        deepcopy(authority)
    with pytest.raises(TypeError):
        pickle_dumps(authority)
    with pytest.raises(TypeError):
        type("AuthoritySubclass", (target_module._RuntimeTargetAuthority,), {})

    profile = _profile(tmp_path)
    invalid_profile = object.__new__(LocalTargetProfile)
    object.__setattr__(invalid_profile, "root", profile.root)
    object.__setattr__(
        invalid_profile,
        "_runtime_authority",
        target_module._RuntimeTargetAuthority(b"x" * 64),
    )
    with pytest.raises(TargetInspectionError) as resolver:
        LocalScopeResolver(invalid_profile)
    assert resolver.value.code is TargetErrorCode.ISOLATION_DENIED
    with pytest.raises(TargetInspectionError) as target:
        LocalInspectionTarget(invalid_profile)
    assert target.value.code is TargetErrorCode.ISOLATION_DENIED

    async def wrong_mount() -> None:
        """Reject a worker witness that does not match configured identity."""
        mismatched = replace(
            profile,
            identity=replace(profile.identity, mount_id="different-mount"),
        )
        with pytest.raises(TargetInspectionError) as rejected:
            await LocalScopeResolver(mismatched).resolve(
                ScopeSelection(ContextKind.LOCAL)
            )
        assert rejected.value.code is TargetErrorCode.MOUNT_DENIED

    run(wrong_mount())

    worker_payload: dict[str, object] = {
        "operation": "witness",
        "root": str(tmp_path),
        "cwd": None,
        "maximum": 16,
        "aggregate_maximum": 32,
        "authority_signature": b64encode(authority._signature).decode(),
        "paths": [],
        "expected_root": None,
    }
    worker_profile = target_module._worker_profile(worker_payload)
    witness = target_module._worker_capture_root(worker_profile)
    assert target_module._worker_response(worker_payload)["mount_id"] == (
        witness.mount_id
    )
    worker_payload["operation"] = "inspect"
    worker_payload["paths"] = ["missing.txt"]
    worker_payload["expected_root"] = {
        "identity": [witness.identity.device, witness.identity.inode],
        "mount_id": witness.mount_id,
        "filesystem_id": witness.filesystem_id,
    }
    assert target_module._worker_response(worker_payload)["snapshots"]
    worker_payload["paths"] = [1]
    assert target_module._worker_response(worker_payload) == {
        "error": TargetErrorCode.PATH_DENIED.value
    }
    del worker_payload["paths"]
    assert target_module._worker_response(worker_payload) == {
        "error": TargetErrorCode.PATH_DENIED.value
    }
    for malformed_root in (None, {}, {"identity": [1], "mount_id": "m"}):
        worker_payload["expected_root"] = malformed_root
        worker_payload["paths"] = []
        assert target_module._worker_response(worker_payload) == {
            "error": TargetErrorCode.WITNESS_STALE.value
        }

    malformed_snapshot = {
        "path": "file.txt",
        "present": True,
        "parent": {
            "path": None,
            "identity": [1, 2],
            "mount_id": "mount",
        },
        "bytes": "AA==",
        "metadata": {"mode": 0o644, "has_bom": False, "representation": "lf"},
        "identity": [3, 4],
        "link_count": 1,
        "classification": MetadataClassification.ORDINARY.value,
    }
    for field, value in (
        ("bytes", 1),
        (
            "metadata",
            {"mode": "bad", "has_bom": False, "representation": "lf"},
        ),
        ("bytes", "!"),
        ("classification", "unknown"),
    ):
        candidate = dict(malformed_snapshot)
        candidate[field] = value
        with pytest.raises(TargetInspectionError):
            target_module._snapshot_from_worker(candidate)

    monkeypatch.setattr(target_module, "find_spec", lambda name: None)
    with pytest.raises(TargetInspectionError):
        target_module._cffi_backend_runtime_path()
    monkeypatch.setattr(target_module, "cryptography_file", None)
    with pytest.raises(TargetInspectionError):
        target_module._worker_seatbelt_profile(profile, (), "")


def test_patch_phase_4_direct_rooted_inspection_branches(
    tmp_path: Path,
) -> None:
    """Exercise rooted helper branches in the parent coverage process."""
    nested = tmp_path / "nested" / "child"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("content")
    (tmp_path / "ordinary.txt").write_text("ordinary")
    worker_profile = target_module._WorkerInspectionProfile(
        tmp_path,
        None,
        64,
        128,
    )
    witness = target_module._worker_capture_root(worker_profile)
    snapshots = target_module._inspect_many(
        worker_profile,
        (
            LogicalPath("nested/child/file.txt"),
            LogicalPath("missing.txt"),
        ),
        witness,
    )
    assert snapshots[0].present and not snapshots[1].present
    parent = ParentWitness(None, witness.identity, witness.mount_id)
    root_fd = open_fd(tmp_path, O_RDONLY)
    try:
        with pytest.raises(TargetInspectionError) as link:
            (tmp_path / "link.txt").symlink_to(tmp_path / "ordinary.txt")
            target_module._snapshot_leaf(
                root_fd,
                LogicalPath("link.txt"),
                "link.txt",
                parent,
                worker_profile,
                witness.mount_id,
            )
        assert link.value.code is TargetErrorCode.LINK_DENIED
        with pytest.raises(TargetInspectionError) as special:
            target_module._snapshot_leaf(
                root_fd,
                LogicalPath("nested"),
                "nested",
                parent,
                worker_profile,
                witness.mount_id,
            )
        assert special.value.code is TargetErrorCode.SPECIAL_FILE_DENIED
    finally:
        close(root_fd)
    hardlink = tmp_path / "hardlink.txt"
    hardlink.hardlink_to(tmp_path / "ordinary.txt")
    fresh_witness = target_module._worker_capture_root(worker_profile)
    with pytest.raises(TargetInspectionError) as hardlink_error:
        target_module._inspect_many(
            worker_profile,
            (LogicalPath("hardlink.txt"),),
            fresh_witness,
        )
    assert hardlink_error.value.code is TargetErrorCode.HARDLINK_DENIED
    assert (
        target_module._classify_metadata(0o100)
        is MetadataClassification.EXECUTABLE
    )


def test_patch_phase_4_remaining_target_statement_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise exceptional target paths in the parent coverage process."""
    profile = _profile(tmp_path)

    async def malformed_root_witness(
        _profile: LocalTargetProfile,
        _operation: str,
        _paths: tuple[LogicalPath, ...],
        _expected_root: target_module.RootWitness | None,
    ) -> dict[str, object]:
        """Return a malformed root-witness response without a worker launch."""
        return {"identity": [], "mount_id": "mount", "filesystem_id": "fs"}

    monkeypatch.setattr(
        target_module,
        "_worker_request",
        malformed_root_witness,
    )
    with pytest.raises(TargetInspectionError):
        run(target_module._worker_root_witness(profile))

    async def missing_filesystem_witness(
        _profile: LocalTargetProfile,
        _operation: str,
        _paths: tuple[LogicalPath, ...],
        _expected_root: target_module.RootWitness | None,
    ) -> dict[str, object]:
        """Return a root witness whose filesystem field is unavailable."""
        return {"identity": [1, 2], "mount_id": "mount"}

    monkeypatch.setattr(
        target_module,
        "_worker_request",
        missing_filesystem_witness,
    )
    with pytest.raises(TargetInspectionError):
        run(target_module._worker_root_witness(profile))

    witness = target_module._capture_root_witness(profile.root)

    async def missing_snapshots(
        _profile: LocalTargetProfile,
        _operation: str,
        _paths: tuple[LogicalPath, ...],
        _expected_root: target_module.RootWitness | None,
    ) -> dict[str, object]:
        """Return an inspection response without required snapshots."""
        return {}

    monkeypatch.setattr(target_module, "_worker_request", missing_snapshots)
    with pytest.raises(TargetInspectionError):
        run(target_module._worker_inspect(profile, (), witness))
    with pytest.raises(TargetInspectionError):
        target_module._validate_aliases(
            (LogicalPath("/".join(("x" * 200,) * 3)),),
            profile,
        )

    worker_profile = target_module._WorkerInspectionProfile(
        tmp_path,
        LogicalPath("one/two"),
        64,
        128,
    )
    (tmp_path / "one" / "two").mkdir(parents=True)
    root_fd = target_module._open_directory(tmp_path)
    try:
        root_status = fstat(root_fd)
        cwd_fd, _ = target_module._open_cwd(
            root_fd,
            worker_profile.cwd,
            target_module._filesystem_id(root_fd),
            target_module._root_mount_id(root_fd, root_status),
        )
        close(cwd_fd)
        (tmp_path / "directory-link").symlink_to(
            tmp_path / "one",
            target_is_directory=True,
        )
        with pytest.raises(TargetInspectionError) as link:
            target_module._open_child_directory(
                root_fd,
                "directory-link",
            )
        assert link.value.code is TargetErrorCode.LINK_DENIED
    finally:
        close(root_fd)

    privileged = tmp_path / "privileged.txt"
    privileged.write_text("private")
    privileged.chmod(0o4644)
    privileged_profile = target_module._WorkerInspectionProfile(
        tmp_path,
        None,
        64,
        128,
    )
    privileged_witness = target_module._worker_capture_root(privileged_profile)
    with pytest.raises(TargetInspectionError) as metadata:
        target_module._inspect_many(
            privileged_profile,
            (LogicalPath("privileged.txt"),),
            privileged_witness,
        )
    assert metadata.value.code is TargetErrorCode.METADATA_DENIED
    privileged.chmod(0o644)

    def worker_unavailable(_path: Path) -> Never:
        """Raise the unavailable failure that inspect-many must translate."""
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)

    monkeypatch.setattr(target_module, "_open_directory", worker_unavailable)
    with pytest.raises(TargetInspectionError) as stale:
        target_module._inspect_many(
            privileged_profile,
            (LogicalPath("missing.txt"),),
            privileged_witness,
        )
    assert stale.value.code is TargetErrorCode.WITNESS_STALE

    def path_denied(_path: Path) -> Never:
        """Raise a non-worker target error that must remain unchanged."""
        raise TargetInspectionError(TargetErrorCode.PATH_DENIED)

    monkeypatch.setattr(target_module, "_open_directory", path_denied)
    with pytest.raises(TargetInspectionError) as denied:
        target_module._inspect_many(
            privileged_profile,
            (LogicalPath("missing.txt"),),
            privileged_witness,
        )
    assert denied.value.code is TargetErrorCode.PATH_DENIED

    source = tmp_path / "changed.txt"
    source.write_text("value")
    descriptor = open_fd(source, O_RDONLY)
    original_status = fstat(descriptor)
    changed_status = SimpleNamespace(
        st_dev=original_status.st_dev,
        st_ino=original_status.st_ino + 1,
        st_nlink=original_status.st_nlink,
        st_size=original_status.st_size,
    )
    statuses = iter((original_status, changed_status))
    monkeypatch.setattr(target_module, "fstat", lambda _fd: next(statuses))
    try:
        with pytest.raises(TargetInspectionError) as stale_read:
            target_module._read_bounded(descriptor, 16)
        assert stale_read.value.code is TargetErrorCode.WITNESS_STALE
    finally:
        close(descriptor)

    class Stream:
        """Provide a controllable byte stream for worker main failures."""

        def __init__(self, value: bytes) -> None:
            """Initialize one standard-stream replacement."""
            self.buffer = BytesIO(value)

    token = b"b" * 32
    monkeypatch.setattr(
        target_module,
        "environ",
        {target_module._WORKER_TOKEN_ENV: token.hex()},
    )
    for envelope in (
        b"[]",
        dumps({"payload": [], "mac": "x"}).encode(),
        dumps({"payload": {}, "mac": 1}).encode(),
        dumps({"payload": {}, "mac": "wrong"}).encode(),
    ):
        monkeypatch.setattr(target_module, "stdin", Stream(envelope))
        monkeypatch.setattr(target_module, "stdout", Stream(b""))
        assert target_module._worker_main() == 2

    authority = _test_runtime_authority(tmp_path)
    valid_payload: dict[str, object] = {
        "operation": "unknown",
        "root": str(tmp_path),
        "cwd": None,
        "maximum": 16,
        "aggregate_maximum": 32,
        "authority_signature": b64encode(authority._signature).decode(),
    }
    assert target_module._worker_response(valid_payload) == {
        "error": TargetErrorCode.CAPABILITY_UNAVAILABLE.value
    }
    for encoded_signature in ("!", b64encode(b"x" * 64).decode()):
        candidate = dict(valid_payload)
        candidate["authority_signature"] = encoded_signature
        with pytest.raises(TargetInspectionError):
            target_module._worker_profile(candidate)
    with pytest.raises(TargetInspectionError):
        target_module._worker_root({"identity": [1, 2], "mount_id": "mount"})

    parent = ParentWitness(None, FileIdentity(1, 2), "mount")
    present = TargetSnapshot(
        LogicalPath("present.txt"),
        True,
        SourceBytes(b"value"),
        MetadataProfile(
            FileMode(0o644),
            False,
            "lf",
        ),
        FileIdentity(3, 4),
        1,
        parent,
    )
    assert target_module._snapshot_to_worker(present)["present"]

    class FailedStatFs:
        """Model Darwin fstatfs returning an untrusted failure."""

        argtypes: object
        restype: object

        def __call__(self, _fd: int, _buffer: object) -> int:
            """Return the native error result."""
            return -1

    class FailedLibc:
        """Expose the native fstatfs failure stub."""

        fstatfs = FailedStatFs()

    monkeypatch.setattr(target_module, "CDLL", lambda _name: FailedLibc())
    with pytest.raises(TargetInspectionError):
        target_module._mount_topology(-1)
    monkeypatch.delenv(target_module._WORKER_TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit) as worker_main:
        run_path(target_module.__file__, run_name="__main__")
    assert worker_main.value.code == 2
