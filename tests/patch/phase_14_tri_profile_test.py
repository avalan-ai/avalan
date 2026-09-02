"""Exercise Phase 14 physical selected runtime profile boundaries."""

from asyncio import run
from dataclasses import replace
from json import loads
from pathlib import Path
from runpy import run_path
from sys import path as sys_path

import pytest
from patch_activation_support import activated_patch_test_profile

from avalan.patch.container_target import (
    ContainerInspectionTarget,
    ContainerPatchRuntimeSettings,
    ContainerSharedRootAuthority,
    _ContainerRuntimeProcess,
    _docker_output,
    _shared_host_root,
    _shared_host_root_is_safe,
)
from avalan.patch.domain import ContextKind, LogicalPath, OperationType
from avalan.patch.durable_approval import HmacDurableApprovalAuthority
from avalan.patch.durable_store import (
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.target import (
    InspectionRequest,
    ScopeSelection,
    TargetErrorCode,
    TargetInspectionError,
)
from avalan.patch.toolset import PatchToolLoader


def _phase_eleven() -> dict[str, object]:
    """Load existing selected-container factories without copying helpers."""
    sys_path.insert(0, "tests/patch")
    try:
        return run_path("tests/patch/phase_11_contract_test.py")
    finally:
        sys_path.remove("tests/patch")


def test_patch_e2e_035_container_shared_root_is_test_only_and_physical(
    tmp_path: Path,
) -> None:
    """Bind the selected container service to the exact shared host root."""

    async def exercise() -> None:
        """Start one real Docker service and observe host-backed effects."""
        phase_eleven = _phase_eleven()
        image_factory = phase_eleven["_test_image"]
        settings_factory = phase_eleven["_settings"]
        configuration_factory = phase_eleven["_configuration"]
        binder_factory = phase_eleven["_binder"]
        clock_type = phase_eleven["_Clock"]
        assert callable(image_factory)
        assert callable(settings_factory)
        assert callable(configuration_factory)
        assert callable(binder_factory)
        assert isinstance(clock_type, type)
        root = tmp_path / "shared-workspace"
        root.mkdir()
        note = root / "container.txt"
        note.write_text("before\n", encoding="utf-8")
        try:
            image = await image_factory()
        except TargetInspectionError as unavailable:
            assert unavailable.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
            pytest.skip("sealed Docker worker image is unavailable locally")
        ordinary_value = settings_factory(root, image)
        assert isinstance(ordinary_value, ContainerPatchRuntimeSettings)
        ordinary = ordinary_value
        shared = replace(
            ordinary,
            shared_root_authority=ContainerSharedRootAuthority.from_bytes(
                b"s" * 32
            ),
        )
        with pytest.raises(TargetInspectionError) as inactive:
            replace(shared, test_profile=False)
        assert inactive.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
        linked = tmp_path / "shared-workspace-link"
        linked.symlink_to(root, target_is_directory=True)
        with pytest.raises(TargetInspectionError) as linked_root:
            replace(shared, seed_root=linked)
        assert linked_root.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
        authority = HmacDurableApprovalAuthority.random()
        store = InMemoryDurablePatchStore(
            InMemoryDurablePatchBackend(approval_verifier=authority)
        )
        configuration = configuration_factory(clock_type(), authority)
        binder = binder_factory(shared, configuration, store)
        try:
            bundle = await PatchToolLoader(
                binder,
                activated_patch_test_profile(),
            ).load(enable_tools=["patch.edit"])
        except TargetInspectionError as unavailable:
            assert unavailable.code in {
                TargetErrorCode.CAPABILITY_UNAVAILABLE,
                TargetErrorCode.WORKER_UNAVAILABLE,
            }
            assert note.read_text(encoding="utf-8") == "before\n"
            return
        assert bundle.toolset is not None
        await bundle.manager.__aenter__()
        try:
            scope = await binder.runtime.resolve(
                ScopeSelection(ContextKind.CONTAINER)
            )
            assert binder.runtime._process.volume_name is None
            container_id = binder.runtime._process._container_id
            assert container_id is not None
            inspected = await _docker_output(
                (
                    "docker",
                    "inspect",
                    container_id,
                )
            )
            assert inspected is not None
            rows = loads(inspected)
            assert type(rows) is list and len(rows) == 1
            row = rows[0]
            assert type(row) is dict
            host_config = row["HostConfig"]
            assert type(host_config) is dict
            assert host_config["ReadonlyRootfs"] is True
            assert host_config["NetworkMode"] == "none"
            mounts = row["Mounts"]
            assert type(mounts) is list
            workspace_mounts = [
                item
                for item in mounts
                if type(item) is dict
                and item.get("Destination") == "/workspace"
            ]
            assert len(workspace_mounts) == 1
            assert workspace_mounts[0]["Type"] == "bind"
            assert workspace_mounts[0]["RW"] is True
            outcome = await bundle.toolset.sdk_host().invoke_json(
                OperationType.EDIT,
                {
                    "path": "container.txt",
                    "edits": [{"old_text": "before\n", "new_text": "after\n"}],
                },
            )
            assert outcome.status.value == "committed"
            assert note.read_text(encoding="utf-8") == "after\n"
            container = await ContainerInspectionTarget(
                binder.runtime
            ).inspect(
                InspectionRequest(scope, (LogicalPath("container.txt"),))
            )
            assert container.snapshots[0].bytes_value is not None
            assert container.snapshots[0].bytes_value._value == b"after\n"
        finally:
            await bundle.manager.__aexit__(None, None, None)
            container_id = binder.runtime._process._container_id
            await binder.runtime.dispose()
        assert container_id is not None
        assert (
            await _docker_output(("docker", "inspect", container_id), False)
            is None
        )
        assert note.read_text(encoding="utf-8") == "after\n"

    run(exercise())


def test_patch_phase_14_shared_root_authority_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject malformed, replaced, or unstatable shared-root witnesses."""
    phase_eleven = _phase_eleven()
    settings_factory = phase_eleven["_settings"]
    assert callable(settings_factory)
    root = tmp_path / "shared-root"
    root.mkdir()
    settings_value = settings_factory(root, "sha256:" + "a" * 64)
    assert isinstance(settings_value, ContainerPatchRuntimeSettings)
    with pytest.raises(TargetInspectionError) as malformed_authority:
        ContainerSharedRootAuthority.from_bytes(b"short")
    assert (
        malformed_authority.value.code
        is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )
    shared = replace(
        settings_value,
        shared_root_authority=ContainerSharedRootAuthority.from_bytes(
            b"a" * 32
        ),
    )
    assert _shared_host_root(shared) == root
    with pytest.raises(TargetInspectionError) as missing_authority:
        _shared_host_root(replace(shared, shared_root_authority=None))
    assert (
        missing_authority.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE
    )

    process = _ContainerRuntimeProcess(shared)
    process._shared_host_root = root
    process._shared_host_root_identity = (
        process._shared_host_root_identity_for(root)
    )
    process._require_shared_host_root()
    process._shared_host_root_identity = (0, 0)
    with pytest.raises(TargetInspectionError) as replaced:
        process._require_shared_host_root()
    assert replaced.value.code is TargetErrorCode.WITNESS_STALE

    def no_stat(self: Path, *, follow_symlinks: bool = True) -> object:
        """Model a host filesystem that cannot retain root metadata."""
        del self, follow_symlinks
        raise OSError("metadata unavailable")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "stat", no_stat)
        with pytest.raises(TargetInspectionError) as unavailable:
            process._shared_host_root_identity_for(root)
        assert unavailable.value.code is TargetErrorCode.WITNESS_STALE

    def no_resolve(self: Path, *, strict: bool = False) -> Path:
        """Model a root that vanishes between profile checks and use."""
        del self, strict
        raise OSError("root unavailable")

    with monkeypatch.context() as patched:
        patched.setattr(
            "avalan.patch.container_target._shared_host_root_is_safe",
            lambda value: value == root,
        )
        patched.setattr(Path, "resolve", no_resolve)
        with pytest.raises(TargetInspectionError) as unresolved:
            _shared_host_root(shared)
        assert unresolved.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE

    with monkeypatch.context() as patched:
        patched.setattr(
            ContainerSharedRootAuthority,
            "_root_receipt",
            lambda self, received_root, resource_digest: "",
        )
        with pytest.raises(TargetInspectionError) as unsealed:
            _shared_host_root(shared)
        assert unsealed.value.code is TargetErrorCode.CAPABILITY_UNAVAILABLE


def test_patch_phase_14_shared_root_safety_requires_live_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject a cached directory type without live host metadata."""
    root = tmp_path / "shared-root"
    root.mkdir()

    def cached_directory(self: Path) -> bool:
        """Model pathlib directory metadata cached before a stat failure."""
        del self
        return True

    def no_stat(self: Path, *, follow_symlinks: bool = True) -> object:
        """Model a host filesystem that cannot provide root metadata."""
        del self, follow_symlinks
        raise OSError("metadata unavailable")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "is_dir", cached_directory)
        patched.setattr(Path, "stat", no_stat)

        assert not _shared_host_root_is_safe(root)
