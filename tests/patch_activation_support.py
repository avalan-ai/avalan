"""Provide explicit sealed activation factories for PATCH integration tests."""

from asyncio import run
from dataclasses import replace
from pathlib import Path
from platform import machine
from secrets import token_bytes
from sys import platform as runtime_platform

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

import avalan.patch.target as target_module
from avalan.patch.activation import (
    PatchActivationPlatform,
    PatchActivationRuntimeFactory,
    PatchCapabilityProfile,
    PatchProfileProofs,
    PatchProfileState,
    _build_activation_factory,
    _build_activation_verifier,
    _manifest,
    _new_activation_authority,
    build_patch_production_manifest,
)
from avalan.patch.container_target import _docker_output
from avalan.patch.domain import (
    ByteSize,
    ContextKind,
    DurationTicks,
    PatchApprovalId,
    PatchContextId,
    PatchDomainId,
    PatchLimits,
    PatchProtocolId,
    PatchTargetId,
    PatchWorkspaceId,
)
from avalan.patch.target import (
    LocalPlatformProfile,
    LocalTargetProfile,
    TargetIdentity,
    TrustedLocalRoot,
)
from avalan.patch.toolset import PatchTestHostProfile


def patch_container_test_image() -> str:
    """Build one offline, architecture-matched PATCH container test image."""
    return run(_build_patch_container_test_image())


async def _build_patch_container_test_image() -> str:
    """Build the sealed test image from committed wheels and base bytes."""
    architectures = {
        "aarch64": "arm64",
        "amd64": "amd64",
        "arm64": "arm64",
        "x86_64": "amd64",
    }
    architecture = architectures.get(machine().lower())
    assert architecture is not None, "unsupported Docker test architecture"
    fixtures = Path(__file__).resolve().parent / "fixtures" / "patch"
    image = await _docker_output(
        (
            "docker",
            "build",
            "--quiet",
            "--network=none",
            "--pull=false",
            "--build-arg",
            "TARGETARCH=" + architecture,
            "--file",
            str(fixtures / "container_worker.Dockerfile"),
            str(fixtures),
        )
    )
    assert image is not None
    return image.strip()


def patch_test_activation_factory(
    profiles: tuple[PatchCapabilityProfile, ...] | None = None,
) -> PatchActivationRuntimeFactory:
    """Return one test-only factory over explicitly sealed profiles."""
    production = build_patch_production_manifest()
    base = production.profiles[0]
    proofs = PatchProfileProofs(
        context=True,
        platform=True,
        filesystem=True,
        target=True,
        protocol=True,
        policy=True,
        approval=True,
        persistence=True,
        surface=True,
        provider_codec=True,
    )
    nonce = token_bytes(8).hex()
    selected_profiles = (
        tuple(
            replace(
                base,
                key=replace(base.key, context=context, platform=platform),
                proofs=proofs,
                state=PatchProfileState.SELECTED,
                selection_rationale=(
                    "Explicit sealed PATCH test runtime " + nonce
                ),
            )
            for context in ContextKind
            for platform in PatchActivationPlatform
        )
        if profiles is None
        else profiles
    )
    manifest = _manifest(
        sources=production.sources,
        schemas=production.schemas,
        protocols=production.protocols,
        profiles=selected_profiles,
    )
    return _build_activation_factory(
        manifest,
        _build_activation_verifier(
            manifest,
            _new_activation_authority(token_bytes(32)),
            production=False,
        ),
    )


def activated_patch_test_profile() -> PatchTestHostProfile:
    """Return a test host profile retaining an explicit sealed factory."""
    return PatchTestHostProfile(
        enabled=True,
        authenticated=True,
        activation_factory=patch_test_activation_factory(),
    )


def phase15_local_target_profile(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> LocalTargetProfile:
    """Return a sealed real local target profile for Phase 15 adapters."""
    signer = Ed25519PrivateKey.generate()
    verifier = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    original_bootstrap = target_module._WORKER_BOOTSTRAP
    child_bootstrap = original_bootstrap.replace(
        "from avalan.patch.target import _worker_main\n"
        "raise SystemExit(_worker_main())",
        "import avalan.patch.target as target_module\n"
        "target_module._RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES = "
        + repr(verifier)
        + "\nraise SystemExit(target_module._worker_main())",
    )
    assert child_bootstrap != original_bootstrap
    monkeypatch.setattr(
        target_module, "_RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES", verifier
    )
    monkeypatch.setattr(target_module, "_WORKER_BOOTSTRAP", child_bootstrap)
    authority = target_module._RuntimeTargetAuthority(
        signer.sign(target_module._runtime_target_authority_message(root))
    )
    trusted_root = TrustedLocalRoot(root, _runtime_authority=authority)
    witness = target_module._capture_root_witness(trusted_root)
    identity = TargetIdentity(
        PatchContextId("context_" + "f" * 16),
        PatchWorkspaceId("workspace_" + "f" * 16),
        PatchDomainId("domain_" + "f" * 16),
        PatchTargetId("target_" + "f" * 16),
        PatchProtocolId("protocol_" + "f" * 16),
        witness.filesystem_id,
        witness.mount_id,
        "policy-phase15",
        "workspace-lease-phase15",
        PatchApprovalId("approval_" + "f" * 16),
    )
    namespace = root.parent / ".avalan-patch-phase15-private"
    namespace.mkdir(mode=0o700)
    return LocalTargetProfile(
        identity,
        trusted_root,
        None,
        PatchLimits(
            ByteSize(10_000),
            ByteSize(20),
            ByteSize(512),
            ByteSize(20),
            ByteSize(20),
            ByteSize(10_000),
            ByteSize(10_000),
            ByteSize(10_000),
            DurationTicks(100),
            DurationTicks(100),
            DurationTicks(100),
        ),
        ByteSize(10_000),
        _runtime_authority=authority,
        platform=(
            LocalPlatformProfile.DARWIN
            if runtime_platform == "darwin"
            else LocalPlatformProfile.LINUX
        ),
        mutation_test_profile=True,
        commit_namespace=namespace,
    )
