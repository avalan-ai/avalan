"""Reject cross-boundary local patch-review capability substitutions."""

from avalan.cli.patch_review import (
    DetachedPatchCliApproval,
    ExactPatchCliPreauthorization,
    LocalPatchReviewTestProfile,
    PatchCliReviewContinuation,
    create_local_patch_review_test_profile,
    resume_local_patch_review,
    run_local_patch_review,
)
from avalan.patch.toolset import PatchSdkHost


def reject_unprepared_host(host: PatchSdkHost) -> LocalPatchReviewTestProfile:
    """Pass a host where one opaque prepared review binding is required."""
    return create_local_patch_review_test_profile(host)


async def reject_wrong_headless_authority(
    profile: LocalPatchReviewTestProfile,
    detached: DetachedPatchCliApproval,
) -> None:
    """Pass detached approval in the exact preauthorization argument."""
    await run_local_patch_review(profile, preauthorization=detached)


async def reject_wrong_continuation(
    profile: LocalPatchReviewTestProfile,
    preauthorization: ExactPatchCliPreauthorization,
) -> None:
    """Pass a preauthorization where durable continuation is required."""
    await resume_local_patch_review(profile, preauthorization)


def reject_continuation_constructor() -> PatchCliReviewContinuation:
    """Construct one factory-only local pending continuation directly."""
    return PatchCliReviewContinuation(None)
