"""Assert the planned local patch-review test profile remains typed."""

from typing import assert_type

from avalan.cli.patch_review import (
    DetachedPatchCliApproval,
    ExactPatchCliPreauthorization,
    LocalPatchReviewTestProfile,
    PatchCliReviewContinuation,
    PatchCliReviewResult,
    create_detached_patch_cli_approval,
    create_exact_patch_cli_preauthorization,
    create_local_patch_review_test_profile,
    prepare_local_patch_review_binding,
    read_local_patch_review_result,
    resume_local_patch_review,
    run_local_patch_review,
)
from avalan.patch.toolset import PatchSdkHost


async def assert_local_patch_review_types(
    host: PatchSdkHost,
    continuation: PatchCliReviewContinuation,
) -> None:
    """Require exact local review, approval, and continuation boundaries."""
    binding = await prepare_local_patch_review_binding(host)
    profile = create_local_patch_review_test_profile(binding)
    assert_type(profile, LocalPatchReviewTestProfile)
    assert_type(
        create_exact_patch_cli_preauthorization(profile),
        ExactPatchCliPreauthorization,
    )
    assert_type(
        create_detached_patch_cli_approval(profile),
        DetachedPatchCliApproval,
    )
    assert_type(
        await run_local_patch_review(profile),
        PatchCliReviewResult,
    )
    assert_type(
        await read_local_patch_review_result(profile),
        PatchCliReviewResult,
    )
    assert_type(
        await resume_local_patch_review(profile, continuation),
        PatchCliReviewResult,
    )
