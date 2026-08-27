"""Assert the planned Phase 12 approver review display stays typed."""

from typing import assert_type

from rich.text import Text

from avalan.patch.projection import (
    ApproverProjectionAuthority,
    ApproverProjectionBoundary,
)
from avalan.patch.review_display import (
    ApproverReviewViewAuthority,
    CompleteDiffPagination,
    ReviewPageIndex,
    TrustedReviewerActionPrompt,
    create_approver_review_view,
    render_review_ansi,
    render_review_json,
    render_review_plain,
    render_review_rich,
    review_pagination,
    trusted_reviewer_action_prompt,
)
from avalan.patch.review_display_codec import (
    ApproverReviewView,
    render_review_log,
)


def assert_approver_review_display_types(
    boundary: ApproverProjectionBoundary,
    authority: ApproverProjectionAuthority,
) -> None:
    """Require detached complete review to remain approver-authorized."""
    view, review_authority = create_approver_review_view(boundary, authority)
    page = ReviewPageIndex(0)
    assert_type(view, ApproverReviewView)
    assert_type(review_authority, ApproverReviewViewAuthority)
    assert_type(review_authority.correlation_id.value, str)
    assert_type(
        review_pagination(view, review_authority), CompleteDiffPagination
    )
    assert_type(
        trusted_reviewer_action_prompt(view, review_authority),
        TrustedReviewerActionPrompt,
    )
    assert_type(render_review_plain(view, review_authority, page), str)
    assert_type(render_review_ansi(view, review_authority, page), str)
    assert_type(render_review_rich(view, review_authority, page), Text)
    assert_type(render_review_json(view, review_authority, page), bytes)
    assert_type(render_review_log(view), bytes)
