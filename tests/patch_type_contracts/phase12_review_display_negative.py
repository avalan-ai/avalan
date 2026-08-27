"""Reject cross-audience use of planned privileged review display values."""

from avalan.patch.projection import (
    ModelProjectionAuthority,
    ModelProjectionBoundary,
)
from avalan.patch.review_display import (
    ApproverReviewViewAuthority,
    ReviewPageIndex,
    create_approver_review_view,
    render_review_json,
    render_review_plain,
)
from avalan.patch.review_display_codec import ApproverReviewView


def reject_model_authority_as_reviewer(
    view: ApproverReviewView,
    authority: ModelProjectionAuthority,
) -> str:
    """Pass a model authority to a reviewer-only display function."""
    return render_review_plain(view, authority, ReviewPageIndex(0))


def reject_approver_authority_as_view_authority(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
) -> ApproverReviewViewAuthority:
    """Return a review view where a review authority is required."""
    return view


def reject_model_boundary_as_review_source(
    boundary: ModelProjectionBoundary,
    authority: ModelProjectionAuthority,
) -> tuple[ApproverReviewView, ApproverReviewViewAuthority]:
    """Construct an approver review from the wrong audience boundary."""
    return create_approver_review_view(boundary, authority)


def reject_log_as_complete_review(
    view: ApproverReviewView,
    authority: ApproverReviewViewAuthority,
) -> str:
    """Treat content-free generic logs as complete privileged JSON review."""
    return render_review_json(view, authority, ReviewPageIndex(0))
