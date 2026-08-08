"""Reject untyped callbacks at the patch strict boundary."""


def reject_untyped_callback(value):
    """Keep one deliberately unannotated callback for the fixture."""
    return value
