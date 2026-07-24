"""Reject synchronous callbacks at the public resumer boundary."""

from avalan.interaction import InputResumer, InputResumptionNotification


def resume_input(notification: InputResumptionNotification) -> None:
    del notification


resumer: InputResumer = resume_input
