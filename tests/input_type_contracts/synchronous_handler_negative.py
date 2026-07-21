"""Reject synchronous callbacks at the public handler boundary."""

from avalan.interaction import (
    InputHandler,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
)


def handle_input(context: InputHandlerContext) -> InputHandlerOutcome:
    return InputHandlerDetached()


handler: InputHandler = handle_input
