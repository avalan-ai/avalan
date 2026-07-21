"""Accept a strictly typed asynchronous input handler."""

from typing import assert_type

from avalan.interaction import (
    InputHandler,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
    InputRequest,
)


async def handle_input(context: InputHandlerContext) -> InputHandlerOutcome:
    assert_type(context.request, InputRequest)
    return InputHandlerDetached()


handler: InputHandler = handle_input
