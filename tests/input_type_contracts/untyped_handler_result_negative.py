"""Reject annotation-free asynchronous input handlers."""

from avalan.interaction import InputHandler


async def handle_input(context):
    return {"request_id": context.request.request_id}


handler: InputHandler = handle_input
