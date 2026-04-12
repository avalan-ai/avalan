import asyncio
from contextlib import AsyncExitStack
from threading import Thread


def close_async_exit_stack(exit_stack: AsyncExitStack) -> None:
    """Close an async exit stack from sync code in a deterministic way."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(exit_stack.aclose())
        return

    def close_in_background() -> None:
        asyncio.run(exit_stack.aclose())

    thread = Thread(target=close_in_background, daemon=False)
    thread.start()
    thread.join()
