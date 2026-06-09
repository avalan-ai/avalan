from asyncio import run as asyncio_run
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


def run_async(coroutine: Coroutine[object, object, T]) -> T:
    try:
        return asyncio_run(coroutine)
    except RuntimeError as error:
        if (
            "asyncio.run() cannot be called from a running event loop"
            not in str(error)
        ):
            raise
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio_run, coroutine)
            return future.result()
