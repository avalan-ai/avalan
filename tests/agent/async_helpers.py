from asyncio import get_running_loop
from asyncio import run as asyncio_run
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        get_running_loop()
    except RuntimeError:
        return asyncio_run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio_run, coroutine).result()
