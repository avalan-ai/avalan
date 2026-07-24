from asyncio import (
    AbstractEventLoop,
    Future,
    Lock,
    get_running_loop,
)
from errno import EBADF, EIO, ENODEV, ENXIO, EPIPE
from locale import getpreferredencoding
from os import (
    O_RDWR,
    device_encoding,
    isatty,
    set_blocking,
)
from os import (
    close as close_fd,
)
from os import (
    name as platform_name,
)
from os import (
    open as open_fd,
)
from os import (
    read as read_fd,
)
from os import (
    write as write_fd,
)
from sys import modules
from types import TracebackType
from typing import Protocol


def _optional_open_flag(name: str) -> int:
    value: object = getattr(modules["os"], name, 0)
    return value if isinstance(value, int) else 0


_DEFAULT_TTY_PATH = "/dev/tty"
_READ_SIZE = 4096
_OPEN_FLAGS = (
    O_RDWR
    | _optional_open_flag("O_CLOEXEC")
    | _optional_open_flag("O_NONBLOCK")
    | _optional_open_flag("O_NOCTTY")
)
_TERMINAL_GONE_ERRNOS = frozenset((EBADF, EIO, ENODEV, ENXIO))
_TERMINAL_WRITE_GONE_ERRNOS = _TERMINAL_GONE_ERRNOS | frozenset((EPIPE,))


class CliInteractionChannelProtocol(Protocol):
    """Provide the async terminal operations used by CLI renderers."""

    async def read_line(self) -> str | None:
        """Read one line or return ``None`` when the terminal is gone."""
        ...

    async def write(self, text: str) -> None:
        """Write text to the interactive control terminal."""
        ...

    async def aclose(self) -> None:
        """Close the interactive control terminal."""
        ...


class CliInteractionChannel:
    """Own one cancellable asynchronous CLI control-terminal channel."""

    def __init__(
        self,
        descriptor: int,
        loop: AbstractEventLoop,
        encoding: str,
    ) -> None:
        assert isinstance(descriptor, int)
        assert descriptor >= 0
        assert isinstance(encoding, str)
        assert encoding
        self._descriptor = descriptor
        self._loop = loop
        self._encoding = encoding
        self._buffer = bytearray()
        self._read_waiter: Future[None] | None = None
        self._write_waiter: Future[None] | None = None
        self._read_active = False
        self._read_registered = False
        self._write_registered = False
        self._closed = False
        self._write_lock = Lock()

    @classmethod
    def open(
        cls,
        tty_path: str = _DEFAULT_TTY_PATH,
    ) -> "CliInteractionChannel | None":
        """Open a controlling terminal with async FD readiness."""
        assert isinstance(tty_path, str)
        path = _control_terminal_path(tty_path)
        try:
            descriptor = open_fd(path, _OPEN_FLAGS)
        except OSError:
            return None

        try:
            if not isatty(descriptor):
                close_fd(descriptor)
                return None
            set_blocking(descriptor, False)
            loop = get_running_loop()
            if not _supports_fd_readiness(loop, descriptor):
                close_fd(descriptor)
                return None
            encoding = device_encoding(descriptor) or getpreferredencoding(
                False
            )
            return cls(descriptor, loop, encoding)
        except (NotImplementedError, OSError, RuntimeError, ValueError):
            close_fd(descriptor)
            return None

    @property
    def closed(self) -> bool:
        """Return whether the channel no longer owns a terminal descriptor."""
        return self._closed

    async def __aenter__(self) -> "CliInteractionChannel":
        self._ensure_loop()
        if self._closed:
            raise RuntimeError("CLI interaction channel is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def read_line(self) -> str | None:
        """Read one terminal line without blocking the event loop."""
        self._ensure_loop()
        if self._read_active:
            raise RuntimeError("CLI interaction channel already has a reader")

        self._read_active = True
        try:
            while True:
                line = self._take_buffered_line()
                if line is not None:
                    return line
                if self._closed:
                    return self._take_buffered_tail()
                if not await self._wait_until_readable():
                    return self._take_buffered_tail()
                if not self._read_available_chunk():
                    return self._take_buffered_tail()
        finally:
            self._read_active = False

    async def write(self, text: str) -> None:
        """Write terminal text without blocking the event loop."""
        assert isinstance(text, str)
        self._ensure_loop()
        data = text.encode(self._encoding, errors="replace")
        async with self._write_lock:
            if self._closed:
                raise RuntimeError("CLI interaction channel is closed")

            offset = 0
            while offset < len(data):
                try:
                    written = write_fd(
                        self._descriptor,
                        data[offset:],
                    )
                except BlockingIOError:
                    if not await self._wait_until_writable():
                        return
                    continue
                except InterruptedError:
                    continue
                except OSError as error:
                    self._finish_terminal()
                    if error.errno in _TERMINAL_WRITE_GONE_ERRNOS:
                        return
                    raise

                if written <= 0:
                    self._finish_terminal()
                    return
                offset += written

    async def aclose(self) -> None:
        """Close the descriptor and wake any blocked channel operation."""
        self._ensure_loop()
        self._buffer.clear()
        self._finish_terminal()

    def _take_buffered_line(self) -> str | None:
        newline = self._buffer.find(b"\n")
        if newline < 0:
            return None
        line = bytes(self._buffer[:newline])
        del self._buffer[: newline + 1]
        if line.endswith(b"\r"):
            line = line[:-1]
        return line.decode(self._encoding, errors="replace")

    def _take_buffered_tail(self) -> str | None:
        if not self._buffer:
            return None
        tail = bytes(self._buffer)
        self._buffer.clear()
        return tail.decode(self._encoding, errors="replace")

    def _read_available_chunk(self) -> bool:
        while True:
            try:
                chunk = read_fd(self._descriptor, _READ_SIZE)
            except BlockingIOError:
                return True
            except InterruptedError:
                continue
            except OSError as error:
                self._finish_terminal()
                if error.errno in _TERMINAL_GONE_ERRNOS:
                    return False
                raise

            if not chunk:
                self._finish_terminal()
                return False
            self._buffer.extend(chunk)
            return True

    async def _wait_until_readable(self) -> bool:
        descriptor = self._descriptor
        waiter = self._loop.create_future()
        self._read_waiter = waiter

        def ready() -> None:
            if not waiter.done():
                waiter.set_result(None)

        try:
            self._loop.add_reader(descriptor, ready)
            self._read_registered = True
        except (NotImplementedError, OSError, ValueError):
            self._read_waiter = None
            self._finish_terminal()
            return False
        try:
            await waiter
        finally:
            if self._read_registered:
                self._loop.remove_reader(descriptor)
                self._read_registered = False
            self._read_waiter = None
        return not self._closed

    async def _wait_until_writable(self) -> bool:
        descriptor = self._descriptor
        waiter = self._loop.create_future()
        self._write_waiter = waiter

        def ready() -> None:
            if not waiter.done():
                waiter.set_result(None)

        try:
            self._loop.add_writer(descriptor, ready)
            self._write_registered = True
        except (NotImplementedError, OSError, ValueError):
            self._write_waiter = None
            self._finish_terminal()
            return False
        try:
            await waiter
        finally:
            if self._write_registered:
                self._loop.remove_writer(descriptor)
                self._write_registered = False
            self._write_waiter = None
        return not self._closed

    def _finish_terminal(self) -> None:
        if self._closed:
            return
        descriptor = self._descriptor
        self._closed = True
        try:
            if self._read_registered:
                self._loop.remove_reader(descriptor)
                self._read_registered = False
        finally:
            try:
                if self._write_registered:
                    self._loop.remove_writer(descriptor)
                    self._write_registered = False
            finally:
                try:
                    close_fd(descriptor)
                finally:
                    for waiter in (self._read_waiter, self._write_waiter):
                        if waiter is not None and not waiter.done():
                            waiter.set_result(None)

    def _ensure_loop(self) -> None:
        if get_running_loop() is not self._loop:
            raise RuntimeError(
                "CLI interaction channel used from a different event loop"
            )


def _control_terminal_path(tty_path: str) -> str:
    if platform_name == "nt" and tty_path == _DEFAULT_TTY_PATH:
        return "CON"
    return tty_path


def _supports_fd_readiness(
    loop: AbstractEventLoop,
    descriptor: int,
) -> bool:
    def ready() -> None:
        return None

    reader_registered = False
    writer_registered = False
    try:
        loop.add_reader(descriptor, ready)
        reader_registered = True
        loop.remove_reader(descriptor)
        reader_registered = False
        loop.add_writer(descriptor, ready)
        writer_registered = True
        loop.remove_writer(descriptor)
        writer_registered = False
    except (NotImplementedError, OSError, ValueError):
        if reader_registered:
            loop.remove_reader(descriptor)
        if writer_registered:
            loop.remove_writer(descriptor)
        return False
    return True
