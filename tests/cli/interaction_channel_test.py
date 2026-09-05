from asyncio import (
    CancelledError,
    Event,
    create_task,
    run,
    sleep,
    wait_for,
)
from errno import EIO
from fcntl import ioctl
from os import (
    _exit,
    dup2,
    pipe,
    set_blocking,
    setsid,
    ttyname,
)
from os import (
    close as close_fd,
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
from pathlib import Path
from pty import openpty
from subprocess import Popen
from sys import argv, executable
from tempfile import NamedTemporaryFile
from termios import TIOCSCTTY
from tty import setraw
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

from avalan.cli import interaction_channel as channel_module
from avalan.cli.interaction_channel import CliInteractionChannel


def _run_default_channel_child(
    slave: int,
    stdin_read: int,
    result_write: int,
) -> None:
    """Open the default channel in one fresh PTY child interpreter."""
    exit_code = 1
    try:
        setsid()
        ioctl(slave, TIOCSCTTY, 0)
        dup2(stdin_read, 0)
        dup2(slave, 1)
        dup2(slave, 2)
        close_fd(stdin_read)
        close_fd(slave)

        async def open_default() -> bool:
            channel = CliInteractionChannel.open()
            if channel is None:
                return False
            await channel.aclose()
            return True

        opened = run(open_default())
        write_fd(result_write, b"opened" if opened else b"unavailable")
        exit_code = 0 if opened else 2
    except Exception as error:
        write_fd(
            result_write,
            f"{type(error).__name__}: {error}".encode(),
        )
    finally:
        close_fd(result_write)
        _exit(exit_code)


class CliInteractionChannelTestCase(IsolatedAsyncioTestCase):
    master: int
    slave: int
    channel: CliInteractionChannel

    async def asyncSetUp(self) -> None:
        self.master, self.slave = openpty()
        setraw(self.slave)
        set_blocking(self.master, False)
        opened = CliInteractionChannel.open(ttyname(self.slave))
        self.assertIsNotNone(opened)
        assert opened is not None
        self.channel = opened

    async def asyncTearDown(self) -> None:
        if not self.channel.closed:
            await self.channel.aclose()
        for descriptor in (self.master, self.slave):
            if descriptor >= 0:
                close_fd(descriptor)

    async def test_reads_lines_and_writes_control_output(self) -> None:
        write_fd(self.master, b"first\r\nsecond\n")

        self.assertEqual(await wait_for(self.channel.read_line(), 1), "first")
        self.assertEqual(await wait_for(self.channel.read_line(), 1), "second")

        await self.channel.write("prompt")
        self.assertEqual(read_fd(self.master, 64), b"prompt")

    async def test_blocked_read_keeps_event_loop_heartbeat_live(self) -> None:
        read_task = create_task(self.channel.read_line())

        ticks = 0
        for _ in range(8):
            await sleep(0)
            ticks += 1

        self.assertEqual(ticks, 8)
        self.assertFalse(read_task.done())
        write_fd(self.master, b"answer\n")
        self.assertEqual(await wait_for(read_task, 1), "answer")

    async def test_cancelled_reader_preserves_bytes_for_next_prompt(
        self,
    ) -> None:
        bytes_read = Event()
        real_read_fd = channel_module.read_fd

        def observed_read(descriptor: int, size: int) -> bytes:
            result = real_read_fd(descriptor, size)
            if result:
                bytes_read.set()
            return result

        with patch.object(
            channel_module,
            "read_fd",
            side_effect=observed_read,
        ):
            cancelled = create_task(self.channel.read_line())
            write_fd(self.master, b"part")
            await wait_for(bytes_read.wait(), 1)
            self.assertFalse(cancelled.done())

            cancelled.cancel()
            with self.assertRaises(CancelledError):
                await cancelled

            write_fd(self.master, b"ial\n")
            self.assertEqual(
                await wait_for(self.channel.read_line(), 1),
                "partial",
            )

    async def test_aclose_wakes_blocked_read_and_is_idempotent(self) -> None:
        read_task = create_task(self.channel.read_line())
        await sleep(0)

        await self.channel.aclose()

        self.assertIsNone(await wait_for(read_task, 1))
        self.assertTrue(self.channel.closed)
        self.assertIsNone(await self.channel.read_line())
        await self.channel.aclose()
        with self.assertRaisesRegex(RuntimeError, "channel is closed"):
            await self.channel.write("late")
        with self.assertRaisesRegex(RuntimeError, "channel is closed"):
            async with self.channel:
                self.fail("closed channel entered")

    async def test_terminal_disappearance_returns_eof_without_hanging(
        self,
    ) -> None:
        read_task = create_task(self.channel.read_line())
        await sleep(0)
        close_fd(self.master)
        self.master = -1

        self.assertIsNone(await wait_for(read_task, 1))
        self.assertTrue(self.channel.closed)
        self.assertIsNone(await self.channel.read_line())

    async def test_disappearance_returns_buffered_tail_once(self) -> None:
        bytes_read = Event()
        real_read_fd = channel_module.read_fd

        def observed_read(descriptor: int, size: int) -> bytes:
            result = real_read_fd(descriptor, size)
            if result:
                bytes_read.set()
            return result

        with patch.object(
            channel_module,
            "read_fd",
            side_effect=observed_read,
        ):
            read_task = create_task(self.channel.read_line())
            write_fd(self.master, b"tail")
            await wait_for(bytes_read.wait(), 1)
            close_fd(self.master)
            self.master = -1

            self.assertEqual(await wait_for(read_task, 1), "tail")
            self.assertIsNone(await self.channel.read_line())

    async def test_rejects_concurrent_reader(self) -> None:
        first = create_task(self.channel.read_line())
        await sleep(0)

        with self.assertRaisesRegex(RuntimeError, "already has a reader"):
            await self.channel.read_line()

        first.cancel()
        with self.assertRaises(CancelledError):
            await first

    async def test_read_retries_interrupt_and_handles_not_ready(self) -> None:
        with patch.object(
            channel_module,
            "read_fd",
            side_effect=BlockingIOError,
        ):
            self.assertTrue(self.channel._read_available_chunk())

        with patch.object(
            channel_module,
            "read_fd",
            side_effect=(InterruptedError(), b"chunk"),
        ):
            self.assertTrue(self.channel._read_available_chunk())
        self.assertEqual(self.channel._take_buffered_tail(), "chunk")

    async def test_read_terminal_disappearance_is_deterministic(
        self,
    ) -> None:
        with patch.object(
            channel_module,
            "read_fd",
            side_effect=OSError(EIO, "terminal gone"),
        ):
            self.assertFalse(self.channel._read_available_chunk())
        self.assertTrue(self.channel.closed)

    async def test_unexpected_read_error_closes_then_propagates(self) -> None:
        error = OSError(999, "unexpected")
        with (
            patch.object(channel_module, "read_fd", side_effect=error),
            self.assertRaises(OSError) as raised,
        ):
            self.channel._read_available_chunk()
        self.assertIs(raised.exception, error)
        self.assertTrue(self.channel.closed)

    async def test_readiness_failure_closes_instead_of_hanging(self) -> None:
        with patch.object(
            self.channel._loop,
            "add_reader",
            side_effect=NotImplementedError,
        ):
            self.assertIsNone(await self.channel.read_line())
        self.assertTrue(self.channel.closed)

    async def test_write_retries_readiness_and_interrupts(self) -> None:
        with patch.object(
            channel_module,
            "write_fd",
            side_effect=(BlockingIOError(), 3),
        ):
            await wait_for(self.channel.write("one"), 1)

        with patch.object(
            channel_module,
            "write_fd",
            side_effect=(InterruptedError(), 3),
        ):
            await self.channel.write("two")

        with patch.object(channel_module, "write_fd", return_value=0):
            await self.channel.write("stop")
        self.assertTrue(self.channel.closed)

    async def test_repeated_readiness_callbacks_are_inert(self) -> None:
        def signal_twice(descriptor: int, callback: object) -> None:
            _ = descriptor
            assert callable(callback)
            callback()
            callback()

        with (
            patch.object(
                self.channel._loop,
                "add_reader",
                side_effect=signal_twice,
            ),
            patch.object(self.channel._loop, "remove_reader"),
        ):
            self.assertTrue(await self.channel._wait_until_readable())

        with (
            patch.object(
                self.channel._loop,
                "add_writer",
                side_effect=signal_twice,
            ),
            patch.object(self.channel._loop, "remove_writer"),
        ):
            self.assertTrue(await self.channel._wait_until_writable())

    async def test_write_disappearance_closes_without_error(
        self,
    ) -> None:
        with patch.object(
            channel_module,
            "write_fd",
            side_effect=OSError(EIO, "terminal gone"),
        ):
            await self.channel.write("prompt")
        self.assertTrue(self.channel.closed)

    async def test_unexpected_write_error_closes_then_propagates(self) -> None:
        error = OSError(999, "unexpected")
        with (
            patch.object(channel_module, "write_fd", side_effect=error),
            self.assertRaises(OSError) as raised,
        ):
            await self.channel.write("prompt")
        self.assertIs(raised.exception, error)
        self.assertTrue(self.channel.closed)

    async def test_write_readiness_failure_is_unavailable(self) -> None:
        with (
            patch.object(
                channel_module,
                "write_fd",
                side_effect=BlockingIOError,
            ),
            patch.object(
                self.channel._loop,
                "add_writer",
                side_effect=NotImplementedError,
            ),
        ):
            await self.channel.write("prompt")
        self.assertTrue(self.channel.closed)

    async def test_close_interrupts_registered_writer(self) -> None:
        with (
            patch.object(
                channel_module,
                "write_fd",
                side_effect=BlockingIOError,
            ),
            patch.object(self.channel._loop, "add_writer"),
            patch.object(self.channel._loop, "remove_writer") as remove_writer,
        ):
            write_task = create_task(self.channel.write("prompt"))
            await sleep(0)
            self.assertTrue(self.channel._write_registered)

            await self.channel.aclose()

            await wait_for(write_task, 1)
            remove_writer.assert_called_once_with(self.channel._descriptor)

    async def test_async_context_closes_channel(self) -> None:
        async with self.channel as entered:
            self.assertIs(entered, self.channel)
            self.assertFalse(entered.closed)

        self.assertTrue(self.channel.closed)

    async def test_missing_non_terminal_and_unsupported_backend_unavailable(
        self,
    ) -> None:
        self.assertIsNone(
            CliInteractionChannel.open("/definitely/missing/avalan-tty")
        )
        with NamedTemporaryFile() as regular:
            self.assertIsNone(CliInteractionChannel.open(regular.name))

        with (
            patch.object(
                channel_module,
                "_supports_fd_readiness",
                return_value=False,
            ),
            patch.object(
                channel_module,
                "close_fd",
                wraps=close_fd,
            ) as close_mock,
        ):
            self.assertIsNone(CliInteractionChannel.open(ttyname(self.slave)))
        close_mock.assert_called_once()

    async def test_default_alias_falls_back_to_concrete_terminal(self) -> None:
        first_descriptor = open_fd(
            ttyname(self.slave),
            channel_module._OPEN_FLAGS,
        )
        second_descriptor = open_fd(
            ttyname(self.slave), channel_module._OPEN_FLAGS
        )
        with (
            patch.object(
                channel_module,
                "_control_terminal_paths",
                return_value=("/dev/tty", ttyname(self.slave)),
            ),
            patch.object(
                channel_module,
                "open_fd",
                side_effect=(first_descriptor, second_descriptor),
            ),
            patch.object(
                channel_module,
                "_supports_fd_readiness",
                side_effect=(False, True),
            ),
            patch.object(
                channel_module,
                "close_fd",
                wraps=close_fd,
            ) as close_mock,
        ):
            opened = CliInteractionChannel.open()
            self.assertIsNotNone(opened)
            assert opened is not None
            self.assertEqual(opened._descriptor, second_descriptor)
            close_mock.assert_called_once_with(first_descriptor)
            await opened.aclose()

    async def test_open_uses_locale_encoding_fallback(self) -> None:
        with (
            patch.object(channel_module, "device_encoding", return_value=None),
            patch.object(
                channel_module,
                "getpreferredencoding",
                return_value="ascii",
            ),
        ):
            fallback = CliInteractionChannel.open(ttyname(self.slave))
        self.assertIsNotNone(fallback)
        assert fallback is not None
        self.assertEqual(fallback._encoding, "ascii")
        await fallback.aclose()

    async def test_different_event_loop_is_rejected(self) -> None:
        with patch.object(
            channel_module,
            "get_running_loop",
            return_value=object(),
        ):
            with self.assertRaisesRegex(RuntimeError, "different event loop"):
                await self.channel.read_line()


class CliInteractionChannelOpenTestCase(TestCase):
    def test_default_path_opens_with_piped_stdin_and_controlling_pty(
        self,
    ) -> None:
        master, slave = openpty()
        stdin_read, stdin_write = pipe()
        result_read, result_write = pipe()
        process = Popen(
            (
                executable,
                str(Path(__file__).resolve()),
                "--default-channel-child",
                str(slave),
                str(stdin_read),
                str(result_write),
            ),
            pass_fds=(slave, stdin_read, result_write),
        )

        try:
            close_fd(slave)
            close_fd(stdin_read)
            close_fd(stdin_write)
            close_fd(result_write)
            status = process.wait(timeout=15)
            result = read_fd(result_read, 4096)
            self.assertEqual(status, 0, result.decode())
            self.assertEqual(result, b"opened")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            close_fd(master)
            close_fd(result_read)

    def test_open_without_running_loop_is_unavailable_and_closes_fd(
        self,
    ) -> None:
        master, slave = openpty()
        try:
            self.assertIsNone(CliInteractionChannel.open(ttyname(slave)))
        finally:
            close_fd(master)
            close_fd(slave)

    def test_default_path_uses_platform_equivalent_on_windows(self) -> None:
        with patch.object(channel_module, "platform_name", "nt"):
            self.assertEqual(
                channel_module._control_terminal_paths("/dev/tty"),
                ("CON",),
            )
            self.assertEqual(
                channel_module._control_terminal_paths("custom"),
                ("custom",),
            )

    def test_posix_default_adds_concrete_standard_terminal_paths(self) -> None:
        with (
            patch.object(channel_module, "platform_name", "posix"),
            patch.object(
                channel_module,
                "_controlling_terminal_name",
                side_effect=(None, "/dev/pts/7", "/dev/pts/7"),
            ),
        ):
            self.assertEqual(
                channel_module._control_terminal_paths("/dev/tty"),
                ("/dev/tty", "/dev/pts/7"),
            )

    def test_posix_default_ignores_unresolvable_standard_terminal(
        self,
    ) -> None:
        with (
            patch.object(channel_module, "platform_name", "posix"),
            patch.object(
                channel_module,
                "_controlling_terminal_name",
                side_effect=(None, None, "/dev/tty"),
            ),
        ):
            self.assertEqual(
                channel_module._control_terminal_paths("/dev/tty"),
                ("/dev/tty",),
            )

    def test_standard_terminal_name_requires_controlling_terminal(
        self,
    ) -> None:
        os_module = channel_module.modules["os"]
        with (
            patch.object(channel_module, "isatty", return_value=True),
            patch.object(os_module, "tcgetpgrp", return_value=41) as process,
            patch.object(
                os_module,
                "ttyname",
                return_value="/dev/pts/7",
            ) as name,
        ):
            self.assertEqual(
                channel_module._controlling_terminal_name(2),
                "/dev/pts/7",
            )
        process.assert_called_once_with(2)
        name.assert_called_once_with(2)

        with (
            patch.object(channel_module, "isatty", return_value=False),
            patch.object(os_module, "tcgetpgrp") as process,
            patch.object(os_module, "ttyname") as name,
        ):
            self.assertIsNone(channel_module._controlling_terminal_name(1))
        process.assert_not_called()
        name.assert_not_called()

    def test_standard_terminal_name_fails_closed(self) -> None:
        os_module = channel_module.modules["os"]
        for error in (NotImplementedError(), OSError(), ValueError()):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(channel_module, "isatty", return_value=True),
                patch.object(os_module, "tcgetpgrp", side_effect=error),
                patch.object(
                    os_module,
                    "ttyname",
                    return_value="/dev/pts/7",
                ),
            ):
                self.assertIsNone(channel_module._controlling_terminal_name(0))

        with (
            patch.object(channel_module, "isatty", return_value=True),
            patch.object(os_module, "tcgetpgrp", return_value=41),
            patch.object(os_module, "ttyname", return_value=""),
        ):
            self.assertIsNone(channel_module._controlling_terminal_name(0))

        with (
            patch.object(channel_module, "isatty", return_value=True),
            patch.object(os_module, "tcgetpgrp", None),
            patch.object(os_module, "ttyname", None),
        ):
            self.assertIsNone(channel_module._controlling_terminal_name(0))

    def test_fd_readiness_probe_fails_closed(self) -> None:
        fake_loop = MagicMock()
        fake_loop.add_reader.side_effect = NotImplementedError

        self.assertFalse(channel_module._supports_fd_readiness(fake_loop, 1))
        fake_loop.remove_reader.assert_not_called()
        fake_loop.remove_writer.assert_not_called()

    def test_fd_readiness_probe_cleans_registered_reader_and_writer(
        self,
    ) -> None:
        reader_loop = MagicMock()
        reader_loop.add_reader.side_effect = lambda descriptor, callback: (
            callback()
        )
        reader_loop.remove_reader.side_effect = (ValueError(), None)
        self.assertFalse(channel_module._supports_fd_readiness(reader_loop, 2))
        self.assertEqual(reader_loop.remove_reader.call_count, 2)

        writer_loop = MagicMock()
        writer_loop.add_reader.side_effect = lambda descriptor, callback: (
            callback()
        )
        writer_loop.add_writer.side_effect = lambda descriptor, callback: (
            callback()
        )
        writer_loop.remove_writer.side_effect = (ValueError(), None)
        self.assertFalse(channel_module._supports_fd_readiness(writer_loop, 3))
        self.assertEqual(writer_loop.remove_writer.call_count, 2)

    def test_optional_open_flag_rejects_non_integer(self) -> None:
        os_module = channel_module.modules["os"]
        with patch.object(
            os_module,
            "_AVALAN_TEST_OPEN_FLAG",
            "invalid",
            create=True,
        ):
            self.assertEqual(
                channel_module._optional_open_flag("_AVALAN_TEST_OPEN_FLAG"),
                0,
            )


def _run_child_from_argv() -> None:
    """Dispatch one fresh-interpreter channel child invocation."""
    assert len(argv) == 5
    assert argv[1] == "--default-channel-child"
    _run_default_channel_child(
        int(argv[2]),
        int(argv[3]),
        int(argv[4]),
    )


if __name__ == "__main__":
    _run_child_from_argv()
