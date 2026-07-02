from asyncio import CancelledError, create_task, sleep, wait_for
from contextlib import suppress
from json import dumps
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event as ThreadEvent
from threading import Lock as ThreadLock
from time import perf_counter
from time import sleep as blocking_sleep
from tracemalloc import get_traced_memory, start, stop
from unittest import IsolatedAsyncioTestCase, main

from avalan.skill import (
    SkillAsyncFileSystem,
    SkillConfiguredSource,
    SkillCursorLimits,
    SkillDiagnosticCode,
    SkillIndexLimits,
    SkillMatchLimits,
    SkillReadLimits,
    SkillRegistry,
    SkillResourceReader,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_match_index,
    build_skill_registry,
    match_skill_registry,
    resolve_skill_sources,
)
from avalan.skill._async import skill_bounded_await


class SkillPerformancePhase14Test(IsolatedAsyncioTestCase):
    async def test_large_registry_match_payload_and_memory_are_bounded(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for skill_number in range(48):
                _write_skill(
                    root / f"skill-{skill_number}" / "SKILL.md",
                    name=f"skill-{skill_number}",
                    description=(
                        "Phase fourteen shared metadata for deterministic "
                        f"matching {skill_number}."
                    ),
                    body=(
                        "# Body\n"
                        f"PRIVATE_BODY_TOKEN_{skill_number}\n"
                        "Use bounded resources only.\n"
                    ),
                )
            registry = await _registry(
                root,
                read_limits=SkillReadLimits(
                    max_bytes_per_read=2048,
                    max_lines_per_read=64,
                ),
                index_limits=SkillIndexLimits(
                    max_skills=64,
                    max_indexed_bytes=100_000,
                ),
            )

            start()
            excerpt_index_limits = SkillIndexLimits(
                max_skills=64,
                max_indexed_bytes=512,
            )
            match_index = await build_skill_match_index(
                registry,
                include_resource_excerpts=True,
                index_limits=excerpt_index_limits,
                match_limits=SkillMatchLimits(
                    max_results=5,
                    max_excerpt_bytes_per_skill=32,
                    max_index_tokens_per_skill=32,
                ),
            )
            matched = await match_skill_registry(
                registry,
                query="phase fourteen shared metadata",
                index=match_index,
                max_results=5,
            )
            _, peak_bytes = get_traced_memory()
            stop()

        self.assertEqual(len(match_index.entries), 48)
        self.assertLessEqual(match_index.indexed_bytes, 512)
        self.assertEqual(matched.status, SkillStatus.OK)
        self.assertLessEqual(len(matched.items), 5)
        self.assertLess(peak_bytes, 20_000_000)
        encoded = dumps(
            (
                registry.as_model_dict(),
                match_index.as_model_dict(),
                matched.as_model_dict(),
            ),
            sort_keys=True,
        )
        self.assertLess(len(encoded.encode("utf-8")), 400_000)
        self.assertNotIn("PRIVATE_BODY_TOKEN", encoded)

    async def test_large_resource_read_window_and_cursors_are_bounded(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            body = "".join(
                f"line {index:03d} payload\n" for index in range(80)
            )
            _write_skill(
                root / "large" / "SKILL.md",
                name="large",
                description="Large resource guidance.",
                body=body,
            )
            registry = await _registry(
                root,
                read_limits=SkillReadLimits(
                    max_bytes_per_read=4096,
                    max_lines_per_read=128,
                ),
            )
            reader = SkillResourceReader()

            first = await reader.read(
                registry,
                "large",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=256,
                    max_lines_per_read=5,
                ),
                cursor_limits=SkillCursorLimits(max_active_cursors=1),
            )
            second = await reader.read(
                registry,
                "large",
                read_limits=SkillReadLimits(
                    max_bytes_per_read=192,
                    max_lines_per_read=4,
                ),
                cursor_limits=SkillCursorLimits(max_active_cursors=1),
            )

        self.assertEqual(first.status, SkillStatus.TRUNCATED)
        self.assertIsNotNone(first.content)
        assert first.content is not None
        self.assertLessEqual(len(first.content.text.encode("utf-8")), 256)
        self.assertLessEqual(len(first.content.text.splitlines()), 5)
        self.assertIsNotNone(first.next_cursor)
        self.assertEqual(second.status, SkillStatus.TRUNCATED)
        self.assertLessEqual(reader.active_cursor_count, 1)
        self.assertLess(
            len(dumps(first.as_model_dict(), sort_keys=True).encode("utf-8")),
            8_192,
        )

    async def test_bounded_async_filesystem_times_out_slow_calls(self) -> None:
        file_system = SkillAsyncFileSystem(max_operation_seconds=0.01)

        with self.assertRaises(TimeoutError):
            await file_system._run(lambda: blocking_sleep(0.05))
        await sleep(0.06)

    async def test_bounded_await_accepts_unbounded_timeout(self) -> None:
        self.assertEqual(
            await skill_bounded_await(
                sleep(0, result="done"),
                timeout_seconds=None,
            ),
            "done",
        )

    async def test_async_filesystem_accepts_unbounded_operations(
        self,
    ) -> None:
        file_system = SkillAsyncFileSystem(max_operation_seconds=None)

        self.assertEqual(await file_system._run(lambda: "done"), "done")

    async def test_timed_out_filesystem_worker_keeps_concurrency_slot(
        self,
    ) -> None:
        file_system = SkillAsyncFileSystem(
            max_concurrency=1,
            max_operation_seconds=0.1,
        )
        release_worker = ThreadEvent()
        first_started = ThreadEvent()
        lock = ThreadLock()
        active_workers = 0
        started_workers = 0
        max_active_workers = 0

        def blocking_call() -> str:
            nonlocal active_workers
            nonlocal max_active_workers
            nonlocal started_workers
            with lock:
                active_workers += 1
                started_workers += 1
                max_active_workers = max(max_active_workers, active_workers)
                if started_workers == 1:
                    first_started.set()
            release_worker.wait(timeout=1.0)
            with lock:
                active_workers -= 1
            return "done"

        try:
            with self.assertRaises(TimeoutError):
                await file_system._run(blocking_call)
            self.assertTrue(first_started.wait(timeout=0.2))

            second = create_task(file_system._run(blocking_call))
            await sleep(0.02)

            with lock:
                self.assertEqual(started_workers, 1)
                self.assertEqual(max_active_workers, 1)
            release_worker.set()
            self.assertEqual(await wait_for(second, timeout=1.0), "done")
            with lock:
                self.assertEqual(max_active_workers, 1)
        finally:
            release_worker.set()

    async def test_cancelled_filesystem_worker_keeps_slot_until_exit(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            resolved_path = root / "resolved.txt"
            resolved_path.write_text("ready", encoding="utf-8")
            file_system = SkillAsyncFileSystem(
                max_concurrency=1,
                max_operation_seconds=0.2,
            )
            release_worker = ThreadEvent()
            first_started = ThreadEvent()
            blocking_path = BlockingResolvePath(
                str(root / "blocked.txt"),
                started=first_started,
                release=release_worker,
                target=resolved_path,
            )
            first = create_task(file_system.resolve_path(blocking_path))

            try:
                await wait_for(
                    _wait_for_thread_event(first_started),
                    timeout=0.5,
                )
                first.cancel()

                with self.assertRaises(CancelledError):
                    await first

                self.assertFalse(release_worker.is_set())
                start_time = perf_counter()
                with self.assertRaises(TimeoutError):
                    await wait_for(
                        file_system.resolve_path(resolved_path),
                        timeout=0.5,
                    )
                self.assertLess(perf_counter() - start_time, 0.4)

                release_worker.set()
                self.assertEqual(
                    await wait_for(
                        file_system.resolve_path(resolved_path),
                        timeout=1.0,
                    ),
                    resolved_path.resolve(strict=True),
                )
            finally:
                release_worker.set()
                if not first.done():
                    first.cancel()
                    with suppress(CancelledError):
                        await first

    async def test_filesystem_semaphore_acquire_timeout_is_bounded(
        self,
    ) -> None:
        file_system = SkillAsyncFileSystem(
            max_concurrency=1,
            max_operation_seconds=0.02,
        )
        release_worker = ThreadEvent()
        first_started = ThreadEvent()

        def blocking_call() -> str:
            first_started.set()
            release_worker.wait(timeout=1.0)
            return "done"

        try:
            with self.assertRaises(TimeoutError):
                await file_system._run(blocking_call)
            self.assertTrue(first_started.wait(timeout=0.2))

            start_time = perf_counter()
            with self.assertRaises(TimeoutError):
                await wait_for(
                    file_system._run(lambda: "queued"),
                    timeout=0.2,
                )
            self.assertLess(perf_counter() - start_time, 0.15)
        finally:
            release_worker.set()
            await sleep(0.05)

    async def test_filesystem_worker_exception_releases_timeout_slot(
        self,
    ) -> None:
        file_system = SkillAsyncFileSystem(
            max_concurrency=1,
            max_operation_seconds=0.02,
        )
        release_worker = ThreadEvent()
        first_started = ThreadEvent()

        def blocking_call() -> str:
            first_started.set()
            release_worker.wait(timeout=1.0)
            raise RuntimeError("worker failed after timeout")

        try:
            with self.assertRaises(TimeoutError):
                await file_system._run(blocking_call)
            self.assertTrue(first_started.wait(timeout=0.2))
            release_worker.set()
            await sleep(0.05)

            self.assertEqual(await file_system._run(lambda: "next"), "next")
        finally:
            release_worker.set()

    async def test_policy_denial_and_stale_resources_are_not_hidden(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            skill_path = root / "pdf" / "SKILL.md"
            _write_skill(
                skill_path,
                name="pdf",
                description="PDF rendering guidance.",
                body="# PDF\nRender.\n",
            )
            denied_registry = await _registry(
                root,
                settings=TrustedSkillSettings(allowed_skill_ids=("docx",)),
            )
            denied = await SkillResourceReader().read(
                denied_registry,
                "pdf",
                file_system=ExplodingFileSystem(),
            )
            registry = await _registry(root)
            skill_path.write_text(
                "---\n"
                "name: pdf\n"
                "description: PDF rendering guidance.\n"
                "---\n"
                "# PDF\nChanged.\n",
                encoding="utf-8",
            )
            stale = await SkillResourceReader().read(registry, "pdf")

        self.assertEqual(denied.status, SkillStatus.POLICY_DENIED)
        self.assertEqual(
            denied.diagnostics[0].code,
            SkillDiagnosticCode.POLICY_DENIED,
        )
        self.assertEqual(stale.status, SkillStatus.STALE)


class ExplodingFileSystem:
    async def resolve_path(self, path: Path) -> Path:
        raise AssertionError("policy denial should not touch the filesystem")

    async def stat_path(self, path: Path) -> stat_result:
        raise AssertionError("policy denial should not touch the filesystem")

    async def lstat_path(self, path: Path) -> stat_result:
        raise AssertionError("policy denial should not touch the filesystem")

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        raise AssertionError("policy denial should not touch the filesystem")

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise AssertionError("policy denial should not touch the filesystem")


class BlockingResolvePath(type(Path())):
    _release: ThreadEvent
    _started: ThreadEvent
    _target: Path

    def __new__(
        cls,
        path: str,
        *,
        started: ThreadEvent,
        release: ThreadEvent,
        target: Path,
    ) -> "BlockingResolvePath":
        instance = super().__new__(cls, path)
        instance._started = started
        instance._release = release
        instance._target = target
        return instance

    def __init__(
        self,
        path: str,
        *,
        started: ThreadEvent,
        release: ThreadEvent,
        target: Path,
    ) -> None:
        # Newer pathlib versions pass subclass constructor kwargs to __init__.
        pass

    def resolve(self, strict: bool = False) -> Path:
        self._started.set()
        self._release.wait(timeout=1.0)
        return self._target.resolve(strict=strict)


async def _registry(
    root: Path,
    *,
    settings: TrustedSkillSettings | None = None,
    read_limits: SkillReadLimits | None = None,
    index_limits: SkillIndexLimits | None = None,
) -> SkillRegistry:
    source_result = await resolve_skill_sources(
        (_config(root),),
        settings=settings,
        read_limits=read_limits,
        index_limits=index_limits,
    )
    return await build_skill_registry(
        source_result,
        settings=settings,
        read_limits=read_limits,
        index_limits=index_limits,
    )


async def _wait_for_thread_event(event: ThreadEvent) -> None:
    while not event.is_set():
        await sleep(0.001)


def _config(root: Path) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label="Workspace Main",
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _write_skill(
    path: Path,
    *,
    name: str,
    description: str,
    body: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
