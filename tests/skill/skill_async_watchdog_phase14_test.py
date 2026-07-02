from asyncio import create_task, sleep, wait_for
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.skill import (
    SkillAsyncFileSystem,
    SkillConfiguredSource,
    SkillObservabilitySettings,
    SkillResourceReader,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.skill.observability import (
    SkillAuditDeliveryError,
    emit_skill_audit_event,
)


class SlowReadFileSystem:
    def __init__(self) -> None:
        self._delegate = SkillAsyncFileSystem()

    async def resolve_path(self, path: Path) -> Path:
        return await self._delegate.resolve_path(path)

    async def stat_path(self, path: Path) -> stat_result:
        return await self._delegate.stat_path(path)

    async def lstat_path(self, path: Path) -> stat_result:
        return await self._delegate.lstat_path(path)

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        return await self._delegate.list_directory(path, limit)

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        await sleep(60)
        return await self._delegate.read_bytes(path, limit)


class SlowAuditPublisher:
    async def trigger(self, event: Event) -> None:
        await sleep(60)


class SkillAsyncWatchdogPhase14Test(IsolatedAsyncioTestCase):
    async def test_registry_build_cancels_against_slow_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")
            source_result = await resolve_skill_sources((_config(root),))
            task = create_task(
                build_skill_registry(
                    source_result,
                    file_system=SlowReadFileSystem(),
                )
            )

            with self.assertRaises(TimeoutError):
                await wait_for(task, timeout=0.05)

        self.assertTrue(task.done())

    async def test_stale_detection_cancels_against_slow_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")
            source_result = await resolve_skill_sources((_config(root),))
            registry = await build_skill_registry(source_result)
            handle = registry.resource_handles[0]
            task = create_task(
                registry.check_resource(
                    handle,
                    file_system=SlowReadFileSystem(),
                )
            )

            with self.assertRaises(TimeoutError):
                await wait_for(task, timeout=0.05)

        self.assertTrue(task.done())

    async def test_bounded_read_cancels_against_slow_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(root / "pdf" / "SKILL.md")
            source_result = await resolve_skill_sources((_config(root),))
            registry = await build_skill_registry(source_result)
            task = create_task(
                SkillResourceReader().read(
                    registry,
                    "pdf",
                    file_system=SlowReadFileSystem(),
                )
            )

            with self.assertRaises(TimeoutError):
                await wait_for(task, timeout=0.05)

        self.assertTrue(task.done())

    async def test_event_delivery_cancels_promptly(self) -> None:
        task = create_task(
            emit_skill_audit_event(
                SlowAuditPublisher(),
                None,
                EventType.SKILL_REGISTRY_BUILD_STARTED,
                {"status": "started"},
            )
        )

        with self.assertRaises(TimeoutError):
            await wait_for(task, timeout=0.05)

        self.assertTrue(task.done())

    async def test_event_delivery_timeout_is_fail_open_by_default(
        self,
    ) -> None:
        await wait_for(
            emit_skill_audit_event(
                SlowAuditPublisher(),
                None,
                EventType.SKILL_REGISTRY_BUILD_STARTED,
                {"status": "started"},
                delivery_timeout_seconds=0.01,
            ),
            timeout=0.2,
        )

    async def test_event_delivery_timeout_honors_fail_closed(self) -> None:
        settings = TrustedSkillSettings(
            observability=SkillObservabilitySettings(audit_fail_closed=True)
        )

        with self.assertRaises(SkillAuditDeliveryError):
            await wait_for(
                emit_skill_audit_event(
                    SlowAuditPublisher(),
                    settings,
                    EventType.SKILL_REGISTRY_BUILD_STARTED,
                    {"status": "started"},
                    delivery_timeout_seconds=0.01,
                ),
                timeout=0.2,
            )


def _config(root: Path) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label="Workspace Main",
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _write_skill(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: pdf\n"
        "description: PDF rendering guidance.\n"
        "---\n"
        "# PDF\nRender pages.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
