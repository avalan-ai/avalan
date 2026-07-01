from asyncio import sleep, wait_for
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.skill import (
    SkillAsyncFileSystem,
    SkillSourceRootConfig,
    WorkspaceSkillSourceAuthority,
    resolve_skill_sources,
)


class SlowResolveFileSystem:
    async def resolve_path(self, path: Path) -> Path:
        await sleep(60)
        return path

    async def stat_path(self, path: Path) -> stat_result:
        raise AssertionError("stat_path should not be reached")

    async def lstat_path(self, path: Path) -> stat_result:
        raise AssertionError("lstat_path should not be reached")

    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        raise AssertionError("list_directory should not be reached")

    async def read_bytes(self, path: Path, limit: int) -> bytes:
        raise AssertionError("read_bytes should not be reached")


class SlowWalkFileSystem(SkillAsyncFileSystem):
    async def list_directory(
        self,
        path: Path,
        limit: int,
    ) -> tuple[Path, ...]:
        await sleep(60)
        return ()


class SkillResolverAsyncWatchdogPhase2Test(IsolatedAsyncioTestCase):
    async def test_stalled_source_resolution_cancels_promptly(self) -> None:
        with TemporaryDirectory() as temporary:
            config = SkillSourceRootConfig(
                label="slow",
                authority=WorkspaceSkillSourceAuthority(),
                root=Path(temporary),
            )

            with self.assertRaises(TimeoutError):
                await wait_for(
                    resolve_skill_sources(
                        (config,),
                        file_system=SlowResolveFileSystem(),
                    ),
                    timeout=0.05,
                )

    async def test_stalled_source_walk_cancels_promptly(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "SKILL.md").write_text("ok", encoding="utf-8")
            config = SkillSourceRootConfig(
                label="slow-walk",
                authority=WorkspaceSkillSourceAuthority(),
                root=root,
            )

            with self.assertRaises(TimeoutError):
                await wait_for(
                    resolve_skill_sources(
                        (config,),
                        file_system=SlowWalkFileSystem(),
                    ),
                    timeout=0.05,
                )


if __name__ == "__main__":
    main()
