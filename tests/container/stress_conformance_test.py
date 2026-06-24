from asyncio import Queue, Task, create_task, sleep
from asyncio import run as run_async
from collections.abc import AsyncIterable, Sequence
from unittest import TestCase, main

from avalan.container import (
    ContainerAsyncBackend,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendContainer,
    ContainerBackendDiagnosticCode,
    ContainerBackendImageResolution,
    ContainerBackendInspection,
    ContainerBackendOperation,
    ContainerBackendOperationResult,
    ContainerBackendProbeResult,
    ContainerBackendStats,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBackendWaitResult,
    ContainerBuildPolicy,
    ContainerCommandPlan,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerImagePolicy,
    ContainerLifecycleDeadlines,
    ContainerLifecycleEventPolicy,
    ContainerLifecyclePhase,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerOutputContract,
    ContainerOutputValidationResult,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerStreamDrainPolicy,
    drain_container_streams,
    run_container_managed_lifecycle,
)

_DIGEST = "a" * 64
_IMAGE = f"ghcr.io/example/stress-tools@sha256:{_DIGEST}"


class ContainerStressConformanceTest(TestCase):
    def test_stream_drain_applies_byte_caps_under_many_chunks(self) -> None:
        chunks = _stress_chunks(90)
        result = drain_container_streams(
            chunks,
            ContainerStreamDrainPolicy(
                max_chunks=4,
                max_bytes=24,
                max_chunk_bytes=7,
                max_stdout_bytes=11,
                max_stderr_bytes=9,
                max_non_output_chunks=2,
                max_non_output_bytes=5,
                preserve_truncated_prefix=True,
            ),
        )
        codes = {diagnostic.code for diagnostic in result.diagnostics}

        self.assertEqual(len(result.chunks), 5)
        self.assertEqual(result.stdout_bytes, 11)
        self.assertEqual(result.stderr_bytes, 9)
        self.assertEqual(
            [chunk.content for chunk in result.chunks],
            [b"stdout-", b"stderr-", b"progr", b"stdo", b"st"],
        )
        self.assertEqual(result.truncated_chunks, 5)
        self.assertEqual(result.dropped_chunks, len(chunks) - 5)
        self.assertIn(ContainerBackendDiagnosticCode.STREAM_TRUNCATED, codes)
        self.assertIn(ContainerBackendDiagnosticCode.EVENT_DROPPED, codes)
        self.assertEqual(result.to_dict()["dropped_chunks"], len(chunks) - 5)

    def test_slow_async_stream_is_drained_after_kept_event_cap(self) -> None:
        chunks = _stress_chunks(72)
        stream = _QueueBackedAsyncStream(chunks)
        backend = _QueueStreamBackend(stream)

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                _run_plan(),
                stream_policy=ContainerStreamDrainPolicy(
                    max_chunks=2,
                    max_bytes=18,
                    max_chunk_bytes=6,
                    max_stdout_bytes=6,
                    max_stderr_bytes=6,
                    max_non_output_chunks=1,
                    max_non_output_bytes=4,
                    preserve_truncated_prefix=True,
                ),
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(stream.produced, len(chunks))
        self.assertEqual(stream.consumed, len(chunks))
        self.assertTrue(stream.producer_done)
        self.assertLessEqual(stream.max_buffered, 1)
        self.assertEqual(len(result.stream.chunks), 3)
        self.assertEqual(result.stream.dropped_chunks, len(chunks) - 3)
        self.assertEqual(result.stream.stdout_bytes, 6)
        self.assertEqual(result.stream.stderr_bytes, 6)
        self.assertEqual(
            [chunk.content for chunk in result.stream.chunks],
            [b"stdout", b"stderr", b"prog"],
        )
        self.assertIn(ContainerBackendOperation.STREAM, backend.operations)
        self.assertTrue(result.cleanup_completed)

    def test_lifecycle_event_cap_drops_excess_phase_events(self) -> None:
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                _run_plan(),
                event_policy=ContainerLifecycleEventPolicy(max_events=3),
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(len(result.events), 3)
        self.assertEqual(
            [event.sequence for event in result.events],
            [0, 1, 2],
        )
        self.assertEqual(
            [event.phase for event in result.events],
            [
                ContainerLifecyclePhase.POLICY_NORMALIZATION,
                ContainerLifecyclePhase.POLICY_NORMALIZATION,
                ContainerLifecyclePhase.BACKEND_SELECTION,
            ],
        )
        self.assertGreater(result.dropped_events, 0)
        self.assertEqual(
            result.execution.metadata["dropped_events"],
            str(result.dropped_events),
        )

    def test_cleanup_deadline_is_reported_after_output_pressure(self) -> None:
        chunks = _stress_chunks(64)
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                operation_delay_seconds={
                    ContainerBackendOperation.REMOVE: 0.02,
                },
                stream_chunks=chunks,
            )
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                _run_plan(),
                deadlines=ContainerLifecycleDeadlines(cleanup_seconds=0.001),
                stream_policy=ContainerStreamDrainPolicy(
                    max_chunks=2,
                    max_bytes=12,
                    max_chunk_bytes=6,
                    max_stdout_bytes=6,
                    max_stderr_bytes=6,
                    preserve_truncated_prefix=True,
                ),
            )
        )
        codes = {diagnostic.code for diagnostic in result.diagnostics}

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertTrue(result.cleanup_uncertain)
        self.assertTrue(result.cleanup_completed)
        self.assertEqual(
            result.execution.metadata["cleanup_uncertain"],
            "true",
        )
        self.assertEqual(len(result.stream.chunks), 2)
        self.assertEqual(result.stream.dropped_chunks, len(chunks) - 2)
        self.assertIn(ContainerBackendDiagnosticCode.STREAM_TRUNCATED, codes)
        self.assertIn(ContainerBackendDiagnosticCode.EVENT_DROPPED, codes)
        self.assertIn(ContainerBackendDiagnosticCode.TIMEOUT, codes)
        self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
        self.assertIn(ContainerBackendOperation.CLEANUP, backend.operations)


class _QueueBackedAsyncStream:
    def __init__(
        self,
        chunks: Sequence[ContainerBackendStreamChunk],
    ) -> None:
        self._chunks = tuple(chunks)
        self._queue: Queue[ContainerBackendStreamChunk | None] = Queue(
            maxsize=1,
        )
        self._producer_task: Task[None] | None = None
        self.produced = 0
        self.consumed = 0
        self.max_buffered = 0

    @property
    def producer_done(self) -> bool:
        return self._producer_task is not None and self._producer_task.done()

    def __aiter__(self) -> "_QueueBackedAsyncStream":
        assert self._producer_task is None, "stream can only be consumed once"
        self._producer_task = create_task(self._produce())
        return self

    async def __anext__(self) -> ContainerBackendStreamChunk:
        item = await self._queue.get()
        if item is None:
            assert self._producer_task is not None
            await self._producer_task
            raise StopAsyncIteration
        self.consumed += 1
        await sleep(0)
        return item

    async def _produce(self) -> None:
        for chunk in self._chunks:
            await self._queue.put(chunk)
            self.produced += 1
            self.max_buffered = max(self.max_buffered, self._queue.qsize())
            await sleep(0)
        await self._queue.put(None)


class _QueueStreamBackend(ContainerAsyncBackend):
    def __init__(self, stream: _QueueBackedAsyncStream) -> None:
        self._backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                stream_chunks=(),
            )
        )
        self._stream = stream

    @property
    def operations(self) -> tuple[ContainerBackendOperation, ...]:
        return self._backend.operations

    async def probe(self) -> ContainerBackendProbeResult:
        return await self._backend.probe()

    async def resolve_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendImageResolution:
        return await self._backend.resolve_image(plan)

    async def pull_image(
        self,
        plan: ContainerRunPlan,
        image: ContainerBackendImageResolution,
    ) -> ContainerBackendOperationResult:
        return await self._backend.pull_image(plan, image)

    async def build_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendOperationResult:
        return await self._backend.build_image(plan)

    async def create(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendContainer:
        return await self._backend.create(plan)

    async def start(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.start(container)

    async def attach(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.attach(container)

    async def stream(
        self,
        container: ContainerBackendContainer,
    ) -> (
        Sequence[ContainerBackendStreamChunk]
        | AsyncIterable[ContainerBackendStreamChunk]
    ):
        await self._backend.stream(container)
        return self._stream

    async def wait(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendWaitResult:
        return await self._backend.wait(container)

    async def inspect(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendInspection:
        return await self._backend.inspect(container)

    async def stats(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStats, ...]:
        return await self._backend.stats(container)

    async def stop(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.stop(container)

    async def kill(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.kill(container)

    async def remove(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.remove(container)

    async def copy_outputs(
        self,
        container: ContainerBackendContainer,
        contract: ContainerOutputContract,
    ) -> ContainerOutputValidationResult:
        return await self._backend.copy_outputs(container, contract)

    async def cleanup(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.cleanup(container)


def _stress_chunks(count: int) -> tuple[ContainerBackendStreamChunk, ...]:
    streams = (
        ContainerBackendStream.STDOUT,
        ContainerBackendStream.STDERR,
        ContainerBackendStream.PROGRESS,
    )
    prefixes = {
        ContainerBackendStream.STDOUT: b"stdout-",
        ContainerBackendStream.STDERR: b"stderr-",
        ContainerBackendStream.PROGRESS: b"progress-",
    }
    return tuple(
        ContainerBackendStreamChunk(
            stream=stream,
            content=prefixes[stream] + f"{sequence:02d}".encode("ascii"),
            sequence=sequence,
        )
        for sequence in range(count)
        for stream in (streams[sequence % len(streams)],)
    )


def _capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        rootless=True,
        build=True,
        pull=True,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE,),
        device_classes=(),
        resource_limits=True,
        streaming_attach=True,
        stats=True,
    )


def _run_plan() -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=ContainerBackend.DOCKER,
        profile_name="stress-profile",
        image=ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=ContainerPullPolicy.NEVER,
            build_policy=ContainerBuildPolicy.DISABLED,
        ),
        command=ContainerCommandPlan(
            tool_name="shell.stress",
            command="stress",
            argv=("stress", "--bounded-output"),
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        policy_version="phase20",
    )


if __name__ == "__main__":
    main()
