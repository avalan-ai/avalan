from asyncio import (
    CancelledError,
    Event,
    Task,
    create_task,
    current_task,
    sleep,
    wait_for,
)
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.server.mcp_tasks import (
    MCP_RELATED_TASK_METADATA_KEY,
    MCP_TASK_PROTOCOL_VERSION,
    MCPTaskCapabilities,
    MCPTaskController,
    MCPTaskOutcome,
    MCPTaskPolicy,
    MCPTaskProtocolError,
    MCPTaskRequest,
    parse_task_request,
    require_related_task_metadata,
    with_related_task_metadata,
    without_related_task_metadata,
)


@dataclass(slots=True)
class _Clock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, milliseconds: int) -> None:
        self.value += timedelta(milliseconds=milliseconds)


@dataclass(frozen=True, slots=True)
class _Principal:
    tenant: str
    subject: str


class MCPTaskContractTest(TestCase):
    def test_capabilities_and_task_request_negotiation(self) -> None:
        self.assertEqual(MCP_TASK_PROTOCOL_VERSION, "2025-11-25")
        self.assertEqual(MCPTaskCapabilities.parse({}).as_dict(), {})
        wire: dict[str, object] = {
            "tasks": {
                "list": {},
                "cancel": {},
                "requests": {
                    "tools": {"call": {}},
                    "sampling": {"createMessage": {}},
                    "elicitation": {"create": {}},
                },
            }
        }
        capabilities = MCPTaskCapabilities.parse(wire)
        self.assertEqual(capabilities.as_dict(), wire["tasks"])
        self.assertTrue(capabilities.supports("tools/call"))
        self.assertFalse(capabilities.supports("prompts/get"))
        self.assertFalse(
            MCPTaskCapabilities.advertise(
                MCPTaskPolicy(),
                requestor=None,
            ).list_tasks
        )
        self.assertTrue(
            MCPTaskCapabilities.advertise(
                MCPTaskPolicy(),
                requestor=_Principal("t", "s"),
            ).list_tasks
        )

        request = parse_task_request(
            {"task": {"ttl": 25, "extension": True}},
            request_type="tools/call",
            capabilities=capabilities,
            execution_task_support="optional",
        )
        self.assertEqual(request, MCPTaskRequest(requested_ttl_ms=25))
        self.assertEqual(
            parse_task_request(
                {"task": {}},
                request_type="elicitation/create",
                capabilities=capabilities,
            ),
            MCPTaskRequest(request_type="elicitation/create"),
        )
        self.assertIsNone(
            parse_task_request(
                {"task": "ignored"},
                request_type="tools/call",
                capabilities=MCPTaskCapabilities(),
                execution_task_support="optional",
            )
        )
        self.assertIsNone(
            parse_task_request(
                {},
                request_type="tools/call",
                capabilities=capabilities,
                execution_task_support="optional",
            )
        )

        invalid_capabilities: tuple[dict[str, object], ...] = (
            {"tasks": []},
            {"tasks": {"list": True}},
            {"tasks": {"cancel": False}},
            {"tasks": {"requests": []}},
            {"tasks": {"requests": {"tools": []}}},
            {"tasks": {"requests": {"tools": {"call": True}}}},
        )
        for value in invalid_capabilities:
            with self.subTest(capabilities=value):
                with self.assertRaises(MCPTaskProtocolError) as raised:
                    MCPTaskCapabilities.parse(value)
                self.assertEqual(raised.exception.code, -32602)

        invalid_requests: tuple[
            tuple[dict[str, object], str, object, str],
            ...,
        ] = (
            ({}, "tools/call", "required", "requires"),
            ({"task": {}}, "tools/call", "forbidden", "does not support"),
            ({"task": []}, "tools/call", "optional", "parameters"),
            (
                {"task": {"ttl": 0}},
                "tools/call",
                "optional",
                "positive integer",
            ),
            (
                {"task": {"ttl": True}},
                "tools/call",
                "optional",
                "positive integer",
            ),
            (
                {"task": {}},
                "tools/call",
                "sometimes",
                "task support",
            ),
            (
                {"task": {}},
                "unknown",
                None,
                "request type",
            ),
        )
        for params, method, mode, message in invalid_requests:
            with self.subTest(params=params, mode=mode):
                with self.assertRaises(MCPTaskProtocolError) as raised:
                    parse_task_request(
                        params,
                        request_type=method,
                        capabilities=capabilities,
                        execution_task_support=cast(Any, mode),
                    )
                self.assertIn(message, raised.exception.message)
        with self.assertRaises(AssertionError):
            parse_task_request(
                {},
                request_type="tools/call",
                capabilities=cast(Any, object()),
            )

    def test_metadata_and_exact_outcome_contract(self) -> None:
        source: dict[str, object] = {
            "content": [{"type": "text", "text": "done"}],
            "_meta": {"trace": "safe"},
        }
        attached = with_related_task_metadata(source, "task-a")
        require_related_task_metadata(attached, "task-a")
        self.assertNotIn(
            MCP_RELATED_TASK_METADATA_KEY,
            cast(dict[str, object], source["_meta"]),
        )
        self.assertEqual(
            without_related_task_metadata(attached),
            source,
        )
        self.assertEqual(
            without_related_task_metadata(
                {
                    "_meta": {
                        MCP_RELATED_TASK_METADATA_KEY: {"taskId": "task-a"}
                    }
                }
            ),
            {},
        )
        self.assertEqual(
            without_related_task_metadata({"_meta": "opaque"}),
            {"_meta": "opaque"},
        )
        self.assertEqual(
            with_related_task_metadata(attached, "task-a"),
            attached,
        )

        invalid: tuple[dict[str, object], ...] = (
            {},
            {"_meta": []},
            {"_meta": {}},
            {"_meta": {MCP_RELATED_TASK_METADATA_KEY: []}},
            {"_meta": {MCP_RELATED_TASK_METADATA_KEY: {"taskId": ""}}},
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(MCPTaskProtocolError):
                    require_related_task_metadata(payload, "task-a")
        with self.assertRaises(MCPTaskProtocolError) as mismatch:
            require_related_task_metadata(
                with_related_task_metadata({}, "task-b"),
                "task-a",
            )
        self.assertEqual(
            mismatch.exception.data,
            {"policy": "avalan", "reason": "state_mismatch"},
        )
        attach_invalid: tuple[
            tuple[dict[str, object], str],
            ...,
        ] = (({"_meta": []}, "task-a"), ({}, ""))
        for payload, task_id in attach_invalid:
            with self.subTest(payload=payload, task_id=task_id):
                with self.assertRaises(MCPTaskProtocolError):
                    with_related_task_metadata(payload, task_id)

        result_source: dict[str, object] = {"content": [{"text": "exact"}]}
        success = MCPTaskOutcome.success(result_source)
        result_source["content"] = []
        projected = success.for_task("task-a")
        assert projected.result is not None
        self.assertEqual(projected.result["content"], [{"text": "exact"}])
        require_related_task_metadata(projected.result, "task-a")
        error_source: dict[str, object] = {
            "code": -32000,
            "message": "failed",
        }
        failure = MCPTaskOutcome.failure(error_source)
        error_source["message"] = "mutated"
        self.assertEqual(
            failure.for_task("task-a").error,
            {"code": -32000, "message": "failed"},
        )
        self.assertEqual(
            MCPTaskOutcome.failure({"code": -32000, "message": ""}).error,
            {"code": -32000, "message": ""},
        )
        for result, error in (
            (None, None),
            ({}, {"code": -1, "message": "bad"}),
            (None, {"code": True, "message": "bad"}),
            (None, {"code": -1}),
        ):
            with self.subTest(result=result, error=error):
                with self.assertRaises(ValueError):
                    MCPTaskOutcome(
                        result=cast(Any, result),
                        error=cast(Any, error),
                    )
        protocol = MCPTaskProtocolError(
            code=-32602,
            message="Denied.",
            data={"policy": "avalan"},
        )
        self.assertEqual(
            protocol.as_error(),
            {
                "code": -32602,
                "message": "Denied.",
                "data": {"policy": "avalan"},
            },
        )
        self.assertEqual(
            MCPTaskProtocolError(
                code=-32602,
                message="Invalid.",
            ).as_error(),
            {"code": -32602, "message": "Invalid."},
        )


class MCPTaskControllerTest(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.clock = _Clock(datetime(2026, 7, 24, tzinfo=UTC))
        self.owner = _Principal("tenant", "owner")
        self.ids = iter(("task-a", "task-b", "task-c", "task-d"))

    def controller(
        self,
        policy: MCPTaskPolicy | None = None,
    ) -> MCPTaskController:
        return MCPTaskController(
            policy,
            clock=self.clock,
            id_factory=lambda: next(self.ids),
        )

    async def test_immediate_creation_and_full_blocking_lifecycle(
        self,
    ) -> None:
        controller = self.controller(
            MCPTaskPolicy(default_ttl_ms=100, maximum_ttl_ms=200)
        )
        creation = await controller.create(
            MCPTaskRequest(requested_ttl_ms=500),
            requestor=self.owner,
        )
        self.assertEqual(
            creation.as_dict(),
            {
                "task": {
                    "taskId": "task-a",
                    "status": "working",
                    "statusMessage": "The operation is in progress.",
                    "createdAt": "2026-07-24T00:00:00Z",
                    "lastUpdatedAt": "2026-07-24T00:00:00Z",
                    "ttl": 200,
                    "pollInterval": 1000,
                }
            },
        )
        self.assertEqual(
            controller.task_request(
                {"task": {"ttl": 20}},
                request_type="tools/call",
                execution_task_support="optional",
                requestor=self.owner,
            ),
            MCPTaskRequest(requested_ttl_ms=20),
        )
        self.assertIn(
            "list",
            cast(
                dict[str, object],
                controller.capability_dict(requestor=self.owner)["tasks"],
            ),
        )
        self.assertNotIn(
            "list",
            cast(
                dict[str, object],
                controller.capability_dict(requestor=None)["tasks"],
            ),
        )

        result = create_task(controller.result("task-a", requestor=self.owner))
        await sleep(0)
        self.assertFalse(result.done())
        self.clock.advance(1)
        self.assertEqual(
            (await creation.handle.transition_input_required())["status"],
            "input_required",
        )
        await sleep(0)
        self.assertFalse(result.done())
        self.clock.advance(1)
        self.assertEqual(
            (await creation.handle.transition_working())["status"],
            "working",
        )
        await sleep(0)
        self.assertFalse(result.done())
        completed = await creation.handle.complete(
            {
                "content": [{"type": "text", "text": "exact"}],
                "_meta": {"trace": "retained"},
            }
        )
        self.assertEqual(completed["status"], "completed")
        outcome = await result
        assert outcome.result is not None
        require_related_task_metadata(outcome.result, "task-a")
        self.assertEqual(
            cast(dict[str, object], outcome.result["_meta"])["trace"],
            "retained",
        )
        self.assertEqual(
            (await controller.get("task-a", requestor=self.owner))[
                "lastUpdatedAt"
            ],
            "2026-07-24T00:00:00.002000Z",
        )

    async def test_failures_and_state_policy_are_terminal(self) -> None:
        controller = self.controller(
            MCPTaskPolicy(
                request_types=frozenset(
                    {
                        "tools/call",
                        "sampling/createMessage",
                        "elicitation/create",
                    }
                )
            )
        )
        tool = await controller.create(MCPTaskRequest(), requestor=self.owner)
        self.assertEqual(
            (await tool.handle.complete({"content": [], "isError": True}))[
                "status"
            ],
            "failed",
        )
        self.assertIs(
            (await controller.result("task-a", requestor=self.owner)).result[
                "isError"
            ],  # type: ignore[index]
            True,
        )
        rpc = await controller.create(MCPTaskRequest(), requestor=self.owner)
        await rpc.handle.fail(
            {
                "code": -32004,
                "message": "Unavailable",
                "data": {"retryable": True},
            }
        )
        self.assertEqual(
            (await controller.result("task-b", requestor=self.owner)).error,
            {
                "code": -32004,
                "message": "Unavailable",
                "data": {"retryable": True},
            },
        )
        for operation in (
            tool.handle.transition_input_required(),
            tool.handle.complete({"content": []}),
            rpc.handle.transition_working(),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(MCPTaskProtocolError) as mismatch:
                    await operation
                self.assertEqual(
                    mismatch.exception.data,
                    {"policy": "avalan", "reason": "state_mismatch"},
                )
        conflict = await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
        )
        with self.assertRaises(MCPTaskProtocolError):
            await conflict.handle.complete(
                with_related_task_metadata({}, "other")
            )
        for method, task_id in (
            ("sampling/createMessage", "sampling"),
            ("elicitation/create", "elicitation"),
        ):
            reverse = await controller.create(
                MCPTaskRequest(request_type=method),
                requestor=self.owner,
                task_id=task_id,
            )
            self.assertEqual(
                (
                    await reverse.handle.complete(
                        {
                            "isError": True,
                            "content": [],
                        }
                    )
                )["status"],
                "completed",
            )
        with self.assertRaises(MCPTaskProtocolError):
            await controller.create(
                MCPTaskRequest(request_type="unknown"),
                requestor=self.owner,
                task_id="unknown",
            )
        with self.assertRaises(MCPTaskProtocolError) as unsupported:
            await MCPTaskController().create(
                MCPTaskRequest(request_type="sampling/createMessage"),
                requestor=self.owner,
            )
        self.assertEqual(unsupported.exception.code, -32601)

    async def test_cancellation_is_sticky_before_bounded_cleanup(
        self,
    ) -> None:
        controller = self.controller(MCPTaskPolicy(cancellation_timeout_ms=10))
        entered = Event()
        release = Event()
        observed: list[object] = []

        async def callback() -> None:
            observed.append(
                (await controller.get("task-a", requestor=self.owner))[
                    "status"
                ]
            )
            entered.set()
            await release.wait()

        creation = await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
            cancellation_callback=callback,
        )
        cancelling = create_task(
            controller.cancel("task-a", requestor=self.owner)
        )
        await entered.wait()
        self.assertTrue(creation.handle.cancellation_requested)
        await creation.handle.wait_cancelled()
        self.assertEqual(
            (await creation.handle.complete({"content": ["late"]}))["status"],
            "cancelled",
        )
        self.assertEqual(
            (await creation.handle.transition_input_required())["status"],
            "cancelled",
        )
        release.set()
        self.assertEqual((await cancelling)["status"], "cancelled")
        self.assertEqual(observed, ["cancelled"])
        self.assertEqual(
            (await controller.result("task-a", requestor=self.owner)).error,
            {"code": -32000, "message": "Request cancelled"},
        )
        with self.assertRaises(MCPTaskProtocolError):
            await controller.cancel("task-a", requestor=self.owner)

        async def failing_callback() -> None:
            raise RuntimeError("private")

        failed = await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
            cancellation_callback=failing_callback,
        )
        with self.assertLogs("avalan.server.mcp_tasks", level="ERROR"):
            self.assertEqual(
                (
                    await controller.cancel(
                        "task-b",
                        requestor=self.owner,
                    )
                )["status"],
                "cancelled",
            )
        self.assertTrue(failed.handle.cancellation_requested)

        async def slow_callback() -> None:
            await Event().wait()

        await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
            cancellation_callback=slow_callback,
        )
        with self.assertLogs("avalan.server.mcp_tasks", level="WARNING"):
            await controller.cancel("task-c", requestor=self.owner)

        stubborn = MCPTaskController(
            MCPTaskPolicy(cancellation_timeout_ms=10),
            id_factory=lambda: "stubborn",
        )
        suppressed = Event()
        finished = Event()
        callback_tasks: list[Task[None]] = []

        async def suppress_twice() -> None:
            task = current_task()
            assert task is not None
            callback_tasks.append(task)
            for _ in range(2):
                try:
                    await Event().wait()
                except CancelledError:
                    suppressed.set()
            try:
                await Event().wait()
            finally:
                finished.set()

        await stubborn.create(
            MCPTaskRequest(),
            requestor=self.owner,
            cancellation_callback=suppress_twice,
        )
        with self.assertLogs("avalan.server.mcp_tasks", level="WARNING"):
            cancelled = await wait_for(
                stubborn.cancel("stubborn", requestor=self.owner),
                timeout=0.2,
            )
        self.assertEqual(cancelled["status"], "cancelled")
        await suppressed.wait()
        await stubborn.close()
        self.assertTrue(finished.is_set())
        self.assertTrue(callback_tasks[0].done())

        detached = MCPTaskController(
            MCPTaskPolicy(cancellation_timeout_ms=5),
            id_factory=lambda: "detached",
        )
        detached_release = Event()
        ignored = Event()
        detached_tasks: list[Task[None]] = []

        async def ignore_cancellation() -> None:
            task = current_task()
            assert task is not None
            detached_tasks.append(task)
            while not detached_release.is_set():
                try:
                    await detached_release.wait()
                except CancelledError:
                    ignored.set()
            raise RuntimeError("private detached failure")

        await detached.create(
            MCPTaskRequest(),
            requestor=self.owner,
            cancellation_callback=ignore_cancellation,
        )
        with self.assertLogs("avalan.server.mcp_tasks", level="WARNING"):
            await detached.cancel("detached", requestor=self.owner)
        await ignored.wait()
        with self.assertLogs("avalan.server.mcp_tasks", level="WARNING"):
            await wait_for(detached.close(), timeout=0.2)
        await detached.close()
        self.assertFalse(detached._cancellation_tasks)
        self.assertFalse(detached_tasks[0].done())
        with self.assertLogs("avalan.server.mcp_tasks", level="WARNING"):
            detached_release.set()
            with self.assertRaises(RuntimeError):
                await wait_for(detached_tasks[0], timeout=0.2)

    async def test_requestor_scoping_pagination_and_method_options(
        self,
    ) -> None:
        controller = self.controller(MCPTaskPolicy(page_size=1))
        other = _Principal("tenant", "other")
        for owner in (self.owner, other, self.owner, None):
            await controller.create(MCPTaskRequest(), requestor=owner)
        first = await controller.list(requestor=self.owner)
        self.assertEqual(
            cast(list[dict[str, object]], first["tasks"])[0]["taskId"],
            "task-a",
        )
        self.assertEqual(first["nextCursor"], "task-a")
        second = await controller.list(
            requestor=self.owner,
            cursor=cast(str, first["nextCursor"]),
        )
        self.assertEqual(
            cast(list[dict[str, object]], second["tasks"])[0]["taskId"],
            "task-c",
        )
        self.assertNotIn("nextCursor", second)
        for cursor, reason in (
            ("", None),
            ("missing", None),
            ("task-b", "authorization"),
        ):
            with self.subTest(cursor=cursor):
                with self.assertRaises(MCPTaskProtocolError) as invalid:
                    await controller.list(
                        requestor=self.owner,
                        cursor=cursor,
                    )
                if reason is not None:
                    self.assertEqual(
                        invalid.exception.data,
                        {"policy": "avalan", "reason": reason},
                    )
        with self.assertRaises(MCPTaskProtocolError) as anonymous:
            await controller.list(requestor=None)
        self.assertEqual(
            anonymous.exception.data,
            {"policy": "avalan", "reason": "absent_requestor"},
        )

        unauthorized = (
            controller.get("task-a", requestor=other),
            controller.result("task-a", requestor=other),
            controller.cancel("task-a", requestor=other),
        )
        for operation in unauthorized:
            with self.assertRaises(MCPTaskProtocolError) as denied:
                await operation
            self.assertEqual(
                denied.exception.data,
                {"policy": "avalan", "reason": "authorization"},
            )

        no_list = self.controller(MCPTaskPolicy(allow_list=False))
        with self.assertRaises(MCPTaskProtocolError) as list_error:
            await no_list.list(requestor=self.owner)
        self.assertEqual(list_error.exception.code, -32601)
        no_cancel = self.controller(MCPTaskPolicy(allow_cancel=False))
        await no_cancel.create(
            MCPTaskRequest(),
            requestor=self.owner,
            task_id="disabled",
        )
        with self.assertRaises(MCPTaskProtocolError) as cancel_error:
            await no_cancel.cancel("disabled", requestor=self.owner)
        self.assertEqual(cancel_error.exception.code, -32601)

    async def test_expiry_retention_disconnect_and_close_release_waiters(
        self,
    ) -> None:
        controller = self.controller(
            MCPTaskPolicy(
                default_ttl_ms=10,
                expired_tombstone_limit=1,
            )
        )
        first = await controller.create(MCPTaskRequest(), requestor=self.owner)
        second = await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
        )
        self.clock.advance(11)
        self.assertEqual(await controller.cleanup(), 2)
        self.assertTrue(first.handle.cancellation_requested)
        self.assertTrue(second.handle.cancellation_requested)
        for task_id, message in (
            ("task-a", "Task not found."),
            ("task-b", "Task has expired."),
        ):
            with self.subTest(task_id=task_id):
                with self.assertRaises(MCPTaskProtocolError) as expired:
                    await controller.get(task_id, requestor=self.owner)
                self.assertEqual(expired.exception.message, message)
        with self.assertRaises(MCPTaskProtocolError) as protected:
            await controller.get(
                "task-b",
                requestor=_Principal("tenant", "other"),
            )
        self.assertEqual(
            protected.exception.data,
            {"policy": "avalan", "reason": "authorization"},
        )
        self.assertEqual(await controller.cleanup_requestor(self.owner), 0)
        with self.assertRaises(MCPTaskProtocolError) as removed:
            await controller.get("task-b", requestor=self.owner)
        self.assertEqual(removed.exception.message, "Task not found.")

        active = await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
        )
        waiting = create_task(
            controller.result("task-c", requestor=self.owner)
        )
        await sleep(0)
        self.assertEqual(await controller.cleanup_requestor(self.owner), 1)
        with self.assertRaises(MCPTaskProtocolError):
            await waiting
        self.assertTrue(active.handle.cancellation_requested)

        closing = await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
        )
        closed_waiter = create_task(
            controller.result("task-d", requestor=self.owner)
        )
        await sleep(0)
        await controller.close()
        await controller.close()
        self.assertTrue(closing.handle.cancellation_requested)
        with self.assertRaises(MCPTaskProtocolError) as closed:
            await closed_waiter
        self.assertEqual(closed.exception.code, -32603)
        with self.assertRaises(MCPTaskProtocolError):
            await controller.create(MCPTaskRequest(), requestor=self.owner)

    async def test_expiry_wait_duplicate_capacity_and_input_contracts(
        self,
    ) -> None:
        expiring = MCPTaskController(
            MCPTaskPolicy(default_ttl_ms=1),
            id_factory=lambda: "expiring",
        )
        await expiring.create(MCPTaskRequest(), requestor=None)
        with self.assertRaises(MCPTaskProtocolError) as expired:
            await expiring.result("expiring", requestor=None)
        self.assertEqual(expired.exception.message, "Task has expired.")

        controller = self.controller(
            MCPTaskPolicy(maximum_tasks_per_requestor=1)
        )
        await controller.create(
            MCPTaskRequest(),
            requestor=self.owner,
            task_id="fixed",
        )
        with self.assertRaises(MCPTaskProtocolError) as duplicate:
            await controller.create(
                MCPTaskRequest(),
                requestor=self.owner,
                task_id="fixed",
            )
        self.assertEqual(
            duplicate.exception.data,
            {"policy": "avalan", "reason": "duplicate_task"},
        )
        with self.assertRaises(MCPTaskProtocolError) as capacity:
            await controller.create(MCPTaskRequest(), requestor=self.owner)
        self.assertEqual(
            capacity.exception.data,
            {"policy": "avalan", "reason": "capacity"},
        )
        for task_id in ("", "missing"):
            with self.assertRaises(MCPTaskProtocolError) as missing:
                await controller.get(task_id, requestor=self.owner)
            self.assertEqual(missing.exception.code, -32602)
        with self.assertRaises(TypeError):
            await controller.create(
                MCPTaskRequest(),
                requestor=cast(Any, []),
            )
        with self.assertRaises(AssertionError):
            await controller.create(cast(Any, object()), requestor=None)
        for ttl in (0, -1, cast(Any, True)):
            with self.assertRaises(MCPTaskProtocolError):
                await controller.create(
                    MCPTaskRequest(requested_ttl_ms=ttl),
                    requestor=None,
                )
        with self.assertRaises(AssertionError):
            await controller.create(
                MCPTaskRequest(),
                requestor=None,
                cancellation_callback=cast(Any, "bad"),
            )
        bad_id = MCPTaskController(
            clock=self.clock,
            id_factory=lambda: "",
        )
        with self.assertRaises(RuntimeError):
            await bad_id.create(MCPTaskRequest(), requestor=None)
        naive = MCPTaskController(clock=lambda: datetime(2026, 7, 24))
        with self.assertRaises(RuntimeError):
            await naive.create(MCPTaskRequest(), requestor=None)
        wrong_clock = MCPTaskController(clock=cast(Any, lambda: "now"))
        with self.assertRaises(RuntimeError):
            await wrong_clock.create(MCPTaskRequest(), requestor=None)
        generated = await MCPTaskController().create(
            MCPTaskRequest(),
            requestor=None,
        )
        self.assertTrue(cast(str, generated.task["taskId"]))

    def test_policy_asserts_invalid_configuration(self) -> None:
        invalid: tuple[dict[str, object], ...] = (
            {"request_types": frozenset({"unknown"})},
            {"allow_cancel": "yes"},
            {"allow_list": 1},
            {"default_ttl_ms": 0},
            {"maximum_ttl_ms": True},
            {"poll_interval_ms": 0},
            {"page_size": 0},
            {"maximum_tasks_per_requestor": 0},
            {"expired_tombstone_limit": 0},
            {"cancellation_timeout_ms": 0},
            {"default_ttl_ms": 2, "maximum_ttl_ms": 1},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(AssertionError):
                    MCPTaskPolicy(**cast(Any, values))
