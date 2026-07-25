"""Exercise executable A2A cells in the structured-input failure matrix."""

from asyncio import Event, create_task, run, sleep, wait_for
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast

from httpx import ASGITransport, AsyncClient

sys_path.append(str(Path(__file__).parent))

import a2a_contract_test as contract  # noqa: E402

from avalan.interaction import (  # noqa: E402
    AnswerProvenance,
    AsyncInteractionBroker,
    Choice,
    ChoiceValue,
    InteractionBrokerRequest,
    InteractionClock,
    InteractionPolicy,
    InteractionTime,
    PresentationHint,
    QuestionId,
    ResolutionStatus,
    SingleSelectionQuestion,
    StateRevision,
    TextQuestion,
)
from avalan.interaction.store import (  # noqa: E402
    InteractionCorrelation,
    TerminalizeInteractionCommand,
)
from avalan.interaction.stores import (  # noqa: E402
    MemoryInteractionStoreFactory,
)

_EVIDENCE_PROPERTY = "failure_matrix_evidence"
_SURFACES = ("a2a-downstream", "a2a-server")


@dataclass(frozen=True, slots=True)
class _PendingTask:
    """Carry public correlation for one retained input-required task."""

    task_id: str
    context_id: str
    request_id: str
    request: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Observation:
    """Carry one exact public failure-matrix observation."""

    transition: tuple[str, str]
    public_result_id: str
    status: tuple[str, str]
    provider_calls: int


class _QuestionBroker:
    """Replace provider questions while preserving the real broker."""

    def __init__(
        self,
        broker: AsyncInteractionBroker,
        questions: tuple[object, ...],
    ) -> None:
        self._broker = broker
        self._questions = questions

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> object:
        """Create the transformed request through the real broker."""
        return await self._broker.request(
            replace(request, questions=cast(Any, self._questions))
        )

    async def inspect(self, query: object) -> object:
        """Delegate one public inspection."""
        return await self._broker.inspect(cast(Any, query))

    async def resolve(self, command: object) -> object:
        """Delegate one public resolution."""
        return await self._broker.resolve(cast(Any, command))

    async def cancel(self, command: object) -> object:
        """Delegate one public cancellation."""
        return await self._broker.cancel(cast(Any, command))

    async def cancel_scope(self, command: object) -> object:
        """Delegate one public scope cancellation."""
        return await self._broker.cancel_scope(cast(Any, command))

    async def wait(self, command: object) -> object:
        """Delegate one public wait."""
        return await self._broker.wait(cast(Any, command))

    async def aclose(self) -> None:
        """Close the real broker."""
        await self._broker.aclose()


class _RetainedResponse:
    """Pause the provider after resumption becomes observable."""

    def __init__(
        self,
        response: AsyncIterator[object],
        resumed: Event,
        release: Event,
    ) -> None:
        self._response = response
        self._resumed = resumed
        self._release = release
        self.input_token_count = 0
        self.output_token_count = 0

    def __aiter__(self) -> "_RetainedResponse":
        return self

    async def __anext__(self) -> object:
        item = await anext(self._response)
        if (
            getattr(getattr(item, "kind", None), "value", None)
            == "interaction.answered"
        ):
            self._resumed.set()
            await self._release.wait()
        return item

    async def aclose(self) -> None:
        """Close the wrapped provider response."""
        close = getattr(self._response, "aclose", None)
        if close is not None:
            await close()


class _RetainedProvider(contract.server_support._FakeProviderOrchestrator):
    """Retain one resumed provider stream for duplicate-resolution probes."""

    def __init__(self) -> None:
        super().__init__()
        self.resumed = Event()
        self.release = Event()

    async def __call__(
        self,
        messages: object,
        settings: object | None = None,
        **kwargs: object,
    ) -> AsyncIterator[object]:
        response = await super().__call__(messages, settings, **kwargs)
        return cast(
            AsyncIterator[object],
            _RetainedResponse(response, self.resumed, self.release),
        )


class _InputSequenceProvider(
    contract.server_support._FakeProviderOrchestrator
):
    """Expose one or more retained broker requests in a single stream."""

    def __init__(self, modes: tuple[object, ...]) -> None:
        super().__init__()
        self._modes = modes

    async def _stream(self, runtime: object) -> AsyncIterator[object]:
        support = contract.server_support
        assert isinstance(runtime, support.AttachedInteractionRuntime)
        origin = support._origin(runtime.actor.principal)
        self.provider_calls += 1
        yield support._projection(
            origin,
            0,
            support.StreamItemKind.STREAM_STARTED,
        )
        sequence = 1
        for index, mode in enumerate(self._modes):
            delivered = Event()

            async def handler(context: object) -> object:
                assert isinstance(context, support.InputHandlerContext)
                self.request = context.request
                self.request_ready.set()
                delivered.set()
                return await runtime.handler(context)

            request_task = create_task(
                runtime.broker.request(
                    support.InteractionBrokerRequest(
                        actor=runtime.actor,
                        origin=origin,
                        mode=mode,
                        reason=f"Confirm continuation {index}.",
                        questions=(
                            support.ConfirmationQuestion(
                                question_id=support.QuestionId("continue"),
                                prompt="Continue?",
                                required=True,
                            ),
                            support.TextQuestion(
                                question_id=support.QuestionId("note"),
                                prompt="Optional note",
                                required=False,
                                constraints=support.TextValidationConstraints(
                                    minimum_length=1,
                                    maximum_length=12,
                                ),
                            ),
                        ),
                        handler=handler,
                        continuation_ttl_seconds=600,
                        advisory_wait_seconds=(
                            1
                            if mode is support.RequirementMode.ADVISORY
                            else None
                        ),
                    )
                )
            )
            self._active_tasks.append(request_task)
            await delivered.wait()
            assert self.request is not None
            correlation = support._correlation(self.request)
            yield support._projection(
                origin,
                sequence,
                support.StreamItemKind.INTERACTION_CREATED,
                correlation,
            )
            sequence += 1
            yield support._projection(
                origin,
                sequence,
                support.StreamItemKind.INTERACTION_PENDING,
                correlation,
            )
            sequence += 1
            result = await request_task
            assert result.delivery is not None
            terminal_request = result.delivery.record.request
            terminal_kind = {
                support.RequestState.ANSWERED: (
                    support.StreamItemKind.INTERACTION_ANSWERED
                ),
                support.RequestState.TIMED_OUT: (
                    support.StreamItemKind.INTERACTION_TIMED_OUT
                ),
            }[terminal_request.state]
            self.provider_calls += 1
            self.continuation_started.set()
            if self.continuation_gate is not None:
                await self.continuation_gate.wait()
            yield support._projection(
                origin,
                sequence,
                terminal_kind,
                correlation,
            )
            sequence += 1
        yield support._projection(
            origin,
            sequence,
            support.StreamItemKind.ANSWER_DELTA,
            text_delta="Confirmed.",
        )
        sequence += 1
        yield support._projection(
            origin,
            sequence,
            support.StreamItemKind.ANSWER_DONE,
        )
        sequence += 1
        yield support._projection(
            origin,
            sequence,
            support.StreamItemKind.STREAM_COMPLETED,
            usage={},
            terminal_outcome=support.StreamTerminalOutcome.COMPLETED,
        )
        yield support._projection(
            origin,
            sequence + 1,
            support.StreamItemKind.STREAM_CLOSED,
        )


class _AdvancingClock(InteractionClock):
    """Expose deterministic manual deadline advancement."""

    def __init__(self) -> None:
        self.wall_time = contract._NOW
        self.monotonic_seconds = 1.0
        self.changed = Event()

    async def read(self) -> InteractionTime:
        """Return the current coherent test time."""
        return InteractionTime.from_clock(
            wall_time=self.wall_time,
            monotonic_seconds=self.monotonic_seconds,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait until manual time reaches one deadline."""
        while self.monotonic_seconds < monotonic_deadline:
            await self.changed.wait()
            self.changed.clear()

    def advance(self, seconds: float) -> None:
        """Advance wall and monotonic time together."""
        self.wall_time += timedelta(seconds=seconds)
        self.monotonic_seconds += seconds
        self.changed.set()


async def _open_broker(clock: InteractionClock) -> AsyncInteractionBroker:
    """Open the real memory broker with an externally controlled clock."""
    policy = InteractionPolicy()
    identifiers = contract.server_support._Ids()
    classifier = contract.server_support._Classifier(policy)
    store = await MemoryInteractionStoreFactory(
        policy=policy,
        clock=clock,
        authorizer=contract.server_support._Authorizer(),
        id_factory=identifiers,
        classifier=classifier,
    ).open()
    return AsyncInteractionBroker(
        store=store,
        clock=clock,
        id_factory=identifiers,
        policy=policy,
        classifier=classifier,
    )


async def _start_pending(
    client: AsyncClient,
    *,
    label: str,
) -> _PendingTask:
    """Start one real A2A task and return its public input correlation."""
    response = await client.post(
        "/a2a",
        headers=contract._A2A_HEADERS,
        json=contract._send_envelope(
            rpc_id=f"rpc-{label}-initial",
            message_id=f"message-{label}-initial",
        ),
    )
    assert response.status_code == 200
    events = contract._sse_events(response.text)
    task = contract._task_from_event(events[0])
    update = contract._status_update(events[-1])
    assert contract._state_from_update(update) == "TASK_STATE_INPUT_REQUIRED"
    request = contract._extension_payload(contract._input_message(update))
    task_id = task.get("id")
    context_id = task.get("contextId")
    request_id = request.get("request_id")
    assert isinstance(task_id, str)
    assert isinstance(context_id, str)
    assert isinstance(request_id, str)
    return _PendingTask(task_id, context_id, request_id, request)


async def _stored_state(client: AsyncClient, task_id: str) -> str:
    """Return one public stored A2A task state."""
    response = await client.get(
        f"/a2a/tasks/{task_id}",
        headers=contract._A2A_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    return contract._state_from_task(body)


def _error(response: object, code: int, data_code: str) -> None:
    """Assert one exact normalized JSON-RPC error."""
    status_code = getattr(response, "status_code")
    assert status_code in {200, 409, 410}
    body = getattr(response, "json")()
    assert isinstance(body, dict)
    error = body.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == code
    data = error.get("data")
    assert isinstance(data, dict)
    assert data.get("code") == data_code


async def _validation_observation(
    condition_id: str,
) -> _Observation:
    """Reject one malformed resolution while retaining pending state."""
    raw_broker: AsyncInteractionBroker | None = None
    answers: Mapping[str, object]
    if condition_id == "INPUT-F-04":
        answers = {"continue": "yes", "note": "okay"}
    elif condition_id == "INPUT-F-05":
        raw_broker = await contract.server_support._open_broker()
        broker = _QuestionBroker(
            raw_broker,
            (
                SingleSelectionQuestion(
                    question_id=QuestionId("choice"),
                    prompt="Choose.",
                    required=True,
                    choices=(
                        Choice(
                            value=ChoiceValue("known"),
                            label="Known",
                        ),
                        Choice(
                            value=ChoiceValue("alternate"),
                            label="Alternate",
                        ),
                    ),
                ),
            ),
        )
        answers = {"choice": "unknown"}
    else:
        assert condition_id == "INPUT-F-06"
        answers = {"note": "okay"}
    selected_broker = cast(Any, broker) if raw_broker is not None else None
    async with contract._server(broker=selected_broker) as (
        app,
        _broker,
        provider,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(client, label=condition_id)
            response = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id=f"rpc-{condition_id}-invalid",
                    message_id=f"message-{condition_id}-invalid",
                    task_id=pending.task_id,
                    context_id=pending.context_id,
                    request_id=pending.request_id,
                    answers=answers,
                    method="SendMessage",
                ),
            )
            _error(response, -32602, "avalan.input.validation")
            assert (
                await _stored_state(client, pending.task_id)
                == "TASK_STATE_INPUT_REQUIRED"
            )
        return _Observation(
            ("pending", "pending"),
            "a2a.invalid_params_error.v1",
            ("jsonrpc_error", "-32602"),
            provider.provider_calls,
        )


async def _unavailable_observation() -> _Observation:
    """Reject input-required work when no capable A2A host is available."""
    isolation = await contract._isolation_observation()
    assert isolation["unavailable_status"] == 503
    body = cast(Mapping[str, object], isolation["unavailable_body"])
    error = cast(Mapping[str, object], body.get("error"))
    assert error.get("code") == -31910
    assert isolation["unavailable_provider_calls"] == 0
    return _Observation(
        ("created", "unavailable"),
        "a2a.unavailable_error.v1",
        ("jsonrpc_error", "-31910"),
        0,
    )


async def _duplicate_observation(*, conflicting: bool) -> _Observation:
    """Repeat or conflict with an answer while its task is working."""
    provider = _RetainedProvider()
    async with contract._server(provider=provider) as (
        app,
        _broker,
        _provider,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(
                client,
                label="conflict" if conflicting else "replay",
            )
            winner = create_task(
                client.post(
                    "/a2a",
                    headers=contract._A2A_HEADERS,
                    json=contract._send_envelope(
                        rpc_id="rpc-winner",
                        message_id="message-winner",
                        task_id=pending.task_id,
                        context_id=pending.context_id,
                        request_id=pending.request_id,
                    ),
                )
            )
            await wait_for(provider.resumed.wait(), timeout=2)
            assert provider.provider_calls == 2
            repeated = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id="rpc-repeated",
                    message_id="message-repeated",
                    task_id=pending.task_id,
                    context_id=pending.context_id,
                    request_id=pending.request_id,
                    answers={
                        "continue": not conflicting,
                        "note": "okay",
                    },
                    method="SendMessage",
                    return_immediately=not conflicting,
                ),
            )
            if conflicting:
                _error(
                    repeated,
                    -31911,
                    "avalan.input.already_resolved",
                )
            else:
                body = repeated.json()
                assert isinstance(body, dict)
                result = body.get("result")
                assert isinstance(result, dict)
                task = result.get("task")
                assert isinstance(task, dict)
                assert contract._state_from_task(task) == "TASK_STATE_WORKING"
            provider.release.set()
            completed = await wait_for(winner, timeout=2)
            assert completed.status_code == 200
        return _Observation(
            ("answered", "answered"),
            (
                "a2a.conflict_error.v1"
                if conflicting
                else "a2a.task_working.v1"
            ),
            (
                ("jsonrpc_error", "-31911")
                if conflicting
                else ("task_state", "TASK_STATE_WORKING")
            ),
            provider.provider_calls,
        )


async def _expired_observation() -> _Observation:
    """Expire a real pending request before its public resolution."""
    clock = _AdvancingClock()
    broker = await _open_broker(clock)
    async with contract._server(broker=broker) as (app, _broker, provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(client, label="expired")
            clock.advance(601)
            await wait_for(provider._active_tasks[0], timeout=2)
            response = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id="rpc-expired",
                    message_id="message-expired",
                    task_id=pending.task_id,
                    context_id=pending.context_id,
                    request_id=pending.request_id,
                    method="SendMessage",
                ),
            )
            _error(response, -31912, "avalan.input.expired")
        return _Observation(
            ("pending", "expired"),
            "a2a.expired_error.v1",
            ("jsonrpc_error", "-31912"),
            provider.provider_calls,
        )


async def _cancelled_observation() -> _Observation:
    """Cancel one real retained A2A input task."""
    async with contract._server() as (app, _broker, provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(client, label="cancelled")
            response = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json={
                    "jsonrpc": "2.0",
                    "id": "rpc-cancelled",
                    "method": "CancelTask",
                    "params": {"id": pending.task_id},
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert isinstance(body, dict)
            result = body.get("result")
            assert isinstance(result, dict)
            assert contract._state_from_task(result) in {
                "TASK_STATE_INPUT_REQUIRED",
                "TASK_STATE_CANCELED",
            }
            await sleep(0)
            assert (
                await _stored_state(client, pending.task_id)
                == "TASK_STATE_CANCELED"
            )
        return _Observation(
            ("pending", "cancelled"),
            "a2a.task_cancelled.v1",
            ("task_state", "TASK_STATE_CANCELED"),
            provider.provider_calls,
        )


async def _superseded_observation() -> _Observation:
    """Supersede one real pending request before public resolution."""
    async with contract._server() as (app, broker, provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(client, label="superseded")
            request = provider.request
            assert request is not None
            result = await broker.terminalize(
                TerminalizeInteractionCommand(
                    actor=contract.server_support._OWNER,
                    correlation=InteractionCorrelation.from_request(request),
                    status=ResolutionStatus.SUPERSEDED,
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    expected_state_revision=StateRevision(
                        request.state_revision
                    ),
                )
            )
            assert (
                result.store_result.record.request.state.value == "superseded"
            )
            response = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id="rpc-superseded",
                    message_id="message-superseded",
                    task_id=pending.task_id,
                    context_id=pending.context_id,
                    request_id=pending.request_id,
                    method="SendMessage",
                ),
            )
            _error(response, -31913, "avalan.input.superseded")
        return _Observation(
            ("pending", "superseded"),
            "a2a.superseded_error.v1",
            ("jsonrpc_error", "-31913"),
            provider.provider_calls,
        )


async def _pending_observation() -> _Observation:
    """Refresh one retained input-required task without guessing an answer."""
    async with contract._server() as (app, _broker, provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(client, label="pending")
            response = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id="rpc-pending-refresh",
                    message_id="message-pending-refresh",
                    task_id=pending.task_id,
                    context_id=pending.context_id,
                    method="SendMessage",
                    extensions=(),
                ),
            )
            body = response.json()
            assert isinstance(body, dict)
            result = body.get("result")
            assert isinstance(result, dict)
            task = result.get("task")
            assert isinstance(task, dict)
            assert (
                contract._state_from_task(task) == "TASK_STATE_INPUT_REQUIRED"
            )
        return _Observation(
            ("pending", "pending"),
            "a2a.task_input_required.v1",
            ("task_state", "TASK_STATE_INPUT_REQUIRED"),
            provider.provider_calls,
        )


async def _ui_hint_observation() -> _Observation:
    """Answer despite an unsupported advisory presentation hint."""
    raw_broker = await contract.server_support._open_broker()
    broker = _QuestionBroker(
        raw_broker,
        (
            TextQuestion(
                question_id=QuestionId("name"),
                prompt="Name?",
                required=True,
                header="Preferred rendering",
                help_text="A single-line control is ideal.",
                presentation_hint=PresentationHint.SINGLE_LINE,
            ),
        ),
    )
    async with contract._server(broker=broker) as (app, _broker, provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(client, label="ui-hint")
            questions = pending.request.get("questions")
            assert isinstance(questions, list)
            assert questions[0]["presentation_hint"] == "single_line"
            response = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id="rpc-ui-hint-answer",
                    message_id="message-ui-hint-answer",
                    task_id=pending.task_id,
                    context_id=pending.context_id,
                    request_id=pending.request_id,
                    answers={"name": "Ada"},
                ),
            )
            events = contract._sse_events(response.text)
            states = tuple(
                contract._state_from_update(contract._status_update(event))
                for event in events
                if "statusUpdate" in contract._result(event)
            )
            assert states == (
                "TASK_STATE_WORKING",
                "TASK_STATE_COMPLETED",
            )
        return _Observation(
            ("pending", "answered"),
            "a2a.task_working.v1",
            ("task_state", "TASK_STATE_WORKING"),
            provider.provider_calls,
        )


async def _extension_required_observation() -> _Observation:
    """Reject a peer that lacks the required structured-input extension."""
    async with contract._server(input_extension_required=True) as (
        app,
        _broker,
        provider,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            response = await client.post(
                "/a2a",
                headers={
                    "A2A-Version": "1.0",
                    "Authorization": "Bearer owner",
                },
                json=contract._send_envelope(
                    rpc_id="rpc-extension-required",
                    message_id="message-extension-required",
                ),
            )
            assert response.status_code == 400
            body = response.json()
            assert isinstance(body, dict)
            error = body.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == -32008
            assert error.get("data") == {
                "code": "avalan.input.extension_required"
            }
        return _Observation(
            ("created", "unavailable"),
            "a2a.extension_required_error.v1",
            ("jsonrpc_error", "-32008"),
            provider.provider_calls,
        )


_EXPECTED = {
    "INPUT-F-01": _Observation(
        ("created", "unavailable"),
        "a2a.unavailable_error.v1",
        ("jsonrpc_error", "-31910"),
        0,
    ),
    "INPUT-F-04": _Observation(
        ("pending", "pending"),
        "a2a.invalid_params_error.v1",
        ("jsonrpc_error", "-32602"),
        1,
    ),
    "INPUT-F-05": _Observation(
        ("pending", "pending"),
        "a2a.invalid_params_error.v1",
        ("jsonrpc_error", "-32602"),
        1,
    ),
    "INPUT-F-06": _Observation(
        ("pending", "pending"),
        "a2a.invalid_params_error.v1",
        ("jsonrpc_error", "-32602"),
        1,
    ),
    "INPUT-F-07": _Observation(
        ("answered", "answered"),
        "a2a.task_working.v1",
        ("task_state", "TASK_STATE_WORKING"),
        2,
    ),
    "INPUT-F-08": _Observation(
        ("answered", "answered"),
        "a2a.conflict_error.v1",
        ("jsonrpc_error", "-31911"),
        2,
    ),
    "INPUT-F-09": _Observation(
        ("pending", "expired"),
        "a2a.expired_error.v1",
        ("jsonrpc_error", "-31912"),
        1,
    ),
    "INPUT-F-10": _Observation(
        ("pending", "cancelled"),
        "a2a.task_cancelled.v1",
        ("task_state", "TASK_STATE_CANCELED"),
        1,
    ),
    "INPUT-F-11": _Observation(
        ("pending", "superseded"),
        "a2a.superseded_error.v1",
        ("jsonrpc_error", "-31913"),
        1,
    ),
    "INPUT-F-12": _Observation(
        ("pending", "pending"),
        "a2a.task_input_required.v1",
        ("task_state", "TASK_STATE_INPUT_REQUIRED"),
        1,
    ),
    "INPUT-F-13": _Observation(
        ("pending", "timed_out"),
        "a2a.task_working.v1",
        ("task_state", "TASK_STATE_WORKING"),
        2,
    ),
    "INPUT-F-14": _Observation(
        ("pending", "answered"),
        "a2a.task_working.v1",
        ("task_state", "TASK_STATE_WORKING"),
        2,
    ),
    "INPUT-F-15": _Observation(
        ("created", "unavailable"),
        "a2a.extension_required_error.v1",
        ("jsonrpc_error", "-32008"),
        0,
    ),
}


async def _observe(condition_id: str) -> _Observation:
    """Dispatch one executable condition to its real boundary probe."""
    if condition_id == "INPUT-F-01":
        return await _unavailable_observation()
    if condition_id in {"INPUT-F-04", "INPUT-F-05", "INPUT-F-06"}:
        return await _validation_observation(condition_id)
    if condition_id == "INPUT-F-07":
        return await _duplicate_observation(conflicting=False)
    if condition_id == "INPUT-F-08":
        return await _duplicate_observation(conflicting=True)
    if condition_id == "INPUT-F-09":
        return await _expired_observation()
    if condition_id == "INPUT-F-10":
        return await _cancelled_observation()
    if condition_id == "INPUT-F-11":
        return await _superseded_observation()
    if condition_id == "INPUT-F-12":
        return await _pending_observation()
    if condition_id == "INPUT-F-13":
        return await _advisory_timeout_observation()
    if condition_id == "INPUT-F-14":
        return await _ui_hint_observation()
    assert condition_id == "INPUT-F-15"
    return await _extension_required_observation()


def _assert_failure(
    condition_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Assert and record one frozen A2A failure-matrix invariant."""
    observation = run(_observe(condition_id))
    assert observation == _EXPECTED[condition_id]
    evidence = [
        {
            "condition_id": condition_id,
            "surface_id": surface,
            "transition_from": observation.transition[0],
            "transition_to": observation.transition[1],
            "public_result_id": observation.public_result_id,
            "public_result": {"redacted": True},
            "status_key": observation.status[0],
            "status_value": observation.status[1],
            "provider_call_count": observation.provider_calls,
            "domain_side_effect_count": 0,
        }
        for surface in _SURFACES
    ]
    record_property(_EVIDENCE_PROPERTY, evidence)


def test_input_f_01(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-01", record_property)


def test_input_f_04(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-04", record_property)


def test_input_f_05(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-05", record_property)


def test_input_f_06(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-06", record_property)


def test_input_f_07(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-07", record_property)


def test_input_f_08(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-08", record_property)


def test_input_f_09(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-09", record_property)


def test_input_f_10(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-10", record_property)


def test_input_f_11(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-11", record_property)


def test_input_f_12(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-12", record_property)


def test_input_f_13(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-13", record_property)


def test_input_f_14(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-14", record_property)


def test_input_f_15(record_property: Callable[[str, object], None]) -> None:
    _assert_failure("INPUT-F-15", record_property)


async def _sequential_pending_observation() -> None:
    """Resolve two sequential requests retained by one provider stream."""
    provider = _InputSequenceProvider(
        (
            contract.server_support.RequirementMode.REQUIRED,
            contract.server_support.RequirementMode.REQUIRED,
        )
    )
    async with contract._server(provider=provider) as (
        app,
        _broker,
        _provider,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            first = await _start_pending(client, label="sequential-first")
            second_response = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id="rpc-sequential-first",
                    message_id="message-sequential-first",
                    task_id=first.task_id,
                    context_id=first.context_id,
                    request_id=first.request_id,
                ),
            )
            second_events = contract._sse_events(second_response.text)
            second_update = contract._status_update(second_events[-1])
            assert (
                contract._state_from_update(second_update)
                == "TASK_STATE_INPUT_REQUIRED"
            )
            second_request = contract._extension_payload(
                contract._input_message(second_update)
            )
            second_request_id = second_request.get("request_id")
            assert isinstance(second_request_id, str)
            assert second_request_id != first.request_id
            completed = await client.post(
                "/a2a",
                headers=contract._A2A_HEADERS,
                json=contract._send_envelope(
                    rpc_id="rpc-sequential-second",
                    message_id="message-sequential-second",
                    task_id=first.task_id,
                    context_id=first.context_id,
                    request_id=second_request_id,
                ),
            )
            completed_events = contract._sse_events(completed.text)
            assert (
                contract._state_from_update(
                    contract._status_update(completed_events[-1])
                )
                == "TASK_STATE_COMPLETED"
            )
            assert provider.provider_calls == 3


def test_a2a_retains_sequential_pending_requests() -> None:
    """Resume each sequential input request without draining past it."""
    run(_sequential_pending_observation())


async def _advisory_timeout_observation() -> _Observation:
    """Observe autonomous A2A continuation after an advisory timeout."""
    clock = _AdvancingClock()
    broker = await _open_broker(clock)
    provider = _InputSequenceProvider(
        (contract.server_support.RequirementMode.ADVISORY,)
    )
    provider.continuation_gate = Event()
    async with contract._server(broker=broker, provider=provider) as (
        app,
        _broker,
        _provider,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            pending = await _start_pending(client, label="advisory-timeout")
            clock.advance(2)
            await wait_for(provider.continuation_started.wait(), timeout=2)
            assert provider.provider_calls == 2
            for _ in range(20):
                if (
                    await _stored_state(client, pending.task_id)
                    == "TASK_STATE_WORKING"
                ):
                    break
                await sleep(0)
            else:
                raise AssertionError("advisory task did not resume to working")
            provider.continuation_gate.set()
            for _ in range(20):
                if (
                    await _stored_state(client, pending.task_id)
                    == "TASK_STATE_COMPLETED"
                ):
                    break
                await sleep(0)
            else:
                raise AssertionError("advisory task did not complete")
            assert provider.provider_calls == 2
        return _Observation(
            ("pending", "timed_out"),
            "a2a.task_working.v1",
            ("task_state", "TASK_STATE_WORKING"),
            provider.provider_calls,
        )


def test_a2a_advisory_timeout_resumes_without_follow_up() -> None:
    """Resume one timed-out advisory without a client follow-up."""
    run(_advisory_timeout_observation())
