import asyncio

from avalan.entities import (
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallResult,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.server.a2a.store import TaskStore
from avalan.server.a2a.router import A2AResponseTranslator


def test_translator_updates_task_store() -> None:
    asyncio.run(_run_translator_flow())


async def _run_translator_flow() -> None:
    store = TaskStore()
    task_id = "task-1"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )

    translator = A2AResponseTranslator(task_id, store)

    base_call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    tool_result = ToolCallResult(
        id="result-1",
        call=base_call,
        result="ok",
        name=base_call.name,
        arguments=base_call.arguments,
    )

    async def stream():
        yield ReasoningToken("thinking")
        yield ToolCallToken(token="", call=base_call)
        yield Event(
            type=EventType.TOOL_RESULT, payload={"result": tool_result}
        )
        yield Token(token="hello")

    await translator.consume(stream())

    task = await store.get_task(task_id)
    assert task["status"] == "completed"
    assert task["messages"][-1]["content"][0]["text"] == "hello"
    assert task["artifacts"][-1]["state"] == "completed"

    events = await store.get_events(task_id)
    event_names = {event["event"] for event in events}
    assert "message.delta" in event_names
    assert "artifact.completed" in event_names
