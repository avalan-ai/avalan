from logging import Logger
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...agent.orchestrator import Orchestrator
from .. import di_get_logger, di_get_orchestrator
from ..a2a.agent import build_agent_card
from ..a2a.schema import dump_payload, task_status_value, validate_task
from ..a2a.store import A2ATaskStore
from ..a2a.translator import A2ATranslator, event_to_sse
from ..entities import ResponsesRequest
from . import orchestrate


router = APIRouter(tags=["a2a"])


def get_task_store(request: Request) -> A2ATaskStore:
    store = getattr(request.app.state, "a2a_task_store", None)
    if not isinstance(store, A2ATaskStore):
        raise HTTPException(status_code=500, detail="A2A task store not available")
    return store


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _prefix(request: Request) -> str:
    prefix = getattr(request.app.state, "a2a_prefix", "/a2a")
    return prefix or ""


@router.get("/.well-known/agent.json")
async def agent_card(
    request: Request,
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
):
    base_url = _base_url(request)
    prefix = _prefix(request)
    card = build_agent_card(
        orchestrator,
        name=request.app.title or "Avalan Agent",
        version=request.app.version or "0.0.0",
        base_url=base_url,
        prefix=prefix,
    )
    return JSONResponse(card)


@router.post("/tasks")
async def create_task(
    request: ResponsesRequest,
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
    store: A2ATaskStore = Depends(get_task_store),
    logger: Logger = Depends(di_get_logger),
):
    input_messages = [msg.model_dump() for msg in request.messages]
    task = store.create_task(input_messages, None)

    response, _, _ = await orchestrate(request, logger, orchestrator)

    translator = A2ATranslator(store, task)

    if request.stream:

        async def event_stream() -> AsyncIterator[str]:
            try:
                for event in await translator.start():
                    yield event_to_sse(event)
                try:
                    async for token in response:
                        for event in await translator.token(token):
                            yield event_to_sse(event)
                except Exception:
                    for event in await translator.finish(succeeded=False):
                        yield event_to_sse(event)
                    raise
                for event in await translator.finish():
                    yield event_to_sse(event)
            finally:
                await orchestrator.sync_messages()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    await translator.start()
    try:
        async for token in response:
            await translator.token(token)
    except Exception:
        await translator.finish(succeeded=False)
        await orchestrator.sync_messages()
        raise
    await translator.finish()
    await orchestrator.sync_messages()

    stored_task = store.get(task.id)
    assert stored_task
    task_data = validate_task(
        {
            "id": stored_task.id,
            "status": task_status_value(stored_task.status),
            "input": stored_task.input_messages,
            "output": stored_task.output_messages,
            "artifacts": list(stored_task.artifacts.values()),
            "metadata": stored_task.metadata,
        }
    )
    return JSONResponse(dump_payload(task_data))


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, store: A2ATaskStore = Depends(get_task_store)):
    task = store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    body = validate_task(
        {
            "id": task.id,
            "status": task_status_value(task.status),
            "input": task.input_messages,
            "output": task.output_messages,
            "artifacts": list(task.artifacts.values()),
            "metadata": task.metadata,
        }
    )
    return JSONResponse(dump_payload(body))


@router.get("/tasks/{task_id}/events")
async def get_task_events(
    task_id: str,
    store: A2ATaskStore = Depends(get_task_store),
):
    try:
        iterator = store.subscribe(task_id)
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=404, detail="Task not found") from exc

    async def stream() -> AsyncIterator[str]:
        async for event in iterator:
            yield event_to_sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
