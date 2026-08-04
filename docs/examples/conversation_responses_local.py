"""Exercise the served Responses contract using a process-local transport."""

from asyncio import run
from json import dumps, loads
from typing import Any, cast

import httpx
from pydantic import ValidationError

from avalan.server.entities import ResponsesCompactRequest, ResponsesRequest
from avalan.server.responses_schema import (
    ResponsesCompactResource,
    ResponsesDeletedResource,
    ResponsesErrorEnvelope,
    ResponsesResource,
)


class LocalResponsesTransport(httpx.AsyncBaseTransport):
    """Provide a deterministic, non-network Responses demonstration."""

    def __init__(self) -> None:
        self._sequence = 0
        self._stored: dict[str, dict[str, object]] = {}

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        """Validate requests and return deterministic local resources."""
        path = request.url.path
        if request.method == "POST" and path == "/v1/responses/compact":
            return self._compact(request)
        if request.method == "POST" and path == "/v1/responses":
            return self._create(request)
        prefix = "/v1/responses/"
        if path.startswith(prefix):
            response_id = path.removeprefix(prefix)
            if request.method == "GET":
                return self._retrieve(request, response_id)
            if request.method == "DELETE":
                return self._delete(request, response_id)
        return self._error(request, 404, "conversation state is unavailable")

    def _create(self, request: httpx.Request) -> httpx.Response:
        try:
            body = ResponsesRequest.model_validate(loads(request.content))
        except (ValidationError, UnicodeDecodeError, ValueError):
            return self._error(request, 400, "conversation input is invalid")
        if body.stream:
            stream_body = "\n".join(
                (
                    "event: response.created",
                    'data: {"type":"response.created"}',
                    "",
                    "event: response.output_text.delta",
                    (
                        'data: {"type":"response.output_text.delta",'
                        '"delta":"local stream answer"}'
                    ),
                    "",
                    "event: response.completed",
                    'data: {"type":"response.completed"}',
                    "",
                    "data: [DONE]",
                    "",
                )
            )
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=stream_body.encode(),
            )
        self._sequence += 1
        response_id = f"resp_avl_local_{self._sequence:04d}"
        previous = body.previous_response_id
        if previous is not None and previous not in self._stored:
            return self._error(
                request,
                404,
                "conversation state is unavailable",
            )
        output_text = (
            "stored continuation answer"
            if previous is not None
            else (
                "stored first answer"
                if body.store
                else "stateless local answer"
            )
        )
        resource = self._resource(response_id, output_text)
        if body.store:
            self._stored[response_id] = resource
        return httpx.Response(200, request=request, json=resource)

    def _compact(self, request: httpx.Request) -> httpx.Response:
        try:
            body = ResponsesCompactRequest.model_validate(
                loads(request.content)
            )
        except (ValidationError, UnicodeDecodeError, ValueError):
            return self._error(request, 400, "conversation input is invalid")
        assert body.input is not None
        compact = {
            "id": "resp_avl_local_compact",
            "created_at": 1_893_456_000,
            "object": "response.compaction",
            "output": [
                {
                    "type": "compaction",
                    "encrypted_content": "redacted-local-fixture",
                }
            ],
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            },
        }
        validated = ResponsesCompactResource.model_validate(compact)
        return httpx.Response(
            200,
            request=request,
            json=validated.model_dump(mode="json"),
        )

    def _retrieve(
        self,
        request: httpx.Request,
        response_id: str,
    ) -> httpx.Response:
        resource = self._stored.get(response_id)
        if resource is None:
            return self._error(
                request,
                404,
                "conversation state is unavailable",
            )
        return httpx.Response(200, request=request, json=resource)

    def _delete(
        self,
        request: httpx.Request,
        response_id: str,
    ) -> httpx.Response:
        if self._stored.pop(response_id, None) is None:
            return self._error(
                request,
                404,
                "conversation state is unavailable",
            )
        deleted = ResponsesDeletedResource.model_validate(
            {
                "id": response_id,
                "object": "response.deleted",
                "deleted": True,
                "metadata": {
                    "avalan_local_deletion": "tombstoned",
                    "avalan_upstream_deletion": "reconciled",
                },
            }
        )
        return httpx.Response(
            200,
            request=request,
            json=deleted.model_dump(mode="json"),
        )

    def _resource(
        self,
        response_id: str,
        output_text: str,
    ) -> dict[str, object]:
        resource = ResponsesResource.model_validate(
            {
                "id": response_id,
                "object": "response",
                "type": "response",
                "created_at": 1_893_456_000,
                "created": 1_893_456_000,
                "model": "deterministic-local-model",
                "status": "completed",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": f"msg_{response_id}",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": output_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "metadata": {
                    "avalan_lifecycle": "published",
                    "avalan_checkpoint_digest": "0" * 64,
                },
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                    "input_text_tokens": 1,
                    "output_text_tokens": 1,
                },
            }
        )
        return cast(dict[str, object], resource.model_dump(mode="json"))

    def _error(
        self,
        request: httpx.Request,
        status: int,
        message: str,
    ) -> httpx.Response:
        envelope = ResponsesErrorEnvelope.model_validate(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "conversation_validation_failed",
                    "param": None,
                }
            }
        )
        return httpx.Response(
            status,
            request=request,
            json=envelope.model_dump(mode="json"),
        )


def _output_text(resource: dict[str, Any]) -> str:
    """Return visible text without inspecting private continuation state."""
    output = cast(list[dict[str, Any]], resource["output"])
    content = cast(list[dict[str, Any]], output[-1]["content"])
    return cast(str, content[-1]["text"])


async def _stream_event_types(client: httpx.AsyncClient) -> list[str]:
    """Consume served events through the terminal marker."""
    event_types: list[str] = []
    async with client.stream(
        "POST",
        "/responses",
        json={
            "model": "deterministic-local-model",
            "input": "Stream locally.",
            "store": False,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = cast(
                dict[str, object], loads(line.removeprefix("data: "))
            )
            event_types.append(cast(str, payload["type"]))
    return event_types


async def run_example() -> dict[str, object]:
    """Run stateless, stored, compact, lifecycle, and stream examples."""
    transport = LocalResponsesTransport()
    async with httpx.AsyncClient(
        base_url="https://local.example.invalid/v1",
        transport=transport,
    ) as client:
        first = (
            (
                await client.post(
                    "/responses",
                    json={
                        "model": "deterministic-local-model",
                        "input": "Start stateless continuity.",
                        "store": False,
                    },
                )
            )
            .raise_for_status()
            .json()
        )
        history = [
            {
                "type": "message",
                "role": "user",
                "content": "Start stateless continuity.",
            },
            *cast(list[dict[str, object]], first["output"]),
            {
                "type": "message",
                "role": "user",
                "content": "Continue with complete ordered replay.",
            },
        ]
        second = (
            (
                await client.post(
                    "/responses",
                    json={
                        "model": "deterministic-local-model",
                        "input": history,
                        "store": False,
                    },
                )
            )
            .raise_for_status()
            .json()
        )
        compact_history = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Start stateless continuity.",
                    }
                ],
            },
            *cast(list[dict[str, object]], first["output"]),
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Continue with complete ordered replay.",
                    }
                ],
            },
        ]
        compact = (
            (
                await client.post(
                    "/responses/compact",
                    json={
                        "model": "deterministic-local-model",
                        "input": compact_history,
                    },
                )
            )
            .raise_for_status()
            .json()
        )

        stored = (
            (
                await client.post(
                    "/responses",
                    json={
                        "model": "deterministic-local-model",
                        "input": "Start disclosed stored continuity.",
                        "store": True,
                    },
                )
            )
            .raise_for_status()
            .json()
        )
        stored_id = cast(str, stored["id"])
        chained = (
            (
                await client.post(
                    "/responses",
                    json={
                        "model": "deterministic-local-model",
                        "input": "Continue by public response ID.",
                        "store": True,
                        "previous_response_id": stored_id,
                    },
                )
            )
            .raise_for_status()
            .json()
        )
        retrieved = (
            (await client.get(f"/responses/{stored_id}"))
            .raise_for_status()
            .json()
        )
        deleted = (
            (await client.delete(f"/responses/{stored_id}"))
            .raise_for_status()
            .json()
        )
        stream_events = await _stream_event_types(client)

    return {
        "stateless_outputs": [_output_text(first), _output_text(second)],
        "compact_output_item_count": len(compact["output"]),
        "stored_output": _output_text(stored),
        "chained_output": _output_text(chained),
        "retrieved_same_public_id": retrieved["id"] == stored_id,
        "deleted": deleted["deleted"],
        "stream_event_types": stream_events,
    }


if __name__ == "__main__":
    print(dumps(run(run_example()), indent=2, sort_keys=True))
