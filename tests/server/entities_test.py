from avalan.server.entities import (
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from avalan.entities import MessageRole
from pydantic import ValidationError
from unittest import TestCase


class ChatEntitiesTestCase(TestCase):
    def test_request_defaults(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="hi")
        req = ChatCompletionRequest(model="m", messages=[msg])
        self.assertEqual(req.temperature, 1.0)
        self.assertEqual(req.top_p, 1.0)
        self.assertEqual(req.n, 1)
        self.assertFalse(req.stream)

    def test_request_validation_error(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="hi")
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="m", messages=[msg], temperature=3.0)

    def test_response_serialization(self) -> None:
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="ok")
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        usage = ChatCompletionUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )
        resp = ChatCompletionResponse(
            id="1",
            created=123,
            model="m",
            choices=[choice],
            usage=usage,
        )
        data = resp.model_dump()
        self.assertEqual(data["choices"][0]["message"]["content"], "ok")
        json_str = resp.model_dump_json()
        self.assertIn('"chat.completion"', json_str)
