from unittest import TestCase

from pydantic import ValidationError

from avalan.entities import MessageRole
from avalan.server.entities import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ContentFile,
    ContentText,
)


class ChatEntitiesTestCase(TestCase):
    def test_request_defaults(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="hi")
        req = ChatCompletionRequest(messages=[msg])
        self.assertIsNone(req.model)
        self.assertEqual(req.temperature, 1.0)
        self.assertEqual(req.top_p, 1.0)
        self.assertEqual(req.n, 1)
        self.assertFalse(req.stream)

    def test_request_validation_error(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="hi")
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="m", messages=[msg], temperature=3.0)

    def test_chat_message_accepts_input_text_blocks(self) -> None:
        msg = ChatMessage(
            role=MessageRole.USER,
            content=[ContentText(type="input_text", text="hi")],
        )

        self.assertEqual(msg.content[0].type, "input_text")
        self.assertEqual(msg.content[0].text, "hi")

    def test_content_file_rejects_empty_file_data(self) -> None:
        with self.assertRaises(ValidationError):
            ContentFile(type="input_file", file_data="")

    def test_content_file_rejects_empty_data_url_payload(self) -> None:
        with self.assertRaises(ValidationError):
            ContentFile(
                type="input_file",
                filename="report.pdf",
                file_data="data:application/pdf;base64,",
            )

    def test_content_file_rejects_nested_empty_file_data(self) -> None:
        with self.assertRaises(ValidationError):
            ContentFile(
                type="input_file",
                file={"file_data": "", "filename": "report.pdf"},
            )

    def test_content_file_accepts_non_empty_data_url_payload(self) -> None:
        content = ContentFile(
            type="input_file",
            filename="report.pdf",
            file_data="data:application/pdf;base64,YWJj",
        )

        self.assertEqual(content.file_data, "data:application/pdf;base64,YWJj")

    def test_content_file_accepts_nested_non_empty_data_url_payload(
        self,
    ) -> None:
        content = ContentFile(
            type="input_file",
            file={
                "file_data": "data:application/pdf;base64,YWJj",
                "filename": "report.pdf",
            },
        )

        self.assertEqual(
            content.file["file_data"], "data:application/pdf;base64,YWJj"
        )

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
