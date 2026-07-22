from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
)
from avalan.model.capability import CorrelatedCapabilityResult
from avalan.model.vendor import TextGenerationVendor


class DummyVendor(TextGenerationVendor):
    async def __call__(self, *args, **kwargs):
        return await super().__call__(*args, **kwargs)


class VendorTemplateMessagesTestCase(IsolatedAsyncioTestCase):
    def test_base_capability_result_message_is_not_implemented(self) -> None:
        result = CorrelatedCapabilityResult(
            call_id="call-1",
            canonical_name="lookup",
            provider_name="lookup",
            payload={"value": 1},
        )

        with self.assertRaises(NotImplementedError):
            TextGenerationVendor.capability_result_message(result)

    def test_non_stream_tool_call_validates_and_defaults_arguments(
        self,
    ) -> None:
        call = TextGenerationVendor.non_stream_tool_call(
            call_id="call-1",
            provider_name="lookup",
            arguments=None,
            capability=None,
            provider_family="openai",
            provider_event_type="chat.completion.tool_call",
        )

        self.assertEqual(call.call_id, "call-1")
        self.assertEqual(call.name, "lookup")
        self.assertEqual(call.arguments, "{}")

        with self.assertRaisesRegex(ValueError, "id must be a non-empty"):
            TextGenerationVendor.non_stream_tool_call(
                call_id="",
                provider_name="lookup",
                arguments={},
                capability=None,
                provider_family="openai",
                provider_event_type="chat.completion.tool_call",
            )
        with self.assertRaisesRegex(ValueError, "object or string"):
            TextGenerationVendor.non_stream_tool_call(
                call_id="call-2",
                provider_name="lookup",
                arguments=["invalid"],
                capability=None,
                provider_family="openai",
                provider_event_type="chat.completion.tool_call",
            )

    async def test_system_prompt_missing_and_template_messages(self) -> None:
        vendor = DummyVendor()
        messages = [
            Message(role=MessageRole.USER, content="str"),
            Message(
                role=MessageRole.USER,
                content=MessageContentText(type="text", text="txt"),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContentImage(
                    type="image_url", image_url={"url": "http://img"}
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContentFile(
                    type="file", file={"file_id": "file-1"}
                ),
            ),
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContentText(type="text", text="a"),
                    MessageContentImage(
                        type="image_url", image_url={"url": "http://b"}
                    ),
                    MessageContentFile(
                        type="file", file={"file_url": "http://file"}
                    ),
                ],
            ),
            Message(role=MessageRole.USER, content=123),
        ]
        self.assertIsNone(vendor._system_prompt(messages))
        tmpl = vendor._template_messages(messages)
        self.assertEqual(
            tmpl,
            [
                {"role": "user", "content": "str"},
                {"role": "user", "content": "txt"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://img"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_id": "file-1"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://b"},
                        },
                        {
                            "type": "file",
                            "file": {"file_url": "http://file"},
                        },
                    ],
                },
                {"role": "user", "content": "123"},
            ],
        )
