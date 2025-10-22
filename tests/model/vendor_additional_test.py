from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    Message,
    MessageContentImage,
    MessageContentText,
    MessageRole,
)
from avalan.model.vendor import TextGenerationVendor


class DummyVendor(TextGenerationVendor):
    async def __call__(self, *args, **kwargs):
        return await super().__call__(*args, **kwargs)


class VendorTemplateMessagesTestCase(IsolatedAsyncioTestCase):
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
                content=[
                    MessageContentText(type="text", text="a"),
                    MessageContentImage(
                        type="image_url", image_url={"url": "http://b"}
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
                        {"type": "text", "text": "a"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://b"},
                        },
                    ],
                },
                {"role": "user", "content": "123"},
            ],
        )
