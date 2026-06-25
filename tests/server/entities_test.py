from unittest import TestCase

from pydantic import ValidationError

from avalan.entities import MessageRole
from avalan.server import entities as server_entities
from avalan.server.entities import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ContentFile,
    ContentText,
    EngineRequest,
    MCPFileDescriptor,
    MCPToolRequest,
    ResponseFormatJSONSchema,
    ResponsesRequest,
)

_REMOTE_RUNTIME_POLICY_KEYS = (
    "capabilities",
    "capability",
    "command_mode",
    "container_flags",
    "env",
    "environment",
    "environment_variables",
    "envvars",
    "gid",
    "platform",
    "privileged",
    "pull_policy",
    "read_only_rootfs",
    "uid",
    "user",
    "workdir",
    "working_directory",
    "workspace",
    "workspace_root",
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

    def test_chat_request_rejects_remote_runtime_authority(self) -> None:
        base = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        invalid_requests = [
            {**base, "container": {"profiles": {}}},
            {
                **base,
                "metadata": {
                    "runtime": {
                        "container": {
                            "image": "registry.example/untrusted:latest"
                        }
                    }
                },
            },
            {**base, "model": "m?backend=container"},
            {
                **base,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "run"},
                    "backend": "container",
                },
            },
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_data": "YWJj",
                                "file": {"mounts": ["/"]},
                            }
                        ],
                    }
                ],
            },
        ]

        for payload in invalid_requests:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    ValidationError,
                    "runtime authority",
                ):
                    ChatCompletionRequest.model_validate(payload)

    def test_chat_request_rejects_runtime_policy_metadata_keys(
        self,
    ) -> None:
        base = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }

        for key in _REMOTE_RUNTIME_POLICY_KEYS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ValidationError,
                    "runtime authority",
                ):
                    ChatCompletionRequest.model_validate(
                        {
                            **base,
                            "metadata": {key: True},
                        }
                    )

    def test_chat_request_ignores_non_runtime_extras(self) -> None:
        req = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"trace_id": "request-1"},
            }
        )

        self.assertEqual(req.model, "m")
        self.assertFalse(hasattr(req, "metadata"))

    def test_chat_request_accepts_safe_container_profile_selector_shape(
        self,
    ) -> None:
        req = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "container": {"profile": "workspace-readonly"},
            }
        )

        self.assertEqual(req.model, "m")

    def test_chat_request_allows_tool_schema_mode_property(self) -> None:
        req = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "set_mode",
                            "parameters": {
                                "type": "object",
                                "$defs": {
                                    "Resources": {
                                        "type": "object",
                                        "properties": {
                                            "environment": {
                                                "title": "Environment",
                                                "type": "string",
                                            },
                                        },
                                    },
                                },
                                "patternProperties": {
                                    "^mode$": {
                                        "title": "Mode Pattern",
                                        "type": "string",
                                    },
                                },
                                "properties": {
                                    "mode": {
                                        "title": "Mode",
                                        "type": "string",
                                    }
                                },
                                "required": ["mode"],
                            },
                        },
                    }
                ],
            }
        )

        assert req.tools is not None
        self.assertIn("mode", req.tools[0].function.parameters.properties)
        self.assertIn("$defs", req.tools[0].function.parameters.model_extra)
        self.assertIn(
            "patternProperties",
            req.tools[0].function.parameters.model_extra,
        )

    def test_chat_request_rejects_metadata_schema_smuggling(self) -> None:
        with self.assertRaisesRegex(ValidationError, "runtime authority"):
            ChatCompletionRequest.model_validate(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {
                        "type": "event",
                        "properties": {"sandboxProfile": "debug"},
                    },
                }
            )

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

    def test_mcp_file_descriptor_accepts_inline_aliases(self) -> None:
        descriptor = MCPFileDescriptor.model_validate(
            {
                "data": "data:text/plain;base64,YWJj",
                "mimeType": "text/plain",
                "filename": "notes.txt",
            }
        )

        self.assertEqual(
            descriptor.as_content_file(),
            {
                "file_data": "data:text/plain;base64,YWJj",
                "mime_type": "text/plain",
                "filename": "notes.txt",
            },
        )

        base64_descriptor = MCPFileDescriptor.model_validate(
            {"base64": "YWJj"}
        )
        self.assertEqual(
            base64_descriptor.as_content_file(), {"file_data": "YWJj"}
        )
        normalized_descriptor = MCPFileDescriptor.model_validate(
            {
                "data": " data:text/plain;base64, YWJj ",
                "mimeType": " text/plain ",
                "filename": " notes.txt ",
            }
        )
        self.assertEqual(
            normalized_descriptor.as_content_file(),
            {
                "file_data": "data:text/plain;base64,YWJj",
                "mime_type": "text/plain",
                "filename": "notes.txt",
            },
        )
        normalized_base64_descriptor = MCPFileDescriptor.model_validate(
            {"base64": " YWJj "}
        )
        self.assertEqual(
            normalized_base64_descriptor.as_content_file(),
            {"file_data": "YWJj"},
        )
        filename_alias_descriptor = MCPFileDescriptor.model_validate(
            {
                "base64": "YWJj",
                "fileName": " notes.txt ",
            }
        )
        self.assertEqual(
            filename_alias_descriptor.as_content_file(),
            {"file_data": "YWJj", "filename": "notes.txt"},
        )

    def test_mcp_file_descriptor_accepts_uri_aliases(self) -> None:
        descriptor = MCPFileDescriptor.model_validate(
            {
                "uri": "mcp://resources/input",
                "mime_type": "application/pdf",
            }
        )

        self.assertEqual(
            descriptor.as_content_file(),
            {
                "file_url": "mcp://resources/input",
                "mime_type": "application/pdf",
            },
        )

        url_descriptor = MCPFileDescriptor.model_validate(
            {"url": "https://example.test/input.png"}
        )
        self.assertEqual(
            url_descriptor.as_content_file(),
            {"file_url": "https://example.test/input.png"},
        )
        normalized_url_descriptor = MCPFileDescriptor.model_validate(
            {"url": " https://example.test/input.png "}
        )
        self.assertEqual(
            normalized_url_descriptor.as_content_file(),
            {"file_url": "https://example.test/input.png"},
        )

    def test_mcp_file_descriptor_rejects_invalid_sources(self) -> None:
        invalid_descriptors: list[object] = [
            {},
            {"data": "data:text/plain;base64"},
            {"data": ""},
            {"data": "not base64"},
            {"data": "YWJj="},
            {"data": "YWJj", "url": "https://example.test/file.txt"},
            {
                "data": "YWJj",
                "mimeType": "text/plain",
                "mime_type": "text/plain",
            },
            {"url": []},
            {"uri": "https://example.test/file.txt", "mimeType": ""},
            {"uri": "https://example.test/file.txt", "filename": ""},
            {
                "uri": "https://example.test/file.txt",
                "filename": "a.txt",
                "file_name": "b.txt",
            },
            "not-an-object",
        ]

        for descriptor in invalid_descriptors:
            with self.subTest(descriptor=descriptor):
                with self.assertRaises(ValidationError):
                    MCPFileDescriptor.model_validate(descriptor)

        with self.assertRaises(ValueError):
            server_entities._validate_base64_file_source(
                "data:text/plain;base64"
            )
        real_b64decode = server_entities.b64decode

        def broken_b64decode(*args: object, **kwargs: object) -> bytes:
            raise server_entities.BinasciiError("bad")

        try:
            server_entities.b64decode = broken_b64decode
            with self.assertRaises(ValueError):
                server_entities._validate_base64_file_source("YWJj")
        finally:
            server_entities.b64decode = real_b64decode

        self.assertEqual(
            server_entities._schema_property({}, "missing"),
            {"type": "string"},
        )

        descriptor = MCPFileDescriptor.model_construct(
            file_data="YWJj",
            file_url="mcp://resources/input",
        )
        with self.assertRaises(ValueError):
            descriptor.validate_sources()

    def test_mcp_tool_request_accepts_text_or_files(self) -> None:
        text_request = MCPToolRequest(input_string="hello")
        file_request = MCPToolRequest(files=[{"base64": "YWJj"}])
        combined_request = MCPToolRequest(
            input_string="summarize",
            files=[{"uri": "mcp://resources/input"}],
        )

        self.assertEqual(text_request.input_string, "hello")
        self.assertEqual(file_request.input_string, None)
        self.assertEqual(file_request.files[0].file_data, "YWJj")
        self.assertEqual(
            combined_request.files[0].file_url, "mcp://resources/input"
        )

    def test_mcp_tool_request_rejects_empty_input(self) -> None:
        with self.assertRaises(ValidationError):
            MCPToolRequest()
        with self.assertRaises(ValidationError):
            MCPToolRequest(input_string=" ")
        with self.assertRaises(ValidationError):
            MCPToolRequest.model_validate("invalid")
        with self.assertRaises(ValidationError):
            MCPToolRequest.model_validate(
                {
                    "input_string": "hello",
                    "files": [{"base64": "YWJj"}],
                    "input_files": [{"uri": "mcp://resources/input"}],
                }
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

    def test_engine_request_requires_uri_or_database(self) -> None:
        self.assertEqual(
            EngineRequest(uri="ai://local/model").uri,
            "ai://local/model",
        )
        self.assertEqual(EngineRequest(database="main").database, "main")
        with self.assertRaises(ValidationError):
            EngineRequest()

    def test_json_schema_response_format_accepts_chat_shape(self) -> None:
        response_format = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema={
                "name": "document",
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": {"title": "Value", "type": "string"}
                    },
                },
            },
        )

        assert response_format.json_schema is not None
        self.assertEqual(response_format.json_schema.name, "document")
        self.assertIsNone(response_format.schema_)

    def test_json_schema_response_format_accepts_responses_shape(
        self,
    ) -> None:
        response_format = ResponseFormatJSONSchema(
            type="json_schema",
            name="document",
            schema={
                "type": "object",
                "properties": {"value": {"title": "Value", "type": "string"}},
            },
            strict=False,
        )

        dumped = response_format.model_dump(by_alias=True, exclude_none=True)
        self.assertEqual(
            dumped,
            {
                "type": "json_schema",
                "name": "document",
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": {"title": "Value", "type": "string"}
                    },
                },
                "strict": False,
            },
        )

    def test_json_schema_response_format_rejects_ambiguous_shapes(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"title": "Value", "type": "string"}},
        }
        with self.assertRaises(ValidationError):
            ResponseFormatJSONSchema(type="json_schema")
        with self.assertRaises(ValidationError):
            ResponseFormatJSONSchema(
                type="json_schema",
                json_schema={"schema": schema},
                schema=schema,
            )
        with self.assertRaises(ValidationError):
            ResponseFormatJSONSchema(
                type="json_schema",
                json_schema={"schema": schema},
                strict=True,
            )
        with self.assertRaises(ValidationError):
            ResponseFormatJSONSchema(
                type="json_schema",
                name="",
                schema=schema,
            )

    def test_responses_request_accepts_text_format(self) -> None:
        req = ResponsesRequest(
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            text={
                "format": {"type": "json_object"},
                "stop": ["DONE"],
            },
        )

        assert req.text is not None
        assert req.text.format is not None
        self.assertEqual(req.text.stop, ["DONE"])
        self.assertEqual(req.text.format.type, "json_object")

    def test_responses_request_accepts_string_input(self) -> None:
        req = ResponsesRequest(
            input="summarize this",
            instructions="top-level guidance",
        )

        self.assertEqual(req.instructions, "top-level guidance")
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].role, MessageRole.USER)
        self.assertEqual(req.messages[0].content, "summarize this")

    def test_responses_request_preserves_message_list_input(self) -> None:
        message = ChatMessage(role=MessageRole.USER, content="hi")
        req = ResponsesRequest(input=[message])

        self.assertEqual(req.messages, [message])

    def test_responses_request_rejects_non_string_instructions(self) -> None:
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="hi", instructions={"raw": "prompt"})

    def test_responses_request_rejects_invalid_input_shape(self) -> None:
        with self.assertRaises(ValidationError):
            ResponsesRequest(input={"role": "user", "content": "hi"})

    def test_responses_request_rejects_ambiguous_text_aliases(self) -> None:
        message = ChatMessage(role=MessageRole.USER, content="hi")
        with self.assertRaises(ValidationError):
            ResponsesRequest(
                input=[message],
                response_format={"type": "json_object"},
                text={"format": {"type": "json_object"}},
            )
        with self.assertRaises(ValidationError):
            ResponsesRequest(
                input=[message],
                stop="END",
                text={"stop": "DONE"},
            )

    def test_responses_request_rejects_remote_runtime_authority(self) -> None:
        invalid_requests = [
            {"input": "hi", "runtime_envelope": {"profile": "unsafe"}},
            {
                "input": "hi",
                "metadata": {"resources": {"cpu_count": 128}},
            },
            {
                "model": "m?ds4_native_backend=metal",
                "input": "hi",
            },
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.test/i.png",
                                    "network": "host",
                                },
                            }
                        ],
                    }
                ]
            },
        ]

        for payload in invalid_requests:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    ValidationError,
                    "runtime authority",
                ):
                    ResponsesRequest.model_validate(payload)

    def test_responses_request_rejects_runtime_policy_extra_keys(
        self,
    ) -> None:
        for key in _REMOTE_RUNTIME_POLICY_KEYS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ValidationError,
                    "runtime authority",
                ):
                    ResponsesRequest.model_validate(
                        {
                            "input": "hi",
                            key: True,
                        }
                    )

    def test_responses_request_accepts_safe_container_profile_selector_shape(
        self,
    ) -> None:
        req = ResponsesRequest.model_validate(
            {
                "input": "hi",
                "container": {"profile": "workspace-readonly"},
            }
        )

        self.assertEqual(req.messages[0].content, "hi")

    def test_responses_request_allows_schema_mode_property(self) -> None:
        req = ResponsesRequest.model_validate(
            {
                "input": "hi",
                "response_format": {
                    "type": "json_schema",
                    "name": "mode_response",
                    "schema": {
                        "type": "object",
                        "$defs": {
                            "Resources": {
                                "type": "object",
                                "properties": {
                                    "environment": {
                                        "title": "Environment",
                                        "type": "string",
                                    },
                                },
                            },
                        },
                        "patternProperties": {
                            "^mode$": {
                                "title": "Mode Pattern",
                                "type": "string",
                            },
                        },
                        "properties": {
                            "mode": {
                                "title": "Mode",
                                "type": "string",
                            }
                        },
                        "required": ["mode"],
                    },
                },
            }
        )

        assert isinstance(req.response_format, ResponseFormatJSONSchema)
        assert req.response_format.schema_ is not None
        self.assertIn("mode", req.response_format.schema_.properties)
        self.assertIn("$defs", req.response_format.schema_.model_extra)
        self.assertIn(
            "patternProperties",
            req.response_format.schema_.model_extra,
        )

    def test_responses_request_rejects_metadata_schema_smuggling(self) -> None:
        with self.assertRaisesRegex(ValidationError, "runtime authority"):
            ResponsesRequest.model_validate(
                {
                    "input": "hi",
                    "metadata": {
                        "type": "event",
                        "properties": {"sandboxProfile": "debug"},
                    },
                }
            )
