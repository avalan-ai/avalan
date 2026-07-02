from types import SimpleNamespace
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
from avalan.skill.entities import (
    SkillSourceConfig,
    WorkspaceSkillSourceAuthority,
)
from avalan.skill.settings import TrustedSkillSettings

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

_MODEL_VISIBLE_REDACTION_SETTINGS = (
    server_entities.ServerOutputRedactionSettings(enabled=True)
)


def _model_visible_redactor(
    *,
    protocol: server_entities.ServerOutputRedactionProtocol = "openai",
    channel: server_entities.ServerOutputRedactionChannel = "answer",
) -> server_entities.ModelVisibleServerProtocolTextRedactor:
    return server_entities.ModelVisibleServerProtocolTextRedactor(
        _MODEL_VISIBLE_REDACTION_SETTINGS,
        protocol=protocol,
        channel=channel,
    )


def _sanitize_model_visible_text(
    value: str,
    *,
    protocol: server_entities.ServerOutputRedactionProtocol = "openai",
    channel: server_entities.ServerOutputRedactionChannel = "answer",
) -> str:
    return server_entities.sanitize_model_visible_server_protocol_text(
        value,
        output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
        protocol=protocol,
        channel=channel,
    )


def _sanitize_server_protocol_text(
    value: str,
    *,
    protocol: server_entities.ServerOutputRedactionProtocol = "openai",
    channel: server_entities.ServerOutputRedactionChannel | None = None,
) -> str:
    return server_entities.sanitize_server_protocol_text(
        value,
        output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
        protocol=protocol,
        channel=channel,
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

    def test_chat_request_rejects_remote_skills_authority_metadata(
        self,
    ) -> None:
        base = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }

        for key in (
            "skills",
            "skill_settings",
            "source_roots",
            "source_authority",
            "read_limits",
            "registry_mutation",
        ):
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

    def test_responses_request_rejects_remote_skills_authority_metadata(
        self,
    ) -> None:
        for payload in (
            {"model": "m", "input": "hi", "metadata": {"skills": {}}},
            {
                "model": "m",
                "input": "hi",
                "metadata": {"source_roots": ["/Users/me/skills"]},
            },
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    ValidationError,
                    "runtime authority",
                ):
                    ResponsesRequest.model_validate(payload)

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

    def test_chat_request_rejects_remote_skills_tool_definition(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "Remote requests cannot define skills tools",
        ):
            ChatCompletionRequest.model_validate(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "skills.read",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                        }
                    ],
                }
            )

    def test_request_skills_tool_definition_validator_ignores_malformed_inputs(
        self,
    ) -> None:
        server_entities._reject_remote_skills_tool_definitions(
            {"function": {"name": "skills.read"}},
            path="request.tools",
        )
        server_entities._reject_remote_skills_tool_definitions(
            [object(), {"type": "function"}, {"function": "invalid"}],
            path="request.tools",
        )

    def test_chat_request_allows_skills_schema_property_name(self) -> None:
        req = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "select_skills",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "skills": {
                                        "title": "Skills",
                                        "type": "string",
                                    }
                                },
                            },
                        },
                    }
                ],
            }
        )

        assert req.tools is not None
        self.assertIn("skills", req.tools[0].function.parameters.properties)

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

    def test_server_protocol_sanitizer_redacts_binary_values(self) -> None:
        sanitized = server_entities.sanitize_server_protocol_value(
            {
                "data": b"secret",
                "nested": [bytearray(b"hidden"), memoryview(b"private")],
            }
        )

        self.assertEqual(
            sanitized,
            {
                "data": "<redacted-bytes>",
                "nested": ["<redacted-bytes>", "<redacted-bytes>"],
            },
        )

    def test_server_protocol_sanitizer_defaults_to_redaction_off(
        self,
    ) -> None:
        text = "See /Users/mariano/skills/demo/SKILL.md"
        value = {
            "content": "secret skill content",
            "path": text,
            "data": b"secret",
        }

        self.assertEqual(
            server_entities.sanitize_server_protocol_text(text),
            text,
        )
        self.assertEqual(
            server_entities.sanitize_server_protocol_value(
                value,
                tool_name="skills.demo",
                protocol="mcp",
            ),
            {
                "content": "secret skill content",
                "path": text,
                "data": "<redacted-bytes>",
            },
        )

    def test_server_protocol_sanitizer_redacts_skills_content_by_rule(
        self,
    ) -> None:
        settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"skills_tool_content"}),
            protocols=frozenset({"mcp"}),
        )
        value = {
            "content": "secret skill content at /Users/mariano/demo.txt",
            "path": "/Users/mariano/demo.txt",
        }

        self.assertEqual(
            server_entities.sanitize_server_protocol_value(
                value,
                tool_name="skills.demo",
                output_redaction_settings=settings,
                protocol="mcp",
            ),
            {
                "content": {
                    "redacted": True,
                    "reason": server_entities.SKILL_CONTENT_REDACTION,
                },
                "path": "/Users/mariano/demo.txt",
            },
        )
        self.assertEqual(
            server_entities.sanitize_server_protocol_value(
                value,
                tool_name="skills.demo",
                output_redaction_settings=settings,
                protocol="openai",
            ),
            value,
        )

    def test_server_protocol_sanitizer_redacts_host_paths_by_rule(
        self,
    ) -> None:
        settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"host_paths"}),
        )
        value = {
            "content": "secret skill content",
            "path": "/Users/mariano/demo.txt",
        }

        self.assertEqual(
            server_entities.sanitize_server_protocol_value(
                value,
                tool_name="skills.demo",
                output_redaction_settings=settings,
                protocol="mcp",
            ),
            {
                "content": "secret skill content",
                "path": "<host-path>/demo.txt",
            },
        )

    def test_unchanneled_protocol_sanitizer_respects_channel_filters(
        self,
    ) -> None:
        settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"host_paths", "skills_tool_content"}),
            channels=frozenset({"reasoning"}),
        )
        text = "failed at /Users/mariano/private.txt"
        value = {
            "content": "private skill content",
            "path": text,
        }

        self.assertEqual(
            server_entities.sanitize_server_protocol_text(
                text,
                output_redaction_settings=settings,
                protocol="mcp",
            ),
            text,
        )
        self.assertEqual(
            server_entities.sanitize_server_protocol_value(
                value,
                tool_name="skills.demo",
                output_redaction_settings=settings,
                protocol="mcp",
            ),
            value,
        )
        self.assertEqual(
            server_entities.sanitize_server_protocol_text(
                text,
                output_redaction_settings=settings,
                protocol="mcp",
                channel="reasoning",
            ),
            "failed at <host-path>/private.txt",
        )

    def test_server_output_redaction_settings_from_state_uses_ctx_fallback(
        self,
    ) -> None:
        ctx_settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            protocols=frozenset({"a2a"}),
        )
        state = SimpleNamespace(
            ctx=server_entities.OrchestratorContext(
                participant_id=None,
                output_redaction_settings=ctx_settings,
            )
        )

        self.assertIs(
            server_entities.server_output_redaction_settings_from_state(state),
            ctx_settings,
        )

        app_state_settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            protocols=frozenset({"mcp"}),
        )
        state.server_output_redaction_settings = app_state_settings
        self.assertIs(
            server_entities.server_output_redaction_settings_from_state(state),
            ctx_settings,
        )

        fallback_state = SimpleNamespace(
            server_output_redaction_settings=app_state_settings
        )
        self.assertIs(
            server_entities.server_output_redaction_settings_from_state(
                fallback_state
            ),
            app_state_settings,
        )

    def test_server_skills_registry_metadata_handles_settings(self) -> None:
        self.assertIsNone(
            server_entities.server_skills_registry_metadata(None)
        )

        skills_settings = TrustedSkillSettings(
            sources=(
                SkillSourceConfig(
                    label="workspace",
                    authority=WorkspaceSkillSourceAuthority(),
                    root_path="/Users/mariano/.codex/skills",
                ),
            )
        )
        context = server_entities.OrchestratorContext(
            participant_id=None,
            skills_settings=skills_settings,
            skills_registry_metadata={"enabled": True},
        )

        self.assertIs(context.skills_settings, skills_settings)
        self.assertEqual(context.skills_registry_metadata, {"enabled": True})

        metadata = server_entities.server_skills_registry_metadata(
            skills_settings
        )

        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["source_labels"], ("workspace",))
        self.assertIn("workspace", metadata["authority_kinds"])

    def test_model_visible_redaction_defaults_to_off(self) -> None:
        text = (
            "# Demo Skill\n\n"
            "Use when handling private operator tasks.\n"
            "Secret body at /Users/mariano/skills/demo/SKILL.md"
        )
        redactor = server_entities.ModelVisibleServerProtocolTextRedactor()

        self.assertEqual(
            server_entities.sanitize_model_visible_server_protocol_text(text),
            text,
        )
        self.assertEqual(redactor.push(text), (text,))
        self.assertEqual(redactor.flush(), ())

    def test_server_output_redaction_settings_apply_granularity(
        self,
    ) -> None:
        settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"host_paths"}),
            protocols=frozenset({"mcp"}),
            channels=frozenset({"reasoning"}),
        )
        text = "See /Users/mariano/private.txt"

        self.assertEqual(
            server_entities.sanitize_model_visible_server_protocol_text(
                text,
                output_redaction_settings=settings,
                protocol="mcp",
                channel="reasoning",
            ),
            "See <host-path>/private.txt",
        )
        self.assertEqual(
            server_entities.sanitize_model_visible_server_protocol_text(
                text,
                output_redaction_settings=settings,
                protocol="mcp",
                channel="answer",
            ),
            text,
        )
        self.assertEqual(
            server_entities.sanitize_model_visible_server_protocol_text(
                text,
                output_redaction_settings=settings,
                protocol="openai",
                channel="reasoning",
            ),
            text,
        )

    def test_model_visible_redactor_can_only_redact_skill_source_paths(
        self,
    ) -> None:
        settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"skill_source_paths"}),
        )
        redactor = server_entities.ModelVisibleServerProtocolTextRedactor(
            settings
        )

        self.assertEqual(redactor.push("# Demo Skill\n\n"), ())
        self.assertEqual(
            redactor.push("Source: /skills/demo/SKILL.md"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )
        self.assertEqual(redactor.push("later"), ())

        body_only = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"skill_body_echoes"}),
        )
        source_text = "# Demo Skill\n\nSource: /skills/demo/SKILL.md"
        self.assertEqual(
            server_entities.sanitize_model_visible_server_protocol_text(
                source_text,
                output_redaction_settings=body_only,
            ),
            source_text,
        )

    def test_model_visible_redactor_handles_empty_and_full_skill_body(
        self,
    ) -> None:
        redactor = _model_visible_redactor()

        self.assertEqual(redactor.push(""), ())
        self.assertEqual(redactor.push("   \n"), ("   \n",))
        self.assertEqual(
            redactor.push("# Demo Skill\n\nUse when private.\n\nSecret body"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )
        self.assertEqual(redactor.push("later token"), ())

        embedded = _model_visible_redactor()
        self.assertEqual(
            embedded.push(
                "Preamble\n# Demo Skill\n\n"
                "Use when private.\n\n"
                "Secret body that is long enough to look like an echo."
            ),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )
        self.assertEqual(embedded.push("later token"), ())

    def test_model_visible_redactor_redacts_skill_source_path_references(
        self,
    ) -> None:
        skill_bodies = (
            "# Demo Skill\n\nSource: ~/.codex/skills/demo/README.md\n",
            "# Demo Skill\n\nSource: /tmp/demo/SKILL.md\n",
            "# Demo Skill\n\nSource: /skills/demo/README.md\n",
            "# Demo Skill\n\nSource: /skills\n",
            "# Demo Skill\n\nSource: \\skills\\demo\\README.md\n",
            "# Demo Skill\n\nSource: \\skills\n",
            "# Demo Skill\n\nSource: C:/skills/demo/README.md\n",
            "# Demo Skill\n\nSource: C:/skills\n",
            "# Demo Skill\n\nSource: C:\\skills\n",
        )

        for text in skill_bodies:
            with self.subTest(text=text):
                self.assertEqual(
                    _sanitize_model_visible_text(text),
                    server_entities.SKILL_CONTENT_REDACTION,
                )

    def test_model_visible_stream_redactor_allows_markdown_starts(
        self,
    ) -> None:
        heading = _model_visible_redactor()

        self.assertEqual(
            heading.push("# Summary\nThis is ordinary markdown.\n"),
            ("# Summary\nThis is ordinary markdown.\n",),
        )
        self.assertEqual(heading.push("Next delta."), ("Next delta.",))

        split = _model_visible_redactor()
        self.assertEqual(split.push("# Summary\n"), ("# Summary\n",))
        self.assertEqual(
            split.push("This is ordinary markdown.\n"),
            ("This is ordinary markdown.\n",),
        )
        self.assertEqual(split.push("Next delta."), ("Next delta.",))

        frontmatter = _model_visible_redactor()
        self.assertEqual(
            frontmatter.push("---\ntitle: Report\n---\nBody.\n"),
            ("---\ntitle: Report\n---\nBody.\n",),
        )
        self.assertEqual(frontmatter.push("Next delta."), ("Next delta.",))

        instructions = _model_visible_redactor()
        self.assertEqual(
            instructions.push("# Summary\nInstructions for deployment.\n"),
            ("# Summary\nInstructions for deployment.\n",),
        )
        self.assertEqual(instructions.push("Next delta."), ("Next delta.",))

        release_notes = (
            "# Release Notes\n\n"
            "Use when evaluating rollout options for the July train.\n"
        )
        self.assertEqual(
            _sanitize_model_visible_text(release_notes),
            release_notes,
        )
        api_guide = "# API Guide\n\nSee docs/skills/demo for details.\n"
        self.assertEqual(
            _sanitize_model_visible_text(api_guide),
            api_guide,
        )
        remote_skill_resource = (
            "# API Guide\n\n"
            "See https://docs.example.com/skills/demo/SKILL.md\n"
        )
        self.assertEqual(
            _sanitize_model_visible_text(remote_skill_resource),
            remote_skill_resource,
        )
        relative_skill_resource = "description: docs/skills/demo/SKILL.md"
        self.assertEqual(
            _sanitize_model_visible_text(relative_skill_resource),
            relative_skill_resource,
        )
        skills_matrix = (
            "# Skills Matrix\n\n"
            "Use when assigning people to delivery rotations.\n"
        )
        self.assertEqual(
            _sanitize_model_visible_text(skills_matrix),
            skills_matrix,
        )

        for skill_body in (
            "# imagegen\n\nUse when creating bitmap assets.\n",
            "# Presentations\n\nUse when building slide decks.\n",
            "# Imagegen\n\nUse when creating bitmap assets.\n",
            "# PDF\n\nUse when extracting document layouts.\n",
            "# PDF Basic\n\nUse when handling simple PDF tasks.\n",
            (
                "# browser:control-in-app-browser\n\n"
                "Use when controlling browser.\n"
            ),
        ):
            with self.subTest(skill_body=skill_body):
                self.assertEqual(
                    _sanitize_model_visible_text(skill_body),
                    server_entities.SKILL_CONTENT_REDACTION,
                )

        for ordinary_markdown in (
            "# ``\n\nUse when writing placeholder sections during drafting.\n",
            (
                "# Quarterly Planning Notes\n\n"
                "Use when evaluating staffing and release tradeoffs.\n"
            ),
            "# Summary\n\nnormal intro\nUse when reviewing plans.\n",
            "# Summary\n\nnormal intro\nSource: /skills/demo/SKILL.md\n",
            (
                "Preamble\n# Summary\n\nnormal intro\n"
                "Source: /skills/demo/SKILL.md\n"
            ),
        ):
            with self.subTest(ordinary_markdown=ordinary_markdown):
                self.assertEqual(
                    _sanitize_model_visible_text(ordinary_markdown),
                    ordinary_markdown,
                )

    def test_model_visible_stream_redactor_releases_safe_heading_prefixes(
        self,
    ) -> None:
        redactor = _model_visible_redactor()

        self.assertEqual(redactor.push("# Summary\n"), ("# Summary\n",))
        self.assertEqual(redactor.push("Instruction"), ("Instruction",))
        self.assertEqual(
            redactor.push("al note for deployment.\n"),
            ("al note for deployment.\n",),
        )
        self.assertEqual(redactor.push("Next delta."), ("Next delta.",))

        ordinary_use_when = _model_visible_redactor()
        ordinary_use_when_start = ordinary_use_when.push("# Summary\n\n")
        self.assertEqual(
            ordinary_use_when_start,
            ("# Summary\n\n",),
        )
        use_when_streamed = (
            *ordinary_use_when_start,
            *ordinary_use_when.push("normal intro\n"),
            *ordinary_use_when.push("Use when reviewing plans.\n"),
            *ordinary_use_when.flush(),
        )
        use_when_text = (
            "# Summary\n\nnormal intro\nUse when reviewing plans.\n"
        )
        self.assertEqual(
            "".join(use_when_streamed),
            _sanitize_model_visible_text(use_when_text),
        )

        ordinary_source = _model_visible_redactor()
        ordinary_source_start = ordinary_source.push("# Summary\n\n")
        self.assertEqual(
            ordinary_source_start,
            ("# Summary\n\n",),
        )
        source_streamed = (
            *ordinary_source_start,
            *ordinary_source.push("normal intro\n"),
            *ordinary_source.push("Source: /skills/demo/SKILL.md\n"),
            *ordinary_source.flush(),
        )
        source_text = (
            "# Summary\n\nnormal intro\nSource: /skills/demo/SKILL.md\n"
        )
        self.assertEqual(
            "".join(source_streamed),
            _sanitize_model_visible_text(source_text),
        )

        long_marker_text = "# Summary\n\nUse when reviewing plans.\n" + (
            "x" * 12001
        )
        long_marker = _model_visible_redactor()
        long_marker_streamed = (
            *long_marker.push("# Summary\n\n"),
            *long_marker.push("Use when reviewing plans.\n" + ("x" * 12001)),
            *long_marker.flush(),
        )
        self.assertEqual(
            "".join(long_marker_streamed),
            _sanitize_model_visible_text(long_marker_text),
        )

        preamble_source = _model_visible_redactor()
        preamble_source_start = preamble_source.push("Preamble\n# Summary\n\n")
        self.assertEqual(
            preamble_source_start,
            ("Preamble\n# Summary\n\n",),
        )
        preamble_source_streamed = (
            *preamble_source_start,
            *preamble_source.push("normal intro\n"),
            *preamble_source.push("Source: /skills/demo/SKILL.md\n"),
            *preamble_source.flush(),
        )
        preamble_source_text = (
            "Preamble\n# Summary\n\nnormal intro\n"
            "Source: /skills/demo/SKILL.md\n"
        )
        self.assertEqual(
            "".join(preamble_source_streamed),
            _sanitize_model_visible_text(preamble_source_text),
        )

        heading_only = _model_visible_redactor()
        self.assertEqual(heading_only.push("# Summary"), ())
        self.assertEqual(heading_only.flush(), ("# Summary",))

        description = _model_visible_redactor()
        self.assertEqual(description.push("description:"), ())
        self.assertEqual(description.push(" Source"), ())
        self.assertEqual(
            description.push(" note for deployment.\n"),
            ("description: Source note for deployment.\n",),
        )

        private_source = _model_visible_redactor()
        self.assertEqual(private_source.push("description: ~/.c"), ())
        self.assertEqual(private_source.push("odex"), ())
        self.assertEqual(
            private_source.push("/skills/demo/README.md"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        for chunks in (
            (
                "description: ",
                "~/",
                ".codex/skills/demo/README.md",
            ),
            ("description: ", "/", "skills/demo/README.md"),
            ("description: ", "C:/", "skills/demo/README.md"),
            ("description: ", "C:\\", "skills\\demo\\README.md"),
        ):
            with self.subTest(chunks=chunks):
                split_source = _model_visible_redactor()
                self.assertEqual(split_source.push(chunks[0]), ())
                self.assertEqual(split_source.push(chunks[1]), ())
                self.assertEqual(
                    split_source.push(chunks[2]),
                    (server_entities.SKILL_CONTENT_REDACTION,),
                )

        for prefix in ("description: ~", "description: C:"):
            with self.subTest(prefix=prefix):
                split_prefix = _model_visible_redactor()
                self.assertEqual(split_prefix.push(prefix), ())
                self.assertTrue(split_prefix.has_pending)

    def test_model_visible_stream_redactor_redacts_split_host_paths(
        self,
    ) -> None:
        normal = _model_visible_redactor()
        self.assertEqual(normal.push("a"), ("a",))
        self.assertEqual(normal.push("b"), ("b",))

        redactor = _model_visible_redactor()

        self.assertEqual(redactor.push("See /Users/mar"), ("See ",))
        self.assertEqual(
            redactor.push("iano/secret.txt and continue."),
            ("<host-path>/secret.txt and continue.",),
        )
        self.assertEqual(redactor.push(" Next delta."), (" Next delta.",))

        windows = _model_visible_redactor()
        self.assertEqual(windows.push("Open C:/Users/mar"), ("Open ",))
        self.assertEqual(
            windows.push("iano/secret.txt now."),
            ("<host-path>/secret.txt now.",),
        )

        drive = _model_visible_redactor()
        self.assertEqual(drive.push("Open C"), ("Open ",))
        self.assertEqual(
            drive.push(":/Users/mariano/secret.txt now."),
            ("<host-path>/secret.txt now.",),
        )

        partial_drive = _model_visible_redactor()
        self.assertEqual(partial_drive.push("Open C:"), ("Open ",))
        self.assertEqual(partial_drive.flush(), ("C:",))

        invalid_windows = _model_visible_redactor()
        self.assertEqual(
            invalid_windows.push("Open C:Users"),
            ("Open C:Users",),
        )

        file_url = _model_visible_redactor()
        self.assertEqual(
            file_url.push("Open file:///tmp/re"), ("Open file://",)
        )
        self.assertEqual(file_url.push("port.txt"), ())
        self.assertEqual(file_url.flush(), ("<host-path>/report.txt",))

        remote = _model_visible_redactor()
        self.assertEqual(
            remote.push("See https://files.example/tmp/re"),
            ("See https://files.example/tmp/re",),
        )
        self.assertEqual(remote.push("port.pdf"), ("port.pdf",))

    def test_model_visible_stream_redactor_flushes_pending_safe_text(
        self,
    ) -> None:
        redactor = _model_visible_redactor()

        self.assertEqual(redactor.push("---\n"), ())
        self.assertEqual(redactor.flush(), ("---\n",))
        self.assertEqual(redactor.push("Next delta."), ("Next delta.",))

        heading_redactor = _model_visible_redactor()
        self.assertEqual(heading_redactor.push("# Imagegen\n"), ())
        self.assertEqual(
            heading_redactor.push("Source: /tmp/report.txt"),
            ("# Imagegen\nSource: ",),
        )
        self.assertEqual(
            heading_redactor.flush(),
            ("<host-path>/report.txt",),
        )

        path_redactor = _model_visible_redactor()
        self.assertEqual(path_redactor.push("# Summary\n"), ("# Summary\n",))
        self.assertEqual(
            path_redactor.push("See /tmp/report.txt\n"),
            ("See <host-path>/report.txt\n",),
        )

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

    def test_responses_request_rejects_remote_skills_tool_definition(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "Remote requests cannot define skills tools",
        ):
            ResponsesRequest.model_validate(
                {
                    "input": "hi",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "skills.read",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                        }
                    ],
                }
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

    def test_server_protocol_text_redacts_representative_host_paths(
        self,
    ) -> None:
        cases = {
            "see /tmp/skills/demo/SKILL.md": "see <host-path>/SKILL.md",
            "see /opt/avalan/logs/run.log": "see <host-path>/run.log",
            "see /Volumes/Data/skills/demo/SKILL.md": (
                "see <host-path>/SKILL.md"
            ),
            r"see C:\Users\mariano\skills\demo\SKILL.md": (
                "see <host-path>/SKILL.md"
            ),
            "see C:/Users/mariano/skills/demo/SKILL.md": (
                "see <host-path>/SKILL.md"
            ),
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    _sanitize_server_protocol_text(source),
                    expected,
                )

    def test_server_protocol_text_handles_repeated_slash_segments(
        self,
    ) -> None:
        posix_path = "/var/" + "!/" * 128 + "SKILL.md"
        windows_path = "A:/Users/" + "!/" * 128 + "SKILL.md"

        self.assertEqual(
            _sanitize_server_protocol_text(f"read {posix_path}"),
            "read <host-path>/SKILL.md",
        )
        self.assertEqual(
            _sanitize_server_protocol_text(f"read {windows_path}"),
            "read <host-path>/SKILL.md",
        )

    def test_server_protocol_text_redacts_consecutive_slash_host_paths(
        self,
    ) -> None:
        text = (
            "read /tmp//skills/demo/SKILL.md "
            "and C:/Users//skills/demo/SKILL.md "
            "and file:///tmp//skills/demo/SKILL.md "
            "but keep https://files.example/tmp//report.pdf"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            "read <host-path>/SKILL.md "
            "and <host-path>/SKILL.md "
            "and file://<host-path>/SKILL.md "
            "but keep https://files.example/tmp//report.pdf",
        )

    def test_server_protocol_text_redacts_trailing_slash_host_paths(
        self,
    ) -> None:
        text = (
            "read /tmp/skills/demo/ "
            "and /tmp//skills/demo/ "
            "and C:/Users/me/skills/demo/ "
            "and file:///tmp//skills/demo/ "
            "but keep https://files.example/tmp//skills/demo/"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            "read <host-path>/demo "
            "and <host-path>/demo "
            "and <host-path>/demo "
            "and file://<host-path>/demo "
            "but keep https://files.example/tmp//skills/demo/",
        )

    def test_server_protocol_text_avoids_path_prefix_false_positives(
        self,
    ) -> None:
        text = (
            "read /tmpfile and /optimize and /VolumesData "
            r"and C:Users\mariano and C:/UsersData/file"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            text,
        )

    def test_server_protocol_text_does_not_redact_remote_url_paths(
        self,
    ) -> None:
        text = (
            "download https://files.example/tmp/report.pdf "
            "and https://files.example/opt/logs/run.log "
            "and https://files.example/Volumes/Data/report.pdf "
            "but read /tmp/local-report.pdf "
            "and file:///tmp/skills/demo/SKILL.md"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            "download https://files.example/tmp/report.pdf "
            "and https://files.example/opt/logs/run.log "
            "and https://files.example/Volumes/Data/report.pdf "
            "but read <host-path>/local-report.pdf "
            "and file://<host-path>/SKILL.md",
        )

    def test_server_protocol_text_preserves_windows_paths_in_remote_urls(
        self,
    ) -> None:
        text = (
            "keep https://files.example/C:/Users "
            "and https://files.example/C:/Users/me/file.txt "
            "but redact C:/Users and C:/Users/me "
            "and file://C:/Users/me/file.txt "
            "and file:///C:/Users/me/file.txt"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            "keep https://files.example/C:/Users "
            "and https://files.example/C:/Users/me/file.txt "
            "but redact <host-path>/Users and <host-path>/me "
            "and file://<host-path>/file.txt "
            "and file:///<host-path>/file.txt",
        )

    def test_server_protocol_text_redacts_wrapped_file_urls(
        self,
    ) -> None:
        text = (
            "read (file:///tmp/secret.txt) "
            "and [file:///tmp/secret.txt] "
            'and "file:///tmp/secret.txt" '
            "and (file://C:/Users/me/secret.txt)"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            "read (file://<host-path>/secret.txt) "
            "and [file://<host-path>/secret.txt] "
            'and "file://<host-path>/secret.txt" '
            "and (file://<host-path>/secret.txt)",
        )

    def test_server_protocol_text_preserves_wrapped_remote_urls(
        self,
    ) -> None:
        text = (
            "keep (https://files.example/tmp/report.pdf) "
            "and [https://files.example/tmp/report.pdf] "
            'and "https://files.example/tmp/report.pdf"'
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            text,
        )

    def test_server_protocol_text_uses_last_scheme_for_attached_labels(
        self,
    ) -> None:
        text = (
            "Source:file:///tmp/skills/demo/SKILL.md "
            "url:file:///tmp/secret.txt "
            "prefix:file://C:/Users/me/secret.txt "
            "Source:https://files.example/tmp/report.pdf "
            "url:https://files.example/C:/Users/me/file.txt"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            "Source:file://<host-path>/SKILL.md "
            "url:file://<host-path>/secret.txt "
            "prefix:file://<host-path>/secret.txt "
            "Source:https://files.example/tmp/report.pdf "
            "url:https://files.example/C:/Users/me/file.txt",
        )

    def test_server_protocol_text_redacts_colon_labeled_local_paths(
        self,
    ) -> None:
        text = (
            "Source:/tmp/skills/demo/SKILL.md "
            "url:/opt/secret.txt "
            "Source:C:/Users/me/secret.txt "
            "Source:https://files.example/tmp/report.pdf "
            "url:https://files.example/C:/Users/me/file.txt"
        )

        self.assertEqual(
            _sanitize_server_protocol_text(text),
            "Source:<host-path>/SKILL.md "
            "url:<host-path>/secret.txt "
            "Source:<host-path>/secret.txt "
            "Source:https://files.example/tmp/report.pdf "
            "url:https://files.example/C:/Users/me/file.txt",
        )

    def test_model_visible_protocol_text_redacts_echoed_skill_body(
        self,
    ) -> None:
        text = (
            "# Demo Skill\n\n"
            "Use when handling private operator tasks.\n\n"
            "Secret skill body: never expose this instruction.\n"
            "Source: C:/Users/mariano/skills/demo/SKILL.md"
        )

        sanitized = _sanitize_model_visible_text(text)

        self.assertEqual(sanitized, server_entities.SKILL_CONTENT_REDACTION)
        self.assertNotIn("Secret skill body", sanitized)
        self.assertNotIn("C:/Users", sanitized)

    def test_model_visible_stream_redactor_holds_split_heading_marker(
        self,
    ) -> None:
        redactor = _model_visible_redactor()

        self.assertEqual(redactor.push("#"), ())
        self.assertEqual(redactor.push(" Demo Skill\n\n"), ())
        self.assertEqual(
            redactor.push(
                "Use when handling private operator tasks.\n"
                "Secret skill body.\n"
                "Source: /tmp/skills/demo/SKILL.md"
            ),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )
        self.assertEqual(redactor.flush(), ())

    def test_model_visible_stream_redactor_holds_skill_heading_candidates(
        self,
    ) -> None:
        body_settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"skill_body_echoes"}),
        )
        body_redactor = server_entities.ModelVisibleServerProtocolTextRedactor(
            body_settings
        )

        self.assertEqual(body_redactor.push("# Demo Skill\n\n"), ())
        self.assertEqual(body_redactor.push("normal intro\n"), ())
        self.assertEqual(
            body_redactor.push("Use when handling private tasks.\nSecret"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        source_settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"skill_source_paths"}),
        )
        source_redactor = (
            server_entities.ModelVisibleServerProtocolTextRedactor(
                source_settings
            )
        )

        self.assertEqual(source_redactor.push("# Demo Skill\n\n"), ())
        self.assertEqual(source_redactor.push("normal intro\n"), ())
        self.assertEqual(
            source_redactor.push("Source: /skills/demo/SKILL.md"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        default_redactor = (
            server_entities.ModelVisibleServerProtocolTextRedactor()
        )
        self.assertEqual(
            default_redactor.push("# Demo Skill\n\n"),
            ("# Demo Skill\n\n",),
        )
        self.assertEqual(
            default_redactor.push("normal intro\n"),
            ("normal intro\n",),
        )
        self.assertEqual(
            default_redactor.push("Use when handling private tasks.\nSecret"),
            ("Use when handling private tasks.\nSecret",),
        )

    def test_model_visible_stream_redactor_holds_title_case_skill_heading(
        self,
    ) -> None:
        body_redactor = _model_visible_redactor()

        self.assertEqual(body_redactor.push("# Imagegen\n\n"), ())
        self.assertEqual(body_redactor.push("normal intro\n"), ())
        self.assertEqual(
            body_redactor.push("Use when creating private bitmap assets."),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        source_redactor = _model_visible_redactor()
        self.assertEqual(source_redactor.push("# Imagegen\n\n"), ())
        self.assertEqual(source_redactor.push("normal intro\n"), ())
        self.assertEqual(
            source_redactor.push("Source: /skills/imagegen/SKILL.md"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

    def test_model_visible_stream_redactor_buffers_body_window_after_heading(
        self,
    ) -> None:
        redactor = _model_visible_redactor()
        near_boundary_body = "x" * 11990

        self.assertEqual(redactor.push("# Imagegen\n\n"), ())
        self.assertEqual(redactor.push(near_boundary_body), ())
        self.assertEqual(
            redactor.push("\nUse when creating private bitmap assets."),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        split_marker_body = "x" * 11997
        split_marker_text = (
            f"# Imagegen\n\n{split_marker_body}\nUse when private.\nSecret"
        )
        self.assertEqual(
            _sanitize_model_visible_text(split_marker_text),
            server_entities.SKILL_CONTENT_REDACTION,
        )
        split_marker = _model_visible_redactor()
        self.assertEqual(split_marker.push("# Imagegen\n\n"), ())
        self.assertEqual(split_marker.push(split_marker_body), ())
        self.assertEqual(split_marker.push("\nUse"), ())
        self.assertEqual(
            split_marker.push(" when private.\nSecret"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        embedded = _model_visible_redactor()
        self.assertEqual(embedded.push("Preamble\n# Imagegen\n\n"), ())
        self.assertEqual(embedded.push(split_marker_body), ())
        self.assertEqual(embedded.push("\nUse"), ())
        self.assertEqual(
            embedded.push(" when private.\nSecret"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        beyond_window_body = "x" * 12001
        beyond_window = _model_visible_redactor()
        self.assertEqual(beyond_window.push("# Imagegen\n\n"), ())
        self.assertEqual(
            beyond_window.push(beyond_window_body),
            ("# Imagegen\n\n" + beyond_window_body,),
        )
        self.assertEqual(beyond_window.push("\nUse"), ("\nUse",))

    def test_model_visible_stream_buffer_public_edges(self) -> None:
        heading_only = _model_visible_redactor()
        self.assertEqual(heading_only.push("# Imagegen"), ())
        self.assertEqual(heading_only.flush(), ("# Imagegen",))

        marker_prefix = _model_visible_redactor()
        self.assertEqual(marker_prefix.push("# Imagegen\n\n"), ())
        self.assertEqual(marker_prefix.push("Use"), ())
        self.assertEqual(marker_prefix.flush(), ("# Imagegen\n\nUse",))

        source_prefix = _model_visible_redactor()
        self.assertEqual(source_prefix.push("# Imagegen\n\n"), ())
        self.assertEqual(source_prefix.push("/sk"), ())
        self.assertEqual(source_prefix.flush(), ("# Imagegen\n\n/sk",))

        frontmatter_source = "---\nSource: /skills/demo/SKILL.md"
        self.assertEqual(
            _sanitize_model_visible_text(frontmatter_source),
            server_entities.SKILL_CONTENT_REDACTION,
        )

        body_only_settings = server_entities.ServerOutputRedactionSettings(
            enabled=True,
            rules=frozenset({"skill_body_echoes"}),
        )
        heading_only_body_redactor = (
            server_entities.ModelVisibleServerProtocolTextRedactor(
                body_only_settings
            )
        )
        self.assertEqual(heading_only_body_redactor.push("# Imagegen"), ())
        self.assertEqual(
            heading_only_body_redactor.flush(),
            ("# Imagegen",),
        )

        source_prefix_body_redactor = (
            server_entities.ModelVisibleServerProtocolTextRedactor(
                body_only_settings
            )
        )
        self.assertEqual(
            source_prefix_body_redactor.push("# Imagegen\n\n"),
            (),
        )
        self.assertEqual(source_prefix_body_redactor.push("/sk"), ())
        self.assertEqual(
            source_prefix_body_redactor.flush(),
            ("# Imagegen\n\n/sk",),
        )

        beyond_window_body_redactor = (
            server_entities.ModelVisibleServerProtocolTextRedactor(
                body_only_settings
            )
        )
        beyond_window_text = "# Imagegen\n\n" + ("x" * 12001) + "Use"
        self.assertEqual(
            beyond_window_body_redactor.push(beyond_window_text),
            (beyond_window_text,),
        )

    def test_model_visible_source_path_window_matches_streaming(
        self,
    ) -> None:
        near_boundary_body = "x" * 11950
        near_boundary_text = (
            "# Imagegen\n\n"
            f"{near_boundary_body}\n"
            "Source: /skills/demo/SKILL.md"
        )
        self.assertEqual(
            _sanitize_model_visible_text(near_boundary_text),
            server_entities.SKILL_CONTENT_REDACTION,
        )
        near_boundary = _model_visible_redactor()
        self.assertEqual(near_boundary.push("# Imagegen\n\n"), ())
        self.assertEqual(near_boundary.push(near_boundary_body), ())
        self.assertEqual(
            near_boundary.push("\nSource: /skills/demo/SKILL.md"),
            (server_entities.SKILL_CONTENT_REDACTION,),
        )

        over_window_body = "x" * 12001
        over_window_text = (
            f"# Imagegen\n\n{over_window_body}\nSource: /skills/demo/SKILL.md"
        )
        self.assertEqual(
            _sanitize_model_visible_text(over_window_text),
            over_window_text,
        )
        over_window = _model_visible_redactor()
        self.assertEqual(over_window.push("# Imagegen\n\n"), ())
        over_window_streamed = (
            *over_window.push(over_window_body),
            *over_window.push("\nSource: /skills/demo/SKILL.md"),
            *over_window.flush(),
        )
        self.assertEqual("".join(over_window_streamed), over_window_text)
