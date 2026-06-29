from unittest import TestCase

from avalan.server.authority import (
    reject_remote_runtime_authority_extra_fields,
    reject_remote_runtime_authority_fields,
    reject_remote_runtime_authority_model_identifier,
    remote_runtime_authority_key,
)


class RemoteRuntimeAuthorityTestCase(TestCase):
    def test_detects_runtime_authority_key_variants(self) -> None:
        self.assertFalse(remote_runtime_authority_key(""))
        self.assertFalse(remote_runtime_authority_key("metadata"))
        self.assertFalse(remote_runtime_authority_key("shellfish"))
        self.assertFalse(remote_runtime_authority_key("shelley"))
        self.assertTrue(remote_runtime_authority_key("runtimeProfile"))
        self.assertTrue(remote_runtime_authority_key("isolation"))
        self.assertTrue(remote_runtime_authority_key("isolation_policy"))
        self.assertTrue(remote_runtime_authority_key("sandboxProfile"))
        self.assertTrue(remote_runtime_authority_key("sandbox_roots"))
        self.assertTrue(remote_runtime_authority_key("allow_pipelines"))
        self.assertTrue(remote_runtime_authority_key("allowShell"))
        self.assertTrue(remote_runtime_authority_key("shell"))
        self.assertTrue(remote_runtime_authority_key("shellRuntime"))
        self.assertTrue(remote_runtime_authority_key("tool_shell_backend"))
        self.assertTrue(
            remote_runtime_authority_key("tool.shell.allow_pipelines")
        )
        self.assertTrue(remote_runtime_authority_key("tool_backend"))
        self.assertTrue(remote_runtime_authority_key("secretName"))
        for key in (
            "approval",
            "capabilities",
            "command_mode",
            "deny_roots",
            "envvars",
            "mode",
            "platform",
            "privileged",
            "pull_policy",
            "read_roots",
            "read_only_rootfs",
            "uid",
            "user",
            "write_roots",
            "workspace_root",
        ):
            with self.subTest(key=key):
                self.assertTrue(remote_runtime_authority_key(key))

    def test_reject_fields_skips_trusted_key_and_walks_sequences(
        self,
    ) -> None:
        reject_remote_runtime_authority_fields(
            {
                "trusted": {"image": "operator-owned"},
                "items": [{"metadata": {"trace_id": "request-1"}}],
            },
            skip_keys=frozenset({"trusted"}),
        )

        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_fields(
                [{"metadata": {"containerImage": "untrusted"}}]
            )

    def test_reject_extra_fields_allows_profile_selector_shape(
        self,
    ) -> None:
        reject_remote_runtime_authority_extra_fields(
            {"containerProfile": "workspace-readonly"},
            allowed_fields=frozenset(),
            allow_container_profile_selector=True,
        )
        reject_remote_runtime_authority_extra_fields(
            "literal",
            allowed_fields=frozenset(),
        )
        reject_remote_runtime_authority_extra_fields(
            {"messages": [{"metadata": {"trace_id": "request-1"}}]},
            allowed_fields=frozenset({"messages"}),
        )
        reject_remote_runtime_authority_extra_fields(
            {"metadata": {"trace_id": "request-1"}},
            allowed_fields=frozenset(),
        )

        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_extra_fields(
                {"containerProfile": {"image": "untrusted"}},
                allowed_fields=frozenset(),
                allow_container_profile_selector=True,
            )

    def test_reject_extra_fields_scans_allowed_field_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_extra_fields(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "hi",
                            "metadata": {"sandboxProfile": "debug"},
                        }
                    ]
                },
                allowed_fields=frozenset({"messages"}),
                allow_container_profile_selector=True,
            )

    def test_allows_json_schema_property_authority_names(self) -> None:
        reject_remote_runtime_authority_fields(
            {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string"},
                        "allow_pipelines": {"type": "boolean"},
                        "resources": {
                            "type": "object",
                            "properties": {
                                "environment": {"type": "string"},
                            },
                        },
                    },
                    "required": ["mode"],
                },
            },
            path="chat.tools[0].function",
        )
        reject_remote_runtime_authority_fields(
            {
                "schema": {
                    "type": ["object", "null"],
                    "properties": {
                        "sandboxProfile": {"type": "string"},
                    },
                },
            },
            path="responses.response_format",
        )
        reject_remote_runtime_authority_fields(
            {
                "schema": {
                    "type": "object",
                    "$defs": {
                        "Resources": {
                            "type": "object",
                            "properties": {
                                "environment": {"type": "string"},
                            },
                        },
                    },
                    "definitions": {
                        "RuntimeProfile": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string"},
                            },
                        },
                    },
                    "patternProperties": {
                        "^mode$": {"type": "string"},
                    },
                    "dependentSchemas": {
                        "resources": {
                            "type": "object",
                            "properties": {
                                "sandboxProfile": {"type": "string"},
                            },
                        },
                    },
                },
            },
            path="responses.response_format",
        )

    def test_rejects_runtime_authority_in_schema_property_definition(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_fields(
                {
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "default": {"sandboxProfile": "debug"},
                            },
                        },
                    },
                },
                path="chat.tools[0].function",
            )

    def test_rejects_property_name_authority_outside_schema(self) -> None:
        for value in (
            {"properties": {"mode": {"type": "string"}}},
            {"$defs": {"sandboxProfile": {"type": "string"}}},
            {"definitions": {"resources": {"type": "object"}}},
            {"patternProperties": {"^mode$": {"type": "string"}}},
            {"dependentSchemas": {"resources": {"type": "object"}}},
            {
                "metadata": {
                    "type": "event",
                    "properties": {"sandboxProfile": "debug"},
                },
            },
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "runtime authority",
                ):
                    reject_remote_runtime_authority_fields(value)

    def test_rejects_schema_declaration_default_authority(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_fields(
                {
                    "schema": {
                        "type": "object",
                        "$defs": {
                            "Resources": {
                                "type": "object",
                                "default": {
                                    "mode": "container",
                                },
                            },
                        },
                    },
                },
                path="responses.response_format",
            )

    def test_rejects_runtime_authority_model_query_params(self) -> None:
        reject_remote_runtime_authority_model_identifier("model-id")
        reject_remote_runtime_authority_model_identifier({"model": "id"})
        reject_remote_runtime_authority_model_identifier(
            "ai://local/model?temperature=0"
        )

        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_model_identifier(
                "ai://local/model?backend=container"
            )
