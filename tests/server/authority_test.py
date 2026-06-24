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
        self.assertTrue(remote_runtime_authority_key("runtimeProfile"))
        self.assertTrue(remote_runtime_authority_key("tool_backend"))
        self.assertTrue(remote_runtime_authority_key("secretName"))
        for key in (
            "capabilities",
            "command_mode",
            "envvars",
            "platform",
            "privileged",
            "pull_policy",
            "read_only_rootfs",
            "uid",
            "user",
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

        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_extra_fields(
                {"containerProfile": {"image": "untrusted"}},
                allowed_fields=frozenset(),
                allow_container_profile_selector=True,
            )

    def test_rejects_runtime_authority_model_query_params(self) -> None:
        reject_remote_runtime_authority_model_identifier(
            "ai://local/model?temperature=0"
        )

        with self.assertRaisesRegex(ValueError, "runtime authority"):
            reject_remote_runtime_authority_model_identifier(
                "ai://local/model?backend=container"
            )
