from unittest import TestCase, main

from avalan.container import (
    ContainerBackend,
    ContainerCommandMode,
    ContainerExecutionScope,
    ContainerNetworkMode,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerPoolingMode,
    ContainerPoolTeardownMode,
    ContainerProfile,
    ContainerProfileBackedToolContract,
    ContainerProfileBackedToolInvocation,
    ContainerProfilePayloadContract,
    ContainerProfilePayloadField,
    ContainerProfilePayloadType,
    ContainerResourceLimits,
    ContainerServiceProfileSpec,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    container_profile_settings,
    container_service_profile,
    normalize_container_run_plan,
)
from avalan.container import profiles as profiles_module

_DIGEST = "9" * 64
_IMAGE = f"ghcr.io/example/service@sha256:{_DIGEST}"


class ContainerProfilesTest(TestCase):
    def test_profile_backed_contract_accepts_model_safe_selector(
        self,
    ) -> None:
        source = ContainerSettingsSource(
            surface=ContainerSurface.SERVER,
            trust_level=ContainerTrustLevel.MODEL,
        )
        contract = _profile_contract()
        loaded = ContainerProfileBackedToolContract.from_dict(
            contract.to_dict()
        )

        invocation = loaded.invocation_from_mapping(
            {
                "profile": "service-small",
                "payload": {
                    "query": "select 1",
                    "limit": 3,
                    "dry_run": True,
                },
            },
            source=source,
        )
        selection = invocation.to_selection(
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION
        )

        self.assertEqual(invocation.profile, "service-small")
        self.assertEqual(
            invocation.to_dict(),
            {
                "profile": "service-small",
                "payload": {
                    "query": "select 1",
                    "limit": 3,
                    "dry_run": True,
                },
            },
        )
        self.assertEqual(selection.profile, "service-small")
        self.assertTrue(selection.required)
        self.assertEqual(
            selection.scope,
            ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        )

    def test_profile_backed_contract_accepts_missing_optional_payload(
        self,
    ) -> None:
        invocation = _profile_contract().invocation_from_mapping(
            {
                "profile": "service-small",
                "payload": {"query": "select 1"},
            }
        )

        self.assertEqual(invocation.payload, {"query": "select 1"})

    def test_profile_payload_contract_normalizes_none_fields(self) -> None:
        contract = ContainerProfilePayloadContract.from_dict({"fields": None})

        self.assertEqual(contract.fields, {})

    def test_model_safe_selector_rejects_runtime_authority_fields(
        self,
    ) -> None:
        source = ContainerSettingsSource(
            surface=ContainerSurface.SERVER,
            trust_level=ContainerTrustLevel.MODEL,
        )
        contract = _profile_contract()
        runtime_fields = (
            "backend",
            "image",
            "mounts",
            "network",
            "devices",
            "secrets",
            "user",
            "runtime_flags",
        )

        for field_name in runtime_fields:
            with self.subTest(field_name=field_name):
                with self.assertRaises(AssertionError):
                    contract.invocation_from_mapping(
                        {
                            "profile": "service-small",
                            "payload": {"query": "select 1"},
                            field_name: "unsafe",
                        },
                        source=source,
                    )
        with self.assertRaises(AssertionError):
            contract.invocation_from_mapping(
                {
                    "profile": "service-small",
                    "payload": {
                        "query": "select 1",
                        "image": _IMAGE,
                    },
                },
                source=source,
            )
        with self.assertRaises(AssertionError):
            ContainerProfilePayloadField(
                name="image",
                payload_type=ContainerProfilePayloadType.STRING,
            )
        for field_name in (
            "command_mode",
            "resources",
            "runtimeProfile",
            "container_profile",
            "containerized",
            "tool_backend",
            "secretName",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(AssertionError):
                    ContainerProfilePayloadField(
                        name=field_name,
                        payload_type=ContainerProfilePayloadType.STRING,
                    )
                with self.assertRaises(AssertionError):
                    ContainerProfileBackedToolInvocation(
                        profile="service-small",
                        payload={field_name: "unsafe"},
                    )
        self.assertFalse(profiles_module._payload_runtime_authority_key(""))

    def test_profile_backed_contract_rejects_unsafe_selectors_and_payload(
        self,
    ) -> None:
        contract = _profile_contract()

        cases = (
            {"profile": "unknown", "payload": {"query": "select 1"}},
            {"profile": "service-small", "payload": {}},
            {
                "profile": "service-small",
                "payload": {"query": "select 1", "limit": True},
            },
            {
                "profile": "service-small",
                "payload": {"query": "select 1", "extra": "no"},
            },
            {
                "profile": "service-small",
                "payload": {"query": "", "limit": 1},
            },
            {
                "profile": "service-small",
                "payload": {"query": "select 1", "limit": 11},
            },
        )

        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(AssertionError):
                    contract.invocation_from_mapping(raw)
        with self.assertRaises(AssertionError):
            ContainerProfilePayloadField(
                name="limit",
                payload_type=ContainerProfilePayloadType.INTEGER,
                min_value=10,
                max_value=1,
            )
        with self.assertRaises(AssertionError):
            ContainerProfilePayloadField(
                name="flag",
                payload_type=ContainerProfilePayloadType.BOOLEAN,
                max_length=1,
            )
        with self.assertRaises(AssertionError):
            ContainerProfilePayloadField(
                name="score",
                payload_type="float",
            )
        with self.assertRaises(AssertionError):
            ContainerProfilePayloadContract(
                fields={
                    "query": ContainerProfilePayloadField(
                        name="other",
                        payload_type=ContainerProfilePayloadType.STRING,
                    )
                }
            )

    def test_service_profile_spec_builds_short_lived_service_profile(
        self,
    ) -> None:
        source = _trusted_source()
        spec = ContainerServiceProfileSpec.from_dict(
            {
                "name": "service-small",
                "image_reference": _IMAGE,
                "health_check_command": ["avalan-health", "--ready"],
                "audit_labels": {
                    "surface": "server",
                    "workload": "short-lived",
                },
                "max_age_seconds": 45,
                "max_uses": 4,
                "idle_ttl_seconds": 10,
                "cleanup_grace_seconds": 3,
                "resources": {
                    "cpu_count": 1,
                    "memory_bytes": 1048576,
                    "pids": 16,
                    "timeout_seconds": 20,
                },
                "output": {"max_stdout_bytes": 4096},
            },
            source=source,
        )

        loaded = ContainerServiceProfileSpec.from_dict(
            spec.to_dict(),
            source=source,
        )
        profile = spec.to_profile()
        round_trip_profile = ContainerProfile.from_dict(profile.to_dict())
        settings = container_profile_settings(
            source=source,
            backend=ContainerBackend.DOCKER,
            profiles=(profile,),
            default_profile=profile.name,
        )
        invocation = ContainerProfileBackedToolInvocation(
            profile=profile.name,
        )
        effective = settings.select_profile(invocation.to_selection())
        helper_profile = container_service_profile(
            name="service-helper",
            image_reference=_IMAGE,
            health_check_command=("avalan-health",),
            audit_labels={"surface": "server"},
            max_age_seconds=30,
            max_uses=2,
            idle_ttl_seconds=5,
            resources=ContainerResourceLimits(
                cpu_count=1,
                memory_bytes=1048576,
                pids=16,
                timeout_seconds=20,
            ),
        )

        self.assertEqual(loaded.to_dict(), spec.to_dict())
        self.assertEqual(round_trip_profile.to_dict(), profile.to_dict())
        self.assertEqual(
            profile.command_mode,
            ContainerCommandMode.SERVICE_COMMAND,
        )
        self.assertEqual(profile.pooling.mode, ContainerPoolingMode.SERVICE)
        self.assertEqual(
            profile.pooling.health_check_command,
            ("avalan-health", "--ready"),
        )
        self.assertEqual(profile.pooling.max_age_seconds, 45)
        self.assertEqual(profile.pooling.max_uses, 4)
        self.assertEqual(profile.pooling.idle_ttl_seconds, 10)
        self.assertEqual(
            profile.pooling.teardown,
            ContainerPoolTeardownMode.REMOVE,
        )
        self.assertFalse(profile.pooling.allow_secret_reuse)
        self.assertEqual(profile.network.mode, ContainerNetworkMode.LOOPBACK)
        self.assertEqual(profile.devices.devices, ())
        self.assertEqual(profile.secrets, ())
        self.assertTrue(profile.read_only_rootfs)
        self.assertEqual(profile.user, "1000:1000")
        self.assertEqual(profile.cleanup.grace_seconds, 3)
        self.assertEqual(profile.resources.cpu_count, 1)
        self.assertEqual(profile.resources.memory_bytes, 1048576)
        self.assertEqual(profile.resources.pids, 16)
        self.assertEqual(profile.resources.timeout_seconds, 20)
        self.assertEqual(effective.profile_name, "service-small")
        self.assertEqual(helper_profile.name, "service-helper")

    def test_service_profile_normalizes_to_service_run_plan(self) -> None:
        source = _trusted_source()
        profile = ContainerServiceProfileSpec.from_dict(
            {
                "name": "service-helper",
                "image_reference": _IMAGE,
                "health_check_command": ["avalan-health"],
                "audit_labels": {"surface": "server"},
                "max_age_seconds": 30,
                "max_uses": 2,
                "idle_ttl_seconds": 5,
                "resources": {
                    "cpu_count": 1,
                    "memory_bytes": 1048576,
                    "pids": 16,
                    "timeout_seconds": 20,
                },
            },
            source=source,
        ).to_profile()
        settings = container_profile_settings(
            source=source,
            backend=ContainerBackend.DOCKER,
            profiles=(profile,),
            default_profile=profile.name,
        )
        effective = settings.select_profile(
            ContainerProfileBackedToolInvocation(
                profile=profile.name,
            ).to_selection()
        )
        plan = normalize_container_run_plan(
            effective,
            ContainerPlanRequest(
                request_kind=ContainerPlanRequestKind.TYPED_TOOL,
                logical_name="service-helper",
                command="serve",
                argv=("serve", "--once"),
            ),
        )

        self.assertEqual(
            plan.command_mode,
            ContainerCommandMode.SERVICE_COMMAND,
        )
        self.assertEqual(
            plan.run_plan.network.mode,
            ContainerNetworkMode.LOOPBACK,
        )
        self.assertEqual(plan.run_plan.devices.devices, ())
        self.assertEqual(plan.run_plan.secret_names, ())
        self.assertEqual(
            plan.run_plan.pooling.mode,
            ContainerPoolingMode.SERVICE,
        )
        self.assertEqual(plan.run_plan.pooling.max_age_seconds, 30)
        self.assertEqual(plan.run_plan.pooling.max_uses, 2)
        self.assertEqual(plan.run_plan.pooling.idle_ttl_seconds, 5)
        self.assertEqual(
            plan.run_plan.pooling.health_check_command,
            ("avalan-health",),
        )
        self.assertFalse(plan.run_plan.pooling.allow_secret_reuse)
        self.assertEqual(plan.run_plan.resources.cpu_count, 1)
        self.assertEqual(plan.run_plan.resources.memory_bytes, 1048576)
        self.assertEqual(plan.run_plan.resources.pids, 16)
        self.assertEqual(plan.run_plan.resources.timeout_seconds, 20)

    def test_service_profile_spec_rejects_untrusted_or_widened_runtime(
        self,
    ) -> None:
        trusted = _trusted_source()
        untrusted = ContainerSettingsSource(
            surface=ContainerSurface.SERVER,
            trust_level=ContainerTrustLevel.MODEL,
        )
        base = _service_spec_mapping()

        with self.assertRaises(AssertionError):
            ContainerServiceProfileSpec.from_dict(base, source=untrusted)
        for field_name in (
            "backend",
            "image",
            "mounts",
            "devices",
            "secrets",
            "user",
            "runtime_flags",
        ):
            raw = dict(base)
            raw[field_name] = "unsafe"
            with self.subTest(field_name=field_name):
                with self.assertRaises(AssertionError):
                    ContainerServiceProfileSpec.from_dict(raw, source=trusted)

        cases = (
            {"health_check_command": []},
            {"audit_labels": {}},
            {"network": "full"},
            {"teardown": "reset"},
            {"max_age_seconds": 0},
            {"resources": {}},
            {"resources": {"cpu_count": 1}},
            {"resources": {"cpu_count": True}},
        )
        for changes in cases:
            raw = dict(base)
            raw.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaises(AssertionError):
                    ContainerServiceProfileSpec.from_dict(raw, source=trusted)


def _profile_contract() -> ContainerProfileBackedToolContract:
    return ContainerProfileBackedToolContract(
        name="generic-service-tool",
        allowed_profiles=("service-small", "service-large"),
        payload_contract=ContainerProfilePayloadContract(
            fields={
                "query": ContainerProfilePayloadField(
                    name="query",
                    payload_type=ContainerProfilePayloadType.STRING,
                    max_length=80,
                ),
                "limit": ContainerProfilePayloadField(
                    name="limit",
                    payload_type=ContainerProfilePayloadType.INTEGER,
                    required=False,
                    min_value=1,
                    max_value=10,
                ),
                "dry_run": ContainerProfilePayloadField(
                    name="dry_run",
                    payload_type=ContainerProfilePayloadType.BOOLEAN,
                    required=False,
                ),
            },
        ),
    )


def _trusted_source() -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=ContainerSurface.SERVER,
        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
    )


def _service_spec_mapping() -> dict[str, object]:
    return {
        "name": "service-small",
        "image_reference": _IMAGE,
        "health_check_command": ["avalan-health"],
        "audit_labels": {"surface": "server"},
        "max_age_seconds": 30,
        "max_uses": 3,
        "idle_ttl_seconds": 5,
        "resources": {
            "cpu_count": 1,
            "memory_bytes": 1048576,
            "pids": 16,
            "timeout_seconds": 20,
        },
    }


if __name__ == "__main__":
    main()
