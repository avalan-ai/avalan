import asyncio
from json import dumps
from logging import Logger
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent import Specification
from avalan.container import (
    ContainerAuditMode,
    ContainerAuditPolicy,
    ContainerBackend,
    ContainerCleanupMode,
    ContainerCleanupPolicy,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerImagePolicy,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerOutputPolicy,
    ContainerProfile,
    ContainerResourceLimits,
    ContainerRuntimeEnvelopeKind,
    ContainerSecretReference,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerToolRuntimeSettings,
    ContainerTrustLevel,
)
from avalan.entities import (
    Backend,
    EngineUri,
    GenerationSettings,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.model import manager as model_manager
from avalan.model import runtime as model_runtime
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import (
    ModelManager,
    ModelRuntimeEnvelopeUnavailableError,
)
from avalan.model.runtime import (
    ModelBackendAcceleratorClass,
    ModelBackendEnvelopeLifecyclePhase,
    ModelBackendEnvelopeLifecycleScript,
    ModelBackendEnvelopeLifecycleStatus,
    ModelBackendRuntimeDiagnostic,
    ModelBackendRuntimePolicy,
    ModelBackendRuntimePolicyError,
    model_backend_accelerator_class,
    simulate_model_backend_envelope_lifecycle,
    trusted_model_backend_profile_selection,
)


class ModelManagerExtraTestCase(TestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.hub.cache_dir = "cache"
        self.logger = MagicMock(spec=Logger)

    def test_parse_uri_invalid_scheme(self):
        manager = ModelManager(self.hub, self.logger)
        with self.assertRaises(ValueError):
            manager.parse_uri("http://openai/gpt-4o")

    def test_parse_uri_params(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://openai/gpt-4o?temperature=0.6&max_new_tokens=8192&backend=mlx"
        )
        self.assertEqual(uri.params["temperature"], 0.6)
        self.assertEqual(uri.params["max_new_tokens"], 8192)
        self.assertEqual(uri.params["backend"], "mlx")

    def test_parse_uri_local_paths(self):
        manager = ModelManager(self.hub, self.logger)

        relative_uri = manager.parse_uri(
            "ai://local/../pyds4/.local/ds4/ds4flash.gguf"
        )
        absolute_uri = manager.parse_uri(
            "ai://local//Users/mariano/Code/ai/pyds4/.local/ds4/ds4flash.gguf"
        )
        encoded_absolute_uri = manager.parse_uri(
            "ai://local/%2FUsers/mariano/Code/ai/some%20dir/model.gguf"
        )

        self.assertEqual(
            relative_uri.model_id, "../pyds4/.local/ds4/ds4flash.gguf"
        )
        self.assertEqual(
            absolute_uri.model_id,
            "/Users/mariano/Code/ai/pyds4/.local/ds4/ds4flash.gguf",
        )
        self.assertEqual(
            encoded_absolute_uri.model_id,
            "/Users/mariano/Code/ai/some dir/model.gguf",
        )

    def test_get_engine_settings_user_password_no_secret(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user="token",
            password="pass",
            vendor="openai",
            model_id="gpt",
            params={},
        )
        settings = manager.get_engine_settings(uri)
        self.assertIsNone(settings.access_token)

    def test_load_passes_arguments(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            result = manager.load(
                uri,
                base_url="url",
                quiet=True,
                attention="sd",
                trust_remote_code=True,
                backend="mlxlm",
            )
        args = get_mock.call_args.args[1]
        self.assertTrue(args["disable_loading_progress_bar"])
        self.assertEqual(args["base_url"], "url")
        self.assertEqual(args["attention"], "sd")
        self.assertEqual(args["backend"], "mlxlm")
        self.assertTrue(args["trust_remote_code"])
        load_mock.assert_called_once_with(
            uri, get_mock.return_value, Modality.TEXT_GENERATION
        )
        self.assertEqual(result, "model")

    def test_model_backend_runtime_envelope_fails_closed(self):
        manager = ModelManager(
            self.hub,
            self.logger,
            container_runtime=_container_runtime_settings(),
        )
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="local-model",
            params={},
        )

        with self.assertRaises(ModelRuntimeEnvelopeUnavailableError) as raised:
            manager.load_engine(uri, TransformerEngineSettings())

        plan = raised.exception.plan
        self.assertEqual(
            plan.envelope_kind,
            ContainerRuntimeEnvelopeKind.MODEL_BACKEND,
        )
        self.assertEqual(plan.envelope_plan.profile_name, "model-runtime")
        self.assertEqual(
            raised.exception.diagnostic["code"],
            "model.runtime_envelope_unavailable",
        )

    def test_model_backend_runtime_envelope_loader_composes_local_model(self):
        class FakeModelEnvelopeLoader:
            trusted_runtime_envelope_runner = True

            def __init__(self) -> None:
                self.plan = None
                self.kwargs = None

            def load_model_backend_envelope(self, plan, **kwargs):
                self.plan = plan
                self.kwargs = kwargs
                return "enveloped-model"

        loader = FakeModelEnvelopeLoader()
        manager = ModelManager(
            self.hub,
            self.logger,
            container_runtime=_container_runtime_settings(),
            model_backend_envelope_loader=loader,
        )
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="local-model",
            params={},
        )

        result = manager.load_engine(uri, TransformerEngineSettings())

        self.assertEqual(result, "enveloped-model")
        assert loader.plan is not None
        self.assertEqual(
            loader.plan.envelope_kind,
            ContainerRuntimeEnvelopeKind.MODEL_BACKEND,
        )
        assert loader.kwargs is not None
        profile_selection = loader.kwargs["profile_selection"]
        self.assertIs(profile_selection.plan, loader.plan)
        profile = profile_selection.to_dict()["effective_settings"]["profile"]
        assert isinstance(profile, dict)
        self.assertEqual(profile["output"], ContainerOutputPolicy().to_dict())
        self.assertEqual(profile["audit"], {"mode": "full"})
        runtime_policy = profile_selection.to_dict()["runtime_policy"]
        assert isinstance(runtime_policy, dict)
        self.assertEqual(
            runtime_policy["output"],
            ContainerOutputPolicy().to_dict(),
        )
        self.assertEqual(runtime_policy["audit"], {"mode": "full"})
        self.assertIn("policy_fingerprint", runtime_policy)
        self.assertIn(
            "selection_fingerprint",
            profile_selection.to_dict(),
        )
        self.assertEqual(
            profile["cleanup"],
            {"mode": "remove", "grace_seconds": 5},
        )

    def test_model_backend_runtime_envelope_loader_must_be_trusted(self):
        class UntrustedModelEnvelopeLoader:
            def load_model_backend_envelope(self, plan, **kwargs):
                return "not-used"

        with self.assertRaisesRegex(
            AssertionError,
            "model backend envelope loader must be trusted",
        ):
            ModelManager(
                self.hub,
                self.logger,
                model_backend_envelope_loader=UntrustedModelEnvelopeLoader(),
            )

    def test_model_backend_runtime_invalid_required_settings_fail_closed(
        self,
    ):
        cases = (
            (
                _container_runtime_settings(
                    surface=ContainerSurface.SERVER,
                    scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                ),
                "model.runtime.surface_invalid",
            ),
            (
                ContainerToolRuntimeSettings(
                    effective_settings=ContainerEffectiveSettings(
                        backend=ContainerBackend.NONE,
                        required=True,
                        scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                        source=ContainerSettingsSource(
                            surface=ContainerSurface.MODEL_BACKEND,
                            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
                        ),
                        policy_version="phase16",
                        profile_registry_id="model",
                    )
                ),
                "model.runtime.disabled",
            ),
        )

        for container_runtime, expected in cases:
            with self.subTest(expected=expected):
                manager = ModelManager(
                    self.hub,
                    self.logger,
                    container_runtime=container_runtime,
                )

                with self.assertRaises(
                    ModelBackendRuntimePolicyError
                ) as raised:
                    manager.load_engine(
                        _local_uri("local-model"),
                        TransformerEngineSettings(),
                    )

                self.assertIn(expected, _policy_codes(raised.exception))

    def test_model_backend_runtime_envelope_ignores_optional_invalid_settings(
        self,
    ):
        manager = ModelManager(
            self.hub,
            self.logger,
            container_runtime=_container_runtime_settings(
                surface=ContainerSurface.SERVER,
                scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                required=False,
            ),
        )

        plan = manager.model_backend_runtime_envelope_plan(
            _local_uri("local-model"),
            TransformerEngineSettings(),
        )

        self.assertIsNone(plan)

    def test_model_backend_runtime_envelope_skips_remote_models(self):
        manager = ModelManager(
            self.hub,
            self.logger,
            container_runtime=_container_runtime_settings(),
        )
        uri = EngineUri(
            host="openai",
            port=None,
            user=None,
            password=None,
            vendor="openai",
            model_id="gpt",
            params={},
        )

        plan = manager.model_backend_runtime_envelope_plan(
            uri,
            TransformerEngineSettings(),
        )

        self.assertIsNone(plan)

    def test_model_backend_runtime_profile_selection_preserves_policy(self):
        resources = ContainerResourceLimits(
            cpu_count=4,
            memory_bytes=4 * 1024 * 1024,
            timeout_seconds=45,
        )
        manager = ModelManager(
            self.hub,
            self.logger,
            container_runtime=_container_runtime_settings(
                mounts=(
                    _workspace_mount(),
                    ContainerMountDeclaration(
                        target="/cache/hf",
                        mount_type=ContainerMountType.CACHE,
                    ),
                ),
                resources=resources,
            ),
        )

        selection = manager.model_backend_runtime_profile_selection(
            _local_uri("local-model"),
            TransformerEngineSettings(device="cpu"),
        )

        assert selection is not None
        self.assertEqual(
            selection.accelerator,
            ModelBackendAcceleratorClass.CPU,
        )
        self.assertEqual(
            selection.plan.run_plan.run_plan.resources.to_dict(),
            resources.to_dict(),
        )
        self.assertEqual(
            selection.plan.run_plan.run_plan.network.to_dict(),
            ContainerNetworkPolicy().to_dict(),
        )
        self.assertEqual(
            selection.plan.envelope_plan.profile_name,
            "model-runtime",
        )
        self.assertEqual(
            selection.plan.envelope_plan.readiness_timeout_seconds,
            30,
        )
        self.assertEqual(
            selection.runtime_policy.output.to_dict(),
            ContainerOutputPolicy().to_dict(),
        )
        self.assertEqual(
            selection.runtime_policy.audit.to_dict(),
            ContainerAuditPolicy(mode=ContainerAuditMode.FULL).to_dict(),
        )
        dumps(selection.to_dict(), sort_keys=True)

    def test_model_backend_runtime_authorizes_accelerator_classes(self):
        cases = (
            (
                TransformerEngineSettings(device="cuda:0"),
                (ContainerDeviceClass.NVIDIA_CDI,),
                ModelBackendAcceleratorClass.NVIDIA_CDI,
            ),
            (
                TransformerEngineSettings(device="rocm"),
                (ContainerDeviceClass.AMD_CDI,),
                ModelBackendAcceleratorClass.AMD_CDI,
            ),
            (
                TransformerEngineSettings(device="vulkan:0"),
                (ContainerDeviceClass.VULKAN_FORWARDED,),
                ModelBackendAcceleratorClass.VULKAN_FORWARDED,
            ),
            (
                TransformerEngineSettings(
                    device="cuda",
                    backend_config={"native_backend": "auto"},
                ),
                (ContainerDeviceClass.NVIDIA_CDI,),
                ModelBackendAcceleratorClass.NVIDIA_CDI,
            ),
        )

        for settings, devices, expected in cases:
            with self.subTest(expected=expected):
                manager = ModelManager(
                    self.hub,
                    self.logger,
                    container_runtime=_container_runtime_settings(
                        devices=devices,
                    ),
                )

                selection = manager.model_backend_runtime_profile_selection(
                    _local_uri("local-model"),
                    settings,
                )

                assert selection is not None
                self.assertEqual(selection.accelerator, expected)

    def test_model_backend_runtime_policy_denies_unsafe_profiles(self):
        cases = (
            (
                _container_runtime_settings(
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.LOOPBACK,
                    ),
                ).effective_settings,
                TransformerEngineSettings(),
                "model.runtime.network_denied",
            ),
            (
                _container_runtime_settings(
                    secrets=(
                        ContainerSecretReference(
                            name="token",
                            env_name="TOKEN",
                        ),
                    ),
                ).effective_settings,
                TransformerEngineSettings(),
                "model.runtime.secret_leakage_risk",
            ),
            (
                _container_runtime_settings(
                    mounts=(
                        _workspace_mount(),
                        ContainerMountDeclaration(
                            source="cache",
                            target="/cache",
                            mount_type=ContainerMountType.CACHE,
                        ),
                    ),
                ).effective_settings,
                TransformerEngineSettings(),
                "model.runtime.cache_misuse",
            ),
            (
                _container_runtime_settings(
                    resources=ContainerResourceLimits(
                        cpu_count=2,
                        timeout_seconds=30,
                    ),
                ).effective_settings,
                TransformerEngineSettings(),
                "model.runtime.resource_mismatch",
            ),
            (
                _container_runtime_settings(
                    output=ContainerOutputPolicy(
                        allow_artifacts=True,
                        max_artifact_bytes=64,
                    ),
                ).effective_settings,
                TransformerEngineSettings(),
                "model.runtime.output_policy_unsafe",
            ),
            (
                _container_runtime_settings(
                    cleanup=ContainerCleanupPolicy(
                        mode=ContainerCleanupMode.QUARANTINE,
                    ),
                ).effective_settings,
                TransformerEngineSettings(),
                "model.runtime.cleanup_policy_unsafe",
            ),
            (
                _container_runtime_settings(
                    audit=ContainerAuditPolicy(
                        mode=ContainerAuditMode.MINIMAL
                    ),
                ).effective_settings,
                TransformerEngineSettings(),
                "model.runtime.audit_insufficient",
            ),
        )

        for settings, engine_settings, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(
                    ModelBackendRuntimePolicyError
                ) as raised:
                    trusted_model_backend_profile_selection(
                        settings,
                        engine_settings=engine_settings,
                        modality=Modality.TEXT_GENERATION,
                        model_id="local-model",
                    )

                self.assertIn(expected, _policy_codes(raised.exception))
                dumps(raised.exception.to_dict(), sort_keys=True)

    def test_model_backend_runtime_policy_denies_authority_mismatch(self):
        digest = (
            "sha256:"
            "1111111111111111111111111111111111111111111111111111111111111111"
        )
        cases = (
            (
                ContainerEffectiveSettings(
                    backend=ContainerBackend.NONE,
                    required=True,
                    scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    source=ContainerSettingsSource(
                        surface=ContainerSurface.MODEL_BACKEND,
                        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
                    ),
                    policy_version="phase16",
                    profile_registry_id="model",
                ),
                "model.runtime.disabled",
            ),
            (
                _container_runtime_settings(
                    surface=ContainerSurface.SERVER,
                ).effective_settings,
                "model.runtime.surface_invalid",
            ),
            (
                _container_runtime_settings(
                    trust_level=ContainerTrustLevel.UNTRUSTED_REQUEST,
                ).effective_settings,
                "model.runtime.profile_untrusted",
            ),
            (
                _container_runtime_settings(
                    scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                ).effective_settings,
                "model.runtime.scope_invalid",
            ),
            (
                ContainerEffectiveSettings(
                    backend=ContainerBackend.DOCKER,
                    required=True,
                    scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
                    source=ContainerSettingsSource(
                        surface=ContainerSurface.MODEL_BACKEND,
                        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
                    ),
                    policy_version="phase16",
                    profile_registry_id="model",
                ),
                "model.runtime.profile_missing",
            ),
            (
                _container_runtime_settings(
                    allowed_profiles=(),
                ).effective_settings,
                "model.runtime.profile_mismatch",
            ),
            (
                _container_runtime_settings(
                    image=ContainerImagePolicy(
                        reference="registry.example/model:latest",
                        digest=digest,
                    ),
                ).effective_settings,
                "model.runtime.image_untrusted",
            ),
        )

        for settings, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(
                    ModelBackendRuntimePolicyError
                ) as raised:
                    trusted_model_backend_profile_selection(
                        settings,
                        engine_settings=TransformerEngineSettings(),
                        modality=Modality.TEXT_GENERATION,
                        model_id="local-model",
                    )

                self.assertIn(expected, _policy_codes(raised.exception))
                self.assertEqual(
                    raised.exception.diagnostic["code"],
                    _policy_codes(raised.exception)[0],
                )

    def test_model_backend_runtime_policy_denies_plan_mismatch(self):
        settings = _container_runtime_settings().effective_settings
        assert settings.profile is not None
        bad_plan = SimpleNamespace(
            envelope_kind=ContainerRuntimeEnvelopeKind.SERVER,
            envelope_plan=SimpleNamespace(
                scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            ),
            run_plan=SimpleNamespace(
                run_plan=SimpleNamespace(
                    profile_name="other-runtime",
                    image=ContainerImagePolicy(
                        reference=(
                            "registry.example/other@sha256:"
                            "2222222222222222222222222222222222222222222222222222222222222222"
                        ),
                    ),
                    mounts=(),
                    secret_names=("token",),
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.LOOPBACK,
                    ),
                    devices=ContainerDevicePolicy(
                        devices=(ContainerDeviceClass.NVIDIA_CDI,),
                    ),
                    resources=ContainerResourceLimits(
                        cpu_count=1,
                        memory_bytes=1,
                        timeout_seconds=1,
                    ),
                )
            ),
        )

        diagnostics = (
            model_runtime._plan_preservation_diagnostics(  # noqa: SLF001
                settings,
                bad_plan,
            )
        )

        self.assertEqual(
            [diagnostic.path for diagnostic in diagnostics],
            [
                "model.container.plan.envelope_kind",
                "model.container.plan.scope",
                "model.container.plan.profile_name",
                "model.container.plan.image",
                "model.container.plan.mounts",
                "model.container.plan.secrets",
                "model.container.plan.network",
                "model.container.plan.devices",
                "model.container.plan.resources",
            ],
        )

    def test_model_backend_runtime_policy_denies_runtime_policy_mismatch(self):
        settings = _container_runtime_settings().effective_settings
        bad_policy = ModelBackendRuntimePolicy(
            output=ContainerOutputPolicy(
                allow_artifacts=True,
                max_artifact_bytes=1,
            ),
            cleanup=ContainerCleanupPolicy(
                mode=ContainerCleanupMode.QUARANTINE,
            ),
            audit=ContainerAuditPolicy(mode=ContainerAuditMode.MINIMAL),
        )

        diagnostics = model_runtime._runtime_policy_preservation_diagnostics(  # noqa: SLF001
            settings,
            bad_policy,
        )

        self.assertEqual(
            [diagnostic.path for diagnostic in diagnostics],
            [
                "model.container.plan.output",
                "model.container.plan.cleanup",
                "model.container.plan.audit",
            ],
        )

    def test_model_backend_runtime_policy_denies_post_plan_diagnostic(self):
        diagnostic = ModelBackendRuntimeDiagnostic(
            code="model.runtime.policy_mismatch",
            path="model.container.plan.image",
            message="Image changed.",
            hint="Recompute the plan.",
        )

        with (
            patch.object(
                model_runtime,
                "_plan_preservation_diagnostics",
                return_value=(diagnostic,),
            ),
            self.assertRaises(ModelBackendRuntimePolicyError) as raised,
        ):
            trusted_model_backend_profile_selection(
                _container_runtime_settings().effective_settings,
                engine_settings=TransformerEngineSettings(),
                modality=Modality.TEXT_GENERATION,
                model_id="local-model",
            )

        self.assertEqual(
            raised.exception.diagnostic["code"],
            "model.runtime.policy_mismatch",
        )

    def test_model_backend_runtime_policy_denies_devices_and_metal(self):
        cases = (
            (
                TransformerEngineSettings(device="tpu"),
                (),
                "model.runtime.device_unsupported",
            ),
            (
                TransformerEngineSettings(device="cuda"),
                (),
                "model.runtime.device_denied",
            ),
            (
                TransformerEngineSettings(device="mps"),
                (ContainerDeviceClass.NVIDIA_CDI,),
                "model.runtime.metal_container_unsupported",
            ),
            (
                TransformerEngineSettings(
                    backend=Backend.DS4,
                    backend_config={"native_backend": "metal"},
                ),
                (),
                "model.runtime.metal_container_unsupported",
            ),
        )

        for settings, devices, expected in cases:
            with self.subTest(expected=expected):
                manager = ModelManager(
                    self.hub,
                    self.logger,
                    container_runtime=_container_runtime_settings(
                        devices=devices,
                    ),
                )

                with self.assertRaises(
                    ModelBackendRuntimePolicyError
                ) as raised:
                    manager.load_engine(_local_uri("local-model"), settings)

                self.assertIn(expected, _policy_codes(raised.exception))

    def test_model_backend_accelerator_classifies_cpu_and_amd_native(self):
        self.assertEqual(
            model_backend_accelerator_class(TransformerEngineSettings()),
            ModelBackendAcceleratorClass.CPU,
        )
        self.assertEqual(
            model_backend_accelerator_class(
                TransformerEngineSettings(
                    backend_config={"native_backend": "hip"},
                )
            ),
            ModelBackendAcceleratorClass.AMD_CDI,
        )
        self.assertEqual(
            model_backend_accelerator_class(
                TransformerEngineSettings(
                    backend_config={"native_backend": object()},
                )
            ),
            ModelBackendAcceleratorClass.UNSUPPORTED,
        )

    def test_model_backend_lifecycle_streaming_health_shutdown(self):
        plan = _model_backend_plan(self.hub, self.logger)

        result = simulate_model_backend_envelope_lifecycle(
            plan,
            ModelBackendEnvelopeLifecycleScript(
                stream_chunks=("hello", " world"),
                result="hello world",
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.stream_chunks, ("hello", " world"))
        self.assertEqual(result.result, "hello world")
        self.assertEqual(
            result.health.status,
            ModelBackendEnvelopeLifecycleStatus.COMPLETED,
        )
        self.assertEqual(
            result.shutdown.phase,
            ModelBackendEnvelopeLifecyclePhase.SHUTDOWN,
        )
        dumps(result.to_dict(), sort_keys=True)

    def test_model_backend_lifecycle_reports_startup_timeout(self):
        plan = _model_backend_plan(self.hub, self.logger)

        result = simulate_model_backend_envelope_lifecycle(
            plan,
            ModelBackendEnvelopeLifecycleScript(startup_timeout=True),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.startup.status,
            ModelBackendEnvelopeLifecycleStatus.FAILED,
        )
        self.assertEqual(
            result.diagnostics[0].code,
            "model.runtime.startup_timeout",
        )
        self.assertEqual(
            result.cleanup.status,
            ModelBackendEnvelopeLifecycleStatus.COMPLETED,
        )

    def test_model_backend_lifecycle_reports_cancel_cleanup_failure(self):
        plan = _model_backend_plan(self.hub, self.logger)

        result = simulate_model_backend_envelope_lifecycle(
            plan,
            ModelBackendEnvelopeLifecycleScript(
                stream_chunks=("a", "b"),
                cancel_after_chunks=1,
                cleanup_failure=True,
                health_failure=True,
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.stream_chunks, ("a",))
        self.assertIsNone(result.result)
        self.assertEqual(
            result.request_streaming.status,
            ModelBackendEnvelopeLifecycleStatus.CANCELLED,
        )
        self.assertEqual(
            result.cancellation.status,
            ModelBackendEnvelopeLifecycleStatus.COMPLETED,
        )
        self.assertEqual(
            {diagnostic.code for diagnostic in result.diagnostics},
            {
                "model.runtime.request_cancelled",
                "model.runtime.health_failed",
                "model.runtime.cleanup_failed",
            },
        )

    def test_model_backend_runtime_fake_e2e_lifecycle_loader(self):
        class FakeModelEnvelopeLoader:
            trusted_runtime_envelope_runner = True

            def __init__(
                self,
                script: ModelBackendEnvelopeLifecycleScript,
            ) -> None:
                self.script = script
                self.lifecycle_result = None

            def load_model_backend_envelope(self, plan, **kwargs):
                self.lifecycle_result = (
                    simulate_model_backend_envelope_lifecycle(
                        plan,
                        self.script,
                    )
                )
                return self.lifecycle_result.result

        cases = (
            (
                "success",
                ModelBackendEnvelopeLifecycleScript(
                    stream_chunks=("a", "b"),
                    result="ab",
                ),
                "ab",
                (),
            ),
            (
                "cancelled",
                ModelBackendEnvelopeLifecycleScript(
                    stream_chunks=("a", "b"),
                    result="ab",
                    cancel_after_chunks=1,
                ),
                None,
                ("model.runtime.request_cancelled",),
            ),
        )

        for name, script, expected_result, expected_codes in cases:
            with self.subTest(name):
                loader = FakeModelEnvelopeLoader(script)
                manager = ModelManager(
                    self.hub,
                    self.logger,
                    container_runtime=_container_runtime_settings(),
                    model_backend_envelope_loader=loader,
                )

                result = manager.load_engine(
                    _local_uri("local-model"),
                    TransformerEngineSettings(),
                )

                self.assertEqual(result, expected_result)
                assert loader.lifecycle_result is not None
                self.assertEqual(
                    [step.phase for step in loader.lifecycle_result.steps],
                    [
                        ModelBackendEnvelopeLifecyclePhase.STARTUP,
                        ModelBackendEnvelopeLifecyclePhase.WARMUP,
                        ModelBackendEnvelopeLifecyclePhase.REQUEST_STREAMING,
                        ModelBackendEnvelopeLifecyclePhase.CANCELLATION,
                        ModelBackendEnvelopeLifecyclePhase.HEALTH,
                        ModelBackendEnvelopeLifecyclePhase.SHUTDOWN,
                        ModelBackendEnvelopeLifecyclePhase.CLEANUP,
                    ],
                )
                self.assertEqual(
                    tuple(
                        diagnostic.code
                        for diagnostic in loader.lifecycle_result.diagnostics
                    ),
                    expected_codes,
                )

    def test_load_backend_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={"backend": "mlx"},
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)
        args = get_mock.call_args.args[1]
        self.assertEqual(args["backend"], Backend.MLXLM)

    def test_get_engine_settings_uses_provider_options_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user=None,
            password=None,
            vendor="openai",
            model_id="deployment",
            params={"azure_api_version": "2025-04-01-preview"},
        )

        settings = manager.get_engine_settings(uri, {})

        self.assertEqual(
            settings.provider_options,
            {"azure_api_version": "2025-04-01-preview"},
        )

    def test_get_engine_settings_merges_generic_provider_options(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user=None,
            password=None,
            vendor="openai",
            model_id="deployment",
            params={"provider_timeout": 30},
        )

        settings = manager.get_engine_settings(
            uri,
            {"provider_options": {"timeout": 60, "region": "eastus"}},
        )

        self.assertEqual(
            settings.provider_options,
            {"timeout": 60, "region": "eastus"},
        )

    def test_backend_ds4_value(self):
        self.assertEqual(Backend("ds4"), Backend.DS4)

    def test_load_ds4_backend_config_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_ctx=4096"
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)

        args = get_mock.call_args.args[1]
        self.assertEqual(args["backend"], Backend.DS4)
        self.assertEqual(args["backend_config"], {"ctx_size": 4096})
        self.assertNotIn("ctx_size", args)
        self.assertNotIn("ds4_ctx", args)

    def test_get_engine_settings_uses_ds4_backend_config_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_ctx=4096"
            "&ds4_native_backend=metal&ds4_native_log=false"
        )

        settings = manager.get_engine_settings(
            uri, settings={"backend": Backend.TRANSFORMERS}
        )

        self.assertEqual(settings.backend, Backend.DS4)
        self.assertEqual(
            settings.backend_config,
            {
                "ctx_size": 4096,
                "native_backend": "metal",
                "native_log": False,
            },
        )

    def test_get_engine_settings_ds4_explicit_config_overrides_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_ctx=2048"
        )

        settings = manager.get_engine_settings(
            uri,
            settings={
                "backend_config": {
                    "ctx_size": 4096,
                    "native_backend": "metal",
                },
            },
        )

        self.assertEqual(settings.backend, Backend.DS4)
        self.assertEqual(
            settings.backend_config,
            {"ctx_size": 4096, "native_backend": "metal"},
        )

    def test_load_ds4_explicit_config_overrides_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4"
            "&ds4_ctx=2048&ds4_native_backend=metal"
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(
                uri,
                backend_config={
                    "ctx_size": 4096,
                    "native_backend": "cuda",
                },
            )

        args = get_mock.call_args.args[1]
        self.assertEqual(
            args["backend_config"],
            {"ctx_size": 4096, "native_backend": "cuda"},
        )

    def test_load_ds4_uri_directional_steering_config(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4"
            "&ds4_directional_steering_file=steer.bin"
            "&ds4_directional_steering_attn=0.5"
            "&ds4_directional_steering_ffn=-0.25"
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)

        args = get_mock.call_args.args[1]
        self.assertEqual(
            args["backend_config"],
            {
                "directional_steering_file": "steer.bin",
                "directional_steering_attn": 0.5,
                "directional_steering_ffn": -0.25,
            },
        )

    def test_load_unknown_ds4_uri_key_raises(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_unknown=1"
        )

        with self.assertRaisesRegex(
            ValueError, "Unknown DS4 configuration key 'ds4_unknown'"
        ):
            manager.load(uri)

    def test_load_invalid_ds4_uri_values_raise(self):
        manager = ModelManager(self.hub, self.logger)
        cases = {
            "ds4_ctx=0": "a positive integer",
            "ds4_mtp_draft_tokens=-1": "a non-negative integer",
            "ds4_kv_disk_space_mb=-1": "a non-negative integer",
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                uri = manager.parse_uri(
                    f"ai://local/./model.gguf?backend=ds4&{query}"
                )
                with self.assertRaisesRegex(ValueError, expected):
                    manager.load(uri)

    def test_ds4_backend_config_from_mapping_rejects_invalid_values(self):
        cases = (
            ({"ds4_mtp_margin": "bad"}, "a number"),
            ({"ds4_mtp_margin": -0.1}, "a non-negative number"),
            ({"ds4_mtp_path": ""}, "a non-empty string"),
            ({"ds4_quality": 1}, "a boolean"),
            (
                {"ds4_native_backend": "rocm"},
                "one of auto, cpu, cuda, metal",
            ),
        )

        for mapping, expected in cases:
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(ValueError, expected):
                    ModelManager.ds4_backend_config_from_mapping(mapping)

    def test_private_ds4_config_helpers_cover_defensive_paths(self):
        self.assertEqual(
            model_manager._normalize_ds4_backend_config(
                {"ds4_unknown": 1, "ds4_ctx": 2048},
                reject_unknown=False,
                allow_normalized_keys=False,
            ),
            {"ctx_size": 2048},
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown DS4 backend configuration key 'unsupported'",
        ):
            model_manager._validate_ds4_config_value(
                "ds4_unsupported",
                "unsupported",
                1,
            )

    def test_load_non_ds4_ignores_ds4_config(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://local/model?backend=mlx&ds4_ctx=4096")
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)

        args = get_mock.call_args.args[1]
        self.assertEqual(args["backend"], Backend.MLXLM)
        self.assertIsNone(args["backend_config"])

    def test_load_engine_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        settings = TransformerEngineSettings()
        with self.assertRaises(NotImplementedError):
            manager.load_engine(uri, settings, "invalid")  # type: ignore[arg-type]

    def test_load_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        with self.assertRaises(NotImplementedError):
            manager.load(uri, modality="invalid")  # type: ignore[arg-type]

    def test_load_output_hidden_states(self):
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        manager = ModelManager(self.hub, self.logger)
        for modality in (Modality.TEXT_GENERATION, Modality.EMBEDDING):
            for value in (True, False, None):
                with self.subTest(modality=modality, value=value):
                    with (
                        patch.object(manager, "get_engine_settings") as ges,
                        patch.object(manager, "load_engine") as le,
                    ):
                        manager._stack.enter_context = MagicMock()
                        ges.return_value = TransformerEngineSettings()
                        le.return_value = "model"
                        result = manager.load(
                            uri, modality=modality, output_hidden_states=value
                        )
                    args = ges.call_args.args[1]
                    if value is None:
                        self.assertNotIn("output_hidden_states", args)
                    else:
                        self.assertEqual(args["output_hidden_states"], value)
                    le.assert_called_once_with(uri, ges.return_value, modality)
                    self.assertEqual(result, "model")

    def test_parse_uri_boolean_params(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://openai/gpt-4o?stream=true&debug=false")
        self.assertTrue(uri.params["stream"])
        self.assertFalse(uri.params["debug"])

    def test_async_context_manager(self):
        manager = ModelManager(self.hub, self.logger)

        async def run():
            async with manager as mm:
                self.assertIs(mm, manager)

        asyncio.run(run())


class ModelManagerEventDispatchTestCase(IsolatedAsyncioTestCase):
    async def test_triggers_events_and_closes_stack(self) -> None:
        hub = MagicMock(spec=HuggingfaceHub)
        hub.cache_dir = "cache"
        logger = MagicMock(spec=Logger)
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        manager = ModelManager(hub, logger, event_manager=event_manager)
        manager._stack.aclose = AsyncMock()  # type: ignore[assignment]

        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        )
        operation = Operation(
            generation_settings=GenerationSettings(),
            input="prompt",
            modality=Modality.TEXT_GENERATION,
            parameters=OperationParameters(
                text=OperationTextParameters(
                    system_prompt="system",
                    developer_prompt="dev",
                )
            ),
        )
        expected = object()

        async def handler(
            engine_uri_arg: EngineUri,
            model_arg: object,
            operation_arg: Operation,
            tool_arg: object,
        ) -> object:
            self.assertIs(engine_uri_arg, engine_uri)
            self.assertIs(operation_arg, operation)
            self.assertIsNone(tool_arg)
            return expected

        model = object()
        context = ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=operation.input,
        )
        task = ModelCall(
            engine_uri=engine_uri,
            model=model,
            operation=operation,
            tool=None,
            context=context,
        )
        with patch(
            "avalan.model.manager.ModalityRegistry.get",
            return_value=handler,
        ):
            result = await manager(task)

        self.assertIs(result, expected)
        self.assertEqual(event_manager.trigger.await_count, 2)
        before_event = event_manager.trigger.await_args_list[0].args[0]
        self.assertEqual(
            before_event.type, EventType.MODEL_MANAGER_CALL_BEFORE
        )
        self.assertEqual(
            before_event.payload["modality"], Modality.TEXT_GENERATION
        )
        self.assertIs(before_event.payload["task"], task)
        self.assertIs(before_event.payload["context"], context)
        after_event = event_manager.trigger.await_args_list[1].args[0]
        self.assertEqual(after_event.type, EventType.MODEL_MANAGER_CALL_AFTER)
        self.assertIs(after_event.payload["result"], expected)
        self.assertIs(after_event.payload["task"], task)
        self.assertIs(after_event.payload["context"], context)
        self.assertIsNotNone(after_event.started)
        self.assertIsNotNone(after_event.finished)
        self.assertIsNotNone(after_event.elapsed)

        manager.__exit__(None, None, None)
        await asyncio.sleep(0)
        manager._stack.aclose.assert_awaited_once()

    async def test_exit_with_running_loop_closes_on_interrupt(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        manager = ModelManager(hub, logger)
        manager._stack = AsyncMock()

        manager.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)
        await asyncio.sleep(0)

        manager._stack.aclose.assert_awaited_once()

    async def test_aexit_awaits_pending_close_on_interrupt(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        manager = ModelManager(hub, logger)
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        async def close_task() -> None:
            close_started.set()
            await release_close.wait()

        pending = asyncio.create_task(close_task())
        manager._pending_exit_task = pending

        exit_task = asyncio.create_task(
            manager.__aexit__(KeyboardInterrupt, KeyboardInterrupt(), None)
        )

        await close_started.wait()
        await asyncio.sleep(0)
        self.assertFalse(exit_task.done())

        release_close.set()
        result = await exit_task

        self.assertFalse(result)
        self.assertFalse(pending.cancelled())
        self.assertIsNone(manager._pending_exit_task)


def _container_runtime_settings(
    *,
    surface: ContainerSurface = ContainerSurface.MODEL_BACKEND,
    scope: ContainerExecutionScope = ContainerExecutionScope.RUNTIME_ENVELOPE,
    trust_level: ContainerTrustLevel = ContainerTrustLevel.TRUSTED_OPERATOR,
    image: ContainerImagePolicy | None = None,
    mounts: tuple[ContainerMountDeclaration, ...] | None = None,
    secrets: tuple[ContainerSecretReference, ...] = (),
    network: ContainerNetworkPolicy | None = None,
    devices: tuple[ContainerDeviceClass, ...] = (),
    resources: ContainerResourceLimits | None = None,
    output: ContainerOutputPolicy | None = None,
    cleanup: ContainerCleanupPolicy | None = None,
    audit: ContainerAuditPolicy | None = None,
    allowed_profiles: tuple[str, ...] | None = None,
    required: bool = True,
) -> ContainerToolRuntimeSettings:
    source = ContainerSettingsSource(
        surface=surface,
        trust_level=trust_level,
    )
    profile = ContainerProfile(
        name="model-runtime",
        image=image
        or ContainerImagePolicy(
            reference=(
                "registry.example/model@sha256:"
                "1111111111111111111111111111111111111111111111111111111111111111"
            ),
        ),
        mounts=mounts if mounts is not None else (_workspace_mount(),),
        secrets=secrets,
        network=network or ContainerNetworkPolicy(),
        devices=ContainerDevicePolicy(devices=devices),
        resources=resources
        or ContainerResourceLimits(
            cpu_count=2,
            memory_bytes=2 * 1024 * 1024,
            timeout_seconds=60,
        ),
        output=output or ContainerOutputPolicy(),
        cleanup=cleanup or ContainerCleanupPolicy(),
        audit=audit or ContainerAuditPolicy(mode=ContainerAuditMode.FULL),
    )
    return ContainerToolRuntimeSettings(
        effective_settings=ContainerEffectiveSettings(
            backend=ContainerBackend.DOCKER,
            required=required,
            scope=scope,
            source=source,
            policy_version="phase16",
            profile_registry_id="model",
            profile_name=profile.name,
            profile=profile,
            allowed_profiles=(
                (profile.name,)
                if allowed_profiles is None
                else allowed_profiles
            ),
        )
    )


def _workspace_mount() -> ContainerMountDeclaration:
    return ContainerMountDeclaration(
        source=".",
        target="/workspace",
        mount_type=ContainerMountType.WORKSPACE,
    )


def _local_uri(model_id: str) -> EngineUri:
    return EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id=model_id,
        params={},
    )


def _model_backend_plan(
    hub: MagicMock,
    logger: MagicMock,
):
    manager = ModelManager(
        hub,
        logger,
        container_runtime=_container_runtime_settings(),
    )
    plan = manager.model_backend_runtime_envelope_plan(
        _local_uri("local-model"),
        TransformerEngineSettings(),
    )
    assert plan is not None
    return plan


def _policy_codes(
    error: ModelBackendRuntimePolicyError,
) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in error.diagnostics)
