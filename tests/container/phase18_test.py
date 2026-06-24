from asyncio import CancelledError, create_task, gather, sleep
from asyncio import run as run_async
from unittest import TestCase, main

from avalan.container import (
    ContainerAuthorityCaps,
    ContainerAuthorizationDecisionType,
    ContainerBackend,
    ContainerBackendContainer,
    ContainerBackendImageResolution,
    ContainerBackendOperation,
    ContainerBackendOperationResult,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBuildCache,
    ContainerBuildCachePolicy,
    ContainerBuildCacheResult,
    ContainerBuildContextPolicy,
    ContainerBuildPolicy,
    ContainerCacheLookupStatus,
    ContainerCacheMode,
    ContainerCommandPlan,
    ContainerEffectiveSettings,
    ContainerEscalationMode,
    ContainerEscalationPolicy,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerImageCache,
    ContainerImageCachePolicy,
    ContainerImagePolicy,
    ContainerImageTrustPolicy,
    ContainerLifecycleResources,
    ContainerManagedLifecycleResult,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerPolicy,
    ContainerPolicyContext,
    ContainerPolicyPlan,
    ContainerPool,
    ContainerPoolDecisionType,
    ContainerPoolingMode,
    ContainerPoolingPolicy,
    ContainerPoolSafetyReport,
    ContainerPoolTeardownMode,
    ContainerProfile,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerReviewSurface,
    ContainerRunPlan,
    ContainerSecretReference,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerStaticPoolSafetyChecker,
    ContainerSurface,
    ContainerTrustLevel,
    ContainerVulnerabilityPolicy,
    ContainerVulnerabilitySeverity,
    container_backend_capability_profile,
    container_build_cache_key,
    container_image_cache_key,
    container_pool_key,
    run_container_managed_lifecycle,
)

_DIGEST = "a" * 64
_DIGEST_ALT = "b" * 64
_CONTEXT_DIGEST = "c" * 64
_IMAGE = f"ghcr.io/example/phase18@sha256:{_DIGEST}"


class ContainerPhase18Test(TestCase):
    def test_image_trust_allows_pinned_allowed_verified_image(self) -> None:
        digest = f"sha256:{_DIGEST}"
        trust = ContainerImageTrustPolicy(
            allowed_registries=("ghcr.io",),
            digest_pins={"ghcr.io/example/phase18": digest},
            require_signature_verification=True,
            verified_digests=(digest,),
            require_attestations=True,
            attestation_references={digest: "attestation://phase18"},
            require_sbom=True,
            sbom_references={digest: "sbom://phase18"},
            vulnerability_policy=ContainerVulnerabilityPolicy.DENY_HIGH_OR_CRITICAL,
            vulnerability_findings={
                digest: ContainerVulnerabilitySeverity.LOW,
            },
            allowed_platforms=("linux/amd64",),
            production=True,
        )
        policy = ContainerPolicy(policy_version="phase18", image_trust=trust)

        decision = policy.authorize(
            _policy_plan(surface=ContainerReviewSurface.SERVER)
        )

        self.assertEqual(
            decision.decision,
            ContainerAuthorizationDecisionType.ALLOW,
        )
        self.assertEqual(
            decision.code,
            "container.allow.preauthorized_profile",
        )
        self.assertTrue(decision.cacheable)
        self.assertEqual(
            trust.to_dict()["sbom_references"],
            {digest: "sbom://phase18"},
        )
        docker_hub = ContainerPolicy(
            policy_version="phase18",
            image_trust=ContainerImageTrustPolicy(
                allowed_registries=("docker.io",)
            ),
        ).authorize(
            _policy_plan(
                profile=_profile(
                    image=ContainerImagePolicy(
                        reference=f"library/phase18@sha256:{_DIGEST}",
                    )
                ),
                surface=ContainerReviewSurface.SERVER,
            )
        )
        self.assertEqual(
            docker_hub.decision,
            ContainerAuthorizationDecisionType.ALLOW,
        )

    def test_image_trust_denies_untrusted_provenance(self) -> None:
        digest = f"sha256:{_DIGEST}"
        cases = (
            (
                ContainerImageTrustPolicy(
                    allowed_registries=("registry.test",)
                ),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image registry is not trusted by policy.",
            ),
            (
                ContainerImageTrustPolicy(production=True),
                _profile(
                    image=ContainerImagePolicy(
                        reference="ghcr.io/example/phase18:latest",
                        digest=digest,
                    )
                ),
                ContainerReviewSurface.INTERACTIVE_CLI,
                (
                    "Production container images must use digest-pinned"
                    " references."
                ),
            ),
            (
                ContainerImageTrustPolicy(),
                _profile(
                    image=ContainerImagePolicy(
                        reference="ghcr.io/example/phase18:latest",
                        digest=digest,
                    )
                ),
                ContainerReviewSurface.SERVER,
                (
                    "Noninteractive container images must use digest-pinned"
                    " references."
                ),
            ),
            (
                ContainerImageTrustPolicy(
                    digest_pins={
                        "ghcr.io/example/phase18": f"sha256:{_DIGEST_ALT}",
                    }
                ),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image digest does not match trusted pin.",
            ),
            (
                ContainerImageTrustPolicy(allowed_platforms=("linux/arm64",)),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image platform is not trusted by policy.",
            ),
            (
                ContainerImageTrustPolicy(
                    require_signature_verification=True,
                    verified_digests=(),
                ),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image signature verification failed.",
            ),
            (
                ContainerImageTrustPolicy(require_attestations=True),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image attestation is missing.",
            ),
            (
                ContainerImageTrustPolicy(require_sbom=True),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image SBOM reference is missing.",
            ),
            (
                ContainerImageTrustPolicy(
                    vulnerability_policy=(
                        ContainerVulnerabilityPolicy.DENY_HIGH_OR_CRITICAL
                    ),
                    vulnerability_findings={
                        digest: ContainerVulnerabilitySeverity.HIGH,
                    },
                ),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image vulnerability policy denied execution.",
            ),
            (
                ContainerImageTrustPolicy(
                    vulnerability_policy=(
                        ContainerVulnerabilityPolicy.DENY_CRITICAL
                    ),
                ),
                _profile(),
                ContainerReviewSurface.SERVER,
                "Container image vulnerability scan is missing.",
            ),
        )

        for trust, profile, surface, explanation in cases:
            with self.subTest(explanation=explanation):
                decision = ContainerPolicy(
                    policy_version="phase18",
                    image_trust=trust,
                ).authorize(_policy_plan(profile=profile, surface=surface))

                self.assertEqual(
                    decision.decision,
                    ContainerAuthorizationDecisionType.DENY,
                )
                self.assertEqual(decision.code, "container.deny.image_trust")
                self.assertEqual(decision.explanation, explanation)

    def test_build_and_cache_value_objects_serialize(self) -> None:
        build_context = ContainerBuildContextPolicy.from_dict(
            {
                "context_path": "oci://trusted/context",
                "context_digest": f"sha256:{_CONTEXT_DIGEST}",
                "context_size_bytes": 1024,
                "allow_remote_context": True,
            }
        )
        image_cache = ContainerImageCachePolicy(
            mode=ContainerCacheMode.READ_WRITE,
            ttl_seconds=30,
        )
        build_cache = ContainerBuildCachePolicy(
            mode=ContainerCacheMode.READ_WRITE,
            ttl_seconds=30,
        )
        image = ContainerImagePolicy.from_dict(
            {
                "reference": _IMAGE,
                "build_policy": "trusted_only",
                "build_context": build_context.to_dict(),
                "image_cache": image_cache.to_dict(),
                "build_cache": build_cache.to_dict(),
            }
        )

        self.assertTrue(image_cache.enabled)
        self.assertTrue(build_cache.enabled)
        self.assertEqual(image.build_context, build_context)
        assert image.build_context is not None
        self.assertEqual(
            image.build_context.to_dict()["context_path"],
            "oci://trusted/context",
        )
        with self.assertRaises(AssertionError):
            ContainerBuildContextPolicy.from_dict(
                {
                    "context_path": ".",
                    "context_digest": f"sha256:{_CONTEXT_DIGEST}",
                    "context_size_bytes": "1024",
                }
            )

    def test_trusted_build_context_and_progress_are_supported(self) -> None:
        plan = _run_plan(image=_build_image())
        backend = _fake_backend(
            build_progress_chunks=(
                ContainerBackendStreamChunk(
                    stream=ContainerBackendStream.PROGRESS,
                    content=b"step 1/2",
                    sequence=0,
                ),
                ContainerBackendStreamChunk(
                    stream=ContainerBackendStream.PROGRESS,
                    content=b"step 2/2",
                    sequence=1,
                ),
            )
        )

        result = run_async(
            backend.run(
                plan, lifecycle_resources=ContainerLifecycleResources()
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertIn(
            ContainerBackendOperation.IMAGE_BUILD, backend.operations
        )
        self.assertEqual(
            [chunk.to_dict()["content"] for chunk in backend.build_progress],
            ["step 1/2", "step 2/2"],
        )
        self.assertIn(_CONTEXT_DIGEST, container_build_cache_key(plan))

    def test_build_context_denies_remote_secrets_network_and_size(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            ContainerBuildContextPolicy(
                context_path="https://example.test/context.tar",
                context_digest=f"sha256:{_CONTEXT_DIGEST}",
                context_size_bytes=1024,
            )
        with self.assertRaises(AssertionError):
            ContainerBuildContextPolicy(
                context_path=".",
                context_digest=f"sha256:{_CONTEXT_DIGEST}",
                context_size_bytes=1024,
                secret_names=("TOKEN",),
            )
        with self.assertRaises(AssertionError):
            ContainerBuildContextPolicy(
                context_path=".",
                context_digest=f"sha256:{_CONTEXT_DIGEST}",
                context_size_bytes=1024,
                network=ContainerNetworkMode.LOOPBACK,
            )
        with self.assertRaises(AssertionError):
            ContainerBuildContextPolicy(
                context_path=".",
                context_digest=f"sha256:{_CONTEXT_DIGEST}",
                context_size_bytes=2048,
                max_context_bytes=1024,
            )

    def test_image_cache_hits_and_stale_keys_do_not_reuse_mutable_tags(
        self,
    ) -> None:
        cache = ContainerImageCache()
        first = _run_plan(
            image=ContainerImagePolicy(
                reference="ghcr.io/example/cache:stable",
                digest=f"sha256:{_DIGEST}",
                image_cache=ContainerImageCachePolicy(
                    mode=ContainerCacheMode.READ_WRITE,
                    ttl_seconds=10,
                ),
            )
        )
        second = _run_plan(
            image=ContainerImagePolicy(
                reference="ghcr.io/example/cache:stable",
                digest=f"sha256:{_DIGEST_ALT}",
                image_cache=ContainerImageCachePolicy(
                    mode=ContainerCacheMode.READ_WRITE,
                    ttl_seconds=10,
                ),
            )
        )

        self.assertEqual(
            cache.lookup(first, now_seconds=1).status,
            ContainerCacheLookupStatus.MISS,
        )
        self.assertEqual(
            cache.store(first, now_seconds=1).status,
            ContainerCacheLookupStatus.HIT,
        )
        self.assertTrue(cache.lookup(first, now_seconds=2).hit)
        self.assertEqual(
            cache.lookup(second, now_seconds=2).status,
            ContainerCacheLookupStatus.MISS,
        )
        self.assertNotEqual(
            container_image_cache_key(first),
            container_image_cache_key(second),
        )
        self.assertEqual(
            cache.lookup(first, now_seconds=12).status,
            ContainerCacheLookupStatus.STALE,
        )

    def test_build_cache_deduplicates_concurrent_builds(self) -> None:
        backend = _fake_backend(delay_build=True)
        cache = ContainerBuildCache()
        plan = _run_plan(image=_build_image())

        first, second = run_async(_build_twice(cache, backend, plan))
        cached = run_async(cache.get_or_build(backend, plan, now_seconds=2))
        stale = run_async(cache.get_or_build(backend, plan, now_seconds=32))

        self.assertEqual(
            backend.operations.count(ContainerBackendOperation.IMAGE_BUILD),
            2,
        )
        self.assertFalse(first.deduplicated)
        self.assertTrue(second.deduplicated)
        self.assertEqual(
            cached.cache.status,
            ContainerCacheLookupStatus.HIT,
        )
        self.assertEqual(
            stale.cache.status,
            ContainerCacheLookupStatus.STALE,
        )
        self.assertIn("operation", cached.to_dict())

    def test_failed_build_does_not_poison_inflight_cache(self) -> None:
        backend = _flaky_build_backend()
        cache = ContainerBuildCache()
        plan = _run_plan(image=_build_image())

        with self.assertRaises(RuntimeError):
            run_async(cache.get_or_build(backend, plan, now_seconds=1))
        retry = run_async(cache.get_or_build(backend, plan, now_seconds=2))
        cached = run_async(cache.get_or_build(backend, plan, now_seconds=3))

        self.assertTrue(retry.operation.ok)
        self.assertEqual(
            retry.cache.status,
            ContainerCacheLookupStatus.MISS,
        )
        self.assertEqual(
            cached.cache.status,
            ContainerCacheLookupStatus.HIT,
        )
        self.assertEqual(backend.build_attempts, 2)

    def test_disabled_cache_and_racing_cached_build_paths(self) -> None:
        image_cache = ContainerImageCache()
        disabled_plan = _run_plan()
        build_cache = ContainerBuildCache()
        disabled_build_plan = _run_plan(
            image=ContainerImagePolicy(
                reference=_IMAGE,
                build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
            )
        )
        racing_cache = ContainerBuildCache()
        racing_plan = _run_plan(image=_build_image())
        stale_racing_cache = ContainerBuildCache()
        stale_racing_plan = _run_plan(image=_build_image())
        operation = ContainerBackendOperationResult(
            operation=ContainerBackendOperation.IMAGE_BUILD,
        )

        class CachedBuildEntry:
            def __init__(
                self,
                result: ContainerBackendOperationResult,
            ) -> None:
                self.result = result

        calls = 0

        def fake_lookup(
            policy: ContainerBuildCachePolicy,
            cache_key: str,
            now_seconds: int,
        ) -> tuple[CachedBuildEntry | None, ContainerCacheLookupStatus]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return None, ContainerCacheLookupStatus.MISS
            return CachedBuildEntry(operation), ContainerCacheLookupStatus.HIT

        setattr(racing_cache, "_lookup_build_entry", fake_lookup)

        stale_calls = 0

        def fake_stale_lookup(
            policy: ContainerBuildCachePolicy,
            cache_key: str,
            now_seconds: int,
        ) -> tuple[None, ContainerCacheLookupStatus]:
            nonlocal stale_calls
            stale_calls += 1
            if stale_calls == 1:
                return None, ContainerCacheLookupStatus.MISS
            return None, ContainerCacheLookupStatus.STALE

        setattr(stale_racing_cache, "_lookup_build_entry", fake_stale_lookup)

        disabled_lookup = image_cache.lookup(disabled_plan, now_seconds=0)
        disabled_store = image_cache.store(disabled_plan, now_seconds=0)
        disabled_build = run_async(
            build_cache.get_or_build(
                _fake_backend(),
                disabled_build_plan,
                now_seconds=0,
            )
        )
        racing_build = run_async(
            racing_cache.get_or_build(
                _fake_backend(),
                racing_plan,
                now_seconds=0,
            )
        )
        stale_racing_build = run_async(
            stale_racing_cache.get_or_build(
                _fake_backend(),
                stale_racing_plan,
                now_seconds=0,
            )
        )

        self.assertEqual(
            disabled_lookup.status,
            ContainerCacheLookupStatus.DISABLED,
        )
        self.assertEqual(
            disabled_lookup.to_dict()["status"],
            "disabled",
        )
        self.assertEqual(
            disabled_store.status,
            ContainerCacheLookupStatus.DISABLED,
        )
        self.assertEqual(
            disabled_build.cache.status,
            ContainerCacheLookupStatus.DISABLED,
        )
        self.assertEqual(
            racing_build.cache.status,
            ContainerCacheLookupStatus.HIT,
        )
        self.assertEqual(
            stale_racing_build.cache.status,
            ContainerCacheLookupStatus.STALE,
        )

    def test_pool_reuse_health_and_eviction_are_enforced(self) -> None:
        plan = _run_plan()
        container = _container("pool-1")
        pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_age_seconds=10,
                max_uses=1,
                idle_ttl_seconds=5,
            )
        )
        service_pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SERVICE,
                max_age_seconds=10,
                max_uses=2,
                idle_ttl_seconds=5,
                health_check_command=("curl", "-f", "http://127.0.0.1"),
                teardown=ContainerPoolTeardownMode.RESET,
                audit_labels={"service": "phase18"},
            )
        )

        pool.offer(plan, container, ContainerPoolSafetyReport(), now_seconds=0)
        reused = pool.acquire(plan, now_seconds=1)
        evicted = pool.acquire(plan, now_seconds=2)
        service_pool.offer(
            plan,
            _container("service-1"),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        service_reuse = service_pool.acquire(plan, now_seconds=1)
        unhealthy = service_pool.offer(
            plan,
            _container("service-2"),
            ContainerPoolSafetyReport(healthy=False),
            now_seconds=2,
        )

        self.assertEqual(reused.decision, ContainerPoolDecisionType.REUSE)
        self.assertTrue(reused.reuse)
        self.assertEqual(reused.to_dict()["decision"], "reuse")
        self.assertEqual(evicted.decision, ContainerPoolDecisionType.EVICT)
        self.assertEqual(
            service_reuse.audit_labels,
            {"service": "phase18"},
        )
        self.assertEqual(
            service_reuse.decision,
            ContainerPoolDecisionType.REUSE,
        )
        self.assertEqual(
            unhealthy.decision,
            ContainerPoolDecisionType.REJECT,
        )

    def test_pool_key_includes_argv_and_mounts(self) -> None:
        plan = _run_plan()
        argv_plan = _run_plan(argv=("echo", "changed"))
        mount_plan = _run_plan(
            mounts=(
                ContainerMountDeclaration(
                    target="/scratch-alt",
                    mount_type=ContainerMountType.SCRATCH,
                    access=ContainerMountAccess.WRITE,
                ),
            )
        )
        pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_uses=3,
            )
        )

        self.assertNotEqual(
            container_pool_key(plan), container_pool_key(argv_plan)
        )
        self.assertNotEqual(
            container_pool_key(plan),
            container_pool_key(mount_plan),
        )
        pool.offer(
            plan,
            _container("base", plan),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )

        self.assertEqual(
            pool.acquire(argv_plan, now_seconds=1).decision,
            ContainerPoolDecisionType.CREATE,
        )
        self.assertEqual(
            pool.acquire(mount_plan, now_seconds=1).decision,
            ContainerPoolDecisionType.CREATE,
        )
        self.assertEqual(
            pool.acquire(plan, now_seconds=1).decision,
            ContainerPoolDecisionType.REUSE,
        )
        self.assertEqual(
            pool.offer(
                argv_plan,
                _container("wrong-fingerprint", plan),
                ContainerPoolSafetyReport(),
                now_seconds=2,
            ).reason,
            "container plan fingerprint does not match pool key",
        )
        self.assertTrue(pool.discard(plan))
        self.assertFalse(pool.discard(plan))

    def test_pool_rejects_contamination_and_secret_reuse_by_default(
        self,
    ) -> None:
        pool = ContainerPool(
            ContainerPoolingPolicy(mode=ContainerPoolingMode.SHORT_LIVED)
        )
        plan = _run_plan()
        secret_plan = _run_plan(secret_names=("TOKEN",))
        cases = (
            ContainerPoolSafetyReport(contaminated=True),
            ContainerPoolSafetyReport(leftover_processes=1),
            ContainerPoolSafetyReport(dirty_scratch=True),
        )

        for safety in cases:
            with self.subTest(safety=safety.to_dict()):
                self.assertEqual(
                    pool.offer(
                        plan,
                        _container("bad"),
                        safety,
                        now_seconds=0,
                    ).decision,
                    ContainerPoolDecisionType.REJECT,
                )
        self.assertEqual(
            pool.offer(
                secret_plan,
                _container("secret"),
                ContainerPoolSafetyReport(),
                now_seconds=0,
            ).decision,
            ContainerPoolDecisionType.REJECT,
        )
        self.assertEqual(
            pool.acquire(secret_plan, now_seconds=0).decision,
            ContainerPoolDecisionType.REJECT,
        )

    def test_pool_disabled_empty_age_idle_and_safety_evictions(self) -> None:
        plan = _run_plan()
        disabled = ContainerPool(ContainerPoolingPolicy())
        empty = ContainerPool(
            ContainerPoolingPolicy(mode=ContainerPoolingMode.SHORT_LIVED)
        )
        age_pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_age_seconds=1,
                max_uses=10,
                idle_ttl_seconds=10,
            )
        )
        idle_pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_age_seconds=10,
                max_uses=10,
                idle_ttl_seconds=1,
            )
        )
        safety_pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_age_seconds=10,
                max_uses=10,
                idle_ttl_seconds=10,
            )
        )

        self.assertEqual(
            disabled.offer(
                plan,
                _container("disabled"),
                ContainerPoolSafetyReport(),
                now_seconds=0,
            ).decision,
            ContainerPoolDecisionType.REJECT,
        )
        self.assertEqual(
            disabled.acquire(plan, now_seconds=0).decision,
            ContainerPoolDecisionType.CREATE,
        )
        self.assertEqual(
            empty.acquire(plan, now_seconds=0).decision,
            ContainerPoolDecisionType.CREATE,
        )
        age_pool.offer(
            plan,
            _container("age"),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        idle_pool.offer(
            plan,
            _container("idle"),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        safety_pool.offer(
            plan,
            _container("safety"),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        safety_entry = next(iter(getattr(safety_pool, "_entries").values()))
        object.__setattr__(
            safety_entry,
            "safety",
            ContainerPoolSafetyReport(contaminated=True),
        )

        self.assertEqual(
            age_pool.acquire(plan, now_seconds=2).reason,
            "pooled container exceeded max age",
        )
        self.assertEqual(
            idle_pool.acquire(plan, now_seconds=2).reason,
            "pooled container exceeded idle ttl",
        )
        self.assertEqual(
            safety_pool.acquire(plan, now_seconds=1).reason,
            "pooled container is contaminated",
        )

    def test_secret_pool_reuse_requires_explicit_combined_risk_policy(
        self,
    ) -> None:
        denied_profile = _profile(
            secrets=(
                ContainerSecretReference(name="token", env_name="TOKEN"),
            ),
            pooling=ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
            ),
        )
        allowed_profile = _profile(
            secrets=(
                ContainerSecretReference(name="token", env_name="TOKEN"),
            ),
            pooling=ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                allow_secret_reuse=True,
            ),
        )

        denied = ContainerPolicy(policy_version="phase18").authorize(
            _policy_plan(profile=denied_profile)
        )
        allowed = ContainerPolicy(policy_version="phase18").authorize(
            _policy_plan(
                escalation=ContainerEscalationMode.PREAUTHORIZED,
                profile=allowed_profile,
            )
        )
        network_pool = ContainerPolicy(policy_version="phase18").authorize(
            _policy_plan(
                escalation=ContainerEscalationMode.PREAUTHORIZED,
                profile=_profile(
                    pooling=ContainerPoolingPolicy(
                        mode=ContainerPoolingMode.SHORT_LIVED,
                    ),
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.LOOPBACK,
                    ),
                ),
            )
        )

        self.assertEqual(
            denied.decision,
            ContainerAuthorizationDecisionType.DENY,
        )
        self.assertEqual(denied.code, "container.deny.secret_reuse")
        self.assertEqual(
            allowed.decision,
            ContainerAuthorizationDecisionType.ALLOW,
        )
        self.assertEqual(
            allowed.code,
            "container.allow.preauthorized_escalation",
        )
        self.assertEqual(
            network_pool.decision,
            ContainerAuthorizationDecisionType.REQUIRES_REVIEW,
        )

    def test_fake_e2e_digest_build_cache_and_pool_paths(self) -> None:
        run_result = run_async(_fake_backend().run(_run_plan()))
        build_backend = _fake_backend()
        build_result = run_async(
            build_backend.run(_run_plan(image=_build_image()))
        )
        cache = ContainerImageCache()
        plan = _run_plan(
            image=ContainerImagePolicy(
                reference=_IMAGE,
                image_cache=ContainerImageCachePolicy(
                    mode=ContainerCacheMode.READ_WRITE,
                    ttl_seconds=30,
                ),
            )
        )
        pool = ContainerPool(
            ContainerPoolingPolicy(mode=ContainerPoolingMode.SHORT_LIVED)
        )

        cache.store(plan, now_seconds=0)
        clean_offer = pool.offer(
            plan,
            _container("clean", plan),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        clean_reuse = pool.acquire(plan, now_seconds=1)
        contaminated = pool.offer(
            plan,
            _container("dirty", plan),
            ContainerPoolSafetyReport(contaminated=True),
            now_seconds=2,
        )

        self.assertEqual(
            run_result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(
            build_result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertIn(
            ContainerBackendOperation.IMAGE_BUILD,
            build_backend.operations,
        )
        self.assertTrue(cache.lookup(plan, now_seconds=1).hit)
        self.assertEqual(
            clean_offer.decision,
            ContainerPoolDecisionType.CREATE,
        )
        self.assertEqual(
            clean_reuse.decision,
            ContainerPoolDecisionType.REUSE,
        )
        self.assertEqual(
            contaminated.decision,
            ContainerPoolDecisionType.REJECT,
        )

    def test_managed_lifecycle_rejects_resolved_digest_mismatch(self) -> None:
        backend = _fake_backend(resolved_digest=f"sha256:{_DIGEST_ALT}")
        plan = _run_plan(
            image=ContainerImagePolicy(
                reference=_IMAGE,
                pull_policy=ContainerPullPolicy.IF_MISSING,
            )
        )

        result = run_async(run_container_managed_lifecycle(backend, plan))

        self.assertEqual(result.execution.status, ContainerResultStatus.DENIED)
        self.assertIn(
            "resolved image digest does not match approved plan",
            result.diagnostics[0].message,
        )
        self.assertIn(
            ContainerBackendOperation.IMAGE_RESOLUTION,
            backend.operations,
        )
        self.assertNotIn(
            ContainerBackendOperation.IMAGE_PULL, backend.operations
        )
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    def test_managed_lifecycle_rejects_resolved_platform_mismatch(
        self,
    ) -> None:
        backend = _platform_mismatch_backend()

        result = run_async(
            run_container_managed_lifecycle(backend, _run_plan())
        )

        self.assertEqual(result.execution.status, ContainerResultStatus.DENIED)
        self.assertIn(
            "resolved image platform does not match approved plan",
            result.diagnostics[0].message,
        )
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    def test_managed_lifecycle_uses_image_and_build_caches(self) -> None:
        backend = _fake_backend()
        image_cache = ContainerImageCache()
        build_cache = ContainerBuildCache()
        pull_plan = _run_plan(
            image=ContainerImagePolicy(
                reference=_IMAGE,
                pull_policy=ContainerPullPolicy.IF_MISSING,
                image_cache=ContainerImageCachePolicy(
                    mode=ContainerCacheMode.READ_WRITE,
                    ttl_seconds=30,
                ),
            )
        )
        build_plan = _run_plan(image=_build_image())

        first_pull = run_async(
            run_container_managed_lifecycle(
                backend,
                pull_plan,
                image_cache=image_cache,
            )
        )
        second_pull = run_async(
            run_container_managed_lifecycle(
                backend,
                pull_plan,
                image_cache=image_cache,
            )
        )
        first_build = run_async(
            run_container_managed_lifecycle(
                backend,
                build_plan,
                build_cache=build_cache,
                now_seconds=3,
            )
        )
        second_build = run_async(
            run_container_managed_lifecycle(
                backend,
                build_plan,
                build_cache=build_cache,
                now_seconds=4,
            )
        )

        self.assertEqual(
            backend.operations.count(ContainerBackendOperation.IMAGE_PULL),
            1,
        )
        self.assertEqual(
            backend.operations.count(ContainerBackendOperation.IMAGE_BUILD),
            1,
        )
        self.assertEqual(
            first_pull.execution.metadata["image_cache_status"],
            "miss",
        )
        self.assertEqual(
            second_pull.execution.metadata["image_cache_status"],
            "hit",
        )
        self.assertEqual(
            first_build.execution.metadata["build_cache_status"],
            "miss",
        )
        self.assertEqual(
            second_build.execution.metadata["build_cache_status"],
            "hit",
        )

    def test_public_backend_run_uses_lifecycle_cache_resources(self) -> None:
        backend = _fake_backend()
        resources = ContainerLifecycleResources()
        pull_plan = _run_plan(
            image=ContainerImagePolicy(
                reference=_IMAGE,
                pull_policy=ContainerPullPolicy.IF_MISSING,
                image_cache=ContainerImageCachePolicy(
                    mode=ContainerCacheMode.READ_WRITE,
                    ttl_seconds=30,
                ),
            )
        )
        build_plan = _run_plan(image=_build_image())

        first_pull = run_async(
            backend.run(pull_plan, lifecycle_resources=resources)
        )
        second_pull = run_async(
            backend.run(pull_plan, lifecycle_resources=resources)
        )
        first_build = run_async(
            backend.run(build_plan, lifecycle_resources=resources)
        )
        second_build = run_async(
            backend.run(build_plan, lifecycle_resources=resources)
        )

        self.assertEqual(
            first_pull.execution.metadata["image_cache_status"],
            "miss",
        )
        self.assertEqual(
            second_pull.execution.metadata["image_cache_status"],
            "hit",
        )
        self.assertEqual(
            first_build.execution.metadata["build_cache_status"],
            "miss",
        )
        self.assertEqual(
            second_build.execution.metadata["build_cache_status"],
            "hit",
        )
        self.assertEqual(
            backend.operations.count(ContainerBackendOperation.IMAGE_PULL),
            1,
        )
        self.assertEqual(
            backend.operations.count(ContainerBackendOperation.IMAGE_BUILD),
            1,
        )

    def test_public_backend_run_denies_build_cache_without_context(
        self,
    ) -> None:
        backend = _fake_backend()
        plan = _run_plan(
            image=ContainerImagePolicy(
                reference=_IMAGE,
                build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
                build_cache=ContainerBuildCachePolicy(
                    mode=ContainerCacheMode.READ_WRITE,
                    ttl_seconds=30,
                ),
            )
        )

        result = run_async(backend.run(plan))

        self.assertEqual(result.execution.status, ContainerResultStatus.DENIED)
        self.assertIn(
            "container build cache requires a trusted build context",
            result.diagnostics[0].message,
        )
        self.assertNotIn(
            ContainerBackendOperation.IMAGE_BUILD,
            backend.operations,
        )

    def test_lifecycle_resources_accept_explicit_state(self) -> None:
        image_cache = ContainerImageCache()
        build_cache = ContainerBuildCache()
        checker = ContainerStaticPoolSafetyChecker()
        resources = ContainerLifecycleResources(
            image_cache=image_cache,
            build_cache=build_cache,
            pool_safety_checker=checker,
        )
        plan = _run_plan(pooling=_short_lived_pooling())

        self.assertIs(resources.image_cache, image_cache)
        self.assertIs(resources.build_cache, build_cache)
        self.assertIs(resources.pool_safety_checker, checker)
        self.assertIs(resources.pool_for(plan), resources.pool_for(plan))

    def test_public_backend_run_rejects_pooling_without_checker(self) -> None:
        backend = _fake_backend()
        plan = _run_plan(pooling=_short_lived_pooling())

        result = run_async(backend.run(plan))

        self.assertEqual(result.execution.status, ContainerResultStatus.DENIED)
        self.assertIn(
            "container pooling requires an explicit safety checker",
            result.diagnostics[0].message,
        )
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    def test_managed_lifecycle_uses_pool_and_rejects_secret_reuse(
        self,
    ) -> None:
        pooled_backend = _fake_backend()
        pooled_plan = _run_plan(pooling=_short_lived_pooling(max_uses=3))
        pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_uses=3,
            )
        )
        secret_backend = _fake_backend()
        secret_plan = _run_plan(
            pooling=_short_lived_pooling(),
            secret_names=("TOKEN",),
        )
        secret_pool = ContainerPool(
            ContainerPoolingPolicy(mode=ContainerPoolingMode.SHORT_LIVED)
        )
        safety_checker = ContainerStaticPoolSafetyChecker()

        first = run_async(
            run_container_managed_lifecycle(
                pooled_backend,
                pooled_plan,
                pool=pool,
                pool_safety_checker=safety_checker,
                now_seconds=1,
            )
        )
        second = run_async(
            run_container_managed_lifecycle(
                pooled_backend,
                pooled_plan,
                pool=pool,
                pool_safety_checker=safety_checker,
                now_seconds=2,
            )
        )
        first_secret = run_async(
            run_container_managed_lifecycle(
                secret_backend,
                secret_plan,
                pool=secret_pool,
                pool_safety_checker=safety_checker,
                now_seconds=1,
            )
        )
        second_secret = run_async(
            run_container_managed_lifecycle(
                secret_backend,
                secret_plan,
                pool=secret_pool,
                pool_safety_checker=safety_checker,
                now_seconds=2,
            )
        )

        self.assertEqual(
            first.execution.status, ContainerResultStatus.COMPLETED
        )
        self.assertEqual(
            first.execution.metadata["pool_offer_decision"],
            "create",
        )
        self.assertEqual(
            second.execution.metadata["pool_decision"],
            "reuse",
        )
        self.assertEqual(
            pooled_backend.operations.count(ContainerBackendOperation.CREATE),
            1,
        )
        self.assertNotIn(
            ContainerBackendOperation.REMOVE, pooled_backend.operations
        )
        self.assertEqual(
            first_secret.execution.metadata["pool_decision"],
            "reject",
        )
        self.assertEqual(
            second_secret.execution.metadata["pool_decision"],
            "reject",
        )
        self.assertEqual(
            secret_backend.operations.count(ContainerBackendOperation.CREATE),
            2,
        )
        self.assertEqual(
            secret_backend.operations.count(ContainerBackendOperation.REMOVE),
            2,
        )

    def test_managed_lifecycle_rejects_unsafe_reused_container(
        self,
    ) -> None:
        backend = _fake_backend(cleanup_uncertain=True)
        plan = _run_plan(pooling=_short_lived_pooling(max_uses=3))
        pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_uses=3,
            )
        )
        pool.offer(
            plan,
            _container("contaminated-reuse", plan),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                plan,
                pool=pool,
                pool_safety_checker=ContainerStaticPoolSafetyChecker(
                    report=ContainerPoolSafetyReport(contaminated=True)
                ),
                now_seconds=1,
            )
        )

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertTrue(result.cleanup_uncertain)
        self.assertEqual(result.execution.metadata["pool_decision"], "reuse")
        self.assertEqual(
            result.execution.metadata["pool_reuse_safety"],
            "pooled container is contaminated",
        )
        self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
        self.assertIn(ContainerBackendOperation.CLEANUP, backend.operations)
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    def test_managed_lifecycle_checker_denies_pool_offer(self) -> None:
        cases = (
            (
                _short_lived_pooling(),
                ContainerPoolSafetyReport(dirty_scratch=True),
                "pooled container scratch is dirty",
            ),
            (
                _short_lived_pooling(),
                ContainerPoolSafetyReport(leftover_processes=1),
                "pooled container has leftover processes",
            ),
            (
                ContainerPoolingPolicy(
                    mode=ContainerPoolingMode.SERVICE,
                    health_check_command=("curl", "-f", "http://127.0.0.1"),
                    audit_labels={"service": "phase18"},
                ),
                ContainerPoolSafetyReport(healthy=False),
                "service pool health check failed",
            ),
        )

        for pooling, report, reason in cases:
            with self.subTest(reason=reason):
                backend = _fake_backend()
                result = run_async(
                    run_container_managed_lifecycle(
                        backend,
                        _run_plan(pooling=pooling),
                        pool_safety_checker=(
                            ContainerStaticPoolSafetyChecker(report=report)
                        ),
                        now_seconds=1,
                    )
                )

                self.assertEqual(
                    result.execution.status,
                    ContainerResultStatus.COMPLETED,
                )
                self.assertEqual(
                    result.execution.metadata["pool_offer_decision"],
                    "reject",
                )
                self.assertEqual(
                    result.execution.metadata["pool_offer_reason"],
                    reason,
                )
                self.assertIn(
                    ContainerBackendOperation.REMOVE,
                    backend.operations,
                )

    def test_managed_lifecycle_cleans_evicted_pooled_container(self) -> None:
        backend = _fake_backend()
        plan = _run_plan(
            pooling=_short_lived_pooling(max_age_seconds=1, max_uses=3)
        )
        pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_age_seconds=1,
                max_uses=3,
            )
        )

        pool.offer(
            plan,
            _container("stale", plan),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        result = run_async(
            run_container_managed_lifecycle(
                backend,
                plan,
                pool=pool,
                pool_safety_checker=ContainerStaticPoolSafetyChecker(),
                now_seconds=3,
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(result.execution.metadata["pool_decision"], "evict")
        self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
        self.assertIn(ContainerBackendOperation.CLEANUP, backend.operations)
        self.assertIn(ContainerBackendOperation.CREATE, backend.operations)

    def test_managed_lifecycle_honors_pool_teardown_mode(self) -> None:
        cases = (
            (ContainerPoolTeardownMode.RESET, False),
            (ContainerPoolTeardownMode.QUARANTINE, True),
        )

        for teardown, quarantined in cases:
            with self.subTest(teardown=teardown):
                backend = _fake_backend()
                pooling = _short_lived_pooling(
                    max_age_seconds=1,
                    max_uses=3,
                    teardown=teardown,
                )
                plan = _run_plan(pooling=pooling)
                pool = ContainerPool(pooling)
                pool.offer(
                    plan,
                    _container(f"stale-{teardown.value}", plan),
                    ContainerPoolSafetyReport(),
                    now_seconds=0,
                )

                result = run_async(
                    run_container_managed_lifecycle(
                        backend,
                        plan,
                        pool=pool,
                        pool_safety_checker=ContainerStaticPoolSafetyChecker(),
                        now_seconds=3,
                    )
                )

                self.assertEqual(
                    result.execution.status,
                    ContainerResultStatus.COMPLETED,
                )
                self.assertEqual(
                    result.execution.metadata["pool_decision"],
                    "evict",
                )
                self.assertNotIn(
                    ContainerBackendOperation.REMOVE,
                    backend.operations,
                )
                self.assertIn(
                    ContainerBackendOperation.CLEANUP,
                    backend.operations,
                )
                self.assertEqual(result.orphan_quarantined, quarantined)

    def test_managed_lifecycle_preserves_quarantine_after_replacement_failure(
        self,
    ) -> None:
        backend = _fake_backend(wait_exit_code=7)
        pooling = _short_lived_pooling(
            max_age_seconds=1,
            max_uses=3,
            teardown=ContainerPoolTeardownMode.QUARANTINE,
        )
        plan = _run_plan(pooling=pooling)
        pool = ContainerPool(pooling)
        pool.offer(
            plan,
            _container("quarantine-before-failure", plan),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                plan,
                pool=pool,
                pool_safety_checker=ContainerStaticPoolSafetyChecker(),
                now_seconds=3,
            )
        )

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertTrue(result.orphan_quarantined)
        self.assertIn(ContainerBackendOperation.CREATE, backend.operations)
        self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
        self.assertIn(ContainerBackendOperation.CLEANUP, backend.operations)

    def test_managed_lifecycle_fails_on_uncertain_eviction_cleanup(
        self,
    ) -> None:
        backend = _fake_backend(cleanup_uncertain=True)
        plan = _run_plan(
            pooling=_short_lived_pooling(max_age_seconds=1, max_uses=3)
        )
        pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_age_seconds=1,
                max_uses=3,
            )
        )

        pool.offer(
            plan,
            _container("uncertain", plan),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        result = run_async(
            run_container_managed_lifecycle(
                backend,
                plan,
                pool=pool,
                pool_safety_checker=ContainerStaticPoolSafetyChecker(),
                now_seconds=3,
            )
        )

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertTrue(result.cleanup_uncertain)
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    def test_managed_lifecycle_reports_cancelled_eviction_cleanup(
        self,
    ) -> None:
        result = run_async(_cancel_eviction_cleanup())

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertIn(
            "pooled container eviction cleanup",
            result.diagnostics[-1].message,
        )

    def test_managed_lifecycle_reports_cancelled_unsafe_reuse_cleanup(
        self,
    ) -> None:
        result = run_async(_cancel_unsafe_reuse_cleanup())

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertIn(
            "unsafe pooled container cleanup",
            result.diagnostics[-1].message,
        )

    def test_managed_lifecycle_discards_failed_reused_container(self) -> None:
        backend = _fake_backend(wait_exit_code=7)
        plan = _run_plan(pooling=_short_lived_pooling(max_uses=3))
        pool = ContainerPool(
            ContainerPoolingPolicy(
                mode=ContainerPoolingMode.SHORT_LIVED,
                max_uses=3,
            )
        )

        pool.offer(
            plan,
            _container("reused-failure", plan),
            ContainerPoolSafetyReport(),
            now_seconds=0,
        )
        result = run_async(
            run_container_managed_lifecycle(
                backend,
                plan,
                pool=pool,
                pool_safety_checker=ContainerStaticPoolSafetyChecker(),
                now_seconds=1,
            )
        )

        self.assertEqual(result.execution.status, ContainerResultStatus.FAILED)
        self.assertEqual(result.execution.metadata["pool_decision"], "reuse")
        self.assertEqual(result.execution.metadata["pool_discarded"], "true")
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)
        self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)

    def test_build_cache_cancelled_inflight_retry_succeeds(self) -> None:
        backend = _slow_build_backend()
        cache = ContainerBuildCache()
        plan = _run_plan(image=_build_image())

        retry = run_async(_cancel_build_then_retry(cache, backend, plan))
        cached = run_async(cache.get_or_build(backend, plan, now_seconds=3))

        self.assertTrue(retry.operation.ok)
        self.assertEqual(retry.cache.status, ContainerCacheLookupStatus.MISS)
        self.assertEqual(cached.cache.status, ContainerCacheLookupStatus.HIT)
        self.assertEqual(backend.build_attempts, 1)

    def test_build_cache_ignores_cancelled_inflight_completion(
        self,
    ) -> None:
        cache = ContainerBuildCache()
        plan = _run_plan(image=_build_image())

        retry = run_async(_complete_cancelled_inflight_build(cache, plan))

        self.assertTrue(retry.operation.ok)
        self.assertEqual(retry.cache.status, ContainerCacheLookupStatus.MISS)


class _FlakyBuildBackend(ContainerFakeBackend):
    def __init__(self) -> None:
        super().__init__(
            ContainerFakeBackendScript(
                capabilities=container_backend_capability_profile(
                    "docker-engine-linux"
                ).capabilities,
            )
        )
        self.build_attempts = 0

    async def build_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendOperationResult:
        self.build_attempts += 1
        if self.build_attempts == 1:
            raise RuntimeError("build failed")
        return await super().build_image(plan)


class _SlowBuildBackend(ContainerFakeBackend):
    def __init__(self) -> None:
        super().__init__(
            ContainerFakeBackendScript(
                capabilities=container_backend_capability_profile(
                    "docker-engine-linux"
                ).capabilities,
            )
        )
        self.build_attempts = 0

    async def build_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendOperationResult:
        self.build_attempts += 1
        await sleep(0.02)
        return await super().build_image(plan)


class _PlatformMismatchBackend(ContainerFakeBackend):
    async def resolve_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendImageResolution:
        resolution = await super().resolve_image(plan)
        return ContainerBackendImageResolution(
            reference=resolution.reference,
            digest=resolution.digest,
            platform="linux/arm64",
            diagnostics=resolution.diagnostics,
        )


async def _build_twice(
    cache: ContainerBuildCache,
    backend: ContainerFakeBackend,
    plan: ContainerRunPlan,
) -> tuple[ContainerBuildCacheResult, ContainerBuildCacheResult]:
    return await gather(
        cache.get_or_build(backend, plan, now_seconds=1),
        cache.get_or_build(backend, plan, now_seconds=1),
    )


async def _cancel_eviction_cleanup() -> ContainerManagedLifecycleResult:
    backend = _fake_backend(delay_remove=True)
    plan = _run_plan(
        pooling=_short_lived_pooling(max_age_seconds=1, max_uses=3)
    )
    pool = ContainerPool(
        ContainerPoolingPolicy(
            mode=ContainerPoolingMode.SHORT_LIVED,
            max_age_seconds=1,
            max_uses=3,
        )
    )
    pool.offer(
        plan,
        _container("cancel-eviction", plan),
        ContainerPoolSafetyReport(),
        now_seconds=0,
    )
    task = create_task(
        run_container_managed_lifecycle(
            backend,
            plan,
            pool=pool,
            pool_safety_checker=ContainerStaticPoolSafetyChecker(),
            now_seconds=3,
        )
    )
    await sleep(0.001)
    task.cancel()
    return await task


async def _cancel_unsafe_reuse_cleanup() -> ContainerManagedLifecycleResult:
    backend = _fake_backend(delay_remove=True)
    plan = _run_plan(pooling=_short_lived_pooling(max_uses=3))
    pool = ContainerPool(
        ContainerPoolingPolicy(
            mode=ContainerPoolingMode.SHORT_LIVED,
            max_uses=3,
        )
    )
    pool.offer(
        plan,
        _container("cancel-unsafe-reuse", plan),
        ContainerPoolSafetyReport(),
        now_seconds=0,
    )
    task = create_task(
        run_container_managed_lifecycle(
            backend,
            plan,
            pool=pool,
            pool_safety_checker=ContainerStaticPoolSafetyChecker(
                report=ContainerPoolSafetyReport(contaminated=True)
            ),
            now_seconds=1,
        )
    )
    await sleep(0.001)
    task.cancel()
    return await task


async def _cancel_build_then_retry(
    cache: ContainerBuildCache,
    backend: _SlowBuildBackend,
    plan: ContainerRunPlan,
) -> ContainerBuildCacheResult:
    task = create_task(cache.get_or_build(backend, plan, now_seconds=1))
    await sleep(0)
    task.cancel()
    try:
        await task
    except CancelledError:
        pass
    return await cache.get_or_build(backend, plan, now_seconds=2)


async def _cancelled_build_result() -> ContainerBackendOperationResult:
    await sleep(0.02)
    return ContainerBackendOperationResult(
        operation=ContainerBackendOperation.IMAGE_BUILD,
    )


async def _complete_cancelled_inflight_build(
    cache: ContainerBuildCache,
    plan: ContainerRunPlan,
) -> ContainerBuildCacheResult:
    task = create_task(_cancelled_build_result())
    cache_key = container_build_cache_key(plan)
    cache._inflight[cache_key] = task
    task.cancel()
    try:
        await task
    except CancelledError:
        pass
    await cache._complete_inflight_build(
        cache_key,
        task,
        plan.image.build_cache,
        1,
    )
    return await cache.get_or_build(_fake_backend(), plan, now_seconds=2)


def _policy_plan(
    *,
    escalation: ContainerEscalationMode = ContainerEscalationMode.DENY,
    profile: ContainerProfile | None = None,
    surface: ContainerReviewSurface = ContainerReviewSurface.INTERACTIVE_CLI,
) -> ContainerPolicyPlan:
    return ContainerPolicyPlan(
        effective_settings=_effective(escalation=escalation, profile=profile),
        context=ContainerPolicyContext(surface=surface, scope_id="phase18"),
        command_fingerprint="shell.echo:ok",
    )


def _effective(
    *,
    escalation: ContainerEscalationMode,
    profile: ContainerProfile | None,
) -> ContainerEffectiveSettings:
    profile = profile or _profile()
    profile = ContainerProfile(
        name=profile.name,
        image=profile.image,
        mounts=profile.mounts,
        secrets=profile.secrets,
        network=profile.network,
        devices=profile.devices,
        resources=profile.resources,
        output=profile.output,
        cleanup=profile.cleanup,
        pooling=profile.pooling,
        audit=profile.audit,
        escalation=ContainerEscalationPolicy(mode=escalation),
        command_mode=profile.command_mode,
        read_only_rootfs=profile.read_only_rootfs,
        user=profile.user,
    )
    settings = ContainerSettings(
        source=ContainerSettingsSource(
            surface=ContainerSurface.CLI,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        backend=ContainerBackend.DOCKER,
        default_profile=profile.name,
        allowed_profiles=(profile.name,),
        profiles={profile.name: profile},
        policy_version="phase18",
    )
    return ContainerAuthorityCaps(settings=settings).merge(())


def _profile(
    *,
    image: ContainerImagePolicy | None = None,
    secrets: tuple[ContainerSecretReference, ...] = (),
    network: ContainerNetworkPolicy | None = None,
    pooling: ContainerPoolingPolicy | None = None,
) -> ContainerProfile:
    return ContainerProfile(
        name="phase18-profile",
        image=image or ContainerImagePolicy(reference=_IMAGE),
        secrets=secrets,
        network=network or ContainerNetworkPolicy(),
        pooling=pooling or ContainerPoolingPolicy(),
    )


def _build_image() -> ContainerImagePolicy:
    return ContainerImagePolicy(
        reference=_IMAGE,
        build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
        build_context=ContainerBuildContextPolicy(
            context_path=".",
            dockerfile_path="Dockerfile",
            dockerignore_path=".dockerignore",
            context_digest=f"sha256:{_CONTEXT_DIGEST}",
            context_size_bytes=2048,
            max_context_bytes=4096,
        ),
        build_cache=ContainerBuildCachePolicy(
            mode=ContainerCacheMode.READ_WRITE,
            ttl_seconds=30,
        ),
    )


def _run_plan(
    *,
    image: ContainerImagePolicy | None = None,
    argv: tuple[str, ...] = ("echo", "ok"),
    mounts: tuple[ContainerMountDeclaration, ...] = (),
    pooling: ContainerPoolingPolicy | None = None,
    secret_names: tuple[str, ...] = (),
) -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=ContainerBackend.DOCKER,
        profile_name="phase18-profile",
        image=image or ContainerImagePolicy(reference=_IMAGE),
        command=ContainerCommandPlan(
            tool_name="shell.echo",
            command="echo",
            argv=argv,
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        mounts=mounts,
        pooling=pooling or ContainerPoolingPolicy(),
        secret_names=secret_names,
        policy_version="phase18",
    )


def _short_lived_pooling(
    *,
    max_age_seconds: int = 60,
    max_uses: int = 1,
    idle_ttl_seconds: int = 30,
    teardown: ContainerPoolTeardownMode = ContainerPoolTeardownMode.REMOVE,
) -> ContainerPoolingPolicy:
    return ContainerPoolingPolicy(
        mode=ContainerPoolingMode.SHORT_LIVED,
        max_age_seconds=max_age_seconds,
        max_uses=max_uses,
        idle_ttl_seconds=idle_ttl_seconds,
        teardown=teardown,
    )


def _fake_backend(
    *,
    build_progress_chunks: tuple[ContainerBackendStreamChunk, ...] = (),
    delay_build: bool = False,
    delay_remove: bool = False,
    resolved_digest: str | None = None,
    cleanup_uncertain: bool = False,
    wait_exit_code: int = 0,
) -> ContainerFakeBackend:
    operation_delay_seconds = {}
    if delay_build:
        operation_delay_seconds[ContainerBackendOperation.IMAGE_BUILD] = 0.01
    if delay_remove:
        operation_delay_seconds[ContainerBackendOperation.REMOVE] = 0.05
    return ContainerFakeBackend(
        ContainerFakeBackendScript(
            capabilities=container_backend_capability_profile(
                "docker-engine-linux"
            ).capabilities,
            build_progress_chunks=build_progress_chunks,
            operation_delay_seconds=operation_delay_seconds,
            resolved_digest=resolved_digest,
            cleanup_uncertain=cleanup_uncertain,
            wait_exit_code=wait_exit_code,
        )
    )


def _flaky_build_backend() -> _FlakyBuildBackend:
    return _FlakyBuildBackend()


def _slow_build_backend() -> _SlowBuildBackend:
    return _SlowBuildBackend()


def _platform_mismatch_backend() -> _PlatformMismatchBackend:
    return _PlatformMismatchBackend(
        ContainerFakeBackendScript(
            capabilities=container_backend_capability_profile(
                "docker-engine-linux"
            ).capabilities,
        )
    )


def _container(
    container_id: str,
    plan: ContainerRunPlan | None = None,
) -> ContainerBackendContainer:
    plan = plan or _run_plan()
    return ContainerBackendContainer(
        container_id=container_id,
        backend=ContainerBackend.DOCKER,
        plan_fingerprint=container_pool_key(plan),
    )


if __name__ == "__main__":
    main()
