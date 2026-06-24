from collections.abc import Iterator
from contextlib import contextmanager
from os import mkfifo
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from unittest import TestCase, main

from avalan.container import (
    ContainerDeviceClass,
    ContainerDevicePlan,
    ContainerDevicePolicy,
    ContainerDevicePolicyLimits,
    ContainerEnvironmentPlan,
    ContainerEnvironmentPolicy,
    ContainerHostPathKind,
    ContainerHostPathPolicy,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerNetworkPolicyLimits,
    ContainerPlannedMount,
    ContainerProcessSecurityPlan,
    ContainerProfile,
    ContainerResourceControl,
    ContainerResourceLimits,
    ContainerResourcePlan,
    ContainerResourcePolicy,
    ContainerSecretPolicy,
    ContainerSecretReference,
    ContainerValidatedHostPath,
    plan_container_environment,
    plan_container_mounts,
    plan_container_secrets,
    redact_host_path,
    validate_container_devices,
    validate_container_network,
    validate_container_process_security,
    validate_container_resources,
    validate_container_security_profile,
    validate_host_path,
)

_DIGEST = "5" * 64
_IMAGE = f"ghcr.io/example/phase5@sha256:{_DIGEST}"


class ContainerSecurityTest(TestCase):
    def test_valid_safe_workspace_and_readonly_input_mounts(self) -> None:
        with _workspace() as root:
            workspace = _mkdir(root, "project")
            input_file = _write(root, "inputs/data.txt")
            plan = plan_container_mounts(
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            source=str(workspace),
                            target="/workspace",
                            mount_type=ContainerMountType.WORKSPACE,
                        ),
                        ContainerMountDeclaration(
                            source=str(input_file),
                            target="/inputs/data.txt",
                            mount_type=ContainerMountType.INPUT,
                        ),
                    )
                ),
                _path_policy(root),
            )

            self.assertEqual(
                [mount.target for mount in plan.mounts],
                ["/inputs/data.txt", "/workspace"],
            )
            self.assertTrue(plan.mounts[0].read_only)
            self.assertEqual(
                plan.to_dict()["mounts"][0]["target"],
                "/inputs/data.txt",
            )
            self.assertEqual(
                plan.mounts[1].source.path_kind,
                ContainerHostPathKind.DIRECTORY,
            )
            self.assertNotIn(str(root), str(plan.to_redacted_dict()))

    def test_valid_scratch_output_cache_and_secret_mounts(self) -> None:
        with _workspace() as root:
            secret_file = _write(root, "secrets/api-token")
            plan = plan_container_mounts(
                _profile(
                    mounts=(
                        ContainerMountDeclaration(
                            target="/scratch",
                            mount_type=ContainerMountType.SCRATCH,
                            access=ContainerMountAccess.WRITE,
                        ),
                        ContainerMountDeclaration(
                            target="/outputs",
                            mount_type=ContainerMountType.OUTPUT,
                            access=ContainerMountAccess.WRITE,
                        ),
                        ContainerMountDeclaration(
                            target="/cache",
                            mount_type=ContainerMountType.CACHE,
                        ),
                        ContainerMountDeclaration(
                            source=str(secret_file),
                            target="/run/secrets/api-token",
                            mount_type=ContainerMountType.SECRET,
                        ),
                    )
                ),
                _path_policy(root),
            )

            by_target = {mount.target: mount for mount in plan.mounts}
            self.assertFalse(by_target["/scratch"].read_only)
            self.assertFalse(by_target["/outputs"].read_only)
            self.assertTrue(by_target["/cache"].read_only)
            self.assertTrue(by_target["/run/secrets/api-token"].read_only)
            self.assertEqual(
                by_target["/cache"].source.path_kind,
                ContainerHostPathKind.MISSING,
            )

    def test_valid_minimal_environment_and_secret_delivery(self) -> None:
        environment = ContainerEnvironmentPolicy(
            variables={"LC_ALL": "C.UTF-8"},
            allowlist=("PATH",),
        )
        env_plan = plan_container_environment(
            environment,
            {"PATH": "/usr/bin", "IGNORED": "yes"},
        )
        secret_plan = plan_container_secrets(
            (
                ContainerSecretReference(
                    name="api-token",
                    env_name="API_TOKEN",
                ),
                ContainerSecretReference(
                    name="mounted-token",
                    mount_path="/run/secrets/token",
                ),
            ),
            ContainerSecretPolicy(
                allowed_secret_names=("api-token", "mounted-token"),
            ),
        )

        self.assertEqual(
            env_plan.to_dict(),
            {"variables": {"LC_ALL": "C.UTF-8", "PATH": "/usr/bin"}},
        )
        self.assertEqual(
            secret_plan.to_redacted_dict()["deliveries"][0]["value"],
            "<redacted>",
        )
        self.assertNotIn("super-secret", str(secret_plan.to_redacted_dict()))

    def test_valid_network_device_and_resource_controls(self) -> None:
        no_network = validate_container_network(
            ContainerNetworkPolicy(),
            ContainerNetworkPolicyLimits(),
        )
        loopback = validate_container_network(
            ContainerNetworkPolicy(mode=ContainerNetworkMode.LOOPBACK),
            ContainerNetworkPolicyLimits(
                allowed_modes=(
                    ContainerNetworkMode.LOOPBACK,
                    ContainerNetworkMode.ALLOWLIST,
                ),
                allowed_egress_hosts=("api.example.test", "8.8.8.8"),
            ),
        )
        allowlist = validate_container_network(
            ContainerNetworkPolicy(
                mode=ContainerNetworkMode.ALLOWLIST,
                egress_allowlist=("8.8.8.8", "api.example.test"),
            ),
            ContainerNetworkPolicyLimits(
                allowed_modes=("allowlist",),
                allowed_egress_hosts=("api.example.test", "8.8.8.8"),
            ),
        )
        full = validate_container_network(
            ContainerNetworkPolicy(mode=ContainerNetworkMode.FULL),
            ContainerNetworkPolicyLimits(allowed_modes=("full",)),
        )
        devices = validate_container_devices(
            ContainerDevicePolicy(devices=(ContainerDeviceClass.CPU,)),
            ContainerDevicePolicyLimits(allowed_devices=("cpu",)),
        )
        resources = validate_container_resources(
            ContainerResourceLimits(
                cpu_count=2,
                memory_bytes=536870912,
                pids=128,
                timeout_seconds=30,
            ),
            ContainerResourcePolicy(
                supported_controls=(
                    ContainerResourceControl.CPU,
                    ContainerResourceControl.MEMORY,
                    ContainerResourceControl.PIDS,
                    ContainerResourceControl.TIMEOUT,
                ),
                required_controls=(
                    ContainerResourceControl.CPU,
                    ContainerResourceControl.MEMORY,
                ),
            ),
        )

        self.assertFalse(no_network.escalation_required)
        self.assertTrue(loopback.escalation_required)
        self.assertTrue(full.escalation_required)
        self.assertEqual(
            allowlist.to_dict()["egress_allowlist"],
            ["8.8.8.8", "api.example.test"],
        )
        self.assertEqual(devices.to_dict(), {"devices": ["cpu"]})
        self.assertEqual(resources.to_dict()["best_effort_unsupported"], [])

    def test_host_path_rejects_traversal_nul_expansion_and_remote(
        self,
    ) -> None:
        with _workspace() as root:
            policy = _path_policy(root)
            for path in (
                f"{root}/../outside",
                f"{root}/bad\x00path",
                "~/secret",
                "$HOME/secret",
                "ssh://host/workspace",
                "//server/share",
            ):
                with self.subTest(path=path):
                    with self.assertRaises(AssertionError):
                        validate_host_path(path, policy, require_exists=False)

    def test_host_path_rejects_hidden_sensitive_credentials_and_vcs(
        self,
    ) -> None:
        with _workspace() as root:
            hidden = _write(root, ".hidden/file.txt")
            credential = _write(root, ".aws/credentials")
            git_config = _write(root, ".git/config")

            with self.assertRaises(AssertionError):
                validate_host_path(str(hidden), _path_policy(root))
            with self.assertRaises(AssertionError):
                validate_host_path(
                    str(credential),
                    _path_policy(root, allow_hidden_paths=True),
                )
            with self.assertRaises(AssertionError):
                validate_host_path(
                    str(git_config),
                    _path_policy(root, allow_hidden_paths=True),
                )
            with self.assertRaises(AssertionError):
                validate_host_path(
                    "/etc/passwd",
                    ContainerHostPathPolicy(
                        allowed_roots=("/",),
                        runtime_shared_roots=("/",),
                    ),
                    require_exists=False,
                )
            for path in ("/private/var/db", "/private/var/root"):
                with self.subTest(path=path):
                    with self.assertRaises(AssertionError):
                        validate_host_path(
                            path,
                            ContainerHostPathPolicy(
                                allowed_roots=("/",),
                                runtime_shared_roots=("/",),
                            ),
                            require_exists=False,
                        )

    def test_host_path_rejects_symlink_special_socket_and_mismatch(
        self,
    ) -> None:
        with _workspace() as root:
            other = _mkdir(root, "shared-other")
            target = _write(root, "target.txt")
            link = root / "link.txt"
            link.symlink_to(target)
            fifo = root / "fifo"
            mkfifo(fifo)

            with self.assertRaises(AssertionError):
                validate_host_path(str(link), _path_policy(root))
            with self.assertRaises(AssertionError):
                validate_host_path(str(fifo), _path_policy(root))
            with self.assertRaises(AssertionError):
                validate_host_path(
                    "/private/var/run/docker.sock",
                    ContainerHostPathPolicy(
                        allowed_roots=("/",),
                        runtime_shared_roots=("/",),
                    ),
                    require_exists=False,
                )
            with self.assertRaises(AssertionError):
                validate_host_path(
                    str(target),
                    ContainerHostPathPolicy(
                        allowed_roots=(str(root),),
                        runtime_shared_roots=(str(other),),
                    ),
                )

    def test_host_path_rejects_missing_required_and_input_without_source(
        self,
    ) -> None:
        with _workspace() as root:
            with self.assertRaises(AssertionError):
                validate_host_path(str(root / "missing"), _path_policy(root))
            with self.assertRaises(AssertionError):
                plan_container_mounts(
                    _profile(
                        mounts=(
                            ContainerMountDeclaration(
                                target="/input",
                                mount_type=ContainerMountType.INPUT,
                            ),
                        )
                    ),
                    _path_policy(root),
                )

    def test_environment_rejects_inheritance_and_raw_secret_names(
        self,
    ) -> None:
        inherited = ContainerEnvironmentPolicy()
        object.__setattr__(inherited, "inherit_host", True)

        with self.assertRaises(AssertionError):
            plan_container_environment(inherited, {"PATH": "/usr/bin"})
        with self.assertRaises(AssertionError):
            plan_container_environment(
                ContainerEnvironmentPolicy(
                    variables={"API_TOKEN": "secret"},
                ),
                {},
            )
        with self.assertRaises(AssertionError):
            plan_container_environment(
                ContainerEnvironmentPolicy(allowlist=("PASSWORD",)),
                {"PASSWORD": "secret"},
            )
        with self.assertRaises(AssertionError):
            ContainerEnvironmentPlan(variables=())

    def test_secret_delivery_rejects_unauthorized_and_disallowed_targets(
        self,
    ) -> None:
        secret = ContainerSecretReference(name="api-token", env_name="TOKEN")
        mounted = ContainerSecretReference(
            name="api-token",
            mount_path="/run/secrets/token",
        )

        with self.assertRaises(AssertionError):
            plan_container_secrets(
                (secret,),
                ContainerSecretPolicy(allowed_secret_names=("other",)),
            )
        with self.assertRaises(AssertionError):
            plan_container_secrets(
                (secret,),
                ContainerSecretPolicy(
                    allowed_secret_names=("api-token",),
                    allow_env_delivery=False,
                ),
            )
        with self.assertRaises(AssertionError):
            plan_container_secrets(
                (mounted,),
                ContainerSecretPolicy(
                    allowed_secret_names=("api-token",),
                    allow_mount_delivery=False,
                ),
            )

    def test_network_rejects_denied_full_network_and_unsafe_egress(
        self,
    ) -> None:
        unsafe_hosts = (
            "127.0.0.1",
            "169.254.169.254",
            "metadata.google.internal",
            "localhost",
            "api.example.test/redirect",
            "https://api.example.test",
            "*.example.test",
            "metadata",
        )
        for host in unsafe_hosts:
            with self.subTest(host=host):
                with self.assertRaises(AssertionError):
                    validate_container_network(
                        ContainerNetworkPolicy(
                            mode=ContainerNetworkMode.ALLOWLIST,
                            egress_allowlist=(host,),
                        ),
                        ContainerNetworkPolicyLimits(
                            allowed_modes=("allowlist",),
                            allowed_egress_hosts=(host,),
                        ),
                    )
        with self.assertRaises(AssertionError):
            validate_container_network(
                ContainerNetworkPolicy(mode=ContainerNetworkMode.FULL),
                ContainerNetworkPolicyLimits(),
            )

    def test_devices_process_and_required_resources_deny(self) -> None:
        with self.assertRaises(AssertionError):
            validate_container_devices(
                ContainerDevicePolicy(
                    devices=(ContainerDeviceClass.NVIDIA_CDI,)
                ),
                ContainerDevicePolicyLimits(
                    allowed_devices=(ContainerDeviceClass.CPU,),
                ),
            )
        with self.assertRaises(AssertionError):
            validate_container_process_security(
                ContainerProcessSecurityPlan(user="0")
            )
        with self.assertRaises(AssertionError):
            validate_container_process_security(
                ContainerProcessSecurityPlan(user="1000", privileged=True)
            )
        with self.assertRaises(AssertionError):
            validate_container_process_security(
                ContainerProcessSecurityPlan(
                    user="1000",
                    capabilities=("NET_ADMIN",),
                )
            )
        with self.assertRaises(AssertionError):
            validate_container_resources(
                ContainerResourceLimits(memory_bytes=1024),
                ContainerResourcePolicy(
                    supported_controls=(ContainerResourceControl.CPU,),
                    required_controls=(ContainerResourceControl.MEMORY,),
                ),
            )

    def test_best_effort_resource_controls_are_recorded(self) -> None:
        plan = validate_container_resources(
            ContainerResourceLimits(pids=32),
            ContainerResourcePolicy(
                supported_controls=(ContainerResourceControl.CPU,),
            ),
        )

        self.assertEqual(
            plan.to_dict()["best_effort_unsupported"],
            ["pids"],
        )

    def test_fake_e2e_profile_validation_redacts_and_marks_combined_risk(
        self,
    ) -> None:
        with _workspace() as root:
            workspace = _mkdir(root, "project")
            profile = _profile(
                mounts=(
                    ContainerMountDeclaration(
                        source=str(workspace),
                        target="/workspace",
                        mount_type=ContainerMountType.WORKSPACE,
                    ),
                ),
                environment=ContainerEnvironmentPolicy(
                    variables={"LC_ALL": "C.UTF-8"},
                ),
                secrets=(
                    ContainerSecretReference(
                        name="api-token",
                        env_name="API_TOKEN",
                    ),
                ),
                network=ContainerNetworkPolicy(
                    mode=ContainerNetworkMode.ALLOWLIST,
                    egress_allowlist=("api.example.test",),
                ),
                devices=ContainerDevicePolicy(
                    devices=(ContainerDeviceClass.CPU,),
                ),
                resources=ContainerResourceLimits(cpu_count=1),
            )
            plan = validate_container_security_profile(
                profile,
                path_policy=_path_policy(root),
                secret_policy=ContainerSecretPolicy(
                    allowed_secret_names=("api-token",),
                ),
                network_policy=ContainerNetworkPolicyLimits(
                    allowed_modes=("allowlist",),
                    allowed_egress_hosts=("api.example.test",),
                ),
                device_policy=ContainerDevicePolicyLimits(
                    allowed_devices=("cpu",),
                ),
                resource_policy=ContainerResourcePolicy(
                    supported_controls=("cpu",),
                ),
                host_environment={},
            )
            redacted = plan.to_redacted_dict()

            self.assertIn("combined_risk", redacted["risk_tags"])
            self.assertNotIn(str(root), str(redacted))
            self.assertIn("<redacted>", str(redacted))
            self.assertIn("<host-path>/project", str(redacted))

    def test_fake_e2e_profile_validation_denies_network_device_and_resource(
        self,
    ) -> None:
        with _workspace() as root:
            for profile in (
                _profile(
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("api.example.test",),
                    )
                ),
                _profile(
                    devices=ContainerDevicePolicy(
                        devices=(ContainerDeviceClass.NVIDIA_CDI,),
                    )
                ),
                _profile(resources=ContainerResourceLimits(memory_bytes=1024)),
            ):
                with self.subTest(profile=profile.to_dict()):
                    with self.assertRaises(AssertionError):
                        validate_container_security_profile(
                            profile,
                            path_policy=_path_policy(root),
                            secret_policy=ContainerSecretPolicy(),
                            network_policy=ContainerNetworkPolicyLimits(),
                            device_policy=ContainerDevicePolicyLimits(
                                allowed_devices=(ContainerDeviceClass.CPU,),
                            ),
                            resource_policy=ContainerResourcePolicy(
                                supported_controls=(
                                    ContainerResourceControl.CPU,
                                ),
                                required_controls=(
                                    ContainerResourceControl.MEMORY,
                                ),
                            ),
                            host_environment={},
                        )

    def test_value_objects_convert_strings_and_serialize(self) -> None:
        path = ContainerValidatedHostPath(
            original_path="/workspace/file.txt",
            normalized_path="/workspace/file.txt",
            allowed_root="/workspace",
            path_kind="file",
        )
        mount = ContainerPlannedMount(
            source=path,
            target="/file.txt",
            mount_type="input",
            access="read",
        )
        devices = ContainerDevicePlan(devices=("cpu",))
        resources = ContainerResourcePlan(
            limits=ContainerResourceLimits(timeout_seconds=5),
            best_effort_unsupported=("timeout",),
        )

        self.assertEqual(path.to_dict()["path_kind"], "file")
        self.assertTrue(mount.read_only)
        self.assertEqual(mount.to_dict()["mount_type"], "input")
        self.assertEqual(devices.to_dict(), {"devices": ["cpu"]})
        self.assertEqual(
            resources.to_dict()["best_effort_unsupported"],
            ["timeout"],
        )
        self.assertEqual(
            redact_host_path("/workspace"),
            "<host-path>/workspace",
        )


@contextmanager
def _workspace() -> Iterator[Path]:
    with TemporaryDirectory(dir=str(Path(gettempdir()).resolve())) as root:
        yield Path(root)


def _path_policy(
    root: str | Path,
    *,
    allow_hidden_paths: bool = False,
) -> ContainerHostPathPolicy:
    return ContainerHostPathPolicy(
        allowed_roots=(str(root),),
        runtime_shared_roots=(str(root),),
        allow_hidden_paths=allow_hidden_paths,
    )


def _mkdir(root: str | Path, relative: str) -> Path:
    path = Path(root) / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(root: str | Path, relative: str) -> Path:
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")
    return path


def _profile(
    *,
    mounts: tuple[ContainerMountDeclaration, ...] = (),
    environment: ContainerEnvironmentPolicy | None = None,
    secrets: tuple[ContainerSecretReference, ...] = (),
    network: ContainerNetworkPolicy | None = None,
    devices: ContainerDevicePolicy | None = None,
    resources: ContainerResourceLimits | None = None,
) -> ContainerProfile:
    return ContainerProfile(
        name="phase5",
        image=ContainerImagePolicy(reference=_IMAGE),
        mounts=mounts,
        environment=environment or ContainerEnvironmentPolicy(),
        secrets=secrets,
        network=network or ContainerNetworkPolicy(),
        devices=devices or ContainerDevicePolicy(),
        resources=resources or ContainerResourceLimits(),
    )


if __name__ == "__main__":
    main()
