# Isolation Execution

Avalan isolation is the runtime policy layer for shell execution and related
runtime surfaces. The policy model is a tagged union: exactly one of `local`,
`sandbox`, or `container` is active for a plan. Sandbox and container policy
fields are separate, and a plan that mixes them is rejected before execution.

Isolation policy is trusted configuration. Models, prompts, remote HTTP
requests, MCP calls, A2A parts, task payloads, and untrusted flow or agent data
must not define images, mounts, sandbox roots, host executables, backend flags,
network policy, devices, resources, secrets, or approval behavior.

## Supported Modes And Backends

| Mode | Runtime boundary | Supported backends | Current use |
| --- | --- | --- | --- |
| `local` | Direct host execution with Avalan tool policy, bounded streams, audit, and approval checks. It is not OS-level isolation. | None. | Host tools that cannot run in a container or sandbox. Elevation into this mode requires explicit approval when requested from an isolated plan. |
| `sandbox` | Host process launched through a generated sandbox profile. | `seatbelt`, `bubblewrap` | Shell tools with host executables and host paths while preserving sandbox path, environment, network, process, output, and cleanup controls. |
| `container` | Container lifecycle managed by a trusted backend. | `docker`, `apple-container` | Shell tools, container-aware task/flow metadata, and runtime envelope planning where a trusted runner supplies an implementation. |

`docker` is the promoted container backend. `apple-container` is opt-in and
available for shell container execution when the Apple `container` CLI is
installed and selected. `podman`, `nerdctl`, containerd/nerdctl, Microsoft
containers, Windows Docker, WSL2, Hyper-V, and `auto` are not executable
container backends. `none` exists only as a disabled container-settings value;
it is not an executable backend.

## Policy Fields

Common isolation fields are:

- `mode`: one of `local`, `sandbox`, or `container`.
- `source`: the trusted surface and trust level that supplied policy.
- `policy_version`: the operator policy version used in fingerprints,
  approvals, durable metadata, and audit.
- `profile_registry_id`: the registry identity for profile selection.
- `default_profile` and `allowed_profiles`: the trusted profile set that
  untrusted selectors may narrow, never widen.
- `required`: whether execution must fail closed instead of falling back.

Local policy fields are `approval_required`, `allowed_roots`,
`executable_allowlist`, `timeout_seconds`, `max_stdout_bytes`, and
`max_stderr_bytes`.

Sandbox policy fields are `backend`, `default_profile`, `allowed_profiles`,
`profiles`, `profile_registry_id`, and `policy_version`. Sandbox profile fields
include `trusted_executables`, `executable_search_roots`, `read_roots`,
`write_roots`, `deny_roots`, `scratch_roots`, `output_roots`, `environment`,
`network`, `resources`, `output`, `child_processes`, `inherited_fds`, and
`cleanup`.

Container policy fields are `backend`, `default_profile`, `allowed_profiles`,
`profiles`, `profile_registry_id`, and `policy_version`. Container profile
fields include digest-pinned `image`, `workspace`, `mounts`, `environment`,
`secrets`, `network`, `devices`, `resources`, `output`, `cleanup`, `pooling`,
`audit`, `escalation`, `command_mode`, `read_only_rootfs`, and `user`.

## Runtime Setup

Install and authorize the selected runtime outside Avalan:

| Backend | Setup notes |
| --- | --- |
| `docker` | Install Docker CLI and a Docker Engine or Docker Desktop daemon. Treat Docker socket access as host-powerful. Rootful Docker requires trusted authorization. Use digest-pinned images. |
| `apple-container` | Install and start Apple `container` on macOS. Use a digest-pinned image matching the selected platform, commonly `linux/arm64`. |
| `seatbelt` | macOS only. Avalan generates a `sandbox-exec` profile from trusted policy. It supports path controls and `none` or loopback network modes; full network and allowlist modes are not supported by this backend. |
| `bubblewrap` | Linux only. Install `bwrap` and ensure user namespaces, bind mounts, and `/proc` mounting are available. Network isolation depends on namespace support; network allowlists are not implemented for this backend. |

Default tests use fake backends and probes. They should not require live
Docker, Apple `container`, `sandbox-exec`, or `bwrap`.

## Agent TOML Examples

Use local shell tools only when direct host execution is intended:

```toml
[tool]
enable = ["shell.rg"]

[tool.shell]
backend = "local"
workspace_root = "."
cwd = "."
max_stdout_bytes = 65536
```

Use a sandbox profile when host tools are needed but sandbox controls can
enforce the request:

```toml
[tool]
enable = ["shell.cat"]

[tool.sandbox]
backend = "seatbelt"
default_profile = "host-tools"

[tool.sandbox.profiles.host-tools]
trusted_executables = ["/bin/cat"]
read_roots = ["/tmp"]
scratch_roots = ["/tmp"]
output_roots = ["/tmp"]
child_processes = "deny"
inherited_fds = "stdio"

[tool.sandbox.profiles.host-tools.network]
mode = "none"

[tool.sandbox.profiles.host-tools.resources]
timeout_seconds = 10
pids = 16

[tool.shell]
backend = "sandbox"

[tool.shell.sandbox]
profile = "host-tools"
required = true
```

Use a container profile when the command can run inside a digest-pinned image:

```toml
[tool]
enable = ["shell.cat"]

[tool.container]
backend = "docker"
default_profile = "workspace-readonly"
allowed_profiles = ["workspace-readonly"]

[tool.container.profiles.workspace-readonly]
image = "ghcr.io/example/avalan-shell@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
workspace_root = "."
network = "none"
pull_policy = "never"
platform = "linux/amd64"

[tool.shell]
backend = "container"

[tool.shell.container]
profile = "workspace-readonly"
required = true
```

These examples configure policy. They do not install runtimes, pull images
unless policy allows it, or grant model-visible runtime authority.

## Environment-Gated Real Tests

The normal suite uses fake backends. Optional live-runtime checks are gated by
environment variables:

| Runtime | Gate | Current tests |
| --- | --- | --- |
| Docker conformance probe | `AVALAN_CONTAINER_DOCKER_E2E=1` | `tests/container/release_conformance_test.py` |
| Docker backend lifecycle | `AVALAN_CONTAINER_DOCKER_E2E=1` and `AVALAN_CONTAINER_DOCKER_E2E_IMAGE=<digest-pinned-image>` | `tests/container/docker_test.py` |
| Apple `container` backend lifecycle and shell toolset | `AVALAN_CONTAINER_APPLE_E2E=1` and `AVALAN_CONTAINER_APPLE_E2E_IMAGE=<digest-pinned-image>` | `tests/container/apple_test.py` |
| Seatbelt real sandbox probe | `AVALAN_ISOLATION_SEATBELT_E2E=1` | `tests/sandbox/real_runtime_e2e_test.py` |
| Bubblewrap real sandbox probe | `AVALAN_ISOLATION_BUBBLEWRAP_E2E=1` | `tests/sandbox/real_runtime_e2e_test.py` |

Run fake-backed conformance and planning checks with:

```bash
poetry run pytest \
  tests/isolation \
  tests/sandbox \
  tests/container/release_conformance_test.py \
  tests/container/fail_closed_conformance_test.py \
  tests/container/watchdog_conformance_test.py \
  tests/container/stress_conformance_test.py \
  tests/tool/shell/container_test.py \
  tests/tool/shell/sandbox_test.py
```

The release gate remains:

```bash
poetry run pytest --verbose -s
make test-coverage
make lint
```

## Platform Limits

Docker Engine on Linux is the promoted container profile. Docker Desktop on
macOS is VM-backed, so shared mount prefixes, path normalization, case
behavior, signal behavior, and resource ceilings are runtime-specific. Apple
`container` is opt-in and currently documented for shell tool execution.

Seatbelt is macOS-only through `sandbox-exec`. Bubblewrap is Linux-only and
depends on kernel namespace support. Both sandbox backends reject plans whose
required controls exceed probed capabilities.

Current container catalog capabilities are CPU-oriented. GPU/device classes,
rootful engines, build support, network allowlists, service pools, and broad
artifact copy behavior require compatible backend capabilities and explicit
operator policy. Metal is not exposed as a generic Linux container GPU.

## Security Posture

Isolation is defense in depth. Avalan still applies host-side policy before
starting any sandbox or container:

- Tool schemas do not expose runtime authority to the model.
- Trusted profiles choose images, roots, mounts, environment variables,
  secrets, networks, devices, resources, output policy, cleanup, and approval
  behavior.
- Untrusted surfaces may select only approved profile names where the surface
  explicitly supports that selector. They cannot define policy.
- Remote server, MCP, and A2A inputs reject runtime-authority fields. Exposed
  remote container profile selectors are allowed only when operator policy
  exposes the selected profile; remote sandbox profile selectors are rejected
  today.
- Host paths are normalized and checked under approved roots. Traversal,
  sensitive paths, runtime sockets, hidden paths, symlink escapes, special
  files, and remote paths are denied or redacted by policy.
- Environment inheritance is denied by default. Only explicit variables and
  allowed host variable names are planned.
- Secrets are referenced by approved names and redacted from diagnostics and
  audit output.
- Streams and generated artifacts are bounded, typed, validated, and redacted
  before becoming tool, task, or audit output.

Do not treat containers or host sandboxes as a complete multi-tenant boundary
without deployment-specific runtime hardening, user namespace or VM-grade
isolation, host-level controls, and operational monitoring.

## Fail-Closed Behavior

Required isolated execution does not silently fall back to a weaker mode.
Avalan fails closed on missing runtime backends, unsupported backend values,
unsupported mode values, capability mismatches, unsafe profile widening,
unavailable profiles, image or pull denials, invalid paths, unsafe secrets,
network denials, timeouts, cancellations, output validation failures, cleanup
uncertainty, stale durable metadata, stale approvals, and unsupported surfaces.

`shell.pipeline` has an additional v1 boundary: full byte-stream pipelines are
supported only by the local composition executor. A sandbox or container plan
with `mode = "pipeline"` or any `stdin_from` byte routing is policy-denied
with a structured result instead of falling back to host execution or lowering
the composition to shell text. Isolated `serial` and `parallel` compositions
without stdin routing may run only through the selected backend's existing
single-command executor. A future trusted structured runner can add isolated
byte pipelines without changing model-facing schemas.

Optional local fallback is allowed only when the profile is not required and
policy permits local execution. Required `sandbox` execution does not become
`local` without review, and required `container` execution does not become
`sandbox` or `local` without review.

Durable task workers and resumed flows revalidate mode, backend, profile
registry, policy version, scope, and plan fingerprint before execution.

## Approval Behavior

Approval decisions are scoped to the canonical plan fingerprint, effective
mode, backend, profile, policy version, review surface, scope, attempt, and
elevation rung. Approvals cannot be reused across broader scopes, changed
profiles, changed backends, changed modes, changed roots, changed resources,
changed images, changed policy versions, or changed attempts.

Interactive CLI runs may pause for review. Strict flows and queued tasks use
durable review records that must match the exact plan before resume or worker
execution. Direct task execution, server requests, MCP, and A2A are
noninteractive; review-required plans are denied rather than waiting for human
input.

## Diagnostics And Audit

The Phase 10 stable isolation diagnostic inventory is:

- `isolation.mode_conflict`
- `isolation.unsupported_mode`
- `isolation.unsupported_backend`
- `isolation.mode_unavailable`
- `isolation.capability_mismatch`
- `isolation.elevation_required`
- `isolation.elevation_denied`
- `isolation.fallback_denied`
- `isolation.approval_stale`
- `isolation.policy_drift`
- `isolation.audit_unavailable`
- `sandbox.provider_unavailable`
- `sandbox.profile_generation_failed`
- `sandbox.path_denied`
- `sandbox.network_unenforceable`
- `container.backend.unavailable`
- `container.backend.capability_mismatch`

Current sandbox backend diagnostics include `sandbox.backend.unavailable`,
`sandbox.backend.capability_mismatch`, `sandbox.backend.executable_denied`,
`sandbox.backend.path_denied`, `sandbox.backend.output_rejected`,
`sandbox.backend.execution_failed`, `sandbox.backend.cleanup_failed`,
`sandbox.backend.cancelled`, `sandbox.backend.timeout`,
`sandbox.backend.stream_truncated`, and
`sandbox.backend.concurrency_limit`.

Current container diagnostics include top-level conformance codes such as
`container.backend_required`, `container.backend_unavailable`, and
`container.unsupported_syntax`; backend codes such as
`container.backend.unavailable`, `container.backend.capability_mismatch`,
`container.backend.rootful_not_authorized`, `container.backend.image_denied`,
`container.backend.pull_denied`, `container.backend.build_denied`,
`container.backend.timeout`, `container.backend.cancelled`,
`container.backend.stream_truncated`, and
`container.backend.cleanup_failed`; and output validation codes such as
`container.output.too_large`, `container.output.symlink_escape`,
`container.output.special_file`, `container.output.traversal`,
`container.output.unsafe_media`, and `container.output.race_detected`.

`isolation.unsupported_syntax` remains a settings-loader diagnostic for
unsupported task and flow syntax, but it is not part of the Phase 10 runtime
failure-mode inventory.

Audit records include requested mode, effective mode, backend, profile,
profile registry, policy version, plan fingerprint, review surface, scope,
attempt, elevation rung, lost controls, diagnostic code, cleanup status, and
redacted metadata. Raw stream bytes, prompts, host paths, provider tokens,
secret names or values, and large metadata fields are redacted or truncated.

## Related Documentation

- [Container execution](CONTAINERS.md) covers container-specific setup,
  examples, conformance, and deferred non-conformance.
- [Tools and reasoning](TOOLS.md) covers shell tool policy and trusted
  deployment configuration.
- [CLI reference](CLI.md) lists the trusted shell, sandbox, and container
  flags.
- [Isolation traceability](ISOLATION_TRACEABILITY.md) maps requirements and
  safety claims to tests.
