# Task Queue Operations

This guide covers production operation for durable task queue mode. It assumes
task definitions already validate locally and that model/provider credentials
are managed outside Avalan.

## Worker Deployment

Queue mode uses PostgreSQL for lifecycle state, queue claims, idempotency,
artifact metadata, sanitized events, and usage inspection. Run producers and
workers against the same migrated schema:

```bash
python3 -m pip install -U "avalan[task,agent,task-pgsql]"
python3 -m pip install -U alembic "SQLAlchemy>=2.0.43,<3.0.0"

export AVALAN_TASK_STORE_DSN="postgresql://..."
export AVALAN_TASK_STORE_SCHEMA="avalan_tasks"
export AVALAN_TASK_HMAC_KEY_ID="prod-v1"
export AVALAN_TASK_HMAC_KEY_B64="base64-encoded-32-byte-key"
export AVALAN_TASK_ARTIFACT_ROOT="/var/lib/avalan/task-artifacts"

avalan task pgsql check
avalan task pgsql migrate head
```

Submit queued work from application hosts or automation:

```bash
avalan task enqueue tasks/report.task.toml \
  --queue reports \
  --input-json @payload.json
```

Run workers as supervised, bounded processes. The CLI worker processes up to
`--limit` runs and then exits, so production deployments should use a process
supervisor, container restart policy, or job controller:

```bash
avalan task worker \
  --queue reports \
  --worker-id "reports-${HOSTNAME}" \
  --lease-seconds 300 \
  --heartbeat-seconds 30 \
  --limit 1000
```

For systemd, keep secrets in an environment file with restricted permissions:

```ini
[Unit]
Description=Avalan task worker
After=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/avalan/task.env
WorkingDirectory=/srv/avalan
ExecStart=/usr/local/bin/avalan task worker --queue reports --limit 1000 --lease-seconds 300 --heartbeat-seconds 30
Restart=always
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=330
User=avalan
Group=avalan

[Install]
WantedBy=multi-user.target
```

For Kubernetes, run workers separately from API or producer pods. Use a
Deployment for continuous processing or a CronJob for scheduled drain windows.
Set `terminationGracePeriodSeconds` longer than the worker heartbeat interval
and close to the claim lease so shutdown can complete or leave work reclaimable:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: avalan-task-worker
spec:
  replicas: 4
  selector:
    matchLabels:
      app: avalan-task-worker
  template:
    metadata:
      labels:
        app: avalan-task-worker
    spec:
      terminationGracePeriodSeconds: 330
      containers:
        - name: worker
          image: your-registry/avalan-worker:latest
          command:
            - avalan
            - task
            - worker
            - --queue
            - reports
            - --limit
            - "1000"
            - --lease-seconds
            - "300"
            - --heartbeat-seconds
            - "30"
          envFrom:
            - secretRef:
                name: avalan-task-env
```

Use multiple queues when workloads have different latency, cost, provider, or
artifact profiles. Keep worker ids stable within a process lifetime and unique
across concurrent workers.

## Required Environment

Set these values for durable queue mode:

| Variable | Purpose |
| --- | --- |
| `AVALAN_TASK_STORE_DSN` or `AVALAN_TASK_PGSQL_DSN` | PostgreSQL DSN for task storage, queue, events, usage, and artifact metadata. |
| `AVALAN_TASK_STORE_SCHEMA` or `AVALAN_TASK_PGSQL_SCHEMA` | Optional schema/search path for isolated deployments. |
| `AVALAN_TASK_HMAC_KEY_ID` | Identifier stored with HMAC summaries. Rotate by adding a new id and keeping old keys available for inspection windows. |
| `AVALAN_TASK_HMAC_KEY_B64` | Base64-encoded HMAC secret for user-controlled identifiers. Do not log this value. |
| `AVALAN_TASK_ARTIFACT_ROOT` | Local artifact byte root for file materialization and retention sweeps. Required when tasks use file inputs or output artifacts. |
| Provider credentials | Model or tool credentials required by the referenced agent or target. Use provider-specific environment variables or secret managers. |

Install only the extras the deployment uses:

| Extra | Use |
| --- | --- |
| `task` | JSON Schema validation for structured task input and output contracts. |
| `agent` | Agent-backed task execution. |
| `task-pgsql` | Durable PostgreSQL task store and queue workers. |
| `task-documents` | Document conversion for file inputs. |
| `task-pdf-images` | PDF page rasterization for image file delivery. |
| `task-prometheus` | Prometheus task metrics sink. |
| `task-otel` | OpenTelemetry task spans and metrics sink. |

Migration commands additionally require Alembic and SQLAlchemy in the operator
environment. They are not installed by the base package or by `task-pgsql`.

Do not put prompts, raw input, output, file bytes, DSNs, passwords, provider
tokens, or stack traces into worker ids, queue names, metadata, metric labels,
or logs. Queue metadata is for safe scheduling information only.

## PostgreSQL Requirements

Use PostgreSQL 14 or newer. The task schema does not require PostgreSQL
extensions. The database role used for migrations needs permission to create
tables, indexes, triggers, and functions in the target database or schema. The
runtime role needs read/write access to the migrated task tables and sequences.

Run migration checks before starting producers or workers:

```bash
avalan task pgsql diagnose
avalan task pgsql check
```

If `check` reports pending migrations, stop workers, apply migrations, run
`check` again, and then restart workers:

```bash
avalan task pgsql migrate head
avalan task pgsql check
```

Configure the PostgreSQL provider with explicit connection limits and SSL
settings. Preserve DSN query parameters such as `sslmode`,
`target_session_attrs`, and `channel_binding`. Size the pool for the total
number of producer and worker processes, leaving capacity for migrations,
inspection, and emergency administration.

Use a statement timeout that is longer than normal claim, heartbeat, and
inspection paths but shorter than the worker lease. Use a lock timeout that
lets migrations and queue operations fail fast instead of blocking
indefinitely. Use an idle transaction timeout to prevent abandoned
transactions from holding locks.

## Routine Operations

Validate task definitions and input before enqueueing:

```bash
avalan task validate tasks/report.task.toml --input-json @payload.json
```

Inspect a run without exposing raw payloads:

```bash
avalan task inspect RUN_ID --after-sequence 0
avalan task events RUN_ID --after-sequence 100
avalan task output RUN_ID
avalan task artifacts RUN_ID
```

Delete expired artifact bytes while preserving sanitized metadata:

```bash
avalan task retention-sweep --purpose input --purpose output --limit 500
```

For queue health, use the SDK or internal operations tooling around
`PgsqlTaskQueue.health(queue_name)`. Track `available`, `scheduled`,
`claimed`, `dead`, `cancel_requested`, `oldest_available_at`, and
`expired_claims`. Alert when expired claims stay nonzero, dead jobs increase
unexpectedly, or the oldest available timestamp ages beyond the workload SLO.

## Failure Runbooks

### Missing Store

Symptoms:

- `avalan task run` without `--ephemeral`, `enqueue`, `worker`, `inspect`,
  `output`, `events`, `artifacts`, or `retention-sweep` reports
  `store.missing`.
- The command output asks for `AVALAN_TASK_STORE_DSN` or `--store-dsn`.

Remediation:

1. For local direct experimentation only, rerun direct tasks with
   `--ephemeral`. Do not use ephemeral mode for queues or inspection.
2. For durable mode, set `AVALAN_TASK_STORE_DSN` or pass `--store-dsn` and set
   `AVALAN_TASK_STORE_SCHEMA` or `--store-schema` when using a non-default
   schema.
3. Run `avalan task pgsql diagnose` and `avalan task pgsql check` from the same
   environment.
4. If migrations are pending, run `avalan task pgsql migrate head`, then retry
   the task command.

### Missing Artifact Root

Symptoms:

- File tasks or retention sweeps report `artifact_store.missing`.
- Local workers can claim runs but fail before materializing file inputs or
  output artifacts.

Remediation:

1. Set `AVALAN_TASK_ARTIFACT_ROOT` to a durable, writable directory shared by
   producers and workers that need local artifact bytes.
2. Verify filesystem ownership, permissions, free space, and mount health for
   the worker user.
3. Restart workers after changing the environment. Do not point workers at a
   temporary directory unless all queued file tasks are intentionally disposable.
4. For production large files, prefer an object-store artifact backend and keep
   PostgreSQL byte storage limited to small payloads.

### Store Outage

Symptoms:

- `avalan task enqueue`, `worker`, `inspect`, or `retention-sweep` fail with a
  store or PostgreSQL error.
- `avalan task pgsql check` cannot connect or reports a migration mismatch.
- Worker logs show low-cardinality categories such as `connection`, `timeout`,
  `deadlock`, `serialization`, `insufficient_privilege`, or `migration`.

Remediation:

1. Stop or scale down workers if the database is unavailable or migration state
   is unknown.
2. Run `avalan task pgsql diagnose` and `avalan task pgsql check` from an
   operator environment with the same DSN and schema.
3. Verify provider status, network policy, SSL requirements, credentials,
   connection limits, and database role privileges.
4. If migrations are pending, take a provider backup or snapshot, run
   `avalan task pgsql migrate head`, and re-run `check`.
5. Restart workers after `check` passes. Do not replay raw inputs manually;
   inspect existing run ids and rely on idempotency for duplicate submissions.

### Unsafe Paths

Symptoms:

- Validation or materialization reports `path_escape`, `unsafe_path`, or
  `input.invalid_file`.
- A descriptor points outside the allowed task root, follows a symlink escape,
  uses an absolute path unexpectedly, or changes while being read.

Remediation:

1. Keep task definitions, agent refs, and local file descriptors under the
   configured execution or upload roots.
2. Replace symlinked uploads with regular files and retry validation before
   execution.
3. Do not add broad filesystem roots to bypass the diagnostic. Move inputs into
   a controlled workspace and preserve only sanitized file identity in
   inspection.
4. If the file grew or changed during read, regenerate the descriptor size and
   digest hints, then resubmit.

### Remote URL Disabled Or Rejected

Symptoms:

- A `remote_url` descriptor reports remote URL disabled or SSRF rejection.
- Redirects, private-network addresses, credentials in URLs, unsupported
  schemes, unknown sizes, slow reads, over-limit content, or MIME mismatches
  are rejected before fetching or dispatch.

Remediation:

1. Use provider-hosted `--hosted-url` only when the target provider is expected
   to fetch the URL. Use `remote_url` only when Avalan should fetch it.
2. Keep remote fetching disabled by default. Enable it only with an explicit
   policy covering allowed schemes, redirect behavior, private network access,
   timeout, and byte cap.
3. Remove URL credentials and move authentication to a safe fetcher or
   provider-owned file reference.
4. Prefer local upload, object-store URI, or provider file id for durable queue
   retries.

### Missing Converter

Symptoms:

- Validation reports `task.file_conversion.unsupported`,
  `task.file_conversion.dependency_missing`, or
  `task.file_delivery.missing_conversion`.
- Local text models reject native PDF, DOCX, image, or binary file blocks.

Remediation:

1. Confirm the task definition lists the conversion name in
   `input.file_conversions`.
2. Pass `--file-conversion field=name` or an SDK descriptor conversion request;
   the definition allow-list does not request conversion automatically.
3. Install the required extra for the converter, such as `task-documents` or
   `task-pdf-images`.
4. Check converter MIME support, input byte cap, output byte cap, and options
   schema before rerunning.

### Over-Limit Files

Symptoms:

- Validation, delivery planning, conversion, or artifact writes reject a file
  with a limit diagnostic.
- The diagnostic names a limit such as `task_file_bytes`,
  `artifact_bytes`, `inline_file_bytes`, `inline_text_bytes`,
  `provider_file_bytes`, `model_context_tokens`, or `conversion_bytes`.

Remediation:

1. Compare the task input limit, artifact policy, provider profile, model
   context budget, and converter cap. The most restrictive limit wins.
2. For direct runs, use provider file ids, hosted URLs, object-store URIs,
   conversion, retrieval, or map-reduce instead of inline delivery.
3. For queued runs, materialize to durable artifacts or provider references
   that survive retries.
4. Increase limits only when privacy, retention, provider, and infrastructure
   constraints permit it.

### Provider Credentials

Symptoms:

- Hosted-provider direct file runs fail with a dependency or provider
  credential diagnostic.
- Provider file ids, hosted URLs, or object-store URIs are rejected for provider
  mismatch or unsupported delivery.

Remediation:

1. Set the provider credentials required by the referenced agent, such as
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or AWS credentials
   and region for Bedrock.
2. Ensure the descriptor provider matches the target model provider. Do not send
   an OpenAI file id to an Anthropic or local target.
3. Verify provider-side file ownership, tenant scope, expiry, MIME type, and
   object-store permissions.
4. Keep raw provider handles out of logs and queue metadata; use sanitized
   inspection to debug.

### Local Dependency Failures

Symptoms:

- Local text or multimodal runs fail before model dispatch with missing backend,
  model file, tokenizer, hardware, or conversion dependencies.
- A local text model rejects native file blocks.

Remediation:

1. Install the backend extra required by the referenced agent, such as
   transformers, MLX-LM, vLLM, or DS4 support.
2. Confirm model files are present or that downloads are allowed in the target
   environment.
3. Match hardware requirements to the backend and model size. For Apple silicon,
   configure MLX or MPS where appropriate; for CUDA deployments, verify GPU
   visibility and memory.
4. Convert or chunk files to text for local text targets. Use local multimodal
   profiles only for supported image, audio, or video MIME types.

### Queue Payload Privacy Failures

Symptoms:

- `enqueue` fails because durable queued payload storage would require raw input
  or file bytes without the configured privacy keys or retention policy.
- Workers cannot reconstruct executable input from sanitized inspection
  summaries.

Remediation:

1. Provide `AVALAN_TASK_HMAC_KEY_ID` and `AVALAN_TASK_HMAC_KEY_B64` for
   user-controlled identities.
2. Configure encrypted payload or artifact storage only when the task privacy
   policy explicitly permits raw retention.
3. Prefer durable provider references or artifact refs for file inputs, not raw
   paths or sanitized summaries.
4. Re-enqueue only through SDK or CLI paths after validation passes. Do not edit
   queue payload rows manually.

### Stuck Claims

Symptoms:

- Queue health shows `claimed` growing while worker throughput is flat.
- `expired_claims` remains above zero after the lease duration.
- Runs stay in `claimed` or `running` after a worker crash.

Remediation:

1. Identify and stop unhealthy workers by worker id. Do not update claim tokens
   or run states manually.
2. Wait for the configured lease to expire. New workers cannot commit with stale
   claim tokens.
3. Run a worker for the affected queue to process available work. If operations
   tooling exposes `PgsqlTaskQueue.abandon_expired`, call it with the queue
   name, configured maximum attempts, and a bounded limit.
4. Inspect affected run ids with `avalan task inspect` and `avalan task events`
   to confirm attempts are retryable, abandoned, failed, cancelled, or
   succeeded.
5. If claims repeatedly expire, increase `--lease-seconds`, lower worker
   concurrency, check provider latency, and verify `--heartbeat-seconds` is
   shorter than the lease.

### Retention Failures

Symptoms:

- `avalan task retention-sweep` reports a retention error.
- Artifact metadata remains inspectable but raw bytes are not deleted by the
  expected deadline.
- Local artifact storage reports missing or inaccessible paths.

Remediation:

1. Verify `AVALAN_TASK_ARTIFACT_ROOT` points to the same durable artifact
   location used by workers.
2. Check filesystem permissions, mount health, available space, and backup or
   object-store lifecycle policies.
3. Run a bounded sweep by purpose, then inspect the reported deleted and lost
   counts:

   ```bash
   avalan task retention-sweep --purpose input --limit 100
   avalan task retention-sweep --purpose output --limit 100
   ```

4. Treat `lost` as an audit state for bytes that are already unavailable.
   Preserve sanitized metadata and avoid recreating raw bytes from prompts,
   model output, or external files unless the original retention policy allows
   it.
5. Fix the artifact backend and rerun sweeps until expired counts stop growing.

### Sink Failures

Symptoms:

- Prometheus or OpenTelemetry counters stop updating.
- Sink health reports `failure_count` greater than zero.
- Task runs still succeed, but external metrics or traces are incomplete.

Remediation:

1. Confirm task inspection still shows sanitized events and usage through the
   PostgreSQL store.
2. Check the sink-specific dependency extra, exporter endpoint, credentials,
   network policy, and collector quotas.
3. Keep sink label values low-cardinality. Do not add run ids, user ids,
   filenames, provider cache handles, prompts, raw model ids, or tool call ids
   as labels or span attributes.
4. Restart workers after fixing the exporter. Missing external telemetry can be
   reconstructed only from sanitized task inspection records, not from raw
   prompts or outputs.

### Privacy Sanitizer Failures

Symptoms:

- Runs fail with a privacy-classified error.
- Sanitized event payloads collapse to safe metadata for unknown or malformed
  raw events.
- Validation rejects raw retention or encryption settings.

Remediation:

1. Treat the failure as closed-by-default behavior. Do not disable privacy
   policy to unblock production traffic.
2. Inspect task validation diagnostics and sanitized events. They intentionally
   omit raw user input, raw output, file bytes, token text, tool payloads,
   exception messages, and stack traces.
3. Verify `AVALAN_TASK_HMAC_KEY_ID` and `AVALAN_TASK_HMAC_KEY_B64` are present
   for policies that HMAC user-controlled values.
4. For raw storage policies, verify explicit retention and encryption key
   configuration before re-enqueueing work.
5. If a new event type is legitimate, add an allowlisted sanitizer mapping and
   tests before enabling that event in production.

### Idempotency Conflicts

Symptoms:

- Duplicate submissions return an existing run or are rejected according to the
  task idempotency policy.
- Operators see a reservation for the same task name, version, spec hash,
  owner scope, input identity, file identity, and idempotency window.

Remediation:

1. Inspect the returned or existing run id. Do not bypass idempotency by
   modifying database rows.
2. Confirm the caller is using the intended owner scope and idempotency window.
3. For genuine duplicates, return the existing run status to the caller.
4. For distinct work that was incorrectly coalesced, change the task input,
   file identity, owner scope, task version, or idempotency window through the
   public task definition and SDK/CLI paths.
5. Rotate HMAC keys only with a planned compatibility window so existing
   reservations remain inspectable until they expire.
