# Structured task input

Avalan agents can pause a model continuation, ask one to three typed
questions, and resume with a correlated answer. The same canonical request
and resolution types are projected through the Python SDK, CLI, HTTP server,
MCP, and A2A.

Task input is for non-secret clarification. It is not an approval,
authentication, steering, or credential channel.

## Canonical behavior

An input request contains:

- a stable request and continuation identity;
- its run, turn, agent, branch, and model-call origin;
- `required` or `advisory` handling;
- a reason and optional context label;
- confirmation, text, multiline text, single-selection, or
  multiple-selection questions;
- immutable continuation and advisory deadlines.

Answers are typed and retain their provenance. A resolution is exactly one
of answered, declined, cancelled, timed out, unavailable, expired, or
superseded.

A required request stops its continuation until a terminal outcome is
committed. An advisory request may continue through a trusted default or
policy outcome after its advisory deadline. The absolute continuation
deadline always remains authoritative.

## Python SDK

Use `run_agent()` for a typed result instead of relying on an exception:

```python
from avalan import (
    AgentRunCompleted,
    AgentRunInputRequired,
    AnswerProvenance,
    InputAnswerSubmission,
    TextAnswer,
    create_attached_input_runtime,
    run_agent,
)


async def answer(context):
    question = context.request.questions[0]
    return InputAnswerSubmission(
        answers=(
            TextAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value="staging",
            ),
        ),
        provenance=AnswerProvenance.HUMAN,
    )


async with await create_attached_input_runtime(answer) as runtime:
    result = await run_agent(
        orchestrator,
        "inspect the deployment",
        interaction_runtime=runtime,
    )
    if isinstance(result, AgentRunCompleted):
        print(result.to_str())
    elif isinstance(result, AgentRunInputRequired):
        print(result.request)
```

The handler is asynchronous and receives the canonical semantic request.
Invalid answers are returned to the handler as a typed validation context;
the canonical request is not rewritten.

For detached work, implement `DurableInputBridge` and call
`create_durable_input_integration()`. The returned bundle contains:

- `runtime`, which stages a portable continuation;
- `headless_policy`, which performs the atomic durable handoff;
- `controller`, which inspects and resolves opaque request and continuation
  references.

Pass the runtime and headless policy to `run_agent()`. Persist the request
payload, continuation payload, and persistence digest in one host
transaction before acknowledging `persist_input()`. Use a unique
`ResolutionIdempotencyKey` for each logical resolution attempt.

The built-in durable provider snapshot is registered for the exact native
OpenAI Responses path and the exact native Azure OpenAI v1 Responses path.
For Azure, the snapshot binds the `azure_openai` provider family, deployment,
configuration revision, original function-call correlation, and encrypted
reasoning replay state. A fresh worker can therefore restore the portable
continuation without retaining the original client or orchestrator.

Avalan does not treat Azure's `Idempotency-Key` header as a provider-side
deduplication guarantee. A persisted continuation that has not reached
provider dispatch can be reclaimed and resumed normally. A crash that makes
the result of an Azure provider dispatch ambiguous remains fenced and is not
automatically replayed. Each claimed Azure continuation uses one provider
attempt: SDK retries and `response.failed` stream retries are disabled after
replay state is restored.

Opaque SDK references are correlation handles, not bearer credentials.
Authenticate the caller in the bridge and authorize the complete execution
scope before inspection or resolution.

## CLI

`avalan agent run` and the equivalent `avl` command use `--tty` (default
`/dev/tty`) as a separate interactive control channel. Model output can
continue on stdout while questions and answers stay on the control terminal.
If no usable control terminal exists when the run starts, the CLI does not
advertise the capability. If the channel is lost after advertisement, the
runtime returns a typed unavailable outcome instead of reading task answers
from an unrelated stream.

The model-facing capability is named `request_user_input`. When it is
available, the model can call it with one to three typed, non-secret
questions. Avalan then pauses the same logical run, renders those questions
on the control terminal, validates the answers, and returns one correlated
result to the model.

Automatic capability advertisement in the current agent-run wiring is
deliberately narrow. It requires all of the following:

- `avalan agent run` or `avl agent run`, not the standalone model runner;
- the exact native `OpenAIModel` and `OpenAIClient` Responses path;
- either an effective HTTPS endpoint on `api.openai.com` (the normal OpenAI
  SDK default satisfies this), or an explicitly configured Azure OpenAI
  resource endpoint on `*.openai.azure.com` or
  `*.cognitiveservices.azure.com` with the exact `/openai/v1/` path, the
  default HTTPS port, no URL query or user information, and either no
  `azure_api_version` option or `azure_api_version=preview`; and
- a usable controlling terminal at `--tty`.

Anthropic, Google, OpenAI-compatible services or other custom base URLs, and
local backends such as transformers, vLLM, MLX-LM, and DS4 do not currently
receive `request_user_input` automatically. Their schema projection or
parsing support does not constitute trusted production advertisement.
Pointing a compatible adapter at `api.openai.com` or an Azure resource does
not make it native, and pointing the native adapter at another endpoint
disables the capability.

Agent instructions can teach a capable model when to use the capability, but
instructions cannot enable it. If the runtime does not advertise
`request_user_input`, merely naming it in a prompt does not install a tool or
turn prose into a trusted input request.

For example:

```toml
[agent]
name = "deployment-planner"
role = "Plan safe application deployments."
instructions = """
Use request_user_input when missing information would materially change the
plan. Ask one to three concise questions, prefer typed choices with stable
values, and explain why the answer is needed. Do not request credentials,
authentication values, secrets, or approval for an action. Do not ask when a
safe answer can be inferred from the task. If request_user_input is not
available, ask the user in ordinary prose instead of inventing a tool call.
"""

[engine]
uri = "ai://env:OPENAI_API_KEY@openai/gpt-5"
```

The equivalent Azure engine uses the deployment name as the model identifier:

```toml
[engine]
uri = "ai://env:AZURE_OPENAI_API_KEY@openai/gpt-5-mini?azure_api_version=preview"
base_url = "https://tenant.openai.azure.com/openai/v1/"
```

The Azure deployment and region must support the Responses API. The current
native Azure integration uses API-key authentication; it does not infer
capability from an arbitrary OpenAI-compatible endpoint.

With a piped initial task, keep clarification on the separate terminal:

```sh
printf '%s\n' 'Plan a deployment for this service.' \
    | avalan agent run deployment-planner.toml --tty /dev/tty
```

The renderer:

- presents required/advisory state, reason, help, defaults, and choices;
- validates typed and multiline answers before submission;
- pauses streamed display while an attached prompt owns the terminal;
- renders model-authored text literally, strips terminal control sequences,
  and never executes markup, links, escapes, or embedded commands;
- does not echo answers as diagnostic output.

Use `--tty PATH` only for a trusted terminal device. Do not redirect it to a
log file or shared input stream.

## HTTP server

OpenAI-compatible clients negotiate the extension with:

```text
Avalan-Extensions: https://avalan.ai/extensions/task-input/v1
```

When attached handling is unavailable, the response exposes a typed
`input_required` envelope with an opaque request identifier. Authenticated
controllers use:

- `GET /v1/input/requests/{request_id}` to inspect;
- `GET /v1/input/requests/{request_id}/poll` to wait or resume a retained
  JSON/SSE segment;
- `POST /v1/input/requests/{request_id}/resolve` to answer or decline;
- `POST /v1/input/requests/{request_id}/cancel` to cancel.

Resolution requests must carry the expected state revision and an
idempotency key. Responses include state/store revision headers and
`Cache-Control: private, no-store`.

Configure a server authentication resolver and an interaction authorizer.
Missing, unauthorized, and out-of-scope identifiers are intentionally
non-enumerating. Never derive a principal from request content.

## MCP

MCP elicitation is pinned to protocol version `2025-11-25`. A client must
negotiate elicitation capability, complete initialization, preserve the
owner-bound session, and support reverse request routing before Avalan
advertises form handling.

Canonical questions are projected into the restricted
`elicitation/create` form grammar. Stable choice values remain distinct from
display labels, multiline support is capability-checked, and responses are
size-bounded and correlated to the pending request.

Passwords, keys, tokens, payment credentials, private keys, MFA values, and
authentication challenges are rejected before form projection. If a product
needs a URL-based sensitive or authentication flow, route it through a
separate host-controlled MCP URL flow; do not return that value as task
input. A client that cannot route and resume the negotiated interaction
receives a typed unavailable outcome.

## A2A

A2A 1.0 negotiation uses the extension URI:

```text
https://avalan.ai/extensions/task-input/v1
```

The agent card advertises the extension. Requests opt in through
`A2A-Extensions`, and structured request/resolution metadata is carried
under the same URI. Input-required work remains in the same A2A task and
context; a correlated message resumes it.

Peers without the extension receive a readable fallback. That fallback is
non-authoritative and literal-safe: controls, newlines, quotes, and
backslashes are escaped, and model-authored markup has no execution
semantics.

## Lifecycle and recovery

The durable lifecycle is:

```text
created -> pending -> answered/declined/cancelled/timed_out/
                      unavailable/expired/superseded
                  -> continuation ready -> claimed -> completed
```

Creation and suspension are atomic. Resolution uses compare-and-swap state
revision checks. The idempotency ledger distinguishes same-key replay,
semantic replay under a new key, and conflicting content. A worker claims a
ready continuation with a lease and fencing token; only the winning claim
may dispatch it.

After a crash:

1. inspect the persisted request and continuation;
2. replay the same idempotency key for the same logical answer;
3. claim only the current store revision;
4. reconstruct from the encrypted portable continuation;
5. reject revision drift or ambiguous provider dispatch;
6. commit completion, then release retained continuation material under the
   configured retention policy.

Do not reconstruct a continuation from logs or a client-supplied prompt.

## Authorization and approval isolation

Every create, inspect, list, wait, resolve, cancel, and supersede operation is
authorized against the authenticated principal and complete run/branch
scope. Parent and sibling branches do not inherit content access merely
because their identifiers are related.

Words such as `yes`, `continue`, or `do it` are ordinary task answers. They
cannot approve a protected tool call. Protected actions must use their
separate confirmation callback and exact tool-call identity. Approval or
authentication responses must never be delivered to the model as a task
answer.

## Privacy and retention

Treat prompts, headers, help text, choice labels and values, defaults,
answers, multiline content, transcripts, observations, and portable
continuations as interaction content.

Durable stores encrypt that content at rest. Authorized reads decrypt only
after a scope-filtered ownership lookup. Retention sweeps invalidate expired
continuations, then delete records, branches, continuations, idempotency
keys, and resumption outbox payloads after the retention deadline.

Lifecycle telemetry is content-free. It may contain opaque hashed
correlation identifiers, state and resolution categories, surface,
wait duration, validation code, duplicate/stale flags, and provenance
category. It must not contain prompts, choices, defaults, answers, raw
continuations, or secret-like values. Diagnostics use stable safe codes.

Task input rejects requests that collect credentials or authentication
material. Submitted free-form values pass through the configured host
classifier. A secret-like rejection leaves the canonical candidate
unchanged for the caller, commits no resolution, and emits no raw value.

## Capacity and cleanup

The default process admission limit is 1,024 pending interactions. Capacity
rejection occurs before model-visible suspension and returns a typed,
content-safe error.

Waiters are event-driven; they do not poll in a busy loop. Any commit that
advances a matching revision wakes all eligible waiters. Resolution,
cancellation, timeout, handler loss, task termination, and runtime close
remove waiter registrations and release controller leases. Closing one
in-memory handle does not cancel work owned by another open handle.

Operators should:

1. configure authentication, authorization, encryption keys, and retention;
2. verify the extension or protocol capability before dispatch;
3. monitor only content-free lifecycle categories and capacity;
4. sweep expired and retention-eligible interactions on a bounded schedule;
5. close runtimes and sessions during shutdown and await their cleanup;
6. keep approval, steering, and authentication on separate trusted channels.
