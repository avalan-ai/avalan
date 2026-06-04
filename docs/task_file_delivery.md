# Task File Delivery

Task file inputs are planned against a model file-delivery profile before they
are sent to an agent target. Unknown providers and unknown local backend
capabilities fail closed.

## Delivery Modes

| Mode | Meaning |
| --- | --- |
| `provider_file_id` | A durable file identifier already owned by the provider. |
| `hosted_url` | A provider-fetchable HTTPS URL. |
| `object_store_uri` | A provider-fetchable object-store URI such as `s3://` or `gs://`. |
| `inline_bytes` | A bounded base64 payload sent in the model request. |
| `inline_text` | Bounded text sent in the prompt or message content. |
| `converted_artifact` | A task artifact converted to another accepted format. |
| `retrieval_context` | Retrieved chunks injected as bounded context. |
| `map_reduce_context` | File content processed through bounded map-reduce steps. |
| `reject` | No safe delivery path is available. |

## Direct Input Descriptors

Direct task runs accept local file paths, explicit JSON descriptors, and
provider-native references. Descriptor hints are safe metadata used for
validation and provider delivery planning; they do not cause raw bytes to be
stored by themselves.

```bash
avalan task run tasks/review.task.toml --ephemeral \
  --provider-file-id document=openai:file_abc123 \
  --file-mime document=application/pdf \
  --file-role document=source \
  --file-size document=2048 \
  --file-sha256 document=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

avalan task run tasks/review.task.toml --ephemeral \
  --hosted-url document=openai:https://files.example.test/report.pdf

avalan task run tasks/review.task.toml --ephemeral \
  --object-store-uri document=google:gs://bucket/report.pdf

avalan task run tasks/review.task.toml --ephemeral \
  --file-descriptor 'document={"source_kind":"remote_url","reference":"https://example.test/report.txt","mime_type":"text/plain"}'

avalan task run tasks/review.task.toml --ephemeral \
  --pdf docs/examples/playground/invoice.pdf \
  --json
```

Use `--file-conversion field=name` or
`--file-conversion field=name:{"option":"value"}` to request a conversion for a
specific descriptor. The task definition's `input.file_conversions` value is an
allow-list of conversion names a run may request; it is not an automatic
conversion request. Provider-native references cannot request conversions,
because the provider already owns the referenced file.

For queued file runs, local files must be materialized into durable artifacts or
represented by durable provider references before workers execute them. Provider
file ids and object-store URIs can survive queue retries when their provider,
owner scope, expiry, MIME type, size bucket, and identity HMAC are recorded.
Expiring provider handles are direct-run-only unless the caller can refresh
them through a separate durable workflow.

Remote URL descriptors are different from provider-fetchable hosted URLs.
`remote_url` asks Avalan to fetch the URL and is disabled unless a remote URL
policy is configured with SSRF protections. `hosted_url` asks the target
provider to fetch a provider-compatible URL and is planned through the model
file-delivery profile.

Task file inputs are scoped to a single task run. The runner validates the
descriptor, materializes local bytes only when the selected delivery plan needs
an artifact, and passes the resulting file block or provider reference to the
target. This path does not populate recent message memory, permanent message
memory, document memory, embeddings, or retrieval stores by default. Document
loading through `avalan memory document ...` is a separate ingestion workflow:
it chunks and indexes document text for later retrieval. A task file enters a
text path only when the task declares an allowed conversion or when the target
profile selects retrieval or map-reduce context for that run.

## SDK Descriptor Examples

```python
from avalan.task import (
    TaskClient,
    TaskFileDescriptor,
    TaskProviderReferenceKind,
)

local_descriptor = TaskClient.local_file(
    "uploads/report.pdf",
    mime_type="application/pdf",
    size_bytes=2048,
    conversions=(TaskClient.file_conversion("text"),),
)

provider_descriptor = TaskFileDescriptor.provider_reference_descriptor(
    "file_abc123",
    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
    provider="openai",
    owner_scope="tenant-a",
    role="source",
    mime_type="application/pdf",
    size_bucket="small",
    identity_hmac="hmac:file",
)
```

See `docs/examples/tasks/file_inputs_sdk.py` for complete SDK helpers covering
local documents, provider file ids, hosted URLs, and object-store URIs.

## Capability Matrix

| Profile | Source kinds | Native modes | Text modes | Object-store schemes | MIME families | Requirements and limits |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | local path, remote URL, artifact, inline bytes | provider file id, hosted URL, inline bytes | inline text, converted artifact, retrieval context, map-reduce context | none by default | all | Requires OpenAI credentials such as `OPENAI_API_KEY`; provider file and context limits come from capability data. |
| Anthropic | local path, remote URL, artifact, inline bytes | provider file id, hosted URL, inline bytes | inline text, converted artifact, retrieval context, map-reduce context | none by default | all | Requires Anthropic credentials such as `ANTHROPIC_API_KEY`; provider file and context limits come from capability data. |
| Google/Gemini | local path, remote URL, artifact, inline bytes | provider file id, hosted URL, object-store URI, inline bytes | inline text, converted artifact, retrieval context, map-reduce context | `gs` | all | Requires Google/Gemini credentials such as `GOOGLE_API_KEY`; `gs://` references must be readable by the provider or runtime identity. |
| Bedrock | local path, remote URL, artifact, inline bytes | object-store URI, inline text-compatible bytes | inline text, converted artifact, retrieval context, map-reduce context | `s3` | text, JSON, Markdown, XML | Requires AWS credentials, region, Bedrock model access, and `s3://` objects readable by the provider/runtime identity. |
| Local text | local path, artifact, inline bytes | none | inline text, converted artifact, retrieval context, map-reduce context | none | text, JSON, Markdown, XML | Requires the local backend extra and model files. Native file blocks are rejected; convert or chunk to text. |
| Local multimodal | local path, artifact, inline bytes | inline bytes | inline text, converted artifact, retrieval context, map-reduce context | none | audio, image, video | Requires local multimodal backend support, hardware appropriate for the model, and local model downloads. PDF, DOCX, and binary files need conversion or are rejected. |
| Unknown | none | none | none | none | none | Fails closed until a provider or backend profile is configured. |

Optional extras are scoped by capability: `task` for structured validation,
`agent` for agent-backed task execution, `task-pgsql` for durable stores and
workers, `task-documents` for document conversion, `task-prometheus` for
metrics, and `task-otel` for traces. Local models may also require backend
extras such as transformers, MLX-LM, vLLM, or DS4 plus the hardware and model
downloads those backends need.

## Limit Composition

Delivery planning composes these limits in the most restrictive direction:

1. Task input limits such as file count and file bytes.
2. Artifact policy limits such as artifact bytes and retention.
3. Provider file limits from provider capability data.
4. Model context limits for inline text, retrieval, and map-reduce context.
5. Converter limits before any full-content conversion is attempted.

The model profile records limit sources as data. Task runners should report
the relevant limit name and a sanitized size bucket rather than leaking exact
file details through events, inspection, metrics, logs, or traces.
