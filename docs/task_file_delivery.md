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

## Capability Matrix

| Profile | Source kinds | Native modes | Text modes | Object-store schemes | MIME families |
| --- | --- | --- | --- | --- | --- |
| OpenAI | local path, remote URL, artifact, inline bytes | provider file id, hosted URL, inline bytes | inline text, converted artifact, retrieval context, map-reduce context | none by default | all |
| Anthropic | local path, remote URL, artifact, inline bytes | provider file id, hosted URL, inline bytes | inline text, converted artifact, retrieval context, map-reduce context | none by default | all |
| Google/Gemini | local path, remote URL, artifact, inline bytes | provider file id, hosted URL, object-store URI, inline bytes | inline text, converted artifact, retrieval context, map-reduce context | `gs` | all |
| Bedrock | local path, remote URL, artifact, inline bytes | object-store URI, inline text-compatible bytes | inline text, converted artifact, retrieval context, map-reduce context | `s3` | text, JSON, Markdown, XML |
| Local text | local path, artifact, inline bytes | none | inline text, converted artifact, retrieval context, map-reduce context | none | text, JSON, Markdown, XML |
| Local multimodal | local path, artifact, inline bytes | inline bytes | inline text, converted artifact, retrieval context, map-reduce context | none | audio, image, text, video, JSON, Markdown, XML |
| Unknown | none | none | none | none | none |

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
