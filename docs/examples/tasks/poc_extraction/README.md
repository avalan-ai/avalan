# Invoice Extraction Task

This directory contains sanitized fixtures for extracting invoice line items
from one PDF. The native task sends the PDF directly to the hosted agent. The
image-flow task renders the PDF into ordered page images with built-in task
conversion, then sends only those generated image files to the same hosted
agent.

Both variants share the same agent instructions, user prompt, reasoning effort,
disabled tool selection, structured output format, and output schema. The only
provider envelope difference is the media block set: the native task sends one
`application/pdf` file block, while the image-flow task sends ordered
`image/png` image blocks.

Default tracked command from the repository root:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml \
  --ephemeral \
  --pdf docs/examples/tasks/poc_extraction/sample.pdf \
  --json \
  --output extraction.json
```

Equivalent low-level file command:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml \
  --ephemeral \
  --file input=docs/examples/tasks/poc_extraction/sample.pdf \
  --file-mime input=application/pdf \
  --json \
  --output extraction.json
```

Equivalent command after changing into this directory:

```bash
poetry run avalan task run task.toml --ephemeral --pdf sample.pdf --json --output extraction.json
```

Image-flow command after changing into this directory:

```bash
poetry run avalan task run image_flow_task.toml --ephemeral --pdf sample.pdf --json --output image.json
```

Equivalent image-flow command from the repository root:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/image_flow_task.toml \
  --ephemeral \
  --pdf docs/examples/tasks/poc_extraction/sample.pdf \
  --json \
  --output image.json
```

Direct flow command after changing into this directory:

```bash
poetry run avalan flow run image_flow.toml --pdf sample.pdf --json --output image.json
```

CLI input files passed with `--pdf` and `--file` resolve from the caller's
current directory. TOML references such as agent, flow, and schema files resolve
from the TOML file that declares them. Output paths resolve from the caller's
current directory.

Validation checks expected for these fixtures:

* `task.toml` and `image_flow_task.toml` both accept one `application/pdf`
  input and reject pre-rendered `image/png` input at the task boundary.
* `image_flow.toml` uses a `file_convert` node with the `pdf_image` converter
  and an agent node configured with `files_input = "render_pages.files"` and
  `file_policy = "replace"`.
* The image-flow provider request must not include the original PDF block when
  replacement is configured.

The tracked sample is intentionally synthetic. It is suitable for validating
prompt composition, inline PDF delivery, reasoning settings, and structured
output shape without customer documents, network access, live services, or
provider credentials.

The image-flow path is semantic parity for this example, not a general
replacement for native PDF understanding. Rasterization drops text layers,
document structure, attachments, signatures, and hidden or non-rendered content.
Quality, latency, and cost depend on DPI, page count, converter behavior, and
provider vision limits.

Live smoke runs should be kept outside the default CI path and skipped unless
all of the following are true:

* A sanitized non-customer PDF is available.
* `AZURE_OPENAI_API_KEY` is set for the target deployment.
* The agent `engine.uri` names the intended deployment.
* The agent `engine.base_url` points at the intended Azure OpenAI `/openai/v1/`
  endpoint.
* The expected output is a JSON object with a non-empty `line_items` array.
