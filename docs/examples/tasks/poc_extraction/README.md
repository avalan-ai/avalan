# Invoice Extraction Task

This directory contains a sanitized direct task fixture for extracting
invoice line items from one PDF. The task owns the input and output
contract, and the agent owns the provider instructions, user prefix, reasoning
effort, file delivery profile, disabled tool selection, and structured output
format.

Default tracked command:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml --ephemeral --pdf ./sample.pdf --json --output extraction.json
```

Equivalent low-level file command:

```bash
poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml \
  --ephemeral \
  --file input=./sample.pdf \
  --file-mime input=application/pdf \
  --json \
  --output extraction.json
```

The tracked sample is intentionally synthetic. It is suitable for validating
prompt composition, inline PDF delivery, reasoning settings, and structured
output shape without customer documents, network access, live services, or
provider credentials.

Live smoke runs should be kept outside the default CI path and skipped unless
all of the following are true:

* A sanitized non-customer PDF is available.
* `AZURE_OPENAI_API_KEY` is set for the target deployment.
* The agent `engine.uri` names the intended deployment.
* The agent `engine.base_url` points at the intended Azure OpenAI `/openai/v1/`
  endpoint.
* The expected output is a JSON object with a non-empty `line_items` array.
