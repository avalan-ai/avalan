# DS4 Native Backend

Avalan can run DS4-supported DeepSeek V4 Flash GGUF files through the
native `ds4` backend. This backend is intentionally model-specific: it is
not a generic GGUF loader, and it should not be used for arbitrary GGUF
models.

Install the optional backend before running DS4 models:

```bash
python3 -m pip install -U "avalan[ds4]"
```

The production targets are macOS arm64 with Metal and Linux with CUDA. The
CPU backend exists for diagnostics and correctness checks only.

DS4 model files are opened directly from the local filesystem. Avalan does
not require `HF_TOKEN` and does not contact Hugging Face for DS4 runs.

## Basic Generation

Set a model path first. The file must be a DS4-supported DeepSeek V4 Flash
GGUF:

```bash
export DS4_MODEL=/path/to/ds4flash.gguf
```

Run greedy generation with an explicit native backend:

```bash
printf '%s\n' 'Write a short greeting.' \
  | avalan model run "ai://local/${DS4_MODEL}?backend=ds4" \
      --ds4-ctx 4096 \
      --ds4-native-backend metal \
      --max-new-tokens 64 \
      --temperature 0
```

Use `--ds4-native-backend cuda` on Linux CUDA builds. Use
`--ds4-native-backend cpu` only for diagnostics.

Relative paths are resolved from the current working directory. Absolute
paths use two slashes after `local`, because the first slash separates the
URI authority from the path and the second slash belongs to the filesystem
path:

```bash
avalan model run "ai://local/../pyds4/.local/ds4/ds4flash.gguf" --backend ds4
avalan model run "ai://local//Users/me/models/ds4flash.gguf" --backend ds4
```

Percent-encode path characters that are not URL-safe. An encoded absolute
path is also accepted:

```bash
avalan model run "ai://local/%2FUsers/me/DS4%20models/ds4flash.gguf" --backend ds4
```

The DS4-specific flags are available on both `model run` and `agent run`:

- `--ds4-ctx`
- `--ds4-native-backend`
- `--ds4-mtp`
- `--ds4-mtp-draft`
- `--ds4-mtp-margin`
- `--ds4-warm-weights`
- `--ds4-quality`
- `--with-ds4-native-log` / `--ds4-native-log` / `--no-ds4-native-log`

## Disabled Reasoning

Disable Avalan reasoning parsing and request DS4's no-thinking mode with
`--no-reasoning`:

```bash
printf '%s\n' 'Answer in one sentence: what is distillation?' \
  | avalan model run \
      "ai://local/${DS4_MODEL}?backend=ds4&ds4_ctx=4096&ds4_native_backend=metal" \
      --no-reasoning \
      --max-new-tokens 64 \
      --temperature 0
```

Reasoning efforts map to DS4 thinking modes when reasoning remains enabled.
For example, `--reasoning-effort high` requests DS4 high thinking mode, and
`--reasoning-effort max` requests max thinking mode.

## URI Backend Config

DS4 backend settings can be provided in the `ai://` URI with `ds4_`
parameters:

```bash
printf '%s\n' 'List two practical GPU memory tips.' \
  | avalan model run \
      "ai://local/${DS4_MODEL}?backend=ds4&ds4_ctx=8192&ds4_native_backend=cuda&ds4_warm_weights=true&ds4_quality=true" \
      --max-new-tokens 96 \
      --temperature 0
```

Common URI keys are:

- `backend=ds4`
- `ds4_ctx`
- `ds4_native_backend`
- `ds4_mtp`
- `ds4_mtp_draft`
- `ds4_mtp_margin`
- `ds4_warm_weights`
- `ds4_quality`
- `ds4_native_log`

Avalan suppresses DS4 native startup logs by default. Use
`--with-ds4-native-log`, `--ds4-native-log`, or `ds4_native_log=true` when
debugging native engine startup.

Advanced URI-only keys include:

- `ds4_directional_steering_file`
- `ds4_directional_steering_attn`
- `ds4_directional_steering_ffn`

MTP and directional steering file paths are validated before native engine
open. Directional steering coefficients require
`ds4_directional_steering_file`.

Unknown `ds4_` URI parameters are rejected so configuration mistakes fail
early.

## Agent Tools

DS4 uses its native DSML tool protocol internally. You can run the same
agent-tool flow as other local backends by pointing the engine URI at the
DS4 GGUF:

```bash
printf '%s\n' 'What is (4 + 6) and then that result times 5, divided by 2?' \
  | avalan agent run \
      --engine-uri "ai://local/${DS4_MODEL}" \
      --backend ds4 \
      --ds4-ctx 4096 \
      --ds4-native-backend metal \
      --tool "math.calculator" \
      --run-max-new-tokens 256 \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --display-events \
      --display-tools
```

Do not override DS4's internal tool protocol with a generic tool format.
Avalan keeps DSML for DS4 prompt rendering and exact replay even when other
backends use JSON, ReAct, OpenAI, or Harmony formats.

## Current Limitations

- DS4 only supports DS4-supported DeepSeek V4 Flash GGUF files in Avalan.
- Generic GGUF models are not supported by this backend.
- Native DS4 tool calls use DSML and are still treated as DS4-specific.
  Avalan renders tool schemas, parses completed DSML tool blocks, streams
  argument deltas, and preserves exact raw DSML replay metadata for session
  alignment.
- CPU inference is a debug/reference path, not a production target.

## Integration Tests

Real-model DS4 tests are skipped unless a model path is configured:

```bash
AVALAN_DS4_MODEL=/path/to/ds4flash.gguf \
AVALAN_DS4_BACKEND=metal \
AVALAN_DS4_CTX=4096 \
poetry run pytest --verbose -s \
  tests/model/ds4_integration_test.py \
  tests/model/ds4_phase3_integration_test.py
```

`AVALAN_DS4_BACKEND` accepts `metal`, `cuda`, or `cpu`; use `cpu` only for
diagnostic runs. `AVALAN_DS4_CTX` defaults to `4096` when omitted.
