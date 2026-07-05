# CLI

The CLI offers the following commands, some of them with multiple subcommands:

* [avalan](#avalan)
* [avalan agent](#avalan-agent)
  * [avalan agent message](#avalan-agent-message)
  * [avalan agent message search](#avalan-agent-message-search)
  * [avalan agent run](#avalan-agent-run)
  * [avalan agent serve](#avalan-agent-serve)
  * [avalan agent proxy](#avalan-agent-proxy)
  * [avalan agent init](#avalan-agent-init)
* [avalan cache](#avalan-cache)
  * [avalan cache delete](#avalan-cache-delete)
  * [avalan cache download](#avalan-cache-download)
  * [avalan cache list](#avalan-cache-list)
* [avalan deploy](#avalan-deploy)
  * [avalan deploy run](#avalan-deploy-run)
* [avalan flow](#avalan-flow)
  * [avalan flow run](#avalan-flow-run)
  * [avalan flow validate](#avalan-flow-validate)
  * [avalan flow compile](#avalan-flow-compile)
  * [avalan flow graph inspect](#avalan-flow-graph-inspect)
  * [avalan flow mermaid](#avalan-flow-mermaid)
  * [avalan flow inspect](#avalan-flow-inspect)
  * [avalan flow trace](#avalan-flow-trace)
  * [avalan flow cancel](#avalan-flow-cancel)
  * [avalan flow resume](#avalan-flow-resume)
* [avalan task](#avalan-task)
  * [avalan task validate](#avalan-task-validate)
  * [avalan task run](#avalan-task-run)
  * [avalan task enqueue](#avalan-task-enqueue)
  * [avalan task inspect](#avalan-task-inspect)
  * [avalan task usage](#avalan-task-usage)
  * [avalan task output](#avalan-task-output)
  * [avalan task events](#avalan-task-events)
  * [avalan task artifacts](#avalan-task-artifacts)
  * [avalan task worker](#avalan-task-worker)
  * [avalan task retention-sweep](#avalan-task-retention-sweep)
  * [avalan task pgsql](#avalan-task-pgsql)
* [avalan memory](#avalan-memory)
  * [avalan memory embeddings](#avalan-memory-embeddings)
  * [avalan memory search](#avalan-memory-search)
  * [avalan memory document](#avalan-memory-document)
  * [avalan memory document index](#avalan-memory-document-index)
* [avalan model](#avalan-model)
  * [avalan model display](#avalan-model-display)
  * [avalan model install](#avalan-model-install)
  * [avalan model run](#avalan-model-run)
  * [avalan model search](#avalan-model-search)
  * [avalan model uninstall](#avalan-model-uninstall)
* [avalan tokenizer](#avalan-tokenizer)
* [avalan train](#avalan-train)
  * [avalan train run](#avalan-train-run)

If you want to list all available commands and global options, run:

```bash
avalan --help
```

If you want help on a specific command, add `--help` to the command, for
example:

```bash
avalan model --help
```

Some commands, like `model`, contain subcommands of their own, which are listed
when showing the help for the command. You can learn about a subcommand (like
`run` for `model`) set of options with:

```bash
avalan model run --help
```

Global options may affect more than one command. For example, to change the
output language from the default english to spanish, add the `locale` option,
specifying `es` as the locale:

```bash
avalan model run meta-llama/Meta-Llama-3-8B-Instruct --locale es
```

The CLI uses the `fancy` theme by default. Add `--theme basic` to opt into a
simpler text-first theme:

```bash
avalan model run MODEL --theme basic
```

![Running the CLI in spanish](https://avalan.ai/images/spanish_translation.png)

You'll need your Huggingface access token exported as `HF_TOKEN`.

> [!TIP]
> If you are on an Apple silicon chip, run the
> [configure_mlx.sh](https://github.com/avalan-ai/avalan/blob/main/scripts/configure_mlx.sh)
> script, created by [@AlexCheema](https://github.com/AlexCheema), which
> empirically reduces the time to first token and the tokens per second ratio.

## Skills Toolset

Skills are trusted instruction resources exposed through the `skills`
namespace. Agents think, tools act, skills teach, and registries disclose.
The CLI can provide trusted skill sources for agent commands; model-facing
tools can then discover or read those skills during the normal tool loop.

Enable the namespace or individual tools with `--tool`:

```bash
echo "Use the PDF skill to decide the review workflow." \
  | avalan agent run docs/examples/agent_skills_pdf.toml \
      --tool skills.match \
      --tool skills.read \
      --tool-skills-source workspace-main=docs/examples/skills \
      --tool-skills-source-authority workspace-main=workspace:docs \
      --tool-skills-skill pdf \
      --tool-skills-bootstrap auto \
      --tool-skills-max-bytes-per-read 65536 \
      --display-tools \
      --display-events
```

Trusted skill source flags are available on `avalan agent message search`,
`avalan agent run`, `avalan agent serve`, `avalan agent proxy`, and
`avalan agent init`:

| Flag | Effect |
| --- | --- |
| `--tool-skills-source LABEL=PATH` | Add a trusted filesystem source. |
| `--tool-skills-file LABEL=PATH` | Add a trusted direct `SKILL.md` or `SKILL-*.md` manifest file source. The label is allowed as a skill ID by default, so use the manifest's normalized `name`/skill ID when relying on auto-enable. |
| `--tool-skills-file-no-auto-enable` | Do not automatically add manifest file labels to the allowed skill IDs. |
| `--tool-skills-source-authority LABEL=KIND[:ID]` | Assign `bundled`, `workspace`, `user_local`, `plugin_provided`, or `preinstalled_remote` authority. |
| `--tool-skills-source-package LABEL=PATH` | Select a package directory under the source. |
| `--tool-skills-source-allow-hidden LABEL` | Allow hidden paths inside that trusted source. |
| `--tool-skills-authority-kind KIND` | Restrict trusted authority kinds. |
| `--tool-skills-skill ID` | Allow only a logical skill ID. |
| `--tool-skills-disable` | Disable trusted skills settings. |
| `--tool-skills-bootstrap {auto,off}` | Control the compact bootstrap instruction. |
| `--tool-skills-diagnostics {off,standard,verbose}` | Control diagnostic path/detail exposure. |
| `--tool-skills-observability {off,standard,verbose}` | Control skill audit event verbosity. |
| `--tool-skills-max-bytes-per-read N` | Bound bytes returned by one read. |
| `--tool-skills-max-lines-per-read N` | Bound lines returned by one read. |
| `--tool-skills-max-skills N` | Bound indexed skills. |
| `--tool-skills-max-resources-per-skill N` | Bound resources declared by one skill. |
| `--tool-skills-max-indexed-bytes N` | Bound indexed registry bytes. |
| `--tool-skills-max-sources N` | Bound trusted source count. |
| `--tool-skills-max-resources-per-source N` | Bound resources scanned per source. |
| `--tool-skills-max-source-depth N` | Bound source directory depth. |
| `--tool-skills-max-files-per-source N` | Bound files scanned per source. |
| `--tool-skills-max-directory-entries-per-source N` | Bound directory entries scanned per source. |
| `--tool-skills-max-active-cursors N` | Bound active read cursors. |
| `--tool-skills-max-cursor-age-seconds N` | Bound cursor lifetime. |

For one tracked skill file, use `--tool-skills-file` instead of trusting a
whole directory. The `LABEL` should match the manifest skill ID unless you
also pass `--tool-skills-skill` explicitly:

```bash
echo "Use the PDF skill to decide the review workflow." \
  | avalan agent run docs/examples/agent_skills_pdf.toml \
      --tool-skills-file pdf=docs/examples/skills/pdf/SKILL.md \
      --tool-skills-source-authority pdf=workspace:docs
```

Operator inspection uses the existing runtime surfaces. Use
`--display-tools` and `--display-events` to inspect `skills.list`,
`skills.match`, `skills.read`, `skills.check`, response envelopes, statuses,
and diagnostics in agent runs. For durable flows and tasks, use
`avalan flow inspect`, `avalan flow trace`, `avalan task inspect`,
`avalan task events`, `avalan task output`, and `avalan task artifacts`.

Flow and task definitions can contain `[skills]` sections, but flow/task CLI
input is not a source authority. Trusted settings for those surfaces come
from SDK, host, registry, or worker configuration; queued workers revalidate
the captured skills identity before execution.

## Shell Toolset

Shell tools are opt-in policy-limited tools for inspecting files under a
configured workspace and creating bounded generated artifacts. Enable the
namespace with TOML:

```toml
[tool]
enable = ["shell.rg", "shell.head"]

[tool.shell]
workspace_root = "."
materialized_input_files_dir = "avalan-input-files"
max_stdout_bytes = 65536
```

Agent CLI commands also accept scalar shell settings:

```bash
avalan agent run \
  --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4.1-mini" \
  --tool shell.rg \
  --tool-shell-workspace-root . \
  --tool-shell-materialized-input-files-dir avalan-input-files \
  --tool-shell-max-stdout-bytes 65536
```

Structured shell pipelines are a separate explicit opt-in. The model receives
`shell.pipeline` as a typed tool with `steps` objects, not as a shell string.
Use these flags only from trusted operator configuration:

| Flag | Effect |
| --- | --- |
| `--tool-shell-max-pipeline-stages N` | Cap the number of stages in one composition. |
| `--tool-shell-max-pipeline-bytes N` | Cap final aggregate pipeline stdout bytes. |
| `--tool-shell-max-intermediate-bytes N` | Cap stdout bytes Avalan retains or routes between stages in buffered transport. |
| `--tool-shell-pipeline-transport {buffered,native}` | Choose whether `shell.pipeline` routes bytes through Avalan buffers or native OS pipes. |
| `--tool-shell-allow-pipelines` | Expose `shell.pipeline` when the enabled tool selection includes `shell.pipeline`, `shell`, or `shell.*`. |

```bash
avalan agent run docs/examples/agent_shell_pipeline.toml \
  --tool shell.pipeline \
  --tool-shell-workspace-root . \
  --tool-shell-max-pipeline-stages 3 \
  --tool-shell-max-pipeline-bytes 1048576 \
  --tool-shell-max-intermediate-bytes 262144 \
  --tool-shell-pipeline-transport buffered \
  --tool-shell-allow-pipelines
```

Flow and task runtimes can use the same pipeline settings when the host
application builds a `ToolManager` for strict tool nodes. The standalone
`avalan flow validate` command uses the default CLI flow registry and does not
accept tool-runtime flags, so it is not the runtime-availability validator for
these examples. The tracked flow-backed pipeline examples are validated in CI
with a configured `ToolManager`:

```bash
poetry run pytest \
  tests/flow/validator_test.py::FlowValidatorTestCase::test_docs_shell_pipeline_flow_examples_validate_with_runtime \
  tests/task/task_loader_test.py::TaskDefinitionLoaderTest::test_docs_shell_pipeline_task_examples_load \
  -q
```

Queued task workers fail closed for `shell.pipeline` unless the worker process
is started with explicit operator pipeline settings:

```bash
avalan task worker \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --queue default \
  --tool shell.pipeline \
  --tool-shell-max-pipeline-stages 3 \
  --tool-shell-pipeline-transport buffered \
  --tool-shell-allow-pipelines
```

Trusted deployment TOML may select isolated shell execution. `[tool.sandbox]`
and `[tool.container]` define operator-controlled profiles; `[tool.shell.sandbox]`
and `[tool.shell.container]` select one of those profiles for shell tools.
Models and untrusted request data must not define images, mounts, sandbox
roots, executable paths, backend flags, network policy, or secrets. Constrain
agent TOML with operator caps before loading files that are not trusted
deployment configuration.

Supported shell execution modes are `local`, `sandbox`, and `container`.
Supported sandbox backends are `seatbelt` and `bubblewrap`; supported
container backends are `docker` and `apple-container`. Container policy flags
require `--tool-shell-backend container`; sandbox policy flags require
`--tool-shell-backend sandbox`. See [Isolation execution](ISOLATION.md) for
trusted policy fields, platform limits, approval behavior, and diagnostics.
Full byte-stream pipelines are local-only; sandbox or container
`mode="pipeline"` and any `stdin_from` composition fail closed.

Use `shell` or `shell.*` to enable the full namespace, or a concrete name such
as `shell.rg` for one command. An empty `[tool.shell]` section enables shell
tools with defaults. Shell settings alone opt the agent CLI into shell tools
without requiring a separate `--tool`; strict flow tool nodes still execute
through the enabled `ToolManager`.

Public shell tools:

| Tool | Use | Dependency group | Typical binary package |
| --- | --- | --- | --- |
| `shell.rg` | Search workspace text with ripgrep. | core | `ripgrep` |
| `shell.head` | Read leading lines from one text file. | core | `coreutils` |
| `shell.tail` | Read trailing lines from one text file. | core | `coreutils` |
| `shell.ls` | List a directory or file path. | core | `coreutils` |
| `shell.cat` | Read a bounded text file. | core | `coreutils` |
| `shell.nl` | Number lines in a bounded text file. | core | `coreutils` |
| `shell.file` | Identify regular file types. | core | `file` |
| `shell.find` | Find entries with constrained selectors. | core | `findutils` |
| `shell.wc` | Count lines, words, or bytes. | core | `coreutils` |
| `shell.awk` | Select constrained fields and lines. | text filters | `gawk` or `mawk` |
| `shell.sed` | Select constrained line ranges and patterns. | text filters | `sed` |
| `shell.jq` | Transform JSON with a constrained jq filter. | JSON | `jq` |
| `shell.pdfinfo` | Inspect PDF metadata and page boxes. | Poppler | `poppler-utils` or `poppler` |
| `shell.pdftotext` | Extract text from a PDF. | Poppler | `poppler-utils` or `poppler` |
| `shell.pdftoppm` | Rasterize bounded PDF pages. | Poppler | `poppler-utils` or `poppler` |
| `shell.reportlab` | Create one bounded generated PDF from text. | Python PDF | `python3` with `avalan` and `reportlab` |
| `shell.pdfplumber` | Extract bounded text or tables from a PDF. | Python PDF | `python3` with `avalan` and `pdfplumber` |
| `shell.pypdf` | Inspect metadata or extract bounded text from a PDF. | Python PDF | `python3` with `avalan` and `pypdf` |
| `shell.tesseract` | Recognize text in an image. | OCR | `tesseract-ocr` or `tesseract` |

Media tools (`shell.pdfinfo`, `shell.pdftotext`, `shell.pdftoppm`,
`shell.reportlab`, `shell.pdfplumber`, `shell.pypdf`, and `shell.tesseract`)
are disabled unless `allow_media_tools = true`. Optional binaries are resolved
at invocation time: if a configured command is not installed, the tool returns
a formatted `command_unavailable` result instead of failing agent loading. The
Python PDF tools resolve a trusted Python executable and also report
`command_unavailable` when the required package cannot be imported. In
container mode, the selected image must make both `avalan` and the target PDF
library importable to that Python interpreter.

The shell toolset does not provide generic shell execution. It never evaluates
model-supplied shell strings, never accepts arbitrary executable paths from
tool calls, and rejects write access. Paths are normalized under
`workspace_root`; hidden, sensitive, symlink, special-file, binary-content, and
size-limit checks are enforced before execution. These checks reduce race and
leak risk, but they cannot eliminate every time-of-check/time-of-use race on a
mutable local filesystem, so run shell-enabled agents only in workspaces whose
contents and binaries you trust.

# avalan

```
usage: avalan [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
              [--device DEVICE]
              [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
              [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE] [--theme {fancy,basic}]
              [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES]
              [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
              [--skip-hub-access-check] [--verbose] [--version]
              [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--help-full]
              {agent,cache,deploy,flow,memory,model,tokenizer,train} ...

Avalan CLI

positional arguments:
  {agent,cache,deploy,flow,memory,model,tokenizer,train}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --help-full           Show help for all commands and subcommands
```

## avalan agent

```
usage: avalan agent [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                    [--device DEVICE]
                    [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                    [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE] [--theme {fancy,basic}]
                    [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES]
                    [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
                    [--skip-hub-access-check] [--verbose] [--version]
                    [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                    {message,run,serve,proxy,init} ...

Manage AI agents

positional arguments:
  {message,run,serve,proxy,init}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

### avalan agent message

```
usage: avalan agent message [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                            [--device DEVICE]
                            [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                            [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                            [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                            [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                            [--verbose] [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                            {search} ...

Manage AI agent messages

positional arguments:
  {search}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

#### avalan agent message search

```
usage: avalan agent message search [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER]
                                   [--tokenizer-subfolder TOKENIZER_SUBFOLDER] [--device DEVICE]
                                   [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                                   [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                                   [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                                   [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                                   [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
                                   [--skip-hub-access-check] [--verbose] [--version]
                                   [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] --function
                                   {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms} --id ID
                                   [--limit LIMIT] --participant PARTICIPANT --session SESSION [--engine-uri ENGINE_URI]
                                   [--engine-base-url ENGINE_BASE_URL] [--name NAME] [--role ROLE] [--task TASK] [--instructions INSTRUCTIONS]
                                   [--goal-instructions GOAL_INSTRUCTIONS] [--system SYSTEM] [--developer DEVELOPER]
                                   [--user USER] [--user-template USER_TEMPLATE] [--memory-recent]
                                   [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                                   [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                                   [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS]
                                   [--memory-engine-overlap MEMORY_ENGINE_OVERLAP]
                                   [--memory-engine-window MEMORY_ENGINE_WINDOW] [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                                   [--maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES] [--run-skip-special-tokens]
                                   [--run-disable-cache]
                                   [--run-cache-strategy 
{dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}]
                                   [--run-temperature RUN_TEMPERATURE] [--run-top-k RUN_TOP_K] [--run-top-p RUN_TOP_P]
                                   [--tool TOOL] [--tools TOOLS] [--tool-browser-engine TOOL_BROWSER_ENGINE]
                                   [--tool-browser-search] [--tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT]
                                   [--tool-browser-search-k TOOL_BROWSER_SEARCH_K] [--tool-browser-debug]
                                   [--tool-browser-debug-url TOOL_BROWSER_DEBUG_URL]
                                   [--tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE]
                                   [--tool-browser-slowdown TOOL_BROWSER_SLOWDOWN] [--tool-browser-devtools]
                                   [--tool-browser-chromium-sandbox]
                                   [--tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH]
                                   [--tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT]
                                   [--tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR] [--tool-browser-is-mobile]
                                   [--tool-browser-has-touch] [--tool-browser-java-script-enabled]
                                   [--tool-database-dsn TOOL_DATABASE_DSN]
                                   [--tool-database-delay-secs TOOL_DATABASE_DELAY_SECS]
                                   [--tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE] [--tool-database-read-only]
                                   [--tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS]
                                   

Search within an agent's message memory

positional arguments:
  specifications_file   File that holds the agent specifications

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --function {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms}
                        Vector function to use for searching
  --id ID
  --limit LIMIT         If specified, load up to these many recent messages
  --participant PARTICIPANT
                        Search messages with given participant
  --session SESSION     Search within the given session

inline agent settings:
  --engine-uri ENGINE_URI
                        Agent engine URI
  --engine-base-url ENGINE_BASE_URL
                        Agent engine provider base URL
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Provider instructions
  --goal-instructions GOAL_INSTRUCTIONS
                        Agent goal instructions
  --system SYSTEM       System prompt
  --developer DEVELOPER
                        Developer prompt
  --user USER           User message template
  --user-template USER_TEMPLATE
                        User message template file
  --memory-recent
  --no-memory-recent
  --memory-permanent-message MEMORY_PERMANENT_MESSAGE
                        Permanent message memory DSN
  --memory-permanent MEMORY_PERMANENT
                        Permanent memory definition namespace@dsn
  --memory-engine-model-id MEMORY_ENGINE_MODEL_ID
                        Sentence transformer model for memory
  --memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS
                        Maximum tokens for memory sentence transformer
  --memory-engine-overlap MEMORY_ENGINE_OVERLAP
                        Overlap size for memory sentence transformer
  --memory-engine-window MEMORY_ENGINE_WINDOW
                        Window size for memory sentence transformer
  --run-max-new-tokens RUN_MAX_NEW_TOKENS
                        Maximum count of tokens on output
  --maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES, --run-maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES
                        Maximum model/tool result cycles for an agent run,
                        or 'unlimited'
  --run-skip-special-tokens
                        Skip special tokens on output
  --run-disable-cache   Disable generation cache
  --run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}
                        Cache implementation to use for generation
  --run-temperature RUN_TEMPERATURE
                        Temperature [0, 1]
  --run-top-k RUN_TOP_K
                        Number of highest probability vocabulary tokens to keep for top-k-filtering.
  --run-top-p RUN_TOP_P
                        If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p
                        or higher are kept for generation.
  --tool TOOL           Enable tool
  --tools TOOLS         Enable tools matching namespace

browser tool settings:
  --tool-browser-engine TOOL_BROWSER_ENGINE
  --tool-browser-search
  --tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT
  --tool-browser-search-k TOOL_BROWSER_SEARCH_K
  --tool-browser-debug
  --tool-browser-debug-url TOOL_BROWSER_DEBUG_URL
  --tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE
  --tool-browser-slowdown TOOL_BROWSER_SLOWDOWN
  --tool-browser-devtools
  --tool-browser-chromium-sandbox
  --tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH
  --tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT
  --tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR
  --tool-browser-is-mobile
  --tool-browser-has-touch
  --tool-browser-java-script-enabled

database tool settings:
  --tool-database-dsn TOOL_DATABASE_DSN
  --tool-database-delay-secs TOOL_DATABASE_DELAY_SECS
  --tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE
  --tool-database-read-only
  --tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS
```

### avalan agent run

```
usage: avalan agent run [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER]
                        [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                        [--device DEVICE]
                        [--parallel {auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,replicate}]
                        [--parallel-count PARALLEL_COUNT]
                        [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE] [--theme {fancy,basic}]
                        [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                        [--backend {transformers,mlx,vllm,ds4}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                        [--no-repl] [--quiet] [--tty TTY] [--record]
                        [--revision REVISION] [--skip-hub-access-check]
                        [--verbose] [--version]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                        [--display-events] [--stats]
                        [--display-pause [DISPLAY_PAUSE]]
                        [--display-probabilities]
                        [--display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM]
                        [--display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM]
                        [--display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]]
                        [--skip-display-reasoning-time] [--display-reasoning]
                        [--display-tokens [DISPLAY_TOKENS]] [--display-tools]
                        [--display-tools-events DISPLAY_TOOLS_EVENTS]
                        [--display-answer-height-expand | --display-answer-height DISPLAY_ANSWER_HEIGHT]
                        [--id ID] [--participant PARTICIPANT] [--conversation]
                        [--watch] [--no-session | --session SESSION]
                        [--skip-load-recent-messages]
                        [--load-recent-messages-limit LOAD_RECENT_MESSAGES_LIMIT]
                        [--sync] [--tools-confirm]
                        [--input-file INPUT_FILE]
                        [--tool-format {json,react,bracket,openai,harmony,dsml}]
                        [--tool-choice TOOL_CHOICE]
                        [--tool-recovery-format {tool_call_block,minimax_xml,tool_code,broad_xml,dsml_leakage,fenced}]
                        [--reasoning-tag {think,channel}] [--ds4-ctx DS4_CTX]
                        [--ds4-native-backend {auto,metal,cuda,cpu}]
                        [--ds4-mtp DS4_MTP] [--ds4-mtp-draft DS4_MTP_DRAFT]
                        [--ds4-mtp-margin DS4_MTP_MARGIN] [--ds4-warm-weights]
                        [--ds4-quality] [--with-ds4-native-log]
                        [--no-ds4-native-log] [--engine-uri ENGINE_URI]
                        [--engine-base-url ENGINE_BASE_URL] [--name NAME]
                        [--role ROLE] [--task TASK] [--instructions INSTRUCTIONS]
                        [--goal-instructions GOAL_INSTRUCTIONS]
                        [--system SYSTEM] [--developer DEVELOPER]
                        [--user USER] [--user-template USER_TEMPLATE]
                        [--memory-recent] [--no-memory-recent]
                        [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                        [--memory-permanent MEMORY_PERMANENT]
                        [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                        [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS]
                        [--memory-engine-overlap MEMORY_ENGINE_OVERLAP]
                        [--memory-engine-window MEMORY_ENGINE_WINDOW]
                        [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                        [--maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES]
                        [--run-skip-special-tokens] [--run-disable-cache]
                        [--run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}]
                        [--run-temperature RUN_TEMPERATURE]
                        [--run-top-k RUN_TOP_K] [--run-top-p RUN_TOP_P]
                        [--reasoning-effort {none,minimal,low,medium,high,xhigh,max}]
                        [--tool TOOL] [--tools TOOLS]
                        [--tool-browser-engine TOOL_BROWSER_ENGINE]
                        [--tool-browser-search]
                        [--tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT]
                        [--tool-browser-search-k TOOL_BROWSER_SEARCH_K]
                        [--tool-browser-debug]
                        [--tool-browser-debug-url TOOL_BROWSER_DEBUG_URL]
                        [--tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE]
                        [--tool-browser-slowdown TOOL_BROWSER_SLOWDOWN]
                        [--tool-browser-devtools]
                        [--tool-browser-chromium-sandbox]
                        [--tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH]
                        [--tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT]
                        [--tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR]
                        [--tool-browser-is-mobile] [--tool-browser-has-touch]
                        [--tool-browser-java-script-enabled]
                        [--tool-database-dsn TOOL_DATABASE_DSN]
                        [--tool-database-delay-secs TOOL_DATABASE_DELAY_SECS]
                        [--tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE]
                        [--tool-database-read-only]
                        [--tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS]
                        [--tool-graph-file TOOL_GRAPH_FILE]
                        [--tool-shell-backend {local,sandbox,container}]
                        [--tool-shell-workspace-root TOOL_SHELL_WORKSPACE_ROOT]
                        [--tool-shell-cwd TOOL_SHELL_CWD]
                        [--tool-shell-default-timeout-seconds TOOL_SHELL_DEFAULT_TIMEOUT_SECONDS]
                        [--tool-shell-max-timeout-seconds TOOL_SHELL_MAX_TIMEOUT_SECONDS]
                        [--tool-shell-max-stdout-bytes TOOL_SHELL_MAX_STDOUT_BYTES]
                        [--tool-shell-max-stderr-bytes TOOL_SHELL_MAX_STDERR_BYTES]
                        [--tool-shell-max-stdin-bytes TOOL_SHELL_MAX_STDIN_BYTES]
                        [--tool-shell-max-pipeline-stages TOOL_SHELL_MAX_PIPELINE_STAGES]
                        [--tool-shell-max-pipeline-bytes TOOL_SHELL_MAX_PIPELINE_BYTES]
                        [--tool-shell-max-intermediate-bytes TOOL_SHELL_MAX_INTERMEDIATE_BYTES]
                        [--tool-shell-pipeline-transport {buffered,native}]
                        [--tool-shell-max-arguments TOOL_SHELL_MAX_ARGUMENTS]
                        [--tool-shell-max-argument-bytes TOOL_SHELL_MAX_ARGUMENT_BYTES]
                        [--tool-shell-max-command-bytes TOOL_SHELL_MAX_COMMAND_BYTES]
                        [--tool-shell-max-path-count TOOL_SHELL_MAX_PATH_COUNT]
                        [--tool-shell-max-glob-count TOOL_SHELL_MAX_GLOB_COUNT]
                        [--tool-shell-max-glob-bytes-per-glob TOOL_SHELL_MAX_GLOB_BYTES_PER_GLOB]
                        [--tool-shell-max-total-glob-bytes TOOL_SHELL_MAX_TOTAL_GLOB_BYTES]
                        [--tool-shell-max-full-file-bytes TOOL_SHELL_MAX_FULL_FILE_BYTES]
                        [--tool-shell-max-rg-columns TOOL_SHELL_MAX_RG_COLUMNS]
                        [--tool-shell-max-rg-context-lines TOOL_SHELL_MAX_RG_CONTEXT_LINES]
                        [--tool-shell-max-rg-matches-per-file TOOL_SHELL_MAX_RG_MATCHES_PER_FILE]
                        [--tool-shell-max-head-lines TOOL_SHELL_MAX_HEAD_LINES]
                        [--tool-shell-max-tail-lines TOOL_SHELL_MAX_TAIL_LINES]
                        [--tool-shell-max-text-filter-input-bytes TOOL_SHELL_MAX_TEXT_FILTER_INPUT_BYTES]
                        [--tool-shell-max-filter-program-bytes TOOL_SHELL_MAX_FILTER_PROGRAM_BYTES]
                        [--tool-shell-max-filter-pattern-bytes TOOL_SHELL_MAX_FILTER_PATTERN_BYTES]
                        [--tool-shell-max-filter-selectors TOOL_SHELL_MAX_FILTER_SELECTORS]
                        [--tool-shell-max-awk-fields TOOL_SHELL_MAX_AWK_FIELDS]
                        [--tool-shell-max-awk-separator-bytes TOOL_SHELL_MAX_AWK_SEPARATOR_BYTES]
                        [--tool-shell-max-json-input-bytes TOOL_SHELL_MAX_JSON_INPUT_BYTES]
                        [--tool-shell-max-jq-filter-bytes TOOL_SHELL_MAX_JQ_FILTER_BYTES]
                        [--tool-shell-max-pdf-input-bytes TOOL_SHELL_MAX_PDF_INPUT_BYTES]
                        [--tool-shell-max-pdf-text-pages TOOL_SHELL_MAX_PDF_TEXT_PAGES]
                        [--tool-shell-max-pdf-raster-pages TOOL_SHELL_MAX_PDF_RASTER_PAGES]
                        [--tool-shell-max-pdf-raster-dpi TOOL_SHELL_MAX_PDF_RASTER_DPI]
                        [--tool-shell-max-raster-long-edge-pixels TOOL_SHELL_MAX_RASTER_LONG_EDGE_PIXELS]
                        [--tool-shell-max-raster-pixels TOOL_SHELL_MAX_RASTER_PIXELS]
                        [--tool-shell-max-output-files TOOL_SHELL_MAX_OUTPUT_FILES]
                        [--tool-shell-max-output-file-bytes TOOL_SHELL_MAX_OUTPUT_FILE_BYTES]
                        [--tool-shell-max-total-output-file-bytes TOOL_SHELL_MAX_TOTAL_OUTPUT_FILE_BYTES]
                        [--tool-shell-max-inline-output-file-bytes TOOL_SHELL_MAX_INLINE_OUTPUT_FILE_BYTES]
                        [--tool-shell-max-ocr-input-bytes TOOL_SHELL_MAX_OCR_INPUT_BYTES]
                        [--tool-shell-max-ocr-pixels TOOL_SHELL_MAX_OCR_PIXELS]
                        [--tool-shell-max-ocr-languages TOOL_SHELL_MAX_OCR_LANGUAGES]
                        [--tool-shell-max-tesseract-dpi TOOL_SHELL_MAX_TESSERACT_DPI]
                        [--tool-shell-stream-read-chunk-bytes TOOL_SHELL_STREAM_READ_CHUNK_BYTES]
                        [--tool-shell-max-concurrent-processes TOOL_SHELL_MAX_CONCURRENT_PROCESSES]
                        [--tool-shell-max-concurrent-heavy-processes TOOL_SHELL_MAX_CONCURRENT_HEAVY_PROCESSES]
                        [--tool-shell-default-pdf-timeout-seconds TOOL_SHELL_DEFAULT_PDF_TIMEOUT_SECONDS]
                        [--tool-shell-max-pdf-timeout-seconds TOOL_SHELL_MAX_PDF_TIMEOUT_SECONDS]
                        [--tool-shell-default-ocr-timeout-seconds TOOL_SHELL_DEFAULT_OCR_TIMEOUT_SECONDS]
                        [--tool-shell-max-ocr-timeout-seconds TOOL_SHELL_MAX_OCR_TIMEOUT_SECONDS]
                        [--tool-shell-tesseract-thread-limit TOOL_SHELL_TESSERACT_THREAD_LIMIT]
                        [--tool-shell-allow-pipelines]
                        [--tool-shell-allow-media-tools]
                        [--tool-shell-allow-absolute-paths]
                        [--tool-shell-allow-symlinks]
                        [--tool-shell-allow-hidden]
                        [--tool-shell-executable-search-path TOOL_SHELL_EXECUTABLE_SEARCH_PATHS]
                        [--tool-shell-executable-path COMMAND=PATH]
                        [specifications_file]

Run an AI agent

positional arguments:
  specifications_file   File that holds the agent specifications

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model
                        from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the
                        tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel {auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,replicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used
                        (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to
                        /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU
                        memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output
                        is displayed in model run (sets --disable-loading-
                        progress-bar, --skip-hub-access-check, --skip-special-
                        tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be
                        regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --display-events      Show non-tool stream events when an orchestrator or
                        agent is involved.
  --stats               Show token generation statistics for streaming output
  --display-pause [DISPLAY_PAUSE]
                        Pause (in ms.) when cycling through selected tokens as
                        defined by --display-probabilities
  --display-probabilities
                        If --display-tokens specified, show also the token
                        probability distribution
  --display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM
                        When --display-probabilities is used, select tokens
                        which logit probability is no higher than this value.
                        Defaults to 0.8
  --display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM
                        When --display-probabilities is used, select tokens
                        that have alternate tokens with a logit probability at
                        least or higher than this value. Defaults to 0.1
  --display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]
                        Display the time it takes to reach the given Nth token
                        (defaults to 256)
  --skip-display-reasoning-time
                        Don't display total reasoning time
  --display-reasoning   Display streamed reasoning text in the live response panel
  --display-tokens [DISPLAY_TOKENS]
                        How many tokens with full information to display at a
                        time
  --display-tools       Show tool lifecycle details for agent or orchestrator
                        runs.
  --display-tools-events DISPLAY_TOOLS_EVENTS
                        How many tool events to show on tool call panel.
                        Defaults to all retained tool events; use 0 to hide
                        completed tool history.
  --display-answer-height-expand
                        Expand answer section to full height
  --display-answer-height DISPLAY_ANSWER_HEIGHT
                        Height of the answer section (defaults to 12)
  --id ID               Use given ID as the agent ID
  --participant PARTICIPANT
                        If specified, this is the participant ID interacting
                        with the agent
  --conversation        Activate conversation mode with the agent
  --watch               Reload agent when the specification file changes (only
                        with --conversation)
  --no-session          If specified, don't use sessions in persistent message
                        memory
  --session SESSION     Continue the conversation on the given session
  --skip-load-recent-messages
                        If specified, skips loading recent messages
  --load-recent-messages-limit LOAD_RECENT_MESSAGES_LIMIT
                        If specified, load up to these many recent messages
  --sync                Don't use an async generator (streaming output)
  --tools-confirm       Confirm tool calls before execution
  --input-file INPUT_FILE
                        Attach a local file as native input for text
                        generation. May be specified multiple times.
  --tool-format {json,react,bracket,openai,harmony,dsml}
                        Tool format
  --tool-choice TOOL_CHOICE
                        Force a tool by canonical name when supported.
  --tool-recovery-format {tool_call_block,minimax_xml,tool_code,broad_xml,dsml_leakage,fenced}
                        Enable a tool-call recovery format
  --reasoning-tag {think,channel}
                        Reasoning tag style

DS4 backend options:
  --ds4-ctx DS4_CTX     DS4 context size
  --ds4-native-backend {auto,metal,cuda,cpu}
                        DS4 native backend
  --ds4-mtp DS4_MTP     DS4 MTP model path
  --ds4-mtp-draft DS4_MTP_DRAFT
                        DS4 MTP draft-token count
  --ds4-mtp-margin DS4_MTP_MARGIN
                        DS4 MTP acceptance margin
  --ds4-warm-weights    Warm DS4 model weights when opening the engine
  --ds4-quality         Enable DS4 quality mode
  --with-ds4-native-log, --ds4-native-log
                        Replay DS4 native stderr emitted while opening the
                        engine
  --no-ds4-native-log   Suppress DS4 native stderr emitted while opening the
                        engine

inline agent settings:
  --engine-uri ENGINE_URI
                        Agent engine URI
  --engine-base-url ENGINE_BASE_URL
                        Agent engine provider base URL
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Provider instructions
  --goal-instructions GOAL_INSTRUCTIONS
                        Agent goal instructions
  --system SYSTEM       System prompt
  --developer DEVELOPER
                        Developer prompt
  --user USER           User message template
  --user-template USER_TEMPLATE
                        User message template file
  --memory-recent
  --no-memory-recent
  --memory-permanent-message MEMORY_PERMANENT_MESSAGE
                        Permanent message memory DSN
  --memory-permanent MEMORY_PERMANENT
                        Permanent memory definition namespace@dsn
  --memory-engine-model-id MEMORY_ENGINE_MODEL_ID
                        Sentence transformer model for memory
  --memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS
                        Maximum tokens for memory sentence transformer
  --memory-engine-overlap MEMORY_ENGINE_OVERLAP
                        Overlap size for memory sentence transformer
  --memory-engine-window MEMORY_ENGINE_WINDOW
                        Window size for memory sentence transformer
  --run-max-new-tokens RUN_MAX_NEW_TOKENS
                        Maximum count of tokens on output
  --maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES, --run-maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES
                        Maximum model/tool result cycles for an agent run,
                        or 'unlimited'
  --run-skip-special-tokens
                        Skip special tokens on output
  --run-disable-cache   Disable generation cache
  --run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}
                        Cache implementation to use for generation
  --run-temperature RUN_TEMPERATURE
                        Temperature [0, 1]
  --run-top-k RUN_TOP_K
                        Number of highest probability vocabulary tokens to
                        keep for top-k-filtering.
  --run-top-p RUN_TOP_P
                        If set to < 1, only the smallest set of most probable
                        tokens with probabilities that add up to top_p or
                        higher are kept for generation.
  --reasoning-effort {none,minimal,low,medium,high,xhigh,max}, --run-reasoning-effort {none,minimal,low,medium,high,xhigh,max}
                        Reasoning effort level
  --tool TOOL           Enable tool
  --tools TOOLS         Enable tools matching namespace

browser tool settings:
  --tool-browser-engine TOOL_BROWSER_ENGINE
  --tool-browser-search
  --tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT
  --tool-browser-search-k TOOL_BROWSER_SEARCH_K
  --tool-browser-debug
  --tool-browser-debug-url TOOL_BROWSER_DEBUG_URL
  --tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE
  --tool-browser-slowdown TOOL_BROWSER_SLOWDOWN
  --tool-browser-devtools
  --tool-browser-chromium-sandbox
  --tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH
  --tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT
  --tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR
  --tool-browser-is-mobile
  --tool-browser-has-touch
  --tool-browser-java-script-enabled

database tool settings:
  --tool-database-dsn TOOL_DATABASE_DSN
  --tool-database-delay-secs TOOL_DATABASE_DELAY_SECS
  --tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE
  --tool-database-read-only
  --tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS

graph tool settings:
  --tool-graph-file TOOL_GRAPH_FILE

shell tool settings:
  --tool-shell-backend {local,sandbox,container}
  --tool-shell-workspace-root TOOL_SHELL_WORKSPACE_ROOT
  --tool-shell-cwd TOOL_SHELL_CWD
  --tool-shell-default-timeout-seconds TOOL_SHELL_DEFAULT_TIMEOUT_SECONDS
  --tool-shell-max-timeout-seconds TOOL_SHELL_MAX_TIMEOUT_SECONDS
  --tool-shell-max-stdout-bytes TOOL_SHELL_MAX_STDOUT_BYTES
  --tool-shell-max-stderr-bytes TOOL_SHELL_MAX_STDERR_BYTES
  --tool-shell-max-stdin-bytes TOOL_SHELL_MAX_STDIN_BYTES
  --tool-shell-max-pipeline-stages TOOL_SHELL_MAX_PIPELINE_STAGES
  --tool-shell-max-pipeline-bytes TOOL_SHELL_MAX_PIPELINE_BYTES
  --tool-shell-max-intermediate-bytes TOOL_SHELL_MAX_INTERMEDIATE_BYTES
  --tool-shell-pipeline-transport {buffered,native}
                        Trusted shell pipeline byte transport.
  --tool-shell-max-arguments TOOL_SHELL_MAX_ARGUMENTS
  --tool-shell-max-argument-bytes TOOL_SHELL_MAX_ARGUMENT_BYTES
  --tool-shell-max-command-bytes TOOL_SHELL_MAX_COMMAND_BYTES
  --tool-shell-max-path-count TOOL_SHELL_MAX_PATH_COUNT
  --tool-shell-max-glob-count TOOL_SHELL_MAX_GLOB_COUNT
  --tool-shell-max-glob-bytes-per-glob TOOL_SHELL_MAX_GLOB_BYTES_PER_GLOB
  --tool-shell-max-total-glob-bytes TOOL_SHELL_MAX_TOTAL_GLOB_BYTES
  --tool-shell-max-full-file-bytes TOOL_SHELL_MAX_FULL_FILE_BYTES
  --tool-shell-max-rg-columns TOOL_SHELL_MAX_RG_COLUMNS
  --tool-shell-max-rg-context-lines TOOL_SHELL_MAX_RG_CONTEXT_LINES
  --tool-shell-max-rg-matches-per-file TOOL_SHELL_MAX_RG_MATCHES_PER_FILE
  --tool-shell-max-head-lines TOOL_SHELL_MAX_HEAD_LINES
  --tool-shell-max-tail-lines TOOL_SHELL_MAX_TAIL_LINES
  --tool-shell-max-text-filter-input-bytes TOOL_SHELL_MAX_TEXT_FILTER_INPUT_BYTES
  --tool-shell-max-filter-program-bytes TOOL_SHELL_MAX_FILTER_PROGRAM_BYTES
  --tool-shell-max-filter-pattern-bytes TOOL_SHELL_MAX_FILTER_PATTERN_BYTES
  --tool-shell-max-filter-selectors TOOL_SHELL_MAX_FILTER_SELECTORS
  --tool-shell-max-awk-fields TOOL_SHELL_MAX_AWK_FIELDS
  --tool-shell-max-awk-separator-bytes TOOL_SHELL_MAX_AWK_SEPARATOR_BYTES
  --tool-shell-max-json-input-bytes TOOL_SHELL_MAX_JSON_INPUT_BYTES
  --tool-shell-max-jq-filter-bytes TOOL_SHELL_MAX_JQ_FILTER_BYTES
  --tool-shell-max-pdf-input-bytes TOOL_SHELL_MAX_PDF_INPUT_BYTES
  --tool-shell-max-pdf-text-pages TOOL_SHELL_MAX_PDF_TEXT_PAGES
  --tool-shell-max-pdf-raster-pages TOOL_SHELL_MAX_PDF_RASTER_PAGES
  --tool-shell-max-pdf-raster-dpi TOOL_SHELL_MAX_PDF_RASTER_DPI
  --tool-shell-max-raster-long-edge-pixels TOOL_SHELL_MAX_RASTER_LONG_EDGE_PIXELS
  --tool-shell-max-raster-pixels TOOL_SHELL_MAX_RASTER_PIXELS
  --tool-shell-max-output-files TOOL_SHELL_MAX_OUTPUT_FILES
  --tool-shell-max-output-file-bytes TOOL_SHELL_MAX_OUTPUT_FILE_BYTES
  --tool-shell-max-total-output-file-bytes TOOL_SHELL_MAX_TOTAL_OUTPUT_FILE_BYTES
  --tool-shell-max-inline-output-file-bytes TOOL_SHELL_MAX_INLINE_OUTPUT_FILE_BYTES
  --tool-shell-max-ocr-input-bytes TOOL_SHELL_MAX_OCR_INPUT_BYTES
  --tool-shell-max-ocr-pixels TOOL_SHELL_MAX_OCR_PIXELS
  --tool-shell-max-ocr-languages TOOL_SHELL_MAX_OCR_LANGUAGES
  --tool-shell-max-tesseract-dpi TOOL_SHELL_MAX_TESSERACT_DPI
  --tool-shell-stream-read-chunk-bytes TOOL_SHELL_STREAM_READ_CHUNK_BYTES
  --tool-shell-max-concurrent-processes TOOL_SHELL_MAX_CONCURRENT_PROCESSES
  --tool-shell-max-concurrent-heavy-processes TOOL_SHELL_MAX_CONCURRENT_HEAVY_PROCESSES
  --tool-shell-default-pdf-timeout-seconds TOOL_SHELL_DEFAULT_PDF_TIMEOUT_SECONDS
  --tool-shell-max-pdf-timeout-seconds TOOL_SHELL_MAX_PDF_TIMEOUT_SECONDS
  --tool-shell-default-ocr-timeout-seconds TOOL_SHELL_DEFAULT_OCR_TIMEOUT_SECONDS
  --tool-shell-max-ocr-timeout-seconds TOOL_SHELL_MAX_OCR_TIMEOUT_SECONDS
  --tool-shell-tesseract-thread-limit TOOL_SHELL_TESSERACT_THREAD_LIMIT
  --tool-shell-allow-pipelines
  --tool-shell-allow-media-tools
  --tool-shell-allow-absolute-paths
  --tool-shell-allow-symlinks
  --tool-shell-allow-hidden
  --tool-shell-executable-search-path TOOL_SHELL_EXECUTABLE_SEARCH_PATHS
                        Add a trusted directory used to resolve shell tools.
  --tool-shell-executable-path COMMAND=PATH
                        Map a shell command to a trusted absolute executable.
  --tool-container-backend {docker,apple-container}
  --tool-container-profile TOOL_CONTAINER_PROFILE
  --tool-container-image TOOL_CONTAINER_IMAGE
  --tool-container-workspace-root TOOL_CONTAINER_WORKSPACE_ROOT
  --tool-container-pull-policy {never,if_missing,always}
  --tool-container-platform TOOL_CONTAINER_PLATFORM
  --tool-container-cpu-count TOOL_CONTAINER_CPU_COUNT
  --tool-container-memory-bytes TOOL_CONTAINER_MEMORY_BYTES
  --tool-container-pids TOOL_CONTAINER_PIDS
  --tool-container-timeout-seconds TOOL_CONTAINER_TIMEOUT_SECONDS
  --tool-container-network-mode {none,loopback,allowlist,full}
  --tool-container-review-mode {deny,require_review,preauthorized}
  --tool-sandbox-backend {seatbelt,bubblewrap}
  --tool-sandbox-profile TOOL_SANDBOX_PROFILE
  --tool-sandbox-trusted-executable TOOL_SANDBOX_TRUSTED_EXECUTABLES
  --tool-sandbox-executable-search-root TOOL_SANDBOX_EXECUTABLE_SEARCH_ROOTS
  --tool-sandbox-read-root TOOL_SANDBOX_READ_ROOTS
  --tool-sandbox-write-root TOOL_SANDBOX_WRITE_ROOTS
  --tool-sandbox-deny-root TOOL_SANDBOX_DENY_ROOTS
  --tool-sandbox-scratch-root TOOL_SANDBOX_SCRATCH_ROOTS
  --tool-sandbox-output-root TOOL_SANDBOX_OUTPUT_ROOTS
  --tool-sandbox-network-mode {none,loopback,allowlist,full}
  --tool-sandbox-network-egress TOOL_SANDBOX_NETWORK_EGRESS
  --tool-sandbox-timeout-seconds TOOL_SANDBOX_TIMEOUT_SECONDS
  --tool-sandbox-pids TOOL_SANDBOX_PIDS
  --tool-sandbox-max-stdout-bytes TOOL_SANDBOX_MAX_STDOUT_BYTES
  --tool-sandbox-max-stderr-bytes TOOL_SANDBOX_MAX_STDERR_BYTES
  --tool-sandbox-max-artifact-bytes TOOL_SANDBOX_MAX_ARTIFACT_BYTES
  --tool-sandbox-allow-artifacts
  --tool-sandbox-child-processes {deny,allow}
  --tool-sandbox-inherited-fds {deny,stdio,explicit}
  --tool-shell-container-profile TOOL_SHELL_CONTAINER_PROFILE
  --tool-shell-container-required
  --tool-shell-sandbox-profile TOOL_SHELL_SANDBOX_PROFILE
  --tool-shell-sandbox-required
```

### avalan agent serve

```
usage: avalan agent serve [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                          [--device DEVICE]
                          [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                          [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                          [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                          [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
                          [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check] [--verbose]
                          [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--id ID]
                          [--participant PARTICIPANT] [--host HOST] [--port PORT] [--mcp-prefix MCP_PREFIX]
                          [--mcp-name MCP_NAME] [--mcp-description MCP_DESCRIPTION] [--openai-prefix OPENAI_PREFIX]
                          [--a2a-prefix A2A_PREFIX] [--a2a-name A2A_NAME] [--a2a-description A2A_DESCRIPTION]
                          [--protocol PROTOCOL] [--reload] [--cors-origin CORS_ORIGIN] [--cors-origin-regex CORS_ORIGIN_REGEX]
                          [--cors-method CORS_METHOD] [--cors-header CORS_HEADER] [--cors-credentials]
                          [--engine-uri ENGINE_URI] [--engine-base-url ENGINE_BASE_URL] [--name NAME] [--role ROLE]
                          [--task TASK] [--instructions INSTRUCTIONS] [--goal-instructions GOAL_INSTRUCTIONS]
                          [--system SYSTEM] [--developer DEVELOPER] [--user USER] [--user-template USER_TEMPLATE]
                          [--memory-recent] [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                          [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                          [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS]
                          [--memory-engine-overlap MEMORY_ENGINE_OVERLAP] [--memory-engine-window MEMORY_ENGINE_WINDOW]
                          [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                          [--maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES] [--run-skip-special-tokens]
                          [--run-disable-cache]
                          [--run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}]
                          [--run-temperature RUN_TEMPERATURE] [--run-top-k RUN_TOP_K] [--run-top-p RUN_TOP_P] [--tool TOOL]
                          [--tools TOOLS] [--tool-browser-engine TOOL_BROWSER_ENGINE] [--tool-browser-search]
                          [--tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT]
                          [--tool-browser-search-k TOOL_BROWSER_SEARCH_K] [--tool-browser-debug]
                          [--tool-browser-debug-url TOOL_BROWSER_DEBUG_URL]
                          [--tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE]
                          [--tool-browser-slowdown TOOL_BROWSER_SLOWDOWN] [--tool-browser-devtools]
                          [--tool-browser-chromium-sandbox] [--tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH]
                          [--tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT]
                          [--tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR] [--tool-browser-is-mobile]
                          [--tool-browser-has-touch] [--tool-browser-java-script-enabled]
                          [--tool-database-dsn TOOL_DATABASE_DSN] [--tool-database-delay-secs TOOL_DATABASE_DELAY_SECS]
                          [--tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE] [--tool-database-read-only]
                          [--tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS]
                          

Serve an AI agent as an API endpoint

positional arguments:
  specifications_file   File that holds the agent specifications

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --id ID               Use given ID as the agent ID
  --participant PARTICIPANT
                        If specified, this is the participant ID interacting with the agent
  --host HOST           Host (defaults to 127.0.0.1)
  --port PORT           Port (defaults to 9001, HAL 9000+1)
  --mcp-prefix MCP_PREFIX
                        URL prefix for MCP endpoints (defaults to /mcp)
  --mcp-name MCP_NAME   MCP tool name for tools/call (defaults to run)
  --mcp-description MCP_DESCRIPTION
                        MCP tool description for tools/list
  --openai-prefix OPENAI_PREFIX
                        URL prefix for OpenAI endpoints (defaults to /v1)
  --a2a-prefix A2A_PREFIX
                        URL prefix for A2A endpoints (defaults to /a2a)
  --a2a-name A2A_NAME   A2A tool name for task execution (defaults to run)
  --a2a-description A2A_DESCRIPTION
                        A2A tool description for the agent card
  --protocol PROTOCOL   Protocol to expose (e.g. openai, openai:responses,completion). May be specified multiple times
  --reload              Hot reload on code changes
  --cors-origin CORS_ORIGIN
                        Allowed CORS origin; may be specified multiple times
  --cors-origin-regex CORS_ORIGIN_REGEX
                        Allowed CORS origin regex
  --cors-method CORS_METHOD
                        Allowed CORS method; may be specified multiple times
  --cors-header CORS_HEADER
                        Allowed CORS header; may be specified multiple times
  --cors-credentials    Allow CORS credentials

inline agent settings:
  --engine-uri ENGINE_URI
                        Agent engine URI
  --engine-base-url ENGINE_BASE_URL
                        Agent engine provider base URL
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Provider instructions
  --goal-instructions GOAL_INSTRUCTIONS
                        Agent goal instructions
  --system SYSTEM       System prompt
  --developer DEVELOPER
                        Developer prompt
  --user USER           User message template
  --user-template USER_TEMPLATE
                        User message template file
  --memory-recent
  --no-memory-recent
  --memory-permanent-message MEMORY_PERMANENT_MESSAGE
                        Permanent message memory DSN
  --memory-permanent MEMORY_PERMANENT
                        Permanent memory definition namespace@dsn
  --memory-engine-model-id MEMORY_ENGINE_MODEL_ID
                        Sentence transformer model for memory
  --memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS
                        Maximum tokens for memory sentence transformer
  --memory-engine-overlap MEMORY_ENGINE_OVERLAP
                        Overlap size for memory sentence transformer
  --memory-engine-window MEMORY_ENGINE_WINDOW
                        Window size for memory sentence transformer
  --run-max-new-tokens RUN_MAX_NEW_TOKENS
                        Maximum count of tokens on output
  --maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES, --run-maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES
                        Maximum model/tool result cycles for an agent run,
                        or 'unlimited'
  --run-skip-special-tokens
                        Skip special tokens on output
  --run-disable-cache   Disable generation cache
  --run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}
                        Cache implementation to use for generation
  --run-temperature RUN_TEMPERATURE
                        Temperature [0, 1]
  --run-top-k RUN_TOP_K
                        Number of highest probability vocabulary tokens to keep for top-k-filtering.
  --run-top-p RUN_TOP_P
                        If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p
                        or higher are kept for generation.
  --tool TOOL           Enable tool
  --tools TOOLS         Enable tools matching namespace

browser tool settings:
  --tool-browser-engine TOOL_BROWSER_ENGINE
  --tool-browser-search
  --tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT
  --tool-browser-search-k TOOL_BROWSER_SEARCH_K
  --tool-browser-debug
  --tool-browser-debug-url TOOL_BROWSER_DEBUG_URL
  --tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE
  --tool-browser-slowdown TOOL_BROWSER_SLOWDOWN
  --tool-browser-devtools
  --tool-browser-chromium-sandbox
  --tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH
  --tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT
  --tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR
  --tool-browser-is-mobile
  --tool-browser-has-touch
  --tool-browser-java-script-enabled

database tool settings:
  --tool-database-dsn TOOL_DATABASE_DSN
  --tool-database-delay-secs TOOL_DATABASE_DELAY_SECS
  --tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE
  --tool-database-read-only
  --tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS
```

### avalan agent proxy

```
usage: avalan agent proxy [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                          [--device DEVICE]
                          [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                          [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                          [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                          [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
                          [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check] [--verbose]
                          [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--id ID]
                          [--participant PARTICIPANT] [--host HOST] [--port PORT] [--mcp-prefix MCP_PREFIX]
                          [--mcp-name MCP_NAME] [--mcp-description MCP_DESCRIPTION] [--openai-prefix OPENAI_PREFIX]
                          [--a2a-prefix A2A_PREFIX] [--a2a-name A2A_NAME] [--a2a-description A2A_DESCRIPTION]
                          [--protocol PROTOCOL] [--reload] [--cors-origin CORS_ORIGIN] [--cors-origin-regex CORS_ORIGIN_REGEX]
                          [--cors-method CORS_METHOD] [--cors-header CORS_HEADER] [--cors-credentials]
                          [--engine-uri ENGINE_URI] [--engine-base-url ENGINE_BASE_URL] [--name NAME] [--role ROLE]
                          [--task TASK] [--instructions INSTRUCTIONS] [--goal-instructions GOAL_INSTRUCTIONS]
                          [--system SYSTEM] [--developer DEVELOPER] [--user USER] [--user-template USER_TEMPLATE]
                          [--memory-recent] [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                          [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                          [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS]
                          [--memory-engine-overlap MEMORY_ENGINE_OVERLAP] [--memory-engine-window MEMORY_ENGINE_WINDOW]
                          [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                          [--maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES] [--run-skip-special-tokens]
                          [--run-disable-cache]
                          [--run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}]
                          [--run-temperature RUN_TEMPERATURE] [--run-top-k RUN_TOP_K] [--run-top-p RUN_TOP_P] [--tool TOOL]
                          [--tools TOOLS] [--tool-browser-engine TOOL_BROWSER_ENGINE] [--tool-browser-search]
                          [--tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT]
                          [--tool-browser-search-k TOOL_BROWSER_SEARCH_K] [--tool-browser-debug]
                          [--tool-browser-debug-url TOOL_BROWSER_DEBUG_URL]
                          [--tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE]
                          [--tool-browser-slowdown TOOL_BROWSER_SLOWDOWN] [--tool-browser-devtools]
                          [--tool-browser-chromium-sandbox] [--tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH]
                          [--tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT]
                          [--tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR] [--tool-browser-is-mobile]
                          [--tool-browser-has-touch] [--tool-browser-java-script-enabled]
                          [--tool-database-dsn TOOL_DATABASE_DSN] [--tool-database-delay-secs TOOL_DATABASE_DELAY_SECS]
                          [--tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE] [--tool-database-read-only]
                          [--tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS]
                          

Serve a proxy agent as an API endpoint

positional arguments:
  specifications_file   File that holds the agent specifications

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --id ID               Use given ID as the agent ID
  --participant PARTICIPANT
                        If specified, this is the participant ID interacting with the agent
  --host HOST           Host (defaults to 127.0.0.1)
  --port PORT           Port (defaults to 9001, HAL 9000+1)
  --mcp-prefix MCP_PREFIX
                        URL prefix for MCP endpoints (defaults to /mcp)
  --mcp-name MCP_NAME   MCP tool name for tools/call (defaults to run)
  --mcp-description MCP_DESCRIPTION
                        MCP tool description for tools/list
  --openai-prefix OPENAI_PREFIX
                        URL prefix for OpenAI endpoints (defaults to /v1)
  --a2a-prefix A2A_PREFIX
                        URL prefix for A2A endpoints (defaults to /a2a)
  --a2a-name A2A_NAME   A2A tool name for task execution (defaults to run)
  --a2a-description A2A_DESCRIPTION
                        A2A tool description for the agent card
  --protocol PROTOCOL   Protocol to expose (e.g. openai, openai:responses,completion). May be specified multiple times
  --reload              Hot reload on code changes
  --cors-origin CORS_ORIGIN
                        Allowed CORS origin; may be specified multiple times
  --cors-origin-regex CORS_ORIGIN_REGEX
                        Allowed CORS origin regex
  --cors-method CORS_METHOD
                        Allowed CORS method; may be specified multiple times
  --cors-header CORS_HEADER
                        Allowed CORS header; may be specified multiple times
  --cors-credentials    Allow CORS credentials

inline agent settings:
  --engine-uri ENGINE_URI
                        Agent engine URI
  --engine-base-url ENGINE_BASE_URL
                        Agent engine provider base URL
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Provider instructions
  --goal-instructions GOAL_INSTRUCTIONS
                        Agent goal instructions
  --system SYSTEM       System prompt
  --developer DEVELOPER
                        Developer prompt
  --user USER           User message template
  --user-template USER_TEMPLATE
                        User message template file
  --memory-recent
  --no-memory-recent
  --memory-permanent-message MEMORY_PERMANENT_MESSAGE
                        Permanent message memory DSN
  --memory-permanent MEMORY_PERMANENT
                        Permanent memory definition namespace@dsn
  --memory-engine-model-id MEMORY_ENGINE_MODEL_ID
                        Sentence transformer model for memory
  --memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS
                        Maximum tokens for memory sentence transformer
  --memory-engine-overlap MEMORY_ENGINE_OVERLAP
                        Overlap size for memory sentence transformer
  --memory-engine-window MEMORY_ENGINE_WINDOW
                        Window size for memory sentence transformer
  --run-max-new-tokens RUN_MAX_NEW_TOKENS
                        Maximum count of tokens on output
  --maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES, --run-maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES
                        Maximum model/tool result cycles for an agent run,
                        or 'unlimited'
  --run-skip-special-tokens
                        Skip special tokens on output
  --run-disable-cache   Disable generation cache
  --run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}
                        Cache implementation to use for generation
  --run-temperature RUN_TEMPERATURE
                        Temperature [0, 1]
  --run-top-k RUN_TOP_K
                        Number of highest probability vocabulary tokens to keep for top-k-filtering.
  --run-top-p RUN_TOP_P
                        If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p
                        or higher are kept for generation.
  --tool TOOL           Enable tool
  --tools TOOLS         Enable tools matching namespace

browser tool settings:
  --tool-browser-engine TOOL_BROWSER_ENGINE
  --tool-browser-search
  --tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT
  --tool-browser-search-k TOOL_BROWSER_SEARCH_K
  --tool-browser-debug
  --tool-browser-debug-url TOOL_BROWSER_DEBUG_URL
  --tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE
  --tool-browser-slowdown TOOL_BROWSER_SLOWDOWN
  --tool-browser-devtools
  --tool-browser-chromium-sandbox
  --tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH
  --tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT
  --tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR
  --tool-browser-is-mobile
  --tool-browser-has-touch
  --tool-browser-java-script-enabled

database tool settings:
  --tool-database-dsn TOOL_DATABASE_DSN
  --tool-database-delay-secs TOOL_DATABASE_DELAY_SECS
  --tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE
  --tool-database-read-only
  --tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS
```

### avalan agent init

```
usage: avalan agent init [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                         [--device DEVICE]
                         [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                         [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                         [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                         [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                         [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                         [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--engine-uri ENGINE_URI]
                         [--engine-base-url ENGINE_BASE_URL] [--name NAME] [--role ROLE] [--task TASK]
                         [--instructions INSTRUCTIONS] [--goal-instructions GOAL_INSTRUCTIONS] [--system SYSTEM]
                         [--developer DEVELOPER] [--user USER] [--user-template USER_TEMPLATE] [--memory-recent]
                         [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                         [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                         [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS] [--memory-engine-overlap MEMORY_ENGINE_OVERLAP]
                         [--memory-engine-window MEMORY_ENGINE_WINDOW] [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                         [--maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES] [--run-skip-special-tokens]
                         [--run-disable-cache]
                         [--run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}]
                         [--run-temperature RUN_TEMPERATURE] [--run-top-k RUN_TOP_K] [--run-top-p RUN_TOP_P] [--tool TOOL]
                         [--tools TOOLS]

Create an agent definition

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)

inline agent settings:
  --engine-uri ENGINE_URI
                        Agent engine URI
  --engine-base-url ENGINE_BASE_URL
                        Agent engine provider base URL
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Provider instructions
  --goal-instructions GOAL_INSTRUCTIONS
                        Agent goal instructions
  --system SYSTEM       System prompt
  --developer DEVELOPER
                        Developer prompt
  --user USER           User message template
  --user-template USER_TEMPLATE
                        User message template file
  --memory-recent
  --no-memory-recent
  --memory-permanent-message MEMORY_PERMANENT_MESSAGE
                        Permanent message memory DSN
  --memory-permanent MEMORY_PERMANENT
                        Permanent memory definition namespace@dsn
  --memory-engine-model-id MEMORY_ENGINE_MODEL_ID
                        Sentence transformer model for memory
  --memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS
                        Maximum tokens for memory sentence transformer
  --memory-engine-overlap MEMORY_ENGINE_OVERLAP
                        Overlap size for memory sentence transformer
  --memory-engine-window MEMORY_ENGINE_WINDOW
                        Window size for memory sentence transformer
  --run-max-new-tokens RUN_MAX_NEW_TOKENS
                        Maximum count of tokens on output
  --maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES, --run-maximum-tool-cycles RUN_MAXIMUM_TOOL_CYCLES
                        Maximum model/tool result cycles for an agent run,
                        or 'unlimited'
  --run-skip-special-tokens
                        Skip special tokens on output
  --run-disable-cache   Disable generation cache
  --run-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}
                        Cache implementation to use for generation
  --run-temperature RUN_TEMPERATURE
                        Temperature [0, 1]
  --run-top-k RUN_TOP_K
                        Number of highest probability vocabulary tokens to keep for top-k-filtering.
  --run-top-p RUN_TOP_P
                        If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p
                        or higher are kept for generation.
  --tool TOOL           Enable tool
  --tools TOOLS         Enable tools matching namespace
```

## avalan cache

```
usage: avalan cache [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                    [--device DEVICE]
                    [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                    [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE] [--theme {fancy,basic}]
                    [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES]
                    [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
                    [--skip-hub-access-check] [--verbose] [--version]
                    [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                    {delete,download,list} ...

Manage models cache

positional arguments:
  {delete,download,list}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

### avalan cache delete

```
usage: avalan cache delete [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                           [--device DEVICE]
                           [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                           [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                           [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                           [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
                           [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check] [--verbose]
                           [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--delete]
                           --model MODEL [--delete-revision DELETE_REVISION]

Delete cached model data

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --delete              Actually delete. If not provided, a dry run is performed and data that would be deleted is shown, yet
                        not deleted
  --model MODEL, -m MODEL
                        Model to delete
  --delete-revision DELETE_REVISION
                        Revision to delete
```

### avalan cache download

```
usage: avalan cache download [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                             [--device DEVICE]
                             [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                             [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                             [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                             [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                             [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                             [--verbose] [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                             [--workers WORKERS] [--local-dir LOCAL_DIR] [--local-dir-symlinks]
                             model

Download model data to cache

positional arguments:
  model                 Model to download

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --workers WORKERS     How many download workers to use
  --local-dir LOCAL_DIR
                        Local directory to download the model to
  --local-dir-symlinks  Use symlinks when downloading to local dir
```

### avalan cache list

```
usage: avalan cache list [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                         [--device DEVICE]
                         [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                         [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                         [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                         [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                         [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                         [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--model MODEL] [--summary]

List cache contents

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --model MODEL         Models to show content details on
  --summary             If specified, when showing one or more models show only summary
```

## avalan deploy

```
usage: avalan deploy [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                     [--device DEVICE]
                     [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                     [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                     [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                     [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                     [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                     [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                     {run} ...

Manage AI deployments

positional arguments:
  {run}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

### avalan deploy run

```
usage: avalan deploy run [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                         [--device DEVICE]
                         [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                         [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                         [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                         [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                         [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                         [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                         deployment

Perform a deployment

positional arguments:
  deployment            Deployment to run

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

## avalan flow

`avalan flow` validates, compiles, renders, compares, runs, inspects, resumes,
cancels, and exports sanitized traces for flow definitions. Strict flow
definitions use declared inputs, declared outputs, explicit entry behavior, and
explicit output selection. Mermaid commands operate on inert Flow Views and do
not execute nodes.

```bash
avalan flow validate flow.toml
avalan flow compile flow.toml --output strict.flow.toml
avalan flow graph inspect flow.toml --json
avalan flow mermaid parse topology.mmd --mode presentation --json
avalan flow run flow.toml --input-json '{"name":"Ada"}' --json
```

Flow subcommands:

| Command | Purpose |
| --- | --- |
| `run` | Execute a local strict flow or a compatible native flow. |
| `validate` | Validate a flow definition without node execution. |
| `compile` | Compile a flow definition to canonical strict TOML. |
| `graph inspect` | Inspect static graph authoring classifications and bindings. |
| `mermaid parse` | Parse Mermaid into an inert Flow View. |
| `mermaid render` | Render a safe Mermaid view. |
| `mermaid compare` | Compare Mermaid topology with a flow definition. |
| `mermaid skeleton` | Create a non-executable flow skeleton from Mermaid topology. |
| `inspect` | Inspect a durable task-backed flow run. |
| `trace` | Export a sanitized durable flow trace. |
| `cancel` | Request cancellation for a durable flow run. |
| `resume` | Resume a paused human-review flow from durable state. |

See [Flow authoring](FLOW_AUTHORING.md) for definition shape, Mermaid modes,
task-backed execution, and human review details.

### avalan flow run

```
usage: avalan flow run [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER]
                       [--tokenizer-subfolder TOKENIZER_SUBFOLDER] [--device DEVICE]
                       [--parallel {auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,replicate}]
                       [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                       [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                       [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                       [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                       [--verbose] [--version]
                       [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--input TASK_INPUT]
                       [--input-json TASK_INPUT_JSON] [--file TASK_FILES] [--file-mime TASK_FILE_MIME_TYPES]
                       [--pdf TASK_PDF] [--json] [--output TASK_OUTPUT_PATH] [--flow-parallel N] [--tool TOOL]
                       [--tools TOOLS] [--tool-browser-engine TOOL_BROWSER_ENGINE] [--tool-browser-search]
                       [--tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT]
                       [--tool-browser-search-k TOOL_BROWSER_SEARCH_K] [--tool-browser-debug]
                       [--tool-browser-debug-url TOOL_BROWSER_DEBUG_URL]
                       [--tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE]
                       [--tool-browser-slowdown TOOL_BROWSER_SLOWDOWN] [--tool-browser-devtools]
                       [--tool-browser-chromium-sandbox] [--tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH]
                       [--tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT]
                       [--tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR] [--tool-browser-is-mobile]
                       [--tool-browser-has-touch] [--tool-browser-java-script-enabled]
                       [--tool-database-dsn TOOL_DATABASE_DSN] [--tool-database-delay-secs TOOL_DATABASE_DELAY_SECS]
                       [--tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE] [--tool-database-read-only]
                       [--tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS]
                       [--tool-graph-file TOOL_GRAPH_FILE] [--tool-shell-backend {local,sandbox,container}]
                       [--tool-shell-workspace-root TOOL_SHELL_WORKSPACE_ROOT] [--tool-shell-cwd TOOL_SHELL_CWD]
                       [--tool-shell-default-timeout-seconds TOOL_SHELL_DEFAULT_TIMEOUT_SECONDS]
                       [--tool-shell-max-timeout-seconds TOOL_SHELL_MAX_TIMEOUT_SECONDS]
                       [--tool-shell-max-stdout-bytes TOOL_SHELL_MAX_STDOUT_BYTES]
                       [--tool-shell-max-stderr-bytes TOOL_SHELL_MAX_STDERR_BYTES]
                       [--tool-shell-max-stdin-bytes TOOL_SHELL_MAX_STDIN_BYTES]
                       [--tool-shell-max-pipeline-stages TOOL_SHELL_MAX_PIPELINE_STAGES]
                       [--tool-shell-max-pipeline-bytes TOOL_SHELL_MAX_PIPELINE_BYTES]
                       [--tool-shell-max-intermediate-bytes TOOL_SHELL_MAX_INTERMEDIATE_BYTES]
                       [--tool-shell-pipeline-transport {buffered,native}]
                       [--tool-shell-max-arguments TOOL_SHELL_MAX_ARGUMENTS]
                       [--tool-shell-max-argument-bytes TOOL_SHELL_MAX_ARGUMENT_BYTES]
                       [--tool-shell-max-command-bytes TOOL_SHELL_MAX_COMMAND_BYTES]
                       [--tool-shell-max-path-count TOOL_SHELL_MAX_PATH_COUNT]
                       [--tool-shell-max-glob-count TOOL_SHELL_MAX_GLOB_COUNT]
                       [--tool-shell-max-glob-bytes-per-glob TOOL_SHELL_MAX_GLOB_BYTES_PER_GLOB]
                       [--tool-shell-max-total-glob-bytes TOOL_SHELL_MAX_TOTAL_GLOB_BYTES]
                       [--tool-shell-max-full-file-bytes TOOL_SHELL_MAX_FULL_FILE_BYTES]
                       [--tool-shell-max-rg-columns TOOL_SHELL_MAX_RG_COLUMNS]
                       [--tool-shell-max-rg-context-lines TOOL_SHELL_MAX_RG_CONTEXT_LINES]
                       [--tool-shell-max-rg-matches-per-file TOOL_SHELL_MAX_RG_MATCHES_PER_FILE]
                       [--tool-shell-max-head-lines TOOL_SHELL_MAX_HEAD_LINES]
                       [--tool-shell-max-tail-lines TOOL_SHELL_MAX_TAIL_LINES]
                       [--tool-shell-max-text-filter-input-bytes TOOL_SHELL_MAX_TEXT_FILTER_INPUT_BYTES]
                       [--tool-shell-max-filter-program-bytes TOOL_SHELL_MAX_FILTER_PROGRAM_BYTES]
                       [--tool-shell-max-filter-pattern-bytes TOOL_SHELL_MAX_FILTER_PATTERN_BYTES]
                       [--tool-shell-max-filter-selectors TOOL_SHELL_MAX_FILTER_SELECTORS]
                       [--tool-shell-max-awk-fields TOOL_SHELL_MAX_AWK_FIELDS]
                       [--tool-shell-max-awk-separator-bytes TOOL_SHELL_MAX_AWK_SEPARATOR_BYTES]
                       [--tool-shell-max-json-input-bytes TOOL_SHELL_MAX_JSON_INPUT_BYTES]
                       [--tool-shell-max-jq-filter-bytes TOOL_SHELL_MAX_JQ_FILTER_BYTES]
                       [--tool-shell-max-pdf-input-bytes TOOL_SHELL_MAX_PDF_INPUT_BYTES]
                       [--tool-shell-max-pdf-text-pages TOOL_SHELL_MAX_PDF_TEXT_PAGES]
                       [--tool-shell-max-pdf-raster-pages TOOL_SHELL_MAX_PDF_RASTER_PAGES]
                       [--tool-shell-max-pdf-raster-dpi TOOL_SHELL_MAX_PDF_RASTER_DPI]
                       [--tool-shell-max-raster-long-edge-pixels TOOL_SHELL_MAX_RASTER_LONG_EDGE_PIXELS]
                       [--tool-shell-max-raster-pixels TOOL_SHELL_MAX_RASTER_PIXELS]
                       [--tool-shell-max-output-files TOOL_SHELL_MAX_OUTPUT_FILES]
                       [--tool-shell-max-output-file-bytes TOOL_SHELL_MAX_OUTPUT_FILE_BYTES]
                       [--tool-shell-max-total-output-file-bytes TOOL_SHELL_MAX_TOTAL_OUTPUT_FILE_BYTES]
                       [--tool-shell-max-inline-output-file-bytes TOOL_SHELL_MAX_INLINE_OUTPUT_FILE_BYTES]
                       [--tool-shell-max-ocr-input-bytes TOOL_SHELL_MAX_OCR_INPUT_BYTES]
                       [--tool-shell-max-ocr-pixels TOOL_SHELL_MAX_OCR_PIXELS]
                       [--tool-shell-max-ocr-languages TOOL_SHELL_MAX_OCR_LANGUAGES]
                       [--tool-shell-max-tesseract-dpi TOOL_SHELL_MAX_TESSERACT_DPI]
                       [--tool-shell-stream-read-chunk-bytes TOOL_SHELL_STREAM_READ_CHUNK_BYTES]
                       [--tool-shell-max-concurrent-processes TOOL_SHELL_MAX_CONCURRENT_PROCESSES]
                       [--tool-shell-max-concurrent-heavy-processes TOOL_SHELL_MAX_CONCURRENT_HEAVY_PROCESSES]
                       [--tool-shell-default-pdf-timeout-seconds TOOL_SHELL_DEFAULT_PDF_TIMEOUT_SECONDS]
                       [--tool-shell-max-pdf-timeout-seconds TOOL_SHELL_MAX_PDF_TIMEOUT_SECONDS]
                       [--tool-shell-default-ocr-timeout-seconds TOOL_SHELL_DEFAULT_OCR_TIMEOUT_SECONDS]
                       [--tool-shell-max-ocr-timeout-seconds TOOL_SHELL_MAX_OCR_TIMEOUT_SECONDS]
                       [--tool-shell-tesseract-thread-limit TOOL_SHELL_TESSERACT_THREAD_LIMIT]
                       [--tool-shell-allow-pipelines]
                       [--tool-shell-allow-media-tools] [--tool-shell-allow-absolute-paths]
                       [--tool-shell-allow-symlinks] [--tool-shell-allow-hidden]
                       [--tool-shell-executable-search-path TOOL_SHELL_EXECUTABLE_SEARCH_PATHS]
                       [--tool-shell-executable-path COMMAND=PATH]
                       flow

Run a given flow

positional arguments:
  flow                  Flow to run

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel {auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,replicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available
                        GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets
                        --disable-loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --input TASK_INPUT    Flow input value.
  --input-json TASK_INPUT_JSON
                        Flow input JSON value or @file.
  --file TASK_FILES     Attach a local flow input file as field=path.
  --file-mime TASK_FILE_MIME_TYPES
                        Set a flow file MIME hint as field=mime/type.
  --pdf TASK_PDF        Attach one top-level PDF file input.
  --json                Print successful flow output as compact JSON.
  --output TASK_OUTPUT_PATH
                        Write successful flow output to a JSON file.
  --flow-parallel N     Maximum number of ready flow nodes to execute in parallel (defaults to the number of CPUs)
  --tool TOOL           Enable a tool for strict flow tool nodes.
  --tools TOOLS         Enable tools matching a namespace for strict flow tool nodes.

browser tool settings:
  --tool-browser-engine TOOL_BROWSER_ENGINE
  --tool-browser-search
  --tool-browser-search-context TOOL_BROWSER_SEARCH_CONTEXT
  --tool-browser-search-k TOOL_BROWSER_SEARCH_K
  --tool-browser-debug
  --tool-browser-debug-url TOOL_BROWSER_DEBUG_URL
  --tool-browser-debug-source TOOL_BROWSER_DEBUG_SOURCE
  --tool-browser-slowdown TOOL_BROWSER_SLOWDOWN
  --tool-browser-devtools
  --tool-browser-chromium-sandbox
  --tool-browser-viewport-width TOOL_BROWSER_VIEWPORT_WIDTH
  --tool-browser-viewport-height TOOL_BROWSER_VIEWPORT_HEIGHT
  --tool-browser-scale-factor TOOL_BROWSER_SCALE_FACTOR
  --tool-browser-is-mobile
  --tool-browser-has-touch
  --tool-browser-java-script-enabled

database tool settings:
  --tool-database-dsn TOOL_DATABASE_DSN
  --tool-database-delay-secs TOOL_DATABASE_DELAY_SECS
  --tool-database-identifier-case TOOL_DATABASE_IDENTIFIER_CASE
  --tool-database-read-only
  --tool-database-allowed-commands TOOL_DATABASE_ALLOWED_COMMANDS

graph tool settings:
  --tool-graph-file TOOL_GRAPH_FILE

shell tool settings:
  --tool-shell-backend {local,sandbox,container}
  --tool-shell-workspace-root TOOL_SHELL_WORKSPACE_ROOT
  --tool-shell-cwd TOOL_SHELL_CWD
  --tool-shell-default-timeout-seconds TOOL_SHELL_DEFAULT_TIMEOUT_SECONDS
  --tool-shell-max-timeout-seconds TOOL_SHELL_MAX_TIMEOUT_SECONDS
  --tool-shell-max-stdout-bytes TOOL_SHELL_MAX_STDOUT_BYTES
  --tool-shell-max-stderr-bytes TOOL_SHELL_MAX_STDERR_BYTES
  --tool-shell-max-stdin-bytes TOOL_SHELL_MAX_STDIN_BYTES
  --tool-shell-max-pipeline-stages TOOL_SHELL_MAX_PIPELINE_STAGES
  --tool-shell-max-pipeline-bytes TOOL_SHELL_MAX_PIPELINE_BYTES
  --tool-shell-max-intermediate-bytes TOOL_SHELL_MAX_INTERMEDIATE_BYTES
  --tool-shell-pipeline-transport {buffered,native}
                        Trusted shell pipeline byte transport.
  --tool-shell-max-arguments TOOL_SHELL_MAX_ARGUMENTS
  --tool-shell-max-argument-bytes TOOL_SHELL_MAX_ARGUMENT_BYTES
  --tool-shell-max-command-bytes TOOL_SHELL_MAX_COMMAND_BYTES
  --tool-shell-max-path-count TOOL_SHELL_MAX_PATH_COUNT
  --tool-shell-max-glob-count TOOL_SHELL_MAX_GLOB_COUNT
  --tool-shell-max-glob-bytes-per-glob TOOL_SHELL_MAX_GLOB_BYTES_PER_GLOB
  --tool-shell-max-total-glob-bytes TOOL_SHELL_MAX_TOTAL_GLOB_BYTES
  --tool-shell-max-full-file-bytes TOOL_SHELL_MAX_FULL_FILE_BYTES
  --tool-shell-max-rg-columns TOOL_SHELL_MAX_RG_COLUMNS
  --tool-shell-max-rg-context-lines TOOL_SHELL_MAX_RG_CONTEXT_LINES
  --tool-shell-max-rg-matches-per-file TOOL_SHELL_MAX_RG_MATCHES_PER_FILE
  --tool-shell-max-head-lines TOOL_SHELL_MAX_HEAD_LINES
  --tool-shell-max-tail-lines TOOL_SHELL_MAX_TAIL_LINES
  --tool-shell-max-text-filter-input-bytes TOOL_SHELL_MAX_TEXT_FILTER_INPUT_BYTES
  --tool-shell-max-filter-program-bytes TOOL_SHELL_MAX_FILTER_PROGRAM_BYTES
  --tool-shell-max-filter-pattern-bytes TOOL_SHELL_MAX_FILTER_PATTERN_BYTES
  --tool-shell-max-filter-selectors TOOL_SHELL_MAX_FILTER_SELECTORS
  --tool-shell-max-awk-fields TOOL_SHELL_MAX_AWK_FIELDS
  --tool-shell-max-awk-separator-bytes TOOL_SHELL_MAX_AWK_SEPARATOR_BYTES
  --tool-shell-max-json-input-bytes TOOL_SHELL_MAX_JSON_INPUT_BYTES
  --tool-shell-max-jq-filter-bytes TOOL_SHELL_MAX_JQ_FILTER_BYTES
  --tool-shell-max-pdf-input-bytes TOOL_SHELL_MAX_PDF_INPUT_BYTES
  --tool-shell-max-pdf-text-pages TOOL_SHELL_MAX_PDF_TEXT_PAGES
  --tool-shell-max-pdf-raster-pages TOOL_SHELL_MAX_PDF_RASTER_PAGES
  --tool-shell-max-pdf-raster-dpi TOOL_SHELL_MAX_PDF_RASTER_DPI
  --tool-shell-max-raster-long-edge-pixels TOOL_SHELL_MAX_RASTER_LONG_EDGE_PIXELS
  --tool-shell-max-raster-pixels TOOL_SHELL_MAX_RASTER_PIXELS
  --tool-shell-max-output-files TOOL_SHELL_MAX_OUTPUT_FILES
  --tool-shell-max-output-file-bytes TOOL_SHELL_MAX_OUTPUT_FILE_BYTES
  --tool-shell-max-total-output-file-bytes TOOL_SHELL_MAX_TOTAL_OUTPUT_FILE_BYTES
  --tool-shell-max-inline-output-file-bytes TOOL_SHELL_MAX_INLINE_OUTPUT_FILE_BYTES
  --tool-shell-max-ocr-input-bytes TOOL_SHELL_MAX_OCR_INPUT_BYTES
  --tool-shell-max-ocr-pixels TOOL_SHELL_MAX_OCR_PIXELS
  --tool-shell-max-ocr-languages TOOL_SHELL_MAX_OCR_LANGUAGES
  --tool-shell-max-tesseract-dpi TOOL_SHELL_MAX_TESSERACT_DPI
  --tool-shell-stream-read-chunk-bytes TOOL_SHELL_STREAM_READ_CHUNK_BYTES
  --tool-shell-max-concurrent-processes TOOL_SHELL_MAX_CONCURRENT_PROCESSES
  --tool-shell-max-concurrent-heavy-processes TOOL_SHELL_MAX_CONCURRENT_HEAVY_PROCESSES
  --tool-shell-default-pdf-timeout-seconds TOOL_SHELL_DEFAULT_PDF_TIMEOUT_SECONDS
  --tool-shell-max-pdf-timeout-seconds TOOL_SHELL_MAX_PDF_TIMEOUT_SECONDS
  --tool-shell-default-ocr-timeout-seconds TOOL_SHELL_DEFAULT_OCR_TIMEOUT_SECONDS
  --tool-shell-max-ocr-timeout-seconds TOOL_SHELL_MAX_OCR_TIMEOUT_SECONDS
  --tool-shell-tesseract-thread-limit TOOL_SHELL_TESSERACT_THREAD_LIMIT
  --tool-shell-allow-pipelines
  --tool-shell-allow-media-tools
  --tool-shell-allow-absolute-paths
  --tool-shell-allow-symlinks
  --tool-shell-allow-hidden
  --tool-shell-executable-search-path TOOL_SHELL_EXECUTABLE_SEARCH_PATHS
                        Add a trusted directory used to resolve shell tools.
  --tool-shell-executable-path COMMAND=PATH
                        Map a shell command to a trusted absolute executable.
  --tool-container-backend {docker,apple-container}
  --tool-container-profile TOOL_CONTAINER_PROFILE
  --tool-container-image TOOL_CONTAINER_IMAGE
  --tool-container-workspace-root TOOL_CONTAINER_WORKSPACE_ROOT
  --tool-container-pull-policy {never,if_missing,always}
  --tool-container-platform TOOL_CONTAINER_PLATFORM
  --tool-container-cpu-count TOOL_CONTAINER_CPU_COUNT
  --tool-container-memory-bytes TOOL_CONTAINER_MEMORY_BYTES
  --tool-container-pids TOOL_CONTAINER_PIDS
  --tool-container-timeout-seconds TOOL_CONTAINER_TIMEOUT_SECONDS
  --tool-container-network-mode {none,loopback,allowlist,full}
  --tool-container-review-mode {deny,require_review,preauthorized}
  --tool-sandbox-backend {seatbelt,bubblewrap}
  --tool-sandbox-profile TOOL_SANDBOX_PROFILE
  --tool-sandbox-trusted-executable TOOL_SANDBOX_TRUSTED_EXECUTABLES
  --tool-sandbox-executable-search-root TOOL_SANDBOX_EXECUTABLE_SEARCH_ROOTS
  --tool-sandbox-read-root TOOL_SANDBOX_READ_ROOTS
  --tool-sandbox-write-root TOOL_SANDBOX_WRITE_ROOTS
  --tool-sandbox-deny-root TOOL_SANDBOX_DENY_ROOTS
  --tool-sandbox-scratch-root TOOL_SANDBOX_SCRATCH_ROOTS
  --tool-sandbox-output-root TOOL_SANDBOX_OUTPUT_ROOTS
  --tool-sandbox-network-mode {none,loopback,allowlist,full}
  --tool-sandbox-network-egress TOOL_SANDBOX_NETWORK_EGRESS
  --tool-sandbox-timeout-seconds TOOL_SANDBOX_TIMEOUT_SECONDS
  --tool-sandbox-pids TOOL_SANDBOX_PIDS
  --tool-sandbox-max-stdout-bytes TOOL_SANDBOX_MAX_STDOUT_BYTES
  --tool-sandbox-max-stderr-bytes TOOL_SANDBOX_MAX_STDERR_BYTES
  --tool-sandbox-max-artifact-bytes TOOL_SANDBOX_MAX_ARTIFACT_BYTES
  --tool-sandbox-allow-artifacts
  --tool-sandbox-child-processes {deny,allow}
  --tool-sandbox-inherited-fds {deny,stdio,explicit}
  --tool-shell-container-profile TOOL_SHELL_CONTAINER_PROFILE
  --tool-shell-container-required
  --tool-shell-sandbox-profile TOOL_SHELL_SANDBOX_PROFILE
  --tool-shell-sandbox-required
```

Run a flow from a TOML definition:

```bash
avalan flow run flow.toml --input "hello"
avalan flow run flow.toml --input-json @payload.json --json --output output.json
avalan flow run flow.toml --file document=invoice.pdf --file-mime document=application/pdf
avalan flow run flow.toml --pdf invoice.pdf --json
```

Input and output flags:

| Flag | Shape |
| --- | --- |
| `--input VALUE` | Single scalar or JSON-shaped value for the flow input contract. |
| `--input-json JSON_OR_@FILE` | JSON input value, or `@path` to read JSON from a file. |
| `--file FIELD=PATH` | Local file descriptor for a file field. |
| `--file-mime FIELD=TYPE` | MIME hint, for example `document=application/pdf`. |
| `--pdf PATH` | Shorthand for one top-level PDF file input. |
| `--json` | Print successful structured output as compact JSON. |
| `--output PATH` | Write successful structured output as compact JSON. |
| `--flow-parallel N` | Maximum ready strict flow nodes to execute in parallel. Defaults to the CPU count. |
| `--tool NAME` | Enable one tool for strict flow tool nodes. |
| `--tools NAMESPACE` | Enable tools matching a namespace for strict flow tool nodes. |

Tool node execution uses the enabled `ToolManager`. Disabled, unknown,
ambiguous, path-like, URI-like, or provider-originated tool refs fail before
execution.

Direct `flow run` uses local task execution context for strict flows. Durable
inspection and resume use task-backed flow runs stored in PostgreSQL.

### avalan flow validate

Validate without node execution:

```bash
avalan flow validate flow.toml
avalan flow validate flow.toml --json
```

JSON output contains `ok` and public `diagnostics` fields. Diagnostics avoid
printing private file paths, prompt text, file bytes, provider bodies, token
text, and raw model output.

### avalan flow compile

Compile a flow definition to canonical strict TOML:

```bash
avalan flow compile flow.toml
avalan flow compile flow.toml --output strict.flow.toml
avalan flow compile flow.toml --check --json
avalan flow compile flow.toml --encoding utf-8
```

Without `--output` or `--check`, canonical strict TOML is printed to stdout.
`--output PATH` writes the canonical strict TOML atomically. `--check`
compiles and validates without writing. `--encoding NAME` selects the local
file encoding and defaults to UTF-8. `--json` prints compact status and public
diagnostics instead of TOML.

### avalan flow graph inspect

Inspect static graph authoring data without reading durable runtime state:

```bash
avalan flow graph inspect flow.toml
avalan flow graph inspect flow.toml --json
```

JSON output includes `schema_version`, actual and decorative nodes,
executable and decorative edges, explicit edge IDs, binding state, generated
strict edges, source spans, and public diagnostics. The command is separate
from `avalan flow inspect`, which reads durable task-backed flow runs. Public
output omits raw Mermaid source, graph labels, private file paths, prompts,
file bytes, provider payloads, model output, and token text.

### avalan flow mermaid

Mermaid commands require an explicit import mode:

```bash
avalan flow mermaid parse topology.mmd --mode presentation --json
avalan flow mermaid render topology.mmd --mode presentation
avalan flow mermaid compare topology.mmd flow.toml --mode executable
avalan flow mermaid skeleton topology.mmd \
  --mode presentation \
  --name generated_flow \
  --flow-version 1
```

`presentation` mode preserves safe inert topology and presentation metadata.
`executable` mode rejects unsafe or unsupported executable-import constructs.
Skeleton output is intentionally non-executable until contracts, mappings,
entry behavior, and output behavior are added.

### avalan flow inspect

Inspect a durable task-backed flow run:

```bash
avalan flow inspect RUN_ID \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --store-schema public \
  --after-sequence 0 \
  --json
```

The snapshot includes safe flow, node, edge, retry, loop, artifact, and review
state. It does not print raw prompts, raw files, private paths, provider
handles, token text, exception messages, or stack traces.

### avalan flow trace

Export a sanitized trace:

```bash
avalan flow trace RUN_ID \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --after-sequence 0 \
  --json
```

Trace export is designed for operational inspection and omits raw sensitive
payloads by default.

### avalan flow cancel

Request cancellation for a durable flow run:

```bash
avalan flow cancel RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN" --json
```

The result includes the run id and updated task run state.

### avalan flow resume

Resume a paused human-review flow:

```bash
avalan flow resume review.flow.toml RUN_ID \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --decision-json '{"review":{"decision":"approved","comment":"ok"}}' \
  --json
```

`--decision-json` may also be `@path` to read the decision object from a local
JSON file. The object is keyed by review node name and each value must match
that node's declared decision schema.

## avalan task

`avalan task` validates task definition TOML files, runs direct tasks, enqueues
durable tasks, inspects sanitized run state, sweeps expired artifact bytes, and
manages task PostgreSQL migrations.

```bash
avalan task validate tasks/example.task.toml
avalan task run tasks/example.task.toml --ephemeral --input "Summarize this"
avalan task enqueue tasks/example.task.toml --queue documents --input-json @payload.json
```

Input flags shared by `validate`, `run`, and `enqueue`:

| Flag | Shape |
| --- | --- |
| `--input VALUE` | Single scalar or JSON-shaped value for the task input contract. |
| `--input-json JSON_OR_@FILE` | JSON input value, or `@path` to read JSON from a file. |
| `--input-FIELD VALUE` | Field-addressed object input such as `--input-question "What changed?"`. |
| `--file FIELD=PATH` | Local file descriptor for a file field. |
| `--file-descriptor FIELD=JSON` | Explicit descriptor JSON for local, artifact, inline, remote, or provider reference inputs. |
| `--provider-file-id FIELD=PROVIDER:ID` | Provider-owned durable file id. |
| `--hosted-url FIELD=PROVIDER:URL` | Provider-fetchable hosted URL. |
| `--object-store-uri FIELD=PROVIDER:URI` | Provider/object-store URI such as `google:gs://bucket/key` or `bedrock:s3://bucket/key`. |
| `--file-mime FIELD=TYPE` | MIME hint, for example `document=application/pdf`. |
| `--file-role FIELD=ROLE` | Role hint passed as safe descriptor metadata. |
| `--file-size FIELD=BYTES` | Size hint used for validation and delivery planning. |
| `--file-sha256 FIELD=HEX` | Digest hint used for validation and materialization. |
| `--file-conversion FIELD=NAME[:JSON]` | Conversion request allowed by the task input contract. |

Provider file ids, hosted URLs, and object-store URIs are provider-native
references. `--file-conversion` applies to local, artifact, inline, or remote
descriptors and is rejected for provider-native references.

### avalan task validate

Validate a task definition without executing it:

```bash
avalan task validate tasks/example.task.toml
```

Validation output aggregates load and contract issues and avoids printing raw
prompts, outputs, file paths, or sensitive execution references.

### avalan task run

Run a direct-mode task. A durable PostgreSQL store is required unless
`--ephemeral` is passed for local experimentation:

```bash
avalan task run docs/examples/tasks/minimal_string_agent.task.toml \
  --ephemeral \
  --input "What changed?"

avalan task run docs/examples/tasks/large_direct_file.task.toml \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --file document=docs/examples/playground/invoice.pdf \
  --file-mime document=application/pdf \
  --file-conversion document=text

avalan task run docs/examples/tasks/provider_reference_direct.task.toml \
  --ephemeral \
  --provider-file-id document=openai:file_abc123 \
  --file-mime document=application/pdf

avalan task run docs/examples/tasks/structured_json.task.toml \
  --ephemeral \
  --input-json '{"question":"What changed?","priority":2}' \
  --json \
  --output result.json

poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml --ephemeral --pdf ./sample.pdf --json --output extraction.json

poetry run avalan task run docs/examples/tasks/poc_extraction/flow_task.toml --ephemeral --pdf ./sample.pdf --json --output extraction.json

poetry run avalan flow run docs/examples/tasks/poc_extraction/flow.toml --pdf ./sample.pdf --json --output extraction.json

poetry run avalan task run docs/examples/tasks/poc_extraction/task.toml \
  --ephemeral \
  --file input=./sample.pdf \
  --file-mime input=application/pdf \
  --json \
  --output extraction.json
```

Use `--store-schema` or `AVALAN_TASK_STORE_SCHEMA` when the task schema is not
on the default search path. `--json` prints exactly one compact JSON document
for successful `json`, `object`, and `array` outputs. `--output PATH` writes the
same structured value atomically as compact JSON with a trailing newline.
`--pdf PATH` is shorthand for one top-level PDF file input and is equivalent to
`--file input=PATH --file-mime input=application/pdf`. Direct ephemeral runs
that materialize local bytes use a temporary artifact root when
`AVALAN_TASK_ARTIFACT_ROOT` is not configured.

Minimal direct PDF extraction files:

```toml
# task.toml
[task]
name = "pdf_extraction"
version = "1"

[input]
type = "file"
mime_types = ["application/pdf"]

[output]
type = "object"
schema_ref = "extraction.schema.json"

[execution]
type = "agent"
ref = "agent.toml"
```

```toml
# agent.toml
[agent]
name = "PDF Extraction"
system = "Extract one JSON object that matches the configured schema."
user = "Analyze the attached PDF and return the requested fields."

[engine]
uri = "ai://env:AZURE_OPENAI_API_KEY@openai/extraction-deployment"
base_url = "https://tenant.openai.azure.com/openai/v1/"

[run]
use_async_generator = false

[run.response_format]
type = "json_schema"
name = "extraction"
strict = true
schema_ref = "extraction.schema.json"
```

Run it with:

```bash
poetry run avalan task run task.toml --ephemeral --pdf ./sample.pdf --json --output extraction.json
```

The optional flow-backed form keeps provider configuration and prompts in the
agent file while `flow.toml` declares the agent node:

```toml
[flow]
name = "pdf_extraction_flow"
entrypoint = "extract"
output_node = "extract"

[flow.input]
name = "input"
type = "file"
mime_types = ["application/pdf"]

[flow.output]
name = "extraction"
type = "object"
schema_ref = "extraction.schema.json"

[nodes.extract]
type = "agent"
ref = "agent.toml"
input = "__task_input__"
```

Run the flow-backed task or the native flow command with the same PDF input:

```bash
poetry run avalan task run flow_task.toml --ephemeral --pdf ./sample.pdf --json --output extraction.json
poetry run avalan flow run flow.toml --pdf ./sample.pdf --json --output extraction.json
```

Task file inputs are run-scoped delivery inputs. They are validated,
materialized only when needed, and sent through the task file-delivery planner;
they are not indexed into message memory or document memory unless an explicit
conversion, retrieval, or map-reduce text strategy is declared.

### avalan task enqueue

Submit a queue-mode task to a durable store:

```bash
avalan task enqueue docs/examples/tasks/queued_file_task.task.toml \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --queue documents \
  --file documents=docs/examples/playground/invoice.pdf \
  --file-mime documents=application/pdf \
  --file-conversion documents=text \
  --wait \
  --wait-timeout 120 \
  --poll-interval 2
```

`enqueue` supports `--store-dsn`, `--store-schema`, `--queue`, `--wait`,
`--wait-timeout`, and `--poll-interval`. Ephemeral storage is rejected for
queued tasks because workers must reconstruct executable input from durable
payloads, artifact references, or durable provider references.

### avalan task inspect

Inspect sanitized run state:

```bash
avalan task inspect RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN" --after-sequence 0
```

The snapshot includes run state and sanitized events after the optional
sequence cursor. It does not print raw prompts, file bytes, file paths, provider
handles, token text, exception messages, or stack traces.

Direct runs with `--ephemeral` are useful for local output checks only. To
inspect records after the process exits, run with a durable store or inspect
the in-memory client from the same SDK process that executed the task.

For live usage smoke, keep inputs, outputs, and summaries in an ignored local
workspace. Run without `--ephemeral`, capture the run id printed after
`Task run completed:`, then call `avalan task inspect` and
`avalan task usage --source exact` against the same durable store. Ephemeral
runs validate output shape only; SDK smoke can use `InMemoryTaskStore` when
inspection happens in the same process.

### avalan task usage

Print usage records and rollups for a run:

```bash
avalan task usage RUN_ID \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --attempt-id ATTEMPT_ID \
  --source exact
```

The output contains each usage record with its attempt id, sequence, source,
counters, privacy-safe metadata, and a `usage_totals` object. The rollup is
computed from the records returned by the selected filters. Missing counters
are printed as `null`; reported zero values remain `0`.

Provider usage support:

| Provider family | Non-streaming | Streaming | Exact cache read | Exact cache creation | Exact reasoning | Unavailable fields |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI / Azure Responses | Supported from provider usage objects. | Supported from terminal response usage events. | `cached_input_tokens` from provider cache-read details. | `null` unless the provider reports a write counter. | `reasoning_tokens` from provider output-token details. | Cache creation when absent; any malformed or missing counter. |
| Anthropic | Supported from provider usage objects. | Supported from terminal or cumulative provider usage events. | `cached_input_tokens` from cache-read counters. | `cache_creation_input_tokens` from provider cache-write counters. | `reasoning_tokens` from provider thinking-token details. | Any counter absent from provider usage; visible thinking text is ignored. |
| Bedrock | Supported from Converse usage metadata. | Supported from terminal Converse stream metadata. | `cached_input_tokens` from cache-read counters. | `cache_creation_input_tokens` from cache-write counters. | `null` unless a model-specific exact field reports it. | General reasoning-token split and malformed counters. |
| Google / Gemini | Supported from `usage_metadata` / `usageMetadata`. | Supported from terminal stream usage metadata. | `cached_input_tokens` from cached-content counters. | `null` for separately created cached content unless reported by the call. | `reasoning_tokens` from thoughts-token counters. | Cache creation and absent provider fields. |
| OpenAI-compatible vendors | Supported when the adapter exposes OpenAI-compatible usage shapes. | Supported when the adapter exposes trustworthy terminal usage. | Supported only when provider cache-read details are present. | `null` unless explicitly reported. | Supported only when completion reasoning details are present. | Provider-specific cache or reasoning fields that are not exposed. |
| Local / estimated vendors | Local token counters are marked `estimated`. | Local stream counters are marked `estimated` only when produced by the local response. | `null`. | `null`. | `null`. | Exact provider cache, cache creation, reasoning, and total-token counters. |

Exact counters are copied only from provider-returned usage metadata. Avalan
does not infer cache hits, cache writes, reasoning tokens, or provider
`total_tokens` from prompt size, local tokenization, repeated prompts, latency,
reasoning settings, or streamed token text.

### avalan task output

Print the sanitized output snapshot for a run:

```bash
avalan task output RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN"
```

### avalan task events

Print sanitized task events:

```bash
avalan task events RUN_ID \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --attempt-id ATTEMPT_ID \
  --after-sequence 100
```

### avalan task artifacts

Print sanitized artifact metadata:

```bash
avalan task artifacts RUN_ID --store-dsn "$AVALAN_TASK_STORE_DSN"
```

Artifact inspection reports references and retention state, not raw bytes,
local storage paths, bucket names, object keys, signed URLs, or credentials.

### avalan task worker

Run a bounded queue worker:

```bash
avalan task worker \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --queue documents \
  --worker-id "documents-worker-1" \
  --lease-seconds 300 \
  --heartbeat-seconds 30 \
  --limit 100
```

`--once` processes at most one available run. `--heartbeat-seconds` must be
shorter than `--lease-seconds`. Workers need the durable store, HMAC key
environment, provider credentials required by the referenced agents, and
`AVALAN_TASK_ARTIFACT_ROOT` when local artifact bytes are used.

### avalan task retention-sweep

Delete expired artifact bytes while keeping sanitized audit metadata:

```bash
avalan task retention-sweep \
  --store-dsn "$AVALAN_TASK_STORE_DSN" \
  --purpose input \
  --purpose output \
  --limit 500
```

`--purpose` may be `input`, `output`, `converted`, or `intermediate` and can be
passed more than once. The command requires `AVALAN_TASK_ARTIFACT_ROOT` for the
local artifact backend.

### avalan task pgsql

Manage PostgreSQL task schema migrations. The command accepts `--dsn` and
`--schema`, or reads `AVALAN_TASK_PGSQL_DSN` and
`AVALAN_TASK_PGSQL_SCHEMA` when the options are omitted.

```bash
avalan task pgsql status
avalan task pgsql check
avalan task pgsql migrate head
avalan task pgsql stamp head
avalan task pgsql diagnose
```

The migration commands require Alembic and SQLAlchemy in the environment that
runs them:

```bash
python3 -m pip install -U alembic "SQLAlchemy>=2.0.43,<3.0.0"
```

Diagnostics report whether a DSN is configured, the schema, the migration
head revision, the version table, and the script location without printing
raw DSNs or credentials.

## avalan memory

```
usage: avalan memory [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                     [--device DEVICE]
                     [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                     [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                     [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                     [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                     [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                     [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                     {embeddings,search,document} ...

Manage memory

positional arguments:
  {embeddings,search,document}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

### avalan memory embeddings

```
usage: avalan memory embeddings [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER]
                                [--tokenizer-subfolder TOKENIZER_SUBFOLDER] [--device DEVICE]
                                [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                                [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                                [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                                [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                                [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                                [--verbose] [--version]
                                [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                                [--base-url BASE_URL] [--load] [--special-token SPECIAL_TOKEN] [--token TOKEN]
                                [--tokenizer TOKENIZER] [--no-display-partitions | --display-partitions DISPLAY_PARTITIONS]
                                [--partition] [--partition-max-tokens PARTITION_MAX_TOKENS]
                                [--partition-overlap PARTITION_OVERLAP] [--partition-window PARTITION_WINDOW]
                                [--compare COMPARE] [--search SEARCH] [--search-k SEARCH_K]
                                [--sort {cosine,dot,l1,l2,pearson}]
                                model

Obtain and manipulate embeddings

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --base-url BASE_URL   If specified and model is a vendor model that supports it,load model using the given base URL
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default, only if model is loaded
  --no-display-partitions
                        If specified, don't display memory partitions
  --display-partitions DISPLAY_PARTITIONS
                        Display up to this many partitions, if more summarize
  --partition           If specified, partition string
  --partition-max-tokens PARTITION_MAX_TOKENS
                        Maximum number of tokens to allow on each partition
  --partition-overlap PARTITION_OVERLAP
                        How many tokens can potentially overlap in different partitions
  --partition-window PARTITION_WINDOW
                        Number of tokens per window when partitioning
  --compare COMPARE     If specified, compare embeddings with this string
  --search SEARCH       If specified, search across embeddings for this string
  --search-k SEARCH_K   How many nearest neighbors to obtain with search
  --sort {cosine,dot,l1,l2,pearson}
                        Sort comparison results using the given similarity measure
```

### avalan memory search

```
usage: avalan memory search [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                            [--device DEVICE]
                            [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                            [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                            [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                            [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                            [--verbose] [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                            [--base-url BASE_URL] [--load] [--special-token SPECIAL_TOKEN] [--token TOKEN]
                            [--tokenizer TOKENIZER] [--no-display-partitions | --display-partitions DISPLAY_PARTITIONS]
                            [--partition] [--partition-max-tokens PARTITION_MAX_TOKENS]
                            [--partition-overlap PARTITION_OVERLAP] [--partition-window PARTITION_WINDOW] --dsn DSN
                            --participant PARTICIPANT --namespace NAMESPACE --function
                            {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms} [--limit LIMIT]
                            model

Search memories

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --base-url BASE_URL   If specified and model is a vendor model that supports it,load model using the given base URL
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default, only if model is loaded
  --no-display-partitions
                        If specified, don't display memory partitions
  --display-partitions DISPLAY_PARTITIONS
                        Display up to this many partitions, if more summarize
  --partition           If specified, partition string
  --partition-max-tokens PARTITION_MAX_TOKENS
                        Maximum number of tokens to allow on each partition
  --partition-overlap PARTITION_OVERLAP
                        How many tokens can potentially overlap in different partitions
  --partition-window PARTITION_WINDOW
                        Number of tokens per window when partitioning
  --dsn DSN             PostgreSQL DSN for searching
  --participant PARTICIPANT
                        Participant ID to search
  --namespace NAMESPACE
                        Namespace to search
  --function {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms}
                        Vector function to use for searching
  --limit LIMIT         Return up to this many memories
```

### avalan memory document

```
usage: avalan memory document [-h] {index} ...

Manage memory indexed documents

positional arguments:
  {index}

options:
  -h, --help  show this help message and exit
```

#### avalan memory document index

```
usage: avalan memory document index [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER]
                                    [--tokenizer-subfolder TOKENIZER_SUBFOLDER] [--device DEVICE]
                                    [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                                    [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                                    [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                                    [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                                    [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
                                    [--skip-hub-access-check] [--verbose] [--version]
                                    [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                                    [--base-url BASE_URL] [--load] [--special-token SPECIAL_TOKEN] [--token TOKEN]
                                    [--tokenizer TOKENIZER]
                                    [--no-display-partitions | --display-partitions DISPLAY_PARTITIONS] [--partition]
                                    [--partition-max-tokens PARTITION_MAX_TOKENS] [--partition-overlap PARTITION_OVERLAP]
                                    [--partition-window PARTITION_WINDOW] [--partitioner {text,code}] [--language LANGUAGE]
                                    [--encoding ENCODING] [--identifier IDENTIFIER] [--title TITLE]
                                    [--description DESCRIPTION] --dsn DSN --participant PARTICIPANT --namespace NAMESPACE
                                    model source

Add a document to the memory index

positional arguments:
  model                 Model to use
  source                Source to index (an URL or a file path)

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --base-url BASE_URL   If specified and model is a vendor model that supports it,load model using the given base URL
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default, only if model is loaded
  --no-display-partitions
                        If specified, don't display memory partitions
  --display-partitions DISPLAY_PARTITIONS
                        Display up to this many partitions, if more summarize
  --partition           If specified, partition string
  --partition-max-tokens PARTITION_MAX_TOKENS
                        Maximum number of tokens to allow on each partition
  --partition-overlap PARTITION_OVERLAP
                        How many tokens can potentially overlap in different partitions
  --partition-window PARTITION_WINDOW
                        Number of tokens per window when partitioning
  --partitioner {text,code}
                        Partitioner to use when indexing a file
  --language LANGUAGE   Programming language for the code partitioner
  --encoding ENCODING   File encoding used when reading a local file
  --identifier IDENTIFIER
                        Identifier for the memory entry (defaults to the source)
  --title TITLE         Title for the memory entry
  --description DESCRIPTION
                        Description for the memory entry
  --dsn DSN             PostgreSQL DSN for storing the document
  --participant PARTICIPANT
                        Participant ID for the memory entry
  --namespace NAMESPACE
                        Namespace for the memory entry
```

## avalan model

```
usage: avalan model [-h] {display,install,run,search,uninstall} ...

Manage a model, showing details, loading or downloading it

positional arguments:
  {display,install,run,search,uninstall}

options:
  -h, --help            show this help message and exit
```

### avalan model display

```
usage: avalan model display [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                            [--device DEVICE]
                            [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                            [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                            [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                            [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                            [--verbose] [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                            [--base-url BASE_URL] [--load] [--special-token SPECIAL_TOKEN] [--token TOKEN]
                            [--tokenizer TOKENIZER] [--sentence-transformer] [--summary]
                            model

Show information about a model

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --base-url BASE_URL   If specified and model is a vendor model that supports it,load model using the given base URL
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default, only if model is loaded
  --sentence-transformer
                        Load the model as a SentenceTransformer model
  --summary
```

### avalan model install

```
usage: avalan model install [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                            [--device DEVICE]
                            [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                            [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                            [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                            [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                            [--verbose] [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                            [--workers WORKERS] [--local-dir LOCAL_DIR] [--local-dir-symlinks]
                            model

Install a model

positional arguments:
  model                 Model to download

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --workers WORKERS     How many download workers to use
  --local-dir LOCAL_DIR
                        Local directory to download the model to
  --local-dir-symlinks  Use symlinks when downloading to local dir
```

### avalan model run

```
usage: avalan model run [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                        [--device DEVICE]
                        [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                        [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                        [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--base-url BASE_URL]
                        [--load] [--special-token SPECIAL_TOKEN] [--token TOKEN] [--tokenizer TOKENIZER] [--display-events] [--stats]
                        [--display-pause [DISPLAY_PAUSE]] [--display-probabilities]
                        [--display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM]
                        [--display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM]
                        [--display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]] [--skip-display-reasoning-time]
                        [--display-reasoning] [--display-tokens [DISPLAY_TOKENS]] [--display-tools]
                        [--display-tools-events DISPLAY_TOOLS_EVENTS]
                        [--display-answer-height-expand | --display-answer-height DISPLAY_ANSWER_HEIGHT]
                        [--attention {eager,flash_attention_2,flex_attention,sdpa}] [--output-hidden-states] [--path PATH]
                        [--checkpoint CHECKPOINT] [--base-model BASE_MODEL] [--upsampler-model UPSAMPLER_MODEL]
                        [--refiner-model REFINER_MODEL] [--audio-reference-path AUDIO_REFERENCE_PATH]
                        [--audio-reference-text AUDIO_REFERENCE_TEXT] [--audio-sampling-rate AUDIO_SAMPLING_RATE]
                        [--vision-threshold VISION_THRESHOLD] [--vision-width VISION_WIDTH]
                        [--vision-color-model {1,L,LA,P,PA,RGB,RGBA,RGBX,CMYK,YCbCr,LAB,HSV,I,F}]
                        [--vision-image-format 
{BMP,DDS,EPS,GIF,ICNS,ICO,IM,JPEG,JPEG2000,MSP,PCX,PNG,PPM,SGI,SPI,TGA,TIFF,WEBP,XBM}]
                        [--vision-high-noise-frac VISION_HIGH_NOISE_FRAC] [--vision-steps VISION_STEPS]
                        [--vision-timestep-spacing {linspace,leading,trailing}]
                        [--vision-beta-schedule {linear,scaled_linear,squaredcos_cap_v2}]
                        [--vision-guidance-scale VISION_GUIDANCE_SCALE] [--vision-reference-path VISION_REFERENCE_PATH]
                        [--vision-negative-prompt VISION_NEGATIVE_PROMPT] [--vision-height VISION_HEIGHT]
                        [--vision-downscale VISION_DOWNSCALE] [--vision-frames VISION_FRAMES]
                        [--vision-denoise-strength VISION_DENOISE_STRENGTH] [--vision-inference-steps VISION_INFERENCE_STEPS]
                        [--vision-decode-timestep VISION_DECODE_TIMESTEP] [--vision-noise-scale VISION_NOISE_SCALE]
                        [--vision-fps VISION_FPS] [--do-sample] [--enable-gradient-calculation] [--use-cache]
                        [--max-new-tokens MAX_NEW_TOKENS]
                        [--modality 
{audio_classification,audio_speech_recognition,audio_text_to_speech,audio_generation,embedding,text_generation,text_question_ans
wering,text_sequence_classification,text_sequence_to_sequence,text_translation,text_token_classification,vision_object_detection
,vision_image_classification,vision_image_to_text,vision_text_to_image,vision_text_to_animation,vision_text_to_video,vision_imag
e_text_to_text,vision_encoder_decoder,vision_semantic_segmentation}]
                        [--min-p MIN_P] [--repetition-penalty REPETITION_PENALTY] [--skip-special-tokens] [--system SYSTEM]
                        [--developer DEVELOPER] [--input-file INPUT_FILE] [--ds4-ctx DS4_CTX]
                        [--ds4-native-backend {auto,metal,cuda,cpu}] [--ds4-mtp DS4_MTP]
                        [--ds4-mtp-draft DS4_MTP_DRAFT] [--ds4-mtp-margin DS4_MTP_MARGIN]
                        [--ds4-warm-weights] [--ds4-quality] [--text-context TEXT_CONTEXT] [--text-labeled-only]
                        [--text-max-length TEXT_MAX_LENGTH] [--text-num-beams TEXT_NUM_BEAMS] [--text-disable-cache]
                        [--text-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}]
                        [--text-from-lang TEXT_FROM_LANG] [--text-to-lang TEXT_TO_LANG] [--start-thinking]
                        [--chat-disable-thinking] [--no-reasoning] [--reasoning-tag {think,channel}]
                        [--reasoning-effort {none,minimal,low,medium,high,xhigh,max}]
                        [--reasoning-max-new-tokens REASONING_MAX_NEW_TOKENS] [--reasoning-stop-on-max-new-tokens]
                        [--stop_on_keyword STOP_ON_KEYWORD] [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P]
                        [--trust-remote-code]
                        model

Run a model

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --base-url BASE_URL   If specified and model is a vendor model that supports it,load model using the given base URL
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default, only if model is loaded
  --display-events      Show non-tool stream events when an orchestrator or agent is involved.
  --stats               Show token generation statistics for streaming output
  --display-pause [DISPLAY_PAUSE]
                        Pause (in ms.) when cycling through selected tokens as defined by --display-probabilities
  --display-probabilities
                        If --display-tokens specified, show also the token probability distribution
  --display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM
                        When --display-probabilities is used, select tokens which logit probability is no higher than this
                        value. Defaults to 0.8
  --display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM
                        When --display-probabilities is used, select tokens that have alternate tokens with a logit
                        probability at least or higher than this value. Defaults to 0.1
  --display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]
                        Display the time it takes to reach the given Nth token (defaults to 256)
  --skip-display-reasoning-time
                        Don't display total reasoning time
  --display-reasoning   Display streamed reasoning text in the live response panel
  --display-tokens [DISPLAY_TOKENS]
                        How many tokens with full information to display at a time
  --display-tools       Show tool lifecycle details for agent or orchestrator runs.
  --display-tools-events DISPLAY_TOOLS_EVENTS
                        How many tool events to show on tool call panel.
                        Defaults to all retained tool events; use 0 to hide
                        completed tool history.
  --display-answer-height-expand
                        Expand answer section to full height
  --display-answer-height DISPLAY_ANSWER_HEIGHT
                        Height of the answer section (defaults to 12)
  --attention {eager,flash_attention_2,flex_attention,sdpa}
                        Attention implementation to use (defaults to best available: None)
  --output-hidden-states
                        Return hidden states for each layer
  --path PATH           Path where to store generated audio or vision output. Only applicable to audio and vision modalities.
  --checkpoint CHECKPOINT
                        AnimateDiff motion adapter checkpoint to use. Only applicable to vision text to video modality.
  --base-model BASE_MODEL
                        ID of the base model for text-to-video generation. Only applicable to vision text to video modality.
  --upsampler-model UPSAMPLER_MODEL
                        Upsampler model to use for text-to-video generation. Only applicable to vision text to video modality.
  --refiner-model REFINER_MODEL
                        Expert model to use for refinement. Only applicable to vision text to image modality.
  --audio-reference-path AUDIO_REFERENCE_PATH
                        Path to existing audio file to use for voice cloning. Only applicable to audio modalities.
  --audio-reference-text AUDIO_REFERENCE_TEXT
                        Text transcript of the reference audio given in --audio-reference-path. Only applicable to audio
                        modalities.
  --audio-sampling-rate AUDIO_SAMPLING_RATE
                        Sampling rate to use for audio generation. Only applicable to audio modalities.
  --vision-threshold VISION_THRESHOLD
                        Score threshold for object detection. Only applicable to vision modalities.
  --vision-width VISION_WIDTH
                        Resize input image to this width before processing. Only applicable to vision image text to text
                        modality.
  --vision-color-model {1,L,LA,P,PA,RGB,RGBA,RGBX,CMYK,YCbCr,LAB,HSV,I,F}
                        Color model for image generation. Only applicable to vision text to image modality.
  --vision-image-format {BMP,DDS,EPS,GIF,ICNS,ICO,IM,JPEG,JPEG2000,MSP,PCX,PNG,PPM,SGI,SPI,TGA,TIFF,WEBP,XBM}
                        Image format to save generated image. Only applicable to vision text to image modality.
  --vision-high-noise-frac VISION_HIGH_NOISE_FRAC
                        High noise fraction for diffusion (controls the split point between the base model and the refiner.
                        Only applicable to vision text to image modality.
  --vision-steps VISION_STEPS
                        Number of denoising (sampling) iterations in the diffusion scheduler. Only applicable to vision text
                        to image modality.
  --vision-timestep-spacing {linspace,leading,trailing}
                        Timestep spacing strategy for the Euler scheduler. Only applicable to vision text to video modality.
  --vision-beta-schedule {linear,scaled_linear,squaredcos_cap_v2}
                        Beta schedule for the Euler scheduler. Only applicable to vision text to video modality.
  --vision-guidance-scale VISION_GUIDANCE_SCALE
                        Guidance scale for text-to-video generation. Only applicable to vision text to video modality.
  --vision-reference-path VISION_REFERENCE_PATH
                        Reference image to guide generation. Only applicable to vision text to video modality.
  --vision-negative-prompt VISION_NEGATIVE_PROMPT
                        Negative prompt for generation. Only applicable to vision text to video modality.
  --vision-height VISION_HEIGHT
                        Height of generated video. Only applicable to vision text to video modality.
  --vision-downscale VISION_DOWNSCALE
                        Downscale factor for upsampling. Only applicable to vision text to video modality.
  --vision-frames VISION_FRAMES
                        Number of frames to generate. Only applicable to vision text to video modality.
  --vision-denoise-strength VISION_DENOISE_STRENGTH
                        Denoise strength for upsampling. Only applicable to vision text to video modality.
  --vision-inference-steps VISION_INFERENCE_STEPS
                        Number of inference steps for upsampler. Only applicable to vision text to video modality.
  --vision-decode-timestep VISION_DECODE_TIMESTEP
                        Decode timestep for video decoding. Only applicable to vision text to video modality.
  --vision-noise-scale VISION_NOISE_SCALE
                        Noise scale for video generation. Only applicable to vision text to video modality.
  --vision-fps VISION_FPS
                        Frames per second for generated video. Only applicable to vision text to video modality.
  --do-sample           Tell if the token generation process should be deterministic or stochastic. When enabled, it's
                        stochastic and uses probability distribution.
  --enable-gradient-calculation
                        Enable gradient calculation.
  --use-cache           Past key values are used to speed up decoding if applicable to model.
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of tokens to generate
  --modality 
{audio_classification,audio_speech_recognition,audio_text_to_speech,audio_generation,embedding,text_generation,text_question_ans
wering,text_sequence_classification,text_sequence_to_sequence,text_translation,text_token_classification,vision_object_detection
,vision_image_classification,vision_image_to_text,vision_text_to_image,vision_text_to_animation,vision_text_to_video,vision_imag
e_text_to_text,vision_encoder_decoder,vision_semantic_segmentation}
  --min-p MIN_P         Minimum token probability, which will be scaled by the probability of the most likely token [0, 1]
  --repetition-penalty REPETITION_PENALTY
                        Exponential penalty on sequences not in the original input. Defaults to 1.0, which means no penalty.
  --skip-special-tokens
                        If specified, skip special tokens when decoding
  --system SYSTEM       Use this as system prompt
  --developer DEVELOPER
                        Use this as developer prompt
  --input-file INPUT_FILE
                        Attach a local file as native input for text generation. May be specified multiple times.
  --text-context TEXT_CONTEXT
                        Context string for question answering
  --text-labeled-only   If specified, only tokens with labels detected are returned. Only applicable to
                        text_token_classification modalities.
  --text-max-length TEXT_MAX_LENGTH
                        The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                        max_new_tokens
  --text-num-beams TEXT_NUM_BEAMS
                        Number of beams for beam search. 1 means no beam search
  --text-disable-cache  If specified, disable generation cache
  --text-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}
                        Cache implementation to use for generation
  --text-from-lang TEXT_FROM_LANG
                        Source language code for text translation
  --text-to-lang TEXT_TO_LANG
                        Destination language code for text translation
  --start-thinking      If specified, assume model response starts with reasoning
  --chat-disable-thinking
                        Disable thinking tokens in chat template
  --no-reasoning        Disable reasoning parser
  --reasoning-tag {think,channel}
                        Reasoning tag style
  --reasoning-effort {none,minimal,low,medium,high,xhigh,max}
                        Reasoning effort level
  --reasoning-max-new-tokens REASONING_MAX_NEW_TOKENS
                        Maximum number of reasoning tokens
  --reasoning-stop-on-max-new-tokens
                        Stop reasoning when maximum tokens are produced
  --stop_on_keyword STOP_ON_KEYWORD
                        Stop token generation when this keyword is found
  --temperature TEMPERATURE
                        Temperature [0, 1]
  --top-k TOP_K         Number of highest probability vocabulary tokens to keep for top-k-filtering.
  --top-p TOP_P         If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p
                        or higher are kept for generation.
  --trust-remote-code

DS4 backend options:
  --ds4-ctx DS4_CTX     DS4 context size
  --ds4-native-backend {auto,metal,cuda,cpu}
                        DS4 native backend
  --ds4-mtp DS4_MTP     DS4 MTP model path
  --ds4-mtp-draft DS4_MTP_DRAFT
                        DS4 MTP draft-token count
  --ds4-mtp-margin DS4_MTP_MARGIN
                        DS4 MTP acceptance margin
  --ds4-warm-weights    Warm DS4 model weights when opening the engine
  --ds4-quality         Enable DS4 quality mode
```

### avalan model search

```
usage: avalan model search [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                           [--device DEVICE]
                           [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                           [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                           [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                           [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
                           [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check] [--verbose]
                           [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                           [--search SEARCH] [--filter FILTER] [--library LIBRARY] [--author AUTHOR] [--gated | --open]
                           [--language LANGUAGE] [--name NAME] [--task TASK] [--tag TAG] [--limit LIMIT]

Search for models

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --search SEARCH       Search for models matching given expression
  --filter FILTER       Filter models on this (e.g: text-classification)
  --library LIBRARY     Filter by library
  --author AUTHOR       Filter by author
  --gated               Only gated models
  --open                Only open models
  --language LANGUAGE   Filter by language
  --name NAME           Filter by model name
  --task TASK           Filter by task
  --tag TAG             Filter by tag
  --limit LIMIT         Maximum number of models to return
```

### avalan model uninstall

```
usage: avalan model uninstall [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                              [--device DEVICE]
                              [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                              [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                              [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                              [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                              [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check]
                              [--verbose] [--version]
                              [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--base-url BASE_URL]
                              [--load] [--special-token SPECIAL_TOKEN] [--token TOKEN] [--tokenizer TOKENIZER] [--delete]
                              model

Uninstall a model

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --base-url BASE_URL   If specified and model is a vendor model that supports it,load model using the given base URL
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default, only if model is loaded
  --delete              Actually delete. If not provided, a dry run is performed and data that would be deleted is shown, yet
                        not deleted
```

## avalan tokenizer

```
usage: avalan tokenizer [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                        [--device DEVICE]
                        [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                        [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                        [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] --tokenizer TOKENIZER
                        [--save SAVE] [--special-token SPECIAL_TOKEN] [--token TOKEN]

Manage tokenizers, loading, modifying and saving them

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --tokenizer TOKENIZER, -t TOKENIZER
                        Tokenizer to load
  --save SAVE           Save tokenizer (useful if modified via --special-token or --token) to given path, only if model is
                        loaded
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer
  --token TOKEN         Token to add to tokenizer
```

## avalan train

```
usage: avalan train [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                    [--device DEVICE]
                    [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                    [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE] [--theme {fancy,basic}]
                    [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}] [--locales LOCALES]
                    [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
                    [--skip-hub-access-check] [--verbose] [--version]
                    [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                    {run} ...

Training

positional arguments:
  {run}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

### avalan train run

```
usage: avalan train run [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                        [--device DEVICE]
                        [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                        [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE] [--theme {fancy,basic}] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm,ds4}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                        [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                        training

Run a given training

positional arguments:
  training              Training to run

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to /root/.cache/huggingface/hub, can also be specified with
                        $HF_HUB_CACHE)
  --subfolder SUBFOLDER
                        Subfolder inside model repository to load the model from
  --tokenizer-subfolder TOKENIZER_SUBFOLDER
                        Subfolder inside model repository to load the tokenizer from
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}
                        Tensor parallelism strategy to use
  --parallel-count PARALLEL_COUNT
                        Number of processes to launch when --parallel is used (defaults to the number of available GPUs)
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --theme {fancy,basic}
                        Theme to use (default is fancy)
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm,ds4}
                        Backend to use (defaults to "transformers")
  --locales LOCALES     Path to locale files (defaults to /workspace/avalan/locale)
  --low-cpu-mem-usage   If specified, loads the model using ~1x model size CPU memory
  --login               Login to main hub (huggingface)
  --no-repl             Don't echo input coming from stdin
  --quiet, -q           If specified, no welcome screen and only model output is displayed in model run (sets --disable-
                        loading-progress-bar, --skip-hub-access-check, --skip-special-tokens automatically)
  --tty TTY             TTY stream to use for interactive prompts
  --record              If specified, the current console output will be regularly saved to SVG files.
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --version             Display this program's version, and exit
  --weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```
