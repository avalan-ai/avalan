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

![Running the CLI in spanish](https://avalan.ai/images/spanish_translation.png)

You'll need your Huggingface access token exported as `HF_TOKEN`.

> [!TIP]
> If you are on an Apple silicon chip, run the
> [configure_mlx.sh](https://github.com/avalan-ai/avalan/blob/main/scripts/configure_mlx.sh)
> script, created by [@AlexCheema](https://github.com/AlexCheema), which
> empirically reduces the time to first token and the tokens per second ratio.

# avalan

```
usage: avalan [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
              [--device DEVICE]
              [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
              [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE]
              [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}] [--locales LOCALES]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                    [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE]
                    [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}] [--locales LOCALES]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                            [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                                   [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                                   [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                                   [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
                                   [--skip-hub-access-check] [--verbose] [--version]
                                   [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] --function
                                   {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms} --id ID
                                   [--limit LIMIT] --participant PARTICIPANT --session SESSION [--engine-uri ENGINE_URI]
                                   [--name NAME] [--role ROLE] [--task TASK] [--instructions INSTRUCTIONS] [--system SYSTEM]
                                   [--developer DEVELOPER] [--user USER] [--user-template USER_TEMPLATE] [--memory-recent]
                                   [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                                   [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                                   [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS]
                                   [--memory-engine-overlap MEMORY_ENGINE_OVERLAP]
                                   [--memory-engine-window MEMORY_ENGINE_WINDOW] [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                                   [--run-skip-special-tokens] [--run-disable-cache]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Agent instructions
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
usage: avalan agent run [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                        [--device DEVICE]
                        [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                        [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                        [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--display-events]
                        [--display-pause [DISPLAY_PAUSE]] [--display-probabilities]
                        [--display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM]
                        [--display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM]
                        [--display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]] [--skip-display-reasoning-time]
                        [--display-tokens [DISPLAY_TOKENS]] [--display-tools] [--display-tools-events DISPLAY_TOOLS_EVENTS]
                        [--display-answer-height-expand | --display-answer-height DISPLAY_ANSWER_HEIGHT] [--id ID]
                        [--participant PARTICIPANT] [--conversation] [--watch] [--no-session | --session SESSION]
                        [--skip-load-recent-messages] [--load-recent-messages-limit LOAD_RECENT_MESSAGES_LIMIT] [--stats]
                        [--sync] [--tools-confirm] [--tool-format {json,react,bracket,openai,harmony}]
                        [--reasoning-tag {think,channel}] [--engine-uri ENGINE_URI] [--name NAME] [--role ROLE] [--task TASK]
                        [--instructions INSTRUCTIONS] [--system SYSTEM] [--developer DEVELOPER] [--user USER]
                        [--user-template USER_TEMPLATE] [--memory-recent] [--no-memory-recent]
                        [--memory-permanent-message MEMORY_PERMANENT_MESSAGE] [--memory-permanent MEMORY_PERMANENT]
                        [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                        [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS] [--memory-engine-overlap MEMORY_ENGINE_OVERLAP]
                        [--memory-engine-window MEMORY_ENGINE_WINDOW] [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                        [--run-skip-special-tokens] [--run-disable-cache]
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
                        

Run an AI agent

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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
  --display-events      If --display-events is specified and there's an orchestrator / agent involved, show the events panel.
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
  --display-tokens [DISPLAY_TOKENS]
                        How many tokens with full information to display at a time
  --display-tools       If --display-events is specified and there's an orchestrator / agent involved, show the events panel.
  --display-tools-events DISPLAY_TOOLS_EVENTS
                        How many tool events to show on tool call panel
  --display-answer-height-expand
                        Expand answer section to full height
  --display-answer-height DISPLAY_ANSWER_HEIGHT
                        Height of the answer section (defaults to 12)
  --id ID               Use given ID as the agent ID
  --participant PARTICIPANT
                        If specified, this is the participant ID interacting with the agent
  --conversation        Activate conversation mode with the agent
  --watch               Reload agent when the specification file changes (only with --conversation)
  --no-session          If specified, don't use sessions in persistent message memory
  --session SESSION     Continue the conversation on the given session
  --skip-load-recent-messages
                        If specified, skips loading recent messages
  --load-recent-messages-limit LOAD_RECENT_MESSAGES_LIMIT
                        If specified, load up to these many recent messages
  --stats               Show token generation statistics for agent output
  --sync                Don't use an async generator (token streaming)
  --tools-confirm       Confirm tool calls before execution
  --tool-format {json,react,bracket,openai,harmony}
                        Tool format
  --reasoning-tag {think,channel}
                        Reasoning tag style

inline agent settings:
  --engine-uri ENGINE_URI
                        Agent engine URI
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Agent instructions
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

### avalan agent serve

```
usage: avalan agent serve [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                          [--device DEVICE]
                          [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                          [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                          [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                          [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
                          [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check] [--verbose]
                          [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--id ID]
                          [--participant PARTICIPANT] [--host HOST] [--port PORT] [--mcp-prefix MCP_PREFIX]
                          [--mcp-name MCP_NAME] [--mcp-description MCP_DESCRIPTION] [--openai-prefix OPENAI_PREFIX]
                          [--a2a-prefix A2A_PREFIX] [--a2a-name A2A_NAME] [--a2a-description A2A_DESCRIPTION]
                          [--protocol PROTOCOL] [--reload] [--cors-origin CORS_ORIGIN] [--cors-origin-regex CORS_ORIGIN_REGEX]
                          [--cors-method CORS_METHOD] [--cors-header CORS_HEADER] [--cors-credentials]
                          [--engine-uri ENGINE_URI] [--name NAME] [--role ROLE] [--task TASK] [--instructions INSTRUCTIONS]
                          [--system SYSTEM] [--developer DEVELOPER] [--user USER] [--user-template USER_TEMPLATE]
                          [--memory-recent] [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                          [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                          [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS]
                          [--memory-engine-overlap MEMORY_ENGINE_OVERLAP] [--memory-engine-window MEMORY_ENGINE_WINDOW]
                          [--run-max-new-tokens RUN_MAX_NEW_TOKENS] [--run-skip-special-tokens] [--run-disable-cache]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Agent instructions
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
                          [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                          [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
                          [--quiet] [--tty TTY] [--record] [--revision REVISION] [--skip-hub-access-check] [--verbose]
                          [--version] [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--id ID]
                          [--participant PARTICIPANT] [--host HOST] [--port PORT] [--mcp-prefix MCP_PREFIX]
                          [--mcp-name MCP_NAME] [--mcp-description MCP_DESCRIPTION] [--openai-prefix OPENAI_PREFIX]
                          [--a2a-prefix A2A_PREFIX] [--a2a-name A2A_NAME] [--a2a-description A2A_DESCRIPTION]
                          [--protocol PROTOCOL] [--reload] [--cors-origin CORS_ORIGIN] [--cors-origin-regex CORS_ORIGIN_REGEX]
                          [--cors-method CORS_METHOD] [--cors-header CORS_HEADER] [--cors-credentials]
                          [--engine-uri ENGINE_URI] [--name NAME] [--role ROLE] [--task TASK] [--instructions INSTRUCTIONS]
                          [--system SYSTEM] [--developer DEVELOPER] [--user USER] [--user-template USER_TEMPLATE]
                          [--memory-recent] [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                          [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                          [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS]
                          [--memory-engine-overlap MEMORY_ENGINE_OVERLAP] [--memory-engine-window MEMORY_ENGINE_WINDOW]
                          [--run-max-new-tokens RUN_MAX_NEW_TOKENS] [--run-skip-special-tokens] [--run-disable-cache]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Agent instructions
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
                         [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
                         [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                         [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                         [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--engine-uri ENGINE_URI]
                         [--name NAME] [--role ROLE] [--task TASK] [--instructions INSTRUCTIONS] [--system SYSTEM]
                         [--developer DEVELOPER] [--user USER] [--user-template USER_TEMPLATE] [--memory-recent]
                         [--no-memory-recent] [--memory-permanent-message MEMORY_PERMANENT_MESSAGE]
                         [--memory-permanent MEMORY_PERMANENT] [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                         [--memory-engine-max-tokens MEMORY_ENGINE_MAX_TOKENS] [--memory-engine-overlap MEMORY_ENGINE_OVERLAP]
                         [--memory-engine-window MEMORY_ENGINE_WINDOW] [--run-max-new-tokens RUN_MAX_NEW_TOKENS]
                         [--run-skip-special-tokens] [--run-disable-cache]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Agent instructions
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
                    [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE]
                    [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}] [--locales LOCALES]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                           [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                           [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                             [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                             [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                         [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                     [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                         [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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

```
usage: avalan flow [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                   [--device DEVICE]
                   [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                   [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE]
                   [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}] [--locales LOCALES]
                   [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record] [--revision REVISION]
                   [--skip-hub-access-check] [--verbose] [--version]
                   [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                   {run} ...

Manage AI flows

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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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

### avalan flow run

```
usage: avalan flow run [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                       [--device DEVICE]
                       [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                       [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                       [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
                       [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                       [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                       [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}]
                       flow

Run a given flow

positional arguments:
  flow                  Flow to run

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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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

## avalan memory

```
usage: avalan memory [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                     [--device DEVICE]
                     [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                     [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                     [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                                [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                                [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                            [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                                    [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                                    [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                            [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                            [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                            [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                        [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl] [--quiet] [--tty TTY] [--record]
                        [--revision REVISION] [--skip-hub-access-check] [--verbose] [--version]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,fp16,fp32,i8,i16,i32,i64,ui8}] [--base-url BASE_URL]
                        [--load] [--special-token SPECIAL_TOKEN] [--token TOKEN] [--tokenizer TOKENIZER] [--display-events]
                        [--display-pause [DISPLAY_PAUSE]] [--display-probabilities]
                        [--display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM]
                        [--display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM]
                        [--display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]] [--skip-display-reasoning-time]
                        [--display-tokens [DISPLAY_TOKENS]] [--display-tools] [--display-tools-events DISPLAY_TOOLS_EVENTS]
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
                        [--developer DEVELOPER] [--text-context TEXT_CONTEXT] [--text-labeled-only]
                        [--text-max-length TEXT_MAX_LENGTH] [--text-num-beams TEXT_NUM_BEAMS] [--text-disable-cache]
                        [--text-cache-strategy {dynamic,static,offloaded_static,sliding_window,hybrid,mamba,quantized}]
                        [--text-from-lang TEXT_FROM_LANG] [--text-to-lang TEXT_TO_LANG] [--start-thinking]
                        [--chat-disable-thinking] [--no-reasoning] [--reasoning-tag {think,channel}]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
  --display-events      If --display-events is specified and there's an orchestrator / agent involved, show the events panel.
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
  --display-tokens [DISPLAY_TOKENS]
                        How many tokens with full information to display at a time
  --display-tools       If --display-events is specified and there's an orchestrator / agent involved, show the events panel.
  --display-tools-events DISPLAY_TOOLS_EVENTS
                        How many tool events to show on tool call panel
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
```

### avalan model search

```
usage: avalan model search [-h] [--cache-dir CACHE_DIR] [--subfolder SUBFOLDER] [--tokenizer-subfolder TOKENIZER_SUBFOLDER]
                           [--device DEVICE]
                           [--parallel 
{auto,colwise,rowwise,colwise_rep,rowwise_rep,local_colwise,local_rowwise,local,gather,local_packed_rowwise,sequence_parallel,re
plicate}]
                           [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                           [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                           [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login] [--no-repl]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                              [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}]
                              [--backend {transformers,mlx,vllm}] [--locales LOCALES] [--low-cpu-mem-usage] [--login]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                        [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                    [--parallel-count PARALLEL_COUNT] [--disable-loading-progress-bar] [--hf-token HF_TOKEN] [--locale LOCALE]
                    [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}] [--locales LOCALES]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
                        [--locale LOCALE] [--loader-class {auto,gemma3,gpt-oss,mistral3}] [--backend {transformers,mlx,vllm}]
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
  --loader-class {auto,gemma3,gpt-oss,mistral3}
                        Loader class to use (defaults to "auto")
  --backend {transformers,mlx,vllm}
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
