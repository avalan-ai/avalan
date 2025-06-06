# CLI

The CLI offers the following commands, some of them with multiple subcommands:

* [agent](#agent): Run and manage AI agents.
  * [agent run](#agent-run)
  * [agent serve](#agent-serve)
  * [agent message search](#agent-message-search)
  * [agent init](#agent-init)
  * [Usage](#usage)
    * [avalan agent](#avalan-agent)
    * [avalan agent message](#avalan-agent-message)
    * [avalan agent message search](#avalan-agent-message-search)
    * [avalan agent run](#avalan-agent-run)
    * [avalan agent serve](#avalan-agent-serve)
    * [avalan agent init](#avalan-agent-init)
* [cache](#cache): Manage the local cache for model data, and download models.
  * [cache delete](#cache-delete)
  * [cache download](#cache-download)
  * [cache list](#cache-list)
  * [Usage](#usage-1)
    * [avalan cache](#avalan-cache)
    * [avalan cache delete](#avalan-cache-delete)
    * [avalan cache download](#avalan-cache-download)
    * [avalan cache list](#avalan-cache-list)
* [deploy](#deploy)
  * [deploy run](#deploy-run)
* [flow](#flow): Execute flows describing multiple processing steps.
  * [flow run](#flow-run)
* [memory](#memory): Generate embeddings, search them and index documents.
  * [memory embeddings](#memory-embeddings)
  * [memory search](#memory-search)
  * [memory document index](#memory-document-index)
  * [Usage](#usage-2)
    * [avalan memory](#avalan-memory)
    * [avalan memory embeddings](#avalan-memory-embeddings)
    * [avalan memory search](#avalan-memory-search)
    * [avalan memory document](#avalan-memory-document)
    * [avalan memory document index](#avalan-memory-document-index)
* [model](#model): Search for models, install and manage them, show their information, and run them.
  * [model display](#model-display)
  * [model install](#model-install)
  * [model run](#model-run)
    * [Quiet mode](#quiet-mode)
    * [Attention implementation](#attention-implementation)
    * [Stopping patterns and token limitation](#stopping-patterns-and-token-limitation)
    * [Reasoning support](#reasoning-support)
    * [Displaying generation details](#displaying-generation-details)
      * [Showing generation performance](#showing-generation-performance)
      * [Probability distributions](#probability-distributions)
  * [model search](#model-search)
  * [model uninstall](#model-uninstall)
  * [Usage](#usage-3)
    * [avalan model](#avalan-model)
    * [avalan model display](#avalan-model-display)
    * [avalan model install](#avalan-model-install)
    * [avalan model run](#avalan-model-run)
    * [avalan model search](#avalan-model-search)
    * [avalan model uninstall](#avalan-model-uninstall)
* [tokenizer](#tokenizer): Manage tokenizers and save them to filesystem.
  * [Adding tokens and special tokens](#adding-tokens-and-special-tokens)
  * [Saving and loading modified tokenizers](#saving-and-loading-modified-tokenizers)
  * [Usage](#usage-4)
    * [avalan tokenizer](#avalan-tokenizer)
* [train](#train)
  * [train run](#train-run)

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

You'll need your Huggingface access token exported as `HF_ACCESS_TOKEN`.

> [!TIP]
> If you are on an Apple silicon chip, run the
> [configure_mlx.sh](https://github.com/avalan-ai/avalan/blob/main/scripts/configure_mlx.sh)
> script, created by [@AlexCheema](https://github.com/AlexCheema), which
> empirically reduces the time to first token and the tokens per second ratio.

## agent

### agent run

Run an AI agent described in a TOML file:

```bash
avalan agent run docs/examples/agent_gettext_translator.toml --quiet
```

### agent serve

Run an agent as an HTTP server:

```bash
avalan agent serve docs/examples/agent_gettext_translator.toml --port 8000
```

### agent message search

Search an agent's message memory:

```bash
avalan agent message search docs/examples/agent_gettext_translator.toml \
    --id AGENT_ID --session SESSION_ID --participant USER_ID \
    --function l2_distance --limit 5
```

### agent init

Generate a TOML template for a new agent. Missing values will be
requested interactively:
```bash
avalan agent init --name "Leo Messi" --engine-uri microsoft/Phi-4-mini-instruct
```

### Usage

#### avalan agent

```text
usage: avalan agent [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                    [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                    [--locale LOCALE] [--loader-class {auto,gemma3,mistral3}]
                    [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                    [--no-repl] [--quiet] [--revision REVISION]
                    [--skip-hub-access-check] [--verbose]
                    [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                    {message,run,serve,init} ...

Manage AI agents

positional arguments:
  {message,run,serve,init}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

#### avalan agent message

```text
usage: avalan agent message [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                            [--disable-loading-progress-bar]
                            [--hf-token HF_TOKEN] [--locale LOCALE]
                            [--loader-class {auto,gemma3,mistral3}]
                            [--locales LOCALES] [--low-cpu-mem-usage]
                            [--login] [--no-repl] [--quiet]
                            [--revision REVISION] [--skip-hub-access-check]
                            [--verbose]
                            [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                            {search} ...

Manage AI agent messages

positional arguments:
  {search}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

#### avalan agent message search

```text
usage: avalan agent message search [-h] [--cache-dir CACHE_DIR]
                                   [--device DEVICE]
                                   [--disable-loading-progress-bar]
                                   [--hf-token HF_TOKEN] [--locale LOCALE]
                                   [--loader-class {auto,gemma3,mistral3}]
                                   [--locales LOCALES] [--low-cpu-mem-usage]
                                   [--login] [--no-repl] [--quiet]
                                   [--revision REVISION]
                                   [--skip-hub-access-check] [--verbose]
                                   [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                                   --function {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms}
                                   --id ID [--limit LIMIT] --participant PARTICIPANT --session SESSION
                                   specifications_file

Search within an agent's message memory

positional arguments:
  specifications_file   File that holds the agent specifications

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --function {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms}
                        Vector function to use for searching
  --id ID
  --limit LIMIT         If specified, load up to these many recent messages
  --participant PARTICIPANT
                        Search messages with given participant
  --session SESSION     Search within the given session
```

#### avalan agent run

```text
usage: avalan agent run [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                        [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE]
                        [--loader-class {auto,gemma3,mistral3}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                        [--no-repl] [--quiet] [--revision REVISION]
                        [--skip-hub-access-check] [--verbose]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                        [--display-pause [DISPLAY_PAUSE]]
                        [--display-probabilities]
                        [--display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM]
                        [--display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM]
                        [--display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]]
                        [--display-tokens [DISPLAY_TOKENS]] [--conversation]
                        [--id ID] [--no-session | --session SESSION]
                        [--skip-load-recent-messages]
                        [--load-recent-messages-limit LOAD_RECENT_MESSAGES_LIMIT]
                        [--participant PARTICIPANT] [--stats] [--sync]
                        [--tty TTY]
                        specifications_file

Run an AI agent

positional arguments:
  specifications_file   File that holds the agent specifications

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
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
  --display-tokens [DISPLAY_TOKENS]
                        How many tokens with full information to display at a
                        time
  --conversation        Activate conversation mode with the agent
  --id ID               Use given ID as the agent ID
  --no-session          If specified, don't use sessions in persistent message
                        memory
  --session SESSION     Continue the conversation on the given session
  --skip-load-recent-messages
                        If specified, skips loading recent messages
  --load-recent-messages-limit LOAD_RECENT_MESSAGES_LIMIT
                        If specified, load up to these many recent messages
  --participant PARTICIPANT
                        If specified, this is the participant ID interacting
                        with the agent
  --stats               Show token generation statistics for agent output
  --sync                Don't use an async generator (token streaming)
  --tty TTY             TTY stream (only applicable if combining
                        --conversation with input piping)
```

#### avalan agent serve

```text
usage: avalan agent serve [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                          [--disable-loading-progress-bar]
                          [--hf-token HF_TOKEN] [--locale LOCALE]
                          [--loader-class {auto,gemma3,mistral3}]
                          [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                          [--no-repl] [--quiet] [--revision REVISION]
                          [--skip-hub-access-check] [--verbose]
                          [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                          [--host HOST] [--port PORT]
                          [--prefix-mcp PREFIX_MCP]
                          [--prefix-openai PREFIX_OPENAI] [--reload]
                          specifications_file

Serve an AI agent as an API endpoint

positional arguments:
  specifications_file   File that holds the agent specifications

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --host HOST           Host (defaults to 127.0.0.1)
  --port PORT           Port (defaults to 9001, HAL 9000+1)
  --prefix-mcp PREFIX_MCP
                        URL prefix for MCP endpoints (defaults to /mcp)
  --prefix-openai PREFIX_OPENAI
                        URL prefix fir OpenAI endpoints (defaults to /v1)
  --reload              Hot reload on code changes
```

#### avalan agent init

```text
usage: avalan agent init [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                         [--disable-loading-progress-bar]
                         [--hf-token HF_TOKEN] [--locale LOCALE]
                         [--loader-class {auto,gemma3,mistral3}]
                         [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                         [--no-repl] [--quiet] [--revision REVISION]
                         [--skip-hub-access-check] [--verbose]
                         [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                         [--name NAME] [--role ROLE] [--task TASK]
                         [--instructions INSTRUCTIONS] [--memory-recent]
                         [--no-memory-recent]
                         [--memory-permanent MEMORY_PERMANENT]
                         [--memory-engine-model-id MEMORY_ENGINE_MODEL_ID]
                         [--engine-uri ENGINE_URI] [--use-cache] [--no-cache]
                         [--max-new-tokens MAX_NEW_TOKENS]
                         [--skip-special-tokens]

Create an agent definition

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --name NAME           Agent name
  --role ROLE           Agent role
  --task TASK           Agent task
  --instructions INSTRUCTIONS
                        Agent instructions
  --memory-recent       Enable recent message memory
  --no-memory-recent
  --memory-permanent MEMORY_PERMANENT
                        Permanent memory DSN
  --memory-engine-model-id MEMORY_ENGINE_MODEL_ID
                        Sentence transformer model for memory
  --engine-uri ENGINE_URI
                        Agent engine URI
  --use-cache           Cache model locally
  --no-cache
  --max-new-tokens MAX_NEW_TOKENS
                        Max new tokens
  --skip-special-tokens
                        Skip special tokens
```

## cache

To run models locally you'll need to cache their data on a filesystem. A
default location of `$HOME/.cache/huggingface/hub` will be assumed, unless
the `--cache-dir` global option is utilized.

### cache delete

You can delete all cached data for a model:

```bash
avalan cache delete --model 'qingy2024/UwU-7B-Instruct'
```

![Deleting a model](https://avalan.ai/images/cache_delete_model.png)

Or you can specify which model revisions to delete:

```bash
avalan cache delete --model 'google/owlvit-base-patch16' \
                    --revision '10e842' \
                    --revision '4b420d'
```

![Deleting all revisions in a model](https://avalan.ai/images/cache_delete_revisions.png)

### cache download

You can pre-emptively download all the needed files to run a really small
model to your local cache:

```bash
avalan cache download --model 'hf-internal-testing/tiny-random-bert'
```

![Downloading a tiny model to cache](https://avalan.ai/images/cache_download.gif)

### cache list

You can inspect the state of your cached models with:

```bash
avalan cache list
```

![Inspecting cached models](https://avalan.ai/images/cache_list.png)

The cache list is sorted by size on disk, starting with the largest models. In
our case, we see our cached models are occupying a total of 436.4 GB.

Let's inspect the cache contents of the `Qwen/Qwen2.5-7B-Instruct` model we
have installed, which has two revisions, using the option `--model` (you can
specify multiple models by adding more `--model` options):

```bash
avalan cache list --model 'Qwen/Qwen2.5-7B-Instruct'
```

![Showing cached model details](https://avalan.ai/images/cache_list_details.png)

> [!NOTE]
> When the same file appears in multiple revisions of a model, that does
> not mean the file is stored multiple times. If a file hasn't changed
> across revisions, a symlink is used, to only keep one version of the file.

### Usage

#### avalan cache

```text
usage: avalan cache [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                    [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                    [--locale LOCALE] [--loader-class {auto,gemma3,mistral3}]
                    [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                    [--no-repl] [--quiet] [--revision REVISION]
                    [--skip-hub-access-check] [--verbose]
                    [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                    {delete,download,list} ...

Manage models cache

positional arguments:
  {delete,download,list}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

#### avalan cache delete

```text
usage: avalan cache delete [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                           [--disable-loading-progress-bar]
                           [--hf-token HF_TOKEN] [--locale LOCALE]
                           [--loader-class {auto,gemma3,mistral3}]
                           [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                           [--no-repl] [--quiet] [--revision REVISION]
                           [--skip-hub-access-check] [--verbose]
                           [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                           [--delete] --model MODEL
                           [--delete-revision DELETE_REVISION]

Delete cached model data

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --delete              Actually delete. If not provided, a dry run is
                        performed and data that would be deleted is shown, yet
                        not deleted
  --model MODEL, -m MODEL
                        Model to delete
  --delete-revision DELETE_REVISION
                        Revision to delete
```

#### avalan cache download

```text
usage: avalan cache download [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                             [--disable-loading-progress-bar]
                             [--hf-token HF_TOKEN] [--locale LOCALE]
                             [--loader-class {auto,gemma3,mistral3}]
                             [--locales LOCALES] [--low-cpu-mem-usage]
                             [--login] [--no-repl] [--quiet]
                             [--revision REVISION] [--skip-hub-access-check]
                             [--verbose]
                             [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                             --model MODEL

Download model data to cache

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --model MODEL, -m MODEL
                        Model to load
```

#### avalan cache list

```text
usage: avalan cache list [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                         [--disable-loading-progress-bar]
                         [--hf-token HF_TOKEN] [--locale LOCALE]
                         [--loader-class {auto,gemma3,mistral3}]
                         [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                         [--no-repl] [--quiet] [--revision REVISION]
                         [--skip-hub-access-check] [--verbose]
                         [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                         [--model MODEL] [--summary]

List cache contents

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --model MODEL         Models to show content details on
  --summary             If specified, when showing one or more models show
                        only summary
```

## deploy

### deploy run

Run a deployment definition:

```bash
avalan deploy run docs/examples/my_deployment.toml
```

## flow

### flow run

Run a flow definition. Provide the path to the flow description file:

```bash
avalan flow run docs/examples/my_flow.toml
```

## memory

### memory embeddings

Generate embeddings from text. You can compare the generated embeddings or
search across them:

```bash
avalan memory embeddings --model microsoft/Phi-4-mini-instruct \
                         --compare 'hello there' \
                         --search 'hola mundo' \
                         --search-k 3
```

Use `--partition` to split long inputs into windows when indexing and
`--display-partitions` to show a summary of the generated partitions.

### memory search

Look for memories stored in PostgreSQL using vector search:

```bash
avalan memory search --dsn postgresql://user:pass@localhost/db \
                    --participant 123e4567-e89b-12d3-a456-426614174000 \
                    --namespace chat \
                    --function l2_distance \
                    --limit 5
```

### memory document index

Add a document to the memory index so its contents become searchable:

```bash
avalan memory document index README.md \
                           --dsn postgresql://user:pass@localhost/db \
                           --participant 123e4567-e89b-12d3-a456-426614174000 \
                           --namespace docs \
                           --partitioner code \
                           --language python \
                           --encoding utf-8
```
Use `--identifier` to override the default identifier (the source path or URL).

## Usage

#### avalan memory

```text
usage: avalan memory [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                     [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                     [--locale LOCALE] [--loader-class {auto,gemma3,mistral3}]
                     [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                     [--no-repl] [--quiet] [--revision REVISION]
                     [--skip-hub-access-check] [--verbose]
                     [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                     {embeddings,search,document} ...

Manage memory

positional arguments:
  {embeddings,search,document}

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
```

#### avalan memory embeddings

```text
usage: avalan memory embeddings [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                                [--disable-loading-progress-bar]
                                [--hf-token HF_TOKEN] [--locale LOCALE]
                                [--loader-class {auto,gemma3,mistral3}]
                                [--locales LOCALES] [--low-cpu-mem-usage]
                                [--login] [--no-repl] [--quiet]
                                [--revision REVISION]
                                [--skip-hub-access-check] [--verbose]
                                [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                                [--load] [--special-token SPECIAL_TOKEN]
                                [--token TOKEN] [--tokenizer TOKENIZER]
                                [--no-display-partitions | --display-partitions DISPLAY_PARTITIONS]
                                [--partition]
                                [--partition-max-tokens PARTITION_MAX_TOKENS]
                                [--partition-overlap PARTITION_OVERLAP]
                                [--partition-window PARTITION_WINDOW]
                                [--compare COMPARE] [--search SEARCH]
                                [--search-k SEARCH_K]
                                [--sort {cosine,dot,l1,l2,pearson}]
                                model

Obtain and manipulate embeddings

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is
                        loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default,
                        only if model is loaded
  --no-display-partitions
                        If specified, don't display memory partitions
  --display-partitions DISPLAY_PARTITIONS
                        Display up to this many partitions, if more summarize
  --partition           If specified, partition string
  --partition-max-tokens PARTITION_MAX_TOKENS
                        Maximum number of tokens to allow on each partition
  --partition-overlap PARTITION_OVERLAP
                        How many tokens can potentially overlap in different
                        partitions
  --partition-window PARTITION_WINDOW
                        Number of tokens per window when partitioning
  --compare COMPARE     If specified, compare embeddings with this string
  --search SEARCH       If specified, search across embeddings for this string
  --search-k SEARCH_K   How many nearest neighbors to obtain with search
  --sort {cosine,dot,l1,l2,pearson}
                        Sort comparison results using the given similarity
                        measure
```

#### avalan memory search

```text
usage: avalan memory search [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                            [--disable-loading-progress-bar]
                            [--hf-token HF_TOKEN] [--locale LOCALE]
                            [--loader-class {auto,gemma3,mistral3}]
                            [--locales LOCALES] [--low-cpu-mem-usage]
                            [--login] [--no-repl] [--quiet]
                            [--revision REVISION] [--skip-hub-access-check]
                            [--verbose]
                            [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                            [--load] [--special-token SPECIAL_TOKEN]
                            [--token TOKEN] [--tokenizer TOKENIZER]
                            [--no-display-partitions | --display-partitions DISPLAY_PARTITIONS]
                            [--partition]
                            [--partition-max-tokens PARTITION_MAX_TOKENS]
                            [--partition-overlap PARTITION_OVERLAP]
                            [--partition-window PARTITION_WINDOW] --dsn DSN
                            --participant PARTICIPANT --namespace NAMESPACE
                            --function {cosine_distance,inner_product,l1_distance,l2_distance,vector_dims,vector_norms}
                            [--limit LIMIT]
                            model

Search memories

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is
                        loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default,
                        only if model is loaded
  --no-display-partitions
                        If specified, don't display memory partitions
  --display-partitions DISPLAY_PARTITIONS
                        Display up to this many partitions, if more summarize
  --partition           If specified, partition string
  --partition-max-tokens PARTITION_MAX_TOKENS
                        Maximum number of tokens to allow on each partition
  --partition-overlap PARTITION_OVERLAP
                        How many tokens can potentially overlap in different
                        partitions
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

#### avalan memory document

```text
usage: avalan memory document [-h] {index} ...

Manage memory indexed documents

positional arguments:
  {index}

options:
  -h, --help  show this help message and exit
```

#### avalan memory document index

```text
usage: avalan memory document index [-h] [--cache-dir CACHE_DIR]
                                    [--device DEVICE]
                                    [--disable-loading-progress-bar]
                                    [--hf-token HF_TOKEN] [--locale LOCALE]
                                    [--loader-class {auto,gemma3,mistral3}]
                                    [--locales LOCALES] [--low-cpu-mem-usage]
                                    [--login] [--no-repl] [--quiet]
                                    [--revision REVISION]
                                    [--skip-hub-access-check] [--verbose]
                                    [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                                    [--load] [--special-token SPECIAL_TOKEN]
                                    [--token TOKEN] [--tokenizer TOKENIZER]
                                    [--no-display-partitions | --display-partitions DISPLAY_PARTITIONS]
                                    [--partition]
                                    [--partition-max-tokens PARTITION_MAX_TOKENS]
                                    [--partition-overlap PARTITION_OVERLAP]
                                    [--partition-window PARTITION_WINDOW]
                                    [--partitioner {text,code}]
                                    [--language LANGUAGE]
                                    [--encoding ENCODING]
                                    [--identifier IDENTIFIER] --dsn DSN
                                    --participant PARTICIPANT --namespace
                                    NAMESPACE
                                    model source

Add a document to the memory index

positional arguments:
  model                 Model to use
  source                Source to index (an URL or a file path)

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is
                        loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER
                        Path to tokenizer to use instead of model's default,
                        only if model is loaded
  --no-display-partitions
                        If specified, don't display memory partitions
  --display-partitions DISPLAY_PARTITIONS
                        Display up to this many partitions, if more summarize
  --partition           If specified, partition string
  --partition-max-tokens PARTITION_MAX_TOKENS
                        Maximum number of tokens to allow on each partition
  --partition-overlap PARTITION_OVERLAP
                        How many tokens can potentially overlap in different
                        partitions
  --partition-window PARTITION_WINDOW
                        Number of tokens per window when partitioning
  --partitioner {text,code}
                        Partitioner to use when indexing a file
  --language LANGUAGE   Programming language for the code partitioner
  --encoding ENCODING   File encoding used when reading a local file
  --identifier IDENTIFIER
                        Identifier for the memory entry (defaults to the
                        source)
  --dsn DSN             PostgreSQL DSN for storing the document
  --participant PARTICIPANT
                        Participant ID for the memory entry
  --namespace NAMESPACE
                        Namespace for the memory entry
```

## model

### model display

You can show detailed model information (such as architectures, vocabulary
size, hidden and attention layers, special tokens, etc) if you load the model.
The command also tells you whether the model fits in your current device:

```bash
avalan model display --load deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

![Looking for models that match search criteria](https://avalan.ai/images/running_show_example.png)

### model install

You can install any of the +1.4 million models available:

```bash
avalan model install microsoft/Phi-4-mini-instruct
```

### model run

You can run a model by entering your prompt at the, well, prompt:

```bash
avalan model run meta-llama/Meta-Llama-3-8B-Instruct
```

You can also specify your prompt by piping it, on this case to a gated repo
(which is why we also `--login`):

```bash
echo 'explain LLM distillation in no more than three paragraphs' |
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct --login
```

#### Quiet mode

If you want to prompt a model and get nothing but its response, try `--quiet`
mode. It will only stream generated tokens directly to output, without any
added statistics or styling:

```bash
echo 'Who is Leo Messi?' |
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct --quiet
```

![Quiet mode](https://avalan.ai/images/running_quiet_mode.gif)

#### Attention implementation

When running a model, by default the best available attention implementation
is utilized. If you'd like to change it, use the `--attention` option,
specifying one of the available implementations: `eager`, `flash_attention_2`
(you'll need CUDA and the [flash-attn](https://pypi.org/project/flash-attn/)
package installed), `sdpa`, and `flex_attention` (only for CUDA):

```bash
echo 'hello, who are you? answer in no less than 100 words' |
    avalan model run deepseek-ai/deepseek-llm-7b-chat --attention sdpa
```

#### Stopping patterns and token limitation

There are multiple ways to stop the inference process. You can choose to limit
the amount of tokens generated with `--max-new-tokens`:

```bash
echo 'Who is Leo Messi?' | \
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct --max-new-tokens 10
```

![Limiting number of generated tokens](https://avalan.ai/images/running_generation_max_new_tokens.png)

You can also stop the token generation when one (or one of many) expression
is found with `--stop-on-keyword` (use as many as needed):

```bash
echo 'Who is Leo Messi?' | \
    avalan model run meta-llama/Meta-Llama-3-8B-Instruct
                     --stop_on_keyword 'Argentina' \
                     --stop_on_keyword 'Barcelona' \
                     --stop_on_keyword 'footballer'
```

![Stopping generation when certain keywords are found](https://avalan.ai/images/running_generation_stop_on_keyword.png)

#### Reasoning support

If you run a model with reasoning support, you'll see the model reasoning
preceeding its response:

```bash
echo 'explain LLM distillation in no more than three paragraphs' |
    avalan model run deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

![Reasoning section for models that support it](https://avalan.ai/images/running_local_inference_with_reasoning.png)

#### Displaying generation details

To get details on the tokens generated by the chosen model, use the
`--display-tokens` option, optionally setting it to the number of tokens with
details to display at a time, for example `--display-tokens 20`.
If the option is present but no value provided, a default of `15` tokens will
be used.

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat --display-tokens
```

> [!IMPORTANT]
> When the option `--display-tokens` is used, inference tokens are displayed
> only after the model has finished producing all tokens, unlike the default
> real token streaming behavior when the option is not present.

When displaying generation details, tokens may (hopefully) advance too rapidly to
follow. You can add a delay between tokens with `--display-pause`. If no value
specified, a default of `500` milliseconds will be used. Following, we are
introducing a much lower `25` milliseconds delay between tokens:

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat \
               --display-tokens \
               --display-pause 25
```

##### Showing generation performance

While the CLI is displaying the generated tokens, you will see some statistics
at the bottom right side:

* `token count`: the total number of tokens that have been generated.
* `ttft`: time to first token, the time it took for the model to generate
the first token.
* `ttnt`: time to Nth token, the time it took for the model to generate the
Nth token (defaulting to 256.)
* `t/s`: tokens per second, on average, how many tokens the model generates
in a second.

You can choose another value for `ttnt`. For example, by setting
`---display-time-to-n-token` to `100` we can learn how long it takes the model
to produce the 100th token:

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat \
                     --display-time-to-n-token 100
```

![Displaying time to 100th token](https://avalan.ai/images/running_local_inference_speed.png)

We can see it took `deepseek-llm-7b-chat` a total of `4.61 seconds` until
generating the 100th token.

##### Probability distributions

If you are interested in seeing the token generation progress, including
details such as token alternatives per generation step with different
distributions, do:

```bash
echo 'hello, who are you? answer in no less than 100 words' | \
    avalan model run deepseek-ai/deepseek-llm-7b-chat \
                     --max-new-tokens 300 \
                     --temperature 0.9 \
                     --do-sample \
                     --display-tokens 15 \
                     --display-probabilities \
                     --display-probabilities-maximum 0.8 \
                     --display-probabilities-sample-minimum 0.1 \
                     --display-pause
```

![Example use of the CLI showing token distributions](https://avalan.ai/images/running_token_distribution_example.gif)

### model search

Let's search for up to two models matching a query (`deepseek r1`) and a
filter (`transformers`), ensuring we are previously logged in to the hub:

```bash
avalan model search 'deepseek r1' \
					--login \
                    --filter 'transformers' \
                    --limit 2
```

![Looking for models that match search criteria](https://avalan.ai/images/running_search_example.png)

### model uninstall

You can uninstall an install model:

```bash
avalan model uninstall microsoft/Phi-4-mini-instruct
```

### Usage

#### avalan model

```text
usage: avalan model [-h] {display,install,run,search,uninstall} ...

Manage a model, showing details, loading or downloading it

positional arguments:
  {display,install,run,search,uninstall}

options:
  -h, --help            show this help message and exit
```

#### avalan model display

```text
usage: avalan model display [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                            [--disable-loading-progress-bar]
                            [--hf-token HF_TOKEN] [--locale LOCALE]
                            [--loader-class {auto,gemma3,mistral3}]
                            [--locales LOCALES] [--low-cpu-mem-usage]
                            [--login] [--no-repl] [--quiet]
                            [--revision REVISION] [--skip-hub-access-check]
                            [--verbose]
                            [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                            [--load] [--special-token SPECIAL_TOKEN]
                            [--token TOKEN] [--tokenizer TOKENIZER]
                            [--sentence-transformer] [--summary]
                            model

Show information about a model

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is
                        loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER Path to tokenizer to use instead of model's default,
                        only if model is loaded
  --sentence-transformer
                        Load the model as a SentenceTransformer model
  --summary
```

#### avalan model install

```text
usage: avalan model install [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                            [--disable-loading-progress-bar]
                            [--hf-token HF_TOKEN] [--locale LOCALE]
                            [--loader-class {auto,gemma3,mistral3}]
                            [--locales LOCALES] [--low-cpu-mem-usage]
                            [--login] [--no-repl] [--quiet]
                            [--revision REVISION] [--skip-hub-access-check]
                            [--verbose]
                            [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                            [--load] [--special-token SPECIAL_TOKEN]
                            [--token TOKEN] [--tokenizer TOKENIZER]
                            model

Install a model

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is
                        loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER Path to tokenizer to use instead of model's default,
                        only if model is loaded
```

#### avalan model run

```text
usage: avalan model run [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                        [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE]
                        [--loader-class {auto,gemma3,mistral3}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                        [--no-repl] [--quiet] [--revision REVISION]
                        [--skip-hub-access-check] [--verbose]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                        [--load] [--special-token SPECIAL_TOKEN]
                        [--token TOKEN] [--tokenizer TOKENIZER]
                        [--display-pause [DISPLAY_PAUSE]]
                        [--display-probabilities]
                        [--display-probabilities-maximum DISPLAY_PROBABILITIES_MAXIMUM]
                        [--display-probabilities-sample-minimum DISPLAY_PROBABILITIES_SAMPLE_MINIMUM]
                        [--display-time-to-n-token [DISPLAY_TIME_TO_N_TOKEN]]
                        [--display-tokens [DISPLAY_TOKENS]]
                        [--attention {eager,flash_attention_2,flex_attention,sdpa}]
                        [--do-sample] [--enable-gradient-calculation]
                        [--use-cache] [--max-new-tokens MAX_NEW_TOKENS]
                        [--min-p MIN_P]
                        [--repetition-penalty REPETITION_PENALTY]
                        [--skip-special-tokens] [--system SYSTEM]
                        [--start-thinking] [--stop_on_keyword STOP_ON_KEYWORD]
                        [--temperature TEMPERATURE] [--top-k TOP_K]
                        [--top-p TOP_P] [--trust-remote-code]
                        model

Run a model

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is
                        loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER Path to tokenizer to use instead of model's default,
                        only if model is loaded
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
  --display-tokens [DISPLAY_TOKENS]
                        How many tokens with full information to display at a
                        time
  --attention {eager,flash_attention_2,flex_attention,sdpa}
                        Attention implementation to use (defaults to best
                        available)
  --do-sample           Tell if the token generation process should be
                        deterministic or stochastic. When enabled, it's
                        stochastic and uses probability distribution.
  --enable-gradient-calculation
                        Enable gradient calculation.
  --use-cache           Past key values are used to speed up decoding if
                        applicable to model.
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of tokens to generate
  --min-p MIN_P         Minimum token probability, which will be scaled by the
                        probability of the most likely token [0, 1]
  --repetition-penalty REPETITION_PENALTY
                        Exponential penalty on sequences not in the original
                        input. Defaults to 1.0, which means no penalty.
  --skip-special-tokens
                        If specified, skip special tokens when decoding
  --system SYSTEM       Use this as system prompt
  --start-thinking      If specified, assume model response starts with
                        reasoning
  --stop_on_keyword STOP_ON_KEYWORD
                        Stop token generation when this keyword is found
  --temperature TEMPERATURE
                        Temperature [0, 1]
  --top-k TOP_K         Number of highest probability vocabulary tokens to
                        keep for top-k-filtering.
  --top-p TOP_P         If set to < 1, only the smallest set of most probable
                        tokens with probabilities that add up to top_p or
                        higher are kept for generation.
  --trust-remote-code
```

#### avalan model search

```text
usage: avalan model search [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                           [--disable-loading-progress-bar]
                           [--hf-token HF_TOKEN] [--locale LOCALE]
                           [--loader-class {auto,gemma3,mistral3}]
                           [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                           [--no-repl] [--quiet] [--revision REVISION]
                           [--skip-hub-access-check] [--verbose]
                           [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                           [--search SEARCH] [--filter FILTER] [--limit LIMIT]

Search for models

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --search SEARCH       Search for models matching given expression
  --filter FILTER       Filter models on this (e.g: text-classification)
  --limit LIMIT         Maximum number of models to return
```

#### avalan model uninstall

```text
usage: avalan model uninstall [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                              [--disable-loading-progress-bar]
                              [--hf-token HF_TOKEN] [--locale LOCALE]
                              [--loader-class {auto,gemma3,mistral3}]
                              [--locales LOCALES] [--low-cpu-mem-usage]
                              [--login] [--no-repl] [--quiet]
                              [--revision REVISION] [--skip-hub-access-check]
                              [--verbose]
                              [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                              [--load] [--special-token SPECIAL_TOKEN]
                              [--token TOKEN] [--tokenizer TOKENIZER]
                              [--delete]
                              model

Uninstall a model

positional arguments:
  model                 Model to use

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --load                If specified, load model and show more information
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer, only when model is
                        loaded
  --token TOKEN         Token to add to tokenizer, only when model is loaded
  --tokenizer TOKENIZER Path to tokenizer to use instead of model's default,
                        only if model is loaded
  --delete              Actually delete. If not provided, a dry run is
                        performed and data that would be deleted is shown, yet
                        not deleted
```

## tokenizer

If you want to see how a tokenizer deals with text, you can have the CLI
ask for the text to tokenize, or provide it via standard input:

```bash
echo 'Leo Messi is the GOAT' |
    avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat'
```

![Tokenization of text](https://avalan.ai/images/running_tokenization_simple_example.png)

### Adding tokens and special tokens

When viewing token displays, you may have noticed some of the token boxes
are colored differently. Two kinds of token that are always going to be colored
are the added token, and the special token, a small subset of tokens the
tokenizer treates differently.

To see this in action, we'll add a token ourselves: `<avalan_special_token>`.
Let's first see how the tokenizer deails with our token when it has no
knowledge of it:

```bash
echo 'is <avalan_special_token> a special token?' | \
    avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat'
```

We see the tokenizer split it as: `<｜begin▁of▁sentence｜>`, `is`, `<`, `aval`,
`an`, `_`, `special`, `_`, `token`, `>`, `a`, `special`, `token`, `?`,
`<｜end▁of▁sentence｜>`.

Now let's run the same, but also add our token to the tokenizer with
`--token` (you can add multiple by adding more arguments,
like: `--token my_token_1 --token my_token_2`):

```bash
echo 'is <avalan_special_token> a special token?' | \
    avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat' \
                     --token '<avalan_special_token>'
```

![Tokenization of an added token unknown to the tokenizer](https://avalan.ai/images/running_tokenization_example.png)

This time, the tokenizer splits it as: `<｜begin▁of▁sentence｜>`, `is`,
`<avalan_special_token>`, `a`, `special`, `token`, `?"`,
`<｜end▁of▁sentence｜>`, so our added token is a mere token for the tokenizer
now, versus previously using 8 tokens for it, a whooping 87.5% in savings :]

### Saving and loading modified tokenizers

If you want to persist your tokenizer modifications, use the `--save` option:

```bash
avalan tokenizer --tokenizer 'deepseek-ai/deepseek-llm-7b-chat' \
                 --token '<avalan_special_token>' \
                 --save './my_custom_tokenizer'
```

![Saving a modified tokenizer](https://avalan.ai/images/running_tokenization_saving_example.png)

Load your modified tokenizer, and see it in action:

```bash
echo 'is <avalan_special_token> a special token?' | \
    avalan tokenizer --tokenizer './my_custom_tokenizer'
```

### Usage

#### avalan tokenizer

```text
usage: avalan tokenizer [-h] [--cache-dir CACHE_DIR] [--device DEVICE]
                        [--disable-loading-progress-bar] [--hf-token HF_TOKEN]
                        [--locale LOCALE]
                        [--loader-class {auto,gemma3,mistral3}]
                        [--locales LOCALES] [--low-cpu-mem-usage] [--login]
                        [--no-repl] [--quiet] [--revision REVISION]
                        [--skip-hub-access-check] [--verbose]
                        [--weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}]
                        --tokenizer TOKENIZER [--save SAVE]
                        [--special-token SPECIAL_TOKEN] [--token TOKEN]

Manage tokenizers, loading, modifying and saving them

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Path to huggingface cache hub (defaults to
                        /root/.cache/huggingface/hub, can also be specified
                        with $HF_HUB_CACHE)
  --device DEVICE       Device to use (cpu, cuda, mps). Defaults to cpu
  --disable-loading-progress-bar
                        If specified, the shard loading progress bar will not
                        be shown
  --hf-token HF_TOKEN   Your Huggingface access token
  --locale LOCALE       Language to use (defaults to en_US)
  --loader-class {auto,gemma3,mistral3}
                        Loader class to use (defaults to "auto")
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
  --revision REVISION   Model revision to use
  --skip-hub-access-check
                        If specified, skip hub model access check
  --verbose, -v         Set verbosity
  --weight-type {auto,bool,bf16,f16,f32,f64,i8,i16,i32,i64,ui8}
                        Weight type to use (defaults to best available)
  --tokenizer TOKENIZER, -t TOKENIZER
                        Tokenizer to load
  --save SAVE           Save tokenizer (useful if modified via --special-token
                        or --token) to given path, only if model is loaded
  --special-token SPECIAL_TOKEN
                        Special token to add to tokenizer
  --token TOKEN         Token to add to tokenizer
```

## train

### train run

Run a training definition:

```bash
avalan train run docs/examples/my_training.toml
```

