# CLI

The CLI offers the following commands, some of them with multiple subcommands:

* [agent](#agent): Run and manage AI agents.
* [cache](#cache): Manage the local cache for model data, and download models.
* [flow](#flow): Execute flows describing multiple processing steps.
* [memory](#memory): Generate embeddings, search them and index documents.
* [model](#model): Search for models, install and manage them, show
their information, and run them.
* [tokenizer](#tokenizer): Manage tokenizers and save them to filesystem.
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

## model

### model display

You can show detailed model information (such as architectures, vocabulary
size, hidden and attention layers, special tokens, etc) if you load the model:

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


## train

### train run

Run a training definition:

```bash
avalan train run docs/examples/my_training.toml
```

