<h1 align="center">avalan</h1>
<h3 align="center">The multi-backend, multi-modal framework for effortless AI agent development, orchestration, and deployment</h3>

<p align="center">
  <img src="https://github.com/avalan-ai/avalan/actions/workflows/test.yml/badge.svg" alt="Tests" />
  <a href="https://coveralls.io/github/avalan-ai/avalan"><img src="https://coveralls.io/repos/github/avalan-ai/avalan/badge.svg" alt="Code test coverage" /></a>
  <img src="https://img.shields.io/github/last-commit/avalan-ai/avalan.svg" alt="Last commit" />
  <img src="https://img.shields.io/github/v/release/avalan-ai/avalan?label=Release" alt="Release" />
  <img src="https://img.shields.io/pypi/l/avalan.svg" alt="License" />
  <a href="https://discord.gg/8Eh9TNvk"><img src="https://img.shields.io/badge/discord-community-blue" alt="Discord Community" /></a>
</p>

avalan empowers developers and enterprises to easily build, orchestrate, and deploy intelligent AI agents—locally or in the cloud—across millions of models via a unified SDK and CLI, featuring multi-backend support ([transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [mlx-lm](https://github.com/ml-explore/mlx-lm)), multi-modal integration (NLP, vision, audio), and native adapters for platforms like OpenRouter, Ollama, OpenAI, DeepSeek, and Gemini. Enhanced by sophisticated memory management, advanced reasoning (including ReACT tooling and adaptive planning), and intuitive pipelines with branching, filtering, and recursive workflows, avalan ensures agents continuously learn and adapt. Comprehensive observability through real-time metrics, event tracing, and statistical dashboards provides deep insights and robust governance, making avalan ideal for everything from individual experimentation to enterprise-scale AI deployments.

# Quick Look

Check out [the CLI documentation](docs/CLI.md) to see what it can do, but if you want to jump right in, you can run any locally installed model and tweak sampling settings like `--temperature`, `--top-p`, and `--top-k`. In this example, we prompt the model as "Aurora" and limit the response to 100 new tokens:

```bash
echo 'Who are you, and who is Leo Messi?' \
  | avalan model run "meta-llama/Meta-Llama-3-8B-Instruct" \
      --system "You are Aurora, a helpful assistant" \
      --max-new-tokens 100 \
      --temperature .1 \
      --top-p .9 \
      --top-k 20
```

Just as easily as you can run local models, you can use vendors. Simply swap in a vendor-backed [engine URI](docs/ai_uri.md) to run on an external API. For instance, to hit OpenAI's GPT-4o endpoint with the same sampling parameters:

```bash
echo 'Who are you, and who is Leo Messi?' \
  | avalan model run "ai://$OPENAI_API_KEY@openai/gpt-4o" \
      --system "You are Aurora, a helpful assistant" \
      --max-new-tokens 100 \
      --temperature .1 \
      --top-p .9 \
      --top-k 20
```

## Tools

Avalan makes it trivial to spin up a chat-based agent that can invoke external tools, even while streaming. Below is an example using a locally installed 8B-parameter LLM, enabling recent memory, and loading a calculator tool. The agent starts with a math question and then keeps the conversation open for follow-up questions:

```bash
echo "What is (4 + 6) and then that result times 5, divided by 2?" \
  | avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "math.calculator" \
      --memory-recent \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools \
      --conversation
```

Check the GPU hard at work towards the bottom:

![Example use of an ephemeral tool agent with memory](https://avalan.ai/images/cli_agent_tool.gif)

Here's a tool using agent that uses the `code.run` tool to execute Python
code built by the agent, and inform the result:

```bash
echo "Create a python function to uppercase a string, split it spaces, and then return the words joined by a dash, and execute the function with the string 'Leo Messi is the greatest footballer of all times'" \
  | avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "code.run" \
      --memory-recent \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools
```

With tooling, agents get real-time knowledge. Here's an 8B model looking for avalan's latest release, using a browser to do so:

```bash
echo "What's avalan's latest release in pypi?" | \
    avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "browser.open" \
      --memory-recent \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools
```

You can point an agent to specific locations for gaining knowledge:

```bash
echo "Tell me what avalan does based on the web page https://raw.githubusercontent.com/avalan-ai/avalan/refs/heads/main/README.md" | \
    avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "browser.open" \
      --memory-recent \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools
```

## Memories

Let's initiate a chat session where we tell the agent our name. Notice the `--memory-permanent-message` option to specify where messages are stored, the `--id` option to uniquely identify the agent, and `--participant` option specifying a user ID:

```bash
echo "Hi Tool, my name is Leo. Nice to meet you." \
  | avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --memory-recent \
      --memory-permanent-message "postgresql://root:password@localhost/avalan" \
      --id "f4fd12f4-25ea-4c81-9514-d31fb4c48128" \
      --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats
```

Let's have our agent be able to tap into past messages by enabling persistent memory and the `memory.message.read` tool. It should be able to find that our name is `Leo` based off the message we previously posted:

```bash
echo "Hi Tool, based on our previous conversations, what's my name?" \
  | avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "memory.message.read" \
      --memory-recent \
      --memory-permanent-message "postgresql://root:password@localhost/avalan" \
      --id "f4fd12f4-25ea-4c81-9514-d31fb4c48128" \
      --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats
```

You can store knowledge in knowledge stores that can be then used by agents to solve problems. Let's start by indexing the rules of the "Truco" card game directly from a website into our knowledge store. Notice the `--dsn` parameter to specify the store location, and the `--namespace` parameter to specify our desired knowledge namespace:

```bash
avalan memory document index \
    --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
    --dsn "postgresql://root:password@localhost/avalan" \
    --namespace "games.cards.truco" \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "https://trucogame.com/pages/reglamento-de-truco-argentino"
```

## Serving agents

Serve your agents on an OpenAI API compatible endpoint:

```bash
avalan agent serve docs/examples/agent_tool.toml -vvv
```

Or build the agent from inline settings and serve its OpenAI API endpoints:

```bash
avalan agent serve \
    --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
    --tool "math.calculator" \
    --memory-recent \
    --run-max-new-tokens 1024 \
    --name "Tool" \
    --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
    -vvv
```

You can hit your tool streaming agent OpenAPI API endpoint just like you
would with OpenAI, just change the `--base-url`:

```bash
echo "What is (4 + 6) and then that result times 5, divided by 2?" | \
    avalan model run "ai://openai" --base-url "http://localhost:9001/v1"
```

## Modalities

avalan supports text, audio and video modalities.

### Audio

#### Speech recognition

Run a speech recognition model:

```bash
avalan model run "facebook/wav2vec2-base-960h" \
    --modality audio_speech_recognition \
    --audio-path oprah.wav \
    --audio-sampling-rate 16000
```

To get the transcript for the given audio file:

```text
AND THEN I GREW UP AND HAD THE ESTEEMED HONOUR OF MEETING HER AND WASN'T
THAT A SURPRISE HERE WAS THIS PETITE ALMOST DELICATE LADY WHO WAS THE
PERSONIFICATION OF GRACE AND GOODNESS
```

#### Text to speech

Create an audio speech from your prompt by cloning Oprah's voice, using an
18-second clip of her [eulogy for Rosa Parks](https://www.americanrhetoric.com/speeches/oprahwinfreyonrosaparks.htm):

```bash
echo "[S1] Leo Messi is the greatest football player of all times." | \
    avalan model run "nari-labs/Dia-1.6B-0626" \
            --modality audio_text_to_speech \
            --audio-path example.wav \
            --audio-reference-path docs/examples/oprah.wav \
            --audio-reference-text "[S1] And then I grew up and had the esteemed honor of meeting her. And wasn't that a surprise. Here was this petite, almost delicate lady who was the personification of grace and goodness."
```

## Code

Through the avalan microframework, you can easily integrate real time token
streaming with your own code, as [this example shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/text_generation.py):

```python
from asyncio import run
from avalan.entities import GenerationSettings
from avalan.model.nlp.text import TextGenerationModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with TextGenerationModel("meta-llama/Meta-Llama-3-8B-Instruct") as lm:
        print("DONE.", flush=True)

        system_prompt = """
            You are Leo Messi, the greatest football/soccer player of all
            times.
        """

        async for token in await lm(
            "Who are you?",
            system_prompt=system_prompt,
            settings=GenerationSettings(temperature=0.9, max_new_tokens=256)
        ):
            print(token, end="", flush=True)

if __name__ == "__main__":
    run(example())
```

Besides natural language processing, you can also work with other types of
models, such as those that handle vision, like the following
[image classification example](https://github.com/avalan-ai/avalan/blob/main/docs/examples/vision_image_classification.py):

```python
from asyncio import run
from avalan.model.vision.detection import ObjectDetectionModel
import os
import sys

async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with ObjectDetectionModel("facebook/detr-resnet-50") as od:
        print(f"DONE. Running classification for {path}", flush=True)

        for entity in await od(path):
            print(entity, flush=True)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)==2 and os.path.isfile(sys.argv[1]) \
           else sys.exit(f"Usage: {sys.argv[0]} <valid_file_path>")
    run(example(path))
```

Looking for sequence to sequence models? Just as easy, like this [summarization
example shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/seq2seq_summarization.py):

```python
from asyncio import run
from avalan.entities import GenerationSettings
from avalan.model.nlp.sequence import SequenceToSequenceModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with SequenceToSequenceModel("facebook/bart-large-cnn") as s:
        print("DONE.", flush=True)

        text = """
            Andres Cuccittini, commonly known as Andy Cucci, is an Argentine
            professional footballer who plays as a forward for the Argentina
            national team. Regarded by many as the greatest footballer of all
            time, Cucci has achieved unparalleled success throughout his career.

            Born on July 25, 1988, in Ushuaia, Argentina, Cucci began playing
            football at a young age and joined the Boca Juniors youth
            academy.
            """

        summary = await s(text, GenerationSettings(num_beams=4, max_length=60))
        print(summary)

if __name__ == "__main__":
    run(example())
```

You can also perform translations, as [the following example shows](https://github.com/avalan-ai/avalan/blob/main/docs/examples/seq2seq_translation.py).
You'll need the `translation` extra installed for this to run:

```python
from asyncio import run
from avalan.entities import GenerationSettings
from avalan.model.nlp.sequence import TranslationModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with TranslationModel("facebook/mbart-large-50-many-to-many-mmt") as t:
        print("DONE.", flush=True)

        text = """
            Lionel Messi, commonly known as Leo Messi, is an Argentine
            professional footballer who plays as a forward for the Argentina
            national team. Regarded by many as the greatest footballer of all
            time, Messi has achieved unparalleled success throughout his career.
        """

        translation = await t(
            text,
            source_language="en_US",
            destination_language="es_XX",
            settings=GenerationSettings(num_beams=4, max_length=512)
        )

        print(" ".join([line.strip() for line in text.splitlines()]).strip())
        print("-" * 12)
        print(translation)

if __name__ == "__main__":
    run(example())
```

You can also create AI agents. Let's create one to handle gettext translations.
Create a file named [agent_gettext_translator.toml](https://github.com/avalan-ai/avalan/blob/main/docs/examples.agent_gettext_translator.toml)
with the following contents:

```toml
[agent]
role = """
You are an expert translator that specializes in translating gettext
translation files.
"""
task = """
Your task is to translate the given gettext template file,
from the original {{source_language}} to {{destination_language}}.
"""
instructions = """
The text to translate is marked with `msgid`, and it's quoted.
Your translation should be defined in `msgstr`.
"""
rules = [
    """
    Ensure you keep the gettext format intact, only altering
    the `msgstr` section.
    """,
    """
    Respond only with the translated file.
    """
]

[template]
source_language = "English"
destination_language = "Spanish"

[engine]
uri = "meta-llama/Meta-Llama-3-8B-Instruct"

[run]
use_cache = true
max_new_tokens = 1024
skip_special_tokens = true
```

You can now run your agent. Let's give it a gettext translation template file,
have our agent translate it for us, and show a visual difference of what the
agent changed:

```bash
icdiff locale/avalan.pot <(
    cat locale/avalan.pot |
        avalan agent run docs/examples/agent_gettext_translator.toml --quiet
)
```

![diff showing what the AI translator agent modified](https://avalan.ai/images/agent_translator_diff.png)

There are more agent, NLP, multimodal, audio, and vision examples in the
[docs/examples](https://github.com/avalan-ai/avalan/blob/main/docs/examples)
folder.

> [!TIP]
> If you are working with avalan, you can run the example scripts by
> specifying the `$PYTHONPATH` environment variable:
>
> ```bash
> PYTHONPATH=src poetry run python docs/examples/text_generation.py
> ```

# Install

If you're on MacOS, you can install avalan with homebrew:

```bash
brew tap avalan-ai/avalan

```

Create your virtual environment and install packages:

```bash
poetry install avalan
```

> [!TIP]
> At time of this writing, while Python 3.12 is stable and available
> in Homebrew, sentenpiece, a package added by the extra `translation`,
> requires Python 3.11, so you may want to force the python version when
> creating the virtual environment: `python-3.11 -m venv .venv/`

> [!TIP]
> If you will be using avalan with a device other than `cuda`, or wish to
> use `--low-cpu-mem-usage` you'll need the CPU packages installed, so run
> `poetry install --extras 'cpu'` You can also specify multiple extras to install,
> for example with:
>
> ```bash
> poetry install avalan --extras 'agent audio cpu memory secrets server test translation vision'
> ```
>
> Or you can install all extras at once with:
>
> ```bash
> poetry install avalan --extras all
> ```

> [!TIP]
> If you are going to be using transformer loading classes that haven't yet
> made it into a transformers package released version, install transformers
> development edition:
> `poetry install git+https://github.com/huggingface/transformers --no-cache`

> [!TIP]
> On MacOS, sentencepiece may have issues while installing. If so,
> ensure Xcode CLI is installed, and install needed Homebrew packages
> with:
>
> `xcode-select --install`
> `brew install cmake pkg-config protobuf sentencepiece`

