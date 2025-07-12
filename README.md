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

Avalan empowers developers and enterprises to build, orchestrate, and deploy intelligent AI agents both locally and in the cloud. It provides a unified SDK and CLI for running millions of models with ease.

**Highlights**

- 🔌 Multi-backend support ([transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [mlx-lm](https://github.com/ml-explore/mlx-lm).)
- 🌐 Multi-modal integration (NLP, vision, audio.)
- 🔗 Native adapters for OpenRouter, Ollama, OpenAI, DeepSeek, and Gemini.
- 🤖 Sophisticated memory management and advanced reasoning (ReACT tooling, adaptive planning.)
- 🔀 Intuitive pipelines with branching, filtering, and recursive workflows.
- 📊 Comprehensive observability through metrics, event tracing, and dashboards.
- 🚀 Deploy your AI workflows to the cloud.
- 💻 Use via the CLI or integrate the Python SDK directly in your code.

These features make avalan ideal for everything from quick experiments to enterprise deployments.

Take a quick look at which models you can use in [Models](#models), the tools available to agents in [Tools](#tools), the memories you can configure in [Memories](#memories), how to build and deploy agents in [Serving agents](#serving-agents), the [framework code](#framework-code) you can reuse, and see every CLI option in the [CLI docs](docs/CLI.md).

## Models

Avalan makes text, audio, and vision models available from the CLI or in your
own code. You can run local models or call vendor models from OpenRouter,
OpenAI, Ollama, DeepSeek and Gemini. It works across engines such as
transformers, vLLM and mlx-lm.

### Audio

#### Speech recognition

Recognize speech using a model:

```bash
avalan model run "facebook/wav2vec2-base-960h" \
    --modality audio_speech_recognition \
    --path oprah.wav \
    --audio-sampling-rate 16000
```

The output is the transcript of the provided audio:

```text
AND THEN I GREW UP AND HAD THE ESTEEMED HONOUR OF MEETING HER AND WASN'T
THAT A SURPRISE HERE WAS THIS PETITE ALMOST DELICATE LADY WHO WAS THE
PERSONIFICATION OF GRACE AND GOODNESS
```

#### Text to speech

Generate speech in Oprah's voice from a prompt using an 18-second clip of her [eulogy for Rosa Parks](https://www.americanrhetoric.com/speeches/oprahwinfreyonrosaparks.htm):

```bash
echo "[S1] Leo Messi is the greatest football player of all times." | \
    avalan model run "nari-labs/Dia-1.6B-0626" \
            --modality audio_text_to_speech \
            --path example.wav \
            --audio-reference-path docs/examples/oprah.wav \
            --audio-reference-text "[S1] And then I grew up and had the esteemed honor of meeting her. And wasn't that a surprise. Here was this petite, almost delicate lady who was the personification of grace and goodness."
```

### Text

#### Text generation

Run a locally installed model and adjust sampling settings such as `--temperature`, `--top-p`, and `--top-k`. The example below prompts as "Aurora" and limits the output to 100 tokens:

```bash
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "meta-llama/Meta-Llama-3-8B-Instruct" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature .1 \
        --top-p .9 \
        --top-k 20
```

Vendor APIs work the same way. Swap in a vendor [engine URI](docs/ai_uri.md) to call an external service. The following example calls OpenAI's GPT-4o with the same sampling settings:

```bash
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "ai://$OPENAI_API_KEY@openai/gpt-4o" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature .1 \
        --top-p .9 \
        --top-k 20
```

#### Question answering

Answer questions from context using a question answering model:

```bash
echo "What sport does Leo play?" \
    | avalan model run "deepset/roberta-base-squad2" \
        --modality "text_question_answering" \
        --text-context "Lionel Messi, known as Leo Messi, is an Argentine professional footballer widely regarded as one of the greatest football players of all time."
```

The answer comes as no surprise:

```text
football
```

#### Sequence classification

Determine sentiment in a piece of text:

```bash
echo "We love Leo Messi." \
    | avalan model run "distilbert-base-uncased-finetuned-sst-2-english" \
        --modality "text_sequence_classification"
```

The result is positive, as expected:

```text
POSITIVE
```

#### Sequence to sequence

Summarize a text with a sequence-to-sequence model:

```bash
echo "
    Andres Cuccittini, commonly known as Andy Cucci, is an Argentine
    professional footballer who plays as a forward for the Argentina
    national team. Regarded by many as the greatest footballer of all
    time, Cucci has achieved unparalleled success throughout his career.

    Born on July 25, 1988, in Ushuaia, Argentina, Cucci began playing
    football at a young age and joined the Boca Juniors youth
    academy.
" | avalan model run "facebook/bart-large-cnn" \
        --modality "text_sequence_to_sequence"
```

The resulting summary:

```text
Andy Cucci is held by many as the greatest footballer of all times.
```

#### Translation

Translate text between languages using a sequence-to-sequence model:

```bash
echo "
    Lionel Messi, commonly known as Leo Messi, is an Argentine
    professional footballer who plays as a forward for the Argentina
    national team. Regarded by many as the greatest footballer of all
    time, Messi has achieved unparalleled success throughout his career.
" | avalan model run "facebook/mbart-large-50-many-to-many-mmt" \
        --modality "text_translation" \
        --text-from-lang "en_US" \
        --text-to-lang "es_XX" \
        --text-num-beams 4 \
        --text-max-length 512
```

The Spanish version of the text:

```text
Lionel Messi, conocido como Leo Messi, es un futbolista argentino profesional
que representa a la Argentina en el equipo nacional. Considerado por muchos
como el mejor futbolista de todos los tiempos, Messi ha conseguido un éxito
sin precedentes durante su carrera.
```

### Vision

#### Image classification

Classify an image (hot dog or not):

```bash
avalan model run "microsoft/resnet-50" \
    --modality vision_image_classification \
    --path docs/examples/cat.jpg
```

The model identifies the image:

```text
┏━━━━━━━━━━━━━━━━━━┓
┃ Label            ┃
┡━━━━━━━━━━━━━━━━━━┩
│ tabby, tabby cat │
└──────────────────┘
```

#### Image to text

Generate a text description for an image:

```bash
avalan model run "salesforce/blip-image-captioning-base" \
    --modality vision_image_to_text \
    --path docs/examples/Example_Image_1.jpg
```

Example output:

```text
a sign for a gas station on the side of a building [SEP]
```

#### Image text to text

Provide an image and an instruction to an `image-text-to-text` model:

```bash
echo "Transcribe the text on this image, keeping format" | \
    avalan model run "ai://local/google/gemma-3-12b-it" \
        --modality vision_image_text_to_text \
        --path docs/examples/typewritten_partial_sheet.jpg \
        --max-new-tokens 1024
```

The transcription (truncated for brevity):

```text
**INTRODUCCIÓN**

Guillermo de Ockham (según se utiliza la grafía latina o la inglesa) es tan célebre como conocido. Su doctrina
suele merecer las más diversas interpretaciones, y su biografía adolece tremendas oscuridades.
```

#### Object detection

Detect objects in an image and list them with accuracy scores:

```bash
avalan model run "facebook/detr-resnet-50" \
    --modality vision_object_detection \
    --path docs/examples/kitchen.jpg
```

Results are sorted by accuracy and include bounding boxes:

```text
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Label        ┃ Score ┃ Box                              ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ refrigerator │  1.00 │ 855.28, 377.27, 1035.67, 679.42  │
├──────────────┼───────┼──────────────────────────────────┤
│ oven         │  1.00 │ 411.62, 570.92, 651.66, 872.05   │
├──────────────┼───────┼──────────────────────────────────┤
│ potted plant │  0.99 │ 1345.95, 498.15, 1430.21, 603.84 │
├──────────────┼───────┼──────────────────────────────────┤
│ sink         │  0.96 │ 1077.43, 631.51, 1367.12, 703.23 │
├──────────────┼───────┼──────────────────────────────────┤
│ potted plant │  0.94 │ 179.69, 557.44, 317.14, 629.77   │
├──────────────┼───────┼──────────────────────────────────┤
│ vase         │  0.83 │ 1357.88, 562.67, 1399.38, 616.44 │
├──────────────┼───────┼──────────────────────────────────┤
│ handbag      │  0.72 │ 287.08, 544.47, 332.73, 602.24   │
├──────────────┼───────┼──────────────────────────────────┤
│ sink         │  0.68 │ 1079.68, 627.04, 1495.40, 714.07 │
├──────────────┼───────┼──────────────────────────────────┤
│ bird         │  0.38 │ 628.57, 536.31, 666.62, 574.39   │
├──────────────┼───────┼──────────────────────────────────┤
│ sink         │  0.35 │ 1077.98, 629.29, 1497.90, 723.95 │
├──────────────┼───────┼──────────────────────────────────┤
│ spoon        │  0.31 │ 646.69, 505.31, 673.04, 543.10   │
└──────────────┴───────┴──────────────────────────────────┘
```

#### Semantic segmentation

Classify each pixel with a semantic segmentation model:

```bash
avalan model run "nvidia/segformer-b0-finetuned-ade-512-512" \
    --modality vision_semantic_segmentation \
    --path docs/examples/kitchen.jpg
```

The output lists each annotation:

```text
┏━━━━━━━━━━━━━━━━━━┓
┃ Label            ┃
┡━━━━━━━━━━━━━━━━━━┩
│ wall             │
├──────────────────┤
│ floor            │
├──────────────────┤
│ ceiling          │
├──────────────────┤
│ windowpane       │
├──────────────────┤
│ cabinet          │
├──────────────────┤
│ door             │
├──────────────────┤
│ plant            │
├──────────────────┤
│ rug              │
├──────────────────┤
│ lamp             │
├──────────────────┤
│ chest of drawers │
├──────────────────┤
│ sink             │
├──────────────────┤
│ refrigerator     │
├──────────────────┤
│ flower           │
├──────────────────┤
│ stove            │
├──────────────────┤
│ kitchen island   │
├──────────────────┤
│ light            │
├──────────────────┤
│ chandelier       │
├──────────────────┤
│ oven             │
├──────────────────┤
│ microwave        │
├──────────────────┤
│ dishwasher       │
├──────────────────┤
│ hood             │
├──────────────────┤
│ vase             │
├──────────────────┤
│ fan              │
└──────────────────┘
```


## Tools

Avalan makes it simple to launch a chat-based agent that can call external tools while streaming tokens. The example below uses a local 8B LLM, enables recent memory, and loads a calculator tool. The agent begins with a math question and remains open for follow-ups:

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

Notice the GPU utilization at the bottom:

![Example use of an ephemeral tool agent with memory](https://avalan.ai/images/cli_agent_tool.gif)

Below is an agent that leverages the `code.run` tool to execute Python code
generated by the model and display the result:

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

Tools give agents real-time knowledge. This example uses an 8B model and a browser tool to find avalan's latest release:

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

You can direct an agent to read specific locations for knowledge:

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

Start a chat session and tell the agent your name. The `--memory-permanent-message` option specifies where messages are stored, `--id` uniquely identifies the agent, and `--participant` sets the user ID:

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

Enable persistent memory and the `memory.message.read` tool so the agent can recall earlier messages. It should discover that your name is `Leo` from the previous conversation:

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

Agents can use knowledge stores to solve problems. Index the rules of the "Truco" card game directly from a website. The `--dsn` parameter sets the store location and `--namespace` chooses the knowledge namespace:

```bash
avalan memory document index \
    --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
    --dsn "postgresql://root:password@localhost/avalan" \
    --namespace "games.cards.truco" \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "https://trucogame.com/pages/reglamento-de-truco-argentino"
```

## Serving agents

Serve your agents on an OpenAI API–compatible endpoint:

```bash
avalan agent serve docs/examples/agent_tool.toml -vvv
```

Or build an agent from inline settings and expose its OpenAI API endpoints:

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

You can call your tool streaming agent's OpenAI-compatible endpoint just like
the real API; simply change `--base-url`:

```bash
echo "What is (4 + 6) and then that result times 5, divided by 2?" | \
    avalan model run "ai://openai" --base-url "http://localhost:9001/v1"
```

## Framework code

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

# Install

On macOS you can install avalan with Homebrew:

```bash
brew tap avalan-ai/avalan
```

On other environments, use poetry to install avalan:

```bash
poetry install avalan
```

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
> On macOS, sentencepiece may have issues during installation. If so,
> ensure Xcode CLI is installed, and install needed Homebrew packages
> with:
>
> `xcode-select --install`
> `brew install cmake pkg-config protobuf sentencepiece`

