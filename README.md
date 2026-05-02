<h1 align="center">avalan</h1>
<h3 align="center">The multi-backend, multi-modal micro-framework for AI agent development, orchestration, and deployment</h3>

<p align="center">
  <a href="https://github.com/avalan-ai/avalan/actions/workflows/test.yml"><img src="https://github.com/avalan-ai/avalan/actions/workflows/test.yml/badge.svg" alt="Tests" /></a>
  <a href="https://coveralls.io/github/avalan-ai/avalan"><img src="https://coveralls.io/repos/github/avalan-ai/avalan/badge.svg" alt="Code test coverage" /></a>
  <img src="https://img.shields.io/github/last-commit/avalan-ai/avalan.svg" alt="Last commit" />
  <a href="https://pypi.org/project/avalan/"><img src="https://img.shields.io/github/v/release/avalan-ai/avalan?label=Release" alt="Release" /></a>
  <a href="https://github.com/avalan-ai/avalan/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/avalan.svg" alt="License" /></a>
  <a href="https://discord.gg/8Eh9TNvk"><img src="https://img.shields.io/badge/discord-community-blue" alt="Discord Community" /></a>
  <a href="https://deepwiki.com/avalan-ai/avalan"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

Avalan is a Python SDK and CLI for building and running AI workflows and agents across local and hosted models.

# ✨ Highlights

- 🔀 **One runtime for local and hosted models** across Hugging Face model ids and
  vendor `ai://` URIs.
- 🎞️ **Multi-modal support** for text, vision, and audio workloads.
- 🔌 **Multiple backends** including
  [`transformers`](https://github.com/huggingface/transformers),
  [`vLLM`](https://github.com/vllm-project/vllm), and
  [`mlx-lm`](https://github.com/ml-explore/mlx-lm).
- 🧰 **Built-in tools and memory** for browser automation, code execution,
  databases, MCP, search, YouTube, and vector-backed retrieval.
- 🧠 **Composable orchestration** with flows, branching, reasoning strategies, and
  observability.
- 🌐 **Open serving surfaces** for OpenAI-compatible APIs, MCP, and A2A.

# 🚀 Start Here

- [Install](#install)
- [Quickstart](#quickstart)
- [Models](#models)
- [Agents](#agents)
- [docs/examples](docs/examples/README.md)
- [docs/CLI.md](docs/CLI.md)
- [docs/ai_uri.md](docs/ai_uri.md)

# 🗂️ Table of Contents

- 📦 [Install](#install)
- ⚡ [Quickstart](#quickstart)
- 🧪 [Models](#models)
- 🎛️ [Modalities](#modalities)
- 🧰 [Tools](#tools)
- 🧠 [Reasoning strategies](#reasoning-strategies)
- 🗃️ [Memories](#memories)
- 🤖 [Agents](#agents)
- 📚 [Documentation & Resources](#documentation--resources)
- 🤝 [Community & Support](#community--support)
- 🧑‍💻 [Contributing](#contributing)

## 📦 Install

Avalan supports Python 3.11 and 3.12. Install the smallest profile that fits
your workflow; the examples later in this README may require additional extras.

### 🐍 Pip (recommended)

Hosted APIs plus tool-enabled or served agents:

```sh
python3 -m pip install -U "avalan[agent,server,tool,vendors]"
```

Broader local development setup with the capabilities used throughout this
README:

```sh
python3 -m pip install -U "avalan[agent,audio,memory,server,tool,translation,vendors,vision]"
```

Add hardware-specific extras when needed:

- `mlx` or `apple` – Apple Silicon acceleration via MLX / MLX-LM.
- `nvidia` – Linux + NVIDIA bundle for vLLM and quantization support.
- `vllm` – the vLLM runtime without the full NVIDIA bundle.
- `quantization` – 4-bit and 8-bit model loading.

For the leanest install, omit the extras list entirely.

### 🍺 Homebrew (macOS)

```sh
brew tap avalan-ai/avalan
brew install avalan
```

### 🛠️ From Source with Poetry

```sh
poetry install --all-extras --with test
```

> [!TIP]
> On macOS ensure the Xcode command line tools are present and install the build dependencies before compiling extras that rely on `sentencepiece`:
>
> ```sh
> xcode-select --install
> brew install cmake pkg-config protobuf sentencepiece
> ```

When you need bleeding-edge `transformers` features, install the latest nightly:

```sh
poetry run pip install --no-cache-dir "git+https://github.com/huggingface/transformers"
```

## ⚡ Quickstart

### 💬 Call a hosted model from the CLI

Export a vendor key, then run:

```sh
export OPENAI_API_KEY=...
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100
```

### 🐍 Use the Python SDK

```python
import asyncio

from avalan.model.nlp.generation import TextGenerationModel

async def main() -> None:
    with TextGenerationModel("ai://env:OPENAI_API_KEY@openai/gpt-4o") as model:
        response = await model("Give me two facts about Leo Messi.")
        print(response)

        async for token in await model(
            "Give me two more facts about Leo Messi.",
            stream=True,
        ):
            print(token, end="", flush=True)

asyncio.run(main())
```

### 🧭 Next steps

- 📚 Browse [docs/examples](docs/examples/README.md) for runnable scripts across
  text, audio, vision, tools, and agent serving.
- 🧪 Jump to [Models](#models) to search and install open models locally.
- 🤖 Jump to [Agents](#agents) to expose an agent over OpenAI-compatible HTTP,
  MCP, or A2A.

## 🧪 Models

Avalan exposes text, audio, and vision models from the CLI and Python. Use bare
model ids for open models and `ai://` engine URIs for vendor-hosted models or
custom endpoints.

### Vendor models

Avalan supports popular vendor models through
[engine URIs](docs/ai_uri.md). The example below uses OpenAI's GPT-4o:

```sh
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature .1 \
        --top-p .9 \
        --top-k 20
```

### Open models

Open models run across engines such as `transformers`, `vllm`, and `mlx`.
Search through millions of them with `avalan model search` using different
filters. The following command looks for up to three text-generation models that
run with the `mlx` backend, match the term `DeepSeek-R1`, and were published by
the MLX community:

```sh
avalan model search --name DeepSeek-R1 \
    --library mlx \
    --task text-generation \
    --author "mlx-community" \
    --limit 3
```

The command returns three matching models:

```text
┌───── 📛 mlx-community/DeepSeek-R1-Distill-Qwen-14B 🧮 N/A params ─────┐
│ ✅ access granted 💼 mlx-community · 📆 updated: 4 months ago         │
│ 📚 transformers · ⚙ text-generation                                   │
└───────────────────────────────────────────────────────────────────────┘
┌───── 📛 mlx-community/DeepSeek-R1-Distill-Qwen-7B 🧮 N/A params ──────┐
│ ✅ access granted 💼 mlx-community · 📆 updated: 4 months ago         │
│ 📚 transformers · ⚙ text-generation                                   │
└───────────────────────────────────────────────────────────────────────┘
┌─ 📛 mlx-community/Unsloth-DeepSeek-R1-Distill-Qwen-14B-4bit 🧮 N/A pa─┐
│ ✅ access granted 💼 mlx-community · 📆 updated: 4 months ago         │
│ 📚 transformers · ⚙ text-generation                                   │
└───────────────────────────────────────────────────────────────────────┘
```

Install the first model:

```sh
avalan model install mlx-community/DeepSeek-R1-Distill-Qwen-14B
```

The model is now ready to use:

```text
┌──── 📛 mlx-community/DeepSeek-R1-Distill-Qwen-14B 🧮 14.8B params ────┐
│ ✅ access granted 💼 mlx-community · 📆 updated: 4 months ago         │
│ 🤖 qwen2 · 📚 transformers · ⚙ text-generation                        │
└───────────────────────────────────────────────────────────────────────┘
💾 Downloading model mlx-community/DeepSeek-R1-Distill-Qwen-14B:

  Fetching 13 files 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ [ 13/13 - 0:04:15 ]

✔ Downloaded model mlx-community/DeepSeek-R1-Distill-Qwen-14B to
/Users/leo/.cache/huggingface/hub/models--mlx-community--DeepSeek-R1-
Distill-Qwen-14B/snapshots/68570f64bcc30966595926e3b7d200a9d77fb1e8
```

Test the model we just installed, specifying `mlx` as the backend:

> [!TIP]
> You can choose your preferred backend using the `--backend` option. For example,
> on Apple Silicon Macs, the `mlx` backend typically offers a 3x speedup
> compared to the default `transformers` backend. On devices with access to
> Nvidia GPUs, models that run on the backend `vllm` are also orders of
> magnitude faster.

```sh
echo 'What is (4 + 6) and then that result times 5, divided by 2?' | \
    avalan model run 'mlx-community/DeepSeek-R1-Distill-Qwen-14B' \
        --temperature 0.6 \
        --max-new-tokens 1024 \
        --start-thinking \
        --backend mlx
```

The output shows the reasoning and the correct final answer:

```text
┌───────────────────────────────────────────────────────────────────────┐
│ ✅ access granted 💼 mlx-community                                    │
└───────────────────────────────────────────────────────────────────────┘

🗣  What is (4 + 6) and then that result times 5, divided by 2?

┌─ mlx-community/DeepSeek-R1-Distill-Qwen-14B reasoning ────────────────┐
│                                                                       │
│ First, I will add 4 and 6 to get the result.                          │
│ Next, I will multiply that sum by 5.                                  │
│ Then, I will divide the product by 2 to find the final answer.        │
│ </think>                                                              │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│    \]                                                                 │
│                                                                       │
│ 3. **Divide the product by 2:**                                       │
│    [                                                                  │
│    50 \div 2 = 25                                                     │
│    \]                                                                 │
│                                                                       │
│ **Final Answer:**                                                     │
│ [                                                                     │
│ \boxed{25}                                                            │
│                                                                       │
└─ 💻 26 tokens in · 🧮 158 token out · 🌱 ttft: 1.14 s · ⚡ 14.90 t/s ─┘
```

### Modalities

The following examples show each modality in action. Use the table of contents below to jump to the
task you need:

* 🎧 [**Audio**](#audio): Turn audio into text or produce speech for accessibility and media.
  - 🦻 [Audio classification](#audio-classification): Label an audio based on sentiment.
  - 🗣️ [Speech recognition](#speech-recognition): Convert spoken audio to text.
  - 🔊 [Text to speech](#text-to-speech): Synthesize natural speech from text.
  - 🎵 [Audio generation](#audio-generation): Generate or continue audio from prompts.
* 📝 [**Text**](#text): Understand, transform, and generate text across core NLP tasks.
  - ❓ [Question answering](#question-answering): Extract precise answers from context.
  - 🏷️ [Sequence classification](#sequence-classification): Classify text into labels.
  - 🔁 [Sequence to sequence](#sequence-to-sequence): Rewrite or transform text-to-text.
  - ✍️ [Text generation](#text-generation): Produce completions and long-form responses.
  - 🧩 [Token classification](#token-classification): Label entities and token-level tags.
  - 🌍 [Translation](#translation): Translate text between languages.
* 👁️ [**Vision**](#vision): Analyze images and create visual outputs from prompts.
  - 🖼️ [Encoder decoder](#encoder-decoder): Caption images with encoder-decoder models.
  - 🧪 [Image classification](#image-classification): Predict image classes and scores.
  - 📝 [Image to text](#image-to-text): Describe image content in natural language.
  - 💬 [Image text to text](#image-text-to-text): Answer questions grounded in images.
  - 🎯 [Object detection](#object-detection): Detect and localize objects.
  - 🧱 [Semantic segmentation](#semantic-segmentation): Label each pixel by class.
  - 🕺 [Text to animation](#text-to-animation): Create animated clips from prompts.
  - 🎨 [Text to image](#text-to-image): Generate images from text prompts.
  - 🎬 [Text to video](#text-to-video): Generate short videos from prompts.

### Audio

#### Audio classification

Determine the sentiment (neutral, happy, angry, sad) of a given audio file:

```sh
avalan model run "superb/hubert-base-superb-er" \
    --modality audio_classification \
    --path docs/examples/playground/oprah.wav \
    --audio-sampling-rate 16000
```

Output:

```text
┏━━━━━━━┳━━━━━━━┓
┃ Label ┃ Score ┃
┡━━━━━━━╇━━━━━━━┩
│ ang   │ 0.49  │
├───────┼───────┤
│ hap   │ 0.45  │
├───────┼───────┤
│ neu   │ 0.04  │
├───────┼───────┤
│ sad   │ 0.02  │
└───────┴───────┘
```

Python:

```python
from avalan.model.audio.classification import AudioClassificationModel

with AudioClassificationModel("superb/hubert-base-superb-er") as model:
    labels = await model("oprah.wav", sampling_rate=16000)
    print(labels)
```
For a runnable script, see [docs/examples/audio_classification.py](docs/examples/audio_classification.py).

#### Speech recognition

Transcribe speech from an audio file:

```sh
avalan model run "facebook/wav2vec2-base-960h" \
    --modality audio_speech_recognition \
    --path docs/examples/playground/oprah.wav \
    --audio-sampling-rate 16000
```

The output is the transcript of the provided audio:

```text
AND THEN I GREW UP AND HAD THE ESTEEMED HONOUR OF MEETING HER AND WASN'T
THAT A SURPRISE HERE WAS THIS PETITE ALMOST DELICATE LADY WHO WAS THE
PERSONIFICATION OF GRACE AND GOODNESS
```

Python:

```python
from avalan.model.audio.speech_recognition import SpeechRecognitionModel

with SpeechRecognitionModel("facebook/wav2vec2-base-960h") as model:
    output = await model("oprah.wav", sampling_rate=16000)
    print(output)
```
For a runnable script, see [docs/examples/audio_speech_recognition.py](docs/examples/audio_speech_recognition.py).

#### Text to speech

Generate speech in Oprah's voice from a text prompt. The example uses an 18-second clip from her [eulogy for Rosa Parks](https://www.americanrhetoric.com/speeches/oprahwinfreyonrosaparks.htm) as a reference:

```sh
echo "[S1] Leo Messi is the greatest football player of all times." | \
    avalan model run "nari-labs/Dia-1.6B-0626" \
            --modality audio_text_to_speech \
            --path example.wav \
            --audio-reference-path docs/examples/playground/oprah.wav \
            --audio-reference-text "[S1] And then I grew up and had the esteemed honor of meeting her. And wasn't that a surprise. Here was this petite, almost delicate lady who was the personification of grace and goodness."
```

Python:

```python
from avalan.model.audio.speech import TextToSpeechModel

with TextToSpeechModel("nari-labs/Dia-1.6B-0626") as model:
    await model(
        "[S1] Leo Messi is the greatest football player of all times.",
        "example.wav",
        reference_path="docs/examples/playground/oprah.wav",
        reference_text=(
            "[S1] And then I grew up and had the esteemed honor of meeting her. "
            "And wasn't that a surprise. Here was this petite, almost delicate "
            "lady who was the personification of grace and goodness."
        ),
    )
```
For a runnable script, see [docs/examples/audio_text_to_speech.py](docs/examples/audio_text_to_speech.py).

#### Audio generation

Create a short melody from a text prompt:

```sh
echo "A funky riff about Leo Messi." |
    avalan model run "facebook/musicgen-small" \
        --modality audio_generation \
        --max-new-tokens 1024 \
        --path melody.wav
```

Python:

```python
from avalan.model.audio.generation import AudioGenerationModel

with AudioGenerationModel("facebook/musicgen-small") as model:
    await model("A funky riff about Leo Messi.", "melody.wav", max_new_tokens=1024)
```
For a runnable script, see [docs/examples/audio_generation.py](docs/examples/audio_generation.py).

### Text

#### Question answering

Answer a question based on context using a question answering model:

```sh
echo "What sport does Leo play?" \
    | avalan model run "deepset/roberta-base-squad2" \
        --modality "text_question_answering" \
        --text-context "Lionel Messi, known as Leo Messi, is an Argentine professional footballer widely regarded as one of the greatest football players of all time."
```

Output:

```text
football
```

Python:

```python
from avalan.model.nlp.question import QuestionAnsweringModel

with QuestionAnsweringModel("deepset/roberta-base-squad2") as model:
    answer = await model(
        "What sport does Leo play?",
        context="Lionel Messi, known as Leo Messi, is an Argentine professional footballer widely regarded as one of the greatest football players of all time."
    )
    print(answer)
```
For a runnable script, see [docs/examples/question_answering.py](docs/examples/question_answering.py).

#### Sequence classification

Classify the sentiment of short text:

```sh
echo "We love Leo Messi." \
    | avalan model run "distilbert-base-uncased-finetuned-sst-2-english" \
        --modality "text_sequence_classification"
```

The result is positive as expected:

```text
POSITIVE
```

Python:

```python
from avalan.model.nlp.sequence import SequenceClassificationModel

with SequenceClassificationModel("distilbert-base-uncased-finetuned-sst-2-english") as model:
    output = await model("We love Leo Messi.")
    print(output)
```
For a runnable script, see [docs/examples/sequence_classification.py](docs/examples/sequence_classification.py).

#### Sequence to sequence

Summarize text using a sequence-to-sequence model:

```sh
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

The summary:

```text
Andres Cuccittini, commonly known as Andy Cucci, is an Argentine professional
footballer. He plays as a forward for the Argentina national team. Cucci began
playing football at the age of 19 in his native Ushuaia.
```

Python:

```python
from avalan.model.nlp.sequence import SequenceToSequenceModel

with SequenceToSequenceModel("facebook/bart-large-cnn") as model:
    output = await model("""
    Andres Cuccittini, commonly known as Andy Cucci, is an Argentine
    professional footballer who plays as a forward for the Argentina
    national team. Regarded by many as the greatest footballer of all
    time, Cucci has achieved unparalleled success throughout his career.

    Born on July 25, 1988, in Ushuaia, Argentina, Cucci began playing
    football at a young age and joined the Boca Juniors youth
    academy.
    """)
    print(output)
```
For a runnable script, see [docs/examples/seq2seq_summarization.py](docs/examples/seq2seq_summarization.py).

#### Text generation

Run a local model and control sampling with `--temperature`, `--top-p`, and `--top-k`. The example instructs the assistant to act as "Aurora" and limits the output to 100 tokens:

```sh
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "meta-llama/Meta-Llama-3-8B-Instruct" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature .1 \
        --top-p .9 \
        --top-k 20 \
        --backend mlx
```

Python:

```python
from avalan.entities import GenerationSettings
from avalan.model.nlp.text.generation import TextGenerationModel

with TextGenerationModel("meta-llama/Meta-Llama-3-8B-Instruct") as model:
    async for token in await model(
        "Who are you, and who is Leo Messi?",
        system_prompt="You are Aurora, a helpful assistant",
        settings=GenerationSettings(
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.9,
            top_k=20
        )
    ):
        print(token, end="", flush=True)
```

Vendor APIs use the same interface. Swap in a vendor [engine URI](docs/ai_uri.md) to call an external service. The example below uses OpenAI's GPT-4o with the same parameters:

```sh
echo "Who are you, and who is Leo Messi?" \
    | avalan model run "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
        --system "You are Aurora, a helpful assistant" \
        --max-new-tokens 100 \
        --temperature .1 \
        --top-p .9 \
        --top-k 20
```

Python:

```python
from avalan.entities import GenerationSettings
from avalan.model.nlp.text.generation import TextGenerationModel

with TextGenerationModel("ai://env:OPENAI_API_KEY@openai/gpt-4o") as model:
    async for token in await model(
        "Who are you, and who is Leo Messi?",
        system_prompt="You are Aurora, a helpful assistant",
        settings=GenerationSettings(
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.9,
            top_k=20
        )
    ):
        print(token, end="", flush=True)
```
For a runnable script, see [docs/examples/text_generation.py](docs/examples/text_generation.py).

Amazon Bedrock models use the same workflow. With your AWS credentials
configured (for example with `AWS_PROFILE` or environment variables),
you can target any Bedrock region via `--base-url`:

```sh
echo "Summarize the latest AWS re:Invent keynote in three bullet points." \
      | avalan model run "ai://bedrock/us.amazon.nova-lite-v1:0" \
          --base-url "us-east-1" \
          --max-new-tokens 256 \
          --temperature .7
```

Example output:

```
- **Hybrid and Multicloud**: AWS expanded its hybrid and multicloud capabilities with new services to help customers seamlessly connect their on-premises environments with AWS, and manage workloads across multiple clouds.

- **Security and Compliance**: AWS announced new security and compliance features to help customers meet their regulatory requirements and protect their data, including new services for data encryption, identity management, and threat detection.

These highlights capture some of the major themes and announcements from the keynote, but there were many more details and product updates as well. Hopefully this summary gives you a good overview! Let me know if you have any other
```

> [!TIP]
> Some Bedrock models are only available through geo-prefixed IDs in a
> given source region, such as `us.anthropic.claude-sonnet-4-6` for US
> Anthropic routing. Those profile IDs change over time, so you can
> inspect currently active options with
> `aws bedrock list-inference-profiles --region us-east-1`. Anthropic
> models can also require submitting the Bedrock use-case details form for
> your account before inference is allowed.

#### Token classification

Classify tokens with labels for Named Entity Recognition (NER) or
Part-of-Speech (POS):

```sh
echo "
    Lionel Messi, commonly known as Leo Messi, is an Argentine
    professional footballer widely regarded as one of the
    greatest football players of all time.
" | avalan model run "dslim/bert-base-NER" \
    --modality text_token_classification \
    --text-labeled-only
```

Output:

```text
┏━━━━━━━━━━━┳━━━━━━━━┓
┃ Token     ┃ Label  ┃
┡━━━━━━━━━━━╇━━━━━━━━┩
│ Lionel    │ B-PER  │
├───────────┼────────┤
│ Me        │ I-PER  │
├───────────┼────────┤
│ ##ssi     │ I-PER  │
├───────────┼────────┤
│ Leo       │ B-PER  │
├───────────┼────────┤
│ Argentine │ B-MISC │
└───────────┴────────┘
```

Python:

```python
from avalan.model.nlp.token import TokenClassificationModel

with TokenClassificationModel("dslim/bert-base-NER") as model:
    labels = await model(
        "Lionel Messi, commonly known as Leo Messi, is an Argentine professional footballer widely regarded as one of the greatest football players of all time.",
        labeled_only=True
    )
    print(labels)
```
For a runnable script, see [docs/examples/token_classification.py](docs/examples/token_classification.py).

#### Translation

Translate text between languages with a sequence-to-sequence model:

```sh
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

Output:

```text
Lionel Messi, conocido comúnmente como Leo Messi, es un futbolista argentino
profesional que representa a la Argentina en el equipo nacional de Argentina.
Considerado por muchos como el mejor futbolista de todos los tiempos, Messi ha
conseguido un éxito sin precedentes en toda su carrera.
```

Python:

```python
from avalan.entities import GenerationSettings
from avalan.model.nlp.sequence import TranslationModel

with TranslationModel("facebook/mbart-large-50-many-to-many-mmt") as model:
    output = await model(
        "Lionel Messi, commonly known as Leo Messi, is an Argentine professional footballer who plays as a forward for the Argentina national team. Regarded by many as the greatest footballer of all time, Messi has achieved unparalleled success throughout his career.",
        source_language="en_US",
        destination_language="es_XX",
        settings=GenerationSettings(
            num_beams=4,
            max_length=512
        )
    )
    print(output)
```
For a runnable script, see [docs/examples/seq2seq_translation.py](docs/examples/seq2seq_translation.py).

### Vision

#### Encoder decoder

Answer questions to extract information from an image, without using OCR.

```sh
echo "<s_docvqa><s_question>
    What is the FACTURA Number?
</s_question><s_answer>" | \
    avalan model run "naver-clova-ix/donut-base-finetuned-docvqa" \
        --modality vision_encoder_decoder \
        --path docs/examples/playground/invoice-factura.png
```

Output:

```
<s_docvqa>
<s_question> What is the FACTURA Number?</s_question>
<s_answer> 0012-00187506</s_answer>
</s>
```

Python:

```python
from avalan.model.vision.decoder import VisionEncoderDecoderModel

with VisionEncoderDecoderModel("naver-clova-ix/donut-base-finetuned-docvqa") as model:
    answer = await model(
        "docs/examples/playground/invoice-factura.png",
        prompt="<s_docvqa><s_question>What is the FACTURA Number?</s_question><s_answer>"
    )
    print(answer)
```
For a runnable script, see [docs/examples/vision_encoder_decoder.py](docs/examples/vision_encoder_decoder.py).

#### Image classification

Classify an image, such as determining whether it is a hot dog, or not a hot dog 🤓:

```sh
avalan model run "microsoft/resnet-50" \
    --modality vision_image_classification \
    --path docs/examples/playground/cat.jpg
```

The model identifies the image:

```text
┏━━━━━━━━━━━━━━━━━━┓
┃ Label            ┃
┡━━━━━━━━━━━━━━━━━━┩
│ tabby, tabby cat │
└──────────────────┘
```

Python:

```python
from avalan.model.vision.image import ImageClassificationModel

with ImageClassificationModel("microsoft/resnet-50") as model:
    output = await model("docs/examples/playground/cat.jpg")
    print(output)
```
For a runnable script, see [docs/examples/vision_image_classification.py](docs/examples/vision_image_classification.py).

#### Image to text

Generate a caption for an image:

```sh
avalan model run "salesforce/blip-image-captioning-base" \
    --modality vision_image_to_text \
    --path docs/examples/playground/Example_Image_1.jpg
```

Example output:

```text
a sign for a gas station on the side of a building [SEP]
```

Python:

```python
from avalan.model.vision.image import ImageToTextModel

with ImageToTextModel("salesforce/blip-image-captioning-base") as model:
    caption = await model("docs/examples/playground/Example_Image_1.jpg")
    print(caption)
```
For a runnable script, see [docs/examples/vision_image_to_text.py](docs/examples/vision_image_to_text.py).

#### Image text to text

Provide an image and an instruction to an `image-text-to-text` model:

```sh
echo "Transcribe the text on this image, keeping format" | \
    avalan model run "ai://local/google/gemma-3-12b-it" \
        --modality vision_image_text_to_text \
        --path docs/examples/playground/typewritten_partial_sheet.jpg \
        --vision-width 512 \
        --max-new-tokens 1024
```

The transcription (truncated for brevity):

```text
**INTRODUCCIÓN**

Guillermo de Ockham (según se utiliza la grafía latina o la inglesa) es tan
célebre como mal conocido. Su doctrina suele merecer las más diversas
interpretaciones, y su biografía adolece tremendas oscuridades.

Aún más, y como dice un renombrado autor, el estudio de su pensamiento "parece,
por la falta de buenas ediciones de sus obras, una consecuencia del ‘anatema’
que, durante siglos, ha pesado sobre el incipor del nominalismo" (1).
```

Python:

```python
from avalan.entities import GenerationSettings
from avalan.model.vision.image import ImageTextToTextModel

with ImageTextToTextModel("google/gemma-3-12b-it") as model:
    output = await model(
        "docs/examples/playground/typewritten_partial_sheet.jpg",
        "Transcribe the text on this image, keeping format",
        settings=GenerationSettings(max_new_tokens=1024),
        width=512
    )
    print(output)
```
For a runnable script, see [docs/examples/vision_ocr.py](docs/examples/vision_ocr.py).

#### Object detection

Detect objects in an image and list them with accuracy scores:

```sh
avalan model run "facebook/detr-resnet-50" \
    --modality vision_object_detection \
    --path docs/examples/playground/kitchen.jpg \
    --vision-threshold 0.3
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

Python:

```python
from avalan.model.vision.detection import ObjectDetectionModel

with ObjectDetectionModel("facebook/detr-resnet-50") as model:
    labels = await model("docs/examples/playground/kitchen.jpg", threshold=0.3)
    print(labels)
```
For a runnable script, see [docs/examples/vision_object_detection.py](docs/examples/vision_object_detection.py).

#### Semantic segmentation

Classify each pixel using a semantic segmentation model:

```sh
avalan model run "nvidia/segformer-b0-finetuned-ade-512-512" \
    --modality vision_semantic_segmentation \
    --path docs/examples/playground/kitchen.jpg
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

Python:

```python
from avalan.model.vision.segmentation import SemanticSegmentationModel

with SemanticSegmentationModel("nvidia/segformer-b0-finetuned-ade-512-512") as model:
    labels = await model("docs/examples/playground/kitchen.jpg")
    print(labels)
```
For a runnable script, see [docs/examples/vision_semantic_segmentation.py](docs/examples/vision_semantic_segmentation.py).

#### Text to animation

Create an animation from a prompt using a base model for styling:

```sh
echo 'A tabby cat slowly walking' | \
    avalan model run "ByteDance/AnimateDiff-Lightning" \
        --modality vision_text_to_animation \
        --base-model "stablediffusionapi/mistoonanime-v30" \
        --checkpoint "animatediff_lightning_4step_diffusers.safetensors" \
        --weight "fp16" \
        --path example_cat_walking.gif \
        --vision-beta-schedule "linear" \
        --vision-guidance-scale 1.0 \
        --vision-steps 4 \
        --vision-timestep-spacing "trailing"
```

And here's the generated anime inspired animation of a walking cat:

![An anime cat slowly walking](https://avalan.ai/images/github/vision_text_to_animation_generated.webp)

Python:

```python
from avalan.entities import EngineSettings
from avalan.model.vision.diffusion import TextToAnimationModel

with TextToAnimationModel("ByteDance/AnimateDiff-Lightning", settings=EngineSettings(base_model_id="stablediffusionapi/mistoonanime-v30", checkpoint="animatediff_lightning_4step_diffusers.safetensors", weight_type="fp16")) as model:
    await model(
        "A tabby cat slowly walking",
        "example_cat_walking.gif",
        beta_schedule="linear",
        guidance_scale=1.0,
        steps=4,
        timestep_spacing="trailing"
    )
```
For a runnable script, see [docs/examples/vision_text_to_animation.py](docs/examples/vision_text_to_animation.py).

#### Text to image

Create an image from a text prompt:

```sh
echo 'Leo Messi petting a purring tubby cat' | \
    avalan model run "stabilityai/stable-diffusion-xl-base-1.0" \
        --modality vision_text_to_image \
        --refiner-model "stabilityai/stable-diffusion-xl-refiner-1.0" \
        --weight "fp16" \
        --path example_messi_petting_cat.jpg \
        --vision-color-model RGB \
        --vision-image-format JPEG \
        --vision-high-noise-frac 0.8 \
        --vision-steps 150
```

Here is the generated image of Leo Messi petting a cute cat:

![Leo Messi petting a cute cat](https://avalan.ai/images/github/vision_text_to_image_generated.webp)

Python:

```python
from avalan.entities import TransformerEngineSettings
from avalan.model.vision.diffusion import TextToImageModel

with TextToImageModel("stabilityai/stable-diffusion-xl-base-1.0", settings=TransformerEngineSettings(refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0", weight_type="fp16")) as model:
    await model(
        "Leo Messi petting a purring tubby cat",
        "example_messi_petting_cat.jpg",
        color_model="RGB",
        image_format="JPEG",
        high_noise_frac=0.8,
        n_steps=150
    )
```
For a runnable script, see [docs/examples/vision_text_to_image.py](docs/examples/vision_text_to_image.py).

#### Text to video

Create an MP4 video from a prompt, using a negative prompt for guardrails and an image as reference:

```sh
echo 'A cute little penguin takes out a book and starts reading it' | \
    avalan model run "Lightricks/LTX-Video-0.9.7-dev" \
        --modality vision_text_to_video \
        --upsampler-model "Lightricks/ltxv-spatial-upscaler-0.9.7" \
        --weight "fp16" \
        --vision-steps 30 \
        --vision-negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
        --vision-inference-steps 10 \
        --vision-reference-path docs/examples/playground/penguin.png \
        --vision-width 832 \
        --vision-height 480 \
        --vision-frames 96 \
        --vision-fps 24 \
        --vision-decode-timestep 0.05 \
        --vision-denoise-strength 0.4 \
        --path example_text_to_video.mp4
```

And here's the generated video:

![A penguin opening a book](https://avalan.ai/images/github/vision_text_to_video_generated.webp)

Python:

```python
from avalan.entities import EngineSettings
from avalan.model.vision.diffusion import TextToVideoModel

with TextToVideoModel("Lightricks/LTX-Video-0.9.7-dev", settings=EngineSettings(upsampler_model_id="Lightricks/ltxv-spatial-upscaler-0.9.7", weight_type="fp16")) as model:
    await model(
        "A cute little penguin takes out a book and starts reading it",
        "worst quality, inconsistent motion, blurry, jittery, distorted",
        "docs/examples/playground/penguin.png",
        "example_text_to_video.mp4",
        steps=30,
        inference_steps=10,
        width=832,
        height=480,
        frames=96,
        fps=24,
        decode_timestep=0.05,
        denoise_strength=0.4
    )
```
For a runnable script, see [docs/examples/vision_text_to_video.py](docs/examples/vision_text_to_video.py).

## Tools

Avalan makes it simple to launch a chat-based agent that can call external tools while streaming tokens. Avalan ships native helpers for `math.calculator`, `graph.*`, `code.run`, `browser.open`, `database.*`, memory, and MCP integrations so agents can reason with numbers, generate charts, execute code, browse the web, and interact with SQL databases from a single prompt.

> [!NOTE]
> Keep a human in the loop by adding `--tools-confirm` when you run an agent. Avalan will ask you to confirm each tool call before it executes, so you retain control over side effects.

### Math toolset (`math.*`)

Use the math toolset whenever your agent needs deterministic arithmetic or algebraic answers. The calculator tool delegates evaluation to SymPy, making it ideal for verifying multi-step computations instead of relying on approximate language model reasoning.

**Available tools**

- `math.calculator(expression: str) -> str`: Evaluate an arithmetic expression (including parentheses and operator precedence) and return the numeric result as a string.

#### Example: `math.calculator`

The example below uses a local 8B LLM, enables recent memory, and loads a calculator tool. The agent begins with a math question and stays open for follow-ups:

```sh
echo "What is (4 + 6) and then that result times 5, divided by 2?" \
  | avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "math.calculator" \
      --memory-recent \
      --run-max-new-tokens 8192 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools \
      --conversation
```

Notice the GPU utilization at the bottom:

![Example use of an ephemeral tool agent with memory](https://github.com/user-attachments/assets/e15cdd4c-f037-4151-88b9-d0acbb22b0ba)

You can give your GPU some breathing room by running the same on a vendor model, like Anthropic:

```sh
echo "What is (4 + 6) and then that result times 5, divided by 2?" \
  | avalan agent run \
      --engine-uri "ai://$ANTHROPIC_API_KEY@anthropic/claude-sonnet-4-6" \
      --tool "math.calculator" \
      --memory-recent \
      --run-max-new-tokens 8192 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools \
      --conversation
```

### Graph toolset (`graph.*`)

Use the graph toolset when an agent should turn structured numbers into static charts that can be embedded in a response or decoded by another tool. Charts render through Matplotlib's headless backend and return JSON-compatible metadata with a base64 data URI. Pass `--tool-graph-file path/to/chart.png` when you also want each rendered chart saved to a local file.

**Available tools**

- `graph.pie(labels: list[str], values: list[float]) -> dict[str, object]`: Create a pie chart from labels and non-negative values.
- `graph.bar(categories: list[str], values: list[float] | None = None, series: dict[str, list[float]] | None = None) -> dict[str, object]`: Create a vertical, horizontal, grouped, or stacked bar chart.
- `graph.line(x_labels: list[str], values: list[float] | None = None, series: dict[str, list[float]] | None = None) -> dict[str, object]`: Create a line chart from ordered labels and one or more numeric series.
- `graph.scatter(x: list[float], y: list[float]) -> dict[str, object]`: Create a scatter plot from paired numeric values.
- `graph.histogram(values: list[float], bins: int = 10) -> dict[str, object]`: Create a histogram from numeric values.

#### Example: `graph.bar`

```sh
echo 'Generate a monthly bar graph for the total revenue from checks successfully matched to their claims for the organization `Example Legal Group`' | \
    avalan agent run \
      --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-5.4" \
      --reasoning-effort xhigh \
      --tool "database" \
      --tool "graph.bar" \
      --tool-database-dsn "postgresql+asyncpg://root:password@localhost:5432/example_app" \
      --tool-graph-file "./monthly-revenue.png" \
      --developer "You are a helpful assistant that answers questions using tools. Inspect the schema first, then query precisely.
Stay read-only." \
      --run-max-new-tokens 25000 \
      --stats \
      --display-tools \
      --display-events
```

### Code toolset (`code.*`)

Reach for the code toolset when the agent should write, execute, or refactor source code in a controlled environment. Execution happens inside a RestrictedPython sandbox and pattern searches are backed by the `ast-grep` CLI, enabling agents to safely prototype logic, manipulate files, or build refactoring plans.

**Available tools**

- `code.run(code: str, *args, **kwargs) -> str`: Execute a snippet that defines a `run` function and return the function result as text, which is useful for testing generated utilities or validating calculations programmatically.
- `code.search.ast.grep(pattern: str, lang: str, rewrite: str | None = None, paths: list[str] | None = None) -> str`: Search or rewrite codebases using structural patterns, helping agents answer "where is this API used?" or propose targeted edits.

#### Example: `code.run`

Below is an agent that leverages the `code.run` tool to execute Python code generated by the model and display the result:

```sh
echo "Create a python function to uppercase a string, split it spaces, and then return the words joined by a dash, and execute the function with the string 'Leo Messi is the greatest footballer of all times'" \
  | avalan agent run \
      --engine-uri 'ai://local/openai/gpt-oss-20b' \
      --backend mlx \
      --tool-format harmony \
      --tool "code.run" \
      --memory-recent \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools \
      --backend mlx
```

### Database toolset (`database.*`)

Connect the database toolset when an agent must inspect schemas, understand query plans, or run SQL against an external data source. Tools share a pooled SQLAlchemy engine, enforce optional read-only policies, and normalize identifier casing so that agents can explore data safely.

When your agent needs live access to data, configure the database toolset. In the example below we point the agent to a Supabase database, and after prompting for sales data we'll see the agent executing `database.tables` and `database.inspect` to understand the schema, before running SQL with `database.run`. For an end-to-end database-to-chart workflow, see the `graph.bar` example above.

> [!IMPORTANT]
> Database sessions are read-only by default (`read_only = true`) and only permit `SELECT` statements unless you relax the policy. Adjust these safeguards with the database tool settings—for example, set `allowed_commands = ["select", "insert"]` (or pass `--tool-database-allowed-commands select,insert` on the CLI) and toggle `read_only` in your agent specification when you need to allow writes.

```sh
echo "Get me revenue per product, sorting by highest selling" | \
    avalan agent run \
      --engine-uri "ai://local/openai/gpt-oss-20b" \
      --backend mlx \
      --tool-format harmony \
      --tool "database" \
      --tool-database-dsn "postgresql+asyncpg://postgres.project_id:password@aws-1-us-east-1.pooler.supabase.com:5432/postgres" \
      --system "Reasoning: high" \
      --developer "You are a helpful assistant that can resolve user data requests using database tools." \
      --stats \
      --display-tools
```

**Available tools**

- `database.count(table_name: str) -> int`: Return the number of rows in a table—handy for quick health checks or progress reporting.
- `database.inspect(table_names: list[str], schema: str | None = None) -> list[Table]`: Describe table columns and foreign keys so the agent can reason about relationships before writing SQL.
- `database.keys(table_name: str, schema: str | None = None) -> list[TableKey]`: Enumerate primary and unique key definitions so the agent understands table-level uniqueness guarantees.
- `database.relationships(table_name: str, schema: str | None = None) -> list[TableRelationship]`: Surface incoming and outgoing foreign key links for a table so the agent can understand join paths and cardinality constraints.
- `database.plan(sql: str) -> QueryPlan`: Request an `EXPLAIN` plan to validate or optimize a generated query.
- `database.run(sql: str) -> list[dict[str, Any]]`: Execute read or write statements (subject to policy) and return result rows for downstream reasoning.
- `database.sample(table_name: str, columns: list[str] | None = None, conditions: str | None = None, order: dict[str, str] | None = None, count: int | None = None) -> list[dict[str, Any]]`: Fetch up to `count` rows (default 10) from a table so agents can preview data, optionally narrowing by columns, SQL conditions, or ordering before crafting more complex queries.
- `database.locks() -> list[DatabaseLock]`: Inspect PostgreSQL, MySQL, and MariaDB lock metadata—including blocking session IDs, lock targets, and whether the lock is granted—to debug contention before running or terminating queries.
- `database.tables() -> dict[str | None, list[str]]`: List tables grouped by schema—useful for schema discovery in unknown databases.
- `database.tasks(running_for: int | None = None) -> list[DatabaseTask]`: Surface long-running queries on PostgreSQL or MySQL so humans can monitor or intervene.
- `database.kill(task_id: str) -> bool`: Cancel a runaway query when safeguards permit it.
- `database.size(table_name: str) -> TableSize`: Summarize how much space a table occupies, including data and index bytes where the backend provides them, so agents can gauge storage usage before recommending optimizations.

### Browser toolset (`browser.*`)

Use the browser toolset to capture live information from the web or intranet sites. The Playwright-backed browser renders pages, converts them to Markdown, and can optionally search the captured content to keep only the most relevant snippets for the agent.

**Available tools**

- `browser.open(url: str) -> str`: Navigate to a URL and return the rendered page in Markdown, optionally narrowed to search results derived from the user prompt.

Tools give agents real-time knowledge. This example uses an 8B model and a browser tool to find avalan's latest release:

```sh
echo "What's avalan's latest release on https://github.com/avalan-ai/avalan/releases" | \
    avalan agent run \
      --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
      --tool "browser.open" \
      --memory-recent \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-events \
      --display-tools \
      --backend mlx
```

You may need to update playwright browser images with `poetry run playwright install`

When using the browser tool to extract knowledge, be mindful of your context window. With OpenAI's gpt-oss-20b, the model processes 7261 input tokens before producing a final response. When browser context search is enabled (using `--tool-browser-search` and `--tool-browser-search-context`), that number decreases to 1443 input tokens, and the response time improves proportionally:

```sh
echo "What's avalan's latest release on https://github.com/avalan-ai/avalan/releases" | \
    avalan agent run \
      --engine-uri 'ai://local/openai/gpt-oss-20b' \
      --tool-format harmony \
      --tool "browser.open" \
      --tool-browser-search \
      --tool-browser-search-context 10 \
      --run-max-new-tokens 1024 \
      --name "Tool" \
      --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
      --stats \
      --display-tools \
      --backend mlx
```

### Memory toolset (`memory.*`)

Add the memory toolset when agents should consult past conversations or long-lived knowledge bases. The tools can retrieve prior user messages, search permanent vector memories, list stored entries, or enumerate available stores so the agent knows where to look.

**Available tools**

- `memory.message.read(search: str) -> str`: Retrieve user-specific context from prior sessions, returning `NOT_FOUND` when no match exists.
- `memory.read(namespace: str, search: str) -> list[PermanentMemoryPartition]`: Fetch chunks of long-term knowledge inside a namespace for grounding responses.
- `memory.list(namespace: str) -> list[Memory]`: Enumerate stored memories in a namespace so the agent can decide which entries to reuse.
- `memory.stores() -> list[PermanentMemoryStore]`: List permanent memory stores available to the agent for broader exploration.

See [Memories](#memories) for sample usage.

### YouTube toolset (`youtube.*`)

Use the YouTube toolset to ground responses in video transcripts—great for summarizing talks or extracting key quotes without manual downloads. Proxy support keeps the integration flexible for restricted networks.

**Available tools**

- `youtube.transcript(video_id: str, languages: Iterable[str] | None = None) -> list[str]`: Fetch ordered transcript snippets for a given video, optionally prioritizing specific languages.

### MCP toolset (`mcp.*`)

Integrate Model Context Protocol (MCP) servers to orchestrate specialized remote tools. The MCP toolset lets avalan agents proxy any MCP-compatible capability via a single tool call.

**Available tools**

- `mcp.call(uri: str, name: str, arguments: dict[str, object] | None) -> list[object]`: Connect to an MCP server and invoke one of its tools with structured arguments, returning the raw MCP responses.

### Search tool (`search_engine.search`)

For quick demos or testing, Avalan also provides a stubbed search tool that illustrates how to wire internet lookups into an agent. Replace its implementation with a real provider to give agents access to live search APIs.

## Reasoning strategies

Avalan supports several reasoning approaches for guiding agents through complex problems.

### Reasoning models

Reasoning models that emit thinking tags are natively supported. Here's OpenAI's gpt-oss 20B solving a simple calculation:

```sh
echo 'What is (4 + 6) and then that result times 5, divided by 2?' | \
    avalan model run 'ai://local/openai/gpt-oss-20b' \
        --max-new-tokens 1024 \
        --backend mlx
```

The response includes the model reasoning, and its final answer:

![OpenAI's reasoning model responding to a math question](https://avalan.ai/images/github/text_generation_reasoning_openai.webp)

Some of them, like `DeepSeek-R1-Distill-Qwen-14B`, assume the model starts thinking without a thinking tag, so we'll use `--start-thinking`:

```sh
echo 'What is (4 + 6) and then that result times 5, divided by 2?' | \
    avalan model run 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B' \
        --temperature 0.6 \
        --max-new-tokens 1024 \
        --start-thinking \
        --backend mlx
```

![DeepSeek's reasoning model responding to a math question](https://avalan.ai/images/github/text_generation_reasoning_deepseek-2.webp)

Nvidia's Nemotron reasoning model solves the same problem easily and doesn't require the `--start-thinking` flag, since it automatically produces think tags. It does so more verbosely, though (**962** output tokens versus DeepSeek's **186** output tokens or OpenAI's more concise **140** tokens), since it detects ambiguity in the `and then that result` part of the prompt and ends up revisiting the essential principles of mathematics, to the point of realizing it's overthinking 🤓

> [!TIP]
> Endless reasoning rants can be stopped by setting `--reasoning-max-new-tokens` to the maximum number of reasoning tokens allowed, and adding `--reasoning-stop-on-max-new-tokens` to finish generation when that limit is reached.

```sh
echo 'What is (4 + 6) and then that result times 5, divided by 2?' | \
    avalan model run "nvidia/OpenReasoning-Nemotron-14B" \
        --weight "bf16" \
        --max-new-tokens 30000 \
        --backend mlx
```

![Nvidia's reasoning model responding to a math question](https://avalan.ai/images/github/text_generation_reasoning_nvidia-2.webp)

When using reasoning models, be mindful of your total token limit. Some reasoning models include limit recommendations on their model cards, like the following model from Z.ai:

```sh
echo 'What is (4 + 6) and then that result times 5, divided by 2?' | \
    avalan model run 'zai-org/GLM-Z1-32B-0414' \
        --temperature 0.6 \
        --top-p .95 \
        --top-k 40 \
        --max-new-tokens 30000 \
        --start-thinking \
        --backend mlx
```

### ReACT

ReACT interleaves reasoning with tool use so an agent can think through steps and take actions in turn.

You can direct an agent to read specific locations for knowledge:

```sh
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
      --display-tools \
      --backend mlx
```

and you'll get the model's interpretation of what Avalan does based on its README.md file on github:

![ReACT browsing tool usage for real-time information](https://avalan.ai/images/github/text_generation_tools_browser.webp)

### Chain-of-Thought

Chain-of-Thought builds sequential reasoning traces to reach an answer for tasks that require intermediate logic.

### Tree-of-Thought

Tree-of-Thought explores multiple branches of reasoning in parallel to select the best path for difficult decisions.

### Plan-and-Reflect

Plan-and-Reflect has the agent outline a plan, act, and then review the results, promoting methodical problem solving.

### Self-Consistency

Self-Consistency samples several reasoning paths and aggregates them to produce more reliable answers.

### Scratchpad-Toolformer

Scratchpad-Toolformer combines an internal scratchpad with learned tool usage to manipulate intermediate results.

### Cascaded Prompting

Cascaded Prompting chains prompts so each step refines the next, ideal for multi-stage instructions.

### Critic-Guided Direction-Following Experts

Critic-Guided Direction-Following Experts use a critic model to guide expert models when strict quality is required.

### Product-of-Experts

Product-of-Experts merges the outputs of several experts to generate answers that benefit from multiple viewpoints.

## Memories

Avalan offers a unified memory API with native implementations for PostgreSQL
(using pgvector), Elasticsearch, AWS Opensearch, and AWS S3 Vectors.

Start a chat session and tell the agent your name. The `--memory-permanent-message` option specifies where messages are stored, `--id` uniquely identifies the agent, and `--participant` sets the user ID:

```sh
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
      --stats \
      --backend mlx
```

Enable persistent memory and the `memory.message.read` tool so the agent can recall earlier messages. It should discover that your name is `Leo` from the previous conversation:

```sh
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
      --stats \
      --backend mlx
```

Agents can use knowledge stores to solve problems. Index the rules of the "Truco" card game directly from a website. The `--dsn` parameter sets the store location and `--namespace` chooses the knowledge namespace:

```sh
avalan memory document index \
    --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
    --dsn "postgresql://root:password@localhost/avalan" \
    --namespace "games.cards.truco" \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "https://trucogame.com/pages/reglamento-de-truco-argentino"
```

Create an agent, give it access to the indexed memory store and the `memory` tool, and your question:

> [!TIP]
> If you rather create a permanent agent, see the equivalent
> [agent_memory.toml](docs/examples/agent_memory.toml) agent definition.

```sh
echo "What does the memory stored in namespace games.cards.truco say about retrucar?" | \
  avalan agent run \
    --engine-uri "ai://local/openai/gpt-oss-20b" \
    --tool-format harmony \
    --tool memory \
    --system "Reasoning: high" \
    --developer "You are an expert assistant that can use the memory tool to answer questions about card games." \
    --memory-recent \
    --memory-engine-model-id "sentence-transformers/all-MiniLM-L6-v2" \
    --memory-engine-max-tokens 500 \
    --memory-engine-overlap 125 \
    --memory-engine-window 250 \
    --memory-permanent "games.cards.truco@postgresql://root:password@localhost/avalan" \
    --run-max-new-tokens 8192 \
    --run-temperature 0.1 \
    --run-top-p 0.9 \
    --run-top-k 40 \
    --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
    --backend mlx
```

And you should get your answer:

```text
<|channel|>analysis<|message|>We need to read memory stored in namespace games.cards.truco. Use memory.read.<|end|><|start|>assistant<|channel|>commentary to=functions.memory.read <|constrain|>json<|message|>{"namespace":"games.cards.truco","search":"retrucar"}<|channel|>analysis<|message|>We need to answer: "What does the memory stored in namespace games.cards.truco say about retrucar?" We have memory read results. We need to summarize what the memory says about retrucar. The memory contains multiple entries. We should extract relevant info: retrucar is the action of raising the bet after a "Truco" call. Only the team that has said "Quiero" can retrucar. The team that has retrucado cannot raise again; only the opposing team can. Also retrucar is only allowed after "Quiero". The order: Truco, Retruco, Vale Cuatro. The team that wants the last bet can retrucar. Also retrucar is only possible by the team that has wanted the last bet. So answer summarizing.<|end|><|start|>assistant<|channel|>final<|message|>In the **games.cards.truco** memory, the rules about **“retrucar”** (the act of raising the stake after a “Truco” call) are summarized as follows:

| What it is | Who can do it | When it can be done | What happens after |
|------------|---------------|---------------------|--------------------|
| **Retruco** – the next level of the bet after “Truco” | **Only the team that has just said “Quiero”** (i.e., the team that accepted the last bet) | Immediately after the opponent’s “Truco” and the team’s “Quiero” | The opposing team must respond with one of the three options: “Quiero”, “No quiero”, or “Retruco” (if they want to raise again). |
| **Restriction** | The team that has already retrucado **cannot raise again**. | The next raise can only come from the **opposing team**. | This is sometimes called “tener el quiero” – the team that has the right to raise next. |
| **Order of calls** | The sequence must be followed: “Truco” → “Retruco” → “Vale Cuatro”. | | |

In short, **retrucar** is the act of raising the bet after a “Truco” call, and it can only be performed by the team that has just accepted the last bet (“Quiero”). Once a team has retrucado, they lose the right to raise again until the other team does.
```

PDF files are natively supported by memories. Let's index an arxiv paper (in its PDF format) from September 2025, considerably after the training date of the model we'll later use to query it:

```sh
avalan memory document index \
    --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
    --dsn "postgresql://root:password@localhost/avalan" \
    --namespace "papers.arxiv" \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "https://arxiv.org/pdf/2509.17325"
```

Now let's ask the memory agent about that paper (notice the description added when defining the memory store with `--memory-permanent`):

```sh
echo "What papers do you have in memory?" | \
  avalan agent run \
    --engine-uri "ai://local/openai/gpt-oss-20b" \
    --tool-format harmony \
    --tool memory \
    --system "Reasoning: high" \
    --developer "You are an expert assistant that can use the memory tool to answer questions about card games." \
    --memory-recent \
    --memory-engine-model-id "sentence-transformers/all-MiniLM-L6-v2" \
    --memory-engine-max-tokens 500 \
    --memory-engine-overlap 125 \
    --memory-engine-window 250 \
    --memory-permanent "papers.arxiv@postgresql://root:password@localhost/avalan,Papers from ARXIV" \
    --run-max-new-tokens 8192 \
    --run-temperature 0.1 \
    --run-top-p 0.9 \
    --run-top-k 40 \
    --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
    --backend mlx
```

And the stored paper is found:

```text
<|channel|>analysis<|message|>The user asks: "What papers do you have in memory?" We need to search memory for stored papers. We have memory functions: memory.list, memory.read, memory.stores. We need to list memory stores, then list memory in a namespace. The user likely wants to know what papers are stored. We need to call memory.stores to see available stores. Then memory.list for each store. But we don't know the namespace. Maybe there's a default namespace like "papers". We can try memory.list with namespace "papers". But we need to check memory.stores first. Let's call memory.stores.<|end|><|start|>assistant<|channel|>commentary to=functions.memory.stores <|constrain|>json<|message|>{}<|channel|>analysis<|message|>We have a memory store "papers.arxiv". We need to list memory stores. The user asked: "What papers do you have in memory?" We should list the papers stored in that memory store. Use memory.list.<|end|><|start|>assistant<|channel|>commentary to=functions.memory.list<|constrain|>json<|message|>{"namespace":"papers.arxiv"}<|channel|>analysis<|message|>We have memory list showing three entries, all same paper. So answer: list the paper(s). Provide details.<|end|><|start|>assistant<|channel|>final<|message|>I have one paper stored in my permanent memory:

| # | Title | Authors (if available) | URL | Summary |
|---|-------|------------------------|-----|---------|
| 1 | **Generalizable End‑to‑End Tool‑Use RL with Synthetic CodeGym** | (not stored in the memory snippet) | https://arxiv.org/pdf/2509.17325 | The paper introduces *CodeGym*, a framework that turns static coding problems into interactive, multi‑turn tool‑use environments. By extracting atomic functions or logic from code, the authors create verifiable, controllable tasks that can be used to train large‑language‑model agents with reinforcement learning. The study shows that models of various sizes and chain‑of‑thought configurations trained in CodeGym generalize well to out‑of‑distribution benchmarks (e.g., an 8.7‑point accuracy gain on the τ‑Bench). The authors provide the CodeGym code on GitHub. |
```

## Agents

You can easily create AI agents from configuration files. Let's create one to handle gettext translations.
Create a file named [agent_gettext_translator.toml](https://github.com/avalan-ai/avalan/blob/main/docs/examples/agent_gettext_translator.toml)
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

```sh
icdiff locale/avalan.pot <(
    cat locale/avalan.pot |
        avalan agent run docs/examples/agent_gettext_translator.toml --quiet
)
```

![diff showing what the AI translator agent modified](https://avalan.ai/images/github/agent_gettext_translator.webp)

There are more agent, NLP, multimodal, audio, and vision examples in the
[docs/examples](docs/examples/README.md)
folder.

### Serving agents

Avalan agents can be exposed over three open protocols: OpenAI-compatible REST endpoints (supporting completions and streaming responses), Model Context Protocol (MCP), and Agent to Agent (A2A) as first-class tools. They are provided by the same `avalan agent serve` process so you can pick what fits your stack today and evolve without lock-in.

> [!TIP]
> Add one or more `--protocol` flags (for example `--protocol openai`) when running `avalan agent serve` to restrict the interfaces you expose without changing your configuration.

All three interfaces support real-time reasoning plus token and tool streaming, letting you observe thoughts, tokens, tool calls, and intermediate results as they happen.

#### OpenAI completion and responses API

Serve your agents on an OpenAI API–compatible endpoint:

```sh
avalan agent serve docs/examples/agent_tool.toml -vvv
```

> [!NOTE]
> Avalan's OpenAI-compatible endpoint supports both the legacy completions API and the newer [Responses API](https://platform.openai.com/docs/guides/migrate-to-responses).

Agents listen on port 9001 by default.

> [!TIP]
> Use `--port` to serve the agent on a different port.

Or build an agent from inline settings and expose its OpenAI API endpoints:

```sh
avalan agent serve \
    --engine-uri "NousResearch/Hermes-3-Llama-3.1-8B" \
    --tool "math.calculator" \
    --memory-recent \
    --run-max-new-tokens 1024 \
    --name "Tool" \
    --role "You are a helpful assistant named Tool, that can resolve user requests using tools." \
    --backend mlx \
    -vvv
```

You can call your tool streaming agent's OpenAI-compatible endpoint just like
the real API; simply change `--base-url`:

```sh
echo "What is (4 + 6) and then that result times 5, divided by 2?" | \
    avalan model run "ai://openai" --base-url "http://localhost:9001/v1"
```
> [!TIP]
> Use `--protocol openai:responses,completion` to enable both OpenAI Responses and Completions endpoints, or narrow the surface by specifying just `responses` or `completion` after the colon.

##### Example: Match a PDF invoice to database records

You can also serve a database-enabled agent and send it a PDF attachment
through the same OpenAI-compatible endpoint. This is useful when the agent
needs to inspect the document, understand your schema, and look up the matching
record in PostgreSQL.

```sh
avalan agent serve \
    --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-5.4" \
    --reasoning-effort xhigh \
    --tool "database" \
    --tool-database-dsn "postgresql+asyncpg://root:password@localhost:5432/invoices_demo" \
    --developer 'You are a helpful assistant that answers questions using the PostgreSQL database tools. Inspect the schema first, then query precisely. Stay read-only. Imported invoices are in table `invoice_import_items` and the customer account reference is stored in field `account_reference`.' \
    --run-max-new-tokens 25000 \
    --protocol openai:responses,completion \
    --host 127.0.0.1 \
    --port 9001 \
    -vvv
```

Now query your agent with a PDF document:

```sh
echo "The attached invoice may match a customer record in the database. Find the matching account and return its account reference ID." \
    | avalan model run "ai://openai" \
        --base-url "http://127.0.0.1:9001/v1" \
        --input-file docs/examples/playground/invoice.pdf
```

Or call the OpenAI Responses endpoint directly with streaming SSE events:

```sh
pdf=docs/examples/playground/invoice.pdf
jq -n \
    --arg filename "${pdf##*/}" \
    --arg data "data:application/pdf;base64,$(base64 < "$pdf" | tr -d '\n')" '
    {
      input: [{
        role: "user",
        content: [
          {
            type: "input_text",
            text: "The attached invoice may match a customer record in the database. Find the matching account and return its account reference ID."
          },
          {
            type: "input_file",
            filename: $filename,
            file_data: $data
          }
        ]
      }],
      stream: true
}' | curl -N "http://127.0.0.1:9001/v1/responses" \
    -H "Content-Type: application/json" \
    -d @-
```

#### MCP server

Avalan also embeds an HTTP MCP server alongside the OpenAI-compatible
endpoints whenever you run `avalan agent serve`. It is mounted at `/mcp` by
default and can be changed with `--mcp-prefix`.

> [!TIP]
> Use the [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) and
> enter your MCP endpoint URL, the value you configured with `--mcp-prefix`
> when running `avalan agent serve` (default: `http://localhost:9001/mcp`).
> Click `Connect`, then `List Tools`, run the tool that appears (it will match
> your `--mcp-name` and `--mcp-description`), and observe the streaming
> notifications and the final response, which includes reasoning and any tool
> calls with their arguments and results.

You can customize the MCP tool identity with `--mcp-name` (defaults to `run`) and `--mcp-description` when running `avalan agent serve`.

> [!TIP]
> Use `--protocol mcp` (optionally along with other `--protocol` flags) to expose only the MCP interface when serving your agent.

#### A2A server

Avalan also embeds an A2A-compatible server alongside the OpenAI-compatible
endpoints whenever you run `avalan agent serve`. It is mounted at `/a2a` by
default and can be configured with `--a2a-prefix`. The A2A surface supports
streaming, including incremental tool calling and intermediate outputs.

> [!TIP]
> Use the [a2a inspector](https://github.com/a2aproject/a2a-inspector) and
> enter your agent card URL, the value you configured with `--a2a-prefix`
> when running `avalan agent serve` (default: `http://localhost:9001/a2a/agent`).
> You can customize the agent identity with `--a2a-name` and
> `--a2a-description`, then observe the streaming notifications, tool calls,
> and final responses.

You can customize the A2A agent identity with `--a2a-name` (defaults to `run`)
and `--a2a-description` when running `avalan agent serve`.

> [!TIP]
> Use `--protocol a2a` (optionally combined with other `--protocol` flags) to expose just the A2A interface for your served agent.

#### Embedding in existing FastAPI apps

If you already run a FastAPI service, reuse the same OpenAI, MCP, or A2A endpoints without spawning a standalone server. Call `avalan.server.register_agent_endpoints` during startup to attach the routers and lifecycle management to your application:

```python
from fastapi import FastAPI
from logging import getLogger

from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.server import register_agent_endpoints

app = FastAPI()
logger = getLogger("my-app")
hub = HuggingfaceHub()

register_agent_endpoints(
    app,
    hub=hub,
    logger=logger,
    specs_path="docs/examples/agent_tool.toml",
    settings=None,
    tool_settings=None,
    mcp_prefix="/mcp",
    openai_prefix="/v1",
    mcp_name="run",
    protocols={"openai": {"responses"}},
)
```

The helper composes with any existing FastAPI lifespan logic, setting up the orchestrator loader only once and wiring the same streaming endpoints that `avalan agent serve` exposes.

#### Proxy agents

The command `agent proxy` serves as a quick way to serve an agent that:

* Wraps a given `--engine-uri`.
* Enables recent message memory.
* Enables persistent message memory (defaulting to pgsql with pgvector.)

For example, to proxy OpenAI's gpt-4o, do:

```sh
avalan agent proxy \
    --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
    --run-max-new-tokens 1024 \
    -v
```

Like `agent serve`, the proxy listens on port 9001 by default.

And you can connect to it from another terminal using `--base-url`:

```sh
echo "What is (4 + 6) and then that result times 5, divided by 2?" | \
    avalan model run "ai://openai" --base-url "http://localhost:9001/v1"
```

## Documentation & Resources

- [docs/examples](docs/examples/README.md) – runnable scripts and sample agent
  configurations.
- [docs/CLI.md](docs/CLI.md) – exhaustive documentation for commands and flags.
- [docs/INSTALL.md](docs/INSTALL.md) – platform-specific installation notes.
- [docs/ai_uri.md](docs/ai_uri.md) – the guide to engine URIs and backend
  selection.
- [docs/tutorials](docs/tutorials) – longer walkthroughs for advanced
  workflows.

## Community & Support

- Join the [Avalan Discord](https://discord.gg/8Eh9TNvk) to ask questions, share workflows, and follow release announcements.
- Browse community answers or ask DeepWiki follow-up questions from the README badge at the top of this page.
- For commercial support, email [avalan@avalan.ai](mailto:avalan@avalan.ai).

## Contributing

We welcome pull requests, issue reports, docs improvements, and new examples.

1. Read the [Code of Conduct](CODE_OF_CONDUCT.md) before you start.
2. Install the development environment with `poetry install --all-extras --with test`.
3. Run `make lint`.
4. Run `poetry run pytest --verbose -s`.

Open a [GitHub issue](https://github.com/avalan-ai/avalan/issues) if you discover bugs or want to propose larger changes.
