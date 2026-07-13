from asyncio import run
from os import environ
from sys import stderr

from avalan.entities import (
    GenerationSettings,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryMode,
    TransformerEngineSettings,
)
from avalan.model.nlp.text.vendor.openai import OpenAIModel
from avalan.model.stream import CanonicalStreamItem, StreamItemKind


async def example() -> None:
    print("Loading model... ", end="", flush=True)

    api_key = environ["OPENAI_API_KEY"]
    assert api_key, "Need an $OPENAI_API_KEY environment variable set"
    settings = TransformerEngineSettings(access_token=api_key)

    with OpenAIModel("gpt-5-mini", settings) as lm:
        print("DONE.", flush=True)

        system_prompt = """
            You are Leo Messi, the greatest football/soccer player of all
            times.
        """

        async for item in await lm(
            "Who are you?",
            system_prompt=system_prompt,
            settings=GenerationSettings(
                max_new_tokens=256,
                reasoning=ReasoningSettings(
                    effort=ReasoningEffort.LOW,
                    summary=ReasoningSummaryMode.CONCISE,
                ),
                use_async_generator=True,
            ),
        ):
            assert isinstance(item, CanonicalStreamItem)
            if item.kind is StreamItemKind.REASONING_DELTA:
                assert item.text_delta is not None
                print(item.text_delta, end="", file=stderr, flush=True)
            elif (
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta is not None
            ):
                print(item.text_delta, end="", flush=True)


if __name__ == "__main__":
    run(example())
