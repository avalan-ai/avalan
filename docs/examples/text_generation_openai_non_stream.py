from asyncio import run
from os import environ

from avalan.entities import (
    GenerationSettings,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryMode,
    TransformerEngineSettings,
)
from avalan.model.nlp.text.vendor.openai import OpenAIModel


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

        answer = await lm(
            "Who are you?",
            system_prompt=system_prompt,
            settings=GenerationSettings(
                max_new_tokens=256,
                reasoning=ReasoningSettings(
                    effort=ReasoningEffort.LOW,
                    summary=ReasoningSummaryMode.CONCISE,
                ),
                use_async_generator=False,
            ),
        )
        # The compatibility text view contains only the final answer. Consume
        # canonical_stream() instead when structured summary items are needed.
        print(await answer.to_str())


if __name__ == "__main__":
    run(example())
