from asyncio import run
from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.vendor.openai import OpenAIModel
from os import environ


async def example() -> None:
    print("Loading model... ", end="", flush=True)

    api_key = environ["OPENAI_API_KEY"]
    assert api_key, "Need an $OPENAI_API_KEY environment variable set"
    settings = TransformerEngineSettings(access_token=api_key)

    with OpenAIModel("gpt-4o", settings) as lm:
        print("DONE.", flush=True)

        system_prompt = """
            You are Leo Messi, the greatest football/soccer player of all
            times.
        """

        answer = await lm(
            "Who are you?",
            system_prompt=system_prompt,
            settings=GenerationSettings(
                temperature=0.9,
                max_new_tokens=256,
                use_async_generator=False,
            ),
        )
        print(answer)


if __name__ == "__main__":
    run(example())
