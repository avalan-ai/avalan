from asyncio import run
from avalan.model.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.vendor.groq import GroqModel
from os import environ

async def example() -> None:
    print("Loading model... ", end="", flush=True)

    api_key = environ["GROQ_API_KEY"]
    assert api_key, "Need an $GROQ_API_KEY environment variable set"
    settings = TransformerEngineSettings(access_token=api_key)

    with GroqModel("llama3-8b-8192", settings) as lm:
        print("DONE.", flush=True)

        system_prompt = """
            You are Leo Messi, the greatest football/soccer player of all
            times.
        """

        async for token in await lm(
            "Who are you?",
            system_prompt=system_prompt,
            settings=GenerationSettings(temperature=0.9, max_new_tokens=256),
        ):
            print(token, end="", flush=True)

if __name__ == "__main__":
    run(example())
