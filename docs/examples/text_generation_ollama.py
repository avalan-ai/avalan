from asyncio import run
from avalan.model.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.vendor.ollama import OllamaModel


async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with OllamaModel("llama3", TransformerEngineSettings()) as lm:
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
