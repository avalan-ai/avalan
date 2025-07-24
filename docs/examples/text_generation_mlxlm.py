from asyncio import run
from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.mlxlm import MlxLmModel


async def example() -> None:
    print("Loading model... ", end="", flush=True)

    settings = TransformerEngineSettings()
    with MlxLmModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", settings) as lm:
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
