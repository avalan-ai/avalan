from asyncio import run
from avalan.model.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.vision.image import ImageTextToTextModel
from os.path import isfile
from sys import argv, exit

async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with ImageTextToTextModel(
        "google/gemma-3-12b-it",
        settings=TransformerEngineSettings(loader_class="gemma3")
    ) as vm:
        print(f"DONE. Running image recognition for {path}", flush=True)

        text = await vm(
            path,
            "Transcribe the text on this image, word for word, "
            "keeping format when possible.",
            settings=GenerationSettings(max_new_tokens=1024)
        )

        print(text, flush=True)

if __name__ == "__main__":
    path = argv[1] if len(argv)==2 and isfile(argv[1]) \
           else exit(f"Usage: {argv[0]} <valid_file_path>")
    run(example(path))

