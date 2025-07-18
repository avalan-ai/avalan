from asyncio import run
from avalan.entities import TransformerEngineSettings
from avalan.model.vision.diffusion import TextToImageModel
from os.path import isfile
from sys import argv, exit


async def example(prompt: str, path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with TextToImageModel(
        "stabilityai/stable-diffusion-xl-base-1.0",
        settings=TransformerEngineSettings(
            refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
            weight_type="fp16"
        )
    ) as diff:
        print(f"DONE. Running image generation to {path}", flush=True)

        caption = await diff(prompt, path)
        print(caption, flush=True)


if __name__ == "__main__":
    usage = f"Usage: {argv[0]} <prompt> <file_path>"
    prompt = argv[1] if len(argv) == 3 else exit(usage)
    path = (
        argv[2]
        if len(argv) == 3 and not isfile(argv[1])
        else exit(usage)
    )
    run(example(prompt, path))
