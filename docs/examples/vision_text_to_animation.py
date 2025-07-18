from asyncio import run
from avalan.entities import TransformerEngineSettings
from avalan.model.vision.animation import TextToAnimationModel
from os.path import isfile
from sys import argv, exit


async def example(prompt: str, path: str) -> None:
    print("Loading model... ", end="", flush=True)
    steps = 4
    with TextToAnimationModel(
        "ByteDance/AnimateDiff-Lightning",
        settings=TransformerEngineSettings(
            base_model_id="stablediffusionapi/mistoonanime-v30",
            checkpoint=f"animatediff_lightning_{steps}step_diffusers.safetensors",
            weight_type="fp16"
        )
    ) as diff:
        print(f"DONE. Running animation generation to {path}", flush=True)

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
