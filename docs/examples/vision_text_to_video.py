#
# RUN:
#
#  pr python docs/examples/vision_text_to_video.py "A cute little penguin takes out a book and starts reading it" "worst quality, inconsistent motion, blurry, jittery, distorted" penguin.png example_text_to_video.mp4
#

from asyncio import run
from avalan.entities import EngineSettings
from avalan.model.vision.diffusion import TextToVideoModel
from os.path import isfile
from sys import argv, exit


async def example(
    prompt: str,
    negative_prompt: str,
    reference_path: str,
    path: str
) -> None:
    print("Loading model... ", end="", flush=True)
    with TextToVideoModel(
        "Lightricks/LTX-Video-0.9.7-dev",
        settings=EngineSettings(
            upsampler_model_id="Lightricks/ltxv-spatial-upscaler-0.9.7",
            weight_type="fp16"
        )
    ) as diff:
        print(f"DONE. Running video generation to {path}", flush=True)

        caption = await diff(prompt, negative_prompt, reference_path, path)
        print(caption, flush=True)


if __name__ == "__main__":
    usage = f"Usage: {argv[0]} <prompt> <negative_prompt> <reference_path> <file_path>"
    prompt = argv[1] if len(argv) == 5 else exit(usage)
    negative_prompt = argv[2] if len(argv) == 5 else exit(usage)
    reference_path = (
        argv[3]
        if len(argv) == 5 and isfile(argv[3])
        else exit(usage)
    )
    path = (
        argv[4]
        if len(argv) == 5 and not isfile(argv[4])
        else exit(usage)
    )
    run(example(prompt, negative_prompt, reference_path, path))
