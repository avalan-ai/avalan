from asyncio import run
from avalan.model.vision.image import ImageToTextModel
import os
import sys

async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with ImageToTextModel("salesforce/blip-image-captioning-base") as img:
        print(f"DONE. Running classification for {path}", flush=True)

        caption = await img(path)
        print(caption, flush=True)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)==2 and os.path.isfile(sys.argv[1]) \
           else sys.exit(f"Usage: {sys.argv[0]} <valid_file_path>")
    run(example(path))

