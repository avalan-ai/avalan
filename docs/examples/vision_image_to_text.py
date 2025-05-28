from asyncio import run
from avalan.model.vision.image import ImageToTextModel
from os.path import isfile
from sys import argv, exit


async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with ImageToTextModel("salesforce/blip-image-captioning-base") as img:
        print(f"DONE. Running classification for {path}", flush=True)

        caption = await img(path)
        print(caption, flush=True)


if __name__ == "__main__":
    path = (
        argv[1]
        if len(argv) == 2 and isfile(argv[1])
        else exit(f"Usage: {argv[0]} <valid_file_path>")
    )
    run(example(path))
