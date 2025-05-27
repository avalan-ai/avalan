from asyncio import run
from avalan.model.vision.image import VisionEncoderDecoderModel
from os.path import isfile
from sys import argv, exit

async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with VisionEncoderDecoderModel("microsoft/trocr-base-printed") as vm:
        print(f"DONE. Running image recognition for {path}", flush=True)

        text = await vm(path)

        print(text, flush=True)

if __name__ == "__main__":
    path = argv[1] if len(argv)==2 and isfile(argv[1]) \
           else exit(f"Usage: {argv[0]} <valid_file_path>")
    run(example(path))

