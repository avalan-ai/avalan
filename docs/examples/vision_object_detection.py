from asyncio import run
from avalan.model.vision.detection import ObjectDetectionModel
import os
import sys


async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with ObjectDetectionModel("facebook/detr-resnet-50") as od:
        print(f"DONE. Running classification for {path}", flush=True)

        for entity in await od(path):
            print(entity, flush=True)


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) == 2 and os.path.isfile(sys.argv[1])
        else sys.exit(f"Usage: {sys.argv[0]} <valid_file_path>")
    )
    run(example(path))
