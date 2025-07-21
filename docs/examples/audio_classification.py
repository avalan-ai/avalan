from asyncio import run
from avalan.model.audio.classification import AudioClassificationModel
import os
import sys


async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with AudioClassificationModel("superb/hubert-base-superb-er") as rec:
        print(f"DONE. Running audio classification for {path}", flush=True)

        labels = await rec(path)
        print(labels, flush=True)


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) == 2 and os.path.isfile(sys.argv[1])
        else sys.exit(f"Usage: {sys.argv[0]} <valid_file_path>")
    )
    run(example(path))
