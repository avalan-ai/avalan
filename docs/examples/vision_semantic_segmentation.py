from asyncio import run
from avalan.model.vision.segmentation import SemanticSegmentationModel
import os
import sys

async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with SemanticSegmentationModel("nvidia/segformer-b0-finetuned-ade-512-512") as seg:
        print(f"DONE. Running classification for {path}", flush=True)

        labels = await seg(path)
        print(labels, flush=True)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)==2 and os.path.isfile(sys.argv[1]) \
           else sys.exit(f"Usage: {sys.argv[0]} <valid_file_path>")
    run(example(path))

