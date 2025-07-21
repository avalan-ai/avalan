from asyncio import run
from avalan.model.audio.generation import AudioGenerationModel
from os.path import isfile
from sys import argv


async def example(prompt: str, path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with AudioGenerationModel("facebook/musicgen-small") as speech:
        print(f"DONE. Generating audio for \"{prompt}\"", flush=True)

        generated_path = await speech(
            text,
            path,
            max_new_tokens=1024
        )
        print(f"DONE. Audio generated in {generated_path}", flush=True)


if __name__ == "__main__":
    usage = f"Usage: {argv[0]} <text> <file_path>"

    text = (
        argv[1]
        if len(argv) == 3
        else exit(usage)
    )
    path = (
        argv[2]
        if len(argv) == 3 and not isfile(argv[2])
        else exit(usage)
    )

    run(example(text, path))
