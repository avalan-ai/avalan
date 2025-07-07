#
# EXAMPLE CALL:
#
# PYTHONPATH=src poetry run python docs/examples/audio_text_to_speech.py '[S1] Leo Messi (clears throat) is the greatest football player of all times. [S2] Of course he is! (claps)' example.wav
#

from asyncio import run
from avalan.model.audio import TextToSpeechModel
import os
import sys


async def example(text: str, path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with TextToSpeechModel("nari-labs/Dia-1.6B-0626") as speech:
        print(f"DONE. Generating speech for \"{text}\"", flush=True)

        generated_path = await speech([text], path, max_new_tokens=2560) # corresponds to around ~20s
        print(f"DONE. Speech generated in {generated_path}", flush=True)


if __name__ == "__main__":
    text = (
        sys.argv[1]
        if len(sys.argv) == 3
        else sys.exit(f"Usage: {sys.argv[0]} <text> <file_path>")
    )
    path = (
        sys.argv[2]
        if len(sys.argv) == 3 and not os.path.isfile(sys.argv[2])
        else sys.exit(f"Usage: {sys.argv[0]} <text> <file_path>")
    )
    run(example(text, path))
