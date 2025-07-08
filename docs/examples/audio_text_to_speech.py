#
# EXAMPLE CALL:
#
# rm example.wav ; PYTHONPATH=src poetry run python docs/examples/audio_text_to_speech.py '[S1] Leo Messi (clears throat) is the greatest football player of all times. [S2] Of course he is! (claps)' example.wav
#
# EXAMPLE CALL WITH VOICE CLONING:
#
# rm example.wav ; PYTHONPATH=src poetry run python docs/examples/audio_text_to_speech.py "[S1] Leo Messi is the greatest football player of all times." example.wav oprah.wav "[S1] And then I grew up and had the esteemed honor of meeting her. And wasn't that a surprise. Here was this petite, almost delicate lady who was the personification of grace and goodness."
#
#

from asyncio import run
from avalan.model.audio import TextToSpeechModel
from os.path import isfile
from sys import argv


async def example(text: str, path: str, reference_path: str | None, reference_text: str | None) -> None:
    print("Loading model... ", end="", flush=True)
    with TextToSpeechModel("nari-labs/Dia-1.6B-0626") as speech:
        print(
            f"DONE. Generating speech for \"{text}\". " + (
                f"Using {reference_path} as voice reference."
                if reference_path else ""
            )
        , flush=True)

        generated_path = await speech(
            text,
            path,
            reference_path=reference_path,
            reference_text=reference_text,
            max_new_tokens=5120 # 128 tokens correspond to ~1s
        )
        print(f"DONE. Speech generated in {generated_path}", flush=True)


if __name__ == "__main__":
    usage = f"Usage: {argv[0]} <text> <file_path> [<reference_path> <reference_text>]"

    text = (
        argv[1]
        if len(argv) >= 3
        else exit(usage)
    )
    path = (
        argv[2]
        if len(argv) >= 3 and not isfile(argv[2])
        else exit(usage)
    )
    reference_path = (
        argv[3]
        if len(argv) == 5 and isfile(argv[3])
        else None
    )
    reference_text = (
        argv[4]
        if len(argv) == 5
        else None
    )

    run(example(text, path, reference_path, reference_text))
