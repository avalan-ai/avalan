from asyncio import run
from avalan.model.audio import SpeechRecognitionModel
import os
import sys


# audio should have a sample rate of 16000. If you need to change the sample
# rate, `brew install sox` and then `sox input.wav -r 16000 output.wav`
async def example(path: str) -> None:
    print("Loading model... ", end="", flush=True)
    with SpeechRecognitionModel("facebook/wav2vec2-base-960h") as rec:
        print(f"DONE. Running speech recognition for {path}", flush=True)

        speech = await rec(path, sampling_rate=16000)
        print(speech, flush=True)


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) == 2 and os.path.isfile(sys.argv[1])
        else sys.exit(f"Usage: {sys.argv[0]} <valid_file_path>")
    )
    run(example(path))
