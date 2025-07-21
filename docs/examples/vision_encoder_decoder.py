#
# EXAMPLE CALLS:
#
# $ poetry run python docs/examples/a01-122-02-00.jpg
#
# $ poetry run python docs/examples/vision_encoder_decoder.py docs/examples/factura-page-1.png "What is the FACTURA number?"

#
from asyncio import run
from avalan.model.vision.decoder import VisionEncoderDecoderModel
from os.path import isfile
from sys import argv, exit


async def example(path: str, question: str | None) -> None:
    print("Loading model... ", end="", flush=True)
    model_id = (
        "naver-clova-ix/donut-base-finetuned-docvqa" if question else
        "microsoft/trocr-base-printed"
    )
    with VisionEncoderDecoderModel(model_id) as vm:
        print(f"DONE. Running image recognition for {path}", flush=True)

        question = (
            f"<s_docvqa><s_question>{question}</s_question><s_answer>"
            if question
            else None
        )

        text = await vm(path, prompt=question or None)

        print(text, flush=True)


if __name__ == "__main__":
    path = (
        argv[1]
        if len(argv) >= 2 and isfile(argv[1])
        else exit(f"Usage: {argv[0]} <valid_file_path> [question]")
    )
    question = (
        argv[2]
        if len(argv) == 3
        else None
    )

    run(example(path, question))
