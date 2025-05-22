from asyncio import run
from avalan.model.nlp.sequence import SequenceClassificationModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with SequenceClassificationModel("distilbert-base-uncased-finetuned-sst-2-english") as sc:
        print("DONE.", flush=True)

        text = "I love Leo Messi"
        sentiment = await sc(text)

        print(f"Text \"{text}\" sentiment is: {sentiment}")

if __name__ == "__main__":
    run(example())

