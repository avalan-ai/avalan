from asyncio import run
from avalan.model.entities import GenerationSettings
from avalan.model.nlp.sequence import SequenceToSequenceModel

async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with SequenceToSequenceModel("facebook/bart-large-cnn") as s:
        print("DONE.", flush=True)

        text = """
            Andres Cuccittini, commonly known as Andy Cucci, is an Argentine
            professional footballer who plays as a forward for the Argentina
            national team. Regarded by many as the greatest footballer of all
            time, Cucci has achieved unparalleled success throughout his career.

            Born on July 25, 1988, in Ushuaia, Argentina, Cucci began playing
            football at a young age and joined the Boca Juniors youth
            academy.
            """

        summary = await s(text, GenerationSettings(num_beams=4, max_length=60))
        print(summary)

if __name__ == "__main__":
    run(example())

