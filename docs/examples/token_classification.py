from asyncio import run
from avalan.model.nlp.token import TokenClassificationModel


async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with TokenClassificationModel("dslim/bert-base-NER") as tc:
        print("DONE.", flush=True)

        text = (
            "Lionel Messi, commonly known as Leo Messi, is an Argentine "
            "professional footballer widely regarded as one of the "
            "greatest football players of all time."
        )

        named_entities = await tc(text, labeled_only=True)
        for token, entity_type in named_entities.items():
            print(f'"{token}": {entity_type}')


if __name__ == "__main__":
    run(example())
