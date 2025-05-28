from asyncio import run
from avalan.model.nlp.question import QuestionAnsweringModel


async def example() -> None:
    print("Loading model... ", end="", flush=True)
    with QuestionAnsweringModel("deepset/roberta-base-squad2") as qa:
        print("DONE.", flush=True)

        add_knowledge = False
        ask_knowledge = False
        ask_professions = False

        context = (
            "Lionel Messi, known as Leo Messi, is an Argentine "
            "professional footballer widely regarded as one of the "
            "greatest football players of all time"
        )

        if add_knowledge:
            context += " and has acted on multiple TV ads"

        question = (
            "Is Leo an actor?"
            if ask_knowledge
            else "What sport does Leo play?"
            if ask_professions
            else "Who is Leo?"
        )

        answer = await qa(question, context=context)
        print(question)
        print("-" * 12)
        print(answer.strip() if answer else "no idea :)")


if __name__ == "__main__":
    run(example())
