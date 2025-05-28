from asyncio import run
from openai import AsyncOpenAI


async def main(client: AsyncOpenAI) -> None:
    stream = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain LLM distillation",
            }
        ],
        model="gpt-4o",
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


client = AsyncOpenAI(base_url="http://localhost:9001/v1", api_key=None)

run(main(client))
