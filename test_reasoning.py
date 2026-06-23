import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1/", api_key="local")
    completion = await client.chat.completions.create(model="qwen3-14b-awq", messages=[{"role": "user", "content": "What is 2+2? Think step by step."}], temperature=0.1, max_tokens=200, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    msg = completion.choices[0].message
    print("All fields:", msg.model_dump())

asyncio.run(main())