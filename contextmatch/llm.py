from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class VLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        concurrency: int = 16,
        timeout: float = 180,
        retries: int = 3,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Model commands require the 'openai' package. "
                "Install the project with: python -m pip install -e ."
            ) from exc
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.retries = retries
        self.client = AsyncOpenAI(
            base_url=base_url.rstrip("/") + "/",
            api_key=api_key,
            timeout=timeout,
        )

    async def complete_json(
        self,
        messages: Sequence[dict[str, str]],
        result_type: type[ModelT],
        *,
        temperature: float = 0,
        max_tokens: int = 700,
        thinking: bool = False,
    ) -> ModelT:
        schema_name = result_type.__name__.lower()
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=list(messages),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "schema": result_type.model_json_schema(),
                            },
                        },
                        extra_body={
                            "chat_template_kwargs": {
                                "enable_thinking": thinking,
                            }
                        },
                    )
                content = completion.choices[0].message.content or completion.choices[0].message.reasoning
                print(f"DEBUG CONTENT: finish_reason={completion.choices[0].finish_reason!r} content={content!r}", flush=True)
                if not content:
                    raise ValueError("model returned empty content")
                return result_type.model_validate(json.loads(content))
            except Exception as exc:  # API and validation errors are retriable.
                print(f"DEBUG ERROR: {type(exc).__name__}: {exc}", flush=True)
                last_error = exc
                if attempt + 1 < self.retries:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"model request failed after {self.retries} attempts") from last_error

    async def close(self) -> None:
        await self.client.close()


async def gather_with_timing(coroutines: Sequence) -> tuple[list, float]:
    started = time.perf_counter()
    results = await asyncio.gather(*coroutines)
    return results, time.perf_counter() - started
