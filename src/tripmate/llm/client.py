"""One client for every model: Ollama, llama-server and hosted APIs all speak this protocol."""

import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local",
        reasoning_effort: str = "",
        timeout: float = 600,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Completion:
        """With a schema, decoding is grammar-constrained: the output always parses."""
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        t0 = time.perf_counter()
        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        usage = res.usage
        return Completion(
            text=res.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            seconds=time.perf_counter() - t0,
        )


def serving_llm(s) -> LLM:
    return LLM(s.llm_base_url, s.llm_model, s.llm_api_key)


def teacher_llm(s) -> LLM:
    return LLM(s.teacher_base_url, s.teacher_model, s.teacher_api_key, s.teacher_reasoning_effort)
