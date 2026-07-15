from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from ollama import chat

from src.utils.retry import retry_with_backoff

logger = logging.getLogger("llm_client")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        ...


class OllamaProvider(BaseLLMProvider):

    def __init__(self, model: str):
        self.model = model

    def complete(
        self,
        system: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        text = response.message.content

        return LLMResponse(
            text=text,
            input_tokens=0,
            output_tokens=0
        )


class MockProvider(BaseLLMProvider):
    """Deterministic offline provider. Produces a plausible JSON annotation
    by hashing the prompt so the same input always yields the same output,
    which makes tests and demo runs reproducible without any network call
    or API key."""

    def __init__(self, label_set: Optional[List[str]] = None, seed: int = 42):
        self.label_set = label_set or ["World", "Sports", "Business", "Sci/Tech"]
        self.seed = seed

    def complete(self, system: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        h = int(hashlib.sha256((user_prompt + system).encode("utf-8")).hexdigest(), 16)
        rng = random.Random(h ^ self.seed)

        # If asked to re-review (system prompt mentions "quality assessor"),
        # nudge confidence upward to simulate a second, more careful pass.
        is_review = "quality assessor" in system.lower()

        label = self.label_set[h % len(self.label_set)]
        # crude keyword heuristic to make mock labels *somewhat* sensible
        text_lower = user_prompt.lower()
        keyword_map = {
            "Sports": ["match", "team", "goal", "tournament", "coach", "league", "player"],
            "Business": ["market", "stock", "company", "revenue", "economy", "trade", "bank"],
            "Sci/Tech": ["technology", "software", "ai", "space", "research", "device", "app"],
            "World": ["government", "election", "president", "country", "minister", "war", "un "],
        }
        for cand_label, keywords in keyword_map.items():
            if cand_label in self.label_set and any(kw in text_lower for kw in keywords):
                label = cand_label
                break

        base_conf = rng.uniform(0.55, 0.95)
        if is_review:
            base_conf = min(base_conf + 0.15, 0.99)
        confidence = round(base_conf, 2)

        payload = {
            "label": label,
            "confidence": confidence,
            "reasoning": "Mock provider heuristic classification based on keyword overlap.",
        }
        text = json.dumps(payload)
        input_tokens = max(len(user_prompt) // 4, 1)
        output_tokens = max(len(text) // 4, 1)
        return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


class LLMClient:
    """Facade used by agents. Wraps whichever provider is configured."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    @classmethod
    def from_settings(cls, settings, label_set: Optional[List[str]] = None) -> "LLMClient":
        if settings.llm.provider == "mock":
            return cls(MockProvider(label_set=label_set))

        if settings.llm.provider == "ollama":
            return cls(
            OllamaProvider(
                model=settings.llm.model
            )
        )
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")
        

    def complete(self, system: str, user_prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> LLMResponse:
        return self.provider.complete(system, user_prompt, max_tokens, temperature)

    @staticmethod
    def parse_json_response(text: str) -> dict:
        """Robustly extract a JSON object from an LLM response, tolerating
        markdown code fences or minor extra text around the JSON body."""
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0)
        return json.loads(text)
