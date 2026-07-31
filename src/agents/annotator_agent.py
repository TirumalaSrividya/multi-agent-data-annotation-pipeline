"""
Annotator Agent
---------------
Uses an LLM to label text samples, enforces JSON output, 
tracks token usage, and sends failed annotations for review.
"""

from __future__ import annotations

import json
from typing import List

from src.agents.base import BaseAgent
from src.llm_client import LLMClient
from src.schemas import Annotation, Sample, SampleStatus
from src.utils.token_budget import TokenBudgetTracker

SYSTEM_PROMPT_TEMPLATE = """You are a meticulous data annotator for a text classification task.

Task: {task_name}
Valid labels (choose EXACTLY one): {label_list}

Instructions:
1. Read the text carefully.
2. Pick the single best-fitting label from the valid label list above.
3. Estimate your confidence in that label as a float between 0.0 and 1.0,
   where 1.0 means you are certain and 0.5 means you are essentially guessing.
4. Give a one-sentence reasoning for your choice.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"label": "<one of the valid labels>", "confidence": <float 0-1>, "reasoning": "<short reason>"}}
"""

FEW_SHOT_EXAMPLES = """
Example 1:
Text: "The central bank raised interest rates by 0.5% to curb inflation."
Output: {{"label": "Business", "confidence": 0.94, "reasoning": "Discusses monetary policy and interest rates, a core business/economics topic."}}

Example 2:
Text: "The striker scored a hat-trick in the final minutes to win the championship."
Output: {{"label": "Sports", "confidence": 0.97, "reasoning": "Describes a goal-scoring event in a sporting championship."}}
"""

USER_PROMPT_TEMPLATE = """{few_shot}
Now classify this text:
Text: "{text}"
Output:"""


class AnnotatorAgent(BaseAgent):
    name = "annotator"

    def __init__(self, llm_client: LLMClient, label_set: List[str], task_name: str, max_tokens_per_call: int = 400):
        super().__init__()
        self.llm_client = llm_client
        self.label_set = label_set
        self.task_name = task_name
        self.max_tokens_per_call = max_tokens_per_call
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_name=task_name, label_list=", ".join(label_set)
        )

    def annotate_batch(self, samples: List[Sample], budget_tracker: TokenBudgetTracker) -> List[Annotation]:
        annotations: List[Annotation] = []

        for sample in samples:
            if not budget_tracker.has_budget(estimated_cost=self.max_tokens_per_call):
                self.logger.warning(
                    "Token budget exhausted (%s); stopping annotation batch early. "
                    "%d/%d samples in this batch were annotated.",
                    budget_tracker, len(annotations), len(samples),
                )
                break

            annotation = self._annotate_one(sample, budget_tracker)
            annotations.append(annotation)

        self.logger.info(
            "Annotated %d/%d samples in batch. Budget: %s", len(annotations), len(samples), budget_tracker
        )
        return annotations

    def _annotate_one(self, sample: Sample, budget_tracker: TokenBudgetTracker) -> Annotation:
        user_prompt = USER_PROMPT_TEMPLATE.format(few_shot=FEW_SHOT_EXAMPLES, text=sample.text)
        try:
            response = self.llm_client.complete(
                system=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens_per_call,
                temperature=0.0,
            )
            budget_tracker.consume(response.total_tokens)
            parsed = LLMClient.parse_json_response(response.text)

            label = str(parsed["label"]).strip()
            confidence = float(parsed.get("confidence", 0.5)) 
            confidence = min(max(confidence, 0.0), 1.0)   # Clamp confidence to the valid range.
            
            if label not in self.label_set:
            
                self.logger.warning("Sample %s: label '%s' not in label set %s", sample.id, label, self.label_set)
                confidence = min(confidence, 0.4)  # Reduce confidence when the model returns an invalid label.

            annotation = Annotation(
                sample_id=sample.id,
                label=label,
                confidence=confidence,
                reasoning=str(parsed.get("reasoning", "")),
                source="annotator",
                tokens_used=response.total_tokens,
                status=SampleStatus.ANNOTATED,
            )
            annotation.record(event="annotated", label=label, confidence=confidence)
            return annotation

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            self._log_error(f"Failed to parse annotation for sample {sample.id}", exc)
            return Annotation(
                sample_id=sample.id,
                label=self.label_set[0],
                confidence=0.0,
                reasoning=f"annotation_error: {exc}",
                source="annotator",
                tokens_used=0,
                status=SampleStatus.UNDER_REVIEW,
            )
        except Exception as exc: 
            self._log_error(f"LLM call failed for sample {sample.id} after retries", exc)
            return Annotation(
                sample_id=sample.id,
                label=self.label_set[0],
                confidence=0.0,
                reasoning=f"llm_call_failed: {exc}",
                source="annotator",
                tokens_used=0,
                status=SampleStatus.UNDER_REVIEW,
            )
