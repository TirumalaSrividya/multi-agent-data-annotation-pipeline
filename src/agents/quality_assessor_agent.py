"""
Quality Assessor Agent
-----------------------
Reviews low-confidence annotations and either accepts,
updates, or flags them for further review.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.agents.base import BaseAgent
from src.llm_client import LLMClient
from src.schemas import Annotation, QualityReview, Sample, SampleStatus
from src.utils.token_budget import TokenBudgetTracker

SYSTEM_PROMPT_TEMPLATE = """You are a senior quality assessor reviewing the work of a junior data annotator.

Task: {task_name}
Valid labels (choose EXACTLY one): {label_list}

You will be given a text, the junior annotator's proposed label, their
confidence, and their reasoning. Critically re-evaluate the classification.
You may keep the same label or correct it. Be honest about your own
uncertainty.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"label": "<one of the valid labels>", "confidence": <float 0-1>, "reasoning": "<short reason>"}}
"""

USER_PROMPT_TEMPLATE = """Text: "{text}"

Junior annotator's proposed label: {orig_label}
Junior annotator's confidence: {orig_confidence}
Junior annotator's reasoning: {orig_reasoning}

Re-evaluate and output your final judgement as JSON.
Output:"""


class QualityAssessorAgent(BaseAgent):
    name = "quality_assessor"

    def __init__(
        self,
        llm_client: LLMClient,
        label_set: List[str],
        task_name: str,
        low_confidence_threshold: float = 0.8,
        agreement_boost: float = 0.15,
        disagreement_penalty: float = 0.2,
        max_tokens_per_call: int = 400,
    ):
        super().__init__()
        self.llm_client = llm_client
        self.label_set = label_set
        self.threshold = low_confidence_threshold
        self.agreement_boost = agreement_boost
        self.disagreement_penalty = disagreement_penalty
        self.max_tokens_per_call = max_tokens_per_call
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_name=task_name, label_list=", ".join(label_set)
        )

    def filter_low_confidence(self, annotations: List[Annotation]) -> Tuple[List[Annotation], List[Annotation]]:
        low, high = [], []
        for a in annotations:
            (low if a.confidence < self.threshold else high).append(a)
        return low, high

    def review_batch(
        self,
        low_confidence_annotations: List[Annotation],
        samples_by_id: Dict[str, Sample],
        budget_tracker: TokenBudgetTracker,
    ) -> Tuple[List[Annotation], List[Annotation]]:
        accepted: List[Annotation] = []
        still_low: List[Annotation] = []

        for annotation in low_confidence_annotations:
            sample = samples_by_id.get(annotation.sample_id)
            if sample is None:
                self.logger.warning("No sample found for annotation %s; skipping review.", annotation.sample_id)
                continue

            if not budget_tracker.has_budget(estimated_cost=self.max_tokens_per_call):
                self.logger.warning("Token budget exhausted during QA review; deferring remaining samples.")
                still_low.append(annotation)
                continue

            reviewed = self._review_one(sample, annotation, budget_tracker)
            if reviewed.confidence >= self.threshold:
                reviewed.status = SampleStatus.ACCEPTED
                accepted.append(reviewed)
            else:
                reviewed.status = SampleStatus.UNDER_REVIEW
                still_low.append(reviewed)

        self.logger.info(
            "QA reviewed %d samples -> %d accepted, %d still low-confidence",
            len(low_confidence_annotations), len(accepted), len(still_low),
        )
        return accepted, still_low

    def _review_one(self, sample: Sample, annotation: Annotation, budget_tracker: TokenBudgetTracker) -> Annotation:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            text=sample.text,
            orig_label=annotation.label,
            orig_confidence=annotation.confidence,
            orig_reasoning=annotation.reasoning or "(none given)",
        )
        try:
            response = self.llm_client.complete(
                system=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens_per_call,
                temperature=0.0,
            )
            budget_tracker.consume(response.total_tokens)
            parsed = LLMClient.parse_json_response(response.text)

            reviewer_label = str(parsed["label"]).strip()
            reviewer_confidence = min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0)
            agreed = reviewer_label == annotation.label

            if agreed:
                final_confidence = min(reviewer_confidence + self.agreement_boost, 0.99)
                final_label = annotation.label
            else:
                final_confidence = max(reviewer_confidence - self.disagreement_penalty, 0.0)
                final_label = reviewer_label if reviewer_label in self.label_set else annotation.label

            review = QualityReview(
                sample_id=sample.id,
                original_label=annotation.label,
                final_label=final_label,
                original_confidence=annotation.confidence,
                final_confidence=final_confidence,
                agreed=agreed,
                reasoning=str(parsed.get("reasoning", "")),
                tokens_used=response.total_tokens,
            )

            annotation.label = review.final_label
            annotation.confidence = review.final_confidence
            annotation.reasoning = review.reasoning
            annotation.source = "quality_assessor"
            annotation.tokens_used += review.tokens_used
            annotation.review_count += 1
            annotation.record(
                event="qa_review", agreed=agreed, final_label=final_label, final_confidence=final_confidence
            )
            return annotation

        except Exception as exc:
            self._log_error(f"QA review failed for sample {sample.id}", exc)
            annotation.review_count += 1
            annotation.record(event="qa_review_failed", error=str(exc))
            return annotation
