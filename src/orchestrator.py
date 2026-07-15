"""
Orchestrator
------------
Coordinates the four agents through the two pipelines described in the
problem statement:

  1. Annotation pipeline (active-learning loop):
       Sampler -> Annotator -> Quality Assessor -> (loop until the pool's
       mean confidence >= threshold, the token budget is exhausted, the
       unlabelled pool is empty, or max_cycles is hit).

  2. Automated training pipeline:
       Trainer agent consumes the accepted, quality-assessed pool and
       returns a TrainingReport.

The orchestrator owns the shared `PipelineState` and is the only place
that mutates it, keeping agent code side-effect-free with respect to
global state (each agent only sees the slice of data it needs and returns
new/updated objects).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from src.agents.annotator_agent import AnnotatorAgent
from src.agents.quality_assessor_agent import QualityAssessorAgent
from src.agents.sampler_agent import SamplerAgent
from src.agents.trainer_agent import TrainerAgent
from src.config import Settings
from src.llm_client import LLMClient
from src.schemas import Annotation, PipelineState, Sample, SampleStatus, TrainingReport
from src.utils.token_budget import TokenBudgetTracker

logger = logging.getLogger("orchestrator")


class PipelineOrchestrator:
    def __init__(self, settings: Settings, llm_client: LLMClient | None = None):
        self.settings = settings
        self.llm_client = llm_client or LLMClient.from_settings(settings, label_set=settings.task.label_set)

        self.sampler = SamplerAgent(top_k_neighbors=settings.annotation.novelty_top_k_neighbors)
        self.annotator = AnnotatorAgent(
            llm_client=self.llm_client,
            label_set=settings.task.label_set,
            task_name=settings.task.name,
            max_tokens_per_call=settings.llm.max_tokens_per_call,
        )
        self.quality_assessor = QualityAssessorAgent(
            llm_client=self.llm_client,
            label_set=settings.task.label_set,
            task_name=settings.task.name,
            low_confidence_threshold=settings.quality_assessor.low_confidence_threshold,
            agreement_boost=settings.quality_assessor.agreement_boost,
            disagreement_penalty=settings.quality_assessor.disagreement_penalty,
            max_tokens_per_call=settings.llm.max_tokens_per_call,
        )
        self.trainer = TrainerAgent(
            label_set=settings.task.label_set,
            trainer_config=settings.trainer,
            random_seed=settings.data.random_seed,
        )

        self.budget_tracker = TokenBudgetTracker(settings.annotation.token_budget)
        self.state = PipelineState()
        self.samples_by_id: Dict[str, Sample] = {}

    # ------------------------------------------------------------------
    # Pipeline 1: Annotation / active learning
    # ------------------------------------------------------------------
    def run_annotation_pipeline(self, unlabelled_pool: List[Sample]) -> PipelineState:
        self.state.unlabelled = list(unlabelled_pool)
        self.samples_by_id = {s.id: s for s in unlabelled_pool}
        cfg = self.settings.annotation

        while True:
            self.state.cycle += 1
            logger.info(
                "=== Annotation cycle %d/%d | labelled=%d | unlabelled=%d | %s ===",
                self.state.cycle, cfg.max_cycles, len(self.state.labelled),
                len(self.state.unlabelled), self.budget_tracker,
            )

            if self.state.cycle > cfg.max_cycles:
                logger.info("Reached max_cycles=%d; stopping annotation pipeline.", cfg.max_cycles)
                break
            if not self.state.unlabelled and not self.state.low_confidence:
                logger.info("No unlabelled samples and no pending low-confidence samples left; stopping.")
                break
            if not self.budget_tracker.has_budget(estimated_cost=self.settings.llm.max_tokens_per_call):
                logger.warning("Token budget exhausted at cycle start; stopping annotation pipeline.")
                break

            # 1. Re-review any carried-over low-confidence samples first.
            if self.state.low_confidence:
                self._run_quality_assessment(list(self.state.low_confidence.values()))

            # 2. Sample a fresh batch by novelty, if there's still budget & pool.
            if self.state.unlabelled and self.budget_tracker.has_budget(self.settings.llm.max_tokens_per_call):
                batch = self.sampler.select_batch(
                    unlabelled=self.state.unlabelled,
                    labelled=self.state.labelled,
                    all_samples_by_id=self.samples_by_id,
                    batch_size=cfg.batch_size,
                )
                batch_ids = {s.id for s in batch}
                self.state.unlabelled = [s for s in self.state.unlabelled if s.id not in batch_ids]

                annotations = self.annotator.annotate_batch(batch, self.budget_tracker)
                self.state.total_tokens_used = self.budget_tracker.used

                low, high = self.quality_assessor.filter_low_confidence(annotations)
                for a in high:
                    a.status = SampleStatus.ACCEPTED
                    self.state.labelled[a.sample_id] = a
                if low:
                    self._run_quality_assessment(low)

            mean_conf = self.state.mean_confidence()
            logger.info(
                "End of cycle %d: labelled=%d mean_confidence=%.3f (target=%.2f)",
                self.state.cycle, len(self.state.labelled), mean_conf, cfg.confidence_threshold,
            )

            if self.state.labelled and mean_conf >= cfg.confidence_threshold and not self.state.low_confidence:
                logger.info("Confidence target reached with no pending low-confidence samples; stopping.")
                break

        logger.info(
            "Annotation pipeline finished after %d cycles: %d labelled samples, "
            "%d still low-confidence, %d tokens used.",
            self.state.cycle, len(self.state.labelled), len(self.state.low_confidence), self.budget_tracker.used,
        )
        return self.state

    def _run_quality_assessment(self, low_confidence_annotations: List[Annotation]) -> None:
        accepted, still_low = self.quality_assessor.review_batch(
            low_confidence_annotations, self.samples_by_id, self.budget_tracker
        )
        for a in accepted:
            self.state.labelled[a.sample_id] = a
            self.state.low_confidence.pop(a.sample_id, None)
        self.state.low_confidence = {a.sample_id: a for a in still_low}
        self.state.total_tokens_used = self.budget_tracker.used

    # ------------------------------------------------------------------
    # Pipeline 2: Automated training
    # ------------------------------------------------------------------
    def run_training_pipeline(self) -> Tuple[TrainingReport, object]:
        if not self.state.labelled:
            raise RuntimeError("No labelled samples available. Run the annotation pipeline first.")

        texts, labels = [], []
        for sample_id, annotation in self.state.labelled.items():
            sample = self.samples_by_id.get(sample_id)
            if sample is None:
                continue
            texts.append(sample.text)
            labels.append(annotation.label)

        logger.info("Handing %d quality-assessed samples to the Trainer agent.", len(texts))
        report, best_model = self.trainer.run(texts, labels)
        return report, best_model
