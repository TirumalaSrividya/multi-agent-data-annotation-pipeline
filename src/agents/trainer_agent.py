"""
Trainer agent.

Trains candidate models, evaluates them, and selects the best model.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

from src.agents.base import BaseAgent
from src.ml.metrics import compute_eval_result
from src.ml.train import train_and_eval_candidate
from src.schemas import ModelEvalResult, TrainingReport

logger = logging.getLogger("agent.trainer")


class TrainerAgent(BaseAgent):
    name = "trainer"

    def __init__(self, label_set: List[str], trainer_config, random_seed: int = 42):
        super().__init__()
        self.label_set = label_set
        self.config = trainer_config
        self.random_seed = random_seed

    def split(
        self, texts: List[str], labels: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        # Split the dataset while preserving label distribution.
        rng = random.Random(self.random_seed)
        by_label: Dict[str, List[int]] = {}
        for index, label in enumerate(labels):
            by_label.setdefault(label, []).append(index)

        train_idx, eval_idx, test_idx = [], [], []
        for label, idxs in by_label.items():
            rng.shuffle(idxs)
            n = len(idxs)
            n_train = max(int(n * self.config.train_split), 1) if n > 2 else max(n - 2, 1)
            n_eval = max(int(n * self.config.eval_split), 1) if n - n_train > 1 else max(n - n_train - 1, 0)
            train_idx += idxs[:n_train]
            eval_idx += idxs[n_train:n_train + n_eval]
            test_idx += idxs[n_train + n_eval:]

        def gather(idxs: List[int]) -> Tuple[List[str], List[str]]:
            return [texts[i] for i in idxs], [labels[i] for i in idxs]

        train_texts, train_labels = gather(train_idx)
        eval_texts, eval_labels = gather(eval_idx)
        test_texts, test_labels = gather(test_idx)

        self.logger.info(
            "Split labelled pool of %d into train=%d eval=%d test=%d",
            len(texts), len(train_texts), len(eval_texts), len(test_texts),
        )
        return train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels

    def run(self, texts: List[str], labels: List[str]) -> Tuple[TrainingReport, object]:
        if len(set(labels)) < 2:
            raise ValueError("Trainer requires at least 2 distinct labels in the labelled pool.")

        train_texts, train_labels, eval_texts, eval_labels, test_texts, test_lables = self.split(texts, labels)
        if not eval_texts or not test_texts:
            raise ValueError(
                "Labelled pool too small to form non-empty eval/test splits. "
                "Run more annotation cycles first."
            )

        candidate_results: List[ModelEvalResult] = []
        trained_models: Dict[str, object] = {}

        for model_name in self.config.candidate_models:
            self.logger.info("Training candidate model: %s", model_name)
            try:
                model, eval_result = train_and_eval_candidate(
                    model_name, self.label_set, train_texts, train_labels, eval_texts, eval_lables, self.config
                )
                candidate_results.append(eval_result)
                trained_models[model_name] = model
            except Exception as exc:
                self._log_error(f"Training failed for candidate '{model_name}'", exc)

        if not candidate_results:
            raise RuntimeError("All candidate models failed to train.")

        best_result = max(candidate_results, key=lambda r: r.macro_f1)
        best_model = trained_models[best_result.model_name]
        self.logger.info(
            "Best candidate on eval: %s (macro_f1=%.3f, accuracy=%.3f)",
            best_result.model_name, best_result.macro_f1, best_result.accuracy,
        )

        test_preds = best_model.predict(test_texts)
        test_result = compute_eval_result(best_result.model_name, test_labels, test_preds, self.label_set)
        target_reached = test_result.accuracy >= self.config.target_accuracy
        decision = "STOP" if target_reached else "REQUEST_MORE_LABELS"

        self.logger.info(
            "Test result for %s: accuracy=%.3f (target=%.3f, reached=%s)",
            best_result.model_name, test_result.accuracy, self.config.target_accuracy, target_reached,
        )

        self.logger.info(
            "Trainer decision: %s",
            decision,
        )

        return TrainingReport(
            candidate_results_eval=candidate_results,
            best_model_name=best_result.model_name,
            test_result=test_result,
            target_accuracy=self.config.target_accuracy,
            target_reached=target_reached,
            decision=decision,
            train_size=len(train_texts),
            eval_size=len(eval_texts),
            test_size=len(test_texts),
        ), best_model
