"""
Training routines used by the trainer agent.

Provides model training, evaluation, and early-stopping logic.
"""
from __future__ import annotations

import copy
import logging
from typing import List, Tuple

from src.ml.metrics import compute_eval_result
from src.ml.models import KNNTextClassifier, TorchSequenceClassifier, build_model
from src.schemas import ModelEvalResult

logger = logging.getLogger("ml.train")


def train_and_eval_candidate(
    model_name: str,
    label_set: List[str],
    train_texts: List[str],
    train_labels: List[str],
    eval_texts: List[str],
    eval_labels: List[str],
    trainer_config,
) -> Tuple[object, ModelEvalResult]:
    model = build_model(model_name, label_set, trainer_config)

    if isinstance(model, KNNTextClassifier):
        model.fit(train_texts, train_labels)
        preds = model.predict(eval_texts)
        result = compute_eval_result(model_name, eval_labels, preds, label_set)
        logger.info("[%s] eval accuracy=%.3f macro_f1=%.3f", model_name, result.accuracy, result.macro_f1)
        return model, result

    if isinstance(model, TorchSequenceClassifier):
        return _train_torch_with_early_stopping(model, train_texts, train_labels, eval_texts, eval_labels, trainer_config, label_set)

    raise TypeError(f"Unsupported model type: {type(model)}")


def _train_torch_with_early_stopping(
    model: TorchSequenceClassifier,
    train_texts: List[str],
    train_labels: List[str],
    eval_texts: List[str],
    eval_labels: List[str],
    trainer_config,
    label_set: List[str],
) -> Tuple[TorchSequenceClassifier, ModelEvalResult]:
    model.init_vocab(train_texts)

    best_state = None
    best_result: ModelEvalResult | None = None
    best_macro_f1 = -1.0
    epochs_since_improvement = 0
    early_stopped = False
    epochs_trained = 0

    for epoch in range(1, trainer_config.max_epochs + 1):
        train_loss = model.train_one_epoch(train_texts, train_labels)
        preds = model.predict(eval_texts)
        result = compute_eval_result(model.name, eval_labels, preds, label_set, epochs_trained=epoch)
        epochs_trained = epoch

        logger.info(
            "[%s] epoch %d/%d train_loss=%.4f eval_acc=%.3f eval_macro_f1=%.3f",
            model.name, epoch, trainer_config.max_epochs, train_loss, result.accuracy, result.macro_f1,
        )

        improved = result.macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = result.macro_f1
            best_state = copy.deepcopy(model.model.state_dict())
            best_result = result
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if result.accuracy >= trainer_config.target_accuracy:
            logger.info(
                "[%s] target accuracy %.3f reached at epoch %d -- early stopping.",
                model.name, trainer_config.target_accuracy, epoch,
            )
            best_state = copy.deepcopy(model.model.state_dict())
            best_result = result
            early_stopped = True
            break

        if epochs_since_improvement >= trainer_config.early_stopping_patience:
            logger.info(
                "[%s] no macro-F1 improvement for %d epochs -- early stopping.",
                model.name, epochs_since_improvement,
            )
            early_stopped = True
            break

    if best_state is not None:
        model.model.load_state_dict(best_state)
    if best_result is not None:
        best_result.epochs_trained = epochs_trained
        best_result.early_stopped = early_stopped

    return model, best_result
