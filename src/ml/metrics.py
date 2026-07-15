"""Per-class and macro classification metrics, used identically for eval
and test splits so numbers are directly comparable."""
from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import precision_recall_fscore_support

from src.schemas import ClassMetrics, ModelEvalResult


def compute_eval_result(
    model_name: str,
    y_true: List[str],
    y_pred: List[str],
    label_names: List[str],
    epochs_trained: int = 0,
    early_stopped: bool = False,
) -> ModelEvalResult:
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, zero_division=0
    )
    per_class: Dict[str, ClassMetrics] = {}
    for i, label in enumerate(label_names):
        per_class[label] = ClassMetrics(
            precision=float(precisions[i]),
            recall=float(recalls[i]),
            f1=float(f1s[i]),
            support=int(supports[i]),
        )

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average="macro", zero_division=0
    )

    return ModelEvalResult(
        model_name=model_name,
        accuracy=accuracy,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        per_class=per_class,
        epochs_trained=epochs_trained,
        early_stopped=early_stopped,
    )
