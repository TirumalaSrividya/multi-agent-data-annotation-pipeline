"""
Typed data contracts shared by every agent. Keeping these in one place is
what gives the multi-agent system a clean, well-defined state flow: every
agent consumes and returns one of these models instead of ad-hoc dicts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SampleStatus(str, Enum):
    UNLABELLED = "unlabelled"
    ANNOTATED = "annotated"          # labelled by Annotator, not yet reviewed
    UNDER_REVIEW = "under_review"    # flagged low-confidence, sent to QA
    ACCEPTED = "accepted"            # confidence >= threshold, final
    REJECTED = "rejected"            # QA could not resolve confidently, requeued


@dataclass
class Sample:
    id: str
    text: str
    true_label: Optional[str] = None  # only present for demo/eval datasets


@dataclass
class Annotation:
    sample_id: str
    label: str
    confidence: float
    reasoning: str = ""
    source: str = "annotator"        # "annotator" | "quality_assessor"
    tokens_used: int = 0
    review_count: int = 0
    status: SampleStatus = SampleStatus.ANNOTATED
    history: List[Dict] = field(default_factory=list)

    def record(self, **kwargs) -> None:
        snapshot = {"ts": time.time(), **kwargs}
        self.history.append(snapshot)


@dataclass
class QualityReview:
    sample_id: str
    original_label: str
    final_label: str
    original_confidence: float
    final_confidence: float
    agreed: bool
    reasoning: str = ""
    tokens_used: int = 0


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ModelEvalResult:
    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, ClassMetrics]
    epochs_trained: int = 0
    early_stopped: bool = False


@dataclass
class TrainingReport:
    candidate_results_eval: List[ModelEvalResult]
    best_model_name: str
    test_result: ModelEvalResult
    target_accuracy: float
    target_reached: bool
    decision: str    
    train_size: int
    eval_size: int
    test_size: int


@dataclass
class PipelineState:
    """Mutable state threaded through the active-learning (annotation) loop."""
    unlabelled: List[Sample] = field(default_factory=list)
    labelled: Dict[str, Annotation] = field(default_factory=dict)   # accepted, final
    pending: Dict[str, Annotation] = field(default_factory=dict)    # annotated, not yet reviewed
    low_confidence: Dict[str, Annotation] = field(default_factory=dict)
    cycle: int = 0
    total_tokens_used: int = 0

    def mean_confidence(self) -> float:
        all_conf = [a.confidence for a in self.labelled.values()]
        if not all_conf:
            return 0.0
        return sum(all_conf) / len(all_conf)
