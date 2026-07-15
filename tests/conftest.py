import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.config import (
    AnnotationConfig,
    ArtifactsConfig,
    DataConfig,
    LLMConfig,
    LoggingConfig,
    QualityAssessorConfig,
    Settings,
    TaskConfig,
    TrainerConfig,
)
from src.llm_client import LLMClient, MockProvider

LABEL_SET = ["World", "Sports", "Business", "Sci/Tech"]


@pytest.fixture
def label_set():
    return list(LABEL_SET)


@pytest.fixture
def mock_llm_client():
    return LLMClient(MockProvider(label_set=LABEL_SET))


@pytest.fixture
def test_settings():
    return Settings(
        anthropic_api_key=None,
        task=TaskConfig(name="news_topic_classification", label_set=LABEL_SET, text_column="text", label_column="label", id_column="id"),
        data=DataConfig(dataset_path="data/sample_news.csv", unlabelled_sample_size=60, random_seed=42),
        llm=LLMConfig(provider="mock", model="mock-model", max_tokens_per_call=200),
        annotation=AnnotationConfig(batch_size=8, token_budget=5000, confidence_threshold=0.8, max_cycles=3),
        quality_assessor=QualityAssessorConfig(low_confidence_threshold=0.8),
        trainer=TrainerConfig(
            candidate_models=["knn"], train_split=0.7, eval_split=0.15, test_split=0.15,
            target_accuracy=0.6, max_epochs=3, early_stopping_patience=2, batch_size=8,
        ),
        logging=LoggingConfig(level="WARNING", log_dir="artifacts/logs", log_file="test.log"),
        artifacts=ArtifactsConfig(),
    )
