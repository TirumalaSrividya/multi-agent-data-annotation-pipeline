"""
Centralised configuration management.

Loads config/config.yaml, allows environment-variable overrides (useful for
secrets like ANTHROPIC_API_KEY and for CI/containerised deployments), and
exposes a single immutable `Settings` object used across the whole pipeline.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TaskConfig(BaseModel):
    name: str
    label_set: List[str]
    text_column: str
    label_column: Optional[str] = None
    id_column: str = "id"


class DataConfig(BaseModel):
    dataset_path: str
    unlabelled_sample_size: int = 400
    random_seed: int = 42


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.2"
    max_tokens_per_call: int = 400
    temperature: float = 0.0
    request_timeout_s: int = 60
    max_retries: int = 5
    backoff_base_s: float = 1.5



class AnnotationConfig(BaseModel):
    batch_size: int = 16
    token_budget: int = 60000
    confidence_threshold: float = 0.8
    max_cycles: int = 6
    novelty_top_k_neighbors: int = 5


class QualityAssessorConfig(BaseModel):
    low_confidence_threshold: float = 0.8
    max_reviews_per_cycle: int = 200
    agreement_boost: float = 0.15
    disagreement_penalty: float = 0.2


class TrainerConfig(BaseModel):
    candidate_models: List[str] = Field(default_factory=lambda: ["knn", "lstm", "rnn"])
    train_split: float = 0.7
    eval_split: float = 0.15
    test_split: float = 0.15
    target_accuracy: float = 0.85
    max_epochs: int = 15
    early_stopping_patience: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    embedding_dim: int = 64
    hidden_dim: int = 64
    vocab_size: int = 8000
    max_seq_len: int = 40


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "artifacts/logs"
    log_file: str = "pipeline.log"


class ArtifactsConfig(BaseModel):
    output_dir: str = "artifacts"
    labelled_pool_file: str = "labelled_pool.csv"
    training_report_file: str = "training_report.json"
    best_model_file: str = "best_model"


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        extra="ignore"
    )

    task: TaskConfig
    data: DataConfig
    llm: LLMConfig
    annotation: AnnotationConfig
    quality_assessor: QualityAssessorConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    artifacts: ArtifactsConfig

    def resolve(self, relative_path: str) -> Path:
        p = Path(relative_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _apply_dotted_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply CLI-style `section.key=value` overrides onto a nested dict."""
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node: Any = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


@lru_cache(maxsize=1)
def get_settings(config_path: Optional[str] = None) -> Settings:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    raw = _load_yaml(path)
    return Settings(**raw)


def load_settings_with_overrides(config_path: Optional[str] = None, overrides: Optional[dict] = None) -> Settings:
    """Non-cached loader used by the CLI when --set overrides are passed."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    raw = _load_yaml(path)
    if overrides:
        raw = _apply_dotted_overrides(raw, overrides)
    return Settings(**raw)
