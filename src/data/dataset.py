"""
Task-agnostic dataset loading. Any CSV with a text column (and optionally
an id column / ground-truth label column for demo purposes) can be plugged
in via config/config.yaml -> task.text_column / task.id_column /
task.label_column, without touching agent code.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.schemas import Sample

logger = logging.getLogger("data.dataset")


def load_unlabelled_pool(
    dataset_path: Path,
    text_column: str,
    id_column: str = "id",
    label_column: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> List[Sample]:
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Point data.dataset_path in "
            f"config.yaml at a CSV with at least a '{text_column}' column."
        )

    df = pd.read_csv(dataset_path)
    if text_column not in df.columns:
        raise ValueError(f"Configured text_column='{text_column}' not found in dataset columns: {list(df.columns)}")

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    if id_column not in df.columns:
        df[id_column] = [f"row_{i}" for i in range(len(df))]

    samples: List[Sample] = []
    for _, row in df.iterrows():
        text = str(row[text_column]).strip()
        if not text or text.lower() == "nan":
            continue
        true_label = str(row[label_column]) if label_column and label_column in df.columns else None
        samples.append(Sample(id=str(row[id_column]), text=text, true_label=true_label))

    logger.info("Loaded %d unlabelled samples from %s", len(samples), dataset_path)
    return samples
