"""
Sampler agent.
Selects the most informative samples for annotation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.agents.base import BaseAgent
from src.data.embeddings import TfidfEmbeddingBackend, novelty_scores
from src.schemas import Annotation, Sample


class SamplerAgent(BaseAgent):
    name = "sampler"

    def __init__(self, top_k_neighbors: int = 5):
        super().__init__()
        self.top_k_neighbors = top_k_neighbors
        self.embedder = TfidfEmbeddingBackend()

    def select_batch(
        self,
        unlabelled: List[Sample],
        labelled: dict[str, Annotation],
        all_samples_by_id: dict[str, Sample],
        batch_size: int,
    ) -> List[Sample]:
        if not unlabelled:
            return []
        if batch_size >= len(unlabelled):
            self.logger.info("Batch size >= remaining pool; selecting all %d remaining samples.", len(unlabelled))
            return list(unlabelled)

        candidate_texts = [s.text for s in unlabelled]
        # Fit the vectorizer on both labelled and unlabelled samples.
        labelled_texts = [all_samples_by_id[sid].text for sid in labelled.keys() if sid in all_samples_by_id]
        self.embedder.fit(candidate_texts + labelled_texts)

        candidate_vectors = self.embedder.transform(candidate_texts)

        if labelled_texts:
            reference_vectors = self.embedder.transform(labelled_texts)
            scores = novelty_scores(candidate_vectors, reference_vectors)
        else:
            scores = self._diversity_scores(candidate_vectors)

        ranked_indices = np.argsort(-scores)[:batch_size]
        selected = [unlabelled[i] for i in ranked_indices]
        self.logger.info(
            "Selected %d/%d samples by novelty (mean score=%.3f)",
            len(selected), len(unlabelled), float(scores[ranked_indices].mean()),
        )
        return selected

    @staticmethod
    def _diversity_scores(vectors: np.ndarray) -> np.ndarray:
        # Calculate diversity scores for the initial batch.
        centroid = vectors.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        if distances.max() > 0:
            distances = distances / distances.max()
        return distances
