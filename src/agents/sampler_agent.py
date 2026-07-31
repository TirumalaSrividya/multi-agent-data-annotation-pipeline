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

        candidate_vecs = self.embedder.transform(candidate_texts)

        if labelled_texts:
            reference_vecs = self.embedder.transform(labelled_texts)
            scores = novelty_scores(candidate_vecs, reference_vecs)
        else:
            scores = self._diversity_scores(candidate_vecs)

        ranked_idx = np.argsort(-scores)[:batch_size]
        selected = [unlabelled[i] for i in ranked_idx]
        self.logger.info(
            "Selected %d/%d samples by novelty (mean score=%.3f)",
            len(selected), len(unlabelled), float(scores[ranked_idx].mean()),
        )
        return selected

    @staticmethod
    def _diversity_scores(vecs: np.ndarray) -> np.ndarray:
        # Calculate diversity scores for the initial batch.
        centroid = vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        if dists.max() > 0:
            dists = dists / dists.max()
        return dists
