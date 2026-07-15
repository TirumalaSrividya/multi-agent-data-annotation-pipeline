"""
Sampler (Data Selection) Agent
-------------------------------
Chooses which unlabelled samples the Annotator should label next, based on
how *novel* each candidate is relative to the samples already in the
labelled pool. This implements task 1.a: "Annotator agent chooses the
samples to label based on how new the sample is to already existing
samples" -- selection is factored out into its own agent to keep concerns
clean (selection strategy vs. LLM prompting live in different files).

Strategy: TF-IDF vectorise all unlabelled candidates and the current
labelled pool, score each candidate by `1 - max_cosine_similarity` to the
labelled pool (novelty_scores in src/data/embeddings.py), and return the
top-k most novel samples. When the labelled pool is empty (first cycle),
every sample is equally novel, so we fall back to a diversity-maximising
greedy pick (farthest-point sampling) among the candidates themselves so
the very first batch is not a redundant cluster.
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
        # Fit the vectoriser on candidates + labelled text so the vocab
        # covers both sets consistently.
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
        """Farthest-point-style diversity score used only when there is no
        labelled pool yet: score each point by its distance to the overall
        centroid, so the first batch spreads across the topic space instead
        of clustering around one dominant theme."""
        centroid = vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        if dists.max() > 0:
            dists = dists / dists.max()
        return dists
