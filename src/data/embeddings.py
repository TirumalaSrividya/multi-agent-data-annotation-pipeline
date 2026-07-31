"""
Embedding utilities for sample selection.
"""
from __future__ import annotations

from typing import List, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingBackend(Protocol):
    def fit(self, texts: List[str]) -> None: ...
    def transform(self, texts: List[str]) -> np.ndarray: ...


class TfidfEmbeddingBackend:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Fit the vectorizer if it has not been initialized.
            return self.vectorizer.fit_transform(texts).toarray()
        return self.vectorizer.transform(texts).toarray()


def novelty_scores(candidate_vecs: np.ndarray, reference_vecs: np.ndarray) -> np.ndarray:
    # Calculate novelty scores for candidate samples.
    if reference_vecs.shape[0] == 0:
        return np.ones(candidate_vecs.shape[0])
    sims = cosine_similarity(candidate_vecs, reference_vecs)
    max_sim = sims.max(axis=1)
    return 1.0 - max_sim
