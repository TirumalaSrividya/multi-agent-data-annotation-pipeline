"""
Embedding backend used purely for *novelty scoring* in the Sampler agent
(NOT for the classifier itself). TF-IDF is used by default because it is
fast, dependency-light, deterministic and needs no network/model download,
which keeps the pipeline runnable in constrained/offline environments.

The interface is intentionally narrow (`fit`, `transform`) so a
sentence-transformer or any other embedding model can be swapped in later
without changing the Sampler agent.
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
            # fit_transform on the fly if the caller never called fit()
            return self.vectorizer.fit_transform(texts).toarray()
        return self.vectorizer.transform(texts).toarray()


def novelty_scores(candidate_vecs: np.ndarray, reference_vecs: np.ndarray) -> np.ndarray:
    """Returns, for each candidate row, `1 - max_cosine_similarity` against
    the reference set (the already-labelled pool). Higher score = more
    novel / less redundant with what has already been annotated. If the
    reference set is empty, every candidate is maximally novel."""
    if reference_vecs.shape[0] == 0:
        return np.ones(candidate_vecs.shape[0])
    sims = cosine_similarity(candidate_vecs, reference_vecs)
    max_sim = sims.max(axis=1)
    return 1.0 - max_sim
