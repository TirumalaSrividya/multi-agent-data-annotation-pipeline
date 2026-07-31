# Models used by the trainer agent

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, max_size: int = 8000):
        self.max_size = max_size
        self.token2idx: Dict[str, int] = {self.PAD: 0, self.UNK: 1}

    def build(self, texts: List[str]) -> None:
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        most_common = counter.most_common(self.max_size - len(self.token2idx))
        for token, _ in most_common:
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.token2idx.get(tok, self.token2idx[self.UNK]) for tok in tokenize(text)][:max_len]
        ids += [self.token2idx[self.PAD]] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)


class KNNTextClassifier:
    name = "knn"

    def __init__(self, n_neighbors: int = 5):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

    def fit(self, texts: List[str], labels: List[str]) -> None:
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def predict(self, texts: List[str]) -> List[str]:
        X = self.vectorizer.transform(texts)
        return list(self.clf.predict(X))


class _SequenceClassifierBase(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int, rnn_cls):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = rnn_cls(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        output, hidden = self.rnn(emb)
        if isinstance(hidden, tuple):  # LSTM returns (h_n, c_n)
            h_n = hidden[0]
        else:
            h_n = hidden
        last_hidden = h_n[-1]
        return self.fc(self.dropout(last_hidden))


class LSTMTorchModule(_SequenceClassifierBase):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__(vocab_size, embedding_dim, hidden_dim, num_classes, nn.LSTM)


class RNNTorchModule(_SequenceClassifierBase):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__(vocab_size, embedding_dim, hidden_dim, num_classes, nn.RNN)


class TorchSequenceClassifier:
   # Provides a common interface for sequence models
 
    def __init__(
        self,
        kind: str,
        label_set: List[str],
        vocab_size: int = 8000,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        max_seq_len: int = 40,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        assert kind in ("lstm", "rnn")
        self.name = kind
        self.label_set = label_set
        self.label2idx = {l: i for i, l in enumerate(label_set)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        self.vocab = Vocabulary(max_size=vocab_size)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = torch.device(device)

        module_cls = LSTMTorchModule if kind == "lstm" else RNNTorchModule
        self._module_cls = module_cls
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def _build(self) -> None:
        self.model = self._module_cls(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.label_set),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        ids = [self.vocab.encode(t, self.max_seq_len) for t in texts]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _encode_labels(self, labels: List[str]) -> torch.Tensor:
        return torch.tensor([self.label2idx[l] for l in labels], dtype=torch.long, device=self.device)

    def init_vocab(self, texts: List[str]) -> None:
        self.vocab.build(texts)
        self._build()

    def train_one_epoch(self, texts: List[str], labels: List[str]) -> float:
        assert self.model is not None
        self.model.train()
        perm = np.random.permutation(len(texts))
        total_loss = 0.0
        n_batches = 0
        criterion = nn.CrossEntropyLoss()
        for start in range(0, len(texts), self.batch_size):
            idx = perm[start:start + self.batch_size]
            batch_texts = [texts[i] for i in idx]
            batch_labels = [labels[i] for i in idx]
            x = self._encode_batch(batch_texts)
            y = self._encode_labels(batch_labels)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def predict(self, texts: List[str]) -> List[str]:
        assert self.model is not None
        self.model.eval()
        preds: List[str] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                x = self._encode_batch(batch)
                logits = self.model(x)
                batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
                preds.extend(self.idx2label[p] for p in batch_preds)
        return preds


def build_model(name: str, label_set: List[str], config) -> object:
    if name == "knn":
        return KNNTextClassifier(n_neighbors=min(5, max(1, len(label_set))))
    if name in ("lstm", "rnn"):
        return TorchSequenceClassifier(
            kind=name,
            label_set=label_set,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            max_seq_len=config.max_seq_len,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
        )
    raise ValueError(f"Unknown model type: {name}")
