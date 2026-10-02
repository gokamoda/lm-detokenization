"""Token and bigram counts of a corpus (used for T^e, Var(e) and the AUROC).

As in the published frequency.py, texts are tokenized without special tokens
and bigrams are counted within each document. Bigram (a, b) is counted at
row a (prefix), column b (suffix) of a sparse vocabulary x vocabulary matrix.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm


@dataclass
class Counts:
    tokens: np.ndarray  # [vocab], int64
    bigrams: sparse.csr_matrix  # [vocab, vocab], int64

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "tokens.npy", self.tokens)
        sparse.save_npz(directory / "bigrams.npz", self.bigrams)

    @classmethod
    def load(cls, directory: Path) -> "Counts":
        return cls(
            tokens=np.load(directory / "tokens.npy"),
            bigrams=sparse.load_npz(directory / "bigrams.npz").tocsr(),
        )


class _BigramAccumulator:
    def __init__(self, vocab_size: int, flush_every: int):
        self.vocab_size = vocab_size
        self.flush_every = flush_every
        self.pending: list[np.ndarray] = []
        self.pending_size = 0
        self.total = sparse.csr_matrix((vocab_size, vocab_size), dtype=np.int64)

    def add(self, ids: np.ndarray) -> None:
        if len(ids) < 2:
            return
        keys = ids[:-1].astype(np.int64) * self.vocab_size + ids[1:]
        self.pending.append(keys)
        self.pending_size += len(keys)
        if self.pending_size >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        keys, counts = np.unique(np.concatenate(self.pending), return_counts=True)
        chunk = sparse.csr_matrix(
            (
                counts.astype(np.int64),
                (keys // self.vocab_size, keys % self.vocab_size),
            ),
            shape=(self.vocab_size, self.vocab_size),
        )
        self.total = self.total + chunk
        self.pending, self.pending_size = [], 0


def count(
    texts: Iterable[str],
    tokenizer,
    vocab_size: int,
    batch_size: int = 1000,
    flush_every: int = 200_000_000,
    total: int | None = None,
) -> Counts:
    tokens = np.zeros(vocab_size, dtype=np.int64)
    bigrams = _BigramAccumulator(vocab_size, flush_every)
    batch: list[str] = []

    def process(batch: list[str]) -> None:
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for token_ids in encoded:
            ids = np.asarray(token_ids, dtype=np.int64)
            tokens[:] += np.bincount(ids, minlength=vocab_size)
            bigrams.add(ids)

    for text in tqdm(texts, total=total, desc="counting", mininterval=10.0):
        batch.append(text)
        if len(batch) == batch_size:
            process(batch)
            batch = []
    if batch:
        process(batch)
    bigrams.flush()
    return Counts(tokens=tokens, bigrams=bigrams.total)
