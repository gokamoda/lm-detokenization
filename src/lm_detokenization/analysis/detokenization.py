"""Token affinity T^ee between a current token (suffix) and every possible past
token (prefix), and how well it ranks the bigrams that actually occur
(Section 4).

As in the published code, token embeddings are divided by sigma averaged over
positions, and the AUROC of a suffix treats each prefix as positive with
weight = its bigram count (prefix, suffix) and as negative (weight 1) if the
bigram never occurs. A suffix without any occurring bigram gets AUROC 0, and
those zeros are included in the mean over suffixes.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score, roc_curve
from torchtyping import TensorType
from tqdm import tqdm

from lm_detokenization.tensor_types import BATCH, HEAD, HIDDEN_DIM, VOCAB
from lm_detokenization.weights import Layer0Weights


@dataclass
class TokenAffinity:
    wte: TensorType[VOCAB, HIDDEN_DIM]  # divided by sigma averaged over positions
    w_qk: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM]

    @classmethod
    def from_weights(cls, weights: Layer0Weights) -> "TokenAffinity":
        sigma = weights.sigma.mean(dim=0)  # float16, as published
        return cls(wte=weights.wte / sigma.unsqueeze(-1), w_qk=weights.qk.w_qk)

    @torch.no_grad()
    def suffix_scores(self, suffix_ids: list[int]) -> TensorType[BATCH, HEAD, VOCAB]:
        """T^ee with the suffix as the current token i and every prefix as j."""
        query = self.wte[suffix_ids]  # [B, D]
        projected = torch.einsum("bd,hde->bhe", query, self.w_qk)
        return torch.einsum("bhe,ve->bhv", projected, self.wte)


def suffix_auroc(scores: np.ndarray, prefix_counts: np.ndarray) -> float:
    """AUROC of one head's scores over prefixes, with bigram counts as weights."""
    positive = prefix_counts > 0
    if not positive.any():
        return 0.0
    weight = np.where(positive, prefix_counts, 1)
    return float(roc_auc_score(positive, scores, sample_weight=weight))


def compute_auroc(
    affinity: TokenAffinity,
    bigram_counts: sparse.csc_matrix,
    suffix_ids: list[int],
    batch_size: int = 64,
) -> pl.DataFrame:
    """AUROC of every head for every suffix: columns suffix_id, head, auroc,
    num_valid_detokenization (total count of bigrams ending in the suffix)."""
    rows = []
    for start in tqdm(range(0, len(suffix_ids), batch_size), desc="AUROC"):
        batch = suffix_ids[start : start + batch_size]
        scores = affinity.suffix_scores(batch).numpy()
        for b, suffix_id in enumerate(batch):
            counts = bigram_counts[:, suffix_id].toarray().ravel()
            for head in range(scores.shape[1]):
                rows.append(
                    {
                        "suffix_id": suffix_id,
                        "head": head,
                        "auroc": suffix_auroc(scores[b, head], counts),
                        "num_valid_detokenization": int(counts.sum()),
                    }
                )
    return pl.DataFrame(rows)


def mean_auroc_by_head(auroc: pl.DataFrame) -> pl.DataFrame:
    return auroc.group_by("head").agg(pl.mean("auroc")).sort("auroc", descending=True)


def top_prefixes(
    affinity: TokenAffinity, suffix_id: int, head: int, k: int
) -> list[tuple[int, int]]:
    """The k prefixes with the highest T^ee, as (rank starting at 1, prefix id)."""
    scores = affinity.suffix_scores([suffix_id])[0, head]
    return [
        (rank + 1, int(i)) for rank, i in enumerate(scores.argsort(descending=True)[:k])
    ]


def roc(
    affinity: TokenAffinity, bigram_counts: sparse.csc_matrix, suffix_id: int, head: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """(false positive rate, true positive rate, AUROC) of one suffix and head."""
    scores = affinity.suffix_scores([suffix_id])[0, head].numpy()
    counts = bigram_counts[:, suffix_id].toarray().ravel()
    positive = counts > 0
    weight = np.where(positive, counts, 1)
    fpr, tpr, _ = roc_curve(positive, scores, sample_weight=weight)
    return fpr, tpr, suffix_auroc(scores, counts)
