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


def batch_auroc(
    scores: TensorType[BATCH, VOCAB], counts: TensorType[BATCH, VOCAB]
) -> np.ndarray:
    """suffix_auroc of each row at once, on the device of the tensors.

    AUROC = sum over positives p and negatives n of w_p w_n [s_p > s_n]
    (ties count 1/2), divided by (sum w_p)(sum w_n). The scores are sorted,
    tied scores are grouped, and the weights are summed in int64 (counts for
    positives, 1 for negatives), doubled so that the halves of ties are
    integers too: the sums are exact. Returns float64 AUROCs (0 for a row
    without positives).
    """
    order = scores.argsort(dim=-1)
    sorted_scores = scores.gather(-1, order)
    positive = counts.to(torch.int64).gather(-1, order)
    del order
    negative = (positive == 0).to(torch.int64)
    starts = torch.ones_like(sorted_scores, dtype=torch.bool)
    starts[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    del sorted_scores
    group = starts.cumsum(dim=-1) - 1  # tie group of each sorted score
    del starts
    group_positive = torch.zeros_like(positive).scatter_add_(-1, group, positive)
    group_negative = torch.zeros_like(negative).scatter_add_(-1, group, negative)
    del group
    total_positive = positive.sum(dim=-1)
    total_negative = negative.sum(dim=-1)
    del positive, negative
    below = group_negative.cumsum(dim=-1) - group_negative  # negatives scored lower
    twice = (group_positive * (2 * below + group_negative)).sum(dim=-1)
    # in float64 on the CPU (MPS has no float64)
    twice, total_positive, total_negative = (
        x.cpu().numpy().astype(np.float64)
        for x in (twice, total_positive, total_negative)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        auroc = twice / (2 * total_positive * total_negative)
    return np.where(total_positive > 0, auroc, 0.0)


def compute_auroc(
    affinity: TokenAffinity,
    bigram_counts: sparse.csc_matrix,
    suffix_ids: list[int],
    batch_size: int = 64,
    device: str | None = None,
) -> pl.DataFrame:
    """AUROC of every head for every suffix: columns suffix_id, head, auroc,
    num_valid_detokenization (total count of bigrams ending in the suffix).

    Without `device`, with scikit-learn on the CPU (as published). With a
    device ("cuda", "mps", "cpu"), with batch_auroc there: the same values up
    to rounding, much faster on a GPU. For GPT-2, a batch of 64 suffixes takes
    a few GB there (4.4 GiB held on MPS); memory grows with batch_size.
    """
    if device is not None:
        return _compute_auroc_on(
            affinity, bigram_counts, suffix_ids, batch_size, device
        )
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


def _compute_auroc_on(
    affinity: TokenAffinity,
    bigram_counts: sparse.csc_matrix,
    suffix_ids: list[int],
    batch_size: int,
    device: str,
) -> pl.DataFrame:
    on_device = TokenAffinity(
        wte=affinity.wte.to(device), w_qk=affinity.w_qk.to(device)
    )
    num_heads = affinity.w_qk.shape[0]
    columns: dict[str, list[np.ndarray]] = {
        "suffix_id": [],
        "head": [],
        "auroc": [],
        "num_valid_detokenization": [],
    }
    for start in tqdm(range(0, len(suffix_ids), batch_size), desc=f"AUROC on {device}"):
        batch = suffix_ids[start : start + batch_size]
        scores = on_device.suffix_scores(batch)  # [B, H, V]
        counts = np.asarray(bigram_counts[:, batch].T.toarray(), dtype=np.int64)
        auroc = batch_auroc(
            scores.reshape(len(batch) * num_heads, -1),
            torch.from_numpy(counts).to(device).repeat_interleave(num_heads, dim=0),
        )
        columns["suffix_id"].append(np.repeat(batch, num_heads))
        columns["head"].append(np.tile(np.arange(num_heads), len(batch)))
        columns["auroc"].append(auroc)
        columns["num_valid_detokenization"].append(
            np.repeat(counts.sum(axis=1), num_heads)
        )
    return pl.DataFrame(
        {name: np.concatenate(parts) for name, parts in columns.items()}
    ).with_columns(
        pl.col("suffix_id").cast(pl.Int64),
        pl.col("head").cast(pl.Int64),
        pl.col("num_valid_detokenization").cast(pl.Int64),
    )


def mean_auroc_by_head(auroc: pl.DataFrame, *, seen_only: bool = False) -> pl.DataFrame:
    """Mean AUROC per head, over all suffixes (those never seen as a suffix
    count as 0, as published) or, with `seen_only`, over the suffixes seen in
    the corpus. A vocabulary with many tokens unseen in the corpus (30% of
    rinna/japanese-gpt-1b's on Japanese Wikipedia) lowers the former."""
    if seen_only:
        auroc = auroc.filter(pl.col("num_valid_detokenization") > 0)
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
