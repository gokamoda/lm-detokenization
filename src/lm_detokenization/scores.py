import torch
from torchtyping import TensorType

from lm_detokenization.tensor_types import BATCH, HEAD, HIDDEN_DIM, SEQUENCE


def compute_compare_score(
    i: TensorType[BATCH, SEQUENCE, HIDDEN_DIM],
    j: TensorType[BATCH, SEQUENCE, HIDDEN_DIM],
    w: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM],
) -> TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE]:
    """Compute i^T W j for every head.

    Parameters
    ----------
    i : TensorType[BATCH, SEQUENCE, HIDDEN_DIM]
    j : TensorType[BATCH, SEQUENCE, HIDDEN_DIM]
    w : TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM]

    Returns
    -------
    TensorType[BATCH, HEAD, SEQUENCE, SEQUENCE]
    """

    return torch.einsum(
        "bshd,bdt->bhst",
        torch.einsum(
            "bsd,hdi->bshi",
            i,
            w.to(i.device),
        ),
        j.transpose(-1, -2),
    )


def compute_self_score(
    j: TensorType[BATCH, SEQUENCE, HIDDEN_DIM],
    w: TensorType[HEAD, HIDDEN_DIM],
) -> TensorType[BATCH, HEAD, SEQUENCE]:
    """Compute w^T j for every head.

    Parameters
    ----------
    j : TensorType[BATCH, SEQUENCE, HIDDEN_DIM]
    w : TensorType[HEAD, HIDDEN_DIM]

    Returns
    -------
    TensorType[BATCH, HEAD, SEQUENCE]
    """
    return torch.einsum(
        "hd,bdt->bht",
        w.to(j.device),
        j.transpose(-1, -2),
    )
