import torch
from torchtyping import TensorType

from .mytorchtyping import BATCH, HEAD, HIDDEN_DIM, POS, SEQUENCE, VOCAB


def get_var_matrix(
    wpe: TensorType[POS, HIDDEN_DIM], wte: TensorType[VOCAB, HIDDEN_DIM]
) -> TensorType[POS, VOCAB]:
    """_summary_

    Parameters
    ----------
    wpe : TensorType[POS, HIDDEN_DIM]
    wte : TensorType[VOCAB, HIDDEN_DIM]

    Returns
    -------
    TensorType[POS, VOCAB]
    """
    _, hidden_dim = wpe.shape
    centering: TensorType[HIDDEN_DIM, HIDDEN_DIM] = (
        torch.diag(torch.ones(hidden_dim)) - 1 / hidden_dim
    )
    wpe: TensorType[POS, HIDDEN_DIM] = wpe @ centering
    wte: TensorType[VOCAB, HIDDEN_DIM] = wte @ centering

    wpe_var: TensorType[POS] = wpe.var(dim=-1, unbiased=False)
    wte_var: TensorType[VOCAB] = wte.var(dim=-1, unbiased=False)
    cov: TensorType[POS, VOCAB] = wpe @ wte.T / hidden_dim

    var_matrix: TensorType[POS, VOCAB] = (
        2 * cov
        + wpe_var.unsqueeze(1).expand(-1, cov.shape[1])  # TensorType[POS, VOCAB]
        + wte_var.unsqueeze(0).expand(cov.shape[0], -1)  # TensorType[POS, VOCAB]
    )
    return var_matrix


def ln_pos(
    x: TensorType[BATCH, HEAD, POS], var_matrix: TensorType[POS, VOCAB]
) -> TensorType[BATCH, HEAD, POS, VOCAB]:
    """_summary_

    Parameters
    ----------
    x : TensorType[BATCH, HEAD, POS]
    var_matrix : TensorType[POS, VOCAB]

    Returns
    -------
    TensorType[BATCH, HEAD, POS, VOCAB]
    """
    vocab_size = var_matrix.shape[1]
    x: TensorType[BATCH, HEAD, POS, VOCAB] = x.unsqueeze(-1).expand(
        -1, -1, -1, vocab_size
    )
    var_matrix: TensorType[POS, VOCAB] = var_matrix.unsqueeze(0).unsqueeze(0)
    ln_result: TensorType[BATCH, HEAD, POS, VOCAB] = x / var_matrix

    return ln_result
