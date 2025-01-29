import torch

from .ln import get_var_matrix, ln_pos


def test_get_var_matrix():
    pos = 3
    vocab = 5
    hidden_dim = 7

    wpe = torch.randn(pos, hidden_dim)
    wte = torch.randn(vocab, hidden_dim)

    var_matrix = get_var_matrix(wpe, wte)

    assert torch.allclose(var_matrix[0, 0], (wpe[0] + wte[0]).var(unbiased=False))


def test_ln_pos():
    batch = 1
    head = 2
    pos = 5
    vocab = 7

    x = torch.randn(batch, head, pos)
    var_matrix = torch.randn(pos, vocab)

    ln_result = ln_pos(x, var_matrix)

    check_pos = 0
    check_vocab = 0
    assert torch.allclose(
        ln_result[0, 0, check_pos, check_vocab],
        x[0, 0, check_pos] / var_matrix[check_pos, check_vocab],
    )
