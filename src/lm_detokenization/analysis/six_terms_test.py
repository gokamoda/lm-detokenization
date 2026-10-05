from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lm_detokenization.analysis.six_terms import (
    MAX_LENGTH,
    TERMS,
    compute_contributions,
    contributions,
)


def published_loop(scores):
    """The contribution loop of the published empirical/six_terms_importance.py."""
    _, n_head, max_length = scores["posj"].shape
    out = torch.zeros(n_head, max_length, len(TERMS))
    for head in range(n_head):
        for length in range(max_length):
            original = (
                scores["embi_embj"][0, head, length, : length + 1]
                + scores["embj"][0, head, : length + 1]
                + scores["posi_posj"][0, head, length, : length + 1]
                + scores["posj"][0, head, : length + 1]
                + scores["embi_posj"][0, head, length, : length + 1]
                + scores["posi_embj"][0, head, length, : length + 1]
            )
            ablated = {
                "embi_embj": original
                - scores["embi_embj"][0, head, length, : length + 1],
                "embj": original - scores["embj"][0, head, : length + 1],
                "posi_posj": original
                - scores["posi_posj"][0, head, length, : length + 1],
                "posj": original - scores["posj"][0, head, : length + 1],
                "embi_posj": original
                - scores["embi_posj"][0, head, length, : length + 1],
                "posi_embj": original
                - scores["posi_embj"][0, head, length, : length + 1],
            }
            weight = torch.nn.functional.softmax(original, dim=-1)
            for t, name in enumerate(TERMS):
                out[head, length, t] = torch.nn.functional.kl_div(
                    input=torch.nn.functional.log_softmax(ablated[name], dim=-1),
                    target=weight,
                    reduction="mean",
                )
    return out


def test_contributions_match_published_loop():
    g = torch.Generator().manual_seed(0)
    heads, length = 3, 17
    scores = {}
    for name in TERMS:
        shape = (
            (1, heads, length)
            if name in ("embj", "posj")
            else (1, heads, length, length)
        )
        scores[name] = 5 * torch.randn(*shape, generator=g)
    torch.testing.assert_close(
        contributions(scores), published_loop(scores), rtol=1e-4, atol=1e-6
    )


class ToyTokenizer:
    """Token id = character code mod 50, called like a HF tokenizer."""

    def __call__(self, text):
        return {"input_ids": [ord(c) % 50 for c in text]}


def accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(accelerator() is None, reason="no GPU")
def test_contributions_on_a_gpu_match_the_cpu(tmp_path):
    g = torch.Generator().manual_seed(0)
    heads, dim = 3, 8
    weights = SimpleNamespace(
        wte=torch.randn(50, dim, generator=g),
        wpe=torch.randn(MAX_LENGTH, dim, generator=g),
        qk=SimpleNamespace(
            w_qk=torch.randn(heads, dim, dim, generator=g),
            b_qk=torch.randn(heads, dim, generator=g),
        ),
        num_heads=heads,
    )
    texts = ["the first document", "a second, longer document " * 3]
    for device in ["cpu", accelerator()]:
        compute_contributions(
            weights, ToyTokenizer(), texts, tmp_path / f"{device}.npy", device=device
        )
    cpu = np.load(tmp_path / "cpu.npy")
    gpu = np.load(tmp_path / f"{accelerator()}.npy")
    np.testing.assert_array_equal(np.isnan(cpu), np.isnan(gpu))
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-5)
