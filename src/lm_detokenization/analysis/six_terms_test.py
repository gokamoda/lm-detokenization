import torch

from lm_detokenization.analysis.six_terms import TERMS, contributions


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
