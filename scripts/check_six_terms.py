"""Check that the six terms reproduce the model's layer-0 attention.

The softmax of the sum of the six terms (computed from the weights, divided
by sqrt(d_head)) should equal the attention weights the model computes, as
the terms depending on i alone cancel in the softmax. Prints, per head, the
largest difference on one row (position 200) of a repeated sentence.

Usage (from the repository root):
    uv run python scripts/check_six_terms.py gpt2
    uv run python scripts/check_six_terms.py rinna/japanese-gpt-1b
"""

import sys

import torch
from feature_extractor.models import load_tokenizer

from lm_detokenization.analysis.empirical_attention import extract_attention_rows
from lm_detokenization.analysis.six_terms import TERMS, _as_matrix, compute_6terms
from lm_detokenization.weights import load_layer0_weights

model = sys.argv[1]
text = (
    "東京は日本の首都であり、政治や経済の中心地である。" * 30
    if "rinna" in model
    else "Tokyo is the capital of Japan and its center of politics and economy. " * 30
)
w = load_layer0_weights(model)
tok = load_tokenizer(model)
scores = compute_6terms(text, tok, w.wte, w.wpe, w.qk.w_qk, w.qk.b_qk)
L = scores["posj"].shape[-1]
total = sum(_as_matrix(scores[n], L).float() for n in TERMS)  # [H, L, L]
future = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
pred = torch.softmax(
    torch.where(future, float("-inf"), total / w.head_dim**0.5), dim=-1
)
pos = min(L - 1, 200)
obs = extract_attention_rows([text], [pos], model)[pos].weights[0].float()  # [H, pos+1]
diff = (pred[:, pos, : pos + 1] - obs).abs().max(dim=-1).values
print(
    f"{model}: L={L}, row i={pos}; max |predicted - observed| per head:",
    [round(float(d), 4) for d in diff],
)
