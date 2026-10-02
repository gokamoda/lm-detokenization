"""Check that the migrated code reproduces the values of tag `published`.

The reference values are made by scripts/make_published_golden.py.
"""

from pathlib import Path

import pytest
import torch
from feature_extractor.models import load_causal_model, load_tokenizer
from transformers import GPT2LMHeadModel

from lm_detokenization.analysis.detokenization import TokenAffinity
from lm_detokenization.analysis.six_terms import compute_6terms
from lm_detokenization.weights import (
    get_position_embedding,
    get_qk_weights,
    get_word_embedding,
    load_layer0_weights,
)

GOLDEN_PATH = Path(__file__).parents[2] / "outputs/golden/published.pt"
MODEL_NAME = "gpt2"

pytestmark = pytest.mark.skipif(
    not GOLDEN_PATH.exists(), reason=f"{GOLDEN_PATH} not found"
)


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN_PATH, weights_only=False)


@pytest.fixture(scope="module")
def model():
    return load_causal_model(MODEL_NAME, device="cpu")


def assert_close(actual: torch.Tensor, expected: torch.Tensor):
    # Folding LN in a different order changes float32 rounding only, so the
    # tolerance is relative to the scale of the values.
    torch.testing.assert_close(
        actual, expected, rtol=1e-5, atol=1e-6 * expected.abs().max().item()
    )


def test_qk_weights(golden, model):
    qk_weights = get_qk_weights(model, layer_index=0)
    assert_close(qk_weights.w_qk, golden["wqkh"])
    assert_close(qk_weights.b_qk, golden["bqwkh"])


def test_six_terms(golden, model):
    tokenizer = load_tokenizer(MODEL_NAME)
    qk_weights = get_qk_weights(model, layer_index=0)
    for prompt, expected in zip(golden["prompts"], golden["six_terms"]):
        # published prepended <|endoftext|> by hand; load_tokenizer does it
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        assert torch.equal(input_ids, expected["input_ids"])

        scores = compute_6terms(
            prompt=prompt,
            tokenizer=tokenizer,
            wte=get_word_embedding(model),
            wpe=get_position_embedding(model),
            w_compare=qk_weights.w_qk,
            w_self=qk_weights.b_qk,
        )
        assert scores.keys() == expected["scores"].keys()
        for key in scores:
            assert_close(scores[key], expected["scores"][key])


def test_detokenization_scores(golden):
    tokenizer = load_tokenizer(MODEL_NAME)
    affinity = TokenAffinity.from_weights(load_layer0_weights(MODEL_NAME))
    for query, expected in golden["detok"].items():
        suffix_id = tokenizer.encode(query, add_special_tokens=False)[-1]
        assert suffix_id == expected["suffix_id"]
        scores = affinity.suffix_scores([suffix_id])[0]
        assert_close(scores.cpu(), expected["scores"])


def test_vs_tptpp_attention_without_bos(golden):
    """Layer 0 attention of the original model, with the published input
    (no <|endoftext|>). Checks that upgrading transformers changed nothing."""
    tokenizer = load_tokenizer(MODEL_NAME)
    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    hf_model.eval()
    for prompt, expected in zip(golden["prompts"], golden["vs_tptpp"]):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        assert torch.equal(inputs.input_ids, expected["input_ids"])
        with torch.no_grad():
            output = hf_model(**inputs, output_attentions=True)
        assert_close(output.attentions[0][0], expected["attn_l0"])
