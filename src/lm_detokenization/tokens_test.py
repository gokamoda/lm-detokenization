import pytest
from feature_extractor.models import load_tokenizer

from lm_detokenization.tokens import encode, suffix_id


def tokens(tokenizer, ids):
    return tokenizer.convert_ids_to_tokens(ids)


@pytest.mark.parametrize(
    ("model", "word", "expected"),
    [
        ("gpt2", "iens", "iens"),
        ("gpt2", " Jackson", "ĠJackson"),
        ("rinna/japanese-gpt2-small", "サピエンス", "ンス"),
        ("rinna/japanese-gpt2-small", "パン", "パン"),  # ▁ + パン already
        ("rinna/japanese-gpt-1b", "パン", "パン"),  # ▁パン alone: the one without ▁
        ("rinna/japanese-gpt-1b", "サピエンス", "ンス"),
    ],
)
def test_suffix_id_is_a_token_inside_a_word(model, word, expected):
    tokenizer = load_tokenizer(model)
    assert tokens(tokenizer, [suffix_id(tokenizer, word)]) == [expected]


def test_encode_gives_bos_to_gpt2_only():
    gpt2 = load_tokenizer("gpt2")
    assert tokens(gpt2, encode(gpt2, "Tokyo"))[0] == "<|endoftext|>"
    rinna = load_tokenizer("rinna/japanese-gpt2-small")
    assert tokens(rinna, encode(rinna, "NASA")) == ["▁", "nasa"]
