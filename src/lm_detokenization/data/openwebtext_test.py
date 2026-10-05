from collections import Counter
from pathlib import Path

import pytest
from corpus_tools import hash_sample, save_hash_sample

from lm_detokenization.data.openwebtext import (
    count_frequency,
    counts_dir,
    get_data,
    load_bigram_counts,
    load_token_counts,
    sample_path,
)


class CharTokenizer:
    """Tokenizes into character codes 0..25 (a..z), like a HF tokenizer call."""

    def __len__(self):
        return 26

    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [[ord(c) - ord("a") for c in t] for t in texts]}


def test_paths():
    assert sample_path() == Path("data/openwebtext/hash_n10000.jsonl")
    assert counts_dir("gpt2") == Path("outputs/freqs/openwebtext/gpt2")
    assert counts_dir("rinna/japanese-gpt-1b") == Path(
        "outputs/freqs/openwebtext/rinna--japanese-gpt-1b"
    )


def test_counts_match_counter_and_are_saved(tmp_path):
    texts = ["abcab", "ba", "a", "", "zzza"] * 7
    tokens, bigrams = count_frequency(texts, CharTokenizer(), tmp_path)

    expected_tokens, expected_bigrams = Counter(), Counter()
    for t in texts:
        ids = [ord(c) - ord("a") for c in t]
        expected_tokens.update(ids)
        expected_bigrams.update(zip(ids[:-1], ids[1:]))  # within a document only

    assert {i: int(c) for i, c in enumerate(tokens) if c} == dict(expected_tokens)
    coo = bigrams.tocoo()
    assert {
        (int(a), int(b)): int(c) for a, b, c in zip(coo.row, coo.col, coo.data)
    } == dict(expected_bigrams)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["1-grams.npy", "2-grams.npz"]
    assert (load_token_counts(tmp_path) == tokens).all()
    assert (load_bigram_counts(tmp_path) != bigrams).nnz == 0


def test_missing_counts(tmp_path):
    with pytest.raises(FileNotFoundError, match="compute_data.sh frequency"):
        load_bigram_counts(tmp_path)


def test_get_data_reads_the_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="compute_data.sh sample"):
        get_data()
    rows = [{"text": f"document {i}"} for i in range(100)]
    save_hash_sample(rows, sample_path(), num_samples=20)
    assert get_data(5) == hash_sample(rows, num_samples=5)
    assert len(get_data()) == 20
