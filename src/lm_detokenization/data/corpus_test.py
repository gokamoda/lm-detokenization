from pathlib import Path

import numpy as np
import pytest
from corpus_tools import hash_sample, save_hash_sample
from corpus_tools.count import save_counts
from scipy import sparse

from lm_detokenization.data.corpus import (
    counts_dir,
    get_data,
    load_bigram_counts,
    load_token_counts,
    sample_path,
)

OPENWEBTEXT_DIR = Path("outputs/corpus-tools/Skylion007--openwebtext/plain_text/train")
WIKIPEDIA_JA_DIR = Path("outputs/corpus-tools/wikimedia--wikipedia/20231101.ja/train")


def test_paths_are_those_of_corpus_tools():
    assert sample_path() == OPENWEBTEXT_DIR / "samples/hash_n10000.jsonl"
    assert counts_dir("gpt2") == OPENWEBTEXT_DIR / "all/gpt2/counts/nobos"
    assert sample_path("wikipedia-ja") == WIKIPEDIA_JA_DIR / "samples/hash_n10000.jsonl"
    assert (
        counts_dir("rinna/japanese-gpt2-small", "wikipedia-ja")
        == WIKIPEDIA_JA_DIR / "all/rinna--japanese-gpt2-small/counts/nobos"
    )


def test_compute_data_writes_where_counts_are_read():
    script = (Path(__file__).parents[3] / "scripts/compute_data.sh").read_text()
    assert "--output-dir outputs/corpus-tools" in script
    assert "--tokenizer-name gpt2" in script


def test_load_counts(tmp_path):
    tokens = np.array([3, 0, 2])
    bigrams = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 0], [2, 0, 0]]))
    save_counts(tokens, tmp_path / "1-grams.npy")
    save_counts(bigrams, tmp_path / "2-grams.npz")
    assert (load_token_counts(tmp_path) == tokens).all()
    assert (load_bigram_counts(tmp_path) != bigrams).nnz == 0


def test_missing_counts(tmp_path):
    with pytest.raises(FileNotFoundError, match="frequency step"):
        load_bigram_counts(tmp_path)


def test_get_data_reads_the_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="sample step"):
        get_data()
    rows = [{"text": f"document {i}"} for i in range(100)]
    save_hash_sample(rows, sample_path(), num_samples=20)
    assert get_data(5) == hash_sample(rows, num_samples=5)
    assert len(get_data()) == 20
