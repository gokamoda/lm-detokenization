from collections import Counter

from lm_detokenization.data.frequency import Counts, count


class CharTokenizer:
    """Tokenizes into character codes 0..25 (a..z), like a HF tokenizer call."""

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [[ord(c) - ord("a") for c in t] for t in texts]}


def test_counts_match_counter(tmp_path):
    texts = ["abcab", "ba", "a", "", "zzza"] * 7
    counts = count(texts, CharTokenizer(), vocab_size=26, batch_size=3, flush_every=5)

    expected_tokens, expected_bigrams = Counter(), Counter()
    for t in texts:
        ids = [ord(c) - ord("a") for c in t]
        expected_tokens.update(ids)
        expected_bigrams.update(zip(ids[:-1], ids[1:]))  # within a document only

    assert {i: int(c) for i, c in enumerate(counts.tokens) if c} == dict(
        expected_tokens
    )
    coo = counts.bigrams.tocoo()
    assert {
        (int(a), int(b)): int(c) for a, b, c in zip(coo.row, coo.col, coo.data)
    } == dict(expected_bigrams)

    counts.save(tmp_path)
    loaded = Counts.load(tmp_path)
    assert (loaded.tokens == counts.tokens).all()
    assert (loaded.bigrams != counts.bigrams).nnz == 0
