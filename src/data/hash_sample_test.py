import random

from data.hash_sample import hash_key, hash_sample, load_hash_sample, save_hash_sample


def make_rows(n: int) -> list[dict]:
    return [{"text": f"document {i}"} for i in range(n)]


def test_smallest_keys_in_order():
    rows = make_rows(100)
    sample = hash_sample(rows, num_samples=10)
    expected = sorted(hash_key(row["text"]) for row in rows)[:10]
    assert [row["hash"] for row in sample] == expected


def test_sample_is_nested():
    rows = make_rows(1000)
    small = hash_sample(rows, num_samples=10)
    large = hash_sample(rows, num_samples=100)
    assert large[:10] == small


def test_independent_of_row_order():
    rows = make_rows(1000)
    shuffled = rows.copy()
    random.Random(0).shuffle(shuffled)
    assert hash_sample(rows, num_samples=50) == hash_sample(shuffled, num_samples=50)


def test_duplicates_kept_once():
    rows = make_rows(100) * 3
    sample = hash_sample(rows, num_samples=100)
    assert len(sample) == 100
    assert len({row["text"] for row in sample}) == 100


def test_fewer_rows_than_requested():
    assert len(hash_sample(make_rows(5), num_samples=10)) == 5


def test_save_and_load(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = make_rows(100)
    save_hash_sample(rows, path, num_samples=20)
    assert load_hash_sample(path) == hash_sample(rows, num_samples=20)
    assert load_hash_sample(path, num_samples=5) == hash_sample(rows, num_samples=5)
