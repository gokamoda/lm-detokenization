"""Draw a fixed random sample from a (streaming) dataset by hashing.

Every row gets the key sha256(row[key_field]), and the sample is the
`num_samples` rows with the smallest keys, written in ascending key order.

- The key does not depend on the content, so the sample is uniformly random.
- The first n rows of a sample are the sample of size n, so a sample can be
  extended (100 -> 500 -> ...) without changing the rows already used.
- The key depends only on the row itself, not on its position, so the sample
  does not change when the dataset is re-sharded or reordered.
- Rows with the same key (identical texts) are kept only once.

The whole dataset is read once, but only `num_samples` rows are kept in memory.
"""

import hashlib
import heapq
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm


def hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_sample(
    rows: Iterable[dict[str, Any]],
    num_samples: int,
    key_field: str = "text",
) -> list[dict[str, Any]]:
    """Return the `num_samples` rows with the smallest hash keys, sorted by key.

    Each returned row has an extra field "hash" holding its key.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    # max-heap of the smallest keys seen so far: (negated key, row)
    heap: list[tuple[int, dict[str, Any]]] = []
    kept_keys: set[str] = set()
    for row in rows:
        key = hash_key(row[key_field])
        if key in kept_keys:
            continue
        neg_key = -int(key, 16)
        if len(heap) < num_samples:
            heapq.heappush(heap, (neg_key, {**row, "hash": key}))
            kept_keys.add(key)
        elif neg_key > heap[0][0]:
            _, removed = heapq.heapreplace(heap, (neg_key, {**row, "hash": key}))
            kept_keys.discard(removed["hash"])
            kept_keys.add(key)

    return [row for _, row in sorted(heap, key=lambda item: -item[0])]


def save_hash_sample(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    num_samples: int,
    key_field: str = "text",
    total: int | None = None,
) -> None:
    """Write `hash_sample(rows, ...)` to `output_path` as jsonl."""
    sample = hash_sample(
        tqdm(rows, total=total, desc="Hashing", mininterval=10.0),
        num_samples=num_samples,
        key_field=key_field,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        for row in sample:
            json.dump(row, output_file, ensure_ascii=False)
            output_file.write("\n")
    print(f"Saved {len(sample)} rows to {output_path}")


def load_hash_sample(path: Path, num_samples: int | None = None) -> list[dict]:
    """Read the first `num_samples` rows (all if None) of a saved sample."""
    rows = []
    with path.open() as input_file:
        for line in input_file:
            if num_samples is not None and len(rows) >= num_samples:
                break
            rows.append(json.loads(line))
    if num_samples is not None and len(rows) < num_samples:
        raise ValueError(
            f"{path} has only {len(rows)} rows, but {num_samples} were requested"
        )
    return rows
