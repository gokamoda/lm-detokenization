"""The OpenWebText sample and the token and bigram counts of OpenWebText.

Both are made with corpus-tools, which also makes them for other projects:

- The sample is the hash sample of corpus-tools (sha256 of the text; see
  corpus_tools.sample). It is saved in data/.
- Texts are tokenized without special tokens and bigrams are counted within
  each document, as in the published frequency.py. Bigram (a, b) is at row a
  (prefix), column b (suffix) of a sparse vocabulary x vocabulary matrix. The
  counts are saved in outputs/.

History: when the paper was written (Nov 2024), get_data() loaded the whole
"openwebtext" dataset (a loading-script dataset on the HF hub) and used
.shuffle(seed=42), taking the first N rows. The dataset has since been
converted to parquet (Dec 2025), and huggingface-hub>=1.0 requires the
"namespace/name" form. Since the row order may have changed, that shuffle no
longer reproduces the published documents, so we switched to a hash-based
sample that does not depend on the row order.
"""

import itertools
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from corpus_tools import load_hash_sample, preset, save_hash_sample
from corpus_tools.corpus import open_rows
from corpus_tools.count import count_ngrams, counts_filename, load_counts, save_counts
from corpus_tools.tokenize import tokenize
from scipy import sparse

OPENWEBTEXT = preset("openwebtext")
# The dataset is read at this commit of its Hub repository, so that the sample
# and the counts can be made again. The sample made from it is byte-identical
# to the one made before the switch to corpus-tools.
OPENWEBTEXT_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
NUM_SAMPLES = 10_000

# Relative to the repository root, where the commands are run.
DATA_DIR = Path("data") / "openwebtext"
COUNTS_ROOT = Path("outputs") / "freqs" / "openwebtext"


def sample_path(num_samples: int = NUM_SAMPLES) -> Path:
    return DATA_DIR / f"hash_n{num_samples}.jsonl"


def counts_dir(model_name: str) -> Path:
    return COUNTS_ROOT / model_name.replace("/", "--")


def make_sample(num_samples: int, output_path: Path) -> int:
    """Stream the whole dataset once and save its hash sample.

    Returns the number of rows saved.
    """
    with open_rows(OPENWEBTEXT, revision=OPENWEBTEXT_REVISION) as (rows, total):
        return save_hash_sample(rows, output_path, num_samples, total=total)


def get_data(num_samples: int | None = None) -> list[dict]:
    """First `num_samples` documents of the OpenWebText sample.

    Make the sample with `bash scripts/compute_data.sh sample` first.
    """
    path = sample_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `bash scripts/compute_data.sh sample` first."
        )
    return load_hash_sample(path, num_samples=num_samples)


@contextmanager
def open_texts(
    max_documents: int | None = None,
) -> Iterator[tuple[Iterator[str], int | None]]:
    """Texts of the whole dataset (streamed) and their number."""
    with open_rows(OPENWEBTEXT, revision=OPENWEBTEXT_REVISION) as (rows, num_rows):
        texts = (row["text"] for row in rows)
        if max_documents is None:
            yield texts, num_rows
        else:
            total = max_documents if num_rows is None else min(num_rows, max_documents)
            yield itertools.islice(texts, max_documents), total


def count_frequency(
    texts: Iterable[str],
    tokenizer,
    output_dir: Path,
    *,
    total: int | None = None,
    flush_every: int = 200_000_000,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Count the tokens and bigrams of `texts` and save them to `output_dir`."""
    counts = count_ngrams(
        tokenize(texts, tokenizer),
        [1, 2],
        len(tokenizer),
        flush_every=flush_every,
        total=total,
    )
    for n, value in counts.items():
        save_counts(value, output_dir / counts_filename(n))
    return counts[1], counts[2]


def load_token_counts(directory: Path) -> np.ndarray:
    """Count of each token, [vocab] (int64)."""
    return _load(directory, 1)


def load_bigram_counts(directory: Path) -> sparse.csr_matrix:
    """Count of each bigram, [vocab (prefix), vocab (suffix)] (int64)."""
    return _load(directory, 2)


def _load(directory: Path, n: int):
    path = directory / counts_filename(n)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `bash scripts/compute_data.sh frequency` first."
        )
    return load_counts(path)
