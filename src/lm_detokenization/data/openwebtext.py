"""The OpenWebText sample and the token and bigram counts of OpenWebText.

Both are made with the corpus-tools command (`bash scripts/compute_data.sh
sample frequency`), from OpenWebText streamed at a fixed commit, and saved
under outputs/corpus-tools in corpus-tools' layout. They are made apart: the
counts are of all of OpenWebText, not of the sample.

- The sample is the hash sample of corpus-tools (sha256 of the text; see
  corpus_tools.sample).
- Texts are tokenized without special tokens and bigrams are counted within
  each document, as in the published frequency.py. Bigram (a, b) is at row a
  (prefix), column b (suffix) of a sparse vocabulary x vocabulary matrix.

History: when the paper was written (Nov 2024), get_data() loaded the whole
"openwebtext" dataset (a loading-script dataset on the HF hub) and used
.shuffle(seed=42), taking the first N rows. The dataset has since been
converted to parquet (Dec 2025), and huggingface-hub>=1.0 requires the
"namespace/name" form. Since the row order may have changed, that shuffle no
longer reproduces the published documents, so we switched to a hash-based
sample that does not depend on the row order.
"""

from pathlib import Path

import numpy as np
from corpus_tools import Store, load_hash_sample, preset
from corpus_tools.count import counts_filename, load_counts
from scipy import sparse

OPENWEBTEXT = preset("openwebtext")
NUM_SAMPLES = 10_000

# The --output-dir of corpus-tools in scripts/compute_data.sh, relative to the
# repository root, where the commands are run.
CORPUS_TOOLS_OUTPUT = Path("outputs") / "corpus-tools"
_STORE = Store(CORPUS_TOOLS_OUTPUT)


def sample_path(num_samples: int = NUM_SAMPLES) -> Path:
    return _STORE.sample_path(OPENWEBTEXT, num_samples)


def counts_dir(model_name: str) -> Path:
    """Counts made by `corpus-tools count --tokenizer-name <model_name>`."""
    return _STORE.tokenizer_dir(OPENWEBTEXT, model_name) / "counts" / "nobos"


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
