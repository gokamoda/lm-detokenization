"""The document sample and the token and bigram counts of a corpus.

Both are made with the corpus-tools command (`scripts/compute_data.sh` for
GPT-2 and OpenWebText, `scripts/compute_data_rinna.sh` for rinna's Japanese
GPT-2 and Japanese Wikipedia), streaming the corpus at a fixed commit, and
saved under outputs/corpus-tools in corpus-tools' layout. They are made
apart: the counts are of the whole corpus, not of the sample.

- The sample is the hash sample of corpus-tools (sha256 of the text; see
  corpus_tools.sample).
- Texts are tokenized without special tokens (by tokenizer-tools, which
  lowercases for rinna's Japanese GPT-2) and bigrams are counted within each
  document, as in the published frequency.py. Bigram (a, b) is at row a
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

# --corpus of the commands -> the corpus of corpus-tools
CORPORA = {
    "openwebtext": preset("openwebtext"),
    "wikipedia-ja": preset("wikipedia", name="20231101.ja"),
}
DEFAULT_CORPUS = "openwebtext"
NUM_SAMPLES = 10_000

# The --output-dir of corpus-tools in scripts/compute_data*.sh, relative to
# the repository root, where the commands are run.
CORPUS_TOOLS_OUTPUT = Path("outputs") / "corpus-tools"
_STORE = Store(CORPUS_TOOLS_OUTPUT)


def sample_path(corpus: str = DEFAULT_CORPUS, num_samples: int = NUM_SAMPLES) -> Path:
    return _STORE.sample_path(CORPORA[corpus], num_samples)


def counts_dir(model_name: str, corpus: str = DEFAULT_CORPUS) -> Path:
    """Counts made by `corpus-tools count --tokenizer-name <model_name>`."""
    return _STORE.tokenizer_dir(CORPORA[corpus], model_name) / "counts" / "nobos"


def get_data(
    num_samples: int | None = None, corpus: str = DEFAULT_CORPUS
) -> list[dict]:
    """First `num_samples` documents of the sample of `corpus`.

    Make the sample with the `sample` step of scripts/compute_data*.sh first.
    """
    path = sample_path(corpus)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Make it with the sample step of "
            "scripts/compute_data*.sh first."
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
            f"{path} not found. Make it with the frequency step of "
            "scripts/compute_data*.sh first."
        )
    return load_counts(path)
