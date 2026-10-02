from pathlib import Path

from data.hash_sample import load_hash_sample

# History: when the paper was written (Nov 2024), get_data() loaded the whole
# "openwebtext" dataset (a loading-script dataset on the HF hub) and used
# .shuffle(seed=42), taking the first N rows. The dataset has since been
# converted to parquet (Dec 2025), and huggingface-hub>=1.0 requires the
# "namespace/name" form. Since the row order may have changed, that shuffle
# no longer reproduces the published documents, so we switched to a hash-based
# sample that does not depend on the row order (see data/hash_sample.py).
OPENWEBTEXT = "Skylion007/openwebtext"
OPENWEBTEXT_SAMPLE_PATH = (
    Path(__file__).parents[2] / "outputs/data/openwebtext_sample.jsonl"
)


def get_data(num_samples: int | None = None) -> list[dict]:
    """First `num_samples` documents of the OpenWebText sample.

    Make the sample with `python src/sample_openwebtext.py` first.
    """
    if not OPENWEBTEXT_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{OPENWEBTEXT_SAMPLE_PATH} not found. "
            "Run `python src/sample_openwebtext.py` first."
        )
    return load_hash_sample(OPENWEBTEXT_SAMPLE_PATH, num_samples=num_samples)
