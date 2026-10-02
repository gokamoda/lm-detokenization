"""Save a fixed random sample of OpenWebText (see data/hash_sample.py).

    python src/sample_openwebtext.py --num-samples 10000

Reads the whole dataset once by streaming (about 24GB of download), but
stores only the sampled rows.
"""

from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset

from data.hash_sample import save_hash_sample
from empirical.utils import OPENWEBTEXT, OPENWEBTEXT_SAMPLE_PATH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--output-path", type=Path, default=OPENWEBTEXT_SAMPLE_PATH)
    args = parser.parse_args()

    dataset = load_dataset(OPENWEBTEXT, split="train", streaming=True)
    save_hash_sample(
        dataset,
        output_path=args.output_path,
        num_samples=args.num_samples,
        total=dataset.info.splits["train"].num_examples,
    )
