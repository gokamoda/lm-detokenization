"""Commands that prepare data: the OpenWebText sample, corpus counts and the
observed attention."""

import argparse
import itertools
from pathlib import Path

from datasets import load_dataset
from feature_extractor.models import load_tokenizer

from lm_detokenization.analysis.empirical_attention import extract_attention_rows
from lm_detokenization.cli.args import COUNTS_DIR, add_model_arg, attention_rows_path
from lm_detokenization.data.frequency import count
from lm_detokenization.data.hash_sample import save_hash_sample
from lm_detokenization.data.openwebtext import (
    OPENWEBTEXT,
    OPENWEBTEXT_SAMPLE_PATH,
    get_data,
)


def sample_openwebtext_main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a fixed random sample of OpenWebText (see data/hash_sample.py). "
        "Reads the whole dataset once by streaming (about 24GB of download), but "
        "stores only the sampled rows."
    )
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


def count_frequency_main() -> None:
    parser = argparse.ArgumentParser(
        description="Token and bigram counts (tokenized without special tokens; "
        "bigrams within each document)."
    )
    add_model_arg(parser)
    parser.add_argument(
        "--source",
        choices=["openwebtext", "sample"],
        default="openwebtext",
        help="The whole OpenWebText (streamed) or the fixed sample.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Only the first documents (for a quick test).",
    )
    parser.add_argument("--output-dir", type=Path, default=COUNTS_DIR)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model_name)
    if args.source == "sample":
        texts = (row["text"] for row in get_data(args.max_documents))
        total = args.max_documents
    else:
        dataset = load_dataset(OPENWEBTEXT, split="train", streaming=True)
        texts = (row["text"] for row in dataset)
        total = dataset.info.splits["train"].num_examples
        if args.max_documents is not None:
            texts = itertools.islice(texts, args.max_documents)
            total = args.max_documents
    counts = count(texts, tokenizer, vocab_size=len(tokenizer), total=total)
    counts.save(args.output_dir)
    print(
        f"saved {args.output_dir}: {int(counts.tokens.sum())} tokens, "
        f"{counts.bigrams.nnz} distinct bigrams"
    )


def attention_rows_main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-0 attention from the given query positions, for the first "
        "documents of the OpenWebText sample (<|endoftext|> prepended, truncated to "
        "1024 tokens)."
    )
    add_model_arg(parser)
    parser.add_argument("--num-documents", type=int, default=100)
    parser.add_argument("--positions", type=int, nargs="+", default=[500])
    args = parser.parse_args()
    texts = [row["text"] for row in get_data(args.num_documents)]
    for position, rows in extract_attention_rows(
        texts, args.positions, args.model_name
    ).items():
        path = attention_rows_path(position)
        rows.save(path)
        print(
            f"saved {path}: {len(rows.document_index)} documents with position {position}"
        )
