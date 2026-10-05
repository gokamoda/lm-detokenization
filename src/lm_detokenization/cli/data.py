"""Commands that prepare data: the OpenWebText sample, corpus counts and the
observed attention."""

import argparse
from pathlib import Path

from corpus_tools.corpus import run_and_exit
from feature_extractor.models import load_tokenizer

from lm_detokenization.analysis.empirical_attention import extract_attention_rows
from lm_detokenization.cli.args import add_model_arg, attention_rows_path
from lm_detokenization.data.openwebtext import (
    NUM_SAMPLES,
    count_frequency,
    counts_dir,
    get_data,
    make_sample,
    open_texts,
    sample_path,
)


def sample_openwebtext_main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a fixed random sample of OpenWebText (the hash sample of "
        "corpus-tools). Reads the whole dataset once by streaming (about 24GB of "
        "download), but stores only the sampled rows."
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Default: data/openwebtext/hash_n<NUM_SAMPLES>.jsonl.",
    )
    args = parser.parse_args()
    output_path = args.output_path or sample_path(args.num_samples)
    num_rows = make_sample(args.num_samples, output_path)
    print(f"saved {output_path}: {num_rows} documents")


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/freqs/openwebtext/<model>.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or counts_dir(args.model_name)
    tokenizer = load_tokenizer(args.model_name)
    if args.source == "sample":
        texts = [row["text"] for row in get_data(args.max_documents)]
        tokens, bigrams = count_frequency(
            texts, tokenizer, output_dir, total=len(texts)
        )
    else:
        with open_texts(args.max_documents) as (texts, total):
            tokens, bigrams = count_frequency(texts, tokenizer, output_dir, total=total)
    print(
        f"saved {output_dir}: {int(tokens.sum())} tokens, "
        f"{bigrams.nnz} distinct bigrams"
    )


# The commands that stream OpenWebText end with corpus-tools' run_and_exit, as a
# process that leaves a stream before its end (--max-documents, an error,
# Ctrl-C) may otherwise never exit.


def sample_openwebtext_cli() -> None:
    run_and_exit(sample_openwebtext_main)


def count_frequency_cli() -> None:
    run_and_exit(count_frequency_main)


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
