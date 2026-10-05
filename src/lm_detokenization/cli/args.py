"""Arguments shared by the commands, and where their outputs go by default.

Paths are relative to the repository root, where the commands are run.
"""

import argparse
from pathlib import Path

from lm_detokenization.data.corpus import CORPORA, DEFAULT_CORPUS, counts_dir
from lm_detokenization.plots.targets import TARGETS, FigureTarget

DEFAULT_MODEL = "gpt2"
OUTPUTS = Path("outputs")


def model_dir(model_name: str) -> str:
    """Directory name of a model's outputs, e.g. gpt2, rinna--japanese-gpt2-small."""
    return model_name.replace("/", "--")


def six_terms_path(model_name: str) -> Path:
    return OUTPUTS / "six_terms" / model_dir(model_name) / "contributions.npy"


def attention_rows_path(model_name: str, position: int) -> Path:
    return OUTPUTS / "attention" / model_dir(model_name) / f"layer00_i{position}.pt"


def auroc_path(model_name: str) -> Path:
    return OUTPUTS / "detokenization" / model_dir(model_name) / "auroc.parquet"


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model name: GPT-2 or a model of the same architecture "
        "(e.g. rinna/japanese-gpt2-small). Outputs are under its name.",
    )


def add_corpus_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPORA),
        default=DEFAULT_CORPUS,
        help="Corpus of the documents and of the counts (default: %(default)s).",
    )


def add_counts_dir_arg(parser: argparse.ArgumentParser) -> None:
    """--corpus and --counts-dir: the token and bigram counts to read."""
    add_corpus_arg(parser)
    parser.add_argument(
        "--counts-dir",
        type=Path,
        default=None,
        help="Token and bigram counts made by the frequency step of "
        "scripts/compute_data*.sh (default: those of --model-name and --corpus "
        "under outputs/corpus-tools).",
    )


def resolve_counts_dir(args: argparse.Namespace) -> Path:
    return args.counts_dir or counts_dir(args.model_name, args.corpus)


def add_figure_args(
    parser: argparse.ArgumentParser, *, width: float, aspect: float
) -> None:
    """--target, --width, --aspect and --output, as in inseg-attention."""
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="naacl2025",
        help="Document whose page size and font size the figure is made for.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=width,
        help="Figure width / textwidth of the target (a naacl2025 column is 0.48125).",
    )
    parser.add_argument(
        "--aspect", type=float, default=aspect, help="Figure height / width."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="PDF (or PNG) to write, e.g. outputs/figures/naacl2025/x.pdf.",
    )


def figure_target(args: argparse.Namespace) -> FigureTarget:
    return TARGETS[args.target]


def heads_arg(parser: argparse.ArgumentParser, default: list[int]) -> None:
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=default,
        help="Heads to plot (-1 alone: all heads).",
    )


def resolve_heads(heads: list[int], num_heads: int) -> list[int]:
    return list(range(num_heads)) if heads == [-1] else heads
