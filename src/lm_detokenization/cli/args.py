"""Arguments shared by the commands, and where their outputs go by default.

Paths are relative to the repository root, where the commands are run.
"""

import argparse
from pathlib import Path

from lm_detokenization.data.openwebtext import counts_dir
from lm_detokenization.plots.targets import TARGETS, FigureTarget

DEFAULT_MODEL = "gpt2"
OUTPUTS = Path("outputs")
SIX_TERMS_PATH = OUTPUTS / "six_terms" / "gpt2" / "contributions.npy"
ATTENTION_DIR = OUTPUTS / "attention" / "gpt2"
AUROC_PATH = OUTPUTS / "detokenization" / "gpt2" / "auroc.parquet"


def attention_rows_path(position: int) -> Path:
    return ATTENTION_DIR / f"layer00_i{position}.pt"


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model name (only GPT-2 is supported).",
    )


def add_counts_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--counts-dir",
        type=Path,
        default=None,
        help="Token and bigram counts made by count-frequency "
        "(default: outputs/freqs/openwebtext/<model>).",
    )


def resolve_counts_dir(args: argparse.Namespace) -> Path:
    return args.counts_dir or counts_dir(args.model_name)


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
