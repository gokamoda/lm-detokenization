"""Commands for the contribution of the six terms (Section 6.1)."""

import argparse
from pathlib import Path

import numpy as np
import torch
from feature_extractor.models import load_tokenizer

from lm_detokenization.analysis.six_terms import compute_contributions
from lm_detokenization.cli.args import (
    SIX_TERMS_PATH,
    add_figure_args,
    add_model_arg,
    figure_target,
)
from lm_detokenization.data.openwebtext import get_data
from lm_detokenization.plots.save import save_figure
from lm_detokenization.plots.six_terms import MEAN_OVER_HEADS, plot_six_terms
from lm_detokenization.weights import load_layer0_weights


def compute_main() -> None:
    parser = argparse.ArgumentParser(
        description="Contribution (KL) of each term, for the first documents of the "
        "OpenWebText sample (<|endoftext|> is prepended to each)."
    )
    add_model_arg(parser)
    parser.add_argument("--num-documents", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=SIX_TERMS_PATH)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Where to compute: cpu, cuda (or cuda:N), mps. About 1 GB of GPU "
        "memory for GPT-2 (the peak is printed for cuda).",
    )
    args = parser.parse_args()
    texts = [row["text"] for row in get_data(args.num_documents)]
    compute_contributions(
        load_layer0_weights(args.model_name),
        load_tokenizer(args.model_name),
        texts,
        args.output,
        device=args.device,
    )
    print(f"saved {args.output}")
    if args.device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated(args.device) / 2**30
        print(f"peak GPU memory: {peak:.2f} GiB")


def plot_main() -> None:
    parser = argparse.ArgumentParser(description="Contribution of each term, per head.")
    add_figure_args(parser, width=1.0, aspect=0.35)
    parser.add_argument("--contributions", type=Path, default=SIX_TERMS_PATH)
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[1, 7],
        help=f"Heads; {MEAN_OVER_HEADS} for the average over heads.",
    )
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--legend", choices=["top", "right", "none"], default="top")
    args = parser.parse_args()
    contributions = np.load(args.contributions, mmap_mode="r")
    fig = plot_six_terms(
        contributions,
        args.heads,
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        ncols=args.ncols,
        legend=args.legend,
    )
    save_figure(fig, args.output, target=figure_target(args))
    print(f"saved {args.output}")
