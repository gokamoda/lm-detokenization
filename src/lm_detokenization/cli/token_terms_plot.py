"""Commands for the plots of the token-derived terms and embedding variances."""

import argparse

from lm_detokenization.analysis.token_terms import tee_sample
from lm_detokenization.cli.args import (
    add_counts_dir_arg,
    add_figure_args,
    add_model_arg,
    figure_target,
    heads_arg,
    resolve_counts_dir,
    resolve_heads,
)
from lm_detokenization.data.corpus import load_token_counts
from lm_detokenization.plots import token_terms as plots
from lm_detokenization.plots.save import save_figure
from lm_detokenization.weights import load_layer0_weights


def _parser(
    description: str, *, width: float, aspect: float
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_model_arg(parser)
    add_figure_args(parser, width=width, aspect=aspect)
    return parser


def _save(fig, args) -> None:
    save_figure(fig, args.output, target=figure_target(args))
    print(f"saved {args.output}")


def tee_main() -> None:
    parser = _parser(
        "Heatmaps of T^ee between tokens sampled from the vocabulary.",
        width=1.0,
        aspect=0.8,
    )
    heads_arg(parser, [1, 7])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--without-ln", action="store_true")
    args = parser.parse_args()
    weights = load_layer0_weights(args.model_name)
    sample = tee_sample(weights, args.n_samples, seed=args.seed)
    fig = plots.plot_tee(
        sample,
        resolve_heads(args.heads, weights.num_heads),
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        ncols=args.ncols,
        without_ln=args.without_ln,
    )
    _save(fig, args)


def te_main() -> None:
    parser = _parser(
        "T^e of every token against its corpus count.", width=1.0, aspect=0.35
    )
    heads_arg(parser, [1, 7])
    add_counts_dir_arg(parser)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--without-ln", action="store_true")
    args = parser.parse_args()
    weights = load_layer0_weights(args.model_name)
    fig = plots.plot_te(
        weights,
        load_token_counts(resolve_counts_dir(args)),
        resolve_heads(args.heads, weights.num_heads),
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        ncols=args.ncols,
        without_ln=args.without_ln,
    )
    _save(fig, args)


def variance_main() -> None:
    parser = _parser(
        "Variance of token and position embeddings.", width=1.0, aspect=0.4
    )
    add_counts_dir_arg(parser)
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=plots.VARIANCE_PANELS,
        default=["wte", "wpe_ends"],
    )
    parser.add_argument("--num-ends", type=int, default=11)
    args = parser.parse_args()
    weights = load_layer0_weights(args.model_name)
    tokens = (
        load_token_counts(resolve_counts_dir(args)) if "wte" in args.panels else None
    )
    fig = plots.plot_variance(
        weights,
        tokens,
        tuple(args.panels),
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        num_ends=args.num_ends,
    )
    _save(fig, args)
