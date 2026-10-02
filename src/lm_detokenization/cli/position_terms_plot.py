"""Commands for the plots of the position-derived terms (Figure 3, Appendix D)."""

import argparse
from pathlib import Path

from lm_detokenization.analysis.empirical_attention import AttentionRows
from lm_detokenization.analysis.position_terms import inverse_sigma_stats
from lm_detokenization.cli.args import (
    add_figure_args,
    add_model_arg,
    attention_rows_path,
    figure_target,
    heads_arg,
    resolve_heads,
)
from lm_detokenization.plots import position_terms as plots
from lm_detokenization.plots.save import save_figure
from lm_detokenization.weights import load_layer0_weights


def _load(args):
    weights = load_layer0_weights(args.model_name)
    return weights, inverse_sigma_stats(weights.sigma)


def _save(fig, args) -> None:
    save_figure(fig, args.output, target=figure_target(args))
    print(f"saved {args.output}")


def _parser(
    description: str, *, width: float, aspect: float
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_model_arg(parser)
    add_figure_args(parser, width=width, aspect=aspect)
    return parser


def tp_main() -> None:
    parser = _parser("T^p_{j,h} for each head.", width=1.0, aspect=0.9)
    heads_arg(parser, [1, 7])
    parser.add_argument("--ncols", type=int, default=1)
    parser.add_argument(
        "--without-ln",
        action="store_true",
        help="Also show b^QK p_j^T next to each head.",
    )
    args = parser.parse_args()
    weights, stats = _load(args)
    fig = plots.plot_tp(
        weights,
        stats,
        resolve_heads(args.heads, weights.num_heads),
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        ncols=args.ncols,
        without_ln=args.without_ln,
    )
    _save(fig, args)


def tpp_main() -> None:
    parser = _parser(
        "T^pp_{i,j,h}: rows are heads, columns are query positions i.",
        width=1.0,
        aspect=0.5,
    )
    heads_arg(parser, [1, 7])
    parser.add_argument("--pos-i", type=int, nargs="+", default=[500])
    parser.add_argument("--without-ln", action="store_true")
    parser.add_argument("--legend", action="store_true")
    args = parser.parse_args()
    weights, stats = _load(args)
    fig = plots.plot_tpp(
        weights,
        stats,
        resolve_heads(args.heads, weights.num_heads),
        args.pos_i,
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        without_ln=args.without_ln,
        legend=args.legend,
    )
    _save(fig, args)


def tp_tpp_main() -> None:
    parser = _parser(
        "T^pp_{i,j,h} + T^p_{j,h} before and/or after softmax.", width=1.0, aspect=0.5
    )
    heads_arg(parser, [1, 7])
    parser.add_argument("--pos-i", type=int, nargs="+", default=[500])
    parser.add_argument(
        "--views", nargs="+", choices=["before", "after"], default=["before", "after"]
    )
    parser.add_argument("--legend", action="store_true")
    args = parser.parse_args()
    weights, stats = _load(args)
    fig = plots.plot_tp_tpp(
        weights,
        stats,
        resolve_heads(args.heads, weights.num_heads),
        args.pos_i,
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        views=tuple(args.views),
        legend=args.legend,
    )
    _save(fig, args)


def undertrained_main() -> None:
    parser = _parser(
        "p_i W^QK p_j^T for the last query positions.", width=1.0, aspect=0.35
    )
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--num-last", type=int, default=5)
    args = parser.parse_args()
    weights, _ = _load(args)
    fig = plots.plot_undertrained(
        weights,
        args.head,
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        num_last=args.num_last,
    )
    _save(fig, args)


def _attention_arg(parser) -> None:
    parser.add_argument("--pos-i", type=int, default=500)
    parser.add_argument(
        "--attention",
        type=Path,
        default=None,
        help="Attention rows made by attention-rows "
        "(default: outputs/attention/gpt2/layer00_i<pos-i>.pt).",
    )


def vs_tptpp_main() -> None:
    parser = _parser(
        "Observed attention from position i vs. the prediction from T^pp + T^p.",
        width=1.0,
        aspect=0.5,
    )
    heads_arg(parser, [1, 7])
    _attention_arg(parser)
    parser.add_argument("--begin-offset", type=int, default=485)
    parser.add_argument(
        "--zoom-only",
        action="store_true",
        help="Only j >= begin-offset (Figure 3 E), not the whole range.",
    )
    parser.add_argument("--legend", action="store_true")
    args = parser.parse_args()
    weights, stats = _load(args)
    rows = AttentionRows.load(args.attention or attention_rows_path(args.pos_i))
    fig = plots.plot_vs_tptpp(
        weights,
        stats,
        rows,
        resolve_heads(args.heads, weights.num_heads),
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        begin_offset=args.begin_offset,
        zoom_only=args.zoom_only,
        legend=args.legend,
    )
    _save(fig, args)


def position_bias_main() -> None:
    parser = _parser(
        "Figure 3: T^p, T^pp, their sum, its softmax and the observed "
        "attention, one row per head.",
        width=1.0,
        aspect=0.4,
    )
    heads_arg(parser, [1, 7])
    _attention_arg(parser)
    parser.add_argument("--begin-offset", type=int, default=485)
    parser.add_argument(
        "--no-observed",
        action="store_true",
        help="Leave out column E (no attention rows needed).",
    )
    args = parser.parse_args()
    weights, stats = _load(args)
    rows = (
        None
        if args.no_observed
        else AttentionRows.load(args.attention or attention_rows_path(args.pos_i))
    )
    fig = plots.plot_position_bias(
        weights,
        stats,
        rows,
        resolve_heads(args.heads, weights.num_heads),
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
        pos_i=args.pos_i,
        begin_offset=args.begin_offset,
    )
    _save(fig, args)
