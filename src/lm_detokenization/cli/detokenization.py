"""Commands for the token affinity T^ee and detokenization (Section 4, Figure 2)."""

import argparse
from pathlib import Path

import polars as pl
from feature_extractor.models import load_tokenizer

from lm_detokenization.analysis.detokenization import (
    TokenAffinity,
    compute_auroc,
    mean_auroc_by_head,
    top_prefixes,
)
from lm_detokenization.cli.args import (
    AUROC_PATH,
    COUNTS_DIR,
    add_figure_args,
    add_model_arg,
    figure_target,
)
from lm_detokenization.data.frequency import Counts
from lm_detokenization.plots.detokenization import plot_roc
from lm_detokenization.plots.save import save_figure
from lm_detokenization.weights import load_layer0_weights


def _affinity(model_name: str) -> TokenAffinity:
    return TokenAffinity.from_weights(load_layer0_weights(model_name))


def _suffix_id(tokenizer, word: str) -> int:
    """Last token of `word`, as in the published search_from_str."""
    return tokenizer.encode(word, add_special_tokens=False)[-1]


def _token_text(tokenizer, token_id: int) -> str:
    """Token as written in the paper, with "_" for a leading space."""
    return tokenizer.convert_ids_to_tokens(token_id).replace("Ġ", "_")


def auroc_main() -> None:
    parser = argparse.ArgumentParser(
        description="AUROC of T^ee for every suffix token and head, with bigram counts "
        "as positives (the mean over suffixes is Figure 2 D)."
    )
    add_model_arg(parser)
    parser.add_argument("--counts-dir", type=Path, default=COUNTS_DIR)
    parser.add_argument(
        "--max-suffixes",
        type=int,
        default=None,
        help="Only the first suffix ids (for a quick test).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=AUROC_PATH)
    args = parser.parse_args()
    affinity = _affinity(args.model_name)
    bigrams = Counts.load(args.counts_dir).bigrams.tocsc()
    vocab_size = affinity.wte.shape[0]
    suffix_ids = list(
        range(
            vocab_size
            if args.max_suffixes is None
            else min(args.max_suffixes, vocab_size)
        )
    )
    auroc = compute_auroc(affinity, bigrams, suffix_ids, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    auroc.write_parquet(args.output)
    print(f"saved {args.output}")
    with pl.Config(tbl_rows=100):
        print(mean_auroc_by_head(auroc))


def auroc_table_main() -> None:
    parser = argparse.ArgumentParser(description="Mean AUROC per head (Figure 2 D).")
    parser.add_argument("--auroc", type=Path, default=AUROC_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Also write a CSV.")
    args = parser.parse_args()
    table = mean_auroc_by_head(pl.read_parquet(args.auroc))
    with pl.Config(tbl_rows=100):
        print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(args.output)
        print(f"saved {args.output}")


def top_prefixes_main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefixes with the highest T^ee for a current token (Figure 2 A)."
    )
    add_model_arg(parser)
    parser.add_argument(
        "--words",
        nargs="+",
        default=["iens", "tarian", " Jackson"],
        help="The last token of each word is the current token.",
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[4, 7])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None, help="Also write a CSV.")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model_name)
    affinity = _affinity(args.model_name)
    rows = []
    for word in args.words:
        suffix_id = _suffix_id(tokenizer, word)
        for head in args.heads:
            for rank, prefix_id in top_prefixes(affinity, suffix_id, head, args.k):
                rows.append(
                    {
                        "head": head,
                        "token_i": _token_text(tokenizer, suffix_id),
                        "token_j": _token_text(tokenizer, prefix_id),
                        "rank": rank,
                        "detokenization": _token_text(tokenizer, prefix_id)
                        + _token_text(tokenizer, suffix_id).lstrip("_"),
                    }
                )
    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=200):
        print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(args.output)
        print(f"saved {args.output}")


def roc_main() -> None:
    parser = argparse.ArgumentParser(description="ROC curves of T^ee (Figure 2 C).")
    add_model_arg(parser)
    add_figure_args(parser, width=0.4, aspect=0.55)
    parser.add_argument("--counts-dir", type=Path, default=COUNTS_DIR)
    parser.add_argument("--word", default="iens")
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 7])
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model_name)
    suffix_id = _suffix_id(tokenizer, args.word)
    text = _token_text(tokenizer, suffix_id)
    fig = plot_roc(
        _affinity(args.model_name),
        Counts.load(args.counts_dir).bigrams.tocsc(),
        [(suffix_id, text, h) for h in args.heads],
        target=figure_target(args),
        width=args.width,
        aspect=args.aspect,
    )
    save_figure(fig, args.output, target=figure_target(args))
    print(f"saved {args.output}")
