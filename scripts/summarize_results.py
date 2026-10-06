"""Print the results of a model as Markdown tables, for notes/.

- auroc: the mean AUROC per head, over all suffixes (those never seen count
  as 0, as in the paper), over the suffixes seen in the corpus, and over
  those weighted by the number of bigrams ending in them.
- attention: where the attention from position 500 goes, averaged over
  documents: to position 0, the far past (1..484), the recent past
  (485..499) and itself (500).

Usage (from the repository root):
    uv run python scripts/summarize_results.py auroc gpt2
    uv run python scripts/summarize_results.py attention rinna/japanese-gpt2-small
"""

import argparse

import polars as pl
import torch

from lm_detokenization.cli.args import attention_rows_path, auroc_path


def auroc_table(model: str) -> str:
    auroc = pl.read_parquet(auroc_path(model))
    seen = auroc.filter(pl.col("num_valid_detokenization") > 0)
    table = (
        auroc.group_by("head")
        .agg(pl.mean("auroc").alias("all"))
        .join(seen.group_by("head").agg(pl.mean("auroc").alias("seen")), on="head")
        .join(
            seen.group_by("head").agg(
                (
                    (pl.col("auroc") * pl.col("num_valid_detokenization")).sum()
                    / pl.col("num_valid_detokenization").sum()
                ).alias("weighted")
            ),
            on="head",
        )
        .sort("seen", descending=True)
    )
    num_heads = table.height
    unseen = (auroc.height - seen.height) // num_heads
    lines = [
        f"Suffixes never seen in the corpus: {unseen} of {auroc.height // num_heads}.",
        "",
        "| Head | " + " | ".join(str(h) for h in table["head"]) + " |",
        "|---|" + "---|" * num_heads,
    ]
    for column, label in [
        ("all", "All"),
        ("seen", "Seen only"),
        ("weighted", "Frequency weighted"),
    ]:
        lines.append(
            f"| {label} | " + " | ".join(f"{v:.3f}" for v in table[column]) + " |"
        )
    return "\n".join(lines)


def attention_table(model: str) -> str:
    rows = torch.load(attention_rows_path(model, 500), weights_only=False)
    weights = rows["weights"].float()  # [document, head, 501]
    lines = [
        f"{weights.shape[0]} documents.",
        "",
        "| Head | Position 0 | Far past (1–484) | Recent past (485–499) | Self (500) |",
        "|---|---|---|---|---|",
    ]
    for head in range(weights.shape[1]):
        mean = weights[:, head].mean(0)
        lines.append(
            f"| {head} | {mean[0]:.3f} | {mean[1:485].sum():.3f} "
            f"| {mean[485:500].sum():.3f} | {mean[500]:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("table", choices=["auroc", "attention"])
    parser.add_argument("model")
    args = parser.parse_args()
    print(
        auroc_table(args.model)
        if args.table == "auroc"
        else attention_table(args.model)
    )


if __name__ == "__main__":
    main()
