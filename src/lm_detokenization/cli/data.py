"""Commands that prepare data from the OpenWebText sample: the observed
attention. The sample and the counts are made with the corpus-tools command
(see scripts/compute_data.sh)."""

import argparse

from lm_detokenization.analysis.empirical_attention import extract_attention_rows
from lm_detokenization.cli.args import add_model_arg, attention_rows_path
from lm_detokenization.data.openwebtext import get_data


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
