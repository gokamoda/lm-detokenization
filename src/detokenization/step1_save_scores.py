from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils.mylogger import init_logging
from utils.path import WORK_DIR

from .socres_and_counts import DetokenizationDataRetriever

LOG_PATH = "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


def single_process(suffix_ids: list[int], process_id: int):
    model_name = "gpt2"
    logger.info("Loading counter...")
    counter = torch.load("freqs/bitoken/openwebtext_gpt2.pt", weights_only=False)
    logger.info("Loaded counter.")

    data_retriever = DetokenizationDataRetriever(
        model_name=model_name, counter=counter, verbose=False
    )
    for suffix_id in tqdm(
        suffix_ids, desc=f"Process {process_id}", position=process_id
    ):
        save_path = Path(f"detokenization_auc2/{suffix_id}.csv")
        # save_path = WORK_DIR.joinpath(
        #     f"detokenization_scores_and_counts/{suffix_id}.jsonl"
        # )

        if save_path.exists():
            df = pl.read_csv(save_path)
            if df.height == 12:
                continue

        df = data_retriever.search_from_id(suffix_id)

        num_heads = df["head"].n_unique()
        num_valid_detokenization = df["count"].sum()

        result = []
        for head in range(num_heads):
            result_tmp = {
                "head": head,
                "id_i": suffix_id,
                "num_valid_detokenization": num_valid_detokenization,
            }
            df_head = df.filter(pl.col("head") == head)

            # AUROC
            y_true = []
            y_score = []
            bitoken_counts_nonzero = df_head.filter(pl.col("count") > 0)[
                "count"
            ].to_numpy()
            if len(bitoken_counts_nonzero) == 0:
                result_tmp["auroc"] = 0.0
                result.append(result_tmp)
            else:
                bitoken_counts_zero = df_head.filter(pl.col("count") == 0)[
                    "count"
                ].to_numpy()

                scores_zero = df_head.filter(pl.col("count") == 0)["score"].to_numpy()
                scores_nonzero = df_head.filter(pl.col("count") > 0)["score"].to_numpy()
                y_true = np.concatenate(
                    [
                        np.zeros(len(bitoken_counts_zero)),
                        np.ones(np.sum(bitoken_counts_nonzero)),
                    ]
                )
                y_score = np.concatenate(
                    [scores_zero, np.repeat(scores_nonzero, bitoken_counts_nonzero)]
                )

                result_tmp["auroc"] = roc_auc_score(y_true, y_score)

                result.append(result_tmp)
        df = pl.DataFrame(result)
        df.write_csv(save_path)

    return process_id


def main(num_workers: int = 1):
    done_path = Path("detokenization_auc2/done.txt")
    if done_path.exists():
        logger.info("Step1 already done.")
        return
    done_path.parent.mkdir(exist_ok=True, parents=True)

    num_vocab = 50257
    # num_vocab = 100

    with Pool(num_workers) as p:
        result = p.starmap_async(
            single_process,
            [(list(range(i, num_vocab, num_workers)), i) for i in range(num_workers)],
        )
        for value in result.get():
            logger.info(f"Finished process {value}")

    with open(Path("detokenization_auc2/done.txt"), "w") as f:
        f.write("done")
