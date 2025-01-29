from pathlib import Path

import polars as pl
from tqdm import tqdm

from utils.mylogger import init_logging
from utils.path import WORK_DIR

LOG_PATH = "Mean AUROC by head.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


def main():
    prior_cols = []
    dfs = []
    for suffix_id in tqdm(range(50257)):
        file_path = Path(f"detokenization_auc2/{suffix_id}.csv")
        df = pl.read_csv(file_path)
        dfs.append(df)

        if prior_cols == df.columns:
            continue
        elif prior_cols == []:
            prior_cols = df.columns
            continue
        else:
            raise ValueError(f"Columns of {file_path} do not match prior columns")

    df = pl.concat(dfs)
    with pl.Config(tbl_rows=100):
        logger.info(
            df.group_by("head")
            .agg(
                pl.mean("auroc"),
            )
            .sort("auroc", descending=True)
        )
