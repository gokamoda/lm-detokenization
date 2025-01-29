from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torchtyping import TensorType
from transformers import AutoModelForCausalLM, AutoTokenizer

from eqmodels.gpt2 import EQGPT2LMHeadModel, compute_compare_score
from utils.ln import get_var_matrix
from utils.mylogger import init_logging
from utils.mytorchtyping import HEAD, HIDDEN_DIM, POS, VOCAB
from utils.path import WORK_DIR

plt.rc("font", size=100)
plt.rcParams["axes.xmargin"] = 0.001

LOG_PATH = "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH, clear=True)


class DetokenizationDataRetriever:
    offset = 0

    def __init__(
        self,
        model_name: str = "gpt2",
        counter: Counter = None,
        verbose: bool = False,
        print_fc=logger.info,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.verbose = verbose
        self.counter = counter

        wpe: TensorType[POS, HIDDEN_DIM] = (
            AutoModelForCausalLM.from_pretrained(model_name)
            .transformer.wpe.weight.detach()
            .cpu()
        )
        wte: TensorType[VOCAB, HIDDEN_DIM] = (
            AutoModelForCausalLM.from_pretrained(model_name)
            .transformer.wte.weight.detach()
            .cpu()
        )
        var_matrix: TensorType[VOCAB] = (
            torch.sqrt(get_var_matrix(wpe, wte) + 1e-5).to(torch.float16).mean(dim=0)
        )
        self.wte: TensorType[VOCAB, HIDDEN_DIM] = wte / var_matrix.unsqueeze(-1)

        self.wqkh = (
            EQGPT2LMHeadModel.from_pretrained(model_name)
            .transformer.h[0]
            .attn.wqkh.detach()
        )

        self.print_fc = print_fc

    def get_detokenization_scores(
        self,
        suffix_id: int,
        wte: TensorType[VOCAB, HIDDEN_DIM],
        wqkh: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM],
    ) -> TensorType[HEAD, VOCAB]:
        scores: TensorType[HEAD, VOCAB] = compute_compare_score(
            i=wte[0, suffix_id : suffix_id + 1].unsqueeze(0), j=wte, w=wqkh.detach()
        ).squeeze()
        return scores

    def search_from_str(self, query: str):
        suffix_id = self.tokenizer.encode(query)[-1 - self.offset]
        if self.verbose:
            self.print_fc(f'suffix: "{self.tokenizer.decode(suffix_id)}" ({suffix_id})')
        return self.search_from_id(suffix_id)

    def search_from_id(self, suffix_id: int) -> pl.DataFrame:
        scores: TensorType[HEAD, VOCAB] = self.get_detokenization_scores(
            suffix_id, self.wte.unsqueeze(0), self.wqkh
        )
        n_head, n_vocab = scores.shape
        df = pl.DataFrame(
            {
                "prefix_id": np.tile(range(n_vocab), n_head),
                "suffix_id": np.repeat(suffix_id, n_head * n_vocab),
                "head": np.repeat(range(n_head), n_vocab),
                "score": scores.cpu().numpy().flatten(),
            }
        )

        if self.counter is not None:
            counts = [
                self.counter[(prefix_id, suffix_id)] for prefix_id in range(n_vocab)
            ]
            df = df.with_columns(pl.Series("count", np.tile(counts, n_head)))
        return df


def main():
    counter = torch.load("freqs/bitoken/wikitext103_gpt2.pt", weights_only=False)
    retriever = DetokenizationDataRetriever(
        model_name="gpt2", verbose=True, counter=counter
    )
    df = retriever.search_from_str(" sapiens")
    print(df)
