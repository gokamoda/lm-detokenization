from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import torch
from torchtyping import TensorType
from tqdm import tqdm
from transformers import GPT2Tokenizer

from eqmodels import EQGPT2LMHeadModel
from eqmodels.gpt2 import (
    compute_compare_score,
    compute_self_score,
)
from utils.mytorchtyping import (
    HEAD,
    HIDDEN_DIM,
    LAYER,
    POS,
    SEQUENCE,
    VOCAB,
)
from utils.path import WORK_DIR

from .utils import get_data

plt.rc("font", size=70)
plt.rcParams["axes.xmargin"] = 0.001
linewidth = 10

NUM_INSTANCES = 5000


def compute_6terms(
    prompt: str,
    tokenizer: GPT2Tokenizer,
    wte: TensorType[VOCAB, HIDDEN_DIM],
    wpe: TensorType[POS, HIDDEN_DIM],
    w_compare: TensorType[HEAD, HIDDEN_DIM, HIDDEN_DIM],
    w_self: TensorType[HEAD, HIDDEN_DIM],
):
    tokens = tokenizer(prompt, return_tensors="pt")

    if tokens.input_ids.shape[1] > 1024:
        tokens.input_ids = tokens.input_ids[:, :1024]
        tokens.attention_mask = tokens.attention_mask[:, :1024]

    tok_emb: TensorType[1, SEQUENCE, HIDDEN_DIM] = wte[tokens.input_ids]
    pos_enc: TensorType[1, SEQUENCE, HIDDEN_DIM] = wpe[
        : tokens.input_ids.shape[1]
    ].unsqueeze(0)

    variances: TensorType[SEQUENCE] = (tok_emb + pos_enc).var(
        dim=-1, keepdim=True, unbiased=False
    )
    variances = torch.sqrt(variances + 1e-5)

    tok_emb = tok_emb / variances
    pos_enc = pos_enc / variances

    embi_embj: TensorType[1, HEAD, SEQUENCE, SEQUENCE] = compute_compare_score(
        tok_emb, tok_emb, w_compare
    )
    embj: TensorType[1, HEAD, SEQUENCE] = compute_self_score(tok_emb, w_self)
    posi_posj: TensorType[1, HEAD, SEQUENCE, SEQUENCE] = compute_compare_score(
        pos_enc, pos_enc, w_compare
    )
    posj: TensorType[1, HEAD, SEQUENCE] = compute_self_score(pos_enc, w_self)
    embi_posj: TensorType[1, HEAD, SEQUENCE, SEQUENCE] = compute_compare_score(
        tok_emb, pos_enc, w_compare
    )
    posi_embj: TensorType[1, HEAD, SEQUENCE, SEQUENCE] = compute_compare_score(
        pos_enc, tok_emb, w_compare
    )

    return {
        "posi_posj": posi_posj,
        "posj": posj,
        "embi_embj": embi_embj,
        "embj": embj,
        "embi_posj": embi_posj,
        "posi_embj": posi_embj,
    }


def main():
    kls = []
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_name = "gpt2"

    wte = (
        EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.wte.weight.detach()
        .cpu()
    )
    wpe = (
        EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.wpe.weight.detach()
        .cpu()
    )
    w_compare = (
        EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.wqkh.detach()
        .cpu()
    )
    w_self = (
        EQGPT2LMHeadModel.from_pretrained(model_name)
        .transformer.h[0]
        .attn.bqwkh.detach()
        .cpu()
    )

    kls = []
    done_ids = []

    path = WORK_DIR.joinpath("latest/6terms_importance_kl.jsonl")
    if path.exists():
        kls_df = pl.read_ndjson(path)
        done_ids = kls_df["instance_id"].unique().to_list()
        print(f"Already done: {len(done_ids)}")

    for r, row in tqdm(enumerate(get_data())):
        if r in done_ids:
            print(f"Skipping {r}")
            continue

        if r == NUM_INSTANCES:
            break
        # https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/interactive_conditional_samples.py
        prompt = "<|endoftext|>" + row["text"]
        scores = compute_6terms(
            prompt=prompt,
            tokenizer=tokenizer,
            wte=wte,
            wpe=wpe,
            w_compare=w_compare,
            w_self=w_self,
        )

        _, n_head, max_length = scores["posj"].shape

        for head in range(n_head):
            for length in range(max_length):
                original_attn_score = (
                    scores["embi_embj"][0, head, length, : length + 1]
                    + scores["embj"][0, head, : length + 1]
                    + scores["posi_posj"][0, head, length, : length + 1]
                    + scores["posj"][0, head, : length + 1]
                    + scores["embi_posj"][0, head, length, : length + 1]
                    + scores["posi_embj"][0, head, length, : length + 1]
                )

                ablated_attn_score = {
                    "embi_embj": original_attn_score
                    - scores["embi_embj"][0, head, length, : length + 1],
                    "embj": original_attn_score - scores["embj"][0, head, : length + 1],
                    "posi_posj": original_attn_score
                    - scores["posi_posj"][0, head, length, : length + 1],
                    "posj": original_attn_score - scores["posj"][0, head, : length + 1],
                    "embi_posj": original_attn_score
                    - scores["embi_posj"][0, head, length, : length + 1],
                    "posi_embj": original_attn_score
                    - scores["posi_embj"][0, head, length, : length + 1],
                }

                original_weight = torch.nn.functional.softmax(
                    original_attn_score, dim=-1
                )
                for k, v in ablated_attn_score.items():
                    kls.append(
                        {
                            "instance_id": r,
                            "head": head,
                            "i": length,
                            "type": k,
                            "kl": torch.nn.functional.kl_div(
                                input=torch.nn.functional.log_softmax(v, dim=-1),
                                target=original_weight,
                                reduction="mean",
                            ).item(),
                        }
                    )

    if path.exists():
        kls_df = pl.concat([kls_df, pl.DataFrame(kls)])
        kls_df.write_ndjson(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(kls).write_ndjson(path)


def vis():
    path = WORK_DIR.joinpath("latest/6terms_importance_kl.jsonl")
    save_dir = Path("figures/6terms_importance")
    save_dir.mkdir(parents=True, exist_ok=True)

    df_original = pl.read_ndjson(path).to_pandas()
    num_heads = 12
    for head in tqdm(range(num_heads + 1), desc="head", total=num_heads + 1):
        if head != num_heads:
            continue
            df = df_original.loc[df_original["head"] == head]
        else:
            print("Average over all heads")
            df = df_original.groupby(["i", "type", "instance_id"]).mean().reset_index()

        df = df.loc[(df["i"] != 0) & (df["i"] != 1023)]

        df = df.loc[df["kl"] > 1e-10]

        # change "embi_embj" to "ee", "embj" to "e", "posi_posj" to "pp", "posj" to "p", "embi_posj" to "ep", "posi_embj" to "pe"
        df["type"] = df["type"].replace(
            {
                "embi_embj": "ee",
                "embj": "e",
                "posi_posj": "pp",
                "posj": "p",
                "embi_posj": "ep",
                "posi_embj": "pe",
            }
        )

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30, 20))

        sns.lineplot(
            data=df,
            x="i",
            y="kl",
            hue="type",
            ax=ax,
            linewidth=linewidth,
            hue_order=["ee", "e", "pp", "p", "ep", "pe"],
        )
        ax.legend(title="Term", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_yscale("log")
        if head != num_heads:
            ax.set_ylabel("$c_{i, " + str(head) + "}$")
        else:
            ax.set_ylabel("$c_{i}$")
        ax.set_xlabel("$i$")
        if head != num_heads:
            fig.savefig(save_dir.joinpath(f"head-{head}.pdf"), bbox_inches="tight")
        else:
            fig.savefig(save_dir.joinpath("head-all.pdf"), bbox_inches="tight")
