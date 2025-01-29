from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM

plt.rc("font", size=70)
plt.rcParams["axes.xmargin"] = 0.01


def main(**kwargs):
    wpe: torch.Tensor = AutoModelForCausalLM.from_pretrained(
        "gpt2"
    ).transformer.wpe.weight.detach()

    linewidth = 10

    fig = plt.figure(figsize=(40, 15))
    top, bottom = fig.subfigures(2, 1)
    tl, tr = top.subplots(1, 2)
    bottom = bottom.subplots(1, 1)

    bottom.plot(wpe.var(correction=False, dim=-1).numpy(), linewidth=linewidth)
    bottom.set_xlabel("Position ($i$)")
    bottom.set_title("Var($p_i$)", y=0.8, loc="left", x=0.02)
    bottom.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    bottom.yaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")

    n = 11
    tl.plot(
        wpe[:n].var(correction=False, dim=-1).numpy(),
        linewidth=linewidth,
        marker="o",
        markersize=20,
    )
    tr.plot(
        wpe[-n:].var(correction=False, dim=-1).numpy(),
        linewidth=linewidth,
        marker="o",
        markersize=20,
    )

    tr.set_xticks(tr.get_xticks()[1:-1])
    tr.set_xticklabels(map(int, tr.get_xticks() + wpe.shape[0] - n))
    tl.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    tl.yaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    tl.set_title("Var($p_i$)", y=0.8, loc="left", x=0.02)

    tr.xaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    tr.yaxis.set_tick_params(width=linewidth, length=linewidth * 2, direction="out")
    tr.set_title("Var($p_i$)", y=0.8, loc="left", x=0.02)

    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True, parents=True)

    fig.savefig(save_dir.joinpath("wpe_variance.pdf"), bbox_inches="tight")
