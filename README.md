

# GPT2 Detokenization
This repository contains code for the paper  
[Weight-based Analysis of Detokenization in Language Models:
Understanding the First Stage of Inference Without Inference](https://arxiv.org/abs/2501.15754)  
by Go Kamoda, Benjamin Heinzerling, Tatsuro Inaba, Keito Kudo, Keisuke Sakaguchi and Kentaro Inui.

## Relation to the published version

The code used for the paper is commit [`6f3951a`](https://github.com/gokamoda/lm-detokenization/tree/6f3951af780b505d91356660c2b45fe541053194) (`main`).
This code has been updated from it, and the OpenWebText documents used for the observed attention and the six-term decomposition differ from those of the paper.
The paper took the first documents of OpenWebText after `.shuffle(seed=42)`, which can no longer be reproduced since the dataset on the Hugging Face Hub was converted to Parquet; this code takes the documents with the smallest sha256 of their text instead (the hash sample of [corpus-tools](https://github.com/gokamoda/corpus-tools)).
The numbers therefore differ from those in the paper, but the conclusions are broadly consistent with it.

## Create environment

```
make install
```

This installs PyTorch 2.7.0 for CUDA 12.6 if `nvidia-smi` finds a GPU, and for CPU otherwise. On a server whose NVIDIA driver is too old for CUDA 12 (e.g. a driver of CUDA 11.4, where PyTorch warns "The NVIDIA driver on your system is too old"), install the CUDA 11.8 build instead:

```
make install-cu118
```

Run every command below from the repository root. Each command has `--help`.

## Data

The figures read data under `outputs/`. Make it with
```
bash scripts/compute_data.sh
```
or only some steps, e.g. `bash scripts/compute_data.sh attention six_terms`.

| Step | Command | Output | Time |
|---|---|---|---|
| `sample` | `corpus-tools sample` | `outputs/corpus-tools/.../samples/hash_n10000.jsonl` | ~30 min (streams ~24GB) |
| `frequency` | `corpus-tools count` | `outputs/corpus-tools/.../all/gpt2/counts/nobos/` | ~2 h; ~30 min with `CPUS=16` |
| `attention` | `uv run attention-rows` | `outputs/attention/gpt2/layer00_i500.pt` | ~20 s |
| `six_terms` | `uv run six-terms` | `outputs/six_terms/gpt2/contributions.npy` | ~45 min on CPU; ~1 min with `DEVICE=cuda` |
| `auroc` | `uv run detok-auroc` | `outputs/detokenization/gpt2/auroc.parquet` | ~2 h 20 min on a server CPU; ~35 s with `DEVICE=cuda` |

`sample` and `frequency` run the [corpus-tools](https://github.com/gokamoda/corpus-tools) command, streaming OpenWebText at a fixed commit, and save under `outputs/corpus-tools/Skylion007--openwebtext/plain_text/train/`. The frequency is of all of OpenWebText, apart from the sample. `CPUS=16 bash scripts/compute_data.sh frequency` counts with 16 threads in all (corpus-tools `--cpus`).

The OpenWebText sample is the 10,000 documents with the smallest sha256 of the text; `attention` and `six_terms` use its first documents.
`DEVICE=cuda bash scripts/compute_data.sh six_terms auroc` computes these two on a GPU (`--device`; about 0.7 and 2.0 GiB of GPU memory for GPT-2), with the same results up to rounding; without it, `auroc` uses scikit-learn on the CPU, as published.

For a quick test, `six-terms --num-documents N` and `detok-auroc --max-suffixes N` work on part of the data (give them an `--output`/`--output-dir` so the full results are not overwritten).

## Figures

Each figure command takes the document it is for and the printed size:
`--target naacl2025|thesis` (page and font size), `--width` (fraction of the textwidth; a naacl2025 column is 0.48125) and `--aspect` (height / width).
```
uv run position-bias-plot --target thesis --width 1.0 --aspect 0.288 --output outputs/figures/x.pdf
```

| Command | Figure |
|---|---|
| `tee-plot` | $T^{ee}$ heatmaps (Fig. 2 B) |
| `roc-plot` | ROC of $T^{ee}$ for a token (Fig. 2 C) |
| `detok-top-prefixes`, `detok-auroc-table` | Tables of Fig. 2 A and D |
| `position-bias-plot` | Fig. 3 in one figure (columns A–E) |
| `tp-plot`, `tpp-plot`, `tp-tpp-plot`, `vs-tptpp-plot` | $T^p$, $T^{pp}$, $T^p + T^{pp}$ before/after softmax, observed attention (parts of Fig. 3, appendix) |
| `six-terms-plot` | Contribution of the six terms (Fig. 4) |
| `te-plot` | $T^e$ against token counts |
| `variance-plot` | Variance of token and position embeddings |
| `undertrained-plot` | Undertrained position embeddings |

The sizes used in each document are recorded in scripts that write all of its figures:
```
bash scripts/naacl2025_figures.sh            # -> outputs/figures/naacl2025/
bash scripts/thesis_figures.sh               # -> outputs/figures/thesis/
bash scripts/thesis_figures.sh position_bias # only the named figures
```
They do not write into the LaTeX projects; copy the figures there after checking them.

To make the data and then every figure from scratch, run `bash scripts/all.sh` (about an hour with `CPUS=16 DEVICE=cuda bash scripts/all.sh` on a server with a GPU).

## Notebooks

- `notebooks/prefix_search.ipynb`: for a current token (suffix), $T^{ee}$ and the OpenWebText bigram count of every past token (prefix), and the ROC of a head. Needs the frequency (`bash scripts/compute_data.sh frequency`).
