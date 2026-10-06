# Experimental results

These notes summarize layer-0 detokenization analyses of GPT-2 and rinna's Japanese GPT (japanese-gpt-1b), organized by model, to help select figures for papers and the dissertation. The values are from `outputs/` as of 2026-10-06.

- [gpt2](gpt2.md) (OpenWebText)
- [rinna/japanese-gpt-1b](rinna-japanese-gpt-1b.md) (Japanese Wikipedia)

## Model configurations

| Model | Layers | Heads | Hidden dimension | Vocabulary size | Corpus |
|---|---|---|---|---|---|
| gpt2 | 12 | 12 | 768 | 50,257 | OpenWebText (commit `79d93d7`) |
| rinna/japanese-gpt-1b | 24 | 16 | 2048 | 44,877 | Japanese Wikipedia 20231101.ja (commit `b04c8d1`) |

- Both models are analyzed only at layer 0.
- Bigram frequencies are computed over the full corpus; observed attention and the six-term decomposition use the first 5,000 documents of the hash sample.
- Tokenizers are loaded through tokenizer-tools, shared by corpus-tools and feature-extractor. GPT-2 prepends `<|endoftext|>`, as in the published code. rinna adds no special tokens, following its model card.

## Validation

For both models, attention weights obtained by applying softmax to the sum of the six terms matched the model's actual layer-0 attention weights within 0.0002 across all heads. This validates weight extraction and the six-term decomposition for both models.

```bash
uv run python scripts/check_six_terms.py gpt2
uv run python scripts/check_six_terms.py rinna/japanese-gpt-1b
```

## Model comparison

Compare AUROC averaged only over suffixes seen in the corpus ("Seen only" in each note). About 30% of the 1b vocabulary never appears in Japanese Wikipedia, so the paper's averaging method, which assigns zero to unseen suffixes, substantially reduces the 1b scores alone.

| | gpt2 | rinna 1b |
|---|---|---|
| Head with the highest AUROC | 7 (0.877) | 15 (0.696) |
| Number of heads with AUROC below 0.5 | 4 (1, 5, 8, 9) | 7 (0, 1, 4, 5, 7, 10, 12) |
| Head with the most attention from position 500 to the recent past (485–499) | 7 (0.830) | 15 (0.721) |
| Attention to position 0 | Nearly zero in every head | At most 0.013 in every head |

Head indices do not correspond across models, which were trained independently.

## Reproduction

```bash
CPUS=16 DEVICE=cuda bash scripts/compute_data.sh                # GPT-2 data
bash scripts/naacl2025_figures.sh                               # GPT-2 figures at paper size
CPUS=16 DEVICE=cuda bash scripts/compute_data_rinna.sh          # rinna 1b data
bash scripts/rinna_figures.sh                                   # rinna 1b figures for all heads
```
