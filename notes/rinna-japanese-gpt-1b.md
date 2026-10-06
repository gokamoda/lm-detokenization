# rinna/japanese-gpt-1b

See [README](README.md) for the configurations and comparison of the two models.It has the same architecture as GPT-2, with 24 layers, 16 heads, and a hidden dimension of 2048; only layer 0 is analyzed. The weights are stored in float16, so layer-0 weights are converted to float32 for analysis.

## Data

Japanese Wikipedia 20231101.ja (commit `b04c8d1`): 1,389,467 documents, 1,412,859,983 tokens, and 52,126,183 distinct bigrams. Observed attention uses 2,475 documents that reach position 500.

```bash
CPUS=16 DEVICE=cuda bash scripts/compute_data_rinna.sh
```

- Tokenizer: English uppercase letters are in the vocabulary, so input is not lowercased. No special tokens are added.
- About 30% of the vocabulary (13,631 tokens) never appears as a suffix in Japanese Wikipedia, including emoji, symbols, and web-specific terms.

## AUROC

The same three averaging methods as for [GPT-2](gpt2.md#auroc) are used.

```bash
uv run python scripts/summarize_results.py auroc rinna/japanese-gpt-1b
```

Suffixes never seen in the corpus: 13631 of 44877.

| Head | 15 | 14 | 6 | 3 | 11 | 2 | 13 | 9 | 8 | 5 | 7 | 12 | 4 | 1 | 10 | 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All | 0.484 | 0.473 | 0.462 | 0.448 | 0.418 | 0.398 | 0.385 | 0.371 | 0.362 | 0.317 | 0.316 | 0.313 | 0.304 | 0.282 | 0.281 | 0.271 |
| Seen only | 0.696 | 0.680 | 0.663 | 0.644 | 0.600 | 0.572 | 0.553 | 0.533 | 0.519 | 0.456 | 0.454 | 0.450 | 0.437 | 0.405 | 0.404 | 0.390 |
| Frequency weighted | 0.709 | 0.747 | 0.639 | 0.615 | 0.536 | 0.516 | 0.478 | 0.539 | 0.499 | 0.476 | 0.520 | 0.527 | 0.372 | 0.388 | 0.423 | 0.334 |

- "All" is below 0.5 for every head because the 30% of unseen suffixes contribute zeros. Use "Seen only" when comparing models.
- The highest "Seen only" AUROC is for head 15 (0.696), below GPT-2 head 7 (0.877).
- Seven heads (0, 1, 4, 5, 7, 10, 12) score below 0.5, compared with four for GPT-2.

## Examples (corresponding to Figure 2 A)

For the suffixes of パン and マンション, the top six prefixes by $T^{ee}$ are listed for the two heads with the highest AUROC (15, 14) and head 6, which was examined in the earlier notebook.

```bash
uv run detok-top-prefixes --model-name rinna/japanese-gpt-1b --words パン マンション --heads 15 14 6 --k 6
```

| suffix | Head | 1st | 2nd | 3rd | 4th | 5th | 6th |
|---|---|---|---|---|---|---|---|
| `パン` | 15 | フライパン | プロパン | のプロパン | _プロパン | アメリカンパン | マレーパン |
| `パン` | 14 | パンパン | _パンパン | パンのパン | パンをパン | パンツパン | マグパン |
| `パン` | 6 | コロナパン | カレーパン | ビールパン | ランドパン | 糖質制限パン | _アフリカパン |
| `ション` | 15 | グローション | ローション | フルエンション | モルション | _ローション | ジェンション |
| `ション` | 14 | ションション | ションはション | ーションション | ッシュション | ッションション | ージョンション |
| `ション` | 6 | ペンション | _ペンション | キシション | フェンション | スキション | フィンション |

`_` represents the word-initial marker `▁`. The table shows concatenated prefixes and suffixes.

- Suffixes use the form that occurs inside a word (`tokens.suffix_id`). The 1b tokenizer encodes パン as a single word-initial token, `▁パン`, so using it directly selects a different token (head 6 then ranks シャツ, スカート, コロナ, バッグ, ジャケット, サラダ highest). マンション splits into `▁マン` + `ション`, so its suffix is `ション`.
- Head 15: Highly ranked prefixes form existing words such as フライパン, プロパン, and ローション, as in Figure 2 A for GPT-2.
- Head 6: For パン, highly ranked prefixes form types of bread, such as カレーパン and 糖質制限パン, consistent with the earlier notebook. For ション, ペンション also ranks first.
- Head 14: Many highly ranked prefixes are the suffix token itself (パン→パン, ション→ション) or tokens containing it. This head appears to attend to repeated tokens rather than join word fragments, consistent with its attention to the current position at position 500 (0.136), the highest across heads. This table does not explain its high AUROC (the highest frequency-weighted value, 0.747).
- The prefix `マン` of マンション is not among the top six for any of these heads.

## Attention destinations from position 500

```bash
uv run python scripts/summarize_results.py attention rinna/japanese-gpt-1b
```

2475 documents.

| Head | Position 0 | Far past (1–484) | Recent past (485–499) | Self (500) |
|---|---|---|---|---|
| 0 | 0.008 | 0.918 | 0.068 | 0.005 |
| 1 | 0.013 | 0.962 | 0.023 | 0.002 |
| 2 | 0.004 | 0.853 | 0.132 | 0.011 |
| 3 | 0.005 | 0.740 | 0.191 | 0.063 |
| 4 | 0.002 | 0.724 | 0.256 | 0.018 |
| 5 | 0.005 | 0.886 | 0.097 | 0.011 |
| 6 | 0.005 | 0.654 | 0.318 | 0.024 |
| 7 | 0.004 | 0.724 | 0.256 | 0.016 |
| 8 | 0.004 | 0.637 | 0.341 | 0.018 |
| 9 | 0.006 | 0.729 | 0.244 | 0.021 |
| 10 | 0.003 | 0.355 | 0.593 | 0.050 |
| 11 | 0.003 | 0.913 | 0.078 | 0.006 |
| 12 | 0.006 | 0.909 | 0.078 | 0.007 |
| 13 | 0.011 | 0.654 | 0.322 | 0.013 |
| 14 | 0.001 | 0.839 | 0.025 | 0.136 |
| 15 | 0.005 | 0.238 | 0.721 | 0.035 |

- Attention to position 0 and the current position is generally small. Many heads distribute attention broadly over the far past.
- Heads 15 (0.721) and 10 (0.593) pay the most attention to the recent past.

## Six-term contributions

`outputs/figures/rinna-japanese-gpt-1b/six_terms_head-all.pdf`. Not yet inspected.

## Figures

`outputs/figures/rinna-japanese-gpt-1b/` (all heads, dissertation size).

```bash
bash scripts/rinna_figures.sh
```

## Candidate heads to highlight (not yet decided)

Heads 15 (highest AUROC and attention to the recent past, with top prefixes forming existing words) and 6 (the パン example) are candidates. Head 14 has high AUROC but its top prefixes repeat the same token, making it less suitable as an example.
