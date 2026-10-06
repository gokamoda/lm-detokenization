# gpt2

See [README](README.md) for the configurations and comparison of the two models.

## Data

OpenWebText (commit `79d93d7`): 8,013,769 documents, 9,032,003,326 tokens, and 137,312,732 distinct bigrams. Observed attention uses the 3,319 documents among the first 5,000 sampled documents that reach position 500.

```bash
CPUS=16 DEVICE=cuda bash scripts/compute_data.sh
```

Frequencies matched the earlier single-process counts exactly. AUROC and the six-term decomposition agreed between GPU and CPU within rounding error.

## AUROC

AUROC measures how well $T^{ee}$ ranks prefixes of observed bigrams. Three averaging methods are reported.

- **All**: As in Figure 2 D of the paper, tokens never seen as suffixes are included in the average with an AUROC of zero.
- **Seen only**: Average over suffixes seen in the corpus.
- **Frequency weighted**: Average over seen suffixes, weighted by the total count of bigrams ending in each suffix.

```bash
uv run python scripts/summarize_results.py auroc gpt2
uv run detok-auroc-table               # All suffixes (the paper's table)
uv run detok-auroc-table --seen-only   # Seen suffixes only
```

Suffixes never seen in the corpus: 103 of 50257.

| Head | 7 | 11 | 6 | 0 | 4 | 3 | 10 | 2 | 8 | 1 | 9 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All | 0.876 | 0.806 | 0.793 | 0.732 | 0.694 | 0.634 | 0.615 | 0.549 | 0.445 | 0.404 | 0.401 | 0.286 |
| Seen only | 0.877 | 0.808 | 0.795 | 0.733 | 0.695 | 0.635 | 0.616 | 0.550 | 0.445 | 0.404 | 0.402 | 0.287 |
| Frequency weighted | 0.865 | 0.788 | 0.568 | 0.591 | 0.830 | 0.846 | 0.733 | 0.591 | 0.687 | 0.620 | 0.616 | 0.613 |

- "All" and "Seen only" are nearly identical: only 0.2% of suffixes are unseen.
- With frequency weighting, heads 3 (0.846) and 4 (0.830) score higher, approaching head 7. These heads also perform well on frequent suffixes.

## Examples (Figure 2 A of the paper)

For each suffix, the top five prefixes by $T^{ee}$ are shown concatenated with the suffix (prefix + suffix). `_` denotes a leading space.

```bash
uv run detok-top-prefixes --words iens tarian " Jackson" --heads 4 7 --k 5
```

| suffix | Head | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|---|
| `iens` | 4 | _sapiens | _famiens | _Sapiens | Hugiens | Patiens |
| `iens` | 7 | Aliens | _sapiens | _aliens | _ALiens | afiens |
| `tarian` | 4 | Libertarian | _Libertarian | nontarian | ilatertarian | _Techntarian |
| `tarian` | 7 | _vegetabletarian | _Partarian | _metaltarian | _propertarian | _Danieltarian |
| `_Jackson` | 4 | _JesseJackson | _ArcheJackson | _TriJackson | _CortJackson | _EmersonJackson |
| `_Jackson` | 7 | _MichaelJackson | _PeterJackson | MichaelJackson | _JesseJackson | PeterJackson |

## Attention destinations from position 500

Attention weights at position $i=500$ are averaged over documents.

```bash
uv run python scripts/summarize_results.py attention gpt2
```

3319 documents.

| Head | Position 0 | Far past (1–484) | Recent past (485–499) | Self (500) |
|---|---|---|---|---|
| 0 | 0.001 | 0.678 | 0.304 | 0.017 |
| 1 | 0.000 | 0.152 | 0.057 | 0.792 |
| 2 | 0.002 | 0.720 | 0.244 | 0.034 |
| 3 | 0.000 | 0.000 | 0.173 | 0.827 |
| 4 | 0.000 | 0.009 | 0.455 | 0.536 |
| 5 | 0.001 | 0.495 | 0.025 | 0.480 |
| 6 | 0.001 | 0.656 | 0.314 | 0.029 |
| 7 | 0.000 | 0.007 | 0.830 | 0.164 |
| 8 | 0.000 | 0.604 | 0.344 | 0.052 |
| 9 | 0.001 | 0.667 | 0.312 | 0.019 |
| 10 | 0.001 | 0.550 | 0.302 | 0.146 |
| 11 | 0.003 | 0.974 | 0.022 | 0.001 |

- Some heads focus on the current position (1, 3), while others focus on the recent past (4, 7).
- All heads pay very little attention to position 0 (`<|endoftext|>`).

## Figures

`outputs/figures/naacl2025/` (paper size) and `outputs/figures/thesis/` (dissertation size).

```bash
bash scripts/naacl2025_figures.sh
bash scripts/thesis_figures.sh
```
