# gpt2

[README](README.md) に 2 モデルの設定と比較がある。

## データ

OpenWebText（commit `79d93d7`）、8,013,769 文書、9,032,003,326 トークン、異なる bigram 137,312,732。observed attention は、サンプルの先頭 5000 文書のうち位置 500 まである 3,319 文書。

```bash
CPUS=16 DEVICE=cuda bash scripts/compute_data.sh
```

頻度は、以前 1 プロセスで数えた頻度と完全に一致した。AUROC と six-terms は、GPU と CPU で丸め誤差の範囲で一致した。

## AUROC

$T^{ee}$ で、出現する bigram の prefix を上位に並べられるか。3 通りの平均を並べる。

- **全体**: 論文の Figure 2 D と同じ。一度も suffix として出現しないトークンは、AUROC を 0 として平均に含める。
- **出現のみ**: 出現する suffix だけで平均する。
- **頻度重み**: 出現する suffix について、その suffix で終わる bigram の総数で重み付けして平均する。

```bash
uv run python scripts/summarize_results.py auroc gpt2
uv run detok-auroc-table               # 全体（論文の表）
uv run detok-auroc-table --seen-only   # 出現のみ
```

Suffixes never seen in the corpus: 103 of 50257.

| ヘッド | 7 | 11 | 6 | 0 | 4 | 3 | 10 | 2 | 8 | 1 | 9 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 全体 | 0.876 | 0.806 | 0.793 | 0.732 | 0.694 | 0.634 | 0.615 | 0.549 | 0.445 | 0.404 | 0.401 | 0.286 |
| 出現のみ | 0.877 | 0.808 | 0.795 | 0.733 | 0.695 | 0.635 | 0.616 | 0.550 | 0.445 | 0.404 | 0.402 | 0.287 |
| 頻度重み | 0.865 | 0.788 | 0.568 | 0.591 | 0.830 | 0.846 | 0.733 | 0.591 | 0.687 | 0.620 | 0.616 | 0.613 |

- 「全体」と「出現のみ」はほぼ同じ（出現しない suffix は 0.2%）。
- 「頻度重み」では、ヘッド 3（0.846）と 4（0.830）が高くなり、ヘッド 7 に近い。頻度の高い suffix では、これらのヘッドもよく当てている。

## 例（論文の Figure 2 A）

suffix ごとに、$T^{ee}$ の上位 5 個の prefix を、つないだ形（prefix + suffix）で示す。`_` は単語の先頭の空白。

```bash
uv run detok-top-prefixes --words iens tarian " Jackson" --heads 4 7 --k 5
```

| suffix | ヘッド | 1 位 | 2 位 | 3 位 | 4 位 | 5 位 |
|---|---|---|---|---|---|---|
| `iens` | 4 | _sapiens | _famiens | _Sapiens | Hugiens | Patiens |
| `iens` | 7 | Aliens | _sapiens | _aliens | _ALiens | afiens |
| `tarian` | 4 | Libertarian | _Libertarian | nontarian | ilatertarian | _Techntarian |
| `tarian` | 7 | _vegetabletarian | _Partarian | _metaltarian | _propertarian | _Danieltarian |
| `_Jackson` | 4 | _JesseJackson | _ArcheJackson | _TriJackson | _CortJackson | _EmersonJackson |
| `_Jackson` | 7 | _MichaelJackson | _PeterJackson | MichaelJackson | _JesseJackson | PeterJackson |

## 位置 500 からの注意の向き先

位置 $i=500$ の注意の重みを、文書について平均した。

```bash
uv run python scripts/summarize_results.py attention gpt2
```

3319 documents.

| ヘッド | 位置 0 | 遠い過去（1〜484） | 直近（485〜499） | 自分（500） |
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

- 自分に集中するヘッド（1, 3）と、直近に集中するヘッド（4, 7）がある。
- 位置 0（`<|endoftext|>`）には、どのヘッドもほとんど向けない。

## 図

`outputs/figures/naacl2025/`（論文の大きさ）と `outputs/figures/thesis/`（博論の大きさ）。

```bash
bash scripts/naacl2025_figures.sh
bash scripts/thesis_figures.sh
```
