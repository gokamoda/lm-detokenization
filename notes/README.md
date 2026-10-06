# 実験結果のメモ

GPT-2 と rinna の日本語 GPT（japanese-gpt-1b）で、層 0 の detokenization を分析した結果を、モデルごとにまとめる。論文や博論で使う図を選ぶときの材料にする。数値は 2026-10-06 時点の `outputs/` から取った。

- [gpt2](gpt2.md)（OpenWebText）
- [rinna/japanese-gpt-1b](rinna-japanese-gpt-1b.md)（日本語 Wikipedia）

## 2 モデルの設定

| モデル | 層 | ヘッド | 次元 | 語彙 | コーパス |
|---|---|---|---|---|---|
| gpt2 | 12 | 12 | 768 | 50,257 | OpenWebText（commit `79d93d7`） |
| rinna/japanese-gpt-1b | 24 | 16 | 2048 | 44,877 | 日本語 Wikipedia 20231101.ja（commit `b04c8d1`） |

- 分析はどのモデルも層 0 だけ。
- 頻度（bigram）はコーパス全体、observed attention と six-terms はハッシュサンプルの先頭 5000 文書から計算する。
- tokenizer は tokenizer-tools で読み込む（corpus-tools も feature-extractor も同じ）。gpt2 は先頭に `<|endoftext|>` を付ける（公開版のコードどおり）。rinna は特殊トークンを付けない（モデルカードどおり）。

## 計算の確かめ

6 項の合計から作った注意の重み（softmax）は、モデルが実際に計算した層 0 の注意の重みと、どちらのモデルも全ヘッドで 0.0002 以内で一致した（rinna/japanese-gpt2-small でも同じ）。重みの取り出しと 6 項分解は、どのモデルでも正しい。

```bash
uv run python scripts/check_six_terms.py gpt2
uv run python scripts/check_six_terms.py rinna/japanese-gpt-1b
```

## 2 モデルの比較

AUROC は、コーパスに出現する suffix だけで平均した値（各ノートの「出現のみ」）で比べる。1b は語彙の 30% が日本語 Wikipedia に出現しないので、論文と同じ平均（出現しない suffix を 0 とする）では、1b だけ大きく下がるため。

| | gpt2 | rinna 1b |
|---|---|---|
| AUROC が最大のヘッド | 7（0.877） | 15（0.696） |
| AUROC が 0.5 未満のヘッド | 4（1, 5, 8, 9） | 7（0, 1, 4, 5, 7, 10, 12） |
| 位置 500 から直近（485〜499）への重みが最大のヘッド | 7（0.830） | 15（0.721） |
| 位置 0 への重み | どのヘッドもほぼ 0 | どのヘッドも 0.013 以下 |

ヘッドの番号はモデルの間で対応しない（別々に学習されたモデル）。

## 作り直し方

```bash
CPUS=16 DEVICE=cuda bash scripts/compute_data.sh                # gpt2 のデータ
bash scripts/naacl2025_figures.sh                               # gpt2 の図（論文の大きさ）
CPUS=16 DEVICE=cuda bash scripts/compute_data_rinna.sh          # rinna 1b のデータ
bash scripts/rinna_figures.sh                                   # rinna 1b の図（全ヘッド）
```
