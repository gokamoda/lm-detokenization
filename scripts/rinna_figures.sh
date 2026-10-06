#!/usr/bin/env bash
set -euo pipefail

# Write the figures of a Japanese GPT of rinna (Japanese Wikipedia) for all
# heads, at the printed size of the PhD thesis (--target thesis; --width is a
# fraction of the textwidth, --aspect is height / width), with the mean AUROC
# per head as CSV. The heads to show alone are chosen after seeing these.
#
# The model is $MODEL: rinna/japanese-gpt-1b (default, 16 heads) or
# rinna/japanese-gpt2-small (12 heads); figures with a row of panels per head (or
# per few heads) are made taller for more heads.
#
# Needs the data made by scripts/compute_data_rinna.sh (with the same MODEL).
#
# Usage (from the repository root):
#   bash scripts/rinna_figures.sh                                  # all figures
#   bash scripts/rinna_figures.sh six_terms                        # only the named ones
#   MODEL=rinna/japanese-gpt2-small bash scripts/rinna_figures.sh  # -> outputs/figures/rinna-japanese-gpt2-small/

MODEL="${MODEL:-rinna/japanese-gpt-1b}"
DEST="${DEST:-outputs/figures/${MODEL//\//-}}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
T="--target thesis"
M="--model-name $MODEL"
C="--corpus wikipedia-ja"
NUM_HEADS="$(uv run python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('$MODEL').n_head)")"
HEADS="$(seq -s ' ' 0 $((NUM_HEADS - 1)))"

# The aspect of a figure made for 12 heads in rows of $2 panels, scaled to
# the rows NUM_HEADS needs.
aspect() {
  awk -v a="$1" -v c="$2" -v n="$NUM_HEADS" \
    'BEGIN { printf "%.3f", a * int((n + c - 1) / c) / int((12 + c - 1) / c) }'
}

auroc_table() {
  uv run detok-auroc-table $M --output "$OUT/auroc_by_head.csv"
  # over the suffixes seen in Japanese Wikipedia only
  uv run detok-auroc-table $M --seen-only --output "$OUT/auroc_by_head_seen.csv"
}

tee_heatmap() {
  uv run tee-plot $T $M --heads -1 --ncols 4 --width 1.0 --aspect "$(aspect 0.85 4)" \
    --output "$OUT/tee_head-all.png"
}

te_scatter() {
  uv run te-plot $T $M $C --heads -1 --ncols 4 --width 1.0 --aspect "$(aspect 0.48 4)" \
    --output "$OUT/te_head-all.png"
}

tp() {
  uv run tp-plot $T $M --heads -1 --ncols 2 --width 1.0 --aspect "$(aspect 0.743 2)" \
    --output "$OUT/tp_head-all.pdf"
}

tpp() {
  uv run tpp-plot $T $M --heads -1 --pos-i 50 500 1000 --legend \
    --width 1.0 --aspect "$(aspect 1.215 1)" \
    --output "$OUT/tpp_head-all.pdf"
}

tp_tpp() {
  uv run tp-tpp-plot $T $M --heads -1 --pos-i 50 500 1000 --views after --legend \
    --width 1.0 --aspect "$(aspect 1.198 1)" \
    --output "$OUT/tptpp_head-all.pdf"
}

position_bias() {
  uv run position-bias-plot $T $M --heads -1 --width 1.0 --aspect "$(aspect 1.73 1)" \
    --output "$OUT/position_bias_head-all.pdf"
}

six_terms() {
  uv run six-terms-plot $T $M --heads $HEADS --ncols 3 \
    --width 1.0 --aspect "$(aspect 0.639 3)" \
    --output "$OUT/six_terms_head-all.pdf"
}

variance() {
  uv run variance-plot $T $M $C --width 0.48 --aspect 0.435 \
    --output "$OUT/variance.pdf"
}

undertrained() {
  uv run undertrained-plot $T $M --head 0 --width 0.48 --aspect 0.295 \
    --output "$OUT/undertrained_head-0.pdf"
}

ALL_FIGURES=(
  auroc_table
  tee_heatmap
  te_scatter
  tp
  tpp
  tp_tpp
  position_bias
  six_terms
  variance
  undertrained
)

if [ "$#" -gt 0 ]; then
  FIGURES=("$@")
else
  FIGURES=("${ALL_FIGURES[@]}")
fi
for figure in "${FIGURES[@]}"; do
  if ! declare -F "$figure" >/dev/null; then
    echo "Unknown figure: $figure (choose from: ${ALL_FIGURES[*]})" >&2
    exit 1
  fi
done
for figure in "${FIGURES[@]}"; do
  "$figure"
done

mkdir -p "$DEST"
mv "$OUT"/* "$DEST"/
echo "Wrote ${#FIGURES[@]} figure(s) into $DEST"
