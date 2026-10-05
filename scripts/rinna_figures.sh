#!/usr/bin/env bash
set -euo pipefail

# Write the figures of rinna/japanese-gpt2-small (Japanese Wikipedia) for all
# heads, at the printed size of the PhD thesis (--target thesis; --width is a
# fraction of the textwidth, --aspect is height / width), with the mean AUROC
# per head as CSV. The heads to show alone are chosen after seeing these.
#
# Needs the data made by scripts/compute_data_rinna.sh.
#
# Usage (from the repository root):
#   bash scripts/rinna_figures.sh              # all figures
#   bash scripts/rinna_figures.sh six_terms    # only the named ones

DEST="${DEST:-outputs/figures/rinna-japanese-gpt2-small}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
T="--target thesis"
M="--model-name rinna/japanese-gpt2-small"
C="--corpus wikipedia-ja"

auroc_table() {
  uv run detok-auroc-table $M --output "$OUT/auroc_by_head.csv"
}

tee_heatmap() {
  uv run tee-plot $T $M --heads -1 --ncols 4 --width 1.0 --aspect 0.85 \
    --output "$OUT/tee_head-all.png"
}

te_scatter() {
  uv run te-plot $T $M $C --heads -1 --ncols 4 --width 1.0 --aspect 0.48 \
    --output "$OUT/te_head-all.png"
}

tp() {
  uv run tp-plot $T $M --heads -1 --ncols 2 --width 1.0 --aspect 0.743 \
    --output "$OUT/tp_head-all.pdf"
}

tpp() {
  uv run tpp-plot $T $M --heads -1 --pos-i 50 500 1000 --legend \
    --width 1.0 --aspect 1.215 \
    --output "$OUT/tpp_head-all.pdf"
}

tp_tpp() {
  uv run tp-tpp-plot $T $M --heads -1 --pos-i 50 500 1000 --views after --legend \
    --width 1.0 --aspect 1.198 \
    --output "$OUT/tptpp_head-all.pdf"
}

position_bias() {
  uv run position-bias-plot $T $M --heads -1 --width 1.0 --aspect 1.73 \
    --output "$OUT/position_bias_head-all.pdf"
}

six_terms() {
  uv run six-terms-plot $T $M --heads 0 1 2 3 4 5 6 7 8 9 10 11 --ncols 3 \
    --width 1.0 --aspect 0.639 \
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
