#!/usr/bin/env bash
set -euo pipefail

# Write the figures of the detokenization chapter of the PhD thesis (phd/thesis)
# at their printed size. --width is a fraction of the thesis textwidth
# (155mm, the width in main.tex); --aspect is height / width.
#
# Figure 2 of the paper (detokenization_token_affinity) also has tables, so
# only its parts are made here: B, C and the tables of A and D as CSV. The
# ranks in A are T^ee with LN, which differ from the head 7 rows of the paper.
#
# Needs the data made by scripts/compute_data.sh.
#
# Usage (from the repository root):
#   bash scripts/thesis_figures.sh                       # all figures
#   bash scripts/thesis_figures.sh position_bias         # only the named ones
#   DEST=outputs/figures/check bash scripts/thesis_figures.sh position_bias
#
# The figures are not written into ../thesis; copy them to
# ../thesis/figures/detokenization/ after checking.

DEST="${DEST:-outputs/figures/thesis}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
T="--target thesis"

# fig:detokenization-term-contribution: 0.82 textwidth.
six_term_contribution() {
  uv run six-terms-plot $T --heads 1 7 --width 0.82 --aspect 0.396 \
    --output "$OUT/detokenization_six_term_contribution.pdf"
}

# fig:detokenization-token-affinity: textwidth; parts B and C.
token_affinity_b_tee() {
  uv run tee-plot $T --heads 1 7 --ncols 2 --width 0.28 --aspect 0.55 \
    --output "$OUT/detokenization_token_affinity_b_tee.pdf"
}

token_affinity_c_roc() {
  uv run roc-plot $T --word iens --heads 1 7 --width 0.28 --aspect 0.55 \
    --output "$OUT/detokenization_token_affinity_c_roc.pdf"
}

token_affinity_tables() {
  uv run detok-top-prefixes --words iens tarian " Jackson" --heads 4 7 --k 5 \
    --output "$OUT/detokenization_token_affinity_a_top_prefixes.csv"
  uv run detok-auroc-table --output "$OUT/detokenization_token_affinity_d_auroc.csv"
  # over the suffixes seen in OpenWebText only
  uv run detok-auroc-table --seen-only --output "$OUT/detokenization_token_affinity_d_auroc_seen.csv"
}

# fig:detokenization-position-bias: textwidth, columns A-E in one figure.
position_bias() {
  uv run position-bias-plot $T --heads 1 7 --width 1.0 --aspect 0.288 \
    --output "$OUT/detokenization_position_bias.pdf"
}

ALL_FIGURES=(
  six_term_contribution
  token_affinity_b_tee
  token_affinity_c_roc
  token_affinity_tables
  position_bias
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
