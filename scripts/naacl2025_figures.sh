#!/usr/bin/env bash
set -euo pipefail

# Write the figures of the NAACL 2025 paper (conferences/naacl2025) at their
# printed size. --width is a fraction of the textwidth (16.0cm); the main text
# is two-column (one column = 0.48125), the appendix is \onecolumn (1.0).
# --aspect is height / width, taken from the figures in the paper.
#
# Figures 2 and 3 are made of parts (Figure 2 also has tables, see
# naacl2025_tables below); Figure 3 is also written as one figure
# (detokenization-position.pdf).
#
# Needs the data made by scripts/compute_data.sh.
#
# Usage (from the repository root):
#   bash scripts/naacl2025_figures.sh                   # all figures
#   bash scripts/naacl2025_figures.sh te_1_7 variance   # only the named ones
#   DEST=outputs/figures/check bash scripts/naacl2025_figures.sh te_1_7
#
# The figures are not written into conferences/naacl2025; copy them there.

DEST="${DEST:-outputs/figures/naacl2025}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
T="--target naacl2025"
COLUMN=0.48125

# Figure 2 (figure*, A and D are tables): B and C.
fig2_b_tee() {
  uv run tee-plot $T --heads 1 7 --ncols 2 --width 0.28 --aspect 0.55 \
    --output "$OUT/fig2_b_tee_head-1-7.pdf"
}

fig2_c_roc() {
  uv run roc-plot $T --word iens --heads 1 7 --width 0.28 --aspect 0.55 \
    --output "$OUT/fig2_c_roc_iens_head-1-7.pdf"
}

# Figure 3 (figure*): one figure, and its columns A-E as parts.
fig3() {
  uv run position-bias-plot $T --heads 1 7 --width 1.0 --aspect 0.288 \
    --output "$OUT/detokenization-position.pdf"
}

fig3_a_tp() {
  uv run tp-plot $T --heads 1 7 --width 0.2 --aspect 1.44 \
    --output "$OUT/fig3_a_tp_head-1-7.pdf"
}

fig3_b_tpp() {
  uv run tpp-plot $T --heads 1 7 --pos-i 500 --width 0.2 --aspect 1.44 \
    --output "$OUT/fig3_b_tpp_head-1-7.pdf"
}

fig3_cd_tp_tpp() {
  uv run tp-tpp-plot $T --heads 1 7 --pos-i 500 --width 0.4 --aspect 0.72 \
    --output "$OUT/fig3_cd_tptpp_head-1-7.pdf"
}

fig3_e_observed() {
  uv run vs-tptpp-plot $T --heads 1 7 --pos-i 500 --zoom-only --width 0.2 --aspect 1.44 \
    --output "$OUT/fig3_e_observed_head-1-7.pdf"
}

# Main text, one column.
six_terms_1_7() {
  uv run six-terms-plot $T --heads 1 7 --width $COLUMN --aspect 0.396 \
    --output "$OUT/6term-importance-1-7.pdf"
}

te_1_7() {
  uv run te-plot $T --heads 1 7 --ncols 2 --width $COLUMN --aspect 0.351 \
    --output "$OUT/te_1-7.pdf"
}

variance() {
  uv run variance-plot $T --width $COLUMN --aspect 0.435 \
    --output "$OUT/variance.pdf"
}

undertrained_head_0() {
  uv run undertrained-plot $T --head 0 --width $COLUMN --aspect 0.295 \
    --output "$OUT/undertrained_head-0.pdf"
}

# Appendix, \onecolumn.
l0tp_head_all() {
  uv run tp-plot $T --heads -1 --ncols 2 --width 1.0 --aspect 0.743 \
    --output "$OUT/l0tp_head-all.pdf"
}

l0tpp_head_all() {
  uv run tpp-plot $T --heads -1 --pos-i 50 500 1000 --legend \
    --width 1.0 --aspect 1.215 \
    --output "$OUT/l0tpp_head-all.pdf"
}

l0tptpp_head_all() {
  uv run tp-tpp-plot $T --heads -1 --pos-i 50 500 1000 --views after --legend \
    --width 1.0 --aspect 1.198 \
    --output "$OUT/l0tptpp_head-all.pdf"
}

l0te_head_all() {
  uv run te-plot $T --heads -1 --ncols 4 --width 1.0 --aspect 0.48 \
    --output "$OUT/l0te_head-all.png"
}

l0tee_head_all() {
  uv run tee-plot $T --heads -1 --ncols 4 --width 1.0 --aspect 0.85 \
    --output "$OUT/l0tee_head-all.png"
}

six_terms_all() {
  uv run six-terms-plot $T --heads 0 1 2 3 4 5 6 7 8 9 10 11 --ncols 3 \
    --width 1.0 --aspect 0.639 \
    --output "$OUT/6term-importance-all.pdf"
}

# Tables of Figure 2 (A, D), as CSV.
naacl2025_tables() {
  uv run detok-top-prefixes --words iens tarian " Jackson" --heads 4 7 --k 5 \
    --output "$OUT/fig2_a_top_prefixes.csv"
  uv run detok-auroc-table --output "$OUT/fig2_d_auroc.csv"
  # not in the paper: over the suffixes seen in OpenWebText only
  uv run detok-auroc-table --seen-only --output "$OUT/fig2_d_auroc_seen.csv"
}

ALL_FIGURES=(
  fig2_b_tee
  fig2_c_roc
  fig3
  fig3_a_tp
  fig3_b_tpp
  fig3_cd_tp_tpp
  fig3_e_observed
  six_terms_1_7
  te_1_7
  variance
  undertrained_head_0
  l0tp_head_all
  l0tpp_head_all
  l0tptpp_head_all
  l0te_head_all
  l0tee_head_all
  six_terms_all
  naacl2025_tables
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
