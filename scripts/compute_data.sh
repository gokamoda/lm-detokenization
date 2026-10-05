#!/usr/bin/env bash
set -euo pipefail

# Compute the data the figure scripts read (the sample under data/, the rest
# under outputs/). The sample and the counts are made with corpus-tools, from
# OpenWebText at a fixed revision (see src/lm_detokenization/data/openwebtext.py).
#
#   sample        data/openwebtext/hash_n10000.jsonl     10,000 documents (~30 min, streams ~24GB)
#   frequency     outputs/freqs/openwebtext/gpt2/        1-grams.npy, 2-grams.npz on all of OpenWebText (~2 h)
#   attention     outputs/attention/gpt2/layer00_i500.pt observed attention from position 500 (~20 s)
#   six_terms     outputs/six_terms/gpt2/contributions.npy  5,000 documents (~1.5 h on CPU)
#   auroc         outputs/detokenization/gpt2/auroc.parquet every suffix token (~1 h; needs frequency)
#
# Usage (from the repository root):
#   bash scripts/compute_data.sh                  # all steps, in the order above
#   bash scripts/compute_data.sh attention auroc  # only the named steps

sample() {
  uv run sample-openwebtext --num-samples 10000
}

frequency() {
  uv run count-frequency --source openwebtext
}

attention() {
  uv run attention-rows --num-documents 100 --positions 500
}

six_terms() {
  uv run six-terms --num-documents 5000
}

auroc() {
  uv run detok-auroc
}

ALL_STEPS=(sample frequency attention six_terms auroc)

if [ "$#" -gt 0 ]; then
  STEPS=("$@")
else
  STEPS=("${ALL_STEPS[@]}")
fi
for step in "${STEPS[@]}"; do
  if ! declare -F "$step" >/dev/null; then
    echo "Unknown step: $step (choose from: ${ALL_STEPS[*]})" >&2
    exit 1
  fi
done
for step in "${STEPS[@]}"; do
  echo "=== $step"
  "$step"
done
