#!/usr/bin/env bash
set -euo pipefail

# Compute the data the figure scripts read (all under outputs/).
#
#   sample        outputs/data/openwebtext_sample.jsonl  10,000 documents (~30 min, streams ~24GB)
#   frequency     outputs/freqs/openwebtext/             token and bigram counts on all of OpenWebText
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
