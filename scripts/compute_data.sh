#!/usr/bin/env bash
set -euo pipefail

# Compute the data the figure scripts read, under outputs/. The OpenWebText
# sample and the counts are made with the corpus-tools command, apart (the
# counts are of all of OpenWebText), streaming OpenWebText at a fixed commit.
# They go under outputs/corpus-tools/Skylion007--openwebtext/plain_text/train/:
#
#   sample        samples/hash_n10000.jsonl              10,000 documents (~30 min, streams ~24GB)
#   frequency     all/gpt2/counts/nobos/                 1-grams.npy, 2-grams.npz on all of OpenWebText
#                                                        (~2 h; ~30 min with CPUS=16)
#   attention     outputs/attention/gpt2/layer00_i500.pt observed attention from position 500 (~20 s)
#   six_terms     outputs/six_terms/gpt2/contributions.npy  5,000 documents (~1.5 h on CPU)
#   auroc         outputs/detokenization/gpt2/auroc.parquet every suffix token (~1 h; needs frequency)
#
# Usage (from the repository root):
#   bash scripts/compute_data.sh                  # all steps, in the order above
#   bash scripts/compute_data.sh attention auroc  # only the named steps
#   CPUS=16 bash scripts/compute_data.sh frequency  # count with 16 threads in all
#                                                   # (corpus-tools --cpus)
#   DEVICE=cuda bash scripts/compute_data.sh six_terms  # compute on a GPU

# OpenWebText is read at this commit of its Hub repository, so that the sample
# and the counts can be made again. The sample made from it is byte-identical
# to the one made before the switch to corpus-tools.
OPENWEBTEXT_REVISION=79d93d786212f7344586290adb811d4ae6a1762c
# The output directory must match CORPUS_TOOLS_OUTPUT in
# src/lm_detokenization/data/openwebtext.py, which reads them.
CORPUS_TOOLS_ARGS=(--corpus openwebtext --revision "$OPENWEBTEXT_REVISION" --output-dir outputs/corpus-tools)

sample() {
  uv run corpus-tools sample "${CORPUS_TOOLS_ARGS[@]}" --num-samples 10000
}

frequency() {
  uv run corpus-tools count "${CORPUS_TOOLS_ARGS[@]}" \
    --tokenizer openai-community/gpt2 --tokenizer-name gpt2 --n 1 2 \
    ${CPUS:+--cpus "$CPUS"}
}

attention() {
  uv run attention-rows --num-documents 100 --positions 500
}

six_terms() {
  uv run six-terms --num-documents 5000 ${DEVICE:+--device "$DEVICE"}
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
