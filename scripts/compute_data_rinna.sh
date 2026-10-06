#!/usr/bin/env bash
set -euo pipefail

# Compute the data of a Japanese GPT of rinna on Japanese Wikipedia
# (20231101.ja), as scripts/compute_data.sh does for GPT-2 on OpenWebText.
# The model is $MODEL: rinna/japanese-gpt-1b (default; the architecture of
# GPT-2 with 24 layers, 16 heads, 2048 dims; the analyses use its first
# layer) or rinna/japanese-gpt2-small (the architecture and size of GPT-2).
# The sample and the counts are made with the corpus-tools command, apart
# (the counts are of all of Japanese Wikipedia), streaming it at a fixed
# commit, under outputs/corpus-tools/wikimedia--wikipedia/20231101.ja/train/:
#
#   sample        samples/hash_n10000.jsonl              10,000 documents (one for all models)
#   frequency     all/<model>/counts/nobos/              1-grams.npy, 2-grams.npz
#   attention     outputs/attention/<model>/layer00_i500.pt
#   six_terms     outputs/six_terms/<model>/contributions.npy
#   auroc         outputs/detokenization/<model>/auroc.parquet (needs frequency)
#
# where <model> is the model name with "--" for "/", e.g.
# rinna--japanese-gpt-1b.
#
# The tokenizer is that of tokenizer-tools (in corpus-tools and
# feature-extractor alike): no special token is added, as in the model cards,
# and for japanese-gpt2-small the text is lowercased (its vocabulary has no
# upper-case letters; that of japanese-gpt-1b has).
#
# Usage (from the repository root):
#   bash scripts/compute_data_rinna.sh                  # all steps, in the order above
#   bash scripts/compute_data_rinna.sh attention auroc  # only the named steps
#   CPUS=16 bash scripts/compute_data_rinna.sh frequency      # corpus-tools --cpus
#   DEVICE=cuda bash scripts/compute_data_rinna.sh six_terms auroc  # on a GPU
#   MODEL=rinna/japanese-gpt2-small CPUS=16 DEVICE=cuda bash scripts/compute_data_rinna.sh

MODEL="${MODEL:-rinna/japanese-gpt-1b}"
# Japanese Wikipedia is read at this commit of its Hub repository, so that
# the sample and the counts can be made again.
WIKIPEDIA_REVISION=b04c8d1ceb2f5cd4588862100d08de323dccfbaa
# The output directory must match CORPUS_TOOLS_OUTPUT in
# src/lm_detokenization/data/corpus.py, which reads them.
CORPUS_TOOLS_ARGS=(--corpus wikipedia --name 20231101.ja --revision "$WIKIPEDIA_REVISION" --output-dir outputs/corpus-tools)
# The same corpus for the commands of this repository.
ARGS=(--model-name "$MODEL" --corpus wikipedia-ja)

sample() {
  uv run corpus-tools sample "${CORPUS_TOOLS_ARGS[@]}" --num-samples 10000
}

frequency() {
  uv run corpus-tools count "${CORPUS_TOOLS_ARGS[@]}" \
    --tokenizer "$MODEL" --tokenizer-name "$MODEL" --n 1 2 \
    ${CPUS:+--cpus "$CPUS"}
}

attention() {
  uv run attention-rows "${ARGS[@]}" --num-documents 5000 --positions 500
}

six_terms() {
  uv run six-terms "${ARGS[@]}" --num-documents 5000 ${DEVICE:+--device "$DEVICE"}
}

auroc() {
  uv run detok-auroc "${ARGS[@]}" ${DEVICE:+--device "$DEVICE"}
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
