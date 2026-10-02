#!/usr/bin/env bash
# Make the fixed OpenWebText sample used by the empirical experiments
# (empirical/six_terms_importance.py, empirical/vs_tptpp.py).
#
# Reads the whole dataset once by streaming (about 24GB of download, ~30 min),
# and writes the 10,000 documents with the smallest sha256(text) to
# outputs/data/openwebtext_sample.jsonl. See src/data/hash_sample.py.
#
# Run from the repository root after `make install`:
#     bash scripts/make_openwebtext_sample.sh
set -euo pipefail

uv run python src/sample_openwebtext.py --num-samples 10000
