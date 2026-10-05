#!/usr/bin/env bash
set -euo pipefail

# Make everything from scratch: the data (scripts/compute_data.sh), then the
# figures and tables of the NAACL 2025 paper and of the PhD thesis. On CPU,
# mostly the frequency counts (~2 h), six_terms (~1.5 h) and auroc (~2 h 20
# min on a server). With CPUS=16 and DEVICE=cuda, about 1 hour: the sample
# (~30 min) and the counts (~30 min); see scripts/compute_data.sh.
#
# Usage (from the repository root):
#   bash scripts/all.sh
#   CPUS=16 DEVICE=cuda bash scripts/all.sh

bash scripts/compute_data.sh
bash scripts/naacl2025_figures.sh
bash scripts/thesis_figures.sh
