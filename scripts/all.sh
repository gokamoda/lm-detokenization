#!/usr/bin/env bash
set -euo pipefail

# Make everything from scratch: the data (scripts/compute_data.sh), then the
# figures and tables of the NAACL 2025 paper and of the PhD thesis. Takes
# about 5 hours on CPU, mostly the frequency counts (~2 h), six_terms
# (~1.5 h) and auroc (~1 h).
#
# Usage (from the repository root):
#   bash scripts/all.sh

bash scripts/compute_data.sh
bash scripts/naacl2025_figures.sh
bash scripts/thesis_figures.sh
