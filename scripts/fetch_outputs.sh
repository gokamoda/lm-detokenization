#!/usr/bin/env bash
set -euo pipefail

# Copy outputs/ of this repository on a server into outputs/ here, with
# rsync over ssh (only changed files; an interrupted copy continues).
#
# The server is read from .server at the repository root (git-ignored, as
# this repository is public), or from the environment:
#   SERVER=user@host                          # as given to ssh
#   REMOTE_DIR=/home/kamoda/lm-detokenization # the repository on the server
#
# Usage (from anywhere; files always go to outputs/ of this repository):
#   bash scripts/fetch_outputs.sh                    # everything below, but golden/
#   bash scripts/fetch_outputs.sh figures attention  # only these directories of outputs/
#   DRY_RUN=1 bash scripts/fetch_outputs.sh          # show what would be copied

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/.server" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/.server"
fi
: "${SERVER:?Set SERVER (e.g. user@host) in $ROOT/.server or the environment}"
: "${REMOTE_DIR:?Set REMOTE_DIR (the repository on the server) in $ROOT/.server or the environment}"

# outputs/golden/ is made here (the reference of the tests), not on servers.
ALL_DIRS=(corpus-tools attention six_terms detokenization figures)

if [ "$#" -gt 0 ]; then
  DIRS=("$@")
else
  DIRS=("${ALL_DIRS[@]}")
fi

OPTIONS=(--archive --human-readable --partial --info=progress2)
if [ -n "${DRY_RUN:-}" ]; then
  OPTIONS+=(--dry-run --itemize-changes)
fi

mkdir -p "$ROOT/outputs"
for dir in "${DIRS[@]}"; do
  echo "=== outputs/$dir from $SERVER:$REMOTE_DIR"
  # trailing slashes: the contents of the remote directory into the local one
  mkdir -p "$ROOT/outputs/$dir"
  rsync "${OPTIONS[@]}" "$SERVER:$REMOTE_DIR/outputs/$dir/" "$ROOT/outputs/$dir/"
done
