#!/usr/bin/env bash
# Download the ATLAS backbone into $HF_HOME.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"

for r in "Qwen/Qwen2.5-7B"; do
  echo "=== downloading: $r ==="
  hf download "$r" --repo-type model
done
