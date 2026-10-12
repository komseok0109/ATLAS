#!/bin/bash
#SBATCH --job-name=maa-seeds
#SBATCH --output=logs/maa-seeds-%j.out
#SBATCH --error=logs/maa-seeds-%j.err
#SBATCH --nodes=1 --cpus-per-task=8 --mem=96G --time=4:00:00 --requeue
# Rebuild the MAABench seed pool (human text) from the public source corpora. CPU only.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
export PYTHONPATH="src:${PYTHONPATH:-}"
python -m maabench.make_seeds --out data/maabench/seeds --oversample 40 --force
echo "=== Done ==="
