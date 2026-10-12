#!/bin/bash
#SBATCH --job-name=maa-inter-train
#SBATCH --output=logs/maa-inter-train-%j.out
#SBATCH --error=logs/maa-inter-train-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400GB
#SBATCH --time=3-00:00:00
# Train ATLAS on maabench (machine-written text only, K-way).
# usage: sbatch scripts/atlas/maabench/inter/train.sh <run_name>

set -euo pipefail

export HUMAN_CLASS=0
_ATLAS_IMPL="${SLURM_SUBMIT_DIR:-.}/scripts/atlas/maabench/_impl/train.sh"
_ATLAS_SNAP="$(mktemp "${TMPDIR:-/tmp}/atlas_impl.XXXXXX")"
cat "$_ATLAS_IMPL" > "$_ATLAS_SNAP"
bash "$_ATLAS_SNAP" "${1:-base}" "${@:2}"
_ATLAS_RC=$?
rm -f "$_ATLAS_SNAP"
exit $_ATLAS_RC
