#!/bin/bash
#SBATCH --job-name=ot-human-collect
#SBATCH --output=logs/ot-human-collect-%j.out
#SBATCH --error=logs/ot-human-collect-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
# Score every axis with a trained checkpoint on openturing (human as a (K+1)-th class).
# usage: sbatch scripts/atlas/openturing/human/collect_scores.sh <run_name>

set -euo pipefail

export HUMAN_CLASS=1
_ATLAS_IMPL="${SLURM_SUBMIT_DIR:-.}/scripts/atlas/openturing/_impl/collect_scores.sh"
_ATLAS_SNAP="$(mktemp "${TMPDIR:-/tmp}/atlas_impl.XXXXXX")"
cat "$_ATLAS_IMPL" > "$_ATLAS_SNAP"
bash "$_ATLAS_SNAP" "${1:-base-hc}" "${@:2}"
_ATLAS_RC=$?
rm -f "$_ATLAS_SNAP"
exit $_ATLAS_RC
