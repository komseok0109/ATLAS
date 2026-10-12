#!/bin/bash
#SBATCH --job-name=ot-human-eval
#SBATCH --output=logs/ot-human-eval-%j.out
#SBATCH --error=logs/ot-human-eval-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
# Evaluate collected scores (CPU only) on openturing (human as a (K+1)-th class).
# usage: sbatch scripts/atlas/openturing/human/eval.sh <run_name>

set -euo pipefail

export HUMAN_CLASS=1
_ATLAS_IMPL="${SLURM_SUBMIT_DIR:-.}/scripts/atlas/openturing/_impl/eval.sh"
_ATLAS_SNAP="$(mktemp "${TMPDIR:-/tmp}/atlas_impl.XXXXXX")"
cat "$_ATLAS_IMPL" > "$_ATLAS_SNAP"
bash "$_ATLAS_SNAP" "${1:-base-hc}" "${@:2}"
_ATLAS_RC=$?
rm -f "$_ATLAS_SNAP"
exit $_ATLAS_RC
