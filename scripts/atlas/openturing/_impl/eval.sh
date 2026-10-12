#!/bin/bash
#SBATCH --job-name=atlas-ot-eval
#SBATCH --output=logs/atlas-ot-eval-%j.out
#SBATCH --error=logs/atlas-ot-eval-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"

RUN_NAME="${1:-base}"
SRCS="llama qwen gemma intel phi mistral solar"
AXES="${AXES:-openturing_id openturing_ood openturing_e1 openturing_e2 openturing_e3 openturing_e4}"
ID_SCORES="$(atlas_read "atlas_results/scores/id/openturing_id/$RUN_NAME" labels_test.npy)"

if [ "${HUMAN_CLASS:-0}" = "1" ]; then GATE="--human_class"; else GATE=""; fi

echo "=== ATLAS eval (openturing): run=${RUN_NAME} axes=[${AXES}] human_class=${HUMAN_CLASS:-0} ==="

PROD_TEMPS="${PROD_TEMPS-1,2,4,8}"   # must match what collect_scores wrote
# Two readouts: argmax over the mean log-prob score, and argmax-prod over PROD_TEMPS.
READOUTS="argmax="
[ -n "$PROD_TEMPS" ] && READOUTS="$READOUTS argmax-prod=$PROD_TEMPS"

fail=0
run() { echo ""; echo "--- $1 ---"; shift
        if "$@"; then echo "--- OK ---"; else echo "--- FAILED (rc=$?) ---"; fail=1; fi; }

for ax in $AXES; do
    if [ "$ax" = "openturing_id" ]; then
        OUT="$ATLAS_RESULTS/atlas_results/id_results/openturing_id/$RUN_NAME"
        [ -d "$ID_SCORES" ] || { echo ""; echo "=== $ax: no scores at $ID_SCORES, skipping ==="; continue; }
        for ro in $READOUTS; do
            run "$ax ${ro%%=*}" python src/atlas/eval.py --scores_dir "$ID_SCORES" --sources $SRCS \
                --prod_temps "${ro#*=}" --clf argmax $GATE --balance_test --seed 42 --out_dir "$OUT/${ro%%=*}"
        done
        continue
    fi

    SD="$(atlas_read "atlas_results/scores/ood/$ax/$RUN_NAME" labels_all.npy)"
    OUT="$ATLAS_RESULTS/atlas_results/ood_results/$ax/$RUN_NAME"
    [ -d "$SD" ] || { echo ""; echo "=== $ax: no scores at $SD, skipping ==="; continue; }
    for ro in $READOUTS; do
        run "$ax ${ro%%=*}" python src/atlas/eval_ood.py --ood_scores_dir "$SD" --id_scores_dir "$ID_SCORES" \
            --ood_sources $SRCS --id_sources $SRCS --prod_temps "${ro#*=}" \
            --clf argmax --fit_on id_train $GATE --balance_test --seed 42 --out_dir "$OUT/${ro%%=*}"
    done
done

echo ""; echo "=== run=$RUN_NAME EXIT=$fail ==="
exit $fail
