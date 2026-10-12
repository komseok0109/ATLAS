#!/bin/bash
#SBATCH --job-name=atlas-maa-eval
#SBATCH --output=logs/atlas-maa-eval-%j.out
#SBATCH --error=logs/atlas-maa-eval-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"

RUN_NAME="${1:-base}"
SRCS="${SOURCES:-gpt claude gemini mistral llama}"
AXES="${AXES:-maabench_id maabench_domain maabench_version maabench_joint}"
ID_SPLITS="${ID_SPLITS:-test val}"
ID_SCORES="$(atlas_read "atlas_results/scores/id/maabench/maabench_id/$RUN_NAME" labels_test.npy)"

if [ "${HUMAN_CLASS:-0}" = "1" ]; then SETTING="human_class"; GATE="--human_class"
else SETTING="inter"; GATE=""; fi

PROD_TEMPS="${PROD_TEMPS-1,2,4,8}"   # must match what collect_scores wrote
# Two readouts: argmax over the mean log-prob score, and argmax-prod over PROD_TEMPS.
READOUTS="argmax="
[ -n "$PROD_TEMPS" ] && READOUTS="$READOUTS argmax-prod=$PROD_TEMPS"

echo "=== ATLAS eval (maabench): run=${RUN_NAME} setting=$SETTING axes=[${AXES}] ==="

fail=0
run() { echo ""; echo "--- $1 ---"; shift
        if "$@"; then echo "--- OK ---"; else echo "--- FAILED (rc=$?) ---"; fail=1; fi; }

for ax in $AXES; do
    if [ "$ax" = "maabench_id" ]; then
        for sp in $ID_SPLITS; do
            bench="maabench_id"; [ "$sp" != "test" ] && bench="maabench_id_$sp"
            OUT="$ATLAS_RESULTS/atlas_results/id_results/$SETTING/maabench/$bench/$RUN_NAME"
            for ro in $READOUTS; do
                run "$bench ($sp) ${ro%%=*}" python src/atlas/eval.py --scores_dir "$ID_SCORES" --sources $SRCS \
                    --eval_split "$sp" --prod_temps "${ro#*=}" \
                    --clf argmax $GATE --balance_test --seed 42 --out_dir "$OUT/${ro%%=*}"
            done
        done
        continue
    fi

    SD="$(atlas_read "atlas_results/scores/ood/maabench/$ax/$RUN_NAME" labels_all.npy)"
    OUT="$ATLAS_RESULTS/atlas_results/ood_results/$SETTING/maabench/$ax/$RUN_NAME"
    [ -d "$SD" ] || { echo ""; echo "=== $ax: no scores at $SD, skipping ==="; continue; }
    for ro in $READOUTS; do
        run "$ax ${ro%%=*}" python src/atlas/eval_ood.py --ood_scores_dir "$SD" --id_scores_dir "$ID_SCORES" \
            --ood_sources $SRCS --id_sources $SRCS --prod_temps "${ro#*=}" \
            --clf argmax --fit_on id_train $GATE --balance_test --seed 42 --out_dir "$OUT/${ro%%=*}"
    done
done

echo ""; echo "=== run=$RUN_NAME EXIT=$fail ==="
exit $fail
