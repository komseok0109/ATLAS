#!/bin/bash
#SBATCH --job-name=atlas-maa-collect
#SBATCH --output=logs/atlas-maa-collect-%j.out
#SBATCH --error=logs/atlas-maa-collect-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"

RUN_NAME="${1:-base}"
MODEL="${2:-}"
MAX_LEN="${3:-512}"
SOURCES="${SOURCES:-gpt claude gemini mistral llama}"
OOD_SOURCES="${OOD_SOURCES:-$SOURCES}"
AXES="${AXES:-maabench_id maabench_domain maabench_version maabench_joint}"
CKPT="$(atlas_read "atlas_results/train/maabench/maabench_id/$RUN_NAME" adapter_model.safetensors)"

if [ -z "${MODEL:-}" ]; then
    _hp="$(dirname "$CKPT")/hyperparams.json"; [ -f "$_hp" ] || _hp="$CKPT/hyperparams.json"
    MODEL="$(python - "$_hp" <<'PYHP' 2>/dev/null
import json, sys
try: print(json.load(open(sys.argv[1]))["base_model"])
except Exception: pass
PYHP
)"
    MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
    echo "[collect] base_model from checkpoint: $MODEL"
fi
DEVICES="${DEVICES:-${SLURM_GPUS_ON_NODE:-4}}"
PSL_FLAG=""; [ "${PER_SOURCE_LORA_B:-1}" = "1" ] && PSL_FLAG="--per_source_lora_b"
PROD_TEMPS="${PROD_TEMPS-1,2,4,8}"   # temperatures of the argmax-prod readout
HC_FLAG=""; [ "${HUMAN_CLASS:-0}" = "1" ] && HC_FLAG="--human_class"
FPC_FLAG=""; [ "${FIT_PER_CLASS:-0}" != "0" ] && FPC_FLAG="--fit_per_class ${FIT_PER_CLASS}"

echo "=== ATLAS collect_scores (maabench): run=${RUN_NAME} axes=[${AXES}] human_class=${HUMAN_CLASS:-0} ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| GPUs', torch.cuda.device_count())"

fail=0
_done_under() {
    local sdir="$1" split="$2" f
    [ -s "$sdir/labels_${split}.npy" ] || return 1
    [ -s "$sdir/scores_${split}.npy" ] || return 1
    if [ -n "$PROD_TEMPS" ]; then
        local IFS=',' t
        for t in $PROD_TEMPS; do
            f="$sdir/scores_prod${t}_${split}.npy"
            [ -s "$f" ] || return 1
        done
    fi
    return 0
}

_already_done() {
    local sdir="$1" split="$2"
    [ "${SKIP_DONE:-1}" = "1" ] || return 1
    _done_under "$sdir" "$split" && return 0
    return 1
}

collect() {
    local sdir="$1" data="$2" dtype="$3" split="$4"; shift 4
    if _already_done "$sdir" "$split"; then
        echo ""; echo "=== $dtype/$split -> $sdir : already complete, skipping (SKIP_DONE=0 to redo) ==="
        return 0
    fi
    echo ""; echo "=== $dtype/$split -> $sdir ==="
    if python -m torch.distributed.run --standalone --nproc_per_node="$DEVICES" src/atlas/collect_scores.py \
        --data "$data" --base_model "$MODEL" \
        --sources $SOURCES \
        --ckpt "$CKPT" --include_human $PSL_FLAG $HC_FLAG --prod_temps "$PROD_TEMPS" \
        --batch_size "${BATCH:-4}" --max_len "$MAX_LEN" \
        --dataset_type "$dtype" --split "$split" --seed 42 "$@" \
        --out_dir "$sdir"; then echo "--- OK ---"
    else echo "--- FAILED (rc=$?) ---"; fail=1; fi
}

echo ""; echo "########## scoring all axes ##########"
AXIS_DIR="${AXIS_DIR:-maabench}"
for ax in $AXES; do
    DATA="$ATLAS_DATA/$AXIS_DIR/${ax}.jsonl"
    if [ ! -f "$DATA" ]; then echo ""; echo "=== $ax: no $DATA, skipping ==="; continue; fi
    if [ "$ax" = "maabench_id" ]; then
        SDIR="$ATLAS_RESULTS/atlas_results/scores/id/maabench/maabench_id/$RUN_NAME"
        for sp in ${ID_SPLITS:-train val test}; do collect "$SDIR" "$DATA" id "$sp" $FPC_FLAG; done
    else
        SDIR="$ATLAS_RESULTS/atlas_results/scores/ood/$AXIS_DIR/$ax/$RUN_NAME"
        collect "$SDIR" "$DATA" ood all --ood_sources $OOD_SOURCES
    fi
done

echo ""; echo "=== run=$RUN_NAME EXIT=$fail ==="
exit $fail
