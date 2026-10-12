#!/bin/bash
#SBATCH --job-name=atlas-ot-collect
#SBATCH --output=logs/atlas-ot-collect-%j.out
#SBATCH --error=logs/atlas-ot-collect-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"

RUN_NAME="${1:-base}"
MODEL="${2:-}"
MAX_LEN="${3:-1024}"
SRCS="llama qwen gemma intel phi mistral solar"
AXES="${AXES:-openturing_id openturing_ood openturing_e1 openturing_e2 openturing_e3 openturing_e4}"
CKPT="$(atlas_read "atlas_results/train/openturing_id/$RUN_NAME" adapter_model.safetensors)"

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
HC_FLAG=""; [ "${HUMAN_CLASS:-0}" = "1" ] && HC_FLAG="--human_class"
FPC_FLAG=""; [ "${FIT_PER_CLASS:-0}" != "0" ] && FPC_FLAG="--fit_per_class ${FIT_PER_CLASS}"
PROD_TEMPS="${PROD_TEMPS-1,2,4,8}"   # temperatures of the argmax-prod readout

echo "=== ATLAS collect_scores (openturing): run=${RUN_NAME} axes=[${AXES}] human_class=${HUMAN_CLASS:-0} ==="
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
        --sources $SRCS \
        --ckpt "$CKPT" $PSL_FLAG $HC_FLAG --prod_temps "$PROD_TEMPS" \
        --batch_size "${BATCH:-4}" --max_len "$MAX_LEN" \
        --dataset_type "$dtype" --split "$split" --seed 42 "$@" \
        --out_dir "$sdir"; then echo "--- OK ---"
    else echo "--- FAILED (rc=$?) ---"; fail=1; fi
}

echo ""; echo "########## scoring all axes ##########"
for ax in $AXES; do
    DATA="$ATLAS_DATA/openturing/${ax}.jsonl"
    if [ ! -f "$DATA" ]; then echo ""; echo "=== $ax: no $DATA, skipping ==="; continue; fi
    if [ "$ax" = "openturing_id" ]; then
        SDIR="$ATLAS_RESULTS/atlas_results/scores/id/openturing_id/$RUN_NAME"
        for sp in train val test; do collect "$SDIR" "$DATA" id "$sp" --include_human $FPC_FLAG; done
    else
        SDIR="$ATLAS_RESULTS/atlas_results/scores/ood/$ax/$RUN_NAME"
        collect "$SDIR" "$DATA" ood all --ood_sources $SRCS
    fi
done

echo ""; echo "=== run=$RUN_NAME EXIT=$fail ==="
exit $fail
