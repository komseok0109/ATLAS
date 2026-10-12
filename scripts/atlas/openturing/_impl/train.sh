#!/bin/bash
#SBATCH --job-name=atlas-train
#SBATCH --output=logs/atlas-train-%j.out
#SBATCH --error=logs/atlas-train-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400GB
#SBATCH --time=3-00:00:00

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"


RUN_NAME="${1:-base}"
MODEL="${2:-Qwen/Qwen2.5-7B}"
DEVICES="${DEVICES:-${SLURM_GPUS_ON_NODE:-4}}"
if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "$DEVICES" -gt "$SLURM_GPUS_ON_NODE" ]; then
    echo "DEVICES=$DEVICES exceeds the allocation (SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE)" >&2
    exit 1
fi
EFF_BATCH="${EFF_BATCH:-128}"
MAX_PER_GPU="${MAX_PER_GPU:-4}"
read -r BATCH_SIZE GRAD_ACCUM <<<"$(atlas_batch "$EFF_BATCH" "$DEVICES" "$MAX_PER_GPU")"

RESUME="${RESUME:-auto}"

PSL_FLAG=""; [ "${PER_SOURCE_LORA_B:-1}" = "1" ] && PSL_FLAG="--per_source_lora_b"
HC_FLAG=""; [ "${HUMAN_CLASS:-0}" = "1" ] && HC_FLAG="--human_class"

GC_FLAG=""; [ "${GRAD_CKPT:-1}" = "1" ] && GC_FLAG="--grad_ckpt"

echo "=== ATLAS training (openturing_id): run=${RUN_NAME} model=${MODEL} bs=${BATCH_SIZE} accum=${GRAD_ACCUM}"\
     "sources=[${SOURCES:-llama qwen gemma intel phi mistral solar}] per_class=${PER_CLASS:-0}"\
     "epochs=${EPOCHS:-5} resume=${RESUME:-off} lam_sft=${LAM_SFT:-0.3} prefix_len=${PREFIX_LEN:-1}"\
     "per_source_lora_b=${PER_SOURCE_LORA_B:-1} human_class=${HUMAN_CLASS:-0} ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| GPUs', torch.cuda.device_count())"

REL="atlas_results/train/openturing_id/$RUN_NAME"

python -m torch.distributed.run --standalone --nproc_per_node="$DEVICES" src/atlas/train.py \
    --data      "${TRAIN_DATA:-$ATLAS_DATA/openturing/openturing_id.jsonl}" \
    --base_model $MODEL \
    --sources   ${SOURCES:-llama qwen gemma intel phi mistral solar} \
    --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "${EPOCHS:-5}" \
    --lr 1e-4 \
    ${PER_CLASS:+--per_class $PER_CLASS} \
    --lam_sft "${LAM_SFT:-0.3}" \
    --prefix_len "${PREFIX_LEN:-1}" \
    --lora_r 16 --lora_alpha 32 \
    --max_len 1024 $GC_FLAG $PSL_FLAG $HC_FLAG \
    --seed "${SEED:-42}" \
    --save_each_epoch ${RESUME:+--resume "$RESUME"} \
    --log_every 20 \
    --wandb_project atlas \
    --wandb_run_name "openturing_id_atlas_$RUN_NAME" \
    --out "$ATLAS_RESULTS/$REL"

echo "=== Done: atlas_results/train/openturing_id/${RUN_NAME}/ ==="
