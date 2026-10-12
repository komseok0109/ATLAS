#!/bin/bash
# Shared environment for every entry point. Override any variable before sourcing.

_atlas_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT="${REPO_ROOT:-$_atlas_repo_root}"
unset _atlas_repo_root

export ATLAS_DATA="${ATLAS_DATA:-$REPO_ROOT/data}"
export ATLAS_RESULTS="${ATLAS_RESULTS:-$REPO_ROOT/results}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Conda env holding requirements.txt. Skipped if the prefix does not exist (use the active env).
export ATLAS_PREFIX="${ATLAS_PREFIX:-$HOME/miniforge3}"
export ATLAS_ENV="${ATLAS_ENV:-atlas}"
[ -d "$ATLAS_PREFIX/envs/$ATLAS_ENV/bin" ] && export PATH="$ATLAS_PREFIX/envs/$ATLAS_ENV/bin:$PATH"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -z "${ATLAS_ATTN:-}" ]; then
    if python -c "import flash_attn" 2>/dev/null; then
        export ATLAS_ATTN=flash_attention_2
    else
        export ATLAS_ATTN=sdpa
    fi
fi

# atlas_batch EFF_BATCH DEVICES [MAX_PER_GPU] -> "per_gpu_batch grad_accum"
atlas_batch() {
    local eff="$1" dev="$2" cap="${3:-0}" per b g
    if [ "$dev" -le 0 ] 2>/dev/null; then echo "atlas_batch: devices must be > 0" >&2; return 1; fi
    per=$(( eff / dev ))
    if [ $(( per * dev )) -ne "$eff" ] || [ "$per" -lt 1 ]; then
        echo "atlas_batch: eff_batch $eff is not divisible by devices $dev" >&2; return 1
    fi
    b="$per"; g=1
    if [ "$cap" -gt 0 ] && [ "$per" -gt "$cap" ]; then
        b="$cap"; g=$(( per / cap ))
        if [ $(( b * g )) -ne "$per" ]; then
            echo "atlas_batch: per-GPU $per is not a multiple of the cap $cap" >&2; return 1
        fi
    fi
    echo "$b $g"
}

# atlas_read REL [FILE] -> absolute path of a results subdirectory
atlas_read() {
    local here="$ATLAS_RESULTS/$1"
    [ -z "${2:-}" ] || [ -e "$here/$2" ] || echo "[atlas_read] missing: $here/$2" >&2
    echo "$here"
}

cd "$REPO_ROOT"

# Optional credentials. Fill them in here, or leave them empty to use the ones already set in
# your shell (or `hf auth login` / `wandb login`).
_HF_TOKEN=""
_WANDB_API_KEY=""
if [ -n "$_HF_TOKEN" ]; then export HF_TOKEN="$_HF_TOKEN"; fi
if [ -n "$_WANDB_API_KEY" ]; then export WANDB_API_KEY="$_WANDB_API_KEY"; fi

if [ -z "${WANDB_MODE:-}" ]; then
    if [ -n "${WANDB_API_KEY:-}" ] || grep -qs "api.wandb.ai" "$HOME/.netrc"; then
        export WANDB_MODE=online
    else
        export WANDB_MODE=offline
    fi
fi
