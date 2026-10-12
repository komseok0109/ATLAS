"""One optimizer contract for every trained method."""
from __future__ import annotations

import os

import torch

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BETAS = (0.9, 0.999)


def build_adamw(params, lr, weight_decay=WEIGHT_DECAY, betas=BETAS, announce=True):
    """AdamW with the shared contract. Pass params as an iterable of tensors or param groups."""
    if announce:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
        if rank == 0:
            print(f"[optim] AdamW lr={lr} weight_decay={weight_decay} betas={betas} "
                  f"grad_clip={GRAD_CLIP} (utils.optim)", flush=True)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def clip_(params, max_norm=GRAD_CLIP):
    """Clip the global grad norm in place; returns the pre-clip norm so it can be logged."""
    if not max_norm:
        return None
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def add_optim_args(p, lr_default):
    """Register --lr / --weight_decay / --grad_clip so every method exposes the same knobs."""
    p.add_argument("--lr", type=float, default=lr_default)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                   help=f"AdamW weight decay. Shared default {WEIGHT_DECAY} (see utils.optim).")
    p.add_argument("--grad_clip", type=float, default=GRAD_CLIP,
                   help=f"Global grad-norm cap; 0 disables. Shared default {GRAD_CLIP}.")
    return p
