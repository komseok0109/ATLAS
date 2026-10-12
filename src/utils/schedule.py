"""One LR schedule, shared by every trained method."""
import math

import torch

WARMUP_RATIO = 0.05
MIN_LR_RATIO = 0.0


def warmup_cosine(optimizer, total_steps, warmup_ratio=WARMUP_RATIO, min_lr_ratio=MIN_LR_RATIO):
    """Linear warmup over `warmup_ratio` of total_steps, then cosine decay to `min_lr_ratio * lr`."""
    total = max(1, int(total_steps))
    warmup = max(1, int(round(total * warmup_ratio)))

    def factor(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total - warmup))
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def load_state(sched, state, prefix="[schedule]"):
    """Restore scheduler state, skipping it if it was saved by a different scheduler class."""
    if not state:
        return False
    try:
        sched.load_state_dict(state)
        return True
    except (KeyError, TypeError, ValueError) as e:
        print(f"{prefix} ignoring incompatible scheduler state ({type(e).__name__}: {e}). "
              f"The optimizer resumes; the LR schedule restarts from step 0.", flush=True)
        return False
