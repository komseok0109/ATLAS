"""Train ATLAS on an ID split."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

from utils.dataset import IDDataset, make_collate
from utils.schedule import WARMUP_RATIO, load_state as load_sched_state, warmup_cosine
from utils.optim import GRAD_CLIP, WEIGHT_DECAY, build_adamw, clip_
from model import ATLASModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to unified JSONL (e.g. maabench_id.jsonl).")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="Output directory for checkpoint.")
    ap.add_argument("--save_each_epoch", action="store_true", help="Save a checkpoint after every epoch.")
    ap.add_argument("--resume", default=None, metavar="DIR|auto", help="Checkpoint dir, or 'auto' for the newest under --out.")

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="See utils.optim.")
    ap.add_argument("--grad_clip", type=float, default=GRAD_CLIP, help="See utils.optim. 0 disables.")
    ap.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO,
                    help="Linear warmup as a fraction of total optimizer steps, then cosine decay.")

    ap.add_argument("--lam_sft", type=float, default=0.3, help="Scale of the SFT loss.")
    ap.add_argument("--init_from", default=None, metavar="DIR",
                    help="Load weights from a checkpoint with a fresh optimizer and LR schedule.")

    ap.add_argument("--human_class", action="store_true", help="Train human as an extra source with its own control embedding.")
    ap.add_argument("--per_class", type=int, default=0, help="Subsample train to N examples per class (stratified). 0 = use all.")

    ap.add_argument("--per_source_lora_b", action="store_true", help="Shared LoRA A with a separate up-projection B_k per source.")
    ap.add_argument("--prefix_len", type=int, default=1,
                    help="Soft control tokens prepended per source. 0 = conditioning through B_k only.")

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--log_every", type=int, default=20)

    ap.add_argument("--wandb_project", type=str, default="atlas")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def epoch_ckpt_dir(out, epoch):
    """Where epoch N's resumable checkpoint lives: {out}/ep{N}, nested under the run."""
    return os.path.join(out, f"ep{epoch}")


def find_latest_epoch_ckpt(out):
    """Highest-numbered resumable epoch checkpoint as (epochs_completed, path), or None."""
    def scan(directory, prefix):
        if not os.path.isdir(directory):
            return None
        best = None
        for name in os.listdir(directory):
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                continue
            path = os.path.join(directory, name)
            if not os.path.exists(os.path.join(path, "train_state.pt")):
                continue
            if best is None or int(suffix) > best[0]:
                best = (int(suffix), path)
        return best

    out = os.path.abspath(out)
    return (scan(out, "ep")
            or scan(os.path.dirname(out) or ".", os.path.basename(out) + "_ep"))


def save_epoch_ckpt(raw_model, opt, sched, epochs_completed, step, path, wandb_id=None):
    """Weights + everything needed to continue: optimizer moments, LR schedule position, counters."""
    raw_model.save_weights(path)
    torch.save({
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epochs_completed": epochs_completed,
        "step": step,
        "wandb_id": wandb_id,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, os.path.join(path, "train_state.pt"))


def main():
    args = parse_args()
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    resume_dir, resume_state = None, None
    if args.resume:
        if args.resume == "auto":
            found = find_latest_epoch_ckpt(args.out)
            if found and is_main:
                print(f"[resume] auto -> {found[1]} ({found[0]} epoch(s) done)")
            elif is_main:
                print(f"[resume] auto: no {args.out}/ep* checkpoint yet, starting fresh")
            resume_dir = found[1] if found else None
        else:
            resume_dir = args.resume
            if not os.path.exists(os.path.join(resume_dir, "train_state.pt")):
                raise FileNotFoundError(f"--resume {resume_dir} has no train_state.pt (not a resumable checkpoint)")
    if resume_dir:
        resume_state = torch.load(os.path.join(resume_dir, "train_state.pt"),
                                  map_location="cpu", weights_only=False)

    if is_main and not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args),
                   id=(resume_state or {}).get("wandb_id"), resume="allow")

    model_sources = args.sources + (["human"] if args.human_class else [])

    load_dir = resume_dir or args.init_from
    lora_r, lora_alpha, lora_targets = args.lora_r, args.lora_alpha, args.lora_target_modules
    if load_dir:
        _cfg = os.path.join(load_dir, "adapter_config.json")
        if os.path.exists(_cfg):
            with open(_cfg) as f:
                _c = json.load(f)
            lora_r, lora_alpha, lora_targets = _c["r"], _c["lora_alpha"], list(_c["target_modules"])
            if is_main:
                print(f"[init] LoRA from ckpt: r={lora_r} alpha={lora_alpha} targets={sorted(lora_targets)}")

    raw_model = ATLASModel(
        base_model_path=args.base_model,
        source_names=model_sources,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        per_source_lora_b=args.per_source_lora_b,
        prefix_len=args.prefix_len,
    )
    if is_main:
        n_tr = sum(p.numel() for p in raw_model.get_trainable_parameters())
        n_ctrl = raw_model.control_embeddings.numel()
        n_ctrl = n_ctrl if raw_model.control_embeddings.requires_grad else 0
        print(f"[params] trainable={n_tr:,}  of which control_embeddings={n_ctrl:,} "
              f"lora={n_tr - n_ctrl:,}  per_source_B={raw_model.per_source_lora_b}"
              )

    if load_dir:
        raw_model.load_weights(load_dir, trainable=True)
        if is_main:
            print(f"[init] weights from {load_dir}"
                  + ("" if resume_dir else " (weights only; optimizer state restarts)"))

    raw_model = raw_model.to(device)

    if args.grad_ckpt:
        raw_model.model.gradient_checkpointing_enable()
        raw_model.model.enable_input_require_grads()

    if is_ddp:
        model = DDP(raw_model, device_ids=[local_rank], static_graph=True)
    else:
        model = raw_model

    ds = IDDataset(args.data, args.sources, split="train",
                   include_human=args.human_class,
                   neg_seed=args.seed + local_rank)
    if args.per_class > 0:
        ds.subsample_stratified(args.per_class, seed=args.seed, verbose=is_main)
    if is_main:
        hc = " (+human class)" if args.human_class else ""
        print(f"train examples: {len(ds)}{hc}  sources={model_sources}")
        print(f"[loss] sft  (lam_sft={args.lam_sft})")

    collate = make_collate(raw_model.tokenizer, args.max_len)
    dl_generator = None
    if is_ddp:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
        dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate, worker_init_fn=lambda wid: random.seed(args.seed + wid))
    else:
        sampler = None
        dl_generator = torch.Generator()
        dl_generator.manual_seed(args.seed)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, worker_init_fn=lambda wid: random.seed(args.seed + wid), generator=dl_generator)

    if is_main:
        print(f"steps/epoch: {len(dl)}")

    opt = build_adamw(raw_model.get_trainable_parameters(), args.lr, args.weight_decay)
    total_steps = (len(dl) // args.grad_accum) * args.epochs
    sched = warmup_cosine(opt, total_steps, args.warmup_ratio)

    model.train()
    step = 0
    start_epoch = 0
    _nan_streak = 0
    NAN_ABORT_STREAK = int(os.environ.get("ATLAS_NAN_ABORT_STREAK", "10"))
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
        load_sched_state(sched, resume_state.get("scheduler"))
        start_epoch = int(resume_state["epochs_completed"])
        step = int(resume_state["step"])
        torch.set_rng_state(resume_state["torch_rng"])
        if resume_state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(resume_state["cuda_rng"])
        if start_epoch >= args.epochs:
            if is_main:
                print(f"[resume] {start_epoch}/{args.epochs} epochs already done, nothing to train")
        elif is_main:
            print(f"[resume] epoch {start_epoch}/{args.epochs}, step {step}/{total_steps}, "
                  f"lr={sched.get_last_lr()[0]:.3e}")

    for ep in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        elif dl_generator is not None:
            dl_generator.manual_seed(args.seed + ep)
        accumulated = False

        pbar = tqdm(dl, desc=f"ep{ep}", disable=not is_main, dynamic_ncols=True)
        for it, batch in enumerate(pbar):
            seq, attn, lm, y = batch
            seq, attn, lm, y = seq.to(device), attn.to(device), lm.to(device), y.to(device)

            d = model(seq, attn, y, loss_mask=lm, lam_sft=args.lam_sft)
            loss = d["loss"]

            if not torch.isfinite(loss):
                _nan_streak += 1
                if _nan_streak >= NAN_ABORT_STREAK:
                    raise RuntimeError(
                        f"loss has been non-finite for {_nan_streak} consecutive steps "
                        f"(step {step}, epoch {ep}); aborting. Try a larger --warmup_ratio or a lower --lr.")
            else:
                _nan_streak = 0

            (loss / args.grad_accum).backward()
            accumulated = True

            if (it + 1) % args.grad_accum == 0:
                clip_(raw_model.get_trainable_parameters(), args.grad_clip)
                opt.step(); sched.step(); opt.zero_grad()
                accumulated = False
                step += 1

                if is_main and step % args.log_every == 0:
                    scalars = {k: v.item() for k, v in d.items()
                               if k != "loss" and torch.is_tensor(v) and v.dim() == 0}
                    parts = " ".join(f"{k}={v:.4f}" for k, v in scalars.items())
                    pbar.set_postfix_str(f"loss={loss.item():.4f} {parts}")
                    print(f"ep{ep} step{step}/{total_steps} loss={loss.item():.4f} {parts}")
                    if not args.no_wandb:
                        wandb.log({"loss": loss.item(), "lr": sched.get_last_lr()[0], "epoch": ep, **scalars}, step=step)

        if accumulated:
            clip_(raw_model.get_trainable_parameters(), args.grad_clip)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1

        if args.save_each_epoch:
            if is_ddp:
                dist.barrier()
            if is_main:
                ckpt_dir = epoch_ckpt_dir(args.out, ep + 1)
                try:
                    save_epoch_ckpt(raw_model, opt, sched, ep + 1, step, ckpt_dir,
                                    wandb_id=(wandb.run.id if (not args.no_wandb and wandb.run) else None))
                    print(f"[ckpt] epoch {ep + 1}/{args.epochs} -> {ckpt_dir}")
                except Exception as e:
                    print(f"[ERROR] epoch ckpt failed: {e}")
                    import traceback; traceback.print_exc()
            if is_ddp:
                dist.barrier()

    if is_ddp:
        dist.barrier()
    if is_main:
        try:
            raw_model.save_weights(args.out)
            print("saved ->", args.out)
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, "hyperparams.json"), "w") as f:
                json.dump(vars(args), f, indent=2, sort_keys=True)
            print("saved hyperparams ->", os.path.join(args.out, "hyperparams.json"))
        except Exception as e:
            print(f"[ERROR] save failed: {e}")
            import traceback; traceback.print_exc()
            fallback = args.out + "_emergency.pt"
            try:
                torch.save({n: p.detach().cpu() for n, p in raw_model.named_parameters()
                            if p.requires_grad}, fallback)
                print(f"[WARN] emergency state_dict -> {fallback}")
            except Exception as e2:
                print(f"[ERROR] emergency save also failed: {e2}")
        finally:
            if not args.no_wandb:
                wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
