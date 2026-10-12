"""Score a dataset with a trained ATLAS checkpoint and save the score matrices."""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.dataset import IDDataset, OODDataset, make_collate, row_ids
from utils.result_io import save_scores
from model import ATLASModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to JSONL dataset.")
    p.add_argument("--base_model", required=True, help="Base model HF name or local path.")
    p.add_argument("--ckpt", required=True, help="ATLASModel checkpoint directory.")
    p.add_argument("--sources", nargs="+", required=True, help="ID source names in training order (for ATLASModel).")
    p.add_argument("--dataset_type", choices=["id", "ood"], required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all"], required=True, help="'train'/'val'/'test' for ID; 'all' for OOD.")
    p.add_argument("--ood_sources", nargs="+", default=None, help="OOD source names in label order (required when --dataset_type ood).")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--prod_temps", default="1,2,4,8",
                   help="Temperatures of the argmax-prod readout, comma-separated ('' = none). "
                        "Each writes scores_prod{T}_{split}.npy.")
    p.add_argument("--per_source_lora_b", action="store_true",
                   help="a2 checkpoint; must match the training flag.")
    p.add_argument("--include_human", action="store_true")
    p.add_argument("--human_class", action="store_true", help="Checkpoint has human as a (K+1)-th control embedding.")
    p.add_argument("--fit_per_class", type=int, default=0, help="Subsample ID train stratified to this many per class before extraction. 0 = no limit.")
    p.add_argument("--test_ratio", type=float, default=0.2, help="OOD held-out fraction passed to OODDataset (only affects split='train'/'test').")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def _reassemble_by_index(all_data, n):
    """Scatter per-rank (scores, labels, idx) back into dataset order by reported index."""
    scores0, labels0, _ = all_data[0]
    scores_full = np.empty((n,) + tuple(scores0.shape[1:]), dtype=scores0.dtype)
    labels_full = np.empty(n, dtype=labels0.dtype)
    seen = np.zeros(n, dtype=bool)
    for scores, labels, idx in all_data:
        idx = np.asarray(idx)
        scores_full[idx] = scores
        labels_full[idx] = labels
        seen[idx] = True
    missing = int((~seen).sum())
    if missing:
        raise RuntimeError(f"DDP reassembly missed {missing}/{n} samples")
    return scores_full, labels_full


@torch.no_grad()
def extract_scores(model, ds, args, device, is_ddp=False, local_rank=0, world_size=1):
    if is_ddp:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False)
        idx_local = np.array(list(sampler), dtype=np.int64)
        dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, collate_fn=make_collate(model.tokenizer, args.max_len))
    else:
        idx_local = np.arange(len(ds), dtype=np.int64)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(model.tokenizer, args.max_len))

    scores, labels = [], []
    for seq, attn, _, y in tqdm(dl, desc="extract", disable=local_rank != 0, dynamic_ncols=True):
        seq, attn = seq.to(device), attn.to(device)
        s = model.get_score_vectors(seq, attn, args._temps).float()      # [B, K, 1+T]
        scores.append(s.cpu().numpy())
        labels.append(y.numpy())

    scores_local = np.concatenate(scores)
    labels_local = np.concatenate(labels)

    if not is_ddp:
        return scores_local, labels_local

    torch.cuda.empty_cache()
    assert len(idx_local) == len(scores_local), f"idx/score length mismatch: {len(idx_local)} vs {len(scores_local)}"
    all_data = [None] * world_size
    dist.all_gather_object(all_data, (scores_local, labels_local, idx_local))
    if local_rank != 0:
        return None, None
    return _reassemble_by_index(all_data, len(ds))


def main():
    args = parse_args()
    if args.dataset_type == "ood" and not args.ood_sources:
        raise ValueError("--ood_sources required when --dataset_type ood")
    args._temps = [float(t) for t in args.prod_temps.replace(":", ",").split(",") if t.strip()]

    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=6))
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    model_sources = args.sources + (["human"] if args.human_class else [])
    include_human = args.include_human or args.human_class

    lora_kw = {}
    _cfg = os.path.join(args.ckpt, "adapter_config.json")
    if os.path.exists(_cfg):
        with open(_cfg) as f:
            _c = json.load(f)
        lora_kw = {"target_modules": tuple(_c["target_modules"]),
                   "lora_r": _c["r"], "lora_alpha": _c["lora_alpha"]}
        if local_rank == 0:
            print(f"[collect] LoRA from ckpt: r={_c['r']} alpha={_c['lora_alpha']} "
                  f"targets={sorted(_c['target_modules'])}", flush=True)

    model = ATLASModel(
        args.base_model, model_sources,
        per_source_lora_b=args.per_source_lora_b,
        **lora_kw,
    )
    model.load_weights(args.ckpt)
    model = model.to(device)
    model.eval()

    if args.dataset_type == "id":
        ds = IDDataset(args.data, args.sources, split=args.split, include_human=include_human)
        if args.fit_per_class > 0 and args.split == "train":
            ds.subsample_stratified(args.fit_per_class, seed=args.seed)
    else:
        ds = OODDataset(
            args.data, args.ood_sources,
            split=args.split, test_ratio=args.test_ratio,
            seed=args.seed, include_human=include_human,
        )

    if is_main:
        print(f"Dataset: {len(ds)} examples  ({args.dataset_type}/{args.split})")

    scores, labels = extract_scores(model, ds, args, device, is_ddp, local_rank, world_size)

    if not is_main:
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rid = row_ids(ds)
    save_scores(args.out_dir, args.split, scores[:, :, 0], labels, ids=rid)
    for j, T in enumerate(args._temps):
        save_scores(args.out_dir, args.split, scores[:, :, 1 + j], labels,
                    name=f"scores_prod{T:g}", ids=rid)
    if args._temps:
        print(f"[prod] also wrote scores_prod{{{','.join(f'{T:g}' for T in args._temps)}}}_{args.split}.npy")

    meta = {
        "ckpt": args.ckpt, "base_model": args.base_model,
        "sources": args.sources, "ood_sources": args.ood_sources,
        "dataset_type": args.dataset_type, "split": args.split,
        "include_human": include_human,
        "seed": args.seed, "n": int(len(labels)),
    }
    if args.human_class:
        meta["human_class"] = True
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved scores_{args.split}.npy + labels_{args.split}.npy → {args.out_dir}/")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
