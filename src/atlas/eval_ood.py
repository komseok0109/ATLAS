"""Evaluate ATLAS scores on an OOD dataset (numpy + sklearn, no GPU)."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.result_io import make_out_dir, save_results_json
from utils.eval_utils import (bad_rows, run_argmax, run_human_class, report_and_save,
                              build_readout, parse_prod_temps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ood_scores_dir", required=True, help="Dir with scores_all.npy + labels_all.npy from collect_scores.py (OOD).")
    p.add_argument("--prod_temps", default="",
                   help="'' = argmax over the mean log-prob score; e.g. '1,2,4,8' = argmax-prod, "
                        "the prod readout averaged over those temperatures (collect_scores.py "
                        "--prod_temps must have written them).")
    p.add_argument("--id_scores_dir", default=None, help="Dir with scores_train.npy + labels_train.npy from collect_scores.py (ID); required for --fit_on id_train.")
    p.add_argument("--ood_sources", nargs="+", required=True, help="OOD source names in label order.")
    p.add_argument("--source_map", nargs="*", default=None, help="Map OOD source → ID score dimension: 'ood_name:id_name' ...")
    p.add_argument("--id_sources", nargs="+", required=True, help="ID source names in training order (determines score column order).")

    p.add_argument("--fit_on", choices=["id_train"], default="id_train", help="Where tau and the balance constants are fit (ID train only).")
    p.add_argument("--clf", choices=["argmax"], default="argmax")
    p.add_argument("--human_class", action="store_true", help="Scores are [N,K+1] from a checkpoint trained with human as a class.")
    p.add_argument("--balance_test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def _parse_source_map(source_map_list):
    if not source_map_list:
        return {}
    return {k.strip(): v.strip() for item in source_map_list for k, v in [item.split(":", 1)]}


def main():
    args = parse_args()
    if args.fit_on == "id_train" and args.id_scores_dir is None:
        raise ValueError("--id_scores_dir required for --fit_on id_train")

    out_dir = make_out_dir(args.out_dir)
    results_store: dict = {}
    source_map = _parse_source_map(args.source_map)

    id_src_to_idx = {s: i for i, s in enumerate(args.id_sources)}
    eval_cols = []
    for ood_src in args.ood_sources:
        id_name = source_map.get(ood_src, ood_src)
        if id_name not in id_src_to_idx:
            raise ValueError(f"OOD source '{ood_src}' (→ '{id_name}') not in --id_sources. Use --source_map.")
        eval_cols.append(id_src_to_idx[id_name])
    args.human = args.human_class
    if args.human_class:
        eval_cols.append(len(args.id_sources))

    human_label = len(args.ood_sources) if args.human else None

    temps = parse_prod_temps(args.prod_temps)
    y_ood = np.load(os.path.join(args.ood_scores_dir, "labels_all.npy"))
    X_ood_full = build_readout(args.ood_scores_dir, "all", temps)

    meta_path = os.path.join(args.ood_scores_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"no meta.json in {args.ood_scores_dir}; cannot determine the collection's "
            f"label order. Re-run collect_scores.py to regenerate it.")
    with open(meta_path) as f:
        collect_ood_sources = json.load(f).get("ood_sources")
    if not collect_ood_sources:
        raise ValueError(f"meta.json in {args.ood_scores_dir} has no 'ood_sources' field.")

    missing = [s for s in args.ood_sources if s not in collect_ood_sources]
    if missing:
        raise ValueError(f"--ood_sources {missing} not found in collected ood_sources {collect_ood_sources} (meta.json)")

    collect_label_to_src = {i: s for i, s in enumerate(collect_ood_sources)}
    collect_label_to_src[len(collect_ood_sources)] = "human"
    eval_label_of = {s: i for i, s in enumerate(args.ood_sources)}
    remap = {}
    for cl, src in collect_label_to_src.items():
        if src == "human":
            if args.human:
                remap[cl] = len(args.ood_sources)
        elif src in eval_label_of:
            remap[cl] = eval_label_of[src]
    keep = np.isin(y_ood, list(remap.keys()))
    if not keep.all():
        print(f"[remap] keeping {int(keep.sum())}/{len(y_ood)} OOD rows "
              f"(collected {collect_ood_sources} → eval {list(args.ood_sources)})")
        X_ood_full, y_ood = X_ood_full[keep], y_ood[keep]
    y_ood = np.array([remap[int(l)] for l in y_ood])

    bads = bad_rows(X_ood_full)
    if bads.sum():
        print(f"[WARN] {bads.sum()} NaN/Inf in OOD scores, dropping")
        X_ood_full, y_ood = X_ood_full[~bads], y_ood[~bads]

    if not args.human_class:
        keep = y_ood < len(args.ood_sources)
        if (~keep).sum():
            print(f"[inter] dropped {int((~keep).sum())} rows outside the K source labels, "
                  f"{int(keep.sum())} left")
            X_ood_full, y_ood = X_ood_full[keep], y_ood[keep]

    X_ood = X_ood_full[:, eval_cols]

    if True:
        X_id_full = build_readout(args.id_scores_dir, "train", temps)
        y_id_full = np.load(os.path.join(args.id_scores_dir, "labels_train.npy"))

        bads_id = bad_rows(X_id_full)
        if bads_id.sum():
            print(f"[WARN] {bads_id.sum()} NaN/Inf in ID train scores, dropping")
            X_id_full, y_id_full = X_id_full[~bads_id], y_id_full[~bads_id]

        Xte = X_ood; yte = y_ood
        human_id_label = len(args.id_sources)
        human_ood_label = len(args.ood_sources)
        if args.human:
            tr_mask = np.isin(y_id_full, eval_cols) | (y_id_full == human_id_label)
        else:
            tr_mask = np.isin(y_id_full, eval_cols)
        label_map = {col: new for new, col in enumerate(eval_cols)}
        if args.human:
            label_map[human_id_label] = human_ood_label
        Xtr = X_id_full[tr_mask][:, eval_cols]
        ytr = np.array([label_map[l] for l in y_id_full[tr_mask]])

    if args.human_class:
        run_human_class(
            Xtr, ytr, Xte, yte, args.ood_sources,
            args.balance_test, args.seed, out_dir, results_store,
            lambda name, yt, yp, lbls, od, rs: report_and_save(name, yt, yp, lbls, out_dir=od, results_store=rs),
        )
        meta = {"fit_on": args.fit_on, "clf": args.clf, "prod_temps": temps, "human_class": True,
                "setting": "human_class",
                "computation": "shared",
                "binary_from": "(K+1) score margin", "multi_from": "(K+1)-way readout, human as a class",
                "num_classes": len(args.ood_sources) + 1,
                "ood_sources": args.ood_sources, "id_sources": args.id_sources, "source_map": source_map}
    elif args.clf == "argmax":
        run_argmax(Xte, yte, args.ood_sources, args.balance_test, args.seed, out_dir, results_store,
                   lambda name, yt, yp, lbls, od, rs: report_and_save(name, yt, yp, lbls, out_dir=od, results_store=rs))
        meta = {
            "fit_on": args.fit_on, "clf": args.clf, "prod_temps": temps, "setting": "inter",
            "ood_sources": args.ood_sources, "id_sources": args.id_sources, "source_map": source_map,
        }
    save_results_json(out_dir, results_store, meta)


if __name__ == "__main__":
    main()
