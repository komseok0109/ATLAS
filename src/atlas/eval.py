"""Evaluate ATLAS scores on the ID dataset (numpy + sklearn, no GPU)."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.result_io import make_out_dir, save_results_json
from utils.eval_utils import (bad_rows, run_argmax, run_human_class, report_and_save,
                              build_readout, parse_prod_temps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_dir", required=True, help="Dir with scores_train/test.npy + labels_train/test.npy from collect_scores.py.")
    p.add_argument("--sources", nargs="+", required=True, help="ID source names in training order.")
    p.add_argument("--clf", choices=["argmax"], default="argmax")
    p.add_argument("--human_class", action="store_true", help="Scores are [N,K+1] from a checkpoint trained with human as a class. --sources lists the K LLM sources.")
    p.add_argument("--balance_test", action="store_true")
    p.add_argument("--prod_temps", default="",
                   help="'' = argmax over the mean log-prob score; e.g. '1,2,4,8' = argmax-prod, "
                        "the prod readout averaged over those temperatures (collect_scores.py "
                        "--prod_temps must have written them).")
    p.add_argument("--eval_split", choices=["test", "val"], default="test",
                   help="Held-out split to evaluate; constants are still fit on train. 'val' needs "
                        "collect_scores.py --split val.")
    p.add_argument("--eval_sources", nargs="+", default=None, help="Subset of sources for an additional sub-evaluation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = make_out_dir(args.out_dir)
    results_store: dict = {}

    d = args.scores_dir
    ytr = np.load(os.path.join(d, "labels_train.npy"))
    yte = np.load(os.path.join(d, f"labels_{args.eval_split}.npy"))
    temps = parse_prod_temps(args.prod_temps)
    Xtr = build_readout(d, "train", temps)
    Xte = build_readout(d, args.eval_split, temps)

    if not args.human_class:
        for nm in ("train", args.eval_split):
            X, y = (Xtr, ytr) if nm == "train" else (Xte, yte)
            keep = y < len(args.sources)
            if (~keep).sum():
                print(f"[inter] {nm}: dropped {(~keep).sum()} rows outside the K source labels, "
                      f"{keep.sum()} left")
                if nm == "train": Xtr, ytr = X[keep], y[keep]
                else: Xte, yte = X[keep], y[keep]

    for tag, X, y in [("train", Xtr, ytr), (args.eval_split, Xte, yte)]:
        bads = bad_rows(X)
        if bads.sum():
            print(f"[WARN] {bads.sum()} NaN/Inf in {tag}, dropping")
            if tag == "train":
                Xtr, ytr = X[~bads], y[~bads]
            else:
                Xte, yte = X[~bads], y[~bads]

    if args.human_class:
        run_human_class(
            Xtr, ytr, Xte, yte, args.sources,
            args.balance_test, args.seed, out_dir, results_store, report_and_save,
        )
        meta = {"scores_dir": args.scores_dir, "sources": args.sources, "setting": "human_class",
                "computation": "shared",
                "binary_from": "(K+1) score margin", "multi_from": "argmax over K columns",
                "clf": args.clf, "prod_temps": temps, "human_class": True,
                "num_classes": len(args.sources) + 1}
    elif args.clf == "argmax":
        run_argmax(Xte, yte, args.sources, args.balance_test, args.seed, out_dir, results_store,
                   report_and_save)
        meta = {"scores_dir": args.scores_dir, "sources": args.sources, "clf": "argmax",
                "prod_temps": temps, "setting": "inter"}

    if args.eval_sources:
        unknown = [s for s in args.eval_sources if s not in args.sources]
        if unknown:
            raise ValueError(f"--eval_sources contains unknown sources: {unknown}")
        eval_cols = [args.sources.index(s) for s in args.eval_sources]
        mask_tr = np.isin(ytr, eval_cols); mask_te = np.isin(yte, eval_cols)
        Xtr_sub = Xtr[mask_tr][:, eval_cols]; Xte_sub = Xte[mask_te][:, eval_cols]
        label_map = {old: new for new, old in enumerate(eval_cols)}
        ytr_sub = np.array([label_map[l] for l in ytr[mask_tr]])
        yte_sub = np.array([label_map[l] for l in yte[mask_te]])
        sc = StandardScaler().fit(Xtr_sub)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
        clf.fit(sc.transform(Xtr_sub), ytr_sub)
        pred_sub = clf.predict(sc.transform(Xte_sub))
        report_and_save(f"test (eval_sources={args.eval_sources})", yte_sub, pred_sub,
                        args.eval_sources, out_dir=out_dir, results_store=results_store)
        meta["eval_sources"] = args.eval_sources

    save_results_json(out_dir, results_store, meta)


if __name__ == "__main__":
    main()
