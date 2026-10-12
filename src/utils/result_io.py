"""Unified result I/O for eval, eval_ood, eval_bank_compare."""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def make_out_dir(path: str | None) -> str | None:
    if path:
        os.makedirs(path, exist_ok=True)
        results_txt = os.path.join(path, "results.txt")
        if os.path.exists(results_txt):
            os.remove(results_txt)
    return path


def save_scores(
        out_dir: str | None, 
        split: str,
        scores: np.ndarray, 
        labels: np.ndarray,
        name: str = "scores",
        ids: list[str] | None = None,
    ) -> None:
    """Save score matrix and labels for one split (+ per-row ids when given)."""
    if out_dir is None:
        return
    np.save(os.path.join(out_dir, f"{name}_{split}.npy"), scores)
    np.save(os.path.join(out_dir, f"labels_{split}.npy"), labels)
    if ids is not None and all(ids):
        np.save(os.path.join(out_dir, f"ids_{split}.npy"), np.asarray(ids))


def _format_report(
        name: str, 
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: list[str],
        label_indices: list[int] | None = None,
        per_source: bool = True
    ) -> tuple[float, float, np.ndarray, str]:
    """Build acc/F1/CM and format as a text block. Returns (acc, f1, cm, text)."""
    if label_indices is None:
        label_indices = list(range(len(label_names)))

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, labels=label_indices,average="macro", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=label_indices)

    W = 10
    lines = [f"\n[{name}] acc={acc:.4f}  macroF1={f1:.4f}"]
    header = " " * W + " " + " ".join(f"{l[:W]:>{W}}" for l in label_names)
    lines.append(header)
    for i, l in enumerate(label_names):
        lines.append(f"{l[:W]:>{W}} " + " ".join(f"{v:{W}d}" for v in cm[i]))

    if per_source:
        lines.append("\nPer-source accuracy:")
        for i, (l, idx) in enumerate(zip(label_names, label_indices)):
            mask = y_true == idx
            if mask.sum():
                acc_i = accuracy_score(y_true[mask], y_pred[mask])
                lines.append(f"  {l:>12}: {acc_i:.4f}  (n={mask.sum()})")

    return acc, f1, cm, "\n".join(lines)


def report_and_save(
        name: str, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        label_names: list[str],
        out_dir: str | None = None,
        results_store: dict | None = None,
        label_indices: list[int] | None = None,
        per_source: bool = True
    ) -> tuple[float, float, np.ndarray]:
    """Print report, append to results.txt, store in results_store dict."""
    acc, f1, cm, text = _format_report(name, y_true, y_pred, label_names, label_indices, per_source)
    print(text)
    if out_dir:
        with open(os.path.join(out_dir, "results.txt"), "a") as fp:
            fp.write(text + "\n")
    if results_store is not None:
        results_store[name] = {"acc": float(acc), "f1": float(f1), "cm": cm.tolist()}
        if name.endswith("(all)") and "human" in [str(l) for l in label_names]:
            results_store["multi"] = {"acc": float(acc), "macro_f1": float(f1),
                                      "labels": list(label_names), "cm": cm.tolist(),
                                      "source": "(K+1)-way readout, human as a class"}
    return acc, f1, cm


def save_results_json(
        out_dir: str | None,
        results_store: dict,
        meta: dict | None = None
    ) -> None:
    """Write results.json with optional meta block."""
    if out_dir is None:
        return
    data: dict = {}
    if meta:
        data["meta"] = meta
    data["results"] = results_store
    with open(os.path.join(out_dir, "results.json"), "w") as fp:
        json.dump(data, fp, indent=2)


def dump_raw_preds(out_dir, y, pred, ids, scores=None):
    """Write per-row predictions when ATLAS_DUMP_PREDS=1; otherwise do nothing."""
    if os.environ.get("ATLAS_DUMP_PREDS", "0") != "1" or not out_dir:
        return
    payload = {"y": np.asarray(y), "pred": np.asarray(pred),
               "ids": np.asarray([str(i) for i in ids])}
    if scores is not None:
        payload["scores"] = np.asarray(scores, dtype=np.float32)
    np.savez_compressed(os.path.join(out_dir, "preds_raw.npz"), **payload)
    print(f"[dump_raw_preds] {out_dir}/preds_raw.npz  n={len(payload['y'])}"
          f"{'  scores=' + str(payload['scores'].shape) if scores is not None else ''}")
