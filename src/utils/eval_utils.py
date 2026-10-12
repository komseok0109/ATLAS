"""Shared evaluation utilities: score readouts and metric reporting."""
import os

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

from utils.result_io import report_and_save


def bad_rows(X):
    """Return boolean mask of rows containing NaN or Inf."""
    return np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)


def balance(scores, labels, extra=None, seed=42):
    """Subsample each class to the size of the smallest class (stratified, without replacement)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    n_per = min((labels == c).sum() for c in classes)
    idx = rng.permutation(np.concatenate([
        rng.choice(np.where(labels == c)[0], n_per, replace=False) for c in classes
    ]))
    if extra is not None:
        return scores[idx], labels[idx], [extra[i] for i in idx]
    return scores[idx], labels[idx]


def build_readout(scores_dir, split, prod_temps=()):
    """The attribution matrix [N, K]: the mean log-prob score (argmax), or with prod_temps the
    argmax-prod readout averaged over those temperatures."""
    if not prod_temps:
        return np.load(os.path.join(scores_dir, f"scores_{split}.npy"))
    cols = []
    for T in prod_temps:
        f = os.path.join(scores_dir, f"scores_prod{T:g}_{split}.npy")
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"{scores_dir} has no {os.path.basename(f)}. Re-run collect_scores.py with "
                f"--prod_temps covering {','.join(f'{t:g}' for t in prod_temps)}.")
        cols.append(np.load(f).astype(np.float64))
    return np.mean(cols, axis=0)


def parse_prod_temps(spec):
    """'1,2,4,8' -> [1.0, 2.0, 4.0, 8.0]; '' -> [] (plain argmax)."""
    return [float(t) for t in str(spec).replace(":", ",").split(",") if t.strip()]


def put_inter(results_store, name, sources):
    """The inter setting's one number, promoted from whatever the reporter already stored."""
    if results_store is None:
        return
    r = results_store.get(name)
    if not r:
        return
    results_store["inter"] = {"acc": float(r["acc"]), "macro_f1": float(r["f1"]),
                              "sources": list(sources), "cm": r.get("cm", []),
                              "source": "argmax over the K machine columns, no human anywhere"}


def put_binary(results_store, score_machine_high, is_machine, source, out_dir=None,
               slot="binary", extra=None, label=None):
    """Fill the binary slot from a machine-high score. slot='binary_alt' for a second detector."""
    if results_store is None:
        return None
    is_machine = np.asarray(is_machine).astype(bool)
    if is_machine.all() or not is_machine.any():
        return None
    det = detection_metrics(score_machine_high, is_machine)
    det.update({"positive_class": "machine", "source": source,
                "machine_prevalence": float(is_machine.mean()), "n": int(is_machine.size)})
    if extra:
        det.update(extra)
    if slot == "binary" and "binary" in results_store:
        slot = "binary_alt"
    if slot == "binary_alt":
        results_store.setdefault("binary_alt", []).append(det)
    else:
        results_store["binary"] = det
    line = (f"\n[{label or slot}, machine=positive]  TPR@1%FPR={100*det['tpr_at_fpr1']:.2f}%  "
            f"AUPR={100*det['aupr']:.2f}%  AUROC={100*det['auroc']:.2f}%"
            f"   ({source}; n={det['n']}, machine {100*det['machine_prevalence']:.1f}%)")
    print(line)
    if out_dir:
        with open(os.path.join(out_dir, "results.txt"), "a") as f:
            f.write(line + "\n")
    return det


def put_multi(results_store, acc, f1, cm, sources, source, extra=None):
    """Fill the multi slot: the (K+1)-way readout, human as one of the classes."""
    if results_store is None:
        return
    d = {"acc": float(acc), "macro_f1": float(f1), "sources": list(sources), "source": source,
         "cm": cm.tolist() if hasattr(cm, "tolist") else cm}
    if extra:
        d.update(extra)
    results_store["multi"] = d


def detection_metrics(score_machine_high, is_machine):
    """The three numbers the MGT-detection literature reports, plus the operating points around them."""
    m = np.asarray(is_machine).astype(int)
    s_ = np.asarray(score_machine_high, dtype=np.float64)
    fp, tp, _ = roc_curve(m, s_)
    at = lambda f: float(tp[min(np.searchsorted(fp, f), len(tp) - 1)])
    fpr_at_tpr95 = float(fp[min(np.searchsorted(tp, 0.95), len(fp) - 1)])
    return {"auroc": float(roc_auc_score(m, s_)),
            "aupr": float(average_precision_score(m, s_)),
            "tpr_at_fpr1": at(0.01), "tpr_at_fpr5": at(0.05),
            "tpr_at_fpr01": at(0.001), "tpr_at_fpr001": at(0.0001),
            "fpr_at_tpr95": fpr_at_tpr95}


def report_detection(S, y_te, human_label, columns=None, out_dir=None, results_store=None,
                     tag="binary", slot="binary"):
    """Human-vs-machine detection metrics for a (K+1)-way scorer, from its own score matrix."""
    S = np.asarray(S, dtype=np.float64)
    if human_label is None or S.ndim != 2:
        return None
    cols_id = list(range(S.shape[1])) if columns is None else [int(c) for c in columns]
    if len(cols_id) != S.shape[1] or human_label not in cols_id:
        return None
    h = cols_id.index(human_label)
    cols = [i for i in range(S.shape[1]) if i != h]
    return put_binary(results_store, S[:, cols].max(axis=1) - S[:, h], y_te != human_label,
                      "(K+1) score margin: max_k S_k - S_human", out_dir=out_dir,
                      slot=slot, label=tag)


def report_inter_llm(name, y_true, y_pred, sources, human_label, out_dir=None, results_store=None):
    """Report acc/F1/confusion matrix on the LLM-only subset, masking human rows internally."""
    mask = (y_true != human_label) if human_label is not None else np.ones(len(y_true), dtype=bool)
    yt, yp = y_true[mask], y_pred[mask]
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", labels=list(range(len(sources))), zero_division=0)
    human_pred_n = int((yp == human_label).sum()) if human_label is not None else 0
    header = f"\n[{name}]  acc={acc:.4f}  macroF1={f1:.4f}"
    if human_pred_n:
        header += f"  predicted-as-human={human_pred_n}/{len(yt)}"
    print(header)
    lines = [header]
    non_human = yp != human_label if human_label is not None else np.ones(len(yp), dtype=bool)
    cm_kk = None
    if non_human.sum() > 0:
        cm_kk = confusion_matrix(yt[non_human], yp[non_human], labels=list(range(len(sources))))
        W = 10
        hdr = " " * W + " " + " ".join(f"{l[:W]:>{W}}" for l in sources)
        print(hdr); lines.append(hdr)
        for i, l in enumerate(sources):
            row = f"{l[:W]:>{W}} " + " ".join(f"{v:{W}d}" for v in cm_kk[i])
            print(row); lines.append(row)
    if out_dir:
        with open(os.path.join(out_dir, "results.txt"), "a") as f:
                f.write("\n".join(lines) + "\n")
    if results_store is not None:
        results_store[name] = {
            "acc": float(acc), "f1": float(f1),
            "predicted_as_human": human_pred_n,
            "cm": cm_kk.tolist() if cm_kk is not None else [],
        }
        if human_label is None:
            results_store.setdefault("inter", {
                "acc": float(acc), "macro_f1": float(f1), "sources": list(sources),
                "cm": cm_kk.tolist() if cm_kk is not None else [],
                "source": "argmax over the K machine columns, no human anywhere"})


def run_human_class(Xtr, ytr, Xte, yte, sources,
                    balance_test, seed, out_dir, results_store, report_fn):
    """(K+1)-way readout with human as an ordinary class. Reports 'test (all)' and 'test (inter-LLM)';
    the latter masks true-human rows and counts a human prediction on an LLM row as an error."""
    human_label = len(sources)
    label_names = list(sources) + ["human"]

    pred = np.argmax(Xte, axis=1)

    report_detection(Xte, yte, human_label, out_dir=out_dir, results_store=results_store)

    yb, pb = yte, pred
    if balance_test:
        _, yb, exl = balance(Xte, yte, extra=list(pred), seed=seed)
        pb = np.array(exl)
    report_fn("test (all)", yb, pb, label_names, out_dir, results_store)

    mll = yte != human_label
    yl, pl = yte[mll], pred[mll]
    if balance_test:
        _, yl, exl = balance(Xte[mll], yte[mll], extra=list(pred[mll]), seed=seed)
        pl = np.array(exl)
    report_inter_llm("test (inter-LLM)", yl, pl, sources, human_label, out_dir=out_dir, results_store=results_store)


def run_argmax(Xte, yte, sources, balance_test, seed, out_dir, results_store, report_fn):
    """K-way attribution by the highest score column."""
    Zte = Xte
    if balance_test:
        Zte, yte = balance(Zte, yte, seed=seed)
    report_fn("test (inter-LLM)", yte, np.argmax(Zte, axis=1), sources, out_dir, results_store)
    put_inter(results_store, "test (inter-LLM)", sources)
