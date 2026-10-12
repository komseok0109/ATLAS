"""Build data/maabench/seeds/seeds.jsonl, the input generate.py consumes."""
import argparse, hashlib, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "src")
from .config import AXES, ID_DOMAINS, OOD_DOMAINS, COUNTS, DOMAINS
from . import corpora

OUT_DIR = "data/maabench/seeds"


def _sid(domain, text):
    return f"{domain}-{hashlib.sha1(text.encode()).hexdigest()[:12]}"


def _max_needed(domain):
    """Number of seeds to draw for a domain. The version and joint axes reuse id_test and domain seeds."""
    if domain in ID_DOMAINS:
        return (COUNTS["id_train"] + COUNTS["id_val"] + COUNTS["id_test"]) // len(ID_DOMAINS)
    return COUNTS["domain"] // len(OOD_DOMAINS)


def _assign(domain, ordered):
    """Split a domain's ordered seeds into axis/split cells. Version seeds are the id_test seeds and
    joint seeds are the domain seeds."""
    rows, i = [], 0
    def take(k, axis, split):
        nonlocal i
        sl = ordered[i:i + k]; i += k
        for s in sl:
            rows.append({**s, "axis": axis, "split": split})
        return sl
    if domain in ID_DOMAINS:
        d = len(ID_DOMAINS)
        take(COUNTS["id_train"] // d, "maabench_id", "train")
        take(COUNTS["id_val"] // d,   "maabench_id", "val")
        test_slice = take(COUNTS["id_test"] // d, "maabench_id", "test")
        for s in test_slice:
            rows.append({**s, "axis": "maabench_version", "split": "test"})
    else:
        d = len(OOD_DOMAINS)
        dom_slice = take(COUNTS["domain"] // d, "maabench_domain", "test")
        for s in dom_slice:
            rows.append({**s, "axis": "maabench_joint", "split": "test"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oversample", type=int, default=30)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta_path = os.path.join(a.out, "seeds.meta.json")
    meta = {"seed": a.seed, "counts": COUNTS, "domains": ID_DOMAINS + OOD_DOMAINS}
    if os.path.exists(meta_path) and not a.force:
        if json.load(open(meta_path)) != meta:
            raise SystemExit(f"{meta_path} exists with different params; pass --force to overwrite")

    all_rows = []
    for domain in ID_DOMAINS + OOD_DOMAINS:
        need = _max_needed(domain)
        print(f"[seeds] {domain}: need {need:,} (oversample x{a.oversample})", flush=True)
        seeds = corpora.load_domain_seeds(domain, need, seed=a.seed, oversample=a.oversample)
        if len(seeds) < need:
            print(f"[warn] {domain}: got {len(seeds):,} < {need:,}", flush=True)
        for s in seeds:
            s["seed_id"] = _sid(domain, s["human_text"])
        seeds.sort(key=lambda s: s["seed_id"])
        all_rows += _assign(domain, seeds)

    out = os.path.join(a.out, "seeds.jsonl")
    with open(out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(meta, open(meta_path, "w"), indent=2)
    from collections import Counter
    c = Counter((r["axis"], r["split"], r["domain"]) for r in all_rows)
    print(f"\n[seeds] wrote {len(all_rows):,} rows -> {out}")
    for k in sorted(c):
        print(f"    {k[0]:15s} {k[1]:6s} {k[2]:14s} {c[k]:>6,}")


if __name__ == "__main__":
    main()
