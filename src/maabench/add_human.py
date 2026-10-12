"""Restore the human rows of the released benchmark from a locally rebuilt seed pool.

Each row's seed_id is "<domain>-<sha1(human_text)[:12]>", so every rebuilt human text is checked
against the release before it is written next to the machine rows for the same seed."""
import argparse
import hashlib
import json
import os

AXES = ["maabench_id", "maabench_domain", "maabench_version", "maabench_joint"]


def _sid(domain, text):
    return f"{domain}-{hashlib.sha1(text.encode()).hexdigest()[:12]}"


def load_seeds(path):
    seeds = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            seeds.setdefault((r["axis"], r["seed_id"]), r)
    return seeds


def human_row(sd, sid):
    return {"id": f"{sid}:human", "text": sd["human_text"], "source": "human",
            "domain": sd["domain"], "split": sd.get("split", "test"),
            "seed_id": sid, "role": "human", "model": "human"}


def restore(axis, src, dst, seeds):
    rows = [json.loads(line) for line in open(src)]
    if any(r["source"] == "human" for r in rows):
        raise ValueError(f"{src} already contains human rows")
    out, missing, bad = [], set(), set()
    for i, r in enumerate(rows):
        out.append(r)
        sid = r["seed_id"]
        last = i + 1 == len(rows) or rows[i + 1]["seed_id"] != sid
        if not last:
            continue
        sd = seeds.get((axis, sid))
        if sd is None:
            missing.add(sid)
        elif _sid(sd["domain"], sd["human_text"]) != sid:
            bad.add(sid)
        else:
            out.append(human_row(sd, sid))
    n_seed = len({r["seed_id"] for r in rows})
    print(f"[add_human] {axis}: {n_seed - len(missing) - len(bad):,}/{n_seed:,} human rows restored"
          f" ({len(missing):,} missing from the seed pool, {len(bad):,} hash mismatches)")
    if missing or bad:
        raise SystemExit(f"[add_human] {axis}: seed pool does not reproduce the release; "
                         f"first missing={sorted(missing)[:3]} mismatched={sorted(bad)[:3]}")
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release_dir", default="data/maabench")
    ap.add_argument("--seeds", default="data/maabench/seeds/seeds.jsonl")
    ap.add_argument("--out_dir", default="data/maabench")
    a = ap.parse_args()
    seeds = load_seeds(a.seeds)
    os.makedirs(a.out_dir, exist_ok=True)
    for axis in AXES:
        src = os.path.join(a.release_dir, f"{axis}.jsonl")
        if not os.path.exists(src):
            print(f"[add_human] {axis}: no {src}, skipping")
            continue
        restore(axis, src, os.path.join(a.out_dir, f"{axis}.jsonl"), seeds)


if __name__ == "__main__":
    main()
