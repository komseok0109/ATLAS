"""Convert OpenTuringBench in-domain variation E4 (Human-Machine Mixing) into the
unified OOD-style JSONL format (test only, no split field)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_TO_SOURCE = {
    "human":                                "human",
    "meta-llama/Llama-3.1-8B-Instruct":     "llama",
    "Qwen/Qwen2.5-7B-Instruct":             "qwen",
    "google/gemma-2-9b-it":                 "gemma",
    "Intel/neural-chat-7b-v3-3":            "intel",
    "microsoft/Phi-3.5-mini-instruct":      "phi",
    "mistralai/Mistral-7B-Instruct-v0.3":   "mistral",
    "upstage/SOLAR-10.7B-Instruct-v1.0":    "solar",
}

SRC_FILE = "human_mix.json"
DOMAIN = "news"
TAG = "e4"


def build(src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    path = src_dir / SRC_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open() as f:
        rows = json.load(f)

    records: list[dict] = []
    for i, row in enumerate(rows):
        text = (row.get("content") or "").strip()
        if not text:
            continue
        source = MODEL_TO_SOURCE[row["model"]]
        url = row.get("url", str(i))
        records.append({
            "id":     f"openturing-{TAG}-{source}-{url}",
            "text":   text,
            "source": source,
            "domain": DOMAIN,
            "url":    url,
        })

    out_path = out_dir / f"openturing_{TAG}.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} rows -> {out_path}")
    for src in sorted(set(r["source"] for r in records)):
        n = sum(1 for r in records if r["source"] == src)
        print(f"  {src}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=Path, default=Path("data/openturing/in_domain_var"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/openturing"))
    args = ap.parse_args()
    build(args.src_dir, args.out_dir)
