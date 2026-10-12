"""Convert OpenTuringBench in-domain filtered files into the unified JSONL format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_TO_SOURCE = {
    "human":                                "human",
    "meta-llama/Llama-3.1-8B-Instruct":    "llama",
    "Qwen/Qwen2.5-7B-Instruct":            "qwen",
    "google/gemma-2-9b-it":                "gemma",
    "Intel/neural-chat-7b-v3-3":           "intel",
    "microsoft/Phi-3.5-mini-instruct":     "phi",
    "mistralai/Mistral-7B-Instruct-v0.3":  "mistral",
    "upstage/SOLAR-10.7B-Instruct-v1.0":   "solar",
}

SPLITS = ["train", "val", "test"]


def build(src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for split in SPLITS:
        path = src_dir / f"filtered_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with path.open() as f:
            rows = json.load(f)

        for row in rows:
            text = (row.get("content") or "").strip()
            if not text:
                continue
            source = MODEL_TO_SOURCE[row["model"]]
            rec_id = f"{row['url']}-{split}-{source}"
            records.append({
                "id":     rec_id,
                "text":   text,
                "source": source,
                "domain": "news",
                "split":  split,
                "url":    row["url"],
            })

    out_path = out_dir / "openturing_id.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} rows -> {out_path}")
    for src in ["human"] + [s for s in MODEL_TO_SOURCE.values() if s != "human"]:
        n = sum(1 for r in records if r["source"] == src)
        print(f"  {src}: {n}")
    for split in SPLITS:
        n = sum(1 for r in records if r["split"] == split)
        print(f"  split={split}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=Path, default=Path("data/openturing/in_domain"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/openturing"))
    args = ap.parse_args()
    build(args.src_dir, args.out_dir)
