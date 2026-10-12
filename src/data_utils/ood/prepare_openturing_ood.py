"""Convert OpenTuringBench out-of-distribution (essay) file into the unified OOD JSONL format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_TO_SOURCE = {
    "meta-llama/Llama-3.1-8B-Instruct":    "llama",
    "Qwen/Qwen2.5-7B-Instruct":            "qwen",
    "google/gemma-2-9b-it":                "gemma",
    "Intel/neural-chat-7b-v3-3":           "intel",
    "microsoft/Phi-3.5-mini-instruct":     "phi",
    "mistralai/Mistral-7B-Instruct-v0.3":  "mistral",
    "upstage/SOLAR-10.7B-Instruct-v1.0":   "solar",
}

OOD_FILES = ["essay.json"]


def build(src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for fname in OOD_FILES:
        domain = fname.replace(".json", "")
        path = src_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with path.open() as f:
            rows = json.load(f)

        for i, row in enumerate(rows):
            text = (row.get("content") or "").strip()
            if not text:
                continue
            source = MODEL_TO_SOURCE[row["model"]]
            url = row.get("url", str(i))
            rec_id = f"openturing-ood-{domain}-{source}-{url}"
            rec = {
                "id":     rec_id,
                "text":   text,
                "source": source,
                "domain": domain,
            }
            if "url" in row:
                rec["url"] = row["url"]
            records.append(rec)

    out_path = out_dir / "openturing_ood.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} rows -> {out_path}")
    for src in sorted(set(r["source"] for r in records)):
        n = sum(1 for r in records if r["source"] == src)
        print(f"  {src}: {n}")
    for domain in ["essay"]:
        n = sum(1 for r in records if r["domain"] == domain)
        print(f"  domain={domain}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=Path, default=Path("data/openturing/ood"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/openturing"))
    args = ap.parse_args()
    build(args.src_dir, args.out_dir)
