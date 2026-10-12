"""Unified dataset classes for ATLAS attribution experiments."""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class IDDataset(Dataset):
    """Loads a unified JSONL that has a 'split' field (e.g. maabench_id.jsonl)."""

    def __init__(
        self,
        jsonl_path: str | Path,
        sources: list[str],
        split: str,
        include_human: bool = False,
        domains: list[str] | None = None,
        group_key: str | None = None,
        n_negatives: int = 0,
        neg_seed: int = 42,
    ):
        self.sources = list(sources)
        self.source_to_label = {s: i for i, s in enumerate(self.sources)}
        self.human_label = len(self.sources)

        self.examples: list[dict] = []
        for rec in _load_jsonl(jsonl_path):
            if rec.get("split") != split:
                continue
            src = rec["source"]
            domain = rec.get("domain", "")
            if domains is not None and domain not in domains:
                continue
            group = rec.get(group_key) if group_key else None
            if src == "human":
                if include_human:
                    self.examples.append({
                        "text": rec["text"],
                        "source": "human",
                        "id": rec.get("id", ""),
                        "label": self.human_label,
                        "domain": domain,
                        "group": group,
                    })
            elif src in self.source_to_label:
                self.examples.append({
                    "text": rec["text"],
                    "source": src,
                    "id": rec.get("id", ""),
                    "label": self.source_to_label[src],
                    "domain": domain,
                    "group": group,
                })

        self.group_key = group_key
        self.n_negatives = n_negatives
        self._neg_rng = random.Random(neg_seed)
        self._by_group: dict | None = None
        self._build_group_index(verbose=True)

    def _build_group_index(self, verbose: bool = False):
        """Index examples by prompt group, for the D1/D2 same-prompt column loss."""
        if self.n_negatives <= 0:
            self._by_group = None
            return
        if self.group_key is None:
            raise ValueError("n_negatives > 0 requires group_key (e.g. 'seed_id')")

        self._by_group = collections.defaultdict(list)
        missing = 0
        for i, ex in enumerate(self.examples):
            if ex["group"] is None:
                missing += 1
                continue
            self._by_group[ex["group"]].append(i)
        if missing:
            raise ValueError(f"{missing} rows have no '{self.group_key}' field; cannot form same-prompt negatives")

        starved = [i for i, ex in enumerate(self.examples)
                   if not self._candidates(i, ex)]
        if starved:
            raise ValueError(
                f"{len(starved)} of {len(self.examples)} examples have no same-prompt other-source "
                f"negative (e.g. index {starved[0]}, group={self.examples[starved[0]]['group']!r}). "
                f"Check --domains / the source list.")
        if verbose:
            sizes = collections.Counter(len(v) for v in self._by_group.values())
            print(f"[negatives] group_key={self.group_key} groups={len(self._by_group)} "
                  f"n_negatives={self.n_negatives} rows-per-group={dict(sorted(sizes.items()))}")

    def _candidates(self, idx: int, ex: dict) -> list[int]:
        """Indices of examples in the same group written by a different source."""
        return [j for j in self._by_group[ex["group"]] if self.examples[j]["source"] != ex["source"]]

    def subsample_stratified(self, per_class: int, seed: int = 42, verbose: bool = True) -> "IDDataset":
        """Keep ~per_class examples per label, balanced across domain buckets."""
        if per_class <= 0:
            return self

        rng = random.Random(seed)
        by_label: dict[int, list[dict]] = collections.defaultdict(list)
        for ex in self.examples:
            by_label[ex["label"]].append(ex)

        label_names = {i: s for i, s in enumerate(self.sources)}
        label_names[self.human_label] = "human"

        result: list[dict] = []
        for label in sorted(by_label):
            items = by_label[label]
            by_domain: dict[str, list[dict]] = collections.defaultdict(list)
            for ex in items:
                by_domain[ex["domain"]].append(ex)
            per_domain = max(1, per_class // len(by_domain))
            kept: list[dict] = []
            for domain_items in by_domain.values():
                rng.shuffle(domain_items)
                kept.extend(domain_items[:per_domain])
            if verbose:
                name = label_names.get(label, str(label))
                print(f"  [subsample] label={name:<12s} raw={len(items):>6d} -> kept={len(kept):>5d}")
            result.extend(kept)

        self.examples = result
        self._build_group_index()
        return self

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        if self.n_negatives <= 0:
            return ex
        cands = self._candidates(idx, ex)
        picked = (self._neg_rng.sample(cands, self.n_negatives)
                  if len(cands) >= self.n_negatives
                  else self._neg_rng.choices(cands, k=self.n_negatives))
        return {**ex, "negatives": [self.examples[j]["text"] for j in picked]}


class OODDataset(Dataset):
    """Load a unified JSONL without a 'split' field and split it per source with a fixed seed."""

    def __init__(
        self,
        jsonl_path: str | Path,
        sources: list[str],
        split: str = "test",
        test_ratio: float = 0.2,
        seed: int = 42,
        include_human: bool = False,
    ):
        self.sources = list(sources)
        self.source_to_label = {s: i for i, s in enumerate(self.sources)}
        self.human_label = len(self.sources)

        tracked = list(self.sources) + (["human"] if include_human else [])
        tracked_set = set(tracked)
        by_source: dict[str, list[dict]] = {s: [] for s in tracked}
        for rec in _load_jsonl(jsonl_path):
            src = rec["source"]
            if src in tracked_set:
                by_source[src].append(rec)

        rng = random.Random(seed)
        self.examples: list[dict] = []
        for src in tracked:
            items = by_source[src]
            rng.shuffle(items)
            if split == "all":
                chosen = items
            else:
                n_test = max(1, int(len(items) * test_ratio))
                chosen = items[:n_test] if split == "test" else items[n_test:]
            label = self.human_label if src == "human" else self.source_to_label[src]
            for item in chosen:
                self.examples.append({
                    "text": item["text"], "source": src, "id": item.get("id", ""),
                    "label": label, "domain": item.get("domain", ""),
                })

        counts = {s: sum(1 for e in self.examples if e["source"] == s) for s in tracked}
        print(f"[OODDataset/{split}] " + "  ".join(f"{s}={n}" for s, n in counts.items()))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def row_ids(ds) -> list[str]:
    """Each row's `id` in dataset order ('' when missing)."""
    return [str(e.get("id", "")) for e in ds.examples]


def make_collate(tokenizer, max_len: int = 512):
    """Returns a collate_fn compatible with both IDDataset and OODDataset."""
    def collate(batch: list[dict]):
        texts = [ex["text"] for ex in batch]
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        enc = tokenizer(
            texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt",
        )
        loss_mask = enc["attention_mask"].clone()
        return enc["input_ids"], enc["attention_mask"], loss_mask, labels
    return collate
