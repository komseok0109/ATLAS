"""Per-domain corpus loaders for the four shared domains (news, wiki, academic, comment)."""
import html
import itertools
import json
import os
import random
import re

from .config import DOMAINS, TARGET_WORDS

_SHUFFLE_BUFFER = 50_000
_META_SCAN_CAP = 5_000_000

DEFAULTS = {
    "news":     {"dataset": "EdinburghNLP/xsum", "config": None, "split": "train", "streaming": False,
                 "body": "document", "fields": {"headline": "summary"}},
    "academic": {"dataset": "gfissore/arxiv-abstracts-2021", "config": None, "split": "train", "streaming": True,
                 "body": "abstract", "fields": {"title": "title", "field": "categories"}},
    "comment":  {"loader": "amazon_join",
                 "base_url": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main",
                 "categories": ["Video_Games", "Office_Products", "Toys_and_Games"],
                 "cutoff_year": 2021},
    "wiki":     {"dataset": "wikimedia/wikipedia", "config": "20231101.en", "split": "train", "streaming": True,
                 "body": "text", "fields": {"title": "title"}, "clean_wiki": True},
}

_HTML_TAG = re.compile(r"<[^>]+>")
_REF_TAG = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/?>", re.I | re.S)
_REFS = re.compile(r"\[\d+\]")
_WIKI_PARENS = re.compile(r"\s*\([^()]*\)")
_SENT_END = re.compile(r"""[.!?]["')\]]?\s""")


def _year_of(val):
    """Best-effort 4-digit year from a date/timestamp string or unix seconds/ms; None if unknown."""
    if val in (None, ""):
        return None
    s = str(val).strip()
    if s.isdigit():
        n = int(s)
        if n >= 100_000_000:
            if n > 100_000_000_000:
                n //= 1000
            return 1970 + n // 31_556_952
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None


def _clean_review(text):
    """Remove HTML markup and entities from an Amazon review and collapse whitespace."""
    t = _HTML_TAG.sub(" ", html.unescape(str(text)))
    return re.sub(r"\s+", " ", t).strip()


def _clean_wiki(text):
    """Remove citation tags, [n] markers and parenthetical asides from Wikipedia text."""
    t = _REF_TAG.sub("", str(text))
    t = _HTML_TAG.sub(" ", t)
    t = _REFS.sub("", t)
    t = _WIKI_PARENS.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _fit_window(text, lo, hi):
    """Fit a text to the [lo, hi] word window. Texts under lo words are dropped; texts over hi words
    are cut at the last sentence boundary within hi and kept only if at least lo words remain."""
    words = str(text).split()
    wc = len(words)
    if wc < lo:
        return None
    if wc <= hi:
        return " ".join(words)
    head = " ".join(words[:hi])
    ends = list(_SENT_END.finditer(head))
    if ends:
        cand = head[:ends[-1].end()].strip()
        if len(cand.split()) >= lo:
            return cand
    return head


def _field_str(v):
    """Render a corpus field as a single-line string, joining list values with spaces."""
    if isinstance(v, (list, tuple)):
        v = " ".join(str(x) for x in v)
    return re.sub(r"\s+", " ", str(v)).strip()


def _stream_jsonl(url, limit=None):
    """Stream a JSONL file over HTTP, yielding one dict per line."""
    import urllib.request
    headers = {"User-Agent": "atlas-benchmark"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    stream = resp
    if resp.headers.get("Content-Encoding") == "gzip" or url.endswith(".gz"):
        import gzip
        stream = gzip.GzipFile(fileobj=resp)
    try:
        for i, raw in enumerate(stream):
            if limit is not None and i >= limit:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        resp.close()


def _amazon_join_seeds(spec, n, seed, lo, hi, oversample):
    """Build comment-domain seeds from Amazon Reviews 2023 by joining reviews with product metadata
    on parent_asin. Reviews without a metadata match are dropped."""
    cutoff = spec.get("cutoff_year")
    cats = spec["categories"]
    base = spec["base_url"].rstrip("/")
    per_cat = -(-n // len(cats))
    rng = random.Random(seed)
    all_seeds, seen = [], set()
    for cat in cats:
        cands = []
        for r in _stream_jsonl(f"{base}/raw/review_categories/{cat}.jsonl", limit=per_cat * oversample):
            if cutoff is not None:
                y = _year_of(r.get("timestamp"))
                if y is not None and y > cutoff:
                    continue
            body = _fit_window(_clean_review(r.get("text") or ""), lo, hi)
            pa = r.get("parent_asin")
            if body is None or not pa:
                continue
            cands.append((pa, r.get("rating"), body))
            if len(cands) >= per_cat * 3:
                break
        rng.shuffle(cands)
        needed = {pa for pa, _, _ in cands}

        lookup = {}
        for m in _stream_jsonl(f"{base}/raw/meta_categories/meta_{cat}.jsonl", limit=_META_SCAN_CAP):
            pa = m.get("parent_asin")
            if pa in needed and pa not in lookup and m.get("title"):
                lookup[pa] = (str(m["title"]), str(m.get("main_category") or cat))
                if len(lookup) >= len(needed):
                    break

        got = miss = 0
        for pa, rating, body in cands:
            info = lookup.get(pa)
            if info is None:
                miss += 1
                continue
            if body in seen:
                continue
            seen.add(body)
            all_seeds.append({"domain": "comment",
                              "fields": {"product": _field_str(info[0]), "category": _field_str(info[1]),
                                         "rating": str(rating)},
                              "human_text": body})
            got += 1
            if got >= per_cat:
                break
        tot = got + miss
        rate = (100 * miss // tot) if tot else 0
        warn = "  ⚠️ high join-miss (check asin coverage / raise _META_SCAN_CAP)" if rate >= 20 else ""
        print(f"[comment/{cat}] joined {got} seeds (join-miss {miss}/{tot} = {rate}%){warn}", flush=True)

    return all_seeds[:n]


def load_domain_seeds(domain, n, seed=42, spec=None, target_words=TARGET_WORDS, oversample=30):
    """Return up to n seed dicts {domain, fields{...}, human_text} for a domain."""
    from datasets import load_dataset
    spec = {**DEFAULTS[domain], **(spec or {})}
    oversample = spec.get("oversample", oversample)
    lo, hi = target_words
    if spec.get("loader") == "amazon_join":
        return _amazon_join_seeds(spec, n, seed, lo, hi, oversample)
    fmap = {sf: col for sf, col in spec["fields"].items() if sf in set(DOMAINS[domain]["fields"])}
    date_col, cutoff_year = spec.get("date_col"), spec.get("cutoff_year")

    cap = n * oversample
    if spec.get("data_files"):
        ds = load_dataset(spec["dataset"], data_files=spec["data_files"], split=spec["split"], streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=_SHUFFLE_BUFFER)
        rows = itertools.islice(ds, cap)
    elif spec["streaming"]:
        ds = load_dataset(spec["dataset"], spec.get("config"), split=spec["split"], streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=_SHUFFLE_BUFFER)
        rows = itertools.islice(ds, cap)
    else:
        ds = load_dataset(spec["dataset"], spec.get("config"), split=f"{spec['split']}[:{cap}]")
        rows = ds.shuffle(seed=seed)

    seeds, seen, date_checked = [], set(), False
    for row in rows:
        if date_col:
            if not date_checked:
                if date_col not in row:
                    print(f"[warn] date_col '{date_col}' absent in {spec['dataset']} → cutoff filter "
                          f"INACTIVE; verify the column name (falling back to corpus-level cutoff).", flush=True)
                date_checked = True
            if cutoff_year is not None:
                y = _year_of(row.get(date_col))
                if y is not None and y > cutoff_year:
                    continue
        body = row.get(spec["body"])
        if not body:
            continue
        if spec.get("clean_wiki"):
            body = _clean_wiki(body)
        body = _fit_window(body, lo, hi)
        if body is None:
            continue
        if body in seen:
            continue
        seen.add(body)
        fields = {}
        for sf, col in fmap.items():
            raw = row.get(col)
            if raw in (None, "", []):
                continue
            val = _field_str(raw)
            if val:
                fields[sf] = val
        for sf, val in (spec.get("const_fields") or {}).items():
            if sf in set(DOMAINS[domain]["fields"]):
                fields[sf] = _field_str(val)
        if len(fields) <= len(spec.get("const_fields") or {}):
            continue
        seeds.append({"domain": domain, "fields": fields, "human_text": body})
        if len(seeds) >= n:
            break

    return seeds[:n]
