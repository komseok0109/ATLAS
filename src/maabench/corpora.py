"""Corpus loaders for the six MAABench domains."""
import json, os, re, itertools, random
from pathlib import Path

from .corpora_base import (DEFAULTS as _BASE_DEFAULTS, load_domain_seeds as _base_load,
                               _fit_window, _field_str, _SHUFFLE_BUFFER)
from .config import DOMAINS, TARGET_WORDS

REVISIONS = {
    "EdinburghNLP/xsum":               "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
    "wikimedia/wikipedia":             "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "gfissore/arxiv-abstracts-2021":   "e4c5fbd4dec8e46a5dc869216fe1c94cc585757a",
    "McAuley-Lab/Amazon-Reviews-2023": "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e",
    "HUPD/hupd":                       "f570a84b03663180b6034c1f7f4c15864f94385e",
    "flax-sentence-embeddings/stackexchange_title_body_jsonl":
                                       "a3d99bf21570ed043e19e41af46f3f19bf4e4bb6",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
CPC_TITLES_PATH = os.environ.get(
    "CPC_TITLES", str(_REPO_ROOT / "assets" / "cpc_subclass_official.json"))

SE_SITES = ["english", "scifi", "diy", "ell", "worldbuilding", "workplace", "travel", "judaism",
            "photo", "law", "biology", "music", "philosophy", "cooking", "movies", "space",
            "aviation", "bicycles", "history", "christianity", "politics", "skeptics", "gardening",
            "boardgames"]
SE_REPO = "flax-sentence-embeddings/stackexchange_title_body_jsonl"
_CODE = re.compile(r"```|~~~|(?:^|\n)(?: {4}|\t)\S")
_HTML = re.compile(r"</?[a-z][^>]*>", re.I)
_LATEX = re.compile(r"\$[^$\n]+\$|\\\(|\\begin\{")


def _clean_cpc(t, maxw=14):
    """Turn a USPTO CPC subclass title into a short domain hint: drop bracketed text, lower-case
    all-caps titles and truncate to maxw words."""
    t = re.sub(r"\{[^}]*\}", " ", t)
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.;-")
    a = [c for c in t if c.isalpha()]
    if a and sum(c.isupper() for c in a) / len(a) > 0.7:
        t = t[:1] + t[1:].lower()
    w = t.split()
    if len(w) > maxw:
        t = " ".join(w[:maxw]).rstrip(" ,;-") + "\u2026"
    return t


def _cpc_titles():
    if not os.path.exists(CPC_TITLES_PATH):
        raise FileNotFoundError(
            f"{CPC_TITLES_PATH} missing (it ships with the repo; or set CPC_TITLES)")
    return json.load(open(CPC_TITLES_PATH))


def _patent_rows(seed):
    """Yield patents from the HUPD January 2016 sample, read from its JSON files directly because
    the HUPD loading script no longer runs under current `datasets`."""
    import tarfile, urllib.request, tempfile
    rev = REVISIONS["HUPD/hupd"]
    url = f"https://huggingface.co/datasets/HUPD/hupd/resolve/{rev}/data/sample-jan-2016.tar.gz"
    cache = os.path.expanduser("~/.cache/maabench"); os.makedirs(cache, exist_ok=True)
    tgz = os.path.join(cache, "hupd-sample-jan-2016.tar.gz")
    if not (os.path.exists(tgz) and os.path.getsize(tgz) > 300_000_000):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = tgz + ".part"
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as w:
            while True:
                chunk = r.read(1 << 20)
                if not chunk: break
                w.write(chunk)
        os.replace(tmp, tgz)
    print(f"[patent] tar {os.path.getsize(tgz)/1e6:.0f} MB", flush=True)
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".json"):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                d = json.load(f)
            except Exception:
                continue
            yield d


def _patent_seeds(n, seed, lo, hi, oversample):
    titles = _cpc_titles()
    seeds, seen = [], set()
    for row in _patent_rows(seed):
        body = row.get("abstract")
        title = (row.get("title") or "").strip()
        cl = row.get("cpc_labels")
        code = (cl[0] if isinstance(cl, list) and cl
                else row.get("main_cpc_label") or row.get("main_ipcr_label") or "")
        code = str(code).strip()
        sub = code[:4].upper() if len(code) >= 4 else ""
        label = titles.get(sub)
        if label:
            label = _clean_cpc(label)
        if not (body and title and label):
            continue
        if label.lower().startswith("indexing scheme"):
            continue
        body = _fit_window(body, lo, hi)
        if body is None or body in seen:
            continue
        seen.add(body)
        seeds.append({"domain": "patent",
                      "fields": {"title": title, "classification": label},
                      "human_text": body})
        if len(seeds) >= n:
            break
    return seeds[:n]


def _se_seeds(n, seed, lo, hi, oversample):
    from huggingface_hub import hf_hub_download
    import gzip
    rng = random.Random(seed)
    per_site = max(1, (n * oversample) // len(SE_SITES))
    rows = []
    for site in SE_SITES:
        try:
            p = hf_hub_download(SE_REPO, f"{site}.stackexchange.com.jsonl.gz", repo_type="dataset",
                                revision=REVISIONS[SE_REPO])
        except Exception as e:
            print(f"[se] skip {site}: {type(e).__name__}", flush=True); continue
        got = 0
        with gzip.open(p, "rt") as f:
            for line in f:
                try: d = json.loads(line)
                except Exception: continue
                t = d.get("texts") or []
                if len(t) < 2:
                    continue
                title, body = (t[0] or "").strip(), (t[1] or "")
                tags = d.get("tags") or []
                if not (title and tags):
                    continue
                if _CODE.search(body) or _HTML.search(body) or _LATEX.search(body):
                    continue
                fit = _fit_window(body, lo, hi)
                if fit is None:
                    continue
                rows.append({"domain": "stackexchange",
                             "fields": {"title": title, "tags": tags, "community": site},
                             "human_text": fit})
                got += 1
                if got >= per_site:
                    break
    rng.shuffle(rows)
    return rows[:n]


def load_domain_seeds(domain, n, seed=42, target_words=TARGET_WORDS, oversample=30):
    lo, hi = target_words
    if domain == "patent":
        return _patent_seeds(n, seed, lo, hi, oversample)
    if domain == "stackexchange":
        return _se_seeds(n, seed, lo, hi, oversample)
    spec = dict(_BASE_DEFAULTS[domain])
    if spec.get("dataset") in REVISIONS:
        spec["revision"] = REVISIONS[spec["dataset"]]
    return _base_load(domain, n, seed=seed, spec=spec, target_words=target_words, oversample=oversample)
