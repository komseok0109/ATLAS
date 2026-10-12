#!/bin/bash
#SBATCH --job-name=maa-seeds-smoke
#SBATCH --output=logs/maa-seeds-smoke-%j.out
#SBATCH --error=logs/maa-seeds-smoke-%j.err
#SBATCH --nodes=1 --cpus-per-task=8 --mem=64G --time=3:00:00
# Quick check that every corpus loader works, on a tiny seed pool.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
export TO_SMOKE=1
python - <<'PY'
import sys; sys.path.insert(0,"src")
import maabench.config as C
C.COUNTS=dict(id_train=40, id_val=8, id_test=8, domain=16, version=12, joint=12)
C.AXES={
 "maabench_id": dict(domains=C.ID_DOMAINS, roles=["train"],
   splits={"train":C.COUNTS["id_train"],"val":C.COUNTS["id_val"],"test":C.COUNTS["id_test"]}),
 "maabench_domain": dict(domains=C.OOD_DOMAINS, roles=["train"], total=C.COUNTS["domain"]),
 "maabench_version": dict(domains=C.ID_DOMAINS, roles=C.TEST_ROLES, total=C.COUNTS["version"]),
 "maabench_joint": dict(domains=C.OOD_DOMAINS, roles=C.TEST_ROLES, total=C.COUNTS["joint"]),
}
import importlib
import maabench.make_seeds as M
importlib.reload(M)
sys.argv=["make_seeds","--out","data/maabench/seeds_smoke","--oversample","50","--force"]
M.main()
PY
echo "=== sample seeds (3 per domain) ==="
python - <<'PY'
import json,collections
rows=[json.loads(l) for l in open("data/maabench/seeds_smoke/seeds.jsonl")]
seen=set()
for r in rows:
    if r["domain"] not in seen:
        seen.add(r["domain"])
        print(f"\n[{r['domain']}] axis={r['axis']} split={r['split']}")
        print(f"  fields={r['fields']}")
        print(f"  body({len(r['human_text'].split())}w): {r['human_text'][:120]}...")
PY
echo "=== Done ==="
