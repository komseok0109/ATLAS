# ATLAS

Code for **ATLAS**, a method for attributing machine-generated text to the model family that
wrote it, and for **MAABench**, the benchmark used to evaluate it.

Given a machine-written document, ATLAS predicts which model family produced it (`gpt`,
`claude`, `gemini`, `mistral`, `llama`). It fine-tunes a single causal LM (Qwen2.5-7B with LoRA)
together with one learned control embedding per source. At test time the document is scored
under each control embedding and the most likely source is chosen.

## Repository layout

```
src/atlas/       model, training, scoring and evaluation
src/utils/       datasets, readouts and metrics
src/maabench/    rebuilds the human-written part of MAABench (optional)
src/data_utils/  converters for OpenTuringBench
scripts/         Slurm entry points
```

## Installation

```bash
conda create -y -n atlas python=3.11 && conda activate atlas
pip install -r requirements.txt          # exact versions: env/freeze-atlas.txt
bash scripts/download_models.sh          # Qwen/Qwen2.5-7B
mkdir -p logs
```

Paths are set in `scripts/env.sh` and can be overridden with environment variables:

| variable | default |
|---|---|
| `ATLAS_DATA` | `./data` |
| `ATLAS_RESULTS` | `./results` |
| `ATLAS_PREFIX`, `ATLAS_ENV` | `~/miniforge3`, `atlas` (conda env added to `PATH`) |

Optional credentials (`HF_TOKEN`, `WANDB_API_KEY`) can be filled in at the top of the
credentials block in `scripts/env.sh`, or set in your shell. Without a W&B key, runs log offline.

## Data

### MAABench

| axis | distribution shift | file |
|---|---|---|
| `maabench_id` | none (train / val / test) | `maabench_id.jsonl` |
| `maabench_domain` | unseen domains (patents, StackExchange) | `maabench_domain.jsonl` |
| `maabench_version` | newer versions of each model family | `maabench_version.jsonl` |
| `maabench_joint` | unseen domains and newer versions | `maabench_joint.jsonl` |

```bash
hf download ming9999/maabench --repo-type dataset --local-dir data/maabench
```

### OpenTuringBench

Download OpenTuringBench from its authors, then convert it into `$ATLAS_DATA/openturing/`:

```bash
PYTHONPATH=src python -m data_utils.id.prepare_openturing_id --help
PYTHONPATH=src python -m data_utils.ood.prepare_openturing_ood --help   # also _e1 ... _e4
```

## Training and evaluation

Each benchmark has three stages:

```bash
sbatch scripts/atlas/maabench/inter/train.sh          <run>
sbatch scripts/atlas/maabench/inter/collect_scores.sh <run>
sbatch scripts/atlas/maabench/inter/eval.sh           <run>
```

Use `scripts/atlas/openturing/...` for OpenTuringBench. The `#SBATCH` headers only request
generic resources; pass `--partition`, `--qos` and so on for your cluster. On a single machine
the scripts can be run with `bash`.

- **train** runs multi-GPU DDP with an effective batch size of 128 (`EFF_BATCH`) regardless of
  the number of GPUs. A checkpoint is saved after every epoch and an interrupted run resumes
  from the last one. `SEED` sets the random seed (default 42).
- **collect_scores** scores every axis with the trained model and saves the score matrices
  under `$ATLAS_RESULTS/atlas_results/scores/`.
- **eval** runs on CPU and writes `results.json` for each axis and readout under
  `$ATLAS_RESULTS/atlas_results/{id,ood}_results/`. The reported metric is the `(inter-LLM)`
  accuracy. There are two readouts:
  - `argmax`: the source whose control embedding gives the document the highest mean token
    log-probability.
  - `argmax-prod` (main): for each token, each source scores `exp(-Σ_j relu(s_j - s_k) / T)`,
    which is 1 when it has the highest log-probability and decays with how far it falls behind
    the others. The scores are averaged over tokens and over `T ∈ {1, 2, 4, 8}` (`PROD_TEMPS`),
    and the highest-scoring source is chosen.

Set `AXES` to evaluate a subset, e.g. `AXES="maabench_id maabench_version" sbatch ...`.

### Multiple seeds

```bash
for s in 42 123 456; do
  SEED=$s sbatch scripts/atlas/maabench/inter/train.sh atlas-s$s
done
```

Evaluation always uses the same test rows, so runs with different seeds are directly comparable.

## Including human-written text

ATLAS can also treat human-written text as one more source, with its own control embedding,
and predict among K+1 classes. This needs the human text, which MAABench does not
redistribute. Rebuild it from the original public corpora and merge it into the downloaded
files:

```bash
sbatch scripts/maabench/build_seeds.sh        # CPU only, several hours
PYTHONPATH=src python -m maabench.add_human   # adds the human rows to data/maabench/*.jsonl
```

Each row's `seed_id` contains a hash of its human text, so `add_human` checks every rebuilt text
against the release and stops if any is missing or different.

Then run the same three stages from `scripts/atlas/maabench/human/` (or
`scripts/atlas/openturing/human/`). In this setting `results.json` reports accuracy over all
rows, human included (`(all)`), alongside the machine-only `(inter-LLM)` accuracy. Use a
different run name from your main runs.

## Citation

```bibtex
@article{ko2026atlas,
  title   = {Who's Behind the Text? Robust Model Authorship Attribution},
  author  = {Minseok Ko and Hyunseok Lee and Woomin Song and Beomjun Kim and Jinwoo Shin and Kunwoong Kim},
  journal = {arXiv preprint arXiv:2610.19583},
  year    = {2026}
}
```
