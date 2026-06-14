# N2P-Experiments

Experimental codebase for the **Neural-to-Program (N2P)** ICLR paper, **Version A**:
*direction-level Exclusivity + external symbolic stub for OOD robustness*, on arithmetic skills, in real mid-scale LMs (GPT-J, Llama-3-8B).

This repo is the **execution arm** of the research wiki that lives one level up
(`../wiki/`). The *why* lives in the wiki; the *how / what-was-run* lives here.
The governing plan is `../wiki/notes/iclr-plan.md`. Read it first.

> **This README is the source of truth for repo structure and protocol.**
> At the start of any session, read this file, then the README of the specific
> `experiments/weekN_*/` you are working in, then `logs/runlog.md` for what has
> already been run. Update this README whenever the structure or protocol changes.

---

## 0. The paper in one paragraph (so every session shares the goal)

Node/head-level circuits overlap heavily across tasks (~78% head overlap,
`[merullo2024]`), so you cannot safely *edit* a model at node granularity. We claim
that a **number-value feature lives in a low-dimensional subspace** that is far more
**cross-task exclusive** than the heads carrying it (**Contribution 4**), and that this
subspace can be **swapped at inference for an exact external module** — buying
out-of-distribution robustness with no collateral damage to sibling tasks that share
the node (**Contribution 3**). The method that produces both results is **dual-site
SAE feature tracking across tasks** on the *real* model (**Contribution 2**). We do
**not** physically prune weights in Version A — the stub is injected at inference.

## 1. Two objects, do not conflate them

This distinction drives the whole experimental design:

- **Operand representation** — the subspace encoding the *value* of a number
  (the generalized helix / Fourier features). It is **shared across every numeric
  task** that reads a number. → This is the **Exclusivity** target (C4) and the thing
  the stub *reads its arguments from*.
- **Operation circuit** — the machinery mapping operands → result (e.g. the "Clock"
  for addition). Task-specific. → This is what the stub *replaces* with an exact
  computation, and where OOD fragility lives (C3).

Exclusivity is measured on the operand representation; the stub spans *read operands
(from the exclusive subspace) → compute exactly → inject the answer*.

## 2. Pipeline (the decided method — see the two approach-decision notes in the wiki)

```
            ┌─ identification ─┐   ┌──────── feature tracking ────────┐   ┌─ substitution ─┐
 prompts ─▶ │ Edge Pruning     │─▶ │ dual-site SAEs (MLP + attn out)  │─▶ │ external stub  │
            │ + chen2025       │   │ → cluster → number subspace      │   │ + W_inject     │
            │ noising+denoising│   │ → causal validate (engels)       │   │ (inference)    │
            └──────────────────┘   └──────────────────────────────────┘   └────────────────┘
 act node-level, replace along the SAE direction/subspace only (not whole node).
```

- **Identification** = *where* is the circuit. Base: **Edge Pruning** `[bhaskar2024]`
  (recovers Tracr ground truth, scales to 13B). Completeness layer:
  **noising + denoising** `[chen2025]` to catch OR-gate backups. Sanity anchor:
  **Edge Pruning on a Tracr program with known ground truth** (Greater-Than secondary).
  *ACDC is not used* — rejected method (see `verified-failure-modes` item 1).
- **Substitution target** = the OPERATION, by injecting the clean **answer**
  `helix(a+b)` at the **write site**. For heuristic-heavy ops (multiplication, division,
  modular) this **bypasses per-digit heuristics**, but only if the completeness pass
  captured all parallel answer paths (write-site-bypass; see the circuit-identification
  wiki note).
- **Feature tracking** = *what* it computes. SAEs at **MLP outputs and attention
  outputs** of retained sites, on the **real model** (not a CLT surrogate). Cluster
  SAE features → recover the (multi-dimensional) number subspace.
- **Substitution** = swap the subspace for an exact module at inference.

## 3. Repo layout

```
N2P-Experiments/
├── README.md                 ← you are here (protocol source of truth)
├── requirements.txt
├── setup/
│   ├── setup_paperspace.sh    # one-time env + persistent /storage HF cache wiring
│   └── download_models.py     # pre-cache GPT-J + Llama-3-8B to /storage
├── src/n2p/
│   ├── config.py              # paths, model registry, device, cache location
│   ├── models.py              # load models via TransformerLens (HookedTransformer)
│   ├── tasks.py               # TASK REGISTRY: prompts/framings (under discussion)
│   ├── number_repr/
│   │   ├── helix.py           # generalized-helix basis + fit [kantamneni2025]
│   │   ├── fourier.py         # DFT / Fourier-feature analysis [zhou2024]
│   │   └── causal.py          # subspace patch + average-ablate-rest [engels2024]
│   └── circuits/
│       └── discovery.py       # Edge Pruning + Tracr ground truth (ACDC NOT used)
├── experiments/
│   ├── week1_number_representation/
│   │   ├── README.md
│   │   ├── run_helix_fit.py
│   │   ├── run_fourier.py
│   │   └── run_causal_validation.py
│   └── week1_circuit_sanity/
│       ├── README.md
│       └── run_greaterthan.py
├── results/                   # outputs (gitignored except manifests)
└── logs/
    └── runlog.md              # APPEND ONE LINE PER RUN (mirrors wiki/log.md style)
```

## 4. Protocol conventions (keep these stable)

- **Determinism:** every run takes a `--seed` (default 0) and writes its full config
  to `results/<exp>/<run_id>/config.json`. `run_id = <date>-<git_short_sha>-<seed>`.
- **Results are immutable per run.** Never overwrite; new run = new `run_id`.
- **Logging:** append one line per run to `logs/runlog.md`:
  `## [YYYY-MM-DD] <script> | model=<m> task=<t> seed=<s> | <one-line result> | results/<path>`
- **Models are referenced by registry key** (`gptj`, `llama3-8b`) from `config.py`,
  never by raw HF id in scripts.
- **No GPU assumptions in code paths that don't need one** (fits/plots can run on the
  cached activations CPU-side); heavy forward passes guard on `config.DEVICE`.
- **Activations cache:** large activation tensors go to `results/<exp>/<run_id>/acts/`
  and are gitignored; only summary stats + plots are committed.

## 5. Running on Paperspace Gradient (A6000, 48 GB)

No local GPU — this repo is pulled on Paperspace and run there.

```bash
# one-time, in a fresh Paperspace notebook/terminal:
git clone <your-fork-url> N2P-Experiments && cd N2P-Experiments
bash setup/setup_paperspace.sh          # wires HF cache to persistent /storage, installs deps
python setup/download_models.py          # caches GPT-J + Llama-3-8B to /storage (persists!)

# week 1:
python experiments/week1_number_representation/run_helix_fit.py --model gptj
python experiments/week1_number_representation/run_fourier.py --model gptj
python experiments/week1_number_representation/run_causal_validation.py --model gptj
python experiments/week1_circuit_sanity/run_discovery_sanity.py --target tracr  # needs Edge-Pruning repo
```

**Model preparation:** use the **frozen** model + few-shot prompting; the number
mechanism is present in strongly-pretrained models without fine-tuning `[zhou2024 §4.3]`.
Fine-tune only as a fallback, and if you do, verify the representation exists frozen
too (so fine-tuning *cleaned* rather than *created* it). See the feature-tracking wiki
note, "Model preparation."

**Why cache to `/storage`:** on Paperspace Gradient, `/storage` is **persistent across
machine restarts** while the home/notebook dir and the default HF cache are **not**.
GPT-J (~12 GB fp16) and Llama-3-8B (~16 GB fp16) are expensive to re-download every
session; caching them to `/storage/hf_cache` once saves that every time.
`setup_paperspace.sh` sets `HF_HOME=/storage/hf_cache` so all of HF/TransformerLens
reuse it.

**Llama-3-8B is gated.** Accept the license on its Hugging Face page and export a token
(`export HF_TOKEN=...`) before `download_models.py`. GPT-J is open, no token needed.

## 6. Status: Week 1 (current)

Goal: establish **feature-level ground truth** — reproduce the number representation
(helix + Fourier) on our models and causally validate it — and a **discovery sanity
check** (reproduce Greater-Than). These de-risk weeks 2–6.

Go/no-go gate (end of week 1): *does the number representation reproduce on our model?*
If a model fails to show the helix/Fourier structure with causal support, switch model
or task before investing in SAE training.

> **Week-1 code is first-draft and untested on GPU** (authored without a GPU). Expect a
> debug pass on first Paperspace run; treat the scripts as a faithful implementation of
> the cited methods, not a turnkey artifact. File issues against `logs/runlog.md`.
