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
│   ├── setup_paperspace.sh    # one-time env + HF cache wiring (../hf_cache, beside repo)
│   └── download_models.py     # pre-cache GPT-J + Llama-3-8B to ../hf_cache
├── src/n2p/
│   ├── config.py              # paths, model registry, device, cache location
│   ├── models.py              # load models via TransformerLens (HookedTransformer)
│   ├── tasks.py               # TASK REGISTRY + the SINGLE framing source: 3 framings
│   │                          #   (symbolic / word / wordproblem) per operation, plus
│   │                          #   prompt build + read-token selection (was prompts.py)
│   ├── number_repr/
│   │   ├── helix.py           # generalized-helix basis + fit + helix-subspace basis [kantamneni2025]
│   │   ├── fourier.py         # DFT / Fourier-feature analysis [zhou2024]
│   │   ├── plotting.py        # layer x frequency summary heatmaps [zhou2024 Fig 3]
│   │   ├── repcli.py          # shared CLI/plot helpers for the 4 representation scripts
│   │   └── causal.py          # subspace patch + average-ablate-rest [engels2024];
│   │                          #   total/direct-effect (TE/DE) path patching [kantamneni2025 Fig 6]
│   └── circuits/
│       └── discovery.py       # Edge Pruning + Tracr ground truth (ACDC NOT used)
├── experiments/
│   ├── week1_number_representation/
│   │   ├── README.md
│   │   ├── run_helix_fit.py
│   │   ├── run_fourier.py                # DFT of resid_post / embeddings (activation space)
│   │   ├── run_fourier_components.py     # DFT of MLP/attn output LOGITS [zhou2024 Fig 2-3]
│   │   ├── run_fourier_components_raw.py # DFT of MLP/attn output ACTIVATIONS (SAE-relevant)
│   │   ├── run_causal_validation.py      # DENOISING helix sufficiency on resid_post [kantamneni2025 §4.4]; helix full/mag/mod + PCA vs full-layer; ALL layers & framings
│   │   ├── run_causal_validation_components.py  # same, at MLP-out / attn-out (--read-token {a,sum}); two plots; shares src/n2p/number_repr/causal_sufficiency.py
│   │   └── run_te_de_probe.py            # per-layer TE/DE write-site localization [kantamneni2025 Fig 6]; full-node vs helix-direction + decomposed logit-diff
│   └── week1_circuit_sanity/
│       ├── README.md
│       └── run_greaterthan.py
├── results/                   # outputs (gitignored except manifests)
│   └── <exp>/<model>/<script>/<operation>/  # human-readable: grouped by MODEL (GPT-J /
│                                   # Llama-3-8B), then one folder per script, sub-folder
│                                   # per OPERATION (addition, multiplication, ...); files
│                                   # carry the layer/site, FRAMING, and read-token.
│                                   # run_meta.json holds date/sha/seed/model/operation/
│                                   # framing/read_token/cmd.
│       # e.g. week1_number_representation/GPT-J/run_fourier/addition/resid_post.L16.symbolic.a.png
│       #      week1_number_representation/GPT-J/run_fourier_components/addition/L16.symbolic.sum.png
│       #      week1_number_representation/GPT-J/run_fourier_components/addition/summary_MLP.sum.png
│       #      week1_number_representation/GPT-J/run_fourier_components/addition/summary_Attn.sum.png
└── logs/
    └── runlog.md              # APPEND ONE LINE PER RUN (mirrors wiki/log.md style)
```

## 4. Protocol conventions (keep these stable)

- **Determinism:** every run takes a `--seed` (default 0).
- **Output layout (human-readable):** outputs go to
  `results/<exp>/<model>/<script>/<operation>/` — **grouped by model** (`GPT-J` /
  `Llama-3-8B`, from `config.model_dir_name`), then **one folder per script, a sub-folder
  per operation** (`addition`, `subtraction`, `multiplication`, ... — the `--operation`,
  whose 3 framings symbolic/word/wordproblem live in `src/n2p/tasks.py`). File names carry
  the layer/site, the **framing**, and the **read-token** (the path doesn't say those):
  e.g. `resid_post.L16.symbolic.a.png`, `L16.symbolic.sum.png`. The across-layers summary
  draws **one panel per framing**: `summary_MLP.<read_token>.png` + `summary_Attn.<read_token>.png`
  for the component scripts, `summary_resid_post.<read_token>.png` for `run_fourier`. Build
  the path via `config.run_dir(exp, seed, model="gptj", label="run_fourier/addition", meta={...})`
  — pass `model=` to get the per-model grouping (omit it for the legacy un-grouped path).
- **Provenance, not immutable dirs.** Re-running a `(script, operation)` **overwrites in
  place**; the exact `date`, `git_sha`, `seed` and command line are recorded in
  `run_meta.json` in that folder (plus any `meta=` fields). This replaces the old
  immutable `<date>-<sha>-s<seed>` run-id dirs — readability over per-run archival. For a
  one-off immutable snapshot, omit `label` (legacy run-id path still works).
- **Logging:** append one line per run to `logs/runlog.md`:
  `## [YYYY-MM-DD] <script> | model=<m> task=<t> seed=<s> | <one-line result> | results/<path>`
- **Models are referenced by registry key** (`gptj`, `llama3-8b`) from `config.py`,
  never by raw HF id in scripts.
- **No GPU assumptions in code paths that don't need one** (fits/plots can run on the
  cached activations CPU-side); heavy forward passes guard on `config.DEVICE`.
- **Activations cache:** large activation tensors go to `results/<exp>/<run_id>/acts/`
  and are gitignored; only summary stats + plots are committed.
- **Activation sweeps skip the unembed:** `causal.cache_number_site_all_layers` (used by
  the helix, Fourier and causal-validation caching paths) stops the forward after the
  deepest cached block (`stop_at_layer`), so the full `[batch, pos, d_vocab]` logits are
  never materialized. All requested layers are still captured in the one sweep (hooks fire
  inside their blocks); logit-lens projections are taken afterward via `W_U[:, number_ids]`.
  This is what lets the all-layer component sweeps fit on large-vocab models
  (Llama-3, `d_vocab=128256`) at `kshot>0` — the unembed was the OOM, not the cache.

## 5. Running on Paperspace Gradient (A6000, 48 GB)

No local GPU — this repo is pulled on Paperspace and run there.

```bash
# one-time, in a fresh Paperspace notebook/terminal:
git clone <your-fork-url> N2P-Experiments && cd N2P-Experiments
bash setup/setup_paperspace.sh          # wires HF cache to ../hf_cache (beside repo), installs deps
python3 setup/download_models.py          # caches GPT-J + Llama-3-8B to ../hf_cache

# week 1:
python3 experiments/week1_number_representation/run_helix_fit.py --model gptj          # --hi default 99; --operation addition (3 framings as panels), --read-token {a,b,sum}
python3 experiments/week1_number_representation/run_fourier.py --model gptj             # embeddings; add --layer N for resid_post, --operation OP --framing F, --read-token *
python3 experiments/week1_number_representation/run_fourier_components.py --model gptj --layer 16 --operation addition --framing symbolic      # MLP/attn-output LOGIT spectra (read sum)
python3 experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --layer 16 --operation addition --framing symbolic  # MLP/attn-output ACTIVATION spectra (read a)
# Surface form is --operation {addition,subtraction,multiplication,mult_const,int_division,
# modular} x --framing {symbolic,word,wordproblem}: symbolic "Compute {a} + {b} =" (context
# BEFORE the operand) / word "{a} plus {b} equals" / wordproblem "I have {a} apples ...".
# --read-token {a,b,sum} picks the position (a/b operands, sum=last token); b is fixed
# (--b_fixed). See src/n2p/tasks.py (FRAMINGS). The bare-number template is dropped.
# across-layers summary (zhou2024 Fig 3): layer x frequency heatmap, ONE PANEL PER FRAMING.
# --summary sweeps all layers (--layers LO HI to restrict; --power-transform
# {amplitude,power,log} default amplitude=sqrt(power)=||C_k||; --cmap, --vmax-percentile).
python3 experiments/week1_number_representation/run_fourier_components.py --model gptj --summary --operation addition --read-token sum  # -> summary_MLP + summary_Attn
python3 experiments/week1_number_representation/run_fourier.py --model gptj --summary --operation addition --read-token a               # -> summary_resid_post (panel per framing)
python3 experiments/week1_number_representation/run_causal_validation.py --model gptj --kshot 4  # DENOISING helix sufficiency on resid_post [kantamneni2025 §4.4]; helix full/mag/mod + PCA-9/27 vs full-layer ceiling; ALL layers & framings; --framing/--layers to restrict
python3 experiments/week1_number_representation/run_causal_validation_components.py --model gptj --kshot 4  # same denoising test at MLP-out / attn-out; --read-token {a,sum}; two plots (MLP, Attn), panel per framing
python3 experiments/week1_number_representation/run_te_de_probe.py --model gptj        # per-layer TE/DE (Fig 6); full-node vs helix-direction(analytic) + Δlogit decomposition; MLP & Attn summaries, panel per framing
python3 experiments/week1_circuit_sanity/run_discovery_sanity.py --target tracr  # needs Edge-Pruning repo
```

**Model preparation:** use the **frozen** model + few-shot prompting; the number
mechanism is present in strongly-pretrained models without fine-tuning `[zhou2024 §4.3]`.
Fine-tune only as a fallback, and if you do, verify the representation exists frozen
too (so fine-tuning *cleaned* rather than *created* it). See the feature-tracking wiki
note, "Model preparation."

**Where the cache lives:** `HF_HOME=../hf_cache` — a sibling of the repo, **never
inside it** (this is a git repo; multi-GB models must not enter the working tree).
Set by both `setup_paperspace.sh` and `download_models.py`, with `config.py`
defaulting to the same path so the loader reuses it. Models land in
`../hf_cache/hub/`. On Paperspace the repo lives at `/notebooks/N2P-Experiments`, so
the cache is `/notebooks/hf_cache` (persistent with the `/notebooks` volume). Export
`HF_HOME` to override. GPT-J (~12 GB fp16) and Llama-3-8B (~16 GB fp16) are expensive
to re-download, so cache once.

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
