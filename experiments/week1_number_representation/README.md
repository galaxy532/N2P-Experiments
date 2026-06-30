# Week 1 — Number-representation reproduction (feature-level ground truth)

**Goal:** confirm the number-value representation exists and is causally real on our
models, so weeks 2–6 (SAE tracking, exclusivity, stub) build on solid ground.
This is go/no-go gate #1.

## Scripts (run in order)

| Script | Reproduces | Pass signal |
|---|---|---|
| `run_helix_fit.py` | generalized helix [kantamneni2025] | helix R² ≥ poly-baseline R² in a layer band (expected ~14–18 GPT-J). **Reconstruction R² is a weak proxy** — the real gate is `run_causal_validation.py` (see `../../../wiki/exp-notes/helix-experiments-week1-results.md`) |
| `run_fourier.py` | Fourier features in **activation space** [zhou2024 §4.1] | sparse power spectrum; a magnitude (low-freq) peak + modular peaks at periods {2,2.5,5,10}; present already in **embeddings**. NB: top-10-by-power is dominated by the low-freq magnitude tail — read the *plot peaks*, not the ranking (see `fourier-experiments-week1-results.md`) |
| `run_fourier_components.py` | per-component **logit** spectra [zhou2024 §3, Figs 2–3] | MLP-output logits dominated by **low-freq** (magnitude/approximation), attention-output logits by **high-freq** periods {2,5,10} (modular/classification) |
| `run_fourier_components_raw.py` | per-component **activation** spectra (SAE-relevant) | MLP-output / attn-output activations sparse in frequency over the **input** number; the object week-2 SAEs ingest |
| `run_causal_validation.py` | causal sufficiency [kantamneni2025 §4.4/Fig 5], **denoising** | helix-full ÷ full-layer restoration ratio near 1.0 **over a layer band**, and helix-full ≥ the PCA-9/27 baseline (the helix *form*, not generic capacity, is causal). Denoising (inject clean helix into corrupt run); all layers + all framings by default. |
| `run_causal_validation_components.py` | same, at MLP-out / attn-out | per-component denoising restoration ÷ whole-component ceiling; which component's helix write is sufficient. `--read-token {a,sum}`. Two plots (MLP, Attn). |

**The two `*_components*` scripts answer different questions.** `run_fourier_components.py`
(logit lens, read at the **sum** token via `--read-token sum`) asks *which residue class a
component promotes* — DFT over the candidate-**answer** axis. `run_fourier_components_raw.py`
(raw activations, input-number sweep like `run_fourier.py`) asks *whether the component's
output representation is sparse in frequency over the **input** number* — the activation-space
object closest to SAE feature tracking.

## Operation, framing & read-token (all four representation scripts)
`--operation` picks the arithmetic operation and `--framing` its surface form; the operand
`a` is swept while `b` is fixed (`--b_fixed`). The three framings (defined once in
`src/n2p/tasks.py`, `FRAMINGS`) are:

| `--framing` | addition example | has b | sum token |
|---|---|---|---|
| `symbolic` | `"Compute {a} + {b} ="` (context-prefixed) | yes | `=` |
| `word` | `"{a} plus {b} equals"` | yes | `equals` |
| `wordproblem` | `"I have {a} apples and then get {b} more, so now I have"` | yes | last word |

`--operation` ∈ {`addition`, `subtraction`, `multiplication`, `mult_const`, `int_division`,
`modular`}; each defines all three framings (`mult_const` has no operand `b`). The old
bare-number template is **dropped** — `symbolic` now carries context *before* `a` (the
"Compute " prefix), which is what the causal-masking point required (see
`fourier-experiments-week1-results.md`); no framing leaves `a` context-free at the first
position except as a side effect of `word` putting `a` first.

`--read-token a` = operand-a token; `b` = operand-b token (rejected on `mult_const`/no-b
framings); `sum` = last token (the only meaningful logit-lens site for
`run_fourier_components`). Defaults: `a` for the activation scripts + helix, `sum` for the
logit-lens script. Per-layer runs take a single `--framing` (default `symbolic`); `--summary`
iterates **all three** framings as panels.

**Memory on large-vocab models.** The all-layer caching sweep
(`causal.cache_number_site_all_layers`) stops the forward after the deepest cached block, so
the full-vocab unembed is never run — this is what keeps the `--summary` component sweeps
(which cache `mlp_out` + `attn_out` for *every* layer) within an A6000's 48 GB on Llama-3
(`d_vocab=128256`) at `--kshot 4`. If a longer-prompt or larger-`kshot` variant still runs
tight, lower `batch_size` in that helper (the remaining cost is the cached activations, not
the logits).

**Model prompt prefix (`--prefix`).** Every prompt is prepended with a model-specific
zero-shot instruction from `config.ModelSpec.prompt_prefix` — GPT-J `"Output ONLY a
number.\n"`, Llama-3 `"The following is a correct math problem. \n"` — because
[kantamneni2025]'s repo found models need different prompts to actually perform (Llama
especially). It is printed at startup (`[prefix] ...`) and recorded in `run_meta.json` +
each `summary`. The prefix is **digit-free**, so operand indexing (first digit-bearing token
= operand `a`) is unchanged. Override with `--prefix "..."`, or `--prefix ""` to ablate.
**NB:** runs recorded before 2026-06-24 have **no** prefix — they are superseded; re-run the
helix/Fourier sweeps under the prefix before comparing.

**Few-shot (`--kshot`, default 0 = zero-shot).** Prepends `k` fixed solved examples (seeded)
before the query. **GPT-J needs few-shot** (e.g. `--kshot 4`) to actually compute the answer
— zero-shot it often just echoes the prompt (probe-confirmed). Few-shot is *required for
validity* at the **read=sum** site and in `run_causal_validation` (no answer → no signal).
For **read=a** (operand) it is *not required* — the operand token encodes its value whether
or not the model answers, and the paper fits the operand helix zero-shot — **but it is not a
no-op**: the operand residual is computed with attention over the shots, so it is
contextualized by them (more so at the deeper/build layers). For a single consistent prompt
regime, run **all** GPT-J work at the same `--kshot`; use zero-shot read=a only if you
deliberately want the pretrained operand encoding in isolation. Operands are located in the
query line (after the last newline), so example digits never confuse the read-token index.
Llama performs zero-shot, so `--kshot 0` is fine for it.

```bash
python experiments/week1_number_representation/run_helix_fit.py        --model gptj                                       # operation=addition, 3 framings as panels
python experiments/week1_number_representation/run_fourier.py          --model gptj                                       # embeddings (activation space)
python experiments/week1_number_representation/run_fourier.py          --model gptj --layer 16 --operation addition --framing symbolic --read-token a
python experiments/week1_number_representation/run_fourier_components.py     --model gptj --layer 16 --operation addition --framing symbolic        # MLP/attn LOGITS at sum token
python experiments/week1_number_representation/run_fourier_components.py     --model gptj --summary --operation addition --read-token sum            # Fig 3 heatmaps -> summary_MLP + summary_Attn
python experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --layer 16 --operation addition --framing symbolic --read-token a   # MLP/attn ACTIVATIONS
python experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --summary --operation addition --read-token a               # Fig 3 heatmaps (activation space)
python experiments/week1_number_representation/run_fourier.py               --model gptj --summary --operation addition --read-token a               # resid_post heatmap (panel per framing)
python experiments/week1_number_representation/run_causal_validation.py --model gptj --kshot 4                                        # resid_post; denoising; all layers + framings
python experiments/week1_number_representation/run_causal_validation.py --model gptj --operation modular --framing symbolic          # any operation/framing (answer from task.fn)
python experiments/week1_number_representation/run_causal_validation.py --model gptj --operation multiplication --layers 10 14 18 22  # restrict the layer-of-intervention sweep
python experiments/week1_number_representation/run_causal_validation_components.py --model gptj --kshot 4                             # MLP-out + attn-out; --read-token {a,sum}; two plots
# then repeat with --model llama3-8b (confirm build_layers in config.py from the helix sweep)
```

**Grid note (both Fourier scripts):** with `--lo 0 --hi 360` the number axis has **361**
points (prime), so the predicted periods 2/2.5/5/10 do not land on DFT bins and peaks
leak (dominant periods read ~10.03 instead of 10.0). For exact bin alignment use **360
sample points** — `--lo 1 --hi 360` or `--lo 0 --hi 359`. Default left at 360.

## Outputs (where each run lands)
Grouped by **model** (`GPT-J` / `Llama-3-8B`), then one folder per script, a sub-folder
per **operation**; file names carry the layer/site, the **framing**, and the **read-token**.

```
results/week1_number_representation/<model>/
├── run_helix_fit/<operation>/        summary.<rt>.json, helix_r2_by_layer.<rt>.png  (panel per framing)
├── run_fourier/<operation>/          embedding.<framing>.<rt>.png|json, resid_post.L<n>.<framing>.<rt>.png|json
│                                     summary_resid_post.<rt>.png|json   (--summary; panel per framing)
├── run_fourier_components/<operation>/     L<n>.<framing>.<rt>.png|json  (logit-lens, MLP|attn panels)
│                                     summary_MLP.<rt>.png + summary_Attn.<rt>.png   (--summary; panel per framing)
├── run_fourier_components_raw/<operation>/ L<n>.<framing>.<rt>.png|json  (raw activations)
│                                     summary_MLP.<rt>.png + summary_Attn.<rt>.png   (--summary; panel per framing)
├── run_causal_validation/<operation>/ causal_validation.<framing>.json (per framing),
│                                     causal_by_layer.summary.png  (denoising, all-layer sweep, panel-row per framing)
├── run_causal_validation_components/<operation>/ causal_components.<framing>.json (per framing; mlp+attn),
│                                     causal_components_MLP.summary.png + causal_components_Attn.summary.png
└── run_te_de_probe/<operation>/      te_de.<framing>.json (per framing),
                                      te_de_summary_MLP.png + te_de_summary_Attn.png  (per-layer TE/DE, panel per framing, 8 curves)
```
`<model>` is `GPT-J` for `--model gptj`, `Llama-3-8B` for `--model llama3-8b`; `<rt>` is the
`--read-token`. Each folder has a `run_meta.json` (date, git sha, seed, model, operation,
framing, read_token, exact command). Re-running overwrites in place — provenance lives in
`run_meta.json`, not the folder name.

## Across-layers summary (`--summary`, [zhou2024] Fig 3)
The three Fourier scripts produce the Figure-3 summary: **x = layer index, y = frequency,
colour = component magnitude**, with **one panel per framing** (`symbolic` / `word` /
`wordproblem`). The two `*_components*` scripts write **two files** — `summary_MLP` and
`summary_Attn` (MLP and attention in separate figures, framings as panels);
`run_fourier.py --summary` writes a single `summary_resid_post` (resid_post has no MLP/attn
split). `--layers LO HI` restricts the band; colour defaults to `--power-transform amplitude`
= `sqrt(mean power)` = `||C_k||` (`power`/`log` also available), robust limits
`--vmax-percentile 99.5`, `--cmap inferno_r`. Outlier components expected at periods ~2, 2.5, 5, 10.

## Causal validation outputs (`run_causal_validation.py`)
**Direction: DENOISING** ([kantamneni2025] §4.4 / Fig-5; switched 2026-06-27 from the
[engels2024] Eq 5-6 noising+average-ablate-rest the old version used — see the approach-decision
note. Results from the old noising version are **stale; re-run**.) The base run is the
**corrupt** prompt `a'+b`; we **inject the clean operand `helix(a)`** into the subspace at the
operand-`a` token, let the network recompute, and read how much the clean answer is restored.
No average-ablation is needed (injecting clean into a corrupt run leaves nothing clean in the
orthogonal complement to leak).

Sweeps **layers of intervention** (`--layers`, default = **ALL layers `0..last`**, no
`build_layers` prior) across **all three framings by default** (`--framing` restricts to one).
Per layer, per method: mean restoration logit-diff `logit[ans(a)] − logit[ans(a')]` and its
ratio to the whole-layer injection (the ceiling); `noop` (unpatched corrupt) is the floor.
Output is **`causal_by_layer.summary.png`** — one panel-row per framing × 2 cols (absolute |
ratio-over-full-layer vs layer) — plus one `causal_validation.<framing>.json` per framing.
(Single-framing run → `causal_by_layer.<framing>.png`.)

- `full_layer` — inject the **whole** clean `resid_post[L]` (per-layer sufficiency ceiling).
- `helix_full` — inject the whole fitted clean helix (linear + all periods + DC). Tracking `full_layer` over a band ⇒ the helix subspace is causally sufficient there.
- `helix_magnitude` / `helix_modular` — the **separable** split: magnitude = linear + DC + `T=100` (low-freq); modular = `T=2,5,10` (high-freq). Which part carries the effect (the stub replaces magnitude exactly, [zhou2024 Table 1]).
- `pca9` / `pca27` — **PCA-injection baseline** ([kantamneni2025] Fig 5): inject the clean activation's top-k PCA projection, no helix assumed. `9` capacity-matched, `27` over-capacity. `helix_full` matching/beating these with far fewer *effective* dims (the helix populates only ~7–8 dims over integers, since `sin(2πa/2)≡0`) is the evidence that the periodic **form**, not raw subspace capacity, is causal.

## Causal validation at the component outputs (`run_causal_validation_components.py`)
The component-level analogue (MLP-out + attn-out, the causal twin of `run_fourier_components_raw`).
Same **denoising** method and full-parity methods as above, but injected at `hook_mlp_out` /
`hook_attn_out` instead of `resid_post`, asking per layer whether the helix *as written by that
one component* is causally sufficient. `--read-token {a,sum}` (default `a`): `a` = operand token
`helix(a)`, `sum` = answer token `helix(a+b)`. **Caveat:** at a single component the operand `a'`
still lives in every other component / the embedding / the non-helix dims, so this is a *partial*
(total-effect) restoration read **relative to the whole-component ceiling** — not absolute
operand sufficiency (that is the resid script). Per-component helix-fit R² is expected to be
lower than on `resid_post`. Output: **two PNGs** `causal_components_MLP.summary.png` /
`causal_components_Attn.summary.png` (panel per framing × abs|ratio) + `causal_components.<framing>.json`
per framing (both sites). The shared sweep for both scripts lives in `src/n2p/number_repr/causal_sufficiency.py`.

**Operation/framing semantics.** The operand-`a` representation (and the fitted helix
basis) is *identical* across operations sharing a pre-`{a}` prefix (causal masking), but the
patch measurement runs the **operation-specific downstream** to the answer — so per-operation
causal sufficiency is a genuinely distinct test (do mult/div/modular read the operand
*helically*, or via heuristics outside the subspace? — the stress-set question). The helix is
fit once and the basis reused across operations; only patch-and-measure repeats. **Cost** is
`layers × n_test × methods` patched forwards (the setup line prints the count) — restrict
`--layers`/`--n_test` for a first pass.

## TE/DE write-site probe (`run_te_de_probe.py`, optional validation)
Not part of the core reproduction sequence — a role-labeling pass that reproduces
[kantamneni2025] Fig 6 on our model and tests where the answer is *written* (the stub
injection site). Per last-token MLP-out / attn-out **at every swept layer** it reports **TE**
(activation patch, downstream recomputes) and **DE** (path patch, downstream frozen to
corrupt), averaged over `(a, a', b)` triples, for **two sender variants**: **full-node** (swap
the whole component output) and **helix-direction** (swap only the helix subspace, writing the
*fitted* `helix(a+b)` — the analytic target, orthogonal complement kept). full-node DE
localizes the write band; helix-direction DE tracking it ⇒ the write effect lives in the helix
direction (N2P "replace along a direction"), now resolved **per layer** rather than at a single
prior-chosen build layer (no `build_layers` prior — 2026-06-27, Option 2).

It also reports an **exploratory decomposed metric** for the TE: `Δlogit[a+b]` (raised the
clean answer) and `−Δlogit[a'+b]` (suppressed the corrupt answer), both baselined against the
unpatched-corrupt run; they sum to `LD_patched − LD_corrupt`, splitting the standard logit-diff
into its "raise clean" vs "suppress corrupt" halves. (b) vs (a) note: we write the *fitted*
helix (analytic), not the empirical projection of the real activation, so the test is of the
specific helix, not merely its learned subspace. This is the probe
`approach-decision-circuit-identification.md` asks for before adopting TE/DE as a validation
layer; it does **not** use Edge Pruning (that is `run_discovery_sanity.py`).

```bash
python experiments/week1_number_representation/run_te_de_probe.py --model gptj                 # all layers, all framings (expensive)
python experiments/week1_number_representation/run_te_de_probe.py --model gptj --layers 8 27    # focus the build/read band
python experiments/week1_number_representation/run_te_de_probe.py --model gptj --framing symbolic  # one framing only
```
Output: `run_te_de_probe/<operation>/te_de.<framing>.json` (one per framing) + two PNGs
**`te_de_summary_MLP.png`** / **`te_de_summary_Attn.png`**, each with one panel per framing and
8 curves (full/helix × TE/DE in LD, plus full/helix × TE in `Δ[a+b]` and `−Δ[a'+b]`). Cost
note: full+helix × TE+DE × {MLP,attn} ≈ 8 patched forwards per layer per triple, over all
layers and framings — restrict
`--layers` and keep `--n_test` modest.

## What each result feeds downstream
- The **build layer** found by `run_helix_fit` → the site where week-2 SAEs are trained.
- The **helix subspace** (`helix.helix_subspace_basis`) → the hypothesized target the week-4 exclusivity test and the week-5 stub read from.
- The **magnitude / modular split** confirmed causally here → the separable magnitude target the stub can replace exactly while leaving the modular part intact.
- The **MLP-low / attn-high split** confirmed by `run_fourier_components.py` → empirical backing for training **dual-site SAEs on both MLP and attention outputs** (weeks 2–3).

## Gotchas (expect a debug pass)
- Only **single-token integers** are used; tokenizers split many numbers. The operand grid
  is built by `tasks.single_token_number_grid`, which validates each candidate against the
  **real prompt** (the operand-`a` token must be exactly one token) and is therefore
  tokenizer-agnostic. This replaced the old `models.number_token_ids`, which probed only the
  space-prefixed form `" {n}"` and returned an **empty grid on Llama-3** (Llama splits the
  leading space into its own token and groups digits in runs of 1–3, so `" 10" → [" ","10"]`
  — fixed 2026-06-23). The usable range still differs per model; `contiguous_prefix` keeps
  the even grid the DFT/helix expect.
- `run_causal_validation` requires single-token **operands** (the patch site is one token),
  but answers are scored on their **first content token**, read as the **BARE** answer token
  (`models.first_answer_token_id(..., space=False)`). The prompt is zero-shot ending in `=`
  with no trailing space; the probe (`probe_answer_token_space.py`, 2026-06-24) confirmed
  GPT-J emits the bare `99` there (not `␣99`) and Llama is identical (its space is always a
  separate token), so bare is correct for both — matching [kantamneni2025]'s bare
  `tokenizer(f'{answer}')`. Multi-token answers (e.g. multiplication) are admissible as a
  leading-digit/magnitude test (`frac_single_token_answers` flags exactness). Triples whose
  two answers share a first token are skipped (logit-diff ~0 by construction). NB GPT-J needs
  **few-shot** to do the task reliably (zero-shot it often echoes the prompt); the answer/sum
  sites and causal runs assume the model actually computes — operand (read=a) runs do not.
- For Llama-3-8B, `build_layers` in `config.py` is an unverified placeholder. As of
  2026-06-27 **no script's default depends on it** — `run_causal_validation` and
  `run_te_de_probe` both sweep all layers, and `run_helix_fit` only *records* it as
  `expected_build_layers` metadata. Still worth setting from the `run_helix_fit` sweep so that
  annotation is meaningful, but it no longer silently drives any intervention band.
- In word/wordproblem framings the operand `a` may not be at the first content position;
  read-token indexing locates operands as the 1st/2nd **digit-bearing** tokens, and literal
  constants (the `3` in `mult_const`, `7` in `modular`) are placed *after* the operands (and
  spelled out in word framings) so they are never mistaken for `b`.
