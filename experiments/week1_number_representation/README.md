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
| `run_causal_validation.py` | causal sufficiency [engels2024 §5] | subspace-patch ÷ full-layer-patch logit-diff ratio near 1.0 |

**The two `*_components*` scripts answer different questions.** `run_fourier_components.py`
(logit lens, answer-token site) asks *which residue class a component promotes* — DFT over
the candidate-**answer** axis. `run_fourier_components_raw.py` (raw activations, input-number
sweep like `run_fourier.py`) asks *whether the component's output representation is sparse in
frequency over the **input** number* — the activation-space object closest to SAE feature
tracking. Same side-by-side MLP|attn layout; different axis.

```bash
python experiments/week1_number_representation/run_helix_fit.py        --model gptj                 # --hi default 99; --context {bare,addition}
python experiments/week1_number_representation/run_fourier.py          --model gptj                 # embeddings (activation space)
python experiments/week1_number_representation/run_fourier.py          --model gptj --layer 16       # resid_post; --context {bare,addition}
python experiments/week1_number_representation/run_fourier_components.py     --model gptj --layer 16      # MLP/attn output LOGITS, side by side; --context {bare,addition}
python experiments/week1_number_representation/run_fourier_components.py     --model gptj --layer 33 --context addition  # canonical Fig 2/3 (answer-token site)
python experiments/week1_number_representation/run_fourier_components.py     --model gptj --summary --context addition   # Fig 3 across-layers heatmap (MLP|attn)
python experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --layer 16      # MLP/attn output ACTIVATIONS, side by side; --context {bare,addition}
python experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --summary       # Fig 3 across-layers heatmap (activation space)
python experiments/week1_number_representation/run_fourier.py               --model gptj --summary       # resid_post across-layers heatmap (single panel)
python experiments/week1_number_representation/run_causal_validation.py --model gptj
# then repeat with --model llama3-8b (confirm build_layers in config.py from the helix sweep)
```

**Grid note (both Fourier scripts):** with `--lo 0 --hi 360` the number axis has **361**
points (prime), so the predicted periods 2/2.5/5/10 do not land on DFT bins and peaks
leak (dominant periods read ~10.03 instead of 10.0). For exact bin alignment use **360
sample points** — `--lo 1 --hi 360` or `--lo 0 --hi 359`. Default left at 360.

## Outputs (where each run lands)
Grouped by **model** (`GPT-J` / `Llama-3-8B`), then one folder per script, a sub-folder
per context; file names carry only the layer/site.

```
results/week1_number_representation/<model>/
├── run_helix_fit/{bare,addition}/             summary.json, helix_r2_by_layer.png
├── run_fourier/{bare,addition}/               embedding.png|json, resid_post.L<n>.png|json
│                                              summary_resid_post.png|json   (--summary)
├── run_fourier_components/{bare,addition}/     L<n>.png|json   (logit-lens, MLP|attn panels)
│                                              summary_layers.png|json       (--summary)
├── run_fourier_components_raw/{bare,addition}/ L<n>.png|json   (raw activations)
│                                              summary_layers.png|json       (--summary)
└── run_causal_validation/addition/            causal_validation.json
```
`<model>` is `GPT-J` for `--model gptj`, `Llama-3-8B` for `--model llama3-8b`
(`config.model_dir_name`).

Each folder also has a `run_meta.json` (date, git sha, seed, model, exact command).
Re-running a `(script, context)` overwrites in place — provenance lives in
`run_meta.json`, not the folder name.

## Across-layers summary (`--summary`, [zhou2024] Fig 3)
The three Fourier scripts also produce the Figure-3 summary: **x = layer index, y =
frequency, colour = component magnitude** across all layers (use `--layers LO HI` to
restrict; the paper used the last 15). The two `*_components*` scripts draw **MLP | attn
side by side**; `run_fourier.py --summary` sweeps `resid_post` (single panel — no
MLP/attn split; embeddings are excluded as a single point). `--context {bare,addition}`
selects the framing, as for the per-layer plots. Colour defaults to
`--power-transform amplitude` = `sqrt(mean power)` = `||C_k||` (per-dim RMS amplitude /
L2 norm of the frequency-k coefficient vector, linear scale); `power` and `log` are also
available. Outlier components are expected at periods ~2, 2.5, 5, 10.

Readability: the colour map defaults to `--cmap inferno_r` (light background, so a
near-zero cell reads light rather than as an aggressive black field), and colour limits
are robust by default (`--vmax-percentile 99.5`) so the dominant low-frequency spike does
not compress the rest of the map into one colour. Set `--vmax-percentile 100` for the
true max, or pass any matplotlib colormap to `--cmap`.

## What each result feeds downstream
- The **build layer** found by `run_helix_fit` → the site where week-2 SAEs are trained.
- The **PCA/helix subspace** → the hypothesized target the week-4 exclusivity test and
  the week-5 stub read from.
- The **low/high-freq split** (`fourier.split_low_high`) → the separable magnitude
  target the stub can replace exactly while leaving the modular part intact.
- The **MLP-low / attn-high split** confirmed by `run_fourier_components.py` → empirical
  backing for training **dual-site SAEs on both MLP and attention outputs** (weeks 2–3).

## Gotchas (expect a debug pass)
- Only **single-token integers** are used; tokenizers split many numbers. Llama-3 and
  GPT-J tokenize numbers differently — the contiguous-range filter handles this but the
  usable range will differ per model.
- `run_causal_validation` assumes the operand and the answer are single tokens; small
  operands / small answers keep this true. Widen ranges only after it runs.
- For Llama-3-8B, `build_layers` in `config.py` is a placeholder `(0,0)` — set it from
  the `run_helix_fit` sweep before running causal validation with the default layer.
