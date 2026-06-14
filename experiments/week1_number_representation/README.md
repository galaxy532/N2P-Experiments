# Week 1 — Number-representation reproduction (feature-level ground truth)

**Goal:** confirm the number-value representation exists and is causally real on our
models, so weeks 2–6 (SAE tracking, exclusivity, stub) build on solid ground.
This is go/no-go gate #1.

## Scripts (run in order)

| Script | Reproduces | Pass signal |
|---|---|---|
| `run_helix_fit.py` | generalized helix [kantamneni2025] | helix R² ≥ poly-baseline R² in a layer band (expected ~14–18 GPT-J) |
| `run_fourier.py` | Fourier features [zhou2024] | sparse power spectrum; dominant periods near {2,5,10,100} + a magnitude (low-freq) peak; present in **embeddings** |
| `run_causal_validation.py` | causal sufficiency [engels2024 §5] | subspace-patch ÷ full-layer-patch logit-diff ratio near 1.0 |

```bash
python experiments/week1_number_representation/run_helix_fit.py        --model gptj
python experiments/week1_number_representation/run_fourier.py          --model gptj          # embeddings
python experiments/week1_number_representation/run_fourier.py          --model gptj --layer 16
python experiments/week1_number_representation/run_causal_validation.py --model gptj
# then repeat with --model llama3-8b (confirm build_layers in config.py from the helix sweep)
```

## What each result feeds downstream
- The **build layer** found by `run_helix_fit` → the site where week-2 SAEs are trained.
- The **PCA/helix subspace** → the hypothesized target the week-4 exclusivity test and
  the week-5 stub read from.
- The **low/high-freq split** (`fourier.split_low_high`) → the separable magnitude
  target the stub can replace exactly while leaving the modular part intact.

## Gotchas (expect a debug pass)
- Only **single-token integers** are used; tokenizers split many numbers. Llama-3 and
  GPT-J tokenize numbers differently — the contiguous-range filter handles this but the
  usable range will differ per model.
- `run_causal_validation` assumes the operand and the answer are single tokens; small
  operands / small answers keep this true. Widen ranges only after it runs.
- For Llama-3-8B, `build_layers` in `config.py` is a placeholder `(0,0)` — set it from
  the `run_helix_fit` sweep before running causal validation with the default layer.
