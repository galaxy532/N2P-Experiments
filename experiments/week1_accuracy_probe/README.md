# Week 1 — Few-shot accuracy probe (task × framing)

**First observations before committing the family.** Measures how well the **frozen**
model already performs each operation under each **framing**, k-shot. Cheap (behavioral,
no activations cached) — run this first.

```bash
python experiments/week1_accuracy_probe/run_accuracy_probe.py --model gptj --kshot 4 --n 100
python experiments/week1_accuracy_probe/run_accuracy_probe.py --model llama3-8b
```

## How to read the output
`results/week1_accuracy_probe/<run_id>/accuracy_table.md` has one row per (task, framing):
- **exact-value acc** — the full greedily-generated integer equals the gold answer. This
  is the reliable metric: robust across tokenizers and answer lengths (it parses the
  decoded integer, so multi-token products like `20*20=400` are scored at full value).
  Controlled by `--gen-tokens` (default 6, enough for the largest product plus a sign).
- **first-token acc** — the first CONTENT (non-space) generated token matches the gold's
  first content token. The cheap literature-style proxy: exact for single-token answers,
  a leading-digit/magnitude approximation otherwise. The "content" qualifier makes it
  meaningful on Llama-3, which emits the leading space as its own token.
- **single-tok answers** — fraction of this cell's answers that are single tokens. When
  low (e.g. multiplication), trust **exact-value acc**, not first-token acc — the gap
  between the two columns is exactly the first-token proxy's optimism.

## What it decides
- **Which (task, framing) pairs are usable** for feature tracking — only ones the model
  actually does (clean activations). Low-accuracy pairs are *not* discarded blindly:
  per [engels2024 §5] a task can be solvable under one framing and not another, so a
  zero in one framing is a finding, not a reason to drop the task.
- **Where the model is weak** — candidates for the write-site-bypass / stub story, and
  the trigger for the fine-tuning-fallback decision (see the feature-tracking wiki note
  "Model preparation"). **Multiplication is expected to be the weakest** clean test of
  whether the model computes vs. memorizes — its accuracy here decides whether we keep
  it as a frozen target or move it to the fine-tuning fallback.

## Model prompt prefix (`--prefix`)
Every few-shot prompt is prepended with a model-specific instruction from
`config.ModelSpec.prompt_prefix` (GPT-J `"Output ONLY a number.\n"`, Llama-3 `"The following
is a correct math problem. \n"`), per [kantamneni2025]'s finding that models need different
prompts to perform. Printed at startup and stored in `accuracy.json`. To measure the
prefix's effect, run with and without it: `--prefix ""` ablates. (Llama's generic "math
problem" generalizes the paper's addition-only string across our task family — this probe is
the check that it still elicits the task.)

## Caveats
- Frozen + few-shot only (no fine-tuning) — that is the protocol default.
- `exact-value acc` removes the multi-token blind spot; `first-token acc` is kept only as
  the comparable proxy. Read the two together (the gap = proxy optimism), per row.
- `greater_than` is excluded (comparison, not numeric next-token) — its accuracy is a
  separate logit-difference metric handled with the discovery task.
