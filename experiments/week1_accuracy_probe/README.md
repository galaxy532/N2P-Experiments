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
- **first-token acc** — top predicted token == first token of the gold answer. Exact
  for single-token answers; a first-token approximation otherwise.
- **single-tok answers** — fraction of this cell's answers that are single tokens.
  If this is low (e.g. multiplication, where products get large/multi-token), the
  accuracy number is only a first-token proxy and needs a generation-based re-check.

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

## Caveats
- Frozen + few-shot only (no fine-tuning) — that is the protocol default.
- Multi-token answers (big products) make this a first-token proxy; flagged per row.
- `greater_than` is excluded (comparison, not numeric next-token) — its accuracy is a
  separate logit-difference metric handled with the discovery task.
