"""Week 1 — frozen few-shot accuracy per task AND per framing (first observations).

Purpose (per the user, 2026-06-14): before committing to the task family, measure how
well the FROZEN model already does each operation under each framing. This tells us
(a) which (task, framing) pairs the model can actually do — only those give clean
activations for feature tracking — and (b) where the model is weak (candidates for the
write-site-bypass / stub story, and for the fine-tuning fallback decision).

Method: k-shot prompting (frozen model, no fine-tuning — see the feature-tracking wiki
note "Model preparation"); greedy next-token; score by whether the top token matches
the FIRST token of the gold answer (exact for single-token answers; first-token
approximation otherwise — reported separately).

    python3 experiments/week1_accuracy_probe/run_accuracy_probe.py --model gptj
    python3 experiments/week1_accuracy_probe/run_accuracy_probe.py --model gptj --kshot 4 --n 100

Output: results/week1_accuracy_probe/<run_id>/accuracy.json + a task×framing table.
This is a behavioral probe (no activations cached) so it is cheap to run first.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks   # noqa: E402


def first_token_id(model, answer: int) -> int:
    return int(model.to_tokens(f" {answer}", prepend_bos=False)[0, 0].item())


def is_single_token(model, answer: int) -> bool:
    return model.to_tokens(f" {answer}", prepend_bos=False).shape[1] == 1


def build_fewshot(template: str, shots, query) -> str:
    """k solved examples in the SAME framing, then the query (no trailing space)."""
    lines = [template.format(a=s["a"], b=s["b"]) + f" {s['answer']}" for s in shots]
    lines.append(template.format(a=query["a"], b=query["b"]))
    return "\n".join(lines)


@torch.no_grad()
def accuracy_for(model, task, framing_name, template, kshot, n, seed):
    data = task.sample(n + kshot, seed=seed)
    shots, queries = data[:kshot], data[kshot:]
    n_total = n_correct = n_single = 0
    for q in queries:
        prompt = build_fewshot(template, shots, q)
        toks = model.to_tokens(prompt)
        logits = model(toks)
        pred = int(logits[0, -1].argmax().item())
        gold = first_token_id(model, q["answer"])
        n_total += 1
        n_correct += int(pred == gold)
        n_single += int(is_single_token(model, q["answer"]))
    return {
        "framing": framing_name, "n": n_total,
        "first_token_acc": round(n_correct / max(n_total, 1), 4),
        "frac_single_token_answers": round(n_single / max(n_total, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task names; default = all except greater_than")
    args = ap.parse_args()

    model = models.load_model(args.model)
    task_names = args.tasks or [n for n in tasks.REGISTRY if n != "greater_than"]

    rows = []
    for tname in task_names:
        task = tasks.get_task(tname)
        for fname, template in tasks.framings_for(task):
            res = accuracy_for(model, task, fname, template, args.kshot, args.n, args.seed)
            res.update(task=tname, tier=task.tier)
            rows.append(res)
            print(f"{tname:16s} {fname:12s} acc={res['first_token_acc']:.3f} "
                  f"(single-tok answers {res['frac_single_token_answers']:.2f})")

    out = config.run_dir("week1_accuracy_probe", args.seed, model=args.model)
    summary = {"model": args.model, "kshot": args.kshot, "n": args.n, "rows": rows}
    (out / "accuracy.json").write_text(json.dumps(summary, indent=2))
    _write_table(rows, out / "accuracy_table.md", args.model, args.kshot)
    print(f"[done] wrote {out}")


def _write_table(rows, path, model, kshot):
    lines = [f"# Few-shot accuracy — {model} (k={kshot})", "",
             "| task | tier | framing | first-token acc | single-tok answers |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['tier']} | {r['framing']} | "
                     f"{r['first_token_acc']:.3f} | {r['frac_single_token_answers']:.2f} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
