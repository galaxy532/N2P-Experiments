"""Week 1 — frozen few-shot accuracy per task AND per framing (first observations).

Purpose (per the user, 2026-06-14): before committing to the task family, measure how
well the FROZEN model already does each operation under each framing. This tells us
(a) which (task, framing) pairs the model can actually do — only those give clean
activations for feature tracking — and (b) where the model is weak (candidates for the
write-site-bypass / stub story, and for the fine-tuning fallback decision).

Method: k-shot prompting (frozen model, no fine-tuning — see the feature-tracking wiki
note "Model preparation"); greedy decoding. TWO scores are reported per (task, framing):
  - exact_value_acc : the full generated integer equals the gold answer (robust across
                      tokenizers and answer lengths — the reliable metric).
  - first_token_acc : the first CONTENT (non-space) token matches the gold's first content
                      token (the cheap literature-style proxy; exact for single-token
                      answers, leading-digit/magnitude otherwise). Derived from the same
                      generation, skipping a leading-space token so it is meaningful on
                      Llama-3 (which emits the space as its own token).

    python3 experiments/week1_accuracy_probe/run_accuracy_probe.py --model gptj
    python3 experiments/week1_accuracy_probe/run_accuracy_probe.py --model gptj --kshot 4 --n 100

Output: results/week1_accuracy_probe/<run_id>/accuracy.json + a task×framing table.
This is a behavioral probe (no activations cached) so it is cheap to run first.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks   # noqa: E402


@torch.no_grad()
def greedy_generate(model, toks, max_new_tokens: int) -> list[int]:
    """Greedy-decode up to `max_new_tokens` ids, stopping at the first newline (few-shot
    lines are newline-separated). Returns the generated ids (newline excluded)."""
    gen: list[int] = []
    cur = toks
    for _ in range(max_new_tokens):
        logits = model(cur)
        nxt = int(logits[0, -1].argmax().item())
        if "\n" in model.to_string([nxt]):
            break
        gen.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return gen


def first_content_token_id(model, ids: list[int]):
    """First non-whitespace token id in `ids` (skips a leading-space token, e.g. Llama-3)."""
    for tid in ids:
        if model.to_string([tid]).strip() != "":
            return tid
    return None


def parse_leading_int(text: str):
    """Leading (optionally signed) integer in the generated text, or None."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def build_fewshot(template: str, shots, query) -> str:
    """k solved examples in the SAME framing, then the query (no trailing space)."""
    lines = [template.format(a=s["a"], b=s["b"]) + f" {s['answer']}" for s in shots]
    lines.append(template.format(a=query["a"], b=query["b"]))
    return "\n".join(lines)


@torch.no_grad()
def accuracy_for(model, task, framing_name, template, kshot, n, seed, gen_tokens, prefix=""):
    data = task.sample(n + kshot, seed=seed)
    shots, queries = data[:kshot], data[kshot:]
    n_total = n_exact = n_first = n_single = 0
    for q in queries:
        toks = model.to_tokens(prefix + build_fewshot(template, shots, q))
        gen_ids = greedy_generate(model, toks, gen_tokens)
        pred_value = parse_leading_int(model.to_string(gen_ids) if gen_ids else "")
        pred_first = first_content_token_id(model, gen_ids)
        gold_first = models.first_answer_token_id(model, q["answer"])
        n_total += 1
        n_exact += int(pred_value == q["answer"])
        n_first += int(pred_first is not None and pred_first == gold_first)
        n_single += int(models.is_single_token_answer(model, q["answer"]))
    return {
        "framing": framing_name, "n": n_total,
        "exact_value_acc": round(n_exact / max(n_total, 1), 4),
        "first_token_acc": round(n_first / max(n_total, 1), 4),
        "frac_single_token_answers": round(n_single / max(n_total, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-tokens", type=int, default=6, dest="gen_tokens",
                    help="max new tokens to greedy-decode per query for exact-value scoring "
                         "(enough for the largest product, e.g. 20*20=400, plus a sign).")
    ap.add_argument("--prefix", default=None,
                    help="model instruction prefix prepended to the few-shot prompt; default "
                         "= config ModelSpec.prompt_prefix for --model. Pass '' to ablate.")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task names; default = all except greater_than")
    args = ap.parse_args()

    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else config.get_model_spec(args.model).prompt_prefix
    print(f"[prefix] {prefix!r}")
    task_names = args.tasks or [n for n in tasks.REGISTRY if n != "greater_than"]

    rows = []
    for tname in task_names:
        task = tasks.get_task(tname)
        for fname, template in tasks.framings_for(task):
            res = accuracy_for(model, task, fname, template, args.kshot, args.n, args.seed,
                               args.gen_tokens, prefix)
            res.update(task=tname, tier=task.tier)
            rows.append(res)
            print(f"{tname:16s} {fname:12s} exact={res['exact_value_acc']:.3f} "
                  f"first-tok={res['first_token_acc']:.3f} "
                  f"(single-tok answers {res['frac_single_token_answers']:.2f})")

    out = config.run_dir("week1_accuracy_probe", args.seed, model=args.model)
    summary = {"model": args.model, "kshot": args.kshot, "n": args.n,
               "gen_tokens": args.gen_tokens, "prefix": prefix, "rows": rows}
    (out / "accuracy.json").write_text(json.dumps(summary, indent=2))
    _write_table(rows, out / "accuracy_table.md", args.model, args.kshot)
    print(f"[done] wrote {out}")


def _write_table(rows, path, model, kshot):
    lines = [f"# Few-shot accuracy — {model} (k={kshot})", "",
             "| task | tier | framing | exact-value acc | first-token acc | "
             "single-tok answers |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['tier']} | {r['framing']} | "
                     f"{r['exact_value_acc']:.3f} | {r['first_token_acc']:.3f} | "
                     f"{r['frac_single_token_answers']:.2f} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
