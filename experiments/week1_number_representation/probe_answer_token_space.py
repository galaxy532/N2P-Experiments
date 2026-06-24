"""PROBE (diagnostic, not a protocol run) — at the answer position, does the model emit a
leading SPACE token before the digits, and does it depend on the prompt FORMAT?

WHY: `run_causal_validation` reads the answer-number token logit at position -1 and the
accuracy probe scores the first token. On GPT-J the leading space merges into the number
token; on Llama-3 the space is its own token, so IF the model emits a space first the read
is one position off. The paper's repo (subhashk01/LLM-addition) reads the BARE digit token
at -1 with a zero-shot, no-trailing-space prompt and gets good Llama accuracy — i.e. in
that format Llama emits the digit directly. This probe checks both formats on OUR models so
we can confirm before trusting first-token metrics.

Two formats per query:
  (A) paper/zero-shot : "<prefix>\n{a}+{b}="     (ends '=', NO trailing space)  <- causal style
  (B) few-shot spaced : k * "{a}+{b}= {ans}\n" then "{a}+{b}="                   <- accuracy style

Run on Paperspace (A6000):
    python3 experiments/week1_number_representation/probe_answer_token_space.py --model llama3-8b
    python3 experiments/week1_number_representation/probe_answer_token_space.py --model gptj
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import models, tasks                           # noqa: E402

# Model-specific zero-shot prefixes used by the paper repo (helix_fitting.ipynb).
PREFIX = {
    "gptj": "Output ONLY a number.\n",
    "llama3-8b": "The following is a correct addition problem. \n",
}


@torch.no_grad()
def greedy(model, prompt: str, k: int) -> list[int]:
    toks = model.to_tokens(prompt)
    out: list[int] = []
    for _ in range(k):
        nxt = int(model(toks)[0, -1].argmax().item())
        out.append(nxt)
        toks = torch.cat([toks, torch.tensor([[nxt]], device=toks.device)], dim=1)
    return out


def first_content(model, ids) -> int | None:
    for tid in ids:
        if model.to_string([int(tid)]).strip() != "":
            return int(tid)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--n", type=int, default=12, help="query prompts to probe")
    ap.add_argument("--gen-tokens", type=int, default=4, dest="gen_tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = models.load_model(args.model)
    prefix = PREFIX.get(args.model, "")
    data = tasks.get_task("addition").sample(args.n + args.kshot, seed=args.seed)
    shots, queries = data[:args.kshot], data[args.kshot:]

    def paper_zeroshot(q):
        return f"{prefix}{q['a']}+{q['b']}="

    def fewshot_spaced(q):
        lines = [f"{s['a']}+{s['b']}= {s['answer']}" for s in shots]
        lines.append(f"{q['a']}+{q['b']}=")
        return "\n".join(lines)

    # answer-token convention candidates (both with prepend_bos=False, so no BOS):
    def bare_id(ans):  return first_content(model, model.to_tokens(f"{ans}", prepend_bos=False)[0])
    def space_id(ans): return first_content(model, model.to_tokens(f" {ans}", prepend_bos=False)[0])

    for label, build in (("A paper/zero-shot (ends '=')", paper_zeroshot),
                         ("B few-shot spaced ('= ans')", fewshot_spaced)):
        print(f"\n===== format {label} | model={args.model} =====")
        n_space = n_bare = n_spacetok = 0
        for q in queries:
            ids = greedy(model, build(q), args.gen_tokens)
            strs = [model.to_string([i]) for i in ids]
            fc = first_content(model, ids)
            first_is_space = bool(strs) and strs[0].strip() == ""
            n_space += int(first_is_space)
            n_bare += int(fc is not None and fc == bare_id(q["answer"]))
            n_spacetok += int(fc is not None and fc == space_id(q["answer"]))
            print(f"  {q['a']:>2}+{q['b']:>2}={q['answer']:>4}  tokens={strs}  "
                  f"first_is_space={first_is_space}  "
                  f"first_content=='{model.to_string([fc]) if fc is not None else None}'")
        N = len(queries)
        print(f"  [summary] first token whitespace: {n_space}/{N} | "
              f"first content == bare '{{ans}}' token: {n_bare}/{N} | "
              f"== ' {{ans}}' token: {n_spacetok}/{N}")

    print("\nInterpretation:")
    print("- If format A shows ~0 whitespace and high 'bare' match -> read the BARE digit at -1")
    print("  (matches the paper); current zero-shot causal/number-rep prompts are fine.")
    print("- If format B shows mostly whitespace -> few-shot spacing induces the space; the")
    print("  accuracy probe already handles it (exact-value parse + content-token first-token).")


if __name__ == "__main__":
    main()
