"""Week 1 — reproduce the generalized helix [kantamneni2025] across layers.

For each layer, cache the residual stream at the number-token position over a sweep
of single-token integers, fit the helix, and compare its R^2 to a matched-parameter
polynomial-of-`a` baseline. A layer band where the helix matches/beats the baseline
is the build region (expected ~14-18 on GPT-J).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_helix_fit.py --model gptj
    python3 experiments/week1_number_representation/run_helix_fit.py --model gptj --context template_3 --read-token a
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models                      # noqa: E402
from n2p.number_repr import helix, prompts           # noqa: E402
from n2p.number_repr.causal import cache_number_site_all_layers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=99)    # [0,99]: matches kantamneni2025's
    # helix-fit range and stays below the 3-digit subspace discontinuity at a=100
    # (PC1 jump, App. B). Use --hi 360 for the wider sweep / DFT structure analysis.
    ap.add_argument("--n_pca", type=int, default=9)
    ap.add_argument("--context", choices=prompts.TEMPLATE_CHOICES, default="template_1",
                    help="prompt framing (see n2p.number_repr.prompts). template_1 is the "
                         "context-free bare operand (original default). template_3/4 put "
                         "addition context before a, as [kantamneni2025] §4.3 does — the "
                         "regime where the model actually deploys the helix.")
    ap.add_argument("--read-token", choices=prompts.READ_TOKEN_CHOICES, default="a",
                    dest="read_token",
                    help="which position to fit the helix on: a (default) = operand token, "
                         "b = second operand, sum = last token.")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for templates that use it (templates 3,4).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    prompts.validate_read_token(args.read_token, args.context)

    spec = config.get_model_spec(args.model)
    model = models.load_model(args.model)

    # Single-token integers only, contiguous.
    tok_map = models.number_token_ids(model, args.lo, args.hi)
    values = np.array(sorted(tok_map))
    # Keep the contiguous prefix from the first value; stop at the first gap so the
    # value grid is evenly sampled (helix fit + later DFT assume contiguity). Warn so
    # dropped numbers are visible rather than silently producing a holey set.
    if values.size:
        gaps = np.where(np.diff(values) != 1)[0]
        if gaps.size:
            cut = int(gaps[0]) + 1
            print(f"[warn] gap after {values[cut-1]} (next single-token value is "
                  f"{values[cut]}); dropping {len(values) - cut} value(s), using "
                  f"contiguous {values[0]}..{values[cut-1]} ({cut} numbers)")
            values = values[:cut]

    # [kantamneni2025] §4.3: fit helix(a) on the residual stream ON TOP OF THE read token
    # within a prompt (where the helix is actually used). b is fixed so all prompts share
    # structure -> the read token sits at a constant index and the batch needs no padding.
    prompt_list = prompts.build_prompts(args.context, values, args.b_fixed)
    token_index = prompts.read_token_index(model, prompt_list[0], args.read_token,
                                           args.context)
    print(f"[{args.context}] read-token={args.read_token} at index {token_index}; "
          f"b={args.b_fixed if prompts.template_has_b(args.context) else '-'}")

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_helix_fit/{args.context}",
                         meta={"script": "run_helix_fit.py", "context": args.context,
                               "read_token": args.read_token})
    # One batched forward sweep caches resid_post at every layer at once (vs. a full
    # forward per layer). Read each layer's activations out of the returned dict.
    hooks = [f"blocks.{layer}.hook_resid_post" for layer in range(spec.n_layers)]
    acts_by_hook = cache_number_site_all_layers(model, prompt_list, hooks, token_index=token_index)
    per_layer = []
    for layer in range(spec.n_layers):
        acts = acts_by_hook[f"blocks.{layer}.hook_resid_post"]  # (N, d)
        fit = helix.fit_helix(acts, values, n_pca=args.n_pca)
        base = helix.baseline_pca_r2(acts, values, n_pca=args.n_pca)
        per_layer.append({"layer": layer, "helix_r2": fit["r2"],
                          "poly_baseline_r2": base,
                          "helix_minus_baseline": fit["r2"] - base})
        print(f"L{layer:02d}  helix R2={fit['r2']:.3f}  poly R2={base:.3f}  "
              f"delta={fit['r2']-base:+.3f}")

    summary = {
        "model": args.model, "hf_id": spec.hf_id, "n_numbers": len(values),
        "value_range": [int(values[0]), int(values[-1])], "n_pca": args.n_pca,
        "context": args.context, "read_token": args.read_token,
        "b_fixed": args.b_fixed if prompts.template_has_b(args.context) else None,
        "per_layer": per_layer,
        "expected_build_layers": list(spec.build_layers),
        "best_layer": max(per_layer, key=lambda r: r["helix_minus_baseline"])["layer"],
    }
    # Folder = <model>/run_helix_fit/<context>; the file name carries the read-token.
    (out / f"summary.{args.read_token}.json").write_text(json.dumps(summary, indent=2))
    _plot(per_layer, out / f"helix_r2_by_layer.{args.read_token}.png", args.model,
          args.context, args.read_token, summary["value_range"])
    print(f"[done] wrote {out} (context={args.context}, read={args.read_token})")


def _plot(per_layer, path, model, context, read_token, value_range):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = [r["layer"] for r in per_layer]
    plt.figure(figsize=(8, 4))
    plt.plot(L, [r["helix_r2"] for r in per_layer], label="helix R²", marker="o", ms=3)
    plt.plot(L, [r["poly_baseline_r2"] for r in per_layer], label="poly baseline R²",
             marker="x", ms=3)
    plt.xlabel("layer"); plt.ylabel("R² of read-token resid_post")
    plt.title(f"Helix vs poly baseline — {model} "
              f"({context}, read={read_token}, a∈[{value_range[0]}..{value_range[1]}])")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
