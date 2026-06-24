"""Week 1 — reproduce the generalized helix [kantamneni2025] across layers.

For each layer, cache the residual stream at the number-token position over a sweep
of single-token integers, fit the helix, and compare its R^2 to a matched-parameter
polynomial-of-`a` baseline. A layer band where the helix matches/beats the baseline
is the build region (expected ~14-18 on GPT-J).

Prompt surface forms are the THREE framings (symbolic / word / wordproblem) defined in
n2p.tasks for the chosen --operation; the helix is fit once per framing and the three are
shown as side-by-side panels. The operand `a` is swept while `b` is FIXED (--b_fixed).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_helix_fit.py --model gptj                       # operation=addition
    python3 experiments/week1_number_representation/run_helix_fit.py --model gptj --operation multiplication --read-token a
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks                 # noqa: E402
from n2p.number_repr import helix, repcli              # noqa: E402
from n2p.number_repr.causal import cache_number_site_all_layers  # noqa: E402


def _fit_one_framing(model, spec, operation, framing, values, args):
    """Fit the helix per layer for a single framing. Returns per_layer list or None if
    the framing is incompatible with the requested read-token (e.g. b on a no-b framing)."""
    if args.read_token == "b" and not tasks.template_has_b(operation, framing):
        print(f"[skip] {operation}/{framing}: --read-token b invalid (no operand b)")
        return None
    prefix = args.prefix if args.prefix is not None else spec.prompt_prefix
    shots = tasks.fewshot_shots(operation, args.kshot, args.seed)
    prompt_list = tasks.build_prompts(operation, framing, values, args.b_fixed,
                                      prefix=prefix, shots=shots)
    token_index = tasks.read_token_index(model, prompt_list[0], args.read_token,
                                         operation, framing)
    print(f"[{operation}/{framing}] read-token={args.read_token} at index {token_index}; "
          f"b={args.b_fixed if tasks.template_has_b(operation, framing) else '-'}")
    hooks = [f"blocks.{layer}.hook_resid_post" for layer in range(spec.n_layers)]
    acts_by_hook = cache_number_site_all_layers(model, prompt_list, hooks,
                                                token_index=token_index)
    per_layer = []
    for layer in range(spec.n_layers):
        acts = acts_by_hook[f"blocks.{layer}.hook_resid_post"]
        fit = helix.fit_helix(acts, values, n_pca=args.n_pca)
        base = helix.baseline_pca_r2(acts, values, n_pca=args.n_pca)
        per_layer.append({"layer": layer, "helix_r2": fit["r2"],
                          "poly_baseline_r2": base,
                          "helix_minus_baseline": fit["r2"] - base})
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--operation", choices=tasks.OPERATION_CHOICES, default="addition",
                    help="which arithmetic operation's prompts to fit (its 3 framings are "
                         "shown as side-by-side panels). See n2p.tasks.FRAMINGS.")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=99)    # [0,99]: matches kantamneni2025's
    # helix-fit range and stays below the 3-digit subspace discontinuity at a=100.
    ap.add_argument("--n_pca", type=int, default=9)
    ap.add_argument("--read-token", choices=tasks.READ_TOKEN_CHOICES, default="a",
                    dest="read_token",
                    help="position to fit the helix on: a (default) = operand token, "
                         "b = second operand, sum = last token.")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for framings that use it.")
    ap.add_argument("--prefix", default=None,
                    help="model instruction prefix prepended to every prompt; default = "
                         "config ModelSpec.prompt_prefix for --model. Pass '' to ablate.")
    ap.add_argument("--kshot", type=int, default=0,
                    help="few-shot solved examples prepended before the query (0 = zero-shot). "
                         "GPT-J needs few-shot (e.g. 4) to reliably perform; only matters for "
                         "read=sum (answer site), not read=a (operand site).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = config.get_model_spec(args.model)
    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else spec.prompt_prefix
    print(f"[prefix] {prefix!r}  [kshot] {args.kshot}")

    # Operand grid validated against the real prompt (symbolic framing as the canonical
    # surface form); contiguous prefix keeps the even grid the helix/DFT expect.
    grid_values, _ = tasks.single_token_number_grid(model, args.operation, "symbolic",
                                                    args.lo, args.hi, b=args.b_fixed)
    values = repcli.contiguous_prefix(np.array(grid_values))

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_helix_fit/{args.operation}",
                         meta={"script": "run_helix_fit.py", "operation": args.operation,
                               "read_token": args.read_token, "prefix": prefix,
                               "kshot": args.kshot})

    by_framing = {}
    for framing in tasks.FRAMING_NAMES:
        if framing not in tasks.FRAMINGS[args.operation]:
            continue
        pl = _fit_one_framing(model, spec, args.operation, framing, values, args)
        if pl is not None:
            by_framing[framing] = pl
            best = max(pl, key=lambda r: r["helix_minus_baseline"])
            print(f"  [{framing}] best layer L{best['layer']} "
                  f"(helix-poly delta={best['helix_minus_baseline']:+.3f})")
    if not by_framing:
        raise SystemExit("no framing produced a fit (check --read-token vs operation)")

    summary = {
        "model": args.model, "hf_id": spec.hf_id, "operation": args.operation,
        "n_numbers": len(values), "value_range": [int(values[0]), int(values[-1])],
        "n_pca": args.n_pca, "read_token": args.read_token, "prefix": prefix,
        "kshot": args.kshot,
        "b_fixed": args.b_fixed, "expected_build_layers": list(spec.build_layers),
        "per_framing": {
            f: {"per_layer": pl,
                "best_layer": max(pl, key=lambda r: r["helix_minus_baseline"])["layer"]}
            for f, pl in by_framing.items()},
    }
    (out / f"summary.{args.read_token}.json").write_text(json.dumps(summary, indent=2))
    _plot(by_framing, out / f"helix_r2_by_layer.{args.read_token}.png", args.model,
          args.operation, args.read_token, summary["value_range"])
    print(f"[done] wrote {out} (operation={args.operation}, read={args.read_token})")


def _plot(by_framing, path, model, operation, read_token, value_range):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    framings = list(by_framing)
    fig, axes = plt.subplots(1, len(framings), figsize=(5.5 * len(framings), 4.0),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, framing in zip(axes, framings):
        pl = by_framing[framing]
        L = [r["layer"] for r in pl]
        ax.plot(L, [r["helix_r2"] for r in pl], label="helix R²", marker="o", ms=3)
        ax.plot(L, [r["poly_baseline_r2"] for r in pl], label="poly baseline R²",
                marker="x", ms=3)
        ax.set_xlabel("layer"); ax.set_title(framing)
    axes[0].set_ylabel("R² of read-token resid_post")
    axes[0].legend()
    fig.suptitle(f"Helix vs poly baseline — {model} — {operation} "
                 f"(read={read_token}, a∈[{value_range[0]}..{value_range[1]}])")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
