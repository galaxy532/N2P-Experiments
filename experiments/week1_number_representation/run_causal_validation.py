"""Week 1 — causal sufficiency of the helix subspace [engels2024 §5 / kantamneni2025].

Pipeline (now swept across layers, and across operations/framings):
  1. For the chosen --operation/--framing, cache the operand-`a` token residual stream
     over a sweep of single-token operands a (b FIXED), fit the helix -> PCA subspace +
     helix map C. This is done PER LAYER in the --layers sweep.
  2. For test triples (a, a', b): take the clean prompt, and PATCH the operand-a site
     toward a' INSIDE the helix subspace while average-ablating the rest. Measure
     logit_diff = logit[first_tok(ans(a'))] - logit[first_tok(ans(a))].
  3. Compare to:
        - no-op (should be ~0 / negative, i.e. still says ans(a)) — the FLOOR,
        - full-layer patch from an "a'..." donor run — the sufficiency UPPER BOUND / ceiling,
        - PCA-reconstruction baseline at k=9 (capacity-matched) and k=27 (over-capacity),
          patching the REAL a' activation's PCA-k reconstruction (no helix assumed) — the
          [kantamneni2025] Fig-5 control that isolates the helix FORM from raw capacity.
  Helix family is also split into MAGNITUDE (linear + T=100) vs MODULAR (T=2,5,10) to see
  which part carries the causal effect, alongside the WHOLE helix.
  Subspace-patch ≈ full-layer-patch  =>  the helix subspace is causally sufficient.

WHY swept across LAYERS (2026-06-22, user-approved): a single build-layer number is only
a go/no-go; the [kantamneni2025] Fig-5 / [engels2024] Fig-6 object is the curve of mean
logit-diff vs LAYER OF INTERVENTION, which shows WHERE sufficiency emerges and decays.
The default sweep is now ALL layers 0..last (2026-06-27, user request): the build_layers
band is a [kantamneni2025] prior on GPT-J and an unverified placeholder on Llama, so it is
NOT taken for granted here; pass --layers LO HI.. to restrict.
WHY swept across OPERATIONS: although the operand-`a` representation (and hence the fitted
helix basis) is IDENTICAL across operations sharing a pre-`{a}` prefix (causal masking;
see exp-notes/helix-experiments-week1-results.md), the patch MEASUREMENT runs the
operation-specific downstream to the answer, so "is the operand-a helix causally
sufficient FOR THE ANSWER" is a genuinely different question per operation — a direct test
of the stress-set hypothesis (do mult/div/modular read the operand helically, or via
heuristics outside the subspace?). The fit is reused; only patch-and-measure repeats.

First-token answers (2026-06-22): scored on the FIRST token of the gold answer (same
convention as run_accuracy_probe.py), so multi-token answers (e.g. multiplication) are
admissible. NB this is then a LEADING-DIGIT / MAGNITUDE test, not a full-value test; for
single-token answers (e.g. modular, small addition) it is exact. Triples whose two answers
share a first token are skipped (the logit-diff would be ~0 by construction).

All framings by default (2026-06-27, user-approved): like the fourier --summary path, the
bare run now sweeps EVERY framing of the operation (symbolic/word/wordproblem) and draws
one panel-row per framing in a single combined figure (causal_by_layer.summary.png), with
one JSON per framing. The fit + patch-and-measure repeats per framing (the surface form
before `{a}` differs, so the fitted helix basis differs) — ~3x the patched forwards, but
no extra VRAM (one framing's tensors live at a time). Pass --framing F to restrict to a
single framing (the old behavior, written to causal_by_layer.<F>.png).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_causal_validation.py --model gptj --operation addition --kshot 4
    python3 experiments/week1_number_representation/run_causal_validation.py --model gptj \
        --operation multiplication --framing symbolic --layers 10 12 14 16 18 20

NOTE: trickiest week-1 script; operand-token indexing and first-token answer constraints
will likely need a first-run debug pass on GPU.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks                 # noqa: E402
from n2p.number_repr import helix, repcli             # noqa: E402
from n2p.number_repr import causal                    # noqa: E402


# Answer-token helpers (model-agnostic, in n2p.models). The prompt is zero-shot ending in
# "=" with NO trailing space, so we read the BARE answer token (space=False): probe-confirmed
# 2026-06-24, GPT-J emits the bare '99' there (not ' 99'), and Llama is identical (its space
# is always a separate token). This matches [kantamneni2025]'s bare tokenizer(f'{answer}').
def first_token_id(model, answer):
    return models.first_answer_token_id(model, answer, space=False)


def is_single_token_answer(model, answer):
    return models.is_single_token_answer(model, answer, space=False)


def resolve_layers(args, spec) -> list[int]:
    """--layers wins; else single --layer; else ALL layers 0..n_layers-1. We deliberately
    do NOT seed the default from spec.build_layers: the GPT-J band is a [kantamneni2025]
    prior and the Llama band is an unverified placeholder, so the sweep should not take
    them for granted (2026-06-27, user request). Restrict with --layers if slow."""
    if args.layers:
        return sorted({L for L in args.layers if 0 <= L < spec.n_layers})
    if args.layer is not None:
        return [args.layer]
    return list(range(spec.n_layers))


def fit_helix_all_layers(model, operation, framing, values, b_fixed, layers, n_pca,
                         pca_dims, prefix="", shots=()):
    """Fit the helix on the operand-a token at EVERY swept layer in one batched cache
    sweep, plus the PCA baselines. Returns {layer: {hook, fitres, site_mean, bases, pca,
    r2}}. The operand-a index is constant across the sweep (fixed prefix + fixed few-shot
    shots + single-token operands), so one token_index batches all prompts."""
    prompts = tasks.build_prompts(operation, framing, values, b_fixed, prefix=prefix,
                                  shots=shots)
    ix = tasks.read_token_index(model, prompts[0], "a", operation, framing)
    hooks = [f"blocks.{L}.hook_resid_post" for L in layers]
    acts_by_hook = causal.cache_number_site_all_layers(model, prompts, hooks,
                                                       token_index=ix)
    vals = np.asarray(values)
    per_layer = {}
    for L in layers:
        hook = f"blocks.{L}.hook_resid_post"
        acts = acts_by_hook[hook]
        fitres = helix.fit_helix(acts, vals, n_pca=n_pca)
        bases = {
            "helix_full": helix.helix_subspace_basis(fitres),
            "helix_magnitude": helix.helix_subspace_basis(
                fitres, periods=(100,), include_linear=True, include_intercept=True),
            "helix_modular": helix.helix_subspace_basis(
                fitres, periods=(2, 5, 10), include_linear=False, include_intercept=False),
        }
        pca_models = {k: PCA(n_components=k).fit(acts) for k in pca_dims
                      if k <= min(acts.shape)}
        per_layer[L] = {"hook": hook, "fitres": fitres, "site_mean": acts.mean(0),
                        "bases": bases, "pca": pca_models, "r2": fitres["r2"]}
    return per_layer, ix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--operation", choices=tasks.OPERATION_CHOICES, default="addition",
                    help="which operation's prompts to validate (its answer comes from "
                         "tasks.REGISTRY[op].fn). See n2p.tasks.FRAMINGS.")
    ap.add_argument("--framing", choices=tasks.FRAMING_NAMES, default=None,
                    help="restrict to a single framing; default = ALL framings of the "
                         "operation, one panel-row each (like the fourier --summary).")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layers of intervention to sweep (Fig-5/6 x-axis). Default = "
                         "ALL layers (0..last); no build_layers prior is assumed.")
    ap.add_argument("--layer", type=int, default=None,
                    help="single-layer shortcut (ignored if --layers given).")
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--n_pca", type=int, default=9, help="PCA dims the HELIX is fit inside")
    ap.add_argument("--pca_dims", type=int, nargs="*", default=[9, 27],
                    help="dims for the PCA-reconstruction baseline (kantamneni2025 Fig 5 "
                         "used 9 = capacity-matched to the helix, and 27 = over-capacity).")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for the helix fit sweep.")
    ap.add_argument("--prefix", default=None,
                    help="model instruction prefix prepended to every prompt; default = "
                         "config ModelSpec.prompt_prefix for --model. Pass '' to ablate.")
    ap.add_argument("--kshot", type=int, default=0,
                    help="few-shot solved examples before each query (0 = zero-shot). GPT-J "
                         "needs few-shot (e.g. 4) to actually answer — the logit-diff is "
                         "meaningless if the clean run does not produce the answer.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = config.get_model_spec(args.model)
    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else spec.prompt_prefix
    shots = tasks.fewshot_shots(args.operation, args.kshot, args.seed)
    print(f"[prefix] {prefix!r}  [kshot] {args.kshot}")
    operation = args.operation
    layers = resolve_layers(args, spec)
    # Default = ALL framings of the operation (one panel-row each), mirroring the fourier
    # --summary path; --framing F restricts to a single framing. read-token is fixed "a".
    framings = [args.framing] if args.framing else repcli.framings_for_summary(operation, "a")
    if not framings:
        raise SystemExit(f"no framing available for operation {operation!r}")
    print(f"[setup] {operation}; framings={framings}; layers={layers[0]}..{layers[-1]} "
          f"({len(layers)} layers) x ~{args.n_test} triples x methods per framing -> "
          f"~{len(framings) * len(layers) * args.n_test} patched forwards total. "
          f"Restrict with --framing / --layers if slow.")

    out = config.run_dir("week1_number_representation", args.seed, model=args.model,
                         label=f"run_causal_validation/{operation}",
                         meta={"script": "run_causal_validation.py", "operation": operation,
                               "framing": framings[0] if len(framings) == 1 else None,
                               "framings": framings, "read_token": "a", "layers": layers,
                               "prefix": prefix, "kshot": args.kshot})

    panels = []  # (framing, per_layer_summary, methods, frac_single)
    for framing in framings:
        per_layer_summary, methods, summary, triples = run_one_framing(
            model, spec, args, operation, framing, prefix, shots, layers)
        (out / f"causal_validation.{framing}.json").write_text(
            json.dumps({"summary": summary, "trials": triples}, indent=2))
        panels.append((framing, per_layer_summary, methods,
                       summary["frac_single_token_answers"]))
        print(json.dumps(summary, indent=2))

    plot_name = (f"causal_by_layer.{framings[0]}.png" if len(framings) == 1
                 else "causal_by_layer.summary.png")
    _plot(panels, out / plot_name, args.model, operation)
    print(f"[done] wrote {out} ({len(framings)} framing(s): {framings})")


def run_one_framing(model, spec, args, operation, framing, prefix, shots, layers):
    """Full fit -> patch-and-measure -> aggregate pipeline for ONE framing. Returns
    (per_layer_summary, methods, summary, triples). Reseeds its own RNG from --seed so
    framings are sampled comparably (identical seed -> identical triples where the
    single-token pools coincide)."""
    rng = random.Random(args.seed)

    # --- single-token operand sweep for the helix fit (same pool as run_helix_fit) ---
    # Validated against the real (operation, framing) prompt so it is correct on any
    # tokenizer (the old space-prefixed number_token_ids returned an empty pool on Llama-3).
    grid_values, _ = tasks.single_token_number_grid(model, operation, framing, 0, 99,
                                                    b=args.b_fixed)
    fit_values = repcli.contiguous_prefix(np.array(grid_values))
    if args.n_fit and len(fit_values) > args.n_fit:
        fit_values = fit_values[:args.n_fit]

    # --- 1. fit helix (+ PCA baselines) at every swept layer ---
    per_layer, _ = fit_helix_all_layers(model, operation, framing, fit_values,
                                        args.b_fixed, layers, args.n_pca, args.pca_dims,
                                        prefix=prefix, shots=shots)
    pca_keys = sorted({k for L in layers for k in per_layer[L]["pca"]})
    helix_keys = ["helix_full", "helix_magnitude", "helix_modular"]
    methods = helix_keys + [f"pca{k}" for k in pca_keys] + ["full_layer", "noop"]

    # test operands stay in the operation's designed a-range (matches tasks.sample);
    # b from its b_range. Both operands are single-token (drawn from the fit pool).
    task = tasks.get_task(operation)
    a_lo, a_hi = task.a_range
    test_pool = [int(v) for v in fit_values if a_lo <= v <= a_hi]
    b_lo, b_hi = task.b_range

    def noop_logit_diff(clean, ans):
        last = model(model.to_tokens(clean))[0, -1]
        return float(last[ans[1]] - last[ans[0]])

    # --- 2/3. patch experiments on held-out triples, swept over layers ---
    diffs = {L: {m: [] for m in methods} for L in layers}
    triples = []
    tries = 0
    while len(triples) < args.n_test and tries < args.n_test * 80:
        tries += 1
        a, ap_ = rng.choice(test_pool), rng.choice(test_pool)
        b = rng.randint(b_lo, b_hi)
        if a == ap_:
            continue
        ans_a, ans_ap = task.fn(a, b), task.fn(ap_, b)
        if ans_a < 0 or ans_ap < 0:                       # skip nonsensical negatives
            continue
        fa, fap = first_token_id(model, ans_a), first_token_id(model, ans_ap)
        if fa == fap:                                     # logit-diff would be ~0
            continue
        ans = (fa, fap)
        clean = tasks.build_prompt(operation, framing, a, b, prefix=prefix, shots=shots)
        donor = tasks.build_prompt(operation, framing, ap_, b, prefix=prefix, shots=shots)
        ix = tasks.read_token_index(model, clean, "a", operation, framing)
        donor_ix = tasks.read_token_index(model, donor, "a", operation, framing)
        hooks = [per_layer[L]["hook"] for L in layers]
        donor_by_hook = causal.cache_number_site_all_layers(model, [donor], hooks,
                                                            token_index=donor_ix)
        nodiff = noop_logit_diff(clean, ans)
        for L in layers:
            pl = per_layer[L]
            hook = pl["hook"]
            for name in helix_keys:
                basis = pl["bases"][name]
                tgt = helix.helix_target_in_basis([ap_], pl["fitres"], basis)[0]
                diffs[L][name].append(causal.subspace_patch_logit_diff(
                    model, clean, hook, basis, tgt, pl["site_mean"],
                    answer_tokens=ans, token_index=ix))
            for k in pca_keys:
                if k not in pl["pca"]:
                    diffs[L][f"pca{k}"].append(float("nan"))
                    continue
                donor_act = donor_by_hook[hook][0]
                tgt_k = pl["pca"][k].transform(donor_act[None, :])[0]
                diffs[L][f"pca{k}"].append(causal.subspace_patch_logit_diff(
                    model, clean, hook, pl["pca"][k].components_.T, tgt_k, pl["site_mean"],
                    answer_tokens=ans, token_index=ix))
            diffs[L]["full_layer"].append(causal.full_layer_patch_logit_diff(
                model, clean, donor, hook, answer_tokens=ans, token_index=ix))
            diffs[L]["noop"].append(nodiff)
        triples.append({"a": a, "ap": ap_, "b": b, "ans_a": ans_a, "ans_ap": ans_ap})

    # --- aggregate per layer ---
    def nanmean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    per_layer_summary = []
    for L in layers:
        means = {m: nanmean(diffs[L][m]) for m in methods}
        full = means["full_layer"]
        ratios = {m: (means[m] / full if abs(full) > 1e-6 else float("nan"))
                  for m in methods}
        per_layer_summary.append({"layer": L, "helix_fit_r2": per_layer[L]["r2"],
                                  "mean_logit_diff": means,
                                  "ratio_over_full_layer": ratios})

    n_single = sum(is_single_token_answer(model, t["ans_a"]) for t in triples)
    summary = {
        "model": args.model, "hf_id": spec.hf_id, "operation": operation,
        "framing": framing, "read_token": "a", "layers": layers, "prefix": prefix,
        "kshot": args.kshot,
        "n_pca": args.n_pca, "pca_dims": pca_keys, "n_test": len(triples),
        "frac_single_token_answers": round(n_single / max(len(triples), 1), 3),
        "helix_ranks": {n: int(b.shape[1])
                        for n, b in per_layer[layers[0]]["bases"].items()},
        "per_layer": per_layer_summary,
        "interpretation": (
            "Curves are mean logit-diff vs LAYER OF INTERVENTION (Fig-5/6 x-axis). "
            "full_layer = causal ceiling; noop ~0/negative = floor. helix_full tracking "
            "full_layer over a layer band => the helix is causally sufficient there. "
            "helix_full matching/beating pca9/pca27 with fewer effective dims => the "
            "periodic FORM, not generic PCA capacity, carries the number. magnitude vs "
            "modular shows which part does the causal work. For multi-token answers this "
            "is a leading-digit/magnitude test (see frac_single_token_answers)."),
    }
    return per_layer_summary, methods, summary, triples


def _plot(panels, path, model, operation):
    """One panel-ROW per framing (absolute | ratio), stacked into one figure — the causal
    analogue of the fourier --summary 'one panel per framing' layout. `panels` is a list of
    (framing, per_layer_summary, methods, frac_single)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.2 * n), squeeze=False, sharex=True)
    for i, (framing, per_layer_summary, methods, frac_single) in enumerate(panels):
        L = [r["layer"] for r in per_layer_summary]
        ax_abs, ax_ratio = axes[i][0], axes[i][1]
        for m in methods:
            style = dict(marker="o", ms=3)
            if m == "full_layer":
                style = dict(marker="s", ms=4, lw=2, color="black")
            elif m == "noop":
                style = dict(marker="x", ms=3, ls="--", color="grey")
            ax_abs.plot(L, [r["mean_logit_diff"][m] for r in per_layer_summary],
                        label=m, **style)
            ax_ratio.plot(L, [r["ratio_over_full_layer"][m] for r in per_layer_summary],
                          label=m, **style)
        ax_abs.set_ylabel("mean logit-diff  (a' vs a)")
        ax_abs.set_title(f"{framing} — absolute  (single-token answers: {frac_single:.0%})")
        ax_ratio.axhline(1.0, color="black", lw=0.6, ls=":")
        ax_ratio.set_ylabel("ratio over full-layer patch")
        ax_ratio.set_title(f"{framing} — ratio (1.0 = full-layer ceiling)")
        ax_ratio.legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("layer of intervention")
    fig.suptitle(f"Causal sufficiency of the helix subspace — {model} — {operation}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
