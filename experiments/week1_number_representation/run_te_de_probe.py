"""Week 1 — total/direct-effect (TE/DE) write-site localization probe, SWEPT PER LAYER with
a direction-restricted (helix) variant and a decomposed logit-diff metric.

The experiment proposed in `../../../wiki/notes/approach-decision-circuit-identification.md`
("Validation layer: total/direct-effect (TE/DE) write-site localization"). It is NOT a
discovery method (that is `week1_circuit_sanity/run_discovery_sanity.py`, Edge Pruning); it
is a *role-labeling* pass over last-token components, reproducing [kantamneni2025] Fig 6 on
our model and testing whether the DIRECT effect (path patching) localizes the answer
write-site — the place the N2P stub injects `helix(a+b)`.

Setup (denoising): clean `a+b=`, corrupt `a'+b=` (same b). For each last-token MLP-out and
attn-out at every swept layer L, measure
  TE = activation patch sender->clean, downstream recomputes  (causal.total_effect)
  DE = path patch sender->clean, downstream FROZEN to corrupt (causal.direct_effect)
averaged over (a, a', b) triples, with the [kantamneni2025] logit diff
  LD = logit[a+b] - logit[a'+b]   (positive = the sender restores the clean answer).

Two SENDER variants per layer (this is the per-layer generalization of the old single-anchor
helix check — 2026-06-27, user-approved Option 2; NO build_layers prior is used anywhere):
  - full-node     : swap the WHOLE component output  (sender_basis=None).
  - helix-direction: fit `helix(a+b)` on the answer token AT THAT LAYER, and swap only the
    helix subspace, writing the *fitted* helix(a+b) (ANALYTIC target, keeping the orthogonal
    complement). If helix-direction DE tracks full-node DE at the write band, the write-site
    direct effect lives in the helix direction — N2P "replace along a direction", per layer.

Decomposed metric (also user-approved, exploratory): for the TE only, additionally report
  Δlogit[a+b]    = logit[a+b]_patched   - logit[a+b]_corrupt   (raised the clean answer)
  -Δlogit[a'+b]  = logit[a'+b]_corrupt  - logit[a'+b]_patched  (suppressed the corrupt answer)
both oriented positive-when-restoring; they SUM to LD_patched - LD_corrupt, splitting the
TE's effect on the logit-diff into "raise clean" vs "suppress corrupt".

Plots: two PNGs (MLP and Attn), each with ONE PANEL PER FRAMING (symbolic / word /
wordproblem, like the fourier --summary), and 8 curves per panel:
  full/helix x TE/DE (LD)  +  full/helix x TE (Δ[a+b])  +  full/helix x TE (-Δ[a'+b]).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_te_de_probe.py --model gptj
    python3 experiments/week1_number_representation/run_te_de_probe.py --model gptj --layers 8 27   # restrict the sweep band (cheaper)
    python3 experiments/week1_number_representation/run_te_de_probe.py --model gptj --framing symbolic  # one framing only

COST: DE path-patching freezes every component, and we now run full+helix x TE+DE x {MLP,attn}
per layer per triple (~8 patched forwards each) over all layers and all framings. Use
--layers / --framing / --n_test to bound it. GPU-only; first run will want a debug pass.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks                  # noqa: E402
from n2p.number_repr import helix, causal, repcli       # noqa: E402


# Model-agnostic single-token answer id. Prompts are zero-shot "{a}+{b}=" (no trailing
# space), so read the BARE answer token (space=False) — GPT-J emits bare '99' there, Llama
# identical. Matches [kantamneni2025]; see models.first_answer_token_id.
def single_token_answer_id(model, n):
    return models.single_token_answer_id(model, n, space=False)


def helix_full_vector(sum_value, fitres) -> np.ndarray:
    """The fitted helix's FULL (absolute) residual-stream reconstruction for one sum value:
    mu + helix_coords(sum) @ P, shape (d_model,). This is the analytic target written into
    the helix subspace by causal._swap_value (it projects onto the basis internally)."""
    P = fitres["pca"].components_                       # (n_pca, d_model)
    mu = fitres["mu"][0]                                # (d_model,)
    coords = helix.helix_coords([sum_value], fitres)[0]  # (n_pca,)
    return mu + coords @ P                              # (d_model,)


# 8 plotted series: (metric_key, label, matplotlib style). Colour encodes node-variant
# (full=C0/C2, helix=C1/C3); linestyle encodes the metric family (solid/dashed = LD TE/DE,
# dotted = Δ[a+b], dash-dot = -Δ[a'+b]).
NODE_VARIANTS = ("full", "helix")
METRICS = ["full_te", "full_de", "helix_te", "helix_de",
           "full_te_draise", "helix_te_draise", "full_te_dsuppress", "helix_te_dsuppress"]
METRIC_STYLE = [
    ("full_te",            "full-node TE (LD)",       dict(color="C0", ls="-",  marker="o", ms=3)),
    ("full_de",            "full-node DE (LD)",       dict(color="C0", ls="--", marker="o", ms=3)),
    ("helix_te",           "helix-dir TE (LD)",       dict(color="C1", ls="-",  marker="s", ms=3)),
    ("helix_de",           "helix-dir DE (LD)",       dict(color="C1", ls="--", marker="s", ms=3)),
    ("full_te_draise",     "full-node TE Δ[a+b]",     dict(color="C2", ls=":",  marker="^", ms=3)),
    ("helix_te_draise",    "helix-dir TE Δ[a+b]",     dict(color="C3", ls=":",  marker="^", ms=3)),
    ("full_te_dsuppress",  "full-node TE −Δ[a'+b]",   dict(color="C2", ls="-.", marker="v", ms=3)),
    ("helix_te_dsuppress", "helix-dir TE −Δ[a'+b]",   dict(color="C3", ls="-.", marker="v", ms=3)),
]


def resolve_sweep(args, spec):
    lo, hi = (args.layers if args.layers is not None else (0, spec.n_layers - 1))
    if not (0 <= lo <= hi <= spec.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{spec.n_layers - 1}]")
    return list(range(lo, hi + 1))


def sample_triples(model, task, test_pool, b_range, n_test, rng):
    """(a, a', b) triples with single-token, distinct clean & corrupt answers. Framing-
    independent (answers depend only on a,b), so sampled once and reused across framings."""
    b_lo, b_hi = b_range
    triples, tries = [], 0
    while len(triples) < n_test and tries < n_test * 60:
        tries += 1
        a, ap_ = rng.choice(test_pool), rng.choice(test_pool)
        b = rng.randint(b_lo, b_hi)
        if a == ap_:
            continue
        ans_clean, ans_corr = task.fn(a, b), task.fn(ap_, b)
        if ans_clean < 0 or ans_corr < 0:
            continue
        clean_id = single_token_answer_id(model, ans_clean)
        corr_id = single_token_answer_id(model, ans_corr)
        if clean_id is None or corr_id is None or clean_id == corr_id:
            continue
        triples.append({"a": a, "ap": ap_, "b": b, "ans_clean": ans_clean,
                        "ans_corr": ans_corr, "clean_id": clean_id, "corr_id": corr_id})
    return triples


def fit_helix_all_layers(model, operation, framing, fit_values, b_fixed, sweep, n_pca,
                         prefix, shots):
    """One cache sweep over the fit prompts; fit helix(a+b) on the ANSWER token at every
    swept layer. Returns {L: {"fitres":..., "basis": U_L (d_model,r), "r2":...}}."""
    prompts = tasks.build_prompts(operation, framing, fit_values, b_fixed, prefix=prefix,
                                  shots=shots)
    sums = np.asarray(fit_values, dtype=np.float64) + b_fixed
    hooks = [f"blocks.{L}.hook_resid_post" for L in sweep]
    acts_by_hook = causal.cache_number_site_all_layers(model, prompts, hooks, token_index=-1)
    per_layer = {}
    for L in sweep:
        fitres = helix.fit_helix(acts_by_hook[f"blocks.{L}.hook_resid_post"], sums,
                                 n_pca=n_pca)
        per_layer[L] = {"fitres": fitres, "basis": helix.helix_subspace_basis(fitres),
                        "r2": float(fitres["r2"])}
    return per_layer


def run_one_framing(model, operation, framing, fit_values, b_fixed, sweep, triples,
                    n_pca, prefix, shots):
    """Per-layer full-node & helix-direction TE/DE (+ TE decomposition) for one framing.
    Returns ({"mlp": per_layer_summary, "attn": per_layer_summary}, {L: helix_r2})."""
    fits = fit_helix_all_layers(model, operation, framing, fit_values, b_fixed, sweep,
                                n_pca, prefix, shots)
    nodes = {"mlp": "hook_mlp_out", "attn": "hook_attn_out"}
    acc = {node: {L: {m: [] for m in METRICS} for L in sweep} for node in nodes}

    for t in triples:
        a, ap_, b = t["a"], t["ap"], t["b"]
        clean = tasks.build_prompt(operation, framing, a, b, prefix=prefix, shots=shots)
        corrupt = tasks.build_prompt(operation, framing, ap_, b, prefix=prefix, shots=shots)
        ans = (t["corr_id"], t["clean_id"])            # (a_id, ap_id) = (corrupt, clean)
        # unpatched-corrupt baseline for the Δ decomposition
        last_c = model(model.to_tokens(corrupt))[0, -1]
        bl_clean, bl_corr = float(last_c[t["clean_id"]]), float(last_c[t["corr_id"]])
        for L in sweep:
            atgt = helix_full_vector(a + b, fits[L]["fitres"])   # analytic helix(a+b)
            basis = fits[L]["basis"]
            for node, hk in nodes.items():
                hook = f"blocks.{L}.{hk}"
                te_full = causal.total_effect_logit_diff(
                    model, clean, corrupt, hook, ans, token_index=-1, return_logits=True)
                de_full = causal.direct_effect_logit_diff(
                    model, clean, corrupt, hook, ans, token_index=-1)
                te_hel = causal.total_effect_logit_diff(
                    model, clean, corrupt, hook, ans, token_index=-1,
                    sender_basis=basis, analytic_target=atgt, return_logits=True)
                de_hel = causal.direct_effect_logit_diff(
                    model, clean, corrupt, hook, ans, token_index=-1,
                    sender_basis=basis, analytic_target=atgt)
                d = acc[node][L]
                d["full_te"].append(te_full["ld"]);   d["full_de"].append(de_full)
                d["helix_te"].append(te_hel["ld"]);   d["helix_de"].append(de_hel)
                d["full_te_draise"].append(te_full["logit_clean"] - bl_clean)
                d["full_te_dsuppress"].append(bl_corr - te_full["logit_corrupt"])
                d["helix_te_draise"].append(te_hel["logit_clean"] - bl_clean)
                d["helix_te_dsuppress"].append(bl_corr - te_hel["logit_corrupt"])

    def mean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    out = {}
    for node in nodes:
        out[node] = [{"layer": L, "helix_r2": fits[L]["r2"],
                      "metrics": {m: mean(acc[node][L][m]) for m in METRICS}}
                     for L in sweep]
    return out, {L: fits[L]["r2"] for L in sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--operation", choices=tasks.OPERATION_CHOICES, default="addition",
                    help="operation whose write-site is probed (interpretation is addition-"
                         "centric — the Clock; other ops are exploratory).")
    ap.add_argument("--framing", choices=tasks.FRAMING_NAMES, default=None,
                    help="restrict to a single framing; default = ALL framings, one panel "
                         "each (like the fourier --summary).")
    ap.add_argument("--layers", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="restrict the TE/DE layer sweep to [LO,HI]. Default = ALL layers "
                         "(0..last); no build_layers prior is assumed.")
    ap.add_argument("--n_fit", type=int, default=150, help="prompts for the per-layer helix fit")
    ap.add_argument("--n_test", type=int, default=15, help="(a,a',b) triples averaged over")
    ap.add_argument("--n_pca", type=int, default=9)
    ap.add_argument("--b_fixed", type=int, default=5)
    ap.add_argument("--prefix", default=None,
                    help="model instruction prefix prepended to every prompt; default = "
                         "config ModelSpec.prompt_prefix for --model. Pass '' to ablate.")
    ap.add_argument("--kshot", type=int, default=0,
                    help="few-shot solved examples before each query (0 = zero-shot).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    spec = config.get_model_spec(args.model)
    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else spec.prompt_prefix
    shots = tasks.fewshot_shots(args.operation, args.kshot, args.seed)
    operation = args.operation
    sweep = resolve_sweep(args, spec)
    framings = [args.framing] if args.framing else repcli.framings_for_summary(operation, "sum")
    if not framings:
        raise SystemExit(f"no framing available for operation {operation!r}")
    print(f"[prefix] {prefix!r}  [kshot] {args.kshot}")
    print(f"[setup] {operation}; framings={framings}; sweep L{sweep[0]}..{sweep[-1]} "
          f"({len(sweep)} layers) x ~{args.n_test} triples x 8 forwards -> "
          f"~{len(framings) * len(sweep) * args.n_test * 8} patched forwards total. "
          f"Restrict with --framing / --layers / --n_test if slow.")

    # canonical single-token operand pool (symbolic grid) for the fit and the triples; the
    # operand token is a single number token in every framing, so this pool is shared.
    grid_values, _ = tasks.single_token_number_grid(model, operation, "symbolic", 0, 99,
                                                    b=args.b_fixed)
    fit_values = repcli.contiguous_prefix(np.array(grid_values))
    if args.n_fit and len(fit_values) > args.n_fit:
        fit_values = fit_values[:args.n_fit]
    task = tasks.get_task(operation)
    a_lo, a_hi = task.a_range
    test_pool = [int(v) for v in fit_values if a_lo <= v <= a_hi]
    triples = sample_triples(model, task, test_pool, task.b_range, args.n_test, rng)
    if not triples:
        raise SystemExit("no valid single-token-answer triples sampled; check operation/range")

    out = config.run_dir("week1_number_representation", args.seed, model=args.model,
                         label=f"run_te_de_probe/{operation}",
                         meta={"script": "run_te_de_probe.py", "operation": operation,
                               "framing": framings[0] if len(framings) == 1 else None,
                               "framings": framings, "read_token": "sum",
                               "sweep_layers": [sweep[0], sweep[-1]], "prefix": prefix,
                               "kshot": args.kshot})

    mlp_panels, attn_panels = [], []
    for framing in framings:
        per_node, r2 = run_one_framing(model, operation, framing, fit_values, args.b_fixed,
                                       sweep, triples, args.n_pca, prefix, shots)
        mlp_panels.append((framing, per_node["mlp"]))
        attn_panels.append((framing, per_node["attn"]))
        (out / f"te_de.{framing}.json").write_text(json.dumps({
            "model": args.model, "hf_id": spec.hf_id, "operation": operation,
            "framing": framing, "read_token": "sum", "sweep_layers": [sweep[0], sweep[-1]],
            "prefix": prefix, "kshot": args.kshot, "n_test": len(triples),
            "helix_r2_by_layer": {int(L): v for L, v in r2.items()},
            "mlp": per_node["mlp"], "attn": per_node["attn"],
            "metric_legend": {
                "*_te/_de": "LD = logit[a+b]-logit[a'+b] under TE / DE, full-node vs helix-direction",
                "*_te_draise": "Δlogit[a+b] = logit[a+b]_patched - logit[a+b]_corrupt (raise clean)",
                "*_te_dsuppress": "−Δlogit[a'+b] = logit[a'+b]_corrupt - logit[a'+b]_patched (suppress corrupt)",
                "note": "draise + dsuppress = LD_patched - LD_corrupt (TE)"},
            "trials": triples,
        }, indent=2))
        print(f"[{framing}] done; helix R² L{sweep[0]}={r2[sweep[0]]:.2f} "
              f"L{sweep[-1]}={r2[sweep[-1]]:.2f}")

    _plot_node("MLP", mlp_panels, out / "te_de_summary_MLP.png", args.model, operation, len(triples))
    _plot_node("Attn", attn_panels, out / "te_de_summary_Attn.png", args.model, operation, len(triples))
    print(f"[done] wrote {out} ({len(framings)} framing(s); MLP + Attn summaries)")


def _plot_node(node_name, panels, path, model, operation, n_test):
    """One PNG per node; one panel-row per framing; 8 curves/panel (see METRIC_STYLE)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.6 * n), squeeze=False, sharex=True)
    for i, (framing, per_layer) in enumerate(panels):
        ax = axes[i][0]
        L = [r["layer"] for r in per_layer]
        for key, label, style in METRIC_STYLE:
            ax.plot(L, [r["metrics"][key] for r in per_layer], label=label, **style)
        ax.axhline(0.0, color="black", lw=0.6, ls=":")
        ax.set_ylabel(f"{framing}\nlogit units")
        if i == 0:
            ax.set_title(f"{node_name}-output TE/DE by layer — {model} — {operation} "
                         f"(answer token; n_test={n_test})  [kantamneni2025 Fig 6]")
        if i == n - 1:
            ax.set_xlabel("layer of intervention")
    axes[0][0].legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
