"""Shared DENOISING subspace-sufficiency sweep for the causal-validation scripts
(run_causal_validation.py on resid_post; run_causal_validation_components.py on
hook_mlp_out / hook_attn_out).

Direction (2026-06-27, user-approved): this follows the [kantamneni2025] §4.4 / Fig-5
DENOISING test, NOT the [engels2024] Eq 5-6 noising+average-ablate-rest interchange the
original resid script used. The base run is the CORRUPT prompt `a'+b`; we INJECT the clean
value into a subspace at the chosen site/token and let the network recompute (a total-effect
patch, causal.total_effect_logit_diff), then read how much the clean answer is restored.
Because we inject clean into a corrupt run, there is NO clean signal in the orthogonal
complement to leak — so NO average-ablation is needed (that device existed only to plug the
within-stream leak of the noising-on-clean direction).

Per layer & site, the methods (full parity with the old resid script):
  noop          : corrupt run, no patch (the FLOOR; says a', so logit-diff is ~0/negative).
  full          : inject the WHOLE clean site activation (the per-site sufficiency CEILING).
  helix_full    : inject the fitted clean helix into its subspace (analytic target).
  helix_magnitude / helix_modular : the separable low-freq / high-freq helix sub-parts.
  pca9 / pca27  : inject the clean activation's top-k PCA projection (capacity-matched /
                  over-capacity baseline, [kantamneni2025] Fig 5) — empirical swap.
Metric: contrast LD = logit[ans(a)] - logit[ans(a')] on the patched run (the clean-minus-
corrupt answer logit-diff; ratio = method / full). Kantamneni's literal §4.4 metric is the
raise-only logit_patched(a+b)-logit_corrupt(a+b); the contrast adds the suppress-corrupt half
and keeps the floor/ceiling framing.

read_token selects WHAT the helix encodes and WHERE we patch:
  "a"   -> operand-a token; helix(a)      (operand representation; matches run_causal_validation
                                           and run_fourier_components_raw)
  "sum" -> answer/last token; helix(a+b)  (answer representation; matches run_fourier_components)
Answer logits are always read at the last position.

Needs a live HookedTransformer (real forward passes); kept out of the scripts so both share
one implementation (lint 2026-06-27).
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.decomposition import PCA

from n2p import models, tasks
from n2p.number_repr import helix, causal, repcli

HELIX_KEYS = ["helix_full", "helix_magnitude", "helix_modular"]


def resolve_layers(layers, layer, n_layers) -> list[int]:
    """--layers wins; else single --layer; else ALL layers 0..n_layers-1 (no build_layers
    prior is assumed — 2026-06-27)."""
    if layers:
        return sorted({L for L in layers if 0 <= L < n_layers})
    if layer is not None:
        return [layer]
    return list(range(n_layers))


def value_of(a, b, read_token):
    """The integer the helix encodes at the read token: the operand a, or the sum a+b."""
    return a if read_token == "a" else a + b


def helix_full_vector(value, fitres) -> np.ndarray:
    """Absolute residual-stream reconstruction of the fitted helix for one value:
    mu + helix_coords(value) @ P, shape (d_model,). causal._swap_value projects this onto
    the requested basis, so the magnitude/modular sub-bases pick out their own part."""
    P = fitres["pca"].components_                       # (n_pca, d_model)
    mu = fitres["mu"][0]                                # (d_model,)
    coords = helix.helix_coords([value], fitres)[0]      # (n_pca,)
    return mu + coords @ P                              # (d_model,)


def _fit_sites(model, operation, framing, fit_values, b_fixed, sites, layers, read_token,
               n_pca, pca_dims, prefix, shots):
    """One cache sweep over the fit prompts; fit helix + PCA baselines at every (site, layer)
    on the read token. `sites` are hook suffixes, e.g. 'hook_resid_post' / 'hook_mlp_out'.
    Returns ({site: {L: {fitres, bases, pca, r2}}}, read_token_index)."""
    prompts = tasks.build_prompts(operation, framing, fit_values, b_fixed, prefix=prefix,
                                  shots=shots)
    tix = tasks.read_token_index(model, prompts[0], read_token, operation, framing)
    targets = np.array([value_of(int(v), b_fixed, read_token) for v in fit_values],
                       dtype=np.float64)
    hooks = [f"blocks.{L}.{s}" for L in layers for s in sites]
    acts_by_hook = causal.cache_number_site_all_layers(model, prompts, hooks, token_index=tix)
    per = {s: {} for s in sites}
    for s in sites:
        for L in layers:
            acts = acts_by_hook[f"blocks.{L}.{s}"]
            fitres = helix.fit_helix(acts, targets, n_pca=n_pca)
            bases = {
                "helix_full": helix.helix_subspace_basis(fitres),
                "helix_magnitude": helix.helix_subspace_basis(
                    fitres, periods=(100,), include_linear=True, include_intercept=True),
                "helix_modular": helix.helix_subspace_basis(
                    fitres, periods=(2, 5, 10), include_linear=False, include_intercept=False),
            }
            pcas = {k: PCA(n_components=k).fit(acts) for k in pca_dims if k <= min(acts.shape)}
            per[s][L] = {"fitres": fitres, "bases": bases, "pca": pcas, "r2": float(fitres["r2"])}
    return per, tix


def sweep_framing(model, operation, framing, *, sites, read_token, layers, task, n_fit,
                  n_pca, pca_dims, n_test, b_fixed, prefix, shots, seed):
    """Full denoising sufficiency sweep for ONE framing across `sites`. Returns a dict:
    {per_site:{site: per_layer_summary}, r2:{site:{L:r2}}, helix_ranks:{site:{name:rank}},
     methods:[...], triples:[...], frac_single:float}."""
    rng = random.Random(seed)

    grid_values, _ = tasks.single_token_number_grid(model, operation, framing, 0, 99,
                                                    b=b_fixed)
    fit_values = repcli.contiguous_prefix(np.array(grid_values))
    if n_fit and len(fit_values) > n_fit:
        fit_values = fit_values[:n_fit]

    per, tix = _fit_sites(model, operation, framing, fit_values, b_fixed, sites, layers,
                          read_token, n_pca, pca_dims, prefix, shots)
    pca_keys = sorted({k for s in sites for L in layers for k in per[s][L]["pca"]})
    methods = HELIX_KEYS + [f"pca{k}" for k in pca_keys] + ["full", "noop"]

    a_lo, a_hi = task.a_range
    b_lo, b_hi = task.b_range
    test_pool = [int(v) for v in fit_values if a_lo <= v <= a_hi]

    diffs = {s: {L: {m: [] for m in methods} for L in layers} for s in sites}
    triples, tries = [], 0
    while len(triples) < n_test and tries < n_test * 80:
        tries += 1
        a, ap_ = rng.choice(test_pool), rng.choice(test_pool)
        b = rng.randint(b_lo, b_hi)
        if a == ap_:
            continue
        ans_a, ans_ap = task.fn(a, b), task.fn(ap_, b)
        if ans_a < 0 or ans_ap < 0:
            continue
        clean_id = models.first_answer_token_id(model, ans_a, space=False)
        corr_id = models.first_answer_token_id(model, ans_ap, space=False)
        if clean_id == corr_id:                          # logit-diff ~0 by construction
            continue
        clean = tasks.build_prompt(operation, framing, a, b, prefix=prefix, shots=shots)
        corrupt = tasks.build_prompt(operation, framing, ap_, b, prefix=prefix, shots=shots)
        ans = (corr_id, clean_id)                        # (a_id, ap_id) = (corrupt, clean)
        # FLOOR: unpatched corrupt run (says a'), clean-minus-corrupt logit-diff (~0/negative)
        last_c = model(model.to_tokens(corrupt))[0, -1]
        noop = float(last_c[clean_id] - last_c[corr_id])
        cval = value_of(a, b, read_token)                # clean value to inject
        for s in sites:
            for L in layers:
                pl = per[s][L]
                hook = f"blocks.{L}.{s}"
                for name in HELIX_KEYS:
                    diffs[s][L][name].append(causal.total_effect_logit_diff(
                        model, clean, corrupt, hook, ans, token_index=tix,
                        sender_basis=pl["bases"][name],
                        analytic_target=helix_full_vector(cval, pl["fitres"])))
                for k in pca_keys:
                    if k not in pl["pca"]:
                        diffs[s][L][f"pca{k}"].append(float("nan"))
                        continue
                    diffs[s][L][f"pca{k}"].append(causal.total_effect_logit_diff(
                        model, clean, corrupt, hook, ans, token_index=tix,
                        sender_basis=pl["pca"][k].components_.T, analytic_target=None))
                diffs[s][L]["full"].append(causal.total_effect_logit_diff(
                    model, clean, corrupt, hook, ans, token_index=tix, sender_basis=None))
                diffs[s][L]["noop"].append(noop)
        triples.append({"a": a, "ap": ap_, "b": b, "ans_a": ans_a, "ans_ap": ans_ap})

    def nanmean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    per_site = {}
    for s in sites:
        summ = []
        for L in layers:
            means = {m: nanmean(diffs[s][L][m]) for m in methods}
            full = means["full"]
            ratios = {m: (means[m] / full if abs(full) > 1e-6 else float("nan"))
                      for m in methods}
            summ.append({"layer": L, "helix_fit_r2": per[s][L]["r2"],
                         "mean_logit_diff": means, "ratio_over_full": ratios})
        per_site[s] = summ

    n_single = sum(models.is_single_token_answer(model, t["ans_a"], space=False)
                   for t in triples)
    return {
        "per_site": per_site,
        "r2": {s: {L: per[s][L]["r2"] for L in layers} for s in sites},
        "helix_ranks": {s: {n: int(b.shape[1])
                            for n, b in per[s][layers[0]]["bases"].items()} for s in sites},
        "methods": methods, "triples": triples,
        "frac_single": round(n_single / max(len(triples), 1), 3),
    }


def plot_sufficiency(panels, path, suptitle, ceiling_label="full-layer"):
    """One figure: one panel-ROW per framing x 2 cols (absolute logit-diff | ratio-to-ceiling).
    `panels` = list of (framing, per_layer_summary, methods, frac_single)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.2 * n), squeeze=False, sharex=True)
    for i, (framing, pls, methods, frac) in enumerate(panels):
        L = [r["layer"] for r in pls]
        ax_abs, ax_ratio = axes[i][0], axes[i][1]
        for m in methods:
            style = dict(marker="o", ms=3)
            if m == "full":
                style = dict(marker="s", ms=4, lw=2, color="black")
            elif m == "noop":
                style = dict(marker="x", ms=3, ls="--", color="grey")
            label = ceiling_label if m == "full" else m
            ax_abs.plot(L, [r["mean_logit_diff"][m] for r in pls], label=label, **style)
            ax_ratio.plot(L, [r["ratio_over_full"][m] for r in pls], label=label, **style)
        ax_abs.set_ylabel(f"{framing}\nmean logit-diff (a vs a')")
        ax_ratio.axhline(1.0, color="black", lw=0.6, ls=":")
        ax_ratio.set_ylabel(f"ratio over {ceiling_label}")
        if i == 0:
            ax_abs.set_title(f"absolute restoration  (single-tok {frac:.0%})")
            ax_ratio.set_title(f"ratio (1.0 = {ceiling_label} ceiling)")
    for ax in axes[-1]:
        ax.set_xlabel("layer of intervention")
    axes[0][1].legend(fontsize=7, ncol=2)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
