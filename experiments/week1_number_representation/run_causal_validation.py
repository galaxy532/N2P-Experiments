"""Week 1 — causal sufficiency of the helix subspace [engels2024 §5 / kantamneni2025].

Pipeline:
  1. On the addition task, cache the operand-`a` token residual stream at the build
     layer over many a, fit the helix -> get the PCA subspace + helix map C.
  2. For test triples (a, a', b): take the clean prompt "a+b=", and PATCH the operand-a
     site toward a' INSIDE the helix subspace while average-ablating the rest.
     Measure logit_diff = logit[a'+b] - logit[a+b].
  3. Compare to:
        - no-op (should be ~0 / negative, i.e. still says a+b) — the FLOOR,
        - full-layer patch from an "a'+b=" donor run — the sufficiency UPPER BOUND / ceiling,
        - PCA-reconstruction baseline at k=9 (capacity-matched) and k=27 (over-capacity),
          patching the REAL a' activation's PCA-k reconstruction (no helix assumed) — the
          [kantamneni2025] Fig-5 control that isolates the helix FORM from raw capacity.
  Helix family is also split into MAGNITUDE (linear + T=100) vs MODULAR (T=2,5,10) to see
  which part carries the causal effect, alongside the WHOLE helix.
  Subspace-patch ≈ full-layer-patch  =>  the helix subspace is causally sufficient.

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python experiments/week1_number_representation/run_causal_validation.py --model gptj

NOTE: trickiest week-1 script; operand-token indexing and single-token answer
constraints will likely need a first-run debug pass on GPU.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models                       # noqa: E402
from n2p.number_repr import helix                     # noqa: E402
from n2p.number_repr import causal                    # noqa: E402


def operand_a_index(model, a, b):
    """Index of the operand-a token in '{a}+{b}=' (after BOS). Assumes a is one token."""
    toks = model.to_str_tokens(f"{a}+{b}=")
    # token 0 is BOS; the first numeric chunk is a. Find first token containing a digit.
    for i, t in enumerate(toks):
        if any(c.isdigit() for c in t):
            return i
    raise ValueError(f"could not locate operand a in {toks}")


def single_token_answer_id(model, n):
    ids = model.to_tokens(f" {n}", prepend_bos=False)[0]
    return int(ids[0]) if ids.shape[0] == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--layer", type=int, default=None, help="build layer; default = spec mid-band")
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--n_pca", type=int, default=9, help="PCA dims the HELIX is fit inside")
    ap.add_argument("--pca_dims", type=int, nargs="*", default=[9, 27],
                    help="dims for the PCA-reconstruction baseline (kantamneni2025 Fig 5 "
                         "used 9 = capacity-matched to the helix, and 27 = over-capacity).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    spec = config.get_model_spec(args.model)
    model = models.load_model(args.model)
    layer = args.layer if args.layer is not None else (sum(spec.build_layers) // 2 or 16)
    hook = f"blocks.{layer}.hook_resid_post"

    # --- 1. fit helix on operand-a token over many single-token a ---
    fit_as = [a for a in range(0, 200) if single_token_answer_id(model, a) is not None]
    rng.shuffle(fit_as); fit_as = sorted(fit_as[:args.n_fit])
    b_fit = 5
    fit_prompts = [f"{a}+{b_fit}=" for a in fit_as]
    idxs = [operand_a_index(model, a, b_fit) for a in fit_as]
    # cache per-prompt operand-a activation
    acts = np.stack([
        causal.cache_number_site(model, [p], hook, token_index=ix)[0]
        for p, ix in zip(fit_prompts, idxs)
    ], axis=0)
    fitres = helix.fit_helix(acts, np.array(fit_as), n_pca=args.n_pca)
    site_mean = acts.mean(0)                                # (d_model,)

    # --- the patch subspaces we compare ---
    # HELIX patches write the *helix prediction* (a constrained trig functional form) into
    # an orthonormal basis of the helix image (see helix.helix_subspace_basis). The split:
    #   helix_full      : whole helix (linear + all periods)
    #   helix_magnitude : linear + T=100  (low-frequency / magnitude — the separable part
    #                     the stub can replace exactly [zhou2024 Table 1])
    #   helix_modular   : T=2,5,10        (high-frequency / modular residues)
    helix_bases = {
        "helix_full": helix.helix_subspace_basis(fitres),
        "helix_magnitude": helix.helix_subspace_basis(fitres, periods=(100,),
                                                      include_linear=True,
                                                      include_intercept=True),
        "helix_modular": helix.helix_subspace_basis(fitres, periods=(2, 5, 10),
                                                    include_linear=False,
                                                    include_intercept=False),
    }

    # PCA BASELINE (the [kantamneni2025] Fig-5 control). Patches the *real* a' activation's
    # top-k PCA reconstruction — NO helix structure assumed. If the 9-param helix matches a
    # k=9 PCA (capacity-matched) or even k=27 PCA (over-capacity), the periodic FUNCTIONAL
    # FORM, not raw subspace dimensionality, is what is causal.
    from sklearn.decomposition import PCA
    pca_models = {k: PCA(n_components=k).fit(acts) for k in args.pca_dims
                  if k <= min(acts.shape)}

    def noop_logit_diff(clean, ans):
        toks = model.to_tokens(clean)
        last = model(toks)[0, -1]
        return float(last[ans[1]] - last[ans[0]])

    # --- 2/3. patch experiments on held-out triples ---
    methods = list(helix_bases) + [f"pca{k}" for k in pca_models] + ["full_layer", "noop"]
    results = []
    tries = 0
    while len(results) < args.n_test and tries < args.n_test * 20:
        tries += 1
        a, ap, b = rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 9)
        if a == ap:
            continue
        ans_a, ans_ap = single_token_answer_id(model, a + b), single_token_answer_id(model, ap + b)
        if ans_a is None or ans_ap is None:
            continue
        ans = (ans_a, ans_ap)
        clean = f"{a}+{b}="
        donor = f"{ap}+{b}="
        ix = operand_a_index(model, a, b)
        row = {"a": a, "ap": ap, "b": b}
        # helix family: target = helix prediction for a' projected into each basis
        for name, basis in helix_bases.items():
            tgt = helix.helix_target_in_basis([ap], fitres, basis)[0]
            row[name] = causal.subspace_patch_logit_diff(
                model, clean, hook, basis, tgt, site_mean, answer_tokens=ans, token_index=ix)
        # PCA baseline: target = the real a' operand activation's PCA-k coords
        if pca_models:
            donor_ix = operand_a_index(model, ap, b)
            donor_act = causal.cache_number_site(model, [donor], hook,
                                                 token_index=donor_ix)[0]   # (d_model,)
            for k, pca_k in pca_models.items():
                tgt_k = pca_k.transform(donor_act[None, :])[0]              # (k,)
                row[f"pca{k}"] = causal.subspace_patch_logit_diff(
                    model, clean, hook, pca_k.components_.T, tgt_k, site_mean,
                    answer_tokens=ans, token_index=ix)
        row["full_layer"] = causal.full_layer_patch_logit_diff(
            model, clean, donor, hook, answer_tokens=ans, token_index=ix)
        row["noop"] = noop_logit_diff(clean, ans)
        results.append(row)

    means = {m: float(np.mean([r[m] for r in results])) for m in methods}
    full_mean = means["full_layer"]
    ratios = {m: float(means[m] / full_mean) if abs(full_mean) > 1e-6 else float("nan")
              for m in methods}
    summary = {
        "model": args.model, "layer": layer, "helix_fit_r2": fitres["r2"],
        "n_pca": args.n_pca, "pca_dims": list(pca_models), "n_test": len(results),
        "helix_ranks": {n: int(b.shape[1]) for n, b in helix_bases.items()},
        "mean_logit_diff": means,
        "ratio_over_full_layer": ratios,
        "interpretation": (
            "full_layer = causal ceiling; noop ~0/negative = floor. helix_full/full near "
            "1.0 => the helix is causally sufficient. helix_full >> pca9/pca27 (or matching "
            "them with far fewer effective dims) => the periodic FORM, not generic PCA "
            "capacity, carries the number. magnitude vs modular shows which part of the "
            "helix is doing the causal work."),
    }
    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label="run_causal_validation/addition",
                         meta={"script": "run_causal_validation.py", "context": "addition"})
    (out / "causal_validation.json").write_text(
        json.dumps({"summary": summary, "trials": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
