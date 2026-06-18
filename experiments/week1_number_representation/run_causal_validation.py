"""Week 1 — causal sufficiency of the helix subspace [engels2024 §5 / kantamneni2025].

Pipeline:
  1. On the addition task, cache the operand-`a` token residual stream at the build
     layer over many a, fit the helix -> get the PCA subspace + helix map C.
  2. For test triples (a, a', b): take the clean prompt "a+b=", and PATCH the operand-a
     site toward a' INSIDE the helix subspace while average-ablating the rest.
     Measure logit_diff = logit[a'+b] - logit[a+b].
  3. Compare to:
        - no-op (should be ~0 / negative, i.e. still says a+b),
        - full-layer patch from an "a'+b=" donor run (the sufficiency UPPER BOUND).
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
    ap.add_argument("--n_pca", type=int, default=9)
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
    pca = fitres["pca"]
    components = pca.components_           # (n_pca, d_model), orthonormal rows
    subspace_basis = components.T          # (d_model, n_pca)
    site_mean = acts.mean(0)               # (d_model,)

    def helix_target_in_subspace(ap_val):
        return helix.helix_coords([ap_val], fitres)[0]      # (n_pca,)

    # --- 2/3. patch experiments on held-out triples ---
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
        clean = f"{a}+{b}="
        donor = f"{ap}+{b}="
        ix = operand_a_index(model, a, b)
        tgt = helix_target_in_subspace(ap)
        sub = causal.subspace_patch_logit_diff(
            model, clean, hook, subspace_basis, tgt, site_mean,
            answer_tokens=(ans_a, ans_ap), token_index=ix)
        full = causal.full_layer_patch_logit_diff(
            model, clean, donor, hook, answer_tokens=(ans_a, ans_ap), token_index=ix)
        results.append({"a": a, "ap": ap, "b": b,
                        "subspace_logit_diff": sub, "full_layer_logit_diff": full})

    arr_sub = np.array([r["subspace_logit_diff"] for r in results])
    arr_full = np.array([r["full_layer_logit_diff"] for r in results])
    summary = {
        "model": args.model, "layer": layer, "helix_fit_r2": fitres["r2"],
        "n_test": len(results),
        "mean_subspace_logit_diff": float(arr_sub.mean()),
        "mean_full_layer_logit_diff": float(arr_full.mean()),
        "subspace_over_full_ratio": float(arr_sub.mean() / max(arr_full.mean(), 1e-6)),
        "interpretation": "ratio near 1.0 => helix subspace is causally sufficient",
    }
    out = config.run_dir("week1_number_representation", args.seed)
    (out / "causal_validation.json").write_text(
        json.dumps({"summary": summary, "trials": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
