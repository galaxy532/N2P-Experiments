"""Week 1 — total/direct-effect (TE/DE) write-site localization probe.

The experiment proposed in `../../../wiki/notes/approach-decision-circuit-identification.md`
("Validation layer: total/direct-effect (TE/DE) write-site localization"). It is NOT a
discovery method (that is `week1_circuit_sanity/run_discovery_sanity.py`, Edge Pruning); it
is a *role-labeling* pass over last-token components, reproducing [kantamneni2025] Fig 6 on
our model and testing whether the DIRECT effect (path patching) localizes the answer
write-site — the place the N2P stub injects `helix(a+b)`.

Setup (denoising): clean `a+b=`, corrupt `a'+b=` (same b). For each last-token MLP-out and
attn-out, measure
  TE = activation patch sender->clean, downstream recomputes  (causal.total_effect)
  DE = path patch sender->clean, downstream FROZEN to corrupt (causal.direct_effect)
  IE = TE - DE
averaged over (a, a', b) triples, with logit diff = logit[a+b] - logit[a'+b] (positive =
the sender restores the clean answer). Pass signal (Fig 6): MLPs dominate DE (write), early
attention dominates IE (routing); cumulative MLP DE saturates at the write-site band.

It also runs a DIRECTION-RESTRICTED variant at the build layer: restrict the sender swap to
the **answer-helix subspace** (fit on the last-token resid_post over a+b) via
`sender_basis`. If the helix-direction DE retains most of the full-node DE, the write-site
is carried by the helix direction — the N2P "replace along a direction" claim, made
measurable (approach-decision-circuit-identification.md, granularity adaptation).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_te_de_probe.py --model gptj
    python3 experiments/week1_number_representation/run_te_de_probe.py --model gptj --layers 12 27   # focus on the build/read band (cheaper)

COST: DE path-patching freezes every component, so this is ~2*n_layers*n_test patched
forwards — use --layers to restrict the band and keep --n_test modest. GPU-only (real
forward passes); first run will want a debug pass like the rest of week 1.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models                        # noqa: E402
from n2p.number_repr import helix, causal              # noqa: E402


def operand_a_index(model, prompt):
    """First digit-bearing token in the prompt (= operand a; b/constants come later)."""
    for i, t in enumerate(model.to_str_tokens(prompt)):
        if any(c.isdigit() for c in t):
            return i
    raise ValueError(f"could not locate operand a in {model.to_str_tokens(prompt)}")


# Model-agnostic single-token answer id (handles tokenizers that split the leading space,
# e.g. Llama-3; the old ' {n}'.shape==1 check returned None for every answer there).
single_token_answer_id = models.single_token_answer_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--layer", type=int, default=None,
                    help="build layer for the answer-helix direction basis; default = spec mid-band")
    ap.add_argument("--layers", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="restrict the TE/DE layer sweep to [LO,HI] (default: all layers).")
    ap.add_argument("--n_fit", type=int, default=150, help="prompts for the helix-basis fit")
    ap.add_argument("--n_test", type=int, default=15, help="(a,a',b) triples averaged over")
    ap.add_argument("--n_pca", type=int, default=9)
    ap.add_argument("--b_fixed", type=int, default=5)
    ap.add_argument("--prefix", default=None,
                    help="model instruction prefix prepended to every prompt; default = "
                         "config ModelSpec.prompt_prefix for --model. Pass '' to ablate.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    spec = config.get_model_spec(args.model)
    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else spec.prompt_prefix
    print(f"[prefix] {prefix!r}")
    build_layer = args.layer if args.layer is not None else (sum(spec.build_layers) // 2 or 16)
    lo, hi = (args.layers if args.layers is not None else (0, spec.n_layers - 1))
    if not (0 <= lo <= hi <= spec.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{spec.n_layers - 1}]")
    sweep = list(range(lo, hi + 1))

    # --- answer-helix direction basis: fit helix(a+b) on the LAST token at the build layer.
    # b fixed, a swept -> the sum a+b varies; helix is fit over the sum.
    build_hook = f"blocks.{build_layer}.hook_resid_post"
    fit_as = [a for a in range(0, 200)
              if single_token_answer_id(model, a + args.b_fixed) is not None]
    rng.shuffle(fit_as); fit_as = sorted(fit_as[:args.n_fit])
    sums = np.array([a + args.b_fixed for a in fit_as])
    fit_prompts = [f"{prefix}{a}+{args.b_fixed}=" for a in fit_as]
    acts = np.stack([causal.cache_number_site(model, [p], build_hook, token_index=-1)[0]
                     for p in fit_prompts], axis=0)
    fitres = helix.fit_helix(acts, sums, n_pca=args.n_pca)
    helix_basis = helix.helix_subspace_basis(fitres)   # answer-helix directions (d_model, r)

    # --- test triples (a, a', b) with single-token clean & corrupt answers ---
    triples = []
    tries = 0
    while len(triples) < args.n_test and tries < args.n_test * 30:
        tries += 1
        a, ap_, b = rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 9)
        if a == ap_:
            continue
        ans_clean = single_token_answer_id(model, a + b)
        ans_corr = single_token_answer_id(model, ap_ + b)
        if ans_clean is None or ans_corr is None:
            continue
        triples.append((a, ap_, b, ans_corr, ans_clean))

    # --- per-layer TE/DE for MLP-out and attn-out at the answer (last) token ---
    def mean_effects(hook_name, sender_basis=None):
        te = de = ie = 0.0
        for (a, ap_, b, ans_corr, ans_clean) in triples:
            clean, corrupt = f"{prefix}{a}+{b}=", f"{prefix}{ap_}+{b}="
            r = causal.te_de_ie(model, clean, corrupt, hook_name,
                                answer_tokens=(ans_corr, ans_clean), token_index=-1,
                                sender_basis=sender_basis)
            te += r["te"]; de += r["de"]; ie += r["ie"]
        n = max(len(triples), 1)
        return {"te": te / n, "de": de / n, "ie": ie / n}

    per_layer = []
    for L in sweep:
        mlp = mean_effects(f"blocks.{L}.hook_mlp_out")
        attn = mean_effects(f"blocks.{L}.hook_attn_out")
        per_layer.append({"layer": L, "mlp": mlp, "attn": attn})
        print(f"L{L:02d}  MLP te={mlp['te']:+.3f} de={mlp['de']:+.3f} | "
              f"attn te={attn['te']:+.3f} de={attn['de']:+.3f}")

    # cumulative MLP direct effect over the sweep -> write-site saturation layer
    mlp_de = np.array([r["mlp"]["de"] for r in per_layer])
    cum = np.cumsum(np.clip(mlp_de, 0, None))
    total = cum[-1] if cum.size and cum[-1] > 0 else 1.0
    sat_idx = int(np.searchsorted(cum, 0.8 * total))
    write_site_layer = sweep[min(sat_idx, len(sweep) - 1)] if sweep else None

    # --- direction-restricted DE/TE at the build layer (MLP-out), helix subspace only ---
    build_mlp = f"blocks.{build_layer}.hook_mlp_out"
    full_node = mean_effects(build_mlp)
    helix_dir = mean_effects(build_mlp, sender_basis=helix_basis)

    summary = {
        "model": args.model, "build_layer": build_layer, "sweep_layers": [lo, hi],
        "helix_fit_r2": fitres["r2"], "helix_rank": int(helix_basis.shape[1]),
        "n_test": len(triples),
        "write_site_layer_80pct_cumMLP_DE": write_site_layer,
        "per_layer": per_layer,
        "build_layer_mlp_full_node": full_node,
        "build_layer_mlp_helix_direction": helix_dir,
        "helix_direction_de_fraction": (
            helix_dir["de"] / full_node["de"] if abs(full_node["de"]) > 1e-6 else float("nan")),
        "interpretation": (
            "Fig-6 pass: MLPs carry DE (write), early attention carries IE (routing). "
            "write_site_layer ~ where cumulative MLP DE saturates = the stub injection band. "
            "helix_direction_de_fraction near 1.0 => the write-site direct effect lives in "
            "the answer-helix subspace (supports replace-along-a-direction)."),
    }
    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label="run_te_de_probe/addition",
                         meta={"script": "run_te_de_probe.py", "context": "addition",
                               "build_layer": build_layer})
    (out / "te_de_probe.json").write_text(json.dumps(summary, indent=2))
    _plot(per_layer, write_site_layer, out / "te_de_by_layer.png", args.model)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))
    print(f"[done] wrote {out}")


def _plot(per_layer, write_site_layer, path, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = [r["layer"] for r in per_layer]
    plt.figure(figsize=(9, 4.5))
    plt.plot(L, [r["mlp"]["te"] for r in per_layer], label="MLP TE", marker="o", ms=3)
    plt.plot(L, [r["mlp"]["de"] for r in per_layer], label="MLP DE", marker="o", ms=3)
    plt.plot(L, [r["attn"]["te"] for r in per_layer], label="attn TE", marker="x", ms=3)
    plt.plot(L, [r["attn"]["de"] for r in per_layer], label="attn DE", marker="x", ms=3)
    if write_site_layer is not None:
        plt.axvline(write_site_layer, color="0.4", ls="--", lw=1,
                    label=f"write-site ~L{write_site_layer}")
    plt.xlabel("layer"); plt.ylabel("mean logit diff (clean − corrupt)")
    plt.title(f"TE/DE by layer (answer token) — {model} — addition  [kantamneni2025 Fig 6]")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
