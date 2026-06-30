"""Week 1 — causal sufficiency of the helix subspace at the COMPONENT outputs (MLP-out and
attention-out), DENOISING direction. The component-level analogue of run_causal_validation
(which acts on resid_post), mirroring how run_fourier_components_raw splits run_fourier into
the two component sites.

Why a separate script (2026-06-27, user request): run_causal_validation patches the
*accumulated* residual stream; this one asks, per layer, whether the helix as written by a
single component — `hook_mlp_out` or `hook_attn_out` — is causally sufficient. Direction is
DENOISING (base = corrupt `a'+b`, inject the clean helix at the component output, recompute,
read restoration); see n2p.number_repr.causal_sufficiency for the shared method.

IMPORTANT interpretation caveat: at a SINGLE component the corrupt operand a' still lives in
every other component, the embedding, and the non-helix dims, so a single-component injection
is a PARTIAL (total-effect) restoration — read it relative to the per-site whole-component
CEILING (`full`), not as absolute operand sufficiency (that is the resid script). Expect the
per-component helix-fit R² to be lower than on resid_post (a component writes only a piece of
the helix) — itself informative about which component builds it.

Two sites -> two PNGs (`causal_components_MLP.summary.png`, `causal_components_Attn.summary.png`),
each with one panel-row per framing x 2 cols (absolute restoration | ratio-to-whole-component),
and one JSON per framing carrying both sites. Methods are full parity with run_causal_validation:
noop / whole-component (`full`) / helix_full / helix_magnitude / helix_modular / pca9 / pca27.

--read-token {a,sum} (default a, matching run_fourier_components_raw): `a` fits/injects helix(a)
at the operand token; `sum` fits/injects helix(a+b) at the answer token. Answer logits are read
at the last position either way. NB at --read-token sum the helix_full/full curves coincide with
the TE curves run_te_de_probe already draws; the new content here is the operand-token (read=a)
test and the magnitude/modular + PCA controls.

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python experiments/week1_number_representation/run_causal_validation_components.py --model gptj --kshot 4
    python experiments/week1_number_representation/run_causal_validation_components.py --model gptj --read-token sum --layers 10 14 18 22
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks                  # noqa: E402
from n2p.number_repr import repcli                       # noqa: E402
from n2p.number_repr import causal_sufficiency as cs     # noqa: E402

SITES = {"MLP": "hook_mlp_out", "Attn": "hook_attn_out"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--operation", choices=tasks.OPERATION_CHOICES, default="addition")
    ap.add_argument("--framing", choices=tasks.FRAMING_NAMES, default=None,
                    help="restrict to a single framing; default = ALL framings, one panel-row each.")
    ap.add_argument("--read-token", choices=["a", "sum"], default="a", dest="read_token",
                    help="a = operand token (helix(a), default, like run_fourier_components_raw); "
                         "sum = answer token (helix(a+b), like run_fourier_components).")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layers of intervention to sweep. Default = ALL layers (0..last).")
    ap.add_argument("--layer", type=int, default=None, help="single-layer shortcut.")
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--n_pca", type=int, default=9, help="PCA dims the HELIX is fit inside")
    ap.add_argument("--pca_dims", type=int, nargs="*", default=[9, 27],
                    help="dims for the PCA-injection baseline (kantamneni2025 Fig 5).")
    ap.add_argument("--b_fixed", type=int, default=5)
    ap.add_argument("--prefix", default=None,
                    help="model instruction prefix; default = config ModelSpec.prompt_prefix. "
                         "Pass '' to ablate.")
    ap.add_argument("--kshot", type=int, default=0,
                    help="few-shot solved examples before each query (GPT-J needs e.g. 4).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = config.get_model_spec(args.model)
    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else spec.prompt_prefix
    shots = tasks.fewshot_shots(args.operation, args.kshot, args.seed)
    operation, rt = args.operation, args.read_token
    layers = cs.resolve_layers(args.layers, args.layer, spec.n_layers)
    framings = [args.framing] if args.framing else repcli.framings_for_summary(operation, rt)
    if not framings:
        raise SystemExit(f"no framing available for operation {operation!r}")
    task = tasks.get_task(operation)
    site_hooks = list(SITES.values())
    print(f"[prefix] {prefix!r}  [kshot] {args.kshot}")
    print(f"[setup] {operation} (DENOISING components, read={rt}); framings={framings}; "
          f"layers={layers[0]}..{layers[-1]} ({len(layers)}) x {len(site_hooks)} sites x "
          f"~{args.n_test} triples per framing.")

    out = config.run_dir("week1_number_representation", args.seed, model=args.model,
                         label=f"run_causal_validation_components/{operation}",
                         meta={"script": "run_causal_validation_components.py",
                               "operation": operation, "read_token": rt, "direction": "denoising",
                               "sites": site_hooks,
                               "framing": framings[0] if len(framings) == 1 else None,
                               "framings": framings, "layers": layers, "prefix": prefix,
                               "kshot": args.kshot})

    panels = {name: [] for name in SITES}                # display-name -> [(framing, ...)]
    for framing in framings:
        res = cs.sweep_framing(model, operation, framing, sites=site_hooks, read_token=rt,
                               layers=layers, task=task, n_fit=args.n_fit, n_pca=args.n_pca,
                               pca_dims=args.pca_dims, n_test=args.n_test, b_fixed=args.b_fixed,
                               prefix=prefix, shots=shots, seed=args.seed)
        (out / f"causal_components.{framing}.json").write_text(json.dumps({
            "model": args.model, "hf_id": spec.hf_id, "operation": operation,
            "framing": framing, "read_token": rt, "direction": "denoising", "layers": layers,
            "prefix": prefix, "kshot": args.kshot, "n_pca": args.n_pca,
            "pca_dims": [k for k in args.pca_dims], "n_test": len(res["triples"]),
            "frac_single_token_answers": res["frac_single"], "helix_ranks": res["helix_ranks"],
            "mlp": res["per_site"]["hook_mlp_out"], "attn": res["per_site"]["hook_attn_out"],
            "interpretation": (
                "Per-component DENOISING restoration vs LAYER. full = whole-component ceiling; "
                "noop = floor. helix_full near full at a layer => that component's helix write "
                "carries the operand effect there (partial; a' still present elsewhere). "
                "Lower per-component helix R² than resid is expected."),
            "trials": res["triples"],
        }, indent=2))
        for name, hook in SITES.items():
            panels[name].append((framing, res["per_site"][hook], res["methods"],
                                 res["frac_single"]))
        print(f"[{framing}] done; MLP R² L{layers[0]}={res['r2']['hook_mlp_out'][layers[0]]:.2f}"
              f"  Attn R² L{layers[0]}={res['r2']['hook_attn_out'][layers[0]]:.2f}")

    for name in SITES:
        cs.plot_sufficiency(
            panels[name], out / f"causal_components_{name}.summary.png",
            f"Causal sufficiency of the helix subspace — {name}-output (denoising) — "
            f"{args.model} — {operation} — read={rt}", ceiling_label=f"whole-{name}")
    print(f"[done] wrote {out} ({len(framings)} framing(s); MLP + Attn summaries)")


if __name__ == "__main__":
    main()
