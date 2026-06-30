"""Week 1 — causal sufficiency of the helix subspace, DENOISING direction
[kantamneni2025 §4.4 / Fig-5].

Direction switch (2026-06-27, user-approved): this script now follows the [kantamneni2025]
§4.4 DENOISING test — base run = CORRUPT prompt `a'+b`, INJECT the clean operand helix(a)
into the subspace at the operand-`a` token, let the network recompute, and read how much the
clean answer is restored. The previous version used the [engels2024] Eq 5-6 NOISING+average-
ablate-rest interchange (push a clean run toward a'); we switched because (a) the helix paper
validates the helix by denoising, (b) it matches the N2P stub's actual "inject helix" op, and
(c) average-ablation is then unnecessary (injecting clean into a corrupt run leaves no clean
signal in the orthogonal complement to leak). NB: results produced by the old noising version
are STALE and must be re-run.

Per layer of intervention (swept; default ALL layers 0..last, no build_layers prior), methods:
  noop        = corrupt run, no patch (FLOOR; says a', logit-diff ~0/negative).
  full_layer  = inject the WHOLE clean resid_post[L] (sufficiency CEILING).
  helix_full / helix_magnitude / helix_modular = inject the clean helix (or its separable
                low-freq magnitude / high-freq modular part) into the helix subspace.
  pca9 / pca27 = inject the clean activation's top-k PCA projection — the [kantamneni2025]
                Fig-5 capacity control (9 capacity-matched, 27 over-capacity).
Metric = contrast logit[ans(a)] - logit[ans(a')] on the patched run; ratio = method/full_layer.
helix_full tracking full_layer over a layer band => the helix subspace is causally sufficient
there; helix_full matching/beating pca9/pca27 with fewer effective dims => the periodic FORM,
not raw capacity, carries the number.

All framings by default (one panel-row each, like the fourier --summary); pass --framing F to
restrict. First-token answers (multi-token answers admissible as a leading-digit test;
frac_single_token_answers flags exactness). The shared sweep lives in
n2p.number_repr.causal_sufficiency (also used by run_causal_validation_components.py).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python experiments/week1_number_representation/run_causal_validation.py --model gptj --kshot 4
    python experiments/week1_number_representation/run_causal_validation.py --model gptj \
        --operation multiplication --framing symbolic --layers 10 12 14 16 18 20

NOTE: trickiest week-1 script; operand-token indexing / first-token answers will likely need a
first-run debug pass on GPU.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks                  # noqa: E402
from n2p.number_repr import repcli                       # noqa: E402
from n2p.number_repr import causal_sufficiency as cs     # noqa: E402

SITE = "hook_resid_post"
INTERPRETATION = (
    "DENOISING restoration: inject clean helix(a) into the corrupt run at the operand-a "
    "site, measure recovery of logit[ans(a)]-logit[ans(a')] vs LAYER OF INTERVENTION. "
    "full_layer = ceiling; noop ~0/negative = floor. helix_full tracking full_layer over a "
    "band => the helix is causally sufficient there. helix_full matching/beating pca9/pca27 "
    "with fewer effective dims => the periodic FORM, not generic PCA capacity, carries the "
    "number. magnitude vs modular shows which part does the causal work. Multi-token answers "
    "make this a leading-digit/magnitude test (see frac_single_token_answers).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--operation", choices=tasks.OPERATION_CHOICES, default="addition")
    ap.add_argument("--framing", choices=tasks.FRAMING_NAMES, default=None,
                    help="restrict to a single framing; default = ALL framings, one panel-row each.")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layers of intervention to sweep. Default = ALL layers (0..last); "
                         "no build_layers prior is assumed.")
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
    operation = args.operation
    layers = cs.resolve_layers(args.layers, args.layer, spec.n_layers)
    framings = [args.framing] if args.framing else repcli.framings_for_summary(operation, "a")
    if not framings:
        raise SystemExit(f"no framing available for operation {operation!r}")
    task = tasks.get_task(operation)
    print(f"[prefix] {prefix!r}  [kshot] {args.kshot}")
    print(f"[setup] {operation} (DENOISING); framings={framings}; layers={layers[0]}.."
          f"{layers[-1]} ({len(layers)}) x ~{args.n_test} triples x methods per framing.")

    out = config.run_dir("week1_number_representation", args.seed, model=args.model,
                         label=f"run_causal_validation/{operation}",
                         meta={"script": "run_causal_validation.py", "operation": operation,
                               "framing": framings[0] if len(framings) == 1 else None,
                               "framings": framings, "read_token": "a", "direction": "denoising",
                               "layers": layers, "prefix": prefix, "kshot": args.kshot})

    panels = []
    for framing in framings:
        res = cs.sweep_framing(model, operation, framing, sites=[SITE], read_token="a",
                               layers=layers, task=task, n_fit=args.n_fit, n_pca=args.n_pca,
                               pca_dims=args.pca_dims, n_test=args.n_test, b_fixed=args.b_fixed,
                               prefix=prefix, shots=shots, seed=args.seed)
        per_layer = res["per_site"][SITE]
        summary = {
            "model": args.model, "hf_id": spec.hf_id, "operation": operation,
            "framing": framing, "read_token": "a", "direction": "denoising", "layers": layers,
            "prefix": prefix, "kshot": args.kshot, "n_pca": args.n_pca,
            "pca_dims": [k for k in args.pca_dims], "n_test": len(res["triples"]),
            "frac_single_token_answers": res["frac_single"],
            "helix_ranks": res["helix_ranks"][SITE], "per_layer": per_layer,
            "interpretation": INTERPRETATION,
        }
        (out / f"causal_validation.{framing}.json").write_text(
            json.dumps({"summary": summary, "trials": res["triples"]}, indent=2))
        panels.append((framing, per_layer, res["methods"], res["frac_single"]))
        print(json.dumps(summary, indent=2))

    plot_name = (f"causal_by_layer.{framings[0]}.png" if len(framings) == 1
                 else "causal_by_layer.summary.png")
    cs.plot_sufficiency(panels, out / plot_name,
                        f"Causal sufficiency of the helix subspace (denoising) — {args.model} "
                        f"— {operation}", ceiling_label="full-layer")
    print(f"[done] wrote {out} ({len(framings)} framing(s): {framings})")


if __name__ == "__main__":
    main()
