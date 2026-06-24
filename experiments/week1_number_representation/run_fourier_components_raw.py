"""Week 1 — component-output Fourier check in ACTIVATION space [zhou2024 §4.1 idea].

The activation-space sibling of run_fourier_components.py. Same layout (MLP output and
attention output of one layer), but it DFTs the RAW component outputs instead of their
logit-lens projection — the object closest to what week-2 SAEs ingest.

Prompt surface form: --operation + --framing (symbolic / word / wordproblem, n2p.tasks).
Operand `a` is swept, `b` FIXED (--b_fixed). --summary sweeps every layer and draws the
Fig-3 layer x frequency heatmap with ONE PANEL PER FRAMING, as TWO files
(summary_MLP.<tok>.png and summary_Attn.<tok>.png).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --layer 16 --operation addition --framing symbolic --read-token a
    python3 experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --summary --operation addition --read-token a

NOTE on --hi 360: 0..360 is 361 points (prime) -> periods read slightly off; use
--lo 1 --hi 360 for exact bins. Default kept at 360.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models, tasks                 # noqa: E402
from n2p.number_repr import fourier, plotting, repcli   # noqa: E402
from n2p.number_repr.causal import cache_number_site_all_layers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer L whose MLP and attention outputs are analyzed "
                         "(required unless --summary).")
    repcli.add_summary_args(ap)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=360)
    ap.add_argument("--operation", choices=tasks.OPERATION_CHOICES, default="addition",
                    help="arithmetic operation whose framings are analyzed.")
    ap.add_argument("--framing", choices=tasks.FRAMING_NAMES, default="symbolic",
                    help="per-layer runs only: which framing (ignored under --summary).")
    ap.add_argument("--read-token", choices=tasks.READ_TOKEN_CHOICES, default="a",
                    dest="read_token",
                    help="a/b = operand tokens, sum = last token.")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for framings that use it.")
    ap.add_argument("--prefix", default=None,
                    help="model instruction prefix prepended to every prompt; default = "
                         "config ModelSpec.prompt_prefix for --model. Pass '' to ablate.")
    ap.add_argument("--kshot", type=int, default=0,
                    help="few-shot solved examples before the query (0 = zero-shot). GPT-J "
                         "needs few-shot for read=sum; not needed for read=a (operand).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.summary and args.layer is None:
        ap.error("--layer is required unless --summary is set")

    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else config.get_model_spec(args.model).prompt_prefix
    shots = tasks.fewshot_shots(args.operation, args.kshot, args.seed)
    print(f"[prefix] {prefix!r}  [kshot] {args.kshot}")
    # Operand grid validated against the real prompt (canonical symbolic framing).
    grid_values, _ = tasks.single_token_number_grid(model, args.operation, "symbolic",
                                                    args.lo, args.hi, b=args.b_fixed)
    values = repcli.contiguous_prefix(np.array(grid_values))
    if values.size < 8:
        raise SystemExit(f"too few single-token numbers in [{args.lo},{args.hi}] for a DFT")

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier_components_raw/{args.operation}",
                         meta={"script": "run_fourier_components_raw.py",
                               "operation": args.operation, "read_token": args.read_token,
                               "layer": args.layer if not args.summary else None,
                               "summary": bool(args.summary),
                               "framing": None if args.summary else args.framing})

    if args.summary:
        _run_summary(model, args, values, out, prefix, shots)
        return

    tasks.validate_read_token(args.read_token, args.operation, args.framing)
    prompt_list = tasks.build_prompts(args.operation, args.framing, values, args.b_fixed,
                                      prefix=prefix, shots=shots)
    token_index = tasks.read_token_index(model, prompt_list[0], args.read_token,
                                         args.operation, args.framing)
    print(f"[{args.operation}/{args.framing}] read-token={args.read_token} at index "
          f"{token_index}; b={args.b_fixed if tasks.template_has_b(args.operation, args.framing) else '-'}")
    mlp_hook = f"blocks.{args.layer}.hook_mlp_out"
    attn_hook = f"blocks.{args.layer}.hook_attn_out"
    caches = cache_number_site_all_layers(model, prompt_list, [mlp_hook, attn_hook],
                                          token_index=token_index)
    specs = {"mlp": fourier.number_dft(caches[mlp_hook], values),
             "attn": fourier.number_dft(caches[attn_hook], values)}
    summary = {
        "model": args.model, "site": f"components_raw.L{args.layer}",
        "operation": args.operation, "framing": args.framing,
        "read_token": args.read_token, "layer": args.layer, "b_fixed": args.b_fixed,
        "n_numbers": int(values.size), "value_range": [int(values[0]), int(values[-1])],
        "mlp_dominant_periods_top10": [float(p) for p in specs["mlp"]["dominant_periods"]],
        "attn_dominant_periods_top10": [float(p) for p in specs["attn"]["dominant_periods"]],
    }
    stem = f"L{args.layer}.{args.framing}.{args.read_token}"
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    repcli.plot_component_spectra(specs, out / f"{stem}.png", model=args.model,
                                  layer=args.layer, operation=args.operation,
                                  framing=args.framing, read_token=args.read_token,
                                  value_unit="activation")
    print(f"[done] MLP top5 periods:  {summary['mlp_dominant_periods_top10'][:5]}")
    print(f"[done] attn top5 periods: {summary['attn_dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out} ({stem})")


def _run_summary(model, args, values, out, prefix="", shots=()):
    """[zhou2024] Fig 3 in ACTIVATION space: one panel per framing, MLP and attention in
    SEPARATE files."""
    framings = repcli.framings_for_summary(args.operation, args.read_token)
    if not framings:
        raise SystemExit("no framing compatible with --read-token for this operation")
    lo, hi = (args.layers if args.layers is not None else (0, model.cfg.n_layers - 1))
    if not (0 <= lo <= hi <= model.cfg.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{model.cfg.n_layers - 1}]")
    layers = list(range(lo, hi + 1))
    mlp_hooks = [f"blocks.{L}.hook_mlp_out" for L in layers]
    attn_hooks = [f"blocks.{L}.hook_attn_out" for L in layers]

    mlp_panels, attn_panels, freqs, dom = [], [], None, {}
    for framing in framings:
        prompt_list = tasks.build_prompts(args.operation, framing, values, args.b_fixed,
                                          prefix=prefix, shots=shots)
        token_index = tasks.read_token_index(model, prompt_list[0], args.read_token,
                                             args.operation, framing)
        caches = cache_number_site_all_layers(model, prompt_list, mlp_hooks + attn_hooks,
                                              token_index=token_index)
        mlp_specs = [fourier.number_dft(caches[f"blocks.{L}.hook_mlp_out"], values)
                     for L in layers]
        attn_specs = [fourier.number_dft(caches[f"blocks.{L}.hook_attn_out"], values)
                      for L in layers]
        mlp_mat, freqs = plotting.stack_power(mlp_specs)
        attn_mat, _ = plotting.stack_power(attn_specs)
        mlp_panels.append((framing, mlp_mat))
        attn_panels.append((framing, attn_mat))
        dom[framing] = {
            "mlp": {int(L): [float(p) for p in s["dominant_periods"][:5]]
                    for L, s in zip(layers, mlp_specs)},
            "attn": {int(L): [float(p) for p in s["dominant_periods"][:5]]
                     for L, s in zip(layers, attn_specs)}}

    for side, panels in (("MLP", mlp_panels), ("Attn", attn_panels)):
        plotting.plot_layer_freq_heatmap(
            panels, freqs, layers, args.operation,
            out / f"summary_{side}.{args.read_token}.png", model=args.model,
            value_unit="activation", transform=args.power_transform, cmap=args.cmap,
            vmax_percentile=args.vmax_percentile,
            title=f"{side}-output activations in Fourier space across layers — "
                  f"{args.model} — {args.operation} — read={args.read_token}")
    summary = {
        "model": args.model, "site": "components_raw.summary", "operation": args.operation,
        "framings": framings, "read_token": args.read_token,
        "transform": args.power_transform, "layers": layers, "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "freqs": [float(f) for f in freqs], "dominant_periods_by_framing": dom,
    }
    (out / f"summary_layers.{args.read_token}.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary heatmaps over layers {lo}..{hi}, framings={framings}")
    print(f"[done] wrote {out}/summary_MLP.{args.read_token}.png + summary_Attn.{args.read_token}.png")


if __name__ == "__main__":
    main()
