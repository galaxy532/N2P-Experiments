"""Week 1 — component-output Fourier check in ACTIVATION space [zhou2024 §4.1 idea].

The activation-space sibling of run_fourier_components.py. Same layout (MLP output and
attention output of one layer, plotted SIDE BY SIDE), but it DFTs the RAW component
outputs instead of their logit-lens projection.

Two different questions (see fourier-experiments-week1-results.md):
  - run_fourier_components.py  (LOGIT space):  which residue class does this component
    PROMOTE?  -> DFT over the candidate-ANSWER axis (via W_U), answer-token site.
  - run_fourier_components_raw.py (ACTIVATION space, here): is this component's OUTPUT
    REPRESENTATION sparse in frequency over the INPUT number? -> DFT the d-dim activation
    vectors over the input-number sweep, power averaged over dims. This is the object
    closest to what week-2 SAEs ingest (SAEs train on these raw outputs, not on logits).

Because a raw activation is just a d-vector with no number axis inside it, the number
axis comes from SWEEPING input numbers — exactly like run_fourier.py — not from a vocab
projection. So --context controls the prompt/token the sweep reads, mirroring
run_fourier.py: bare = isolated number token ' {n}'; addition = operand-a token in
'{a}+{b}=' (b fixed).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --layer 16
    python3 experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --layer 16 --context addition
    python3 experiments/week1_number_representation/run_fourier_components_raw.py --model gptj --summary

NOTE on --hi 360: with --lo 0 inclusive, 0..360 is 361 points (prime), so periods
2/2.5/5/10 do not land on DFT bins and read slightly off (e.g. 10.03); cosmetic only.
Use --lo 1 --hi 360 or --lo 0 --hi 359 for exact-integer periods. Default kept at 360.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models                       # noqa: E402
from n2p.number_repr import fourier, plotting          # noqa: E402
from n2p.number_repr.causal import cache_number_site_all_layers  # noqa: E402


def _operand_a_index(model, prompt):
    """Index of the operand-a token in an '{a}+{b}=' prompt (after BOS). First position
    whose token contains a digit. Mirrors run_fourier / run_helix_fit so the scripts read
    the same site."""
    for i, t in enumerate(model.to_str_tokens(prompt)):
        if any(c.isdigit() for c in t):
            return i
    raise ValueError(f"could not locate operand a in {model.to_str_tokens(prompt)}")


def _contiguous_prefix(values):
    """Keep the contiguous integer prefix (the DFT assumes an even grid). Warn on gaps."""
    values = np.array(sorted(values))
    if values.size:
        gaps = np.where(np.diff(values) != 1)[0]
        if gaps.size:
            cut = int(gaps[0]) + 1
            print(f"[warn] gap after {values[cut-1]} (next single-token value is "
                  f"{values[cut]}); dropping {len(values) - cut} value(s), using "
                  f"contiguous {values[0]}..{values[cut-1]} ({cut} numbers)")
            values = values[:cut]
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer L whose MLP output and attention output are analyzed "
                         "(required unless --summary, which sweeps all layers).")
    ap.add_argument("--summary", action="store_true",
                    help="instead of one layer, sweep every layer and draw the "
                         "[zhou2024] Fig 3 layer x frequency heatmap (MLP | attn side "
                         "by side) in ACTIVATION space, colour = component magnitude.")
    ap.add_argument("--layers", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--summary only: restrict to the inclusive layer band [LO,HI] "
                         "(default: all layers; the paper used the last 15).")
    ap.add_argument("--power-transform", choices=["amplitude", "power", "log"],
                    default="amplitude", dest="power_transform",
                    help="--summary colour scale: amplitude=sqrt(mean power)=||C_k|| "
                         "(default, linear); power=raw; log=log10.")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=360)
    ap.add_argument("--context", choices=["bare", "addition"], default="bare",
                    help="bare (default): isolated number token ' {n}'. addition: operand-a "
                         "token inside '{a}+{b}=' (b fixed). Mirrors run_fourier.py.")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for '{a}+{b}=' prompts when --context addition.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.summary and args.layer is None:
        ap.error("--layer is required unless --summary is set")

    model = models.load_model(args.model)
    tok_map = models.number_token_ids(model, args.lo, args.hi)
    values = _contiguous_prefix(tok_map)
    if values.size < 8:
        raise SystemExit(f"too few single-token numbers in [{args.lo},{args.hi}] for a DFT")

    if args.context == "addition":
        prompts = [f"{int(v)}+{args.b_fixed}=" for v in values]
        token_index = _operand_a_index(model, prompts[0])
        print(f"[context=addition] b={args.b_fixed}; operand-a token index={token_index}")
    else:
        prompts = [f" {int(v)}" for v in values]
        token_index = -1
        print(f"[context=bare] {len(prompts)} isolated number prompts")

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier_components_raw/{args.context}",
                         meta={"script": "run_fourier_components_raw.py",
                               "context": args.context,
                               "layer": args.layer if not args.summary else None,
                               "summary": bool(args.summary)})

    if args.summary:
        _run_summary(model, args, prompts, token_index, values, out)
        return

    mlp_hook = f"blocks.{args.layer}.hook_mlp_out"
    attn_hook = f"blocks.{args.layer}.hook_attn_out"
    caches = cache_number_site_all_layers(model, prompts, [mlp_hook, attn_hook],
                                          token_index=token_index)

    specs = {
        "mlp": fourier.number_dft(caches[mlp_hook], values),
        "attn": fourier.number_dft(caches[attn_hook], values),
    }
    site = f"components_raw.L{args.layer}"
    summary = {
        "model": args.model, "site": site, "context": args.context,
        "layer": args.layer,
        "b_fixed": args.b_fixed if args.context == "addition" else None,
        "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "mlp_dominant_periods_top10": [float(p) for p in specs["mlp"]["dominant_periods"]],
        "attn_dominant_periods_top10": [float(p) for p in specs["attn"]["dominant_periods"]],
    }
    # Folder = <model>/run_fourier_components_raw/<context>; file only needs the layer.
    (out / f"L{args.layer}.json").write_text(json.dumps(summary, indent=2))
    _plot(specs, out / f"L{args.layer}.png",
          args.model, args.layer, args.context)
    print(f"[done] MLP top5 periods:  {summary['mlp_dominant_periods_top10'][:5]}")
    print(f"[done] attn top5 periods: {summary['attn_dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out} (site={site}, context={args.context})")


def _run_summary(model, args, prompts, token_index, values, out):
    """[zhou2024] Fig 3 in ACTIVATION space: sweep every layer, build the
    layer x frequency heatmap of MLP and attention raw-output magnitudes, side by side."""
    lo, hi = (args.layers if args.layers is not None else (0, model.cfg.n_layers - 1))
    if not (0 <= lo <= hi <= model.cfg.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{model.cfg.n_layers - 1}]")
    layers = list(range(lo, hi + 1))
    mlp_hooks = [f"blocks.{L}.hook_mlp_out" for L in layers]
    attn_hooks = [f"blocks.{L}.hook_attn_out" for L in layers]
    caches = cache_number_site_all_layers(model, prompts, mlp_hooks + attn_hooks,
                                          token_index=token_index)
    mlp_specs = [fourier.number_dft(caches[f"blocks.{L}.hook_mlp_out"], values)
                 for L in layers]
    attn_specs = [fourier.number_dft(caches[f"blocks.{L}.hook_attn_out"], values)
                  for L in layers]

    mlp_mat, freqs = plotting.stack_power(mlp_specs)
    attn_mat, _ = plotting.stack_power(attn_specs)
    plotting.plot_layer_freq_heatmap(
        [("MLP output", mlp_mat), ("Attention output", attn_mat)],
        freqs, layers, args.context, out / "summary_layers.png",
        model=args.model, value_unit="activation", transform=args.power_transform,
        title=f"Component-output activations in Fourier space across layers — "
              f"{args.model} (context={args.context})")
    summary = {
        "model": args.model, "site": "components_raw.summary", "context": args.context,
        "transform": args.power_transform,
        "b_fixed": args.b_fixed if args.context == "addition" else None,
        "layers": layers, "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "freqs": [float(f) for f in freqs],
        "mlp_dominant_periods_by_layer": {
            int(L): [float(p) for p in s["dominant_periods"][:5]]
            for L, s in zip(layers, mlp_specs)},
        "attn_dominant_periods_by_layer": {
            int(L): [float(p) for p in s["dominant_periods"][:5]]
            for L, s in zip(layers, attn_specs)},
    }
    (out / "summary_layers.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary heatmap over layers {lo}..{hi} "
          f"(transform={args.power_transform})")
    print(f"[done] wrote {out}/summary_layers.png (context={args.context})")


def _annotate(ax, spec, k=6):
    for idx in spec["dominant_freq_idx"][:k]:
        f = spec["freqs"][idx]
        if f <= 0:
            continue
        ax.annotate(f"T={1.0/f:.2f}", (f, spec["power"][idx]),
                    textcoords="offset points", xytext=(0, 4), fontsize=7, ha="center")


def _plot(specs, path, model, layer, context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
    for ax, name, title in ((axes[0], "mlp", "MLP output"),
                            (axes[1], "attn", "Attention output")):
        s = specs[name]
        ax.plot(s["freqs"], s["power"], marker="o", ms=2)
        _annotate(ax, s)
        ax.set_xlabel("frequency (cycles / integer)")
        ax.set_title(f"{title} — L{layer}")
    axes[0].set_ylabel("mean power (activation)")
    fig.suptitle(f"Component-output activations in Fourier space — {model} "
                 f"— L{layer} (context={context})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
