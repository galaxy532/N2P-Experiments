"""Week 1 — component-output Fourier check in LOGIT space [zhou2024] (Figs 2-3).

Reproduces zhou2024 Sec 3: the per-component LOGIT contribution of a single layer's MLP /
attention output, transformed into Fourier space. For one layer L we read the MLP output
and the attention output (at the chosen --read-token), project EACH through the unembedding
W_U onto the single-token number ids, DFT the resulting logit-over-number signal, and plot
the two power spectra SIDE BY SIDE.

Why logit space and why the sum (answer) token (see fourier-experiments-week1-results.md):
a single component's logit contribution is a broad PERIODIC WAVE over the whole number line
(L33 MLP "favors even numbers"; L40 attention "favors mod 5 and mod 10"), NOT a spike. Its
content is the PERIOD. The logit lens W_U·h is the NEXT-token readout, so it is only
meaningful where the model is predicting the answer — the SUM token (last token before the
answer). Reading it at the operand tokens (a/b) is the "uninteresting" case and is offered
only for comparison.

Operand `a` is SWEPT; `b` is FIXED (`--b_fixed`); power is averaged over the swept a's (the
controlled analogue of zhou2024's "across all test data" averaging). Framing is --context
(template_1..4); position is --read-token {a,b,sum} (default sum = answer token).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_fourier_components.py --model gptj --layer 16 --context template_3
    python3 experiments/week1_number_representation/run_fourier_components.py --model gptj --summary --context template_3   # Fig-3 heatmap, answer token

NOTE on the DFT: W_U projection uses the raw unembedding only (models load via
from_pretrained_no_processing, so ln_final is NOT folded). The final-LayerNorm scale is a
positive per-example scalar; it rescales a spectrum but does not move WHICH periods are
present, and we average POWER across examples. Pass --apply-ln to fold it in if wanted.

NOTE on --hi 360: with --lo 0 inclusive, 0..360 is 361 points (prime) -> periods
2/2.5/5/10 do NOT land on DFT bins and read ~10.03. For exact bins use --lo 1 --hi 360 or
--lo 0 --hi 359. Default kept at 360.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models                       # noqa: E402
from n2p.number_repr import fourier, plotting, prompts  # noqa: E402
from n2p.number_repr.causal import cache_number_site_all_layers  # noqa: E402


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


def _logit_matrix(model, acts, number_ids, apply_ln):
    """Project component outputs (n_examples, d) onto the number-token logits.

    Returns (N_numbers, n_examples): column j is the logit-over-number signal that
    example j's component output writes, so number_dft DFTs along the candidate-number
    axis and averages power across the swept-a examples (the Fig 3 averaging)."""
    A = torch.tensor(acts, dtype=model.cfg.dtype, device=model.cfg.device)  # (e, d)
    if apply_ln:
        A = model.ln_final(A)
    W_num = model.W_U[:, number_ids]                  # (d, N)
    logits = (A @ W_num).float().cpu().numpy()         # (e, N)
    return logits.T                                    # (N, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer L whose MLP output and attention output are analyzed "
                         "(required unless --summary, which sweeps all layers).")
    ap.add_argument("--summary", action="store_true",
                    help="instead of one layer, sweep every layer and draw the "
                         "[zhou2024] Fig 3 layer x frequency heatmap (MLP | attn side "
                         "by side), colour = component magnitude.")
    ap.add_argument("--layers", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--summary only: restrict to the inclusive layer band [LO,HI] "
                         "(default: all layers; the paper used the last 15).")
    ap.add_argument("--power-transform", choices=["amplitude", "power", "log"],
                    default="amplitude", dest="power_transform",
                    help="--summary colour scale: amplitude=sqrt(mean power)=||C_k|| "
                         "(default, linear); power=raw; log=log10.")
    ap.add_argument("--cmap", default="inferno_r",
                    help="--summary colormap (default inferno_r, light background so "
                         "near-zero cells read light, not black).")
    ap.add_argument("--vmax-percentile", type=float, default=99.5,
                    dest="vmax_percentile",
                    help="--summary robust colour-limit percentile (default 99.5 so a "
                         "single low-freq spike doesn't wash out the rest; 100=true max).")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=360)
    ap.add_argument("--context", choices=prompts.TEMPLATE_CHOICES, default="template_1",
                    help="prompt framing (see n2p.number_repr.prompts). The canonical "
                         "zhou2024 Fig 2/3 setting is an addition template (template_3/4) "
                         "read at --read-token sum.")
    ap.add_argument("--read-token", choices=prompts.READ_TOKEN_CHOICES, default="sum",
                    dest="read_token",
                    help="which position to read the component output at before W_U. "
                         "sum (default) = last/answer token (the meaningful logit-lens "
                         "site); a/b = operand tokens (comparison only).")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for templates that use it (templates 3,4).")
    ap.add_argument("--apply-ln", action="store_true",
                    help="fold the final LayerNorm before W_U (off by default; see header).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.summary and args.layer is None:
        ap.error("--layer is required unless --summary is set")
    prompts.validate_read_token(args.read_token, args.context)

    model = models.load_model(args.model)
    tok_map = models.number_token_ids(model, args.lo, args.hi)
    values = _contiguous_prefix(np.array(sorted(tok_map)))   # candidate-number (logit) axis
    if values.size < 8:
        raise SystemExit(f"too few single-token numbers in [{args.lo},{args.hi}] for a DFT")
    number_ids = [tok_map[int(v)] for v in values]

    prompt_list = prompts.build_prompts(args.context, values, args.b_fixed)
    token_index = prompts.read_token_index(model, prompt_list[0], args.read_token,
                                           args.context)
    print(f"[{args.context}] read-token={args.read_token} at index {token_index}; "
          f"b={args.b_fixed if prompts.template_has_b(args.context) else '-'}; "
          f"sweep a over {values.size} values")

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier_components/{args.context}",
                         meta={"script": "run_fourier_components.py",
                               "context": args.context, "read_token": args.read_token,
                               "layer": args.layer if not args.summary else None,
                               "summary": bool(args.summary)})

    if args.summary:
        _run_summary(model, args, prompt_list, token_index, values, number_ids, out)
        return

    mlp_hook = f"blocks.{args.layer}.hook_mlp_out"
    attn_hook = f"blocks.{args.layer}.hook_attn_out"
    caches = cache_number_site_all_layers(model, prompt_list, [mlp_hook, attn_hook],
                                          token_index=token_index)
    specs = {}
    for name, hook in (("mlp", mlp_hook), ("attn", attn_hook)):
        logit_mat = _logit_matrix(model, caches[hook], number_ids, args.apply_ln)
        specs[name] = fourier.number_dft(logit_mat, values)

    site = f"components.L{args.layer}"
    summary = {
        "model": args.model, "site": site, "context": args.context,
        "read_token": args.read_token, "layer": args.layer, "apply_ln": bool(args.apply_ln),
        "b_fixed": args.b_fixed if prompts.template_has_b(args.context) else None,
        "n_examples": len(prompt_list), "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "mlp_dominant_periods_top10": [float(p) for p in specs["mlp"]["dominant_periods"]],
        "attn_dominant_periods_top10": [float(p) for p in specs["attn"]["dominant_periods"]],
    }
    (out / f"L{args.layer}.{args.read_token}.json").write_text(json.dumps(summary, indent=2))
    _plot(specs, out / f"L{args.layer}.{args.read_token}.png",
          args.model, args.layer, args.context, args.read_token)
    print(f"[done] MLP top5 periods:  {summary['mlp_dominant_periods_top10'][:5]}")
    print(f"[done] attn top5 periods: {summary['attn_dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out} (site={site}, context={args.context}, read={args.read_token})")


def _run_summary(model, args, prompt_list, token_index, values, number_ids, out):
    """[zhou2024] Fig 3: sweep every layer, build the layer x frequency heatmap of MLP and
    attention component-output LOGIT magnitudes, side by side."""
    lo, hi = (args.layers if args.layers is not None else (0, model.cfg.n_layers - 1))
    if not (0 <= lo <= hi <= model.cfg.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{model.cfg.n_layers - 1}]")
    layers = list(range(lo, hi + 1))
    mlp_hooks = [f"blocks.{L}.hook_mlp_out" for L in layers]
    attn_hooks = [f"blocks.{L}.hook_attn_out" for L in layers]
    caches = cache_number_site_all_layers(model, prompt_list, mlp_hooks + attn_hooks,
                                          token_index=token_index)
    mlp_specs, attn_specs = [], []
    for L in layers:
        mlp_lm = _logit_matrix(model, caches[f"blocks.{L}.hook_mlp_out"],
                               number_ids, args.apply_ln)
        attn_lm = _logit_matrix(model, caches[f"blocks.{L}.hook_attn_out"],
                                number_ids, args.apply_ln)
        mlp_specs.append(fourier.number_dft(mlp_lm, values))
        attn_specs.append(fourier.number_dft(attn_lm, values))

    mlp_mat, freqs = plotting.stack_power(mlp_specs)
    attn_mat, _ = plotting.stack_power(attn_specs)
    plotting.plot_layer_freq_heatmap(
        [("MLP output", mlp_mat), ("Attention output", attn_mat)],
        freqs, layers, args.context, out / f"summary_layers.{args.read_token}.png",
        model=args.model, value_unit="logit", transform=args.power_transform,
        cmap=args.cmap, vmax_percentile=args.vmax_percentile,
        title=f"Component-output logits in Fourier space across layers — "
              f"{args.model} — {args.context} — read={args.read_token}")
    summary = {
        "model": args.model, "site": "components.summary", "context": args.context,
        "read_token": args.read_token, "apply_ln": bool(args.apply_ln),
        "transform": args.power_transform,
        "b_fixed": args.b_fixed if prompts.template_has_b(args.context) else None,
        "layers": layers, "n_examples": len(prompt_list), "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "freqs": [float(f) for f in freqs],
        "mlp_dominant_periods_by_layer": {
            int(L): [float(p) for p in s["dominant_periods"][:5]]
            for L, s in zip(layers, mlp_specs)},
        "attn_dominant_periods_by_layer": {
            int(L): [float(p) for p in s["dominant_periods"][:5]]
            for L, s in zip(layers, attn_specs)},
    }
    (out / f"summary_layers.{args.read_token}.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary heatmap over layers {lo}..{hi} "
          f"(transform={args.power_transform}, read={args.read_token})")
    print(f"[done] wrote {out}/summary_layers.{args.read_token}.png (context={args.context})")


def _annotate(ax, spec, k=6):
    """Mark the top-k dominant components (by power) with their period."""
    for idx in spec["dominant_freq_idx"][:k]:
        f = spec["freqs"][idx]
        if f <= 0:
            continue
        ax.annotate(f"T={1.0/f:.2f}", (f, spec["power"][idx]),
                    textcoords="offset points", xytext=(0, 4), fontsize=7, ha="center")


def _plot(specs, path, model, layer, context, read_token):
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
    axes[0].set_ylabel("mean logit power")
    fig.suptitle(f"Component-output logits in Fourier space — {model} "
                 f"— L{layer} ({context}, read={read_token})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
