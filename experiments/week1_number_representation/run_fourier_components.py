"""Week 1 — component-output Fourier check in LOGIT space [zhou2024] (Figs 2-3).

Reproduces zhou2024 Sec 3: the per-component LOGIT contribution of a single layer's
MLP / attention output, transformed into Fourier space. For one layer L we read the
MLP output and the attention output, project EACH through the unembedding W_U onto the
single-token number ids, DFT the resulting logit-over-number signal, and plot the two
power spectra SIDE BY SIDE.

Why logit space and not the raw activation (see the long discussion / the exp-note
fourier-experiments-week1-results.md): a single component's logit contribution is a
broad PERIODIC WAVE over the whole number line (the L33 MLP "favors even numbers";
L40 attention "favors mod 5 and mod 10", zhou2024 Fig 20), NOT a spike. Its content is
the PERIOD, a frequency-domain property invisible to a top-k logit readout, and the
signal is dense in the number basis but SPARSE in the frequency basis. The MLP=low-freq
(magnitude/approximation) vs attention=high-freq (modular/classification) split is a
logit-space claim, so it is reproduced here, not in the activation-space run_fourier.py.

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # bare number prompts (default), layer 16:
    python3 experiments/week1_number_representation/run_fourier_components.py --model gptj --layer 16
    # addition prompts (the canonical Fig 2/3 setting, answer-token site):
    python3 experiments/week1_number_representation/run_fourier_components.py --model gptj --layer 27 --context addition
    python3 experiments/week1_number_representation/run_fourier_components.py --model gptj --summary

NOTE on the DFT: W_U projection uses the raw unembedding only (models load via
from_pretrained_no_processing, so ln_final is NOT folded and we do not apply it). The
final-LayerNorm scale is a positive per-example scalar; it rescales a spectrum but does
not move WHICH periods are present, and we average POWER across examples, so the period
structure (the thing we read) is unaffected. Pass --apply-ln to fold it in if wanted.

NOTE on --hi 360: with --lo 0 inclusive, 0..360 is 361 sample points (prime) -> the
predicted periods 2/2.5/5/10 do NOT land on DFT bins (freq k/361), so peaks LEAK across
neighbouring bins and dominant periods read ~10.03 instead of 10.0. For exact bin
alignment use 360 sample POINTS, e.g. --lo 1 --hi 360 or --lo 0 --hi 359. Default kept
at 360 per request; the contiguous-range filter warns if tokenization drops values.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models                       # noqa: E402
from n2p.number_repr import fourier, plotting          # noqa: E402
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
    example j's component output writes, so number_dft DFTs along the number axis and
    averages power across examples (the Fig 3 averaging)."""
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
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=360)
    ap.add_argument("--context", choices=["bare", "addition"], default="bare",
                    help="bare (default): sweep isolated number tokens ' {n}', read the "
                         "component output at that token. addition: '{a}+{b}=' prompts, "
                         "read the component output at the answer (last) token over a "
                         "sample of (a,b) pairs — the canonical zhou2024 Fig 2/3 setting.")
    ap.add_argument("--n_examples", type=int, default=128,
                    help="number of (a,b) prompts to average power over in --context "
                         "addition (ignored for bare, which uses the full number sweep).")
    ap.add_argument("--b_lo", type=int, default=1,
                    help="addition: inclusive low for the sampled second operand b.")
    ap.add_argument("--b_hi", type=int, default=9,
                    help="addition: inclusive high for the sampled second operand b.")
    ap.add_argument("--apply-ln", action="store_true",
                    help="fold the final LayerNorm before W_U (off by default; see header).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.summary and args.layer is None:
        ap.error("--layer is required unless --summary is set")

    rng = random.Random(args.seed)
    model = models.load_model(args.model)
    tok_map = models.number_token_ids(model, args.lo, args.hi)
    values = _contiguous_prefix(tok_map)               # candidate-number (logit) axis
    if values.size < 8:
        raise SystemExit(f"too few single-token numbers in [{args.lo},{args.hi}] for a DFT")
    number_ids = [tok_map[int(v)] for v in values]
    vset = set(int(v) for v in values)

    if args.context == "addition":
        # Sample single-token (a, b) with single-token answer a+b in range. We read the
        # component output at the answer ('=') token; the candidate-number axis is the
        # set of possible answers (zhou2024 analyse the answer/last token).
        a_pool = [int(v) for v in values]
        prompts = []
        tries = 0
        while len(prompts) < args.n_examples and tries < args.n_examples * 50:
            tries += 1
            a = rng.choice(a_pool)
            b = rng.randint(args.b_lo, args.b_hi)
            if (a + b) in vset and b in vset:
                prompts.append(f"{a}+{b}=")
        if not prompts:
            raise SystemExit("could not build any single-token '{a}+{b}=' prompts in range")
        token_index = -1                               # answer token
        print(f"[context=addition] {len(prompts)} prompts, b in [{args.b_lo},{args.b_hi}], "
              f"answer-token site")
    else:
        # Bare: each input number is one example; magnitude spectra are shift-invariant,
        # so averaging power over inputs extracts WHICH periods the component writes.
        prompts = [f" {int(v)}" for v in values]
        token_index = -1                               # the number token
        print(f"[context=bare] {len(prompts)} isolated number prompts")

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier_components/{args.context}",
                         meta={"script": "run_fourier_components.py",
                               "context": args.context,
                               "layer": args.layer if not args.summary else None,
                               "summary": bool(args.summary)})

    if args.summary:
        _run_summary(model, args, prompts, token_index, values, number_ids, out)
        return

    mlp_hook = f"blocks.{args.layer}.hook_mlp_out"
    attn_hook = f"blocks.{args.layer}.hook_attn_out"
    caches = cache_number_site_all_layers(model, prompts, [mlp_hook, attn_hook],
                                          token_index=token_index)
    specs = {}
    for name, hook in (("mlp", mlp_hook), ("attn", attn_hook)):
        logit_mat = _logit_matrix(model, caches[hook], number_ids, args.apply_ln)
        specs[name] = fourier.number_dft(logit_mat, values)

    site = f"components.L{args.layer}"
    summary = {
        "model": args.model, "site": site, "context": args.context,
        "layer": args.layer, "apply_ln": bool(args.apply_ln),
        "n_examples": len(prompts), "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "mlp_dominant_periods_top10": [float(p) for p in specs["mlp"]["dominant_periods"]],
        "attn_dominant_periods_top10": [float(p) for p in specs["attn"]["dominant_periods"]],
    }
    # Folder = <model>/run_fourier_components/<context>; file only needs the layer.
    (out / f"L{args.layer}.json").write_text(json.dumps(summary, indent=2))
    _plot(specs, out / f"L{args.layer}.png",
          args.model, args.layer, args.context)
    print(f"[done] MLP top5 periods:  {summary['mlp_dominant_periods_top10'][:5]}")
    print(f"[done] attn top5 periods: {summary['attn_dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out} (site={site}, context={args.context})")


def _run_summary(model, args, prompts, token_index, values, number_ids, out):
    """[zhou2024] Fig 3: sweep every layer, build the layer x frequency heatmap of MLP
    and attention component-output LOGIT magnitudes, side by side."""
    lo, hi = (args.layers if args.layers is not None else (0, model.cfg.n_layers - 1))
    if not (0 <= lo <= hi <= model.cfg.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{model.cfg.n_layers - 1}]")
    layers = list(range(lo, hi + 1))
    mlp_hooks = [f"blocks.{L}.hook_mlp_out" for L in layers]
    attn_hooks = [f"blocks.{L}.hook_attn_out" for L in layers]
    # One forward sweep caches every layer's MLP and attn output.
    caches = cache_number_site_all_layers(model, prompts, mlp_hooks + attn_hooks,
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
        freqs, layers, args.context, out / "summary_layers.png",
        model=args.model, value_unit="logit", transform=args.power_transform,
        title=f"Component-output logits in Fourier space across layers — "
              f"{args.model} (context={args.context})")
    summary = {
        "model": args.model, "site": "components.summary", "context": args.context,
        "apply_ln": bool(args.apply_ln), "transform": args.power_transform,
        "layers": layers, "n_examples": len(prompts), "n_numbers": int(values.size),
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
    """Mark the top-k dominant components (by power) with their period."""
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
    axes[0].set_ylabel("mean logit power")
    fig.suptitle(f"Component-output logits in Fourier space — {model} "
                 f"— L{layer} (context={context})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
