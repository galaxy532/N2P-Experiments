"""Week 1 — Fourier-feature check [zhou2024].

Take the per-number vectors (default: the token EMBEDDINGS, where the mechanism is
claimed to originate; optionally a residual-stream layer) and show the power spectrum
is sparse, with dominant low-freq (magnitude) and high-freq (modular) components.

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_fourier.py --model gptj
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --layer 16
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --context addition
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --summary
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
    """Index of the operand-a token in an '{a}+{b}=' prompt (after BOS). Assumes a is a
    single token; returns the first position whose token contains a digit. Mirrors
    run_helix_fit / run_causal_validation so the scripts read the same site."""
    for i, t in enumerate(model.to_str_tokens(prompt)):
        if any(c.isdigit() for c in t):
            return i
    raise ValueError(f"could not locate operand a in {model.to_str_tokens(prompt)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=360)
    ap.add_argument("--layer", type=int, default=None,
                    help="if set, analyze resid_post at this layer; else token embeddings")
    ap.add_argument("--summary", action="store_true",
                    help="sweep resid_post across all layers and draw the [zhou2024] "
                         "Fig 3 layer x frequency heatmap (single panel; resid_post has "
                         "no MLP/attn split). Overrides --layer. Embeddings are one "
                         "point, so excluded from the sweep.")
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
    ap.add_argument("--context", choices=["bare", "addition"], default="bare",
                    help="bare: isolated number token ' {a}'. addition: operand-a token "
                         "inside an '{a}+{b}=' prompt ([kantamneni2025] §4.3). Only applies "
                         "when --layer is set — token embeddings are context-free, so "
                         "--context is ignored for the embedding site.")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for '{a}+{b}=' prompts when --context "
                         "addition and --layer is set.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = models.load_model(args.model)
    tok_map = models.number_token_ids(model, args.lo, args.hi)
    values = np.array(sorted(tok_map))
    # Keep the contiguous prefix from the first value; stop at the first gap. The DFT
    # assumes an evenly sampled integer grid, so a holey set silently corrupts the
    # spectrum. Warn so dropped numbers are visible.
    if values.size:
        gaps = np.where(np.diff(values) != 1)[0]
        if gaps.size:
            cut = int(gaps[0]) + 1
            print(f"[warn] gap after {values[cut-1]} (next single-token value is "
                  f"{values[cut]}); dropping {len(values) - cut} value(s), using "
                  f"contiguous {values[0]}..{values[cut-1]} ({cut} numbers)")
            values = values[:cut]

    if args.summary:
        _run_summary(model, args, values)
        return

    if args.layer is None:
        # Embedding matrix rows for the number tokens — context-free by construction.
        if args.context == "addition":
            print("[warn] --context addition has no effect on the embedding site "
                  "(token embeddings carry no prompt context); analyzing W_E rows.")
        ids = torch.tensor([tok_map[int(v)] for v in values], device=model.cfg.device)
        acts = model.W_E[ids].float().cpu().numpy()    # (N, d)
        site = "embedding"
        ctx_tag = "bare"
    else:
        if args.context == "addition":
            # operand-a token inside '{a}+{b}=' (b fixed -> equal-length, constant index)
            prompts = [f"{n}+{args.b_fixed}=" for n in values]
            token_index = _operand_a_index(model, prompts[0])
            print(f"[context=addition] b={args.b_fixed}; operand-a token index={token_index}")
        else:
            prompts = [f" {n}" for n in values]        # bare number; space-prefixed last token
            token_index = -1
        hook = f"blocks.{args.layer}.hook_resid_post"
        acts = cache_number_site_all_layers(model, prompts, [hook],
                                            token_index=token_index)[hook]  # batched fwd
        site = f"resid_post.L{args.layer}"
        ctx_tag = args.context

    spec = fourier.number_dft(acts, values)
    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier/{ctx_tag}",
                         meta={"script": "run_fourier.py", "context": ctx_tag, "site": site})
    summary = {
        "model": args.model, "site": site, "context": ctx_tag,
        "b_fixed": args.b_fixed if (args.layer is not None and args.context == "addition") else None,
        "n_numbers": len(values),
        "value_range": [int(values[0]), int(values[-1])],
        "dominant_periods_top10": [float(p) for p in spec["dominant_periods"]],
    }
    # Folder already encodes script + context (run_fourier/<ctx_tag>); the file name only
    # needs the site (embedding vs resid_post.L*), which the path does not say.
    (out / f"{site}.json").write_text(json.dumps(summary, indent=2))
    _plot(spec, out / f"{site}.png", args.model, site, ctx_tag)
    print(f"[done] dominant periods (top 5): {summary['dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out} (site={site}, context={ctx_tag})")


def _run_summary(model, args, values):
    """[zhou2024] Fig 3 for the residual stream: sweep resid_post across layers and draw
    a single layer x frequency heatmap (resid_post has no MLP/attn split)."""
    if args.context == "addition":
        prompts = [f"{int(v)}+{args.b_fixed}=" for v in values]
        token_index = _operand_a_index(model, prompts[0])
        print(f"[context=addition] b={args.b_fixed}; operand-a token index={token_index}")
    else:
        prompts = [f" {int(v)}" for v in values]
        token_index = -1
        print(f"[context=bare] {len(prompts)} isolated number prompts")

    lo, hi = (args.layers if args.layers is not None else (0, model.cfg.n_layers - 1))
    if not (0 <= lo <= hi <= model.cfg.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{model.cfg.n_layers - 1}]")
    layers = list(range(lo, hi + 1))
    hooks = [f"blocks.{L}.hook_resid_post" for L in layers]
    caches = cache_number_site_all_layers(model, prompts, hooks, token_index=token_index)
    specs = [fourier.number_dft(caches[f"blocks.{L}.hook_resid_post"], values)
             for L in layers]
    mat, freqs = plotting.stack_power(specs)

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier/{args.context}",
                         meta={"script": "run_fourier.py", "context": args.context,
                               "site": "resid_post.summary", "summary": True})
    plotting.plot_layer_freq_heatmap(
        [("resid_post", mat)], freqs, layers, args.context,
        out / "summary_resid_post.png", model=args.model, value_unit="activation",
        transform=args.power_transform, cmap=args.cmap,
        vmax_percentile=args.vmax_percentile,
        title=f"Residual-stream Fourier components across layers — {args.model} "
              f"(context={args.context})")
    summary = {
        "model": args.model, "site": "resid_post.summary", "context": args.context,
        "transform": args.power_transform, "layers": layers,
        "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "freqs": [float(f) for f in freqs],
        "dominant_periods_by_layer": {
            int(L): [float(p) for p in s["dominant_periods"][:5]]
            for L, s in zip(layers, specs)},
    }
    (out / "summary_resid_post.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary heatmap over layers {lo}..{hi} "
          f"(transform={args.power_transform})")
    print(f"[done] wrote {out}/summary_resid_post.png (context={args.context})")


def _plot(spec, path, model, site, context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(spec["freqs"], spec["power"], marker="o", ms=2)
    plt.xlabel("frequency (cycles / integer)"); plt.ylabel("mean power")
    plt.title(f"Number DFT — {model} — {site} (context={context})"); plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
