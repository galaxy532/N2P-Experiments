"""Week 1 — Fourier-feature check [zhou2024], activation space.

Take the per-number vectors (default: the token EMBEDDINGS, where the mechanism is
claimed to originate; optionally a residual-stream layer) and show the power spectrum is
sparse, with dominant low-freq (magnitude) and high-freq (modular) components. We DFT the
RAW activation over a sweep of the operand `a` (no unembedding) — the object relevant to
operand-subspace feature tracking. See the wiki exp-note for why activations are read at
the operand token (not unembedded).

Prompt framing is selected by --context (template_1..4, see n2p.number_repr.prompts) and
the analyzed position by --read-token {a,b,sum}; the operand `a` is swept while `b` is
FIXED (`--b_fixed`). Because the model is causal, only templates that put context BEFORE
`a` can change the operand representation.

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_fourier.py --model gptj                              # embeddings
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --layer 16                   # resid_post L16, operand-a token, template_1
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --layer 16 --context template_3 --read-token a
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --summary --context template_3 --read-token sum

NOTE on --hi 360: with --lo 0 inclusive, 0..360 is 361 points (prime), so periods
2/2.5/5/10 do not land on DFT bins and read slightly off (e.g. 10.03). Use --lo 1 --hi 360
or --lo 0 --hi 359 for exact-integer periods. Default kept at 360.
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
    ap.add_argument("--context", choices=prompts.TEMPLATE_CHOICES, default="template_1",
                    help="prompt framing (see n2p.number_repr.prompts). template_1 is the "
                         "bare operand baseline; template_3/4 put addition context before "
                         "a. Ignored for the embedding site (W_E rows are context-free).")
    ap.add_argument("--read-token", choices=prompts.READ_TOKEN_CHOICES, default="a",
                    dest="read_token",
                    help="which position to analyze: a/b = operand tokens, sum = last "
                         "token. Embedding site only supports 'a' (the swept operand).")
    ap.add_argument("--b_fixed", type=int, default=5,
                    help="fixed second operand b for templates that use it (templates 3,4).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    prompts.validate_read_token(args.read_token, args.context)

    model = models.load_model(args.model)
    tok_map = models.number_token_ids(model, args.lo, args.hi)
    values = _contiguous_prefix(np.array(sorted(tok_map)))

    if args.summary:
        _run_summary(model, args, values)
        return

    if args.layer is None:
        # Embedding matrix rows for the swept operand — context-free by construction.
        if args.context != "template_1":
            print(f"[warn] --context {args.context} has no effect on the embedding site "
                  "(token embeddings carry no prompt context); analyzing W_E rows.")
        if args.read_token != "a":
            print(f"[warn] embedding site only varies with the swept operand; "
                  f"--read-token {args.read_token} treated as 'a' (W_E rows).")
        ids = torch.tensor([tok_map[int(v)] for v in values], device=model.cfg.device)
        acts = model.W_E[ids].float().cpu().numpy()    # (N, d)
        site = "embedding"
        read_token = "a"
    else:
        prompt_list = prompts.build_prompts(args.context, values, args.b_fixed)
        token_index = prompts.read_token_index(model, prompt_list[0], args.read_token,
                                               args.context)
        print(f"[{args.context}] read-token={args.read_token} at index {token_index}; "
              f"b={args.b_fixed if prompts.template_has_b(args.context) else '-'}")
        hook = f"blocks.{args.layer}.hook_resid_post"
        acts = cache_number_site_all_layers(model, prompt_list, [hook],
                                            token_index=token_index)[hook]
        site = f"resid_post.L{args.layer}"
        read_token = args.read_token

    spec = fourier.number_dft(acts, values)
    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier/{args.context}",
                         meta={"script": "run_fourier.py", "context": args.context,
                               "site": site, "read_token": read_token})
    summary = {
        "model": args.model, "site": site, "context": args.context,
        "read_token": read_token,
        "b_fixed": args.b_fixed if (args.layer is not None
                                    and prompts.template_has_b(args.context)) else None,
        "n_numbers": len(values),
        "value_range": [int(values[0]), int(values[-1])],
        "dominant_periods_top10": [float(p) for p in spec["dominant_periods"]],
    }
    # Folder encodes model + script + template; the file name carries the site + read-token.
    (out / f"{site}.{read_token}.json").write_text(json.dumps(summary, indent=2))
    _plot(spec, out / f"{site}.{read_token}.png", args.model, site, args.context, read_token)
    print(f"[done] dominant periods (top 5): {summary['dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out} (site={site}, context={args.context}, read={read_token})")


def _run_summary(model, args, values):
    """[zhou2024] Fig 3 for the residual stream: sweep resid_post across layers and draw
    a single layer x frequency heatmap (resid_post has no MLP/attn split)."""
    prompt_list = prompts.build_prompts(args.context, values, args.b_fixed)
    token_index = prompts.read_token_index(model, prompt_list[0], args.read_token,
                                           args.context)
    print(f"[{args.context}] read-token={args.read_token} at index {token_index}; "
          f"b={args.b_fixed if prompts.template_has_b(args.context) else '-'}")

    lo, hi = (args.layers if args.layers is not None else (0, model.cfg.n_layers - 1))
    if not (0 <= lo <= hi <= model.cfg.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{model.cfg.n_layers - 1}]")
    layers = list(range(lo, hi + 1))
    hooks = [f"blocks.{L}.hook_resid_post" for L in layers]
    caches = cache_number_site_all_layers(model, prompt_list, hooks, token_index=token_index)
    specs = [fourier.number_dft(caches[f"blocks.{L}.hook_resid_post"], values)
             for L in layers]
    mat, freqs = plotting.stack_power(specs)

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier/{args.context}",
                         meta={"script": "run_fourier.py", "context": args.context,
                               "site": "resid_post.summary", "read_token": args.read_token,
                               "summary": True})
    plotting.plot_layer_freq_heatmap(
        [("resid_post", mat)], freqs, layers, args.context,
        out / f"summary_resid_post.{args.read_token}.png", model=args.model,
        value_unit="activation", transform=args.power_transform, cmap=args.cmap,
        vmax_percentile=args.vmax_percentile,
        title=f"Residual-stream Fourier components across layers — {args.model} "
              f"— {args.context} — read={args.read_token}")
    summary = {
        "model": args.model, "site": "resid_post.summary", "context": args.context,
        "read_token": args.read_token, "transform": args.power_transform, "layers": layers,
        "b_fixed": args.b_fixed if prompts.template_has_b(args.context) else None,
        "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "freqs": [float(f) for f in freqs],
        "dominant_periods_by_layer": {
            int(L): [float(p) for p in s["dominant_periods"][:5]]
            for L, s in zip(layers, specs)},
    }
    (out / f"summary_resid_post.{args.read_token}.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary heatmap over layers {lo}..{hi} "
          f"(transform={args.power_transform}, read={args.read_token})")
    print(f"[done] wrote {out}/summary_resid_post.{args.read_token}.png "
          f"(context={args.context})")


def _plot(spec, path, model, site, context, read_token):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(spec["freqs"], spec["power"], marker="o", ms=2)
    plt.xlabel("frequency (cycles / integer)"); plt.ylabel("mean power")
    plt.title(f"Number DFT — {model} — {site} ({context}, read={read_token})")
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
