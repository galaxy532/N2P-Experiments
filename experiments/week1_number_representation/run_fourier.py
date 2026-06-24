"""Week 1 — Fourier-feature check [zhou2024], activation space.

Take the per-number vectors (default: the token EMBEDDINGS, where the mechanism is
claimed to originate; optionally a residual-stream layer) and show the power spectrum is
sparse, with dominant low-freq (magnitude) and high-freq (modular) components. We DFT the
RAW activation over a sweep of the operand `a` (no unembedding).

Prompt surface form: --operation picks the arithmetic operation, --framing one of its
three framings (symbolic / word / wordproblem, see n2p.tasks). --read-token {a,b,sum}
selects the position; `a` is swept while `b` is FIXED (--b_fixed). --summary sweeps
resid_post across layers and draws the [zhou2024] Fig-3 layer x frequency heatmap with one
panel PER FRAMING (single map — resid_post has no MLP/attn split).

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 experiments/week1_number_representation/run_fourier.py --model gptj                                  # embeddings
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --layer 16 --operation addition --framing symbolic --read-token a
    python3 experiments/week1_number_representation/run_fourier.py --model gptj --summary --operation addition --read-token a

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
from n2p import config, models, tasks                 # noqa: E402
from n2p.number_repr import fourier, plotting, repcli   # noqa: E402
from n2p.number_repr.causal import cache_number_site_all_layers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=360)
    ap.add_argument("--layer", type=int, default=None,
                    help="if set, analyze resid_post at this layer; else token embeddings")
    repcli.add_summary_args(ap)
    ap.add_argument("--operation", choices=tasks.OPERATION_CHOICES, default="addition",
                    help="arithmetic operation whose framings are analyzed.")
    ap.add_argument("--framing", choices=tasks.FRAMING_NAMES, default="symbolic",
                    help="per-layer runs only: which framing to analyze (ignored under "
                         "--summary, which sweeps all three). Ignored for the embedding "
                         "site (W_E rows are context-free).")
    ap.add_argument("--read-token", choices=tasks.READ_TOKEN_CHOICES, default="a",
                    dest="read_token",
                    help="a/b = operand tokens, sum = last token. Embedding site only "
                         "supports 'a' (the swept operand).")
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

    model = models.load_model(args.model)
    prefix = args.prefix if args.prefix is not None else config.get_model_spec(args.model).prompt_prefix
    shots = tasks.fewshot_shots(args.operation, args.kshot, args.seed)
    print(f"[prefix] {prefix!r}  [kshot] {args.kshot}")
    # Operand grid validated against the real prompt (canonical symbolic framing; operand
    # single-token-ness is framing-independent). id_map -> in-context vocab id for W_E rows.
    grid_values, id_map = tasks.single_token_number_grid(model, args.operation, "symbolic",
                                                        args.lo, args.hi, b=args.b_fixed)
    values = repcli.contiguous_prefix(np.array(grid_values))

    out = config.run_dir("week1_number_representation", args.seed,
                         model=args.model,
                         label=f"run_fourier/{args.operation}",
                         meta={"script": "run_fourier.py", "operation": args.operation,
                               "read_token": args.read_token,
                               "summary": bool(args.summary),
                               "framing": None if args.summary else args.framing})

    if args.summary:
        _run_summary(model, args, values, out, prefix, shots)
        return

    if args.layer is None:
        # Embedding matrix rows for the swept operand — context-free by construction.
        if args.read_token != "a":
            print(f"[warn] embedding site only varies with the swept operand; "
                  f"--read-token {args.read_token} treated as 'a' (W_E rows).")
        ids = torch.tensor([id_map[int(v)] for v in values], device=model.cfg.device)
        acts = model.W_E[ids].float().cpu().numpy()
        site, read_token, framing = "embedding", "a", "embedding"
    else:
        tasks.validate_read_token(args.read_token, args.operation, args.framing)
        prompt_list = tasks.build_prompts(args.operation, args.framing, values, args.b_fixed,
                                          prefix=prefix, shots=shots)
        token_index = tasks.read_token_index(model, prompt_list[0], args.read_token,
                                             args.operation, args.framing)
        print(f"[{args.operation}/{args.framing}] read-token={args.read_token} at index "
              f"{token_index}; b={args.b_fixed if tasks.template_has_b(args.operation, args.framing) else '-'}")
        hook = f"blocks.{args.layer}.hook_resid_post"
        acts = cache_number_site_all_layers(model, prompt_list, [hook],
                                            token_index=token_index)[hook]
        site, read_token, framing = f"resid_post.L{args.layer}", args.read_token, args.framing

    spec = fourier.number_dft(acts, values)
    summary = {
        "model": args.model, "site": site, "operation": args.operation, "framing": framing,
        "read_token": read_token,
        "b_fixed": args.b_fixed if args.layer is not None else None,
        "n_numbers": len(values), "value_range": [int(values[0]), int(values[-1])],
        "dominant_periods_top10": [float(p) for p in spec["dominant_periods"]],
    }
    stem = f"{site}.{framing}.{read_token}"
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    repcli.plot_single_spectrum(spec, out / f"{stem}.png", model=args.model, site=site,
                                operation=args.operation, framing=framing,
                                read_token=read_token)
    print(f"[done] dominant periods (top 5): {summary['dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out} (site={site}, op={args.operation}, framing={framing}, "
          f"read={read_token})")


def _run_summary(model, args, values, out, prefix="", shots=()):
    """[zhou2024] Fig 3 for the residual stream: one panel per framing, each a
    layer x frequency heatmap (resid_post has no MLP/attn split)."""
    framings = repcli.framings_for_summary(args.operation, args.read_token)
    if not framings:
        raise SystemExit("no framing compatible with --read-token for this operation")
    lo, hi = (args.layers if args.layers is not None else (0, model.cfg.n_layers - 1))
    if not (0 <= lo <= hi <= model.cfg.n_layers - 1):
        raise SystemExit(f"--layers {lo} {hi} out of range [0,{model.cfg.n_layers - 1}]")
    layers = list(range(lo, hi + 1))
    hooks = [f"blocks.{L}.hook_resid_post" for L in layers]

    panels, freqs, dom_by_framing = [], None, {}
    for framing in framings:
        prompt_list = tasks.build_prompts(args.operation, framing, values, args.b_fixed,
                                          prefix=prefix, shots=shots)
        token_index = tasks.read_token_index(model, prompt_list[0], args.read_token,
                                             args.operation, framing)
        caches = cache_number_site_all_layers(model, prompt_list, hooks,
                                              token_index=token_index)
        specs = [fourier.number_dft(caches[f"blocks.{L}.hook_resid_post"], values)
                 for L in layers]
        mat, freqs = plotting.stack_power(specs)
        panels.append((framing, mat))
        dom_by_framing[framing] = {int(L): [float(p) for p in s["dominant_periods"][:5]]
                                   for L, s in zip(layers, specs)}

    plotting.plot_layer_freq_heatmap(
        panels, freqs, layers, args.operation,
        out / f"summary_resid_post.{args.read_token}.png", model=args.model,
        value_unit="activation", transform=args.power_transform, cmap=args.cmap,
        vmax_percentile=args.vmax_percentile,
        title=f"Residual-stream Fourier components across layers — {args.model} "
              f"— {args.operation} — read={args.read_token}")
    summary = {
        "model": args.model, "site": "resid_post.summary", "operation": args.operation,
        "framings": framings, "read_token": args.read_token,
        "transform": args.power_transform, "layers": layers, "n_numbers": int(values.size),
        "value_range": [int(values[0]), int(values[-1])],
        "freqs": [float(f) for f in freqs],
        "dominant_periods_by_framing_layer": dom_by_framing,
    }
    (out / f"summary_resid_post.{args.read_token}.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] summary heatmap over layers {lo}..{hi}, framings={framings} "
          f"(read={args.read_token})")
    print(f"[done] wrote {out}/summary_resid_post.{args.read_token}.png")


if __name__ == "__main__":
    main()
