"""Week 1 — Fourier-feature check [zhou2024].

Take the per-number vectors (default: the token EMBEDDINGS, where the mechanism is
claimed to originate; optionally a residual-stream layer) and show the power spectrum
is sparse, with dominant low-freq (magnitude) and high-freq (modular) components.

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python experiments/week1_number_representation/run_fourier.py --model gptj
    python experiments/week1_number_representation/run_fourier.py --model gptj --layer 16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p import config, models                       # noqa: E402
from n2p.number_repr import fourier                    # noqa: E402
from n2p.number_repr.causal import cache_number_site_all_layers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gptj")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=360)
    ap.add_argument("--layer", type=int, default=None,
                    help="if set, analyze resid_post at this layer; else token embeddings")
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

    if args.layer is None:
        # Embedding matrix rows for the number tokens.
        ids = torch.tensor([tok_map[int(v)] for v in values], device=model.cfg.device)
        acts = model.W_E[ids].float().cpu().numpy()    # (N, d)
        site = "embedding"
    else:
        prompts = [f" {n}" for n in values]
        hook = f"blocks.{args.layer}.hook_resid_post"
        acts = cache_number_site_all_layers(model, prompts, [hook])[hook]  # batched fwd
        site = f"resid_post.L{args.layer}"

    spec = fourier.number_dft(acts, values)
    out = config.run_dir("week1_number_representation", args.seed)
    summary = {
        "model": args.model, "site": site, "n_numbers": len(values),
        "dominant_periods_top10": [float(p) for p in spec["dominant_periods"]],
    }
    (out / f"fourier_{site}.json").write_text(json.dumps(summary, indent=2))
    _plot(spec, out / f"fourier_{site}.png", args.model, site)
    print(f"[done] dominant periods (top 5): {summary['dominant_periods_top10'][:5]}")
    print(f"[done] wrote {out}")


def _plot(spec, path, model, site):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(spec["freqs"], spec["power"], marker="o", ms=2)
    plt.xlabel("frequency (cycles / integer)"); plt.ylabel("mean power")
    plt.title(f"Number DFT — {model} — {site}"); plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
