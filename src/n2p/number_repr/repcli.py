"""Shared CLI + plotting helpers for the week-1 number-representation scripts
(run_fourier, run_fourier_components, run_fourier_components_raw, run_helix_fit).

Extracted to remove the duplicated _contiguous_prefix / _framings_for_summary /
_annotate / _plot and the repeated --summary argparse block that the four scripts
otherwise carried verbatim (lint 2026-06-21). Pure numpy + matplotlib (imported lazily);
no torch, so it stays importable on CPU-only machines.
"""
from __future__ import annotations

import numpy as np

from n2p import tasks


def contiguous_prefix(values):
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


def framings_for_summary(operation, read_token):
    """All framings of `operation` compatible with `read_token` (skip no-b framings when
    reading the b operand). Used by every --summary path."""
    out = []
    for f in tasks.FRAMING_NAMES:
        if f not in tasks.FRAMINGS[operation]:
            continue
        if read_token == "b" and not tasks.template_has_b(operation, f):
            print(f"[skip] {operation}/{f}: --read-token b invalid (no operand b)")
            continue
        out.append(f)
    return out


def add_summary_args(ap):
    """Register the shared across-layers summary flags ([zhou2024] Fig 3) on a parser.
    Dest names: summary, layers, power_transform, cmap, vmax_percentile."""
    ap.add_argument("--summary", action="store_true",
                    help="sweep every layer; draw the [zhou2024] Fig 3 layer x frequency "
                         "heatmap with one panel PER FRAMING.")
    ap.add_argument("--layers", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--summary only: restrict to the inclusive layer band [LO,HI].")
    ap.add_argument("--power-transform", choices=["amplitude", "power", "log"],
                    default="amplitude", dest="power_transform",
                    help="--summary colour scale (default amplitude=sqrt(mean power)).")
    ap.add_argument("--cmap", default="inferno_r", help="--summary colormap.")
    ap.add_argument("--vmax-percentile", type=float, default=99.5, dest="vmax_percentile",
                    help="--summary robust colour-limit percentile (default 99.5).")


def annotate_periods(ax, spec, k=6):
    """Mark the top-k dominant components (by power) with their period on a spectrum axis."""
    for idx in spec["dominant_freq_idx"][:k]:
        f = spec["freqs"][idx]
        if f <= 0:
            continue
        ax.annotate(f"T={1.0/f:.2f}", (f, spec["power"][idx]),
                    textcoords="offset points", xytext=(0, 4), fontsize=7, ha="center")


def plot_component_spectra(specs, path, *, model, layer, operation, framing, read_token,
                           value_unit):
    """Two-panel MLP | attention per-layer spectrum plot, shared by run_fourier_components
    (value_unit='logit') and run_fourier_components_raw (value_unit='activation')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
    for ax, name, title in ((axes[0], "mlp", "MLP output"),
                            (axes[1], "attn", "Attention output")):
        s = specs[name]
        ax.plot(s["freqs"], s["power"], marker="o", ms=2)
        annotate_periods(ax, s)
        ax.set_xlabel("frequency (cycles / integer)")
        ax.set_title(f"{title} — L{layer}")
    axes[0].set_ylabel(f"mean {value_unit} power")
    fig.suptitle(f"Component-output {value_unit}s in Fourier space — {model} — L{layer} "
                 f"({operation}/{framing}, read={read_token})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_single_spectrum(spec, path, *, model, site, operation, framing, read_token):
    """Single-axis per-layer spectrum plot (run_fourier: resid_post / embedding)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(spec["freqs"], spec["power"], marker="o", ms=2)
    plt.xlabel("frequency (cycles / integer)")
    plt.ylabel("mean power")
    plt.title(f"Number DFT — {model} — {site} ({operation}/{framing}, read={read_token})")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
