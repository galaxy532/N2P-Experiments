"""Shared plotting for the Fourier sweeps — the layer x frequency "summary" view.

Reproduces the across-layers summary of [zhou2024] Figure 3: instead of one power
spectrum for a single layer, build a 2-D map with LAYER on the x-axis, FREQUENCY on the
y-axis, and colour = the Fourier-component magnitude at that (layer, frequency). For the
component scripts that split MLP vs attention, the two maps are drawn SIDE BY SIDE,
exactly as in Figure 3. Outlier components are expected at periods ~2, 2.5, 5, 10.

Colour axis (default ``amplitude``): ``number_dft`` returns, per frequency k,
``power_k = mean_dim |F_k|^2`` — the mean over the d feature dims of the squared
Fourier-coefficient magnitude. Its square root ``sqrt(power_k) = ||C_k||_2 / sqrt(d)``
is the per-dimension RMS amplitude, i.e. proportional to the L2 norm of the
frequency-k coefficient VECTOR C_k. Plotting that on a LINEAR colour scale shows true
component magnitudes while compressing the low-frequency (magnitude) dominance enough to
keep the modular peaks visible. ``power`` (raw, low-freq dominates) and ``log`` are
available via the same switch.
"""
from __future__ import annotations

import numpy as np

# Periods called out in [zhou2024] (Figs 2-3): magnitude-ish low freq + modular peaks.
REFERENCE_PERIODS = (2.0, 2.5, 5.0, 10.0)


def apply_transform(power: np.ndarray, transform: str = "amplitude") -> np.ndarray:
    """Map mean-power -> the value shown in the heatmap.

    - ``amplitude`` (default): ``sqrt(power)`` = ||C_k||_2 / sqrt(d), an L2-norm-like
      amplitude on a linear scale.
    - ``power``: raw mean power (energy); the low-frequency peak dominates.
    - ``log``: ``log10(power)`` for a wide dynamic range.
    """
    power = np.asarray(power, dtype=np.float64)
    if transform == "amplitude":
        return np.sqrt(np.clip(power, 0, None))
    if transform == "power":
        return power
    if transform == "log":
        return np.log10(np.clip(power, 1e-12, None))
    raise ValueError(f"unknown transform {transform!r}; use amplitude|power|log")


def stack_power(specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of ``fourier.number_dft`` results (one per layer, in layer order)
    into a (n_layers, n_freqs) power matrix. All layers must share the frequency grid
    (they do when the number axis is the same length across layers)."""
    if not specs:
        raise ValueError("stack_power got no per-layer specs")
    freqs = np.asarray(specs[0]["freqs"])
    for s in specs:
        if not np.array_equal(np.asarray(s["freqs"]), freqs):
            raise ValueError("all layers must share the same frequency grid for the "
                             "summary heatmap (equal number-axis length).")
    M = np.stack([np.asarray(s["power"]) for s in specs], axis=0)  # (n_layers, n_freqs)
    return M, freqs


def _transform_label(transform: str, value_unit: str) -> str:
    if transform == "amplitude":
        return f"||C_k||  (√mean {value_unit} power)"
    if transform == "power":
        return f"mean {value_unit} power"
    return f"log10 mean {value_unit} power"


def _draw_panel(ax, layers, freqs, matrix, transform, cmap, vmin, vmax):
    """One layer x frequency heatmap. ``matrix`` is (n_layers, n_freqs)."""
    Z = apply_transform(matrix, transform).T            # -> (n_freqs, n_layers)
    # pcolormesh edges so each cell is centred on its (layer, freq).
    layers = np.asarray(layers, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    x_edges = np.concatenate([[layers[0] - 0.5],
                              (layers[:-1] + layers[1:]) / 2.0,
                              [layers[-1] + 0.5]])
    if freqs.size > 1:
        df = freqs[1] - freqs[0]
        y_edges = np.concatenate([[freqs[0] - df / 2], (freqs[:-1] + freqs[1:]) / 2,
                                  [freqs[-1] + df / 2]])
    else:
        y_edges = np.array([freqs[0] - 0.5, freqs[0] + 0.5])
    mesh = ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="flat")
    # Mark the reference periods (freq = 1/period) on a right-hand period axis.
    for p in REFERENCE_PERIODS:
        f = 1.0 / p
        if freqs[0] <= f <= freqs[-1]:
            ax.axhline(f, color="white", lw=0.5, ls="--", alpha=0.45)
    ax.set_xlabel("layer index")
    return mesh


def plot_layer_freq_heatmap(panels, freqs, layers, context, path, *, model,
                            value_unit="logit", transform="amplitude", title=None,
                            cmap="magma"):
    """Draw the Figure-3-style summary and save it to ``path``.

    Args:
        panels:     list of ``(panel_title, power_matrix)`` where ``power_matrix`` is
                    (n_layers, n_freqs) mean power. One entry -> single panel; two
                    entries -> side-by-side (e.g. ("MLP output", ...), ("Attention
                    output", ...)).
        freqs:      (n_freqs,) frequency grid (cycles / integer), shared by all panels.
        layers:     (n_layers,) layer indices on the x-axis.
        context:    "bare" | "addition" — recorded in the title (the addition/bare
                    context this summary was computed in).
        value_unit: "logit" or "activation" — only affects the colourbar label.
        transform:  amplitude | power | log (see ``apply_transform``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Shared colour range across panels so MLP and attn are directly comparable.
    transformed = [apply_transform(m, transform) for _, m in panels]
    vmin = float(min(t.min() for t in transformed))
    vmax = float(max(t.max() for t in transformed))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.2), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]
    mesh = None
    for ax, (panel_title, matrix) in zip(axes, panels):
        mesh = _draw_panel(ax, layers, freqs, matrix, transform, cmap, vmin, vmax)
        ax.set_title(panel_title)
    axes[0].set_ylabel("frequency (cycles / integer)")
    cbar = fig.colorbar(mesh, ax=list(axes), fraction=0.046, pad=0.02)
    cbar.set_label(_transform_label(transform, value_unit))
    fig.suptitle(title or f"Fourier components across layers — {model} "
                 f"(context={context})")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
