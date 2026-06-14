"""Fourier-feature analysis following [zhou2024] ("Pre-trained LMs use Fourier
Features to Compute Addition", NeurIPS 2024, arXiv:2406.03445).

Claim: the number representation is SPARSE IN THE FREQUENCY DOMAIN, and this
structure originates in the pre-trained number embeddings. We take the matrix of
per-number vectors (embeddings, or any per-number residual-stream site), run a DFT
along the number axis, and look for a small set of dominant frequencies (low-freq =
magnitude, high-freq = modular).
"""
from __future__ import annotations

import numpy as np


def number_dft(acts: np.ndarray, values: np.ndarray):
    """DFT of per-number vectors along the number axis.

    Args:
        acts:   (N, d) vectors, row i corresponds to number values[i].
        values: (N,) integers; MUST be a contiguous range for a clean DFT
                (e.g. 0..N-1). Non-contiguous input is sorted & checked.
    Returns dict with the power spectrum averaged over the d feature dims and the
    ranked dominant frequencies.
    """
    acts = np.asarray(acts, dtype=np.float64)
    values = np.asarray(values)
    order = np.argsort(values)
    values, acts = values[order], acts[order]
    if not np.array_equal(values, np.arange(values[0], values[0] + len(values))):
        raise ValueError("number_dft expects a contiguous integer range of values.")

    acts = acts - acts.mean(0, keepdims=True)
    F = np.fft.rfft(acts, axis=0)               # (N//2+1, d)
    power = (np.abs(F) ** 2).mean(axis=1)        # avg power per frequency over dims
    freqs = np.fft.rfftfreq(len(values), d=1.0)  # cycles per integer step
    ranked = np.argsort(power)[::-1]
    return {
        "freqs": freqs, "power": power,
        "dominant_freq_idx": ranked.tolist(),
        "dominant_periods": [(1.0 / freqs[i] if freqs[i] > 0 else np.inf)
                             for i in ranked[:10]],
    }


def split_low_high(acts: np.ndarray, values: np.ndarray, cutoff_period: float = 10.0):
    """Separate the magnitude (low-frequency) and modular (high-frequency) parts,
    the separability [zhou2024] Table 1 relies on. Returns (low_recon, high_recon)
    with the same shape as `acts` (a clean target for the stub's magnitude swap)."""
    acts = np.asarray(acts, dtype=np.float64)
    order = np.argsort(values)
    inv = np.argsort(order)
    A = acts[order] - acts.mean(0, keepdims=True)
    F = np.fft.rfft(A, axis=0)
    freqs = np.fft.rfftfreq(A.shape[0], d=1.0)
    keep_low = freqs <= (1.0 / cutoff_period)
    F_low = F.copy(); F_low[~keep_low] = 0
    F_high = F.copy(); F_high[keep_low] = 0
    low = np.fft.irfft(F_low, n=A.shape[0], axis=0)[inv]
    high = np.fft.irfft(F_high, n=A.shape[0], axis=0)[inv]
    return low, high
