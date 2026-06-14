"""Generalized-helix basis and fit, following [kantamneni2025] ("Language Models Use
Trigonometry to Do Addition", arXiv:2502.00873).

A number a is hypothesized to be represented as a generalized helix:
    B(a) = [ a,  cos(2pi a / T_1), sin(2pi a / T_1),  ...,  cos(2pi a / T_k), sin(2pi a / T_k) ]
with periods T = [2, 5, 10, 100] (the paper's default). We fit a linear map between
this analytic basis and the model's residual-stream representation of the number
token, in a low-dimensional PCA subspace, and report how well the helix explains it.

This module is pure numpy/torch on cached activations — no model forward here.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PERIODS = (2, 5, 10, 100)


def helix_basis(values: np.ndarray, periods=DEFAULT_PERIODS) -> np.ndarray:
    """Return B with shape (len(values), 1 + 2*len(periods)).

    Column 0 is the linear magnitude term; then (cos, sin) per period.
    """
    values = np.asarray(values, dtype=np.float64)
    cols = [values]  # linear term
    for T in periods:
        ang = 2.0 * np.pi * values / T
        cols.append(np.cos(ang))
        cols.append(np.sin(ang))
    return np.stack(cols, axis=1)


def _design_matrix(values, periods):
    """Helix basis with an intercept column appended (for fitting). The intercept
    lets the fit match mean-centered PCA targets; without it the (uncentered) linear
    magnitude term cannot reproduce a centered target. Shape (N, helix_dim + 1)."""
    B = helix_basis(values, periods)                # (N, helix_dim)
    ones = np.ones((B.shape[0], 1))
    return np.concatenate([B, ones], axis=1)        # last column = intercept


def fit_helix(
    acts: np.ndarray,
    values: np.ndarray,
    periods=DEFAULT_PERIODS,
    n_pca: int = 9,
):
    """Fit the helix to per-number activations.

    Args:
        acts:   (N, d_model) residual-stream vectors for N numbers (one layer/site).
        values: (N,) the integer each row corresponds to.
        n_pca:  dimensionality of the PCA subspace the helix is fit inside
                (paper uses a low-D subspace; 9 is a reasonable default — 1 linear
                + 4 periods * 2).

    Returns dict with the fitted linear map C (design_dim -> n_pca), the PCA object,
    the reconstruction R^2 in PCA space, and the per-number residuals.

    Method (paper, App. C / our few-readings-circuit-finding-5 derivation):
      1. center activations, PCA to n_pca dims  -> H_pca (N, n_pca)
      2. build helix design matrix B|1 (N, design_dim)
      3. least-squares  H_pca ~ (B|1) @ C        -> C (design_dim, n_pca)
      4. R^2 of the fit = variance of H_pca explained by the fit.
    Use `helix_coords(values, fitres)` to get target coordinates for a new value.
    """
    from sklearn.decomposition import PCA

    acts = np.asarray(acts, dtype=np.float64)
    mu = acts.mean(0, keepdims=True)
    pca = PCA(n_components=n_pca)
    H_pca = pca.fit_transform(acts - mu)            # (N, n_pca)

    X = _design_matrix(values, periods)             # (N, design_dim) incl. intercept
    C, *_ = np.linalg.lstsq(X, H_pca, rcond=None)   # (design_dim, n_pca)
    H_hat = X @ C
    ss_res = ((H_pca - H_hat) ** 2).sum()
    ss_tot = ((H_pca - H_pca.mean(0)) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "C": C, "pca": pca, "mu": mu, "periods": list(periods),
        "n_pca": n_pca, "r2": float(r2),
        "H_pca": H_pca, "H_hat": H_hat,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def helix_coords(values, fitres) -> np.ndarray:
    """Coordinates (in the fitted PCA subspace) the helix predicts for `values`.
    Shape (len(values), n_pca). This is what the causal patch writes as the clean
    target for a different value a'."""
    X = _design_matrix(values, fitres["periods"])
    return X @ fitres["C"]


def baseline_pca_r2(acts: np.ndarray, values: np.ndarray, n_pca: int = 9) -> float:
    """Matched-dimension control: how well does a generic degree-k polynomial of `a`
    (same #params as the helix) explain the same PCA subspace? The helix is only
    interesting if it MATCHES or BEATS this despite being more constrained
    (the [kantamneni2025] capacity argument; cf. few-readings-circuit-finding-5).
    """
    from sklearn.decomposition import PCA

    acts = np.asarray(acts, dtype=np.float64)
    H_pca = PCA(n_components=n_pca).fit_transform(acts - acts.mean(0, keepdims=True))
    deg = 2 * len(DEFAULT_PERIODS)  # match helix param count
    V = np.vander(np.asarray(values, float), N=deg + 1, increasing=True)
    Vh, *_ = np.linalg.lstsq(V, H_pca, rcond=None)
    res = ((H_pca - V @ Vh) ** 2).sum()
    tot = ((H_pca - H_pca.mean(0)) ** 2).sum()
    return float(1.0 - res / max(tot, 1e-12))
