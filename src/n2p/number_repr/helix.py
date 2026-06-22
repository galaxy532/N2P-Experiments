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


def _design_row_indices(all_periods, periods, include_linear: bool,
                        include_intercept: bool) -> list[int]:
    """Indices into the design matrix (`_design_matrix`: [linear, cos T1, sin T1, ...,
    cos Tk, sin Tk, intercept]) for the requested helix components. The intercept (last
    column) is the DC term — include it so the patch reproduces the FULL helix
    reconstruction mu + helix_coords(a')@P (it is constant in `a`, but it sets the
    absolute level the model expects); group it with the magnitude/low-freq part."""
    idx = []
    if include_linear:
        idx.append(0)                       # linear magnitude term
    for i, T in enumerate(all_periods):
        if periods is None or T in periods:
            idx += [1 + 2 * i, 2 + 2 * i]   # (cos, sin) for period T
    if include_intercept:
        idx.append(1 + 2 * len(all_periods))  # intercept (DC) = last design column
    return idx


def helix_subspace_basis(fitres, periods=None, include_linear: bool = True,
                         include_intercept: bool = True, rtol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis (d_model, r) of the HELIX IMAGE in residual-stream space.

    `periods` selects which periodic components span the subspace (None = all fitted
    periods); `include_linear` toggles the linear magnitude term. This lets the caller
    patch only part of the helix — e.g. the MAGNITUDE part (include_linear=True,
    periods=(100,) — low frequency) vs the MODULAR part (include_linear=False,
    periods=(2,5,10) — high frequency), the separable split the stub story relies on
    [zhou2024 Table 1].

    The fit lives in PCA coordinates: for value v the helix predicts the PCA coords
    helix_coords(v) = X(v) @ C; lifted back to the residual stream these are
    helix_coords(v) @ P, with P = pca.components_ (n_pca, d_model). Every vector the
    helix can ever produce therefore lies in the span of the columns of
    (C @ P).T = P.T @ C.T; we orthonormalize them so a causal patch writes ONLY inside
    the helix subspace, not the full top-n_pca PCA subspace.

    Note: when the helix image fills the top-n_pca PCA subspace (rank(C) == n_pca AND
    every basis function is well-identified over the value range) this subspace
    COINCIDES with the old `components.T` basis, so patching is numerically identical.
    It diverges in two cases — both desirable: (1) n_pca exceeds the helix rank (the
    PCA subspace carries non-helix capacity we must NOT patch), and (2) a periodic basis
    function is near-degenerate over the sampled range (e.g. T=100 barely completes one
    period over a in [0,99], so its direction is collinear with the linear term and
    poorly identified). `rtol` drops those ill-conditioned directions so the patch only
    writes into genuinely helix-spanned directions. The dropped rank is reported by the
    caller; if it is far below 1 + 2*len(periods), reconsider the period set vs range
    (cf. the [zhou2024]/[kantamneni2025] §4.1 range-sensitivity caveat).
    """
    P = fitres["pca"].components_            # (n_pca, d_model), orthonormal rows
    C = fitres["C"]                          # (design_dim, n_pca)
    rows = _design_row_indices(fitres["periods"], periods, include_linear, include_intercept)
    M = P.T @ C[rows].T                      # (d_model, len(rows)): selected helix directions
    # SVD (rank-revealing) so degenerate directions — e.g. sin(2*pi*a/2)==0 over integers —
    # are ordered LAST and dropped cleanly. (Plain QR can place a zero pivot mid-diagonal,
    # which would keep a garbage column and drop a real helix direction.)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((s > rtol * s.max()).sum()) if s.size else 0  # helix effective rank
    return U[:, :rank]


def helix_target_in_basis(values, fitres, basis: np.ndarray) -> np.ndarray:
    """Coordinates, in `basis` (the d_model x r matrix from `helix_subspace_basis`), of
    the helix prediction for `values`. Feed straight to `subspace_patch_logit_diff` as
    the clean target. Shape (len(values), r)."""
    P = fitres["pca"].components_                       # (n_pca, d_model)
    delta = helix_coords(values, fitres) @ P            # (len(values), d_model) helix offset
    return delta @ basis                                # (len(values), r)


def baseline_pca_r2(acts: np.ndarray, values: np.ndarray, n_pca: int = 9,
                    periods=DEFAULT_PERIODS) -> float:
    """Matched-capacity control, following [kantamneni2025] §4.4 exactly: a polynomial
    fit with basis terms B(a) = [a, a^2, ..., a^(2k+1)] — the SAME number of
    non-constant terms (2k+1) as the helix (one linear + 2k periodic) — fit to the same
    PCA subspace. The helix is only interesting if it MATCHES or BEATS this despite
    being constrained to periodic terms (the [kantamneni2025] capacity argument;
    cf. few-readings-circuit-finding-5).

    Capacity matching: the helix design (`_design_matrix`) is its 2k+1 basis functions
    plus one intercept; this baseline is its 2k+1 polynomial terms plus one intercept —
    matched at 2k+1 non-constant terms with one shared (uncounted) intercept each. (The
    paper lists 2k+1 basis functions and a polynomial basis [a, ..., a^(2k+1)] with no
    separate intercept; a default linear regression supplies an uncounted intercept on
    both sides, which we replicate. The previous version used [a^0..a^(2k)], giving the
    helix one extra effective DoF.)

    `a` is standardized before forming powers purely for numerical conditioning: an
    affine reparametrization of the input leaves the polynomial column space — hence the
    R^2 — unchanged, while keeping a^(2k+1) from overflowing.
    """
    from sklearn.decomposition import PCA

    acts = np.asarray(acts, dtype=np.float64)
    H_pca = PCA(n_components=n_pca).fit_transform(acts - acts.mean(0, keepdims=True))
    k = len(periods)
    deg = 2 * k + 1                                   # highest power = 2k+1 (matches helix's 2k+1 basis fns)
    a = np.asarray(values, dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-12)            # standardize: R^2-invariant, fixes conditioning
    powers = np.stack([a ** p for p in range(1, deg + 1)], axis=1)  # [a^1 .. a^(2k+1)], no constant
    V = np.concatenate([powers, np.ones((powers.shape[0], 1))], axis=1)  # + intercept (matches helix)
    Vh, *_ = np.linalg.lstsq(V, H_pca, rcond=None)
    res = ((H_pca - V @ Vh) ** 2).sum()
    tot = ((H_pca - H_pca.mean(0)) ** 2).sum()
    return float(1.0 - res / max(tot, 1e-12))
