"""Causal validation of a hypothesized number subspace, following the
[engels2024] geometry-probe + subspace-patch + average-ablate-the-rest method
(§5.1, Eq 5-6) and [kantamneni2025] activation patching.

Logic (sufficiency test): if we OVERWRITE the hypothesized subspace toward a clean
target value a' (e.g. the helix fit for a') while AVERAGE-ABLATING every other
dimension at that site (replace with the across-prompt mean, to block backup paths
from leaking the original answer), and the model's logit difference moves to a' as
much as a full-layer patch would, then that subspace is causally sufficient for the
number's effect on the computation.

These functions need a live HookedTransformer (real forward passes).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def cache_number_site(model, prompts, hook_name, token_index=-1):
    """Run prompts, return (acts, ) at `hook_name` for `token_index`.

    acts: (len(prompts), d_model) numpy. token_index=-1 = last token; for operand
    tracking pass the index of the number token in the prompt.
    """
    acts = []
    for p in prompts:
        toks = model.to_tokens(p)
        _, cache = model.run_with_cache(toks, names_filter=hook_name)
        acts.append(cache[hook_name][0, token_index].float().cpu().numpy())
    return np.stack(acts, axis=0)


@torch.no_grad()
def subspace_patch_logit_diff(
    model,
    clean_prompt: str,
    hook_name: str,
    subspace_basis: np.ndarray,      # (d_model, k) orthonormal columns spanning the subspace
    target_vec_in_subspace: np.ndarray,  # (k,) coords of the clean target a' in that basis
    site_mean: np.ndarray,           # (d_model,) across-prompt mean for average-ablation
    answer_tokens: tuple[int, int],  # (logit id for a-answer, logit id for a'-answer)
    token_index: int = -1,
):
    """Patch: at `hook_name`/`token_index`, replace the activation by
        site_mean  +  subspace_basis @ target_vec_in_subspace
    i.e. average-ablate everything, then write the clean target *only* inside the
    subspace. Return logit_diff = logit[a'] - logit[a].

    A large positive logit_diff (toward a') = the subspace is causally carrying the
    number. Compare against the full-layer-patch upper bound and the no-op baseline
    in the runner.
    """
    U = torch.tensor(subspace_basis, dtype=model.cfg.dtype, device=model.cfg.device)  # (d,k)
    t = torch.tensor(target_vec_in_subspace, dtype=model.cfg.dtype, device=model.cfg.device)  # (k,)
    mean = torch.tensor(site_mean, dtype=model.cfg.dtype, device=model.cfg.device)  # (d,)
    new_vec = mean + U @ t  # (d,)

    def hook(act, hook):  # act: (batch, seq, d)
        act[:, token_index, :] = new_vec
        return act

    toks = model.to_tokens(clean_prompt)
    logits = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hook)])
    last = logits[0, -1]
    a_id, ap_id = answer_tokens
    return float(last[ap_id] - last[a_id])


@torch.no_grad()
def full_layer_patch_logit_diff(model, clean_prompt, corrupt_prompt, hook_name,
                                answer_tokens, token_index=-1):
    """Upper bound: patch the WHOLE activation at the site from the corrupt run
    (the standard activation-patching ceiling). a'/a answer ids as above."""
    toks_corr = model.to_tokens(corrupt_prompt)
    _, cache = model.run_with_cache(toks_corr, names_filter=hook_name)
    donor = cache[hook_name][0, token_index]

    def hook(act, hook):
        act[:, token_index, :] = donor
        return act

    toks = model.to_tokens(clean_prompt)
    logits = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hook)])
    last = logits[0, -1]
    a_id, ap_id = answer_tokens
    return float(last[ap_id] - last[a_id])


def orthonormalize(C: np.ndarray) -> np.ndarray:
    """Columns of the fitted helix map C live in PCA space; turn an arbitrary basis
    into an orthonormal one (QR) so the patch writes into a clean subspace."""
    Q, _ = np.linalg.qr(C)
    return Q
