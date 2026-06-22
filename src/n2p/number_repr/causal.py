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
def cache_number_site_all_layers(model, prompts, hook_names, token_index=-1,
                                 batch_size=256):
    """Cache MANY sites in one forward sweep per batch, instead of one pass per site.

    A single forward already computes every layer's residual, so caching N sites costs
    the same forward as caching one. This replaces `len(hook_names)` separate
    `cache_number_site` calls (which each re-run the whole model) with one batched
    sweep — e.g. for a per-layer helix scan, ~n_layers x fewer forward passes.

    Args:
        hook_names:  iterable of hook strings (e.g. every blocks.L.hook_resid_post).
        token_index: position to read (-1 = last token = the number token here).
        batch_size:  prompts per forward. Prompts here are equal length (single
                     number tokens), so they batch without padding; lower this only
                     if a longer-prompt variant runs out of memory.

    Returns: dict {hook_name -> (len(prompts), d_model) float32 numpy}, matching what
    `cache_number_site` returns per hook.
    """
    names = list(hook_names)
    nameset = set(names)
    if not prompts:
        d = model.cfg.d_model
        return {h: np.empty((0, d), dtype=np.float32) for h in names}

    chunks = {h: [] for h in names}
    for i in range(0, len(prompts), batch_size):
        toks = model.to_tokens(prompts[i:i + batch_size])  # (b, seq); equal len -> no pad
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in nameset)
        for h in names:
            chunks[h].append(cache[h][:, token_index].float().cpu().numpy())
    return {h: np.concatenate(v, axis=0) for h, v in chunks.items()}


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


# =====================================================================================
# Total / direct / indirect effect (activation vs path patching)
# -------------------------------------------------------------------------------------
# Implements the [kantamneni2025] Fig-6 decomposition (canonical defs in the wiki concept
# ablation-and-patching.md). A *denoising* setup: base run = CORRUPT prompt, the sender is
# patched to its CLEAN value.
#   total_effect  : patch sender->clean, let all downstream components RECOMPUTE freely.
#   direct_effect : patch sender->clean, FREEZE every other component output to its CORRUPT
#                   (base) value so only the sender's direct residual->logits path moves.
#   indirect_effect = total - direct.
# `sender_basis` (d_model, k) optionally restricts the swap to a subspace (e.g. an SAE /
# helix direction) — the N2P "replace along a direction, not the whole node" granularity
# (see approach-decision-circuit-identification.md). Scope: last-token (answer) analysis,
# patching at `token_index`; this matches the Fig-6 use. GPU-untested (no model offline).
# =====================================================================================

def component_output_hooks(model):
    """All per-component output hooks (attn_out + mlp_out) for every layer, forward order.
    These are the nodes frozen for a direct-effect (path-patching) measurement."""
    names = []
    for L in range(model.cfg.n_layers):
        names.append(f"blocks.{L}.hook_attn_out")
        names.append(f"blocks.{L}.hook_mlp_out")
    return names


def _cache_at(model, prompt, hook_names, token_index):
    """Cache the listed hooks at `token_index` on one prompt -> {hook: (d_model,) tensor}."""
    nameset = set(hook_names)
    toks = model.to_tokens(prompt)
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n in nameset)
    return {h: cache[h][0, token_index] for h in hook_names}


def _swap_value(cur, clean_val, basis):
    """New value to write at the patched position. cur: (batch, d_model); clean_val:
    (d_model,). basis None -> full swap to clean; else swap only the projection onto
    span(basis): cur + (clean-cur) projected into the subspace."""
    if basis is None:
        return clean_val
    U = torch.as_tensor(basis, dtype=cur.dtype, device=cur.device)   # (d, k)
    delta = clean_val.unsqueeze(0) - cur                              # (b, d)
    proj = (delta @ U) @ U.t()                                       # (b, d)
    return cur + proj


@torch.no_grad()
def total_effect_logit_diff(model, clean_prompt, corrupt_prompt, sender_hook,
                            answer_tokens, token_index=-1, sender_basis=None):
    """TOTAL effect of `sender_hook`: patch it to its clean value on the corrupt run and
    let the rest of the network recompute. Returns logit[ap]-logit[a] (answer_tokens =
    (a_id, ap_id) = (corrupt-answer, clean-answer))."""
    clean_val = _cache_at(model, clean_prompt, [sender_hook], token_index)[sender_hook]

    def hook(act, hook):
        act[:, token_index, :] = _swap_value(act[:, token_index, :], clean_val, sender_basis)
        return act

    toks = model.to_tokens(corrupt_prompt)
    logits = model.run_with_hooks(toks, fwd_hooks=[(sender_hook, hook)])
    last = logits[0, -1]
    a_id, ap_id = answer_tokens
    return float(last[ap_id] - last[a_id])


@torch.no_grad()
def direct_effect_logit_diff(model, clean_prompt, corrupt_prompt, sender_hook,
                             answer_tokens, frozen_hooks=None, token_index=-1,
                             sender_basis=None):
    """DIRECT effect of `sender_hook` (path patching): patch it to clean while FREEZING
    every hook in `frozen_hooks` (default: all component outputs except the sender) to its
    CORRUPT (base) value, so the sender only reaches the logits via the direct residual
    path. Same return convention as total_effect_logit_diff. IE = TE - DE."""
    if frozen_hooks is None:
        frozen_hooks = component_output_hooks(model)
    frozen = [h for h in frozen_hooks if h != sender_hook]
    clean_val = _cache_at(model, clean_prompt, [sender_hook], token_index)[sender_hook]
    corrupt_vals = _cache_at(model, corrupt_prompt, frozen, token_index)

    fwd_hooks = []

    def sender_fn(act, hook):
        act[:, token_index, :] = _swap_value(act[:, token_index, :], clean_val, sender_basis)
        return act

    fwd_hooks.append((sender_hook, sender_fn))

    def _freeze(value):
        def fn(act, hook):
            act[:, token_index, :] = value
            return act
        return fn

    for h, v in corrupt_vals.items():
        fwd_hooks.append((h, _freeze(v)))

    toks = model.to_tokens(corrupt_prompt)
    logits = model.run_with_hooks(toks, fwd_hooks=fwd_hooks)
    last = logits[0, -1]
    a_id, ap_id = answer_tokens
    return float(last[ap_id] - last[a_id])


@torch.no_grad()
def te_de_ie(model, clean_prompt, corrupt_prompt, sender_hook, answer_tokens,
             frozen_hooks=None, token_index=-1, sender_basis=None):
    """Convenience: return {'te','de','ie','de_over_te'} for one sender. `sender_basis`
    restricts the intervention to an SAE/helix subspace (direction-level DE/TE)."""
    te = total_effect_logit_diff(model, clean_prompt, corrupt_prompt, sender_hook,
                                 answer_tokens, token_index, sender_basis)
    de = direct_effect_logit_diff(model, clean_prompt, corrupt_prompt, sender_hook,
                                  answer_tokens, frozen_hooks, token_index, sender_basis)
    return {"te": te, "de": de, "ie": te - de,
            "de_over_te": (de / te) if abs(te) > 1e-9 else float("nan")}
