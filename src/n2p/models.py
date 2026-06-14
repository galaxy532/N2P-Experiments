"""Model loading via TransformerLens. One entry point: load_model(key)."""
from __future__ import annotations

import functools

import torch

from .config import DEVICE, DTYPE, get_model_spec


@functools.lru_cache(maxsize=2)
def load_model(key: str):
    """Return a HookedTransformer for the registry key. Cached within a process.

    For the 8B model we use from_pretrained_no_processing to keep memory down and
    avoid weight-folding surprises during activation patching; for GPT-J the
    processed load is fine and gives the usual centered/folded conveniences.
    """
    from transformer_lens import HookedTransformer

    spec = get_model_spec(key)
    common = dict(device=DEVICE, dtype=DTYPE)
    if spec.n_layers >= 32:  # 8B-scale: minimize preprocessing/memory
        model = HookedTransformer.from_pretrained_no_processing(spec.tl_name, **common)
    else:
        model = HookedTransformer.from_pretrained(spec.tl_name, **common)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def number_token_ids(model, lo: int = 0, hi: int = 999) -> dict[int, int]:
    """Map integer n -> single-token id, for integers tokenized as ONE token.

    Many tokenizers split multi-digit numbers; we keep only those encoded as a
    single token (the regime where the per-number residual stream is well defined).
    A leading space is prepended because most BPE number tokens are space-prefixed.
    """
    out = {}
    for n in range(lo, hi + 1):
        toks = model.to_tokens(f" {n}", prepend_bos=False)[0]
        if toks.shape[0] == 1:
            out[n] = int(toks[0].item())
    return out
