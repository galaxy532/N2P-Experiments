"""Model loading via TransformerLens. One entry point: load_model(key)."""
from __future__ import annotations

import functools

import torch

from .config import DEVICE, DTYPE, get_model_spec


@functools.lru_cache(maxsize=2)
def load_model(key: str):
    """Return a HookedTransformer for the registry key. Cached within a process.

    All models load via from_pretrained_no_processing. Under reduced precision (fp16)
    the processed path folds/centers weights with an fp32 upcast that transiently
    spikes CPU RAM (OOM-killed GPT-J even at 44 GB) and is numerically discouraged by
    TransformerLens. We don't need the centered/folded conveniences: the helix fit
    mean-centers activations itself, and no_processing is also the patching-friendly
    load (no weight-folding surprises during activation patching).
    """
    from transformer_lens import HookedTransformer

    spec = get_model_spec(key)
    common = dict(device=DEVICE, dtype=DTYPE)
    model = HookedTransformer.from_pretrained_no_processing(spec.tl_name, **common)
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
