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


# --- model-agnostic number tokenization primitives -----------------------------
# Single-token-ness of a number depends on the tokenizer AND on whether a leading space
# attaches to the digits. GPT-2/GPT-J merge " 7" into one token; Llama-3 splits the space
# off and tokenizes digit-runs of 1-3, so " 7" -> [" ", "7"] and the single token is the
# bare "7". The old number_token_ids probed only the space-prefixed form and therefore
# returned an EMPTY map on Llama-3. These helpers try both forms; the *operand-position*
# grid (where context matters) lives in tasks.single_token_number_grid.

def single_token_id(model, n) -> int | None:
    """Vocab id of integer `n` if it is ONE token, trying the space-prefixed form first
    (GPT-2/GPT-J) then the bare form (Llama-3). None if neither is single-token."""
    for form in (f" {n}", f"{n}"):
        toks = model.to_tokens(form, prepend_bos=False)[0]
        if toks.shape[0] == 1:
            return int(toks[0].item())
    return None


def first_answer_token_id(model, answer, *, space: bool = True) -> int:
    """First CONTENT (non-whitespace) token id of `answer`.

    `space` selects which prompt FORMAT the answer is emitted in, because the right token
    differs by format (probe-confirmed 2026-06-24):
      - space=True  -> the answer follows a space (few-shot "= 42"). GPT-J emits the
        space-merged token ' 42'; Llama emits [' ','42'] and the leading space is skipped.
        Use for the accuracy probe.
      - space=False -> the answer immediately follows "=" with NO space (zero-shot,
        "{a}+{b}="). GPT-J emits the BARE '42'; Llama is identical (its space is always a
        separate token). Use for the fixed-position logit reads in run_causal_validation /
        run_te_de_probe, and matches [kantamneni2025]'s bare `tokenizer(f'{answer}')`.
    """
    s = f" {answer}" if space else f"{answer}"
    str_toks = model.to_str_tokens(s, prepend_bos=False)
    ids = model.to_tokens(s, prepend_bos=False)[0]
    for t, i in zip(str_toks, ids):
        if t.strip() != "":
            return int(i.item())
    return int(ids[0].item())


def is_single_token_answer(model, answer, *, space: bool = True) -> bool:
    """True if `answer` is a single CONTENT token in the chosen format (see
    first_answer_token_id for the space= meaning)."""
    s = f" {answer}" if space else f"{answer}"
    str_toks = model.to_str_tokens(s, prepend_bos=False)
    return sum(1 for t in str_toks if t.strip() != "") == 1


def single_token_answer_id(model, answer, *, space: bool = True) -> int | None:
    """first_answer_token_id if the answer is a single content token (in the chosen
    format), else None."""
    return (first_answer_token_id(model, answer, space=space)
            if is_single_token_answer(model, answer, space=space) else None)
