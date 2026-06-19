"""Prompt templates + read-token selection for the number-representation probes.

Shared by run_fourier.py, run_fourier_components.py, run_fourier_components_raw.py and
run_helix_fit.py so all four scripts sweep the operand `a` through the SAME prompts and
read the SAME token positions.

Motivation (see wiki exp-note fourier-experiments-week1-results.md, 2026-06-19): under a
causal model a token's representation only depends on the prefix up to itself, so an
operand placed at the FIRST content position can never pick up task context — which is why
the old bare-vs-addition plots were identical. The templates below put increasingly rich
framing BEFORE `a`, so context can actually reach the operand position:

    template_1  " {a}"                                    bare operand (baseline; no b)
    template_2  "The number {a}"                          minimal context before a (no b)
    template_3  "What is the sum of {a} and {b}? Answer:" natural-language addition (sum=':')
    template_4  "Compute {a} + {b} ="                     terse symbolic addition (sum='=')

The operand is always SPACE-PREFIXED so the `a`-token is the same token id across all four
templates (in GPT-2/GPT-J BPE, "5" and " 5" are different tokens).

`--read-token` selects WHICH position to analyze:
    a    -> the operand-`a` token (its internal representation)
    b    -> the operand-`b` token (only for templates that have a second operand)
    sum  -> the last token (the ':' / '=' whose next-token prediction is the answer);
            for the logit-lens script this is the only meaningful site (zhou2024 Fig 2/3).
For templates without a `sum`/`b` (1, 2) `sum` collapses to the last/`a` token.
"""
from __future__ import annotations

# text uses {a} / {b}; the operand is space-prefixed for stable tokenization.
TEMPLATES: dict[str, dict] = {
    "template_1": {"text": " {a}", "has_b": False,
                   "desc": "bare operand (baseline; context-free)"},
    "template_2": {"text": "The number {a}", "has_b": False,
                   "desc": "minimal context before the operand"},
    "template_3": {"text": "What is the sum of {a} and {b}? Answer:", "has_b": True,
                   "desc": "natural-language addition framing (sum token ':')"},
    "template_4": {"text": "Compute {a} + {b} =", "has_b": True,
                   "desc": "terse symbolic addition framing (sum token '=')"},
}

TEMPLATE_CHOICES = list(TEMPLATES)
READ_TOKEN_CHOICES = ["a", "b", "sum"]


def template_has_b(name: str) -> bool:
    return TEMPLATES[name]["has_b"]


def template_text(name: str) -> str:
    return TEMPLATES[name]["text"]


def build_prompt(name: str, a: int, b: int | None = None) -> str:
    """Build one prompt for operand value `a` (and fixed `b` if the template uses it)."""
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; known: {TEMPLATE_CHOICES}")
    spec = TEMPLATES[name]
    if spec["has_b"] and b is None:
        raise ValueError(f"{name} needs a second operand b")
    return spec["text"].format(a=a, b=("" if b is None else b))


def build_prompts(name: str, values, b: int | None = None) -> list[str]:
    """Prompts for the full operand-`a` sweep (b fixed)."""
    return [build_prompt(name, int(a), b) for a in values]


def _digit_positions(model, prompt: str) -> list[int]:
    return [i for i, t in enumerate(model.to_str_tokens(prompt))
            if any(c.isdigit() for c in t)]


def read_token_index(model, prompt: str, read_token: str, template: str) -> int:
    """Token index to read for this (prompt, read_token, template).

    a / b -> the first / second digit-bearing token (operands are the only digits in our
    templates and `a` precedes `b`). sum -> -1 (last token). Raises if `b` is requested
    on a template without a second operand.
    """
    if read_token not in READ_TOKEN_CHOICES:
        raise ValueError(f"unknown read_token {read_token!r}; use {READ_TOKEN_CHOICES}")
    if read_token == "sum":
        return -1
    pos = _digit_positions(model, prompt)
    if read_token == "a":
        if not pos:
            raise ValueError(f"no operand-a token found in {model.to_str_tokens(prompt)}")
        return pos[0]
    # read_token == "b"
    if not template_has_b(template):
        raise ValueError(f"--read-token b is invalid for {template} (no second operand)")
    if len(pos) < 2:
        raise ValueError(f"no operand-b token found in {model.to_str_tokens(prompt)}")
    return pos[1]


def validate_read_token(read_token: str, template: str) -> None:
    """Cheap up-front check (no model needed) for obviously invalid combinations."""
    if read_token == "b" and not template_has_b(template):
        raise SystemExit(f"--read-token b is invalid for {template} (no second operand b); "
                         f"use a template with b ({[t for t in TEMPLATE_CHOICES if template_has_b(t)]}).")
