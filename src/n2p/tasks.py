"""Task registry: arithmetic programs and their prompt framings.

STATUS (revised 2026-06-14): program family = CLEAN-CORE + STRESS-SET (see
../wiki/notes/iclr-plan.md "Program family"). Anchors (addition, greater_than) locked.

  - clean-core  : value-level mechanism expected (clean operand subspace likely)
                  -> carries the cross-task EXCLUSIVITY test.
  - stress-set  : per-digit-heuristic-likely (multiplication, division, modular)
                  -> tests whether WRITE-SITE INJECTION bypasses the heuristics
                     (../wiki/notes/approach-decision-circuit-identification.md).
  - framing-variant : same op, different surface form -> framing protocol
                  (../wiki/notes/approach-decision-feature-tracking.md "Framing protocol").
  - transcendentals (sin/exp) are an OPTIONAL probe, not core (a base model may have
    no clean circuit -> possible no-circuit-found result). Not included by default.

Design principle (see README §1): separate the OPERAND representation (shared
number-value subspace, the exclusivity target) from the OPERATION (task-specific; the
stub replaces the OPERATION, by injecting the clean ANSWER at the write site). The
family (a) shares the operand representation across clean-core, and (b) stresses the
substitution with heuristic-heavy ops.

Each Task yields (prompt, answer, operands) tuples; `operand_token_index` marks which
token position carries the number whose representation we track.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Task:
    name: str
    tier: str                       # "clean-core" | "stress-set" | "framing-variant"
    template: str                   # uses {a}, {b}
    fn: Callable[[int, int], int]   # ground-truth program
    a_range: tuple[int, int] = (0, 99)
    b_range: tuple[int, int] = (0, 99)
    anchor: bool = False            # doubles as ground truth / locked
    note: str = ""

    def sample(self, n: int, seed: int = 0):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            a = rng.randint(*self.a_range)
            b = rng.randint(*self.b_range)
            out.append({
                "prompt": self.template.format(a=a, b=b),
                "answer": self.fn(a, b),
                "a": a, "b": b,
            })
        return out


# --- CLEAN-CORE (value-level mechanism expected; carries the EXCLUSIVITY test) ---
ADDITION = Task(
    "addition", "clean-core", template="{a}+{b}=", fn=lambda a, b: a + b, anchor=True,
    note="Clock/helix mechanism established [kantamneni2025][zhou2024]. PRIMARY anchor + ground truth.",
)
GREATER_THAN = Task(
    "greater_than", "clean-core",
    # Canonical Hanna et al. form: predict a 2-digit year > {a} (here {b} is the gold YY).
    template="The war lasted from the year 17{a} to the year 17",
    fn=lambda a, b: int(b > a), anchor=True, a_range=(2, 98), b_range=(2, 98),
    note="Comparison/greater-than. Known circuit [hanna]. DISCOVERY anchor (Edge Pruning on its "
         "own GT dataset, not this string). Excluded from the numeric accuracy probe.",
)
SUBTRACTION = Task("subtraction", "clean-core", "{a}-{b}=", lambda a, b: a - b,
                   note="Inverse of addition; should share the operand subspace.")
MULT_CONST = Task("mult_const", "clean-core", "{a}*3=", lambda a, b: a * 3,
                  note="Constant multiplier isolates a single operand's representation.")

# --- STRESS-SET (per-digit-heuristic-likely; tests WRITE-SITE injection bypass) --
# May NOT have a clean operand subspace [nikankin2024][lindsey2025]; the test is
# whether injecting the clean ANSWER at the write site bypasses the heuristics.
MULTIPLICATION = Task("multiplication", "stress-set", "{a}*{b}=", lambda a, b: a * b,
                      a_range=(2, 20), b_range=(2, 20),
                      note="Full a*b: most likely digit-heuristic; primary bypass test.")
INT_DIVISION = Task("int_division", "stress-set", "{a}//{b}=", lambda a, b: a // max(b, 1),
                    b_range=(1, 20),
                    note="[kantamneni2025] §4.5: helix underperforms here — informative.")
MODULAR = Task("modular", "stress-set", "({a}+{b}) mod 7 =", lambda a, b: (a + b) % 7,
               note="High-frequency Fourier component [zhou2024]; modular vs magnitude split.")

# NOTE: framing VARIANTS (same op, different surface form) are no longer separate Task
# objects. They live once in FRAMINGS below (symbolic / word / wordproblem) and are
# iterated by both the accuracy probe and the number-representation scripts. This removes
# the old duplication between the standalone addition_words/addition_wordproblem Tasks,
# the EXTRA_FRAMINGS dict, and number_repr/prompts.py's template_1..4.


REGISTRY: dict[str, Task] = {t.name: t for t in [
    ADDITION, GREATER_THAN, SUBTRACTION, MULT_CONST,          # clean-core
    MULTIPLICATION, INT_DIVISION, MODULAR,                    # stress-set
]}


def by_tier(tier: str) -> list[Task]:
    return [t for t in REGISTRY.values() if t.tier == tier]


# --- Framings (surface forms) per task -----------------------------------------
# THE single source of truth for prompt surface forms (replaces number_repr/prompts.py).
# Three framings per numeric task, in this order:
#   symbolic    : context-PREFIXED symbolic form, e.g. "Compute {a} + {b} =". The
#                 "Compute " prefix puts context BEFORE the operand, so reading the
#                 operand-a representation is not done at the first content position
#                 (the causal-masking point in fourier-experiments-week1-results.md).
#                 This replaces the old bare-number template_1 (dropped) and the terse
#                 template_4.
#   word        : word operator, e.g. "{a} plus {b} equals".
#   wordproblem : natural-language scenario.
# Both the accuracy probe and the number-representation scripts iterate these. Operands
# are space-separated so the {a}/{b} tokens are stable across framings; literal constants
# (the 3 in mult_const, the 7 in modular) always come AFTER the operands so the
# digit-position read-token logic is unambiguous, and are spelled out ("three"/"seven")
# in word/wordproblem framings to avoid stray digit tokens.
FRAMING_NAMES = ["symbolic", "word", "wordproblem"]

FRAMINGS: dict[str, dict[str, str]] = {
    "addition": {
        "symbolic": "Compute {a} + {b} =",
        "word": "{a} plus {b} equals",
        "wordproblem": "I have {a} apples and then get {b} more, so now I have",
    },
    "subtraction": {
        "symbolic": "Compute {a} - {b} =",
        "word": "{a} minus {b} equals",
        "wordproblem": "I have {a} apples and then give away {b}, so now I have",
    },
    "multiplication": {
        "symbolic": "Compute {a} * {b} =",
        "word": "{a} times {b} equals",
        "wordproblem": "I have {a} baskets holding {b} apples each, so in total I have",
    },
    "mult_const": {
        "symbolic": "Compute {a} * 3 =",
        "word": "{a} times three equals",
        "wordproblem": "I have {a} baskets holding three apples each, so in total I have",
    },
    "int_division": {
        "symbolic": "Compute {a} // {b} =",
        "word": "{a} divided by {b}, rounded down, is",
        "wordproblem": "I share {a} apples equally among {b} friends, so each friend gets",
    },
    "modular": {
        "symbolic": "Compute ({a} + {b}) mod 7 =",
        "word": "the remainder when {a} plus {b} is divided by seven is",
        "wordproblem": "I take {a} plus {b} steps around a clock with seven hours, landing on",
    },
    # greater_than is a comparison (its own canonical .template); excluded here.
}

# Operations exposed to the number-representation scripts as --operation.
OPERATION_CHOICES = list(FRAMINGS)
READ_TOKEN_CHOICES = ["a", "b", "sum"]


def framing_template(operation: str, framing: str) -> str:
    if operation not in FRAMINGS:
        raise KeyError(f"unknown operation {operation!r}; known: {OPERATION_CHOICES}")
    fr = FRAMINGS[operation]
    if framing not in fr:
        raise KeyError(f"unknown framing {framing!r} for {operation}; known: {list(fr)}")
    return fr[framing]


def framings_for(task: Task) -> list[tuple[str, str]]:
    """[(framing_name, template), ...], symbolic first. Source of truth for the accuracy
    probe and feature tracking. Falls back to the task's canonical .template for tasks
    without a framing set (e.g. greater_than)."""
    fr = FRAMINGS.get(task.name)
    if fr is None:
        return [("symbolic", task.template)]
    return [(name, fr[name]) for name in FRAMING_NAMES if name in fr]


def template_has_b(operation: str, framing: str) -> bool:
    return "{b}" in framing_template(operation, framing)


# --- prompt construction + read-token selection (was number_repr/prompts.py) ----
def fewshot_shots(operation: str, kshot: int, seed: int = 0):
    """`kshot` solved (a, b, answer) example tuples for few-shot context, deterministic by
    `seed` (drawn from the task's operand ranges; answer = task.fn). Empty if kshot <= 0."""
    if kshot <= 0:
        return ()
    task = get_task(operation)
    return tuple((d["a"], d["b"], d["answer"]) for d in task.sample(kshot, seed=seed))


def build_prompt(operation: str, framing: str, a: int, b: int | None = None,
                 *, prefix: str = "", shots=()) -> str:
    """Build one prompt for operand value `a` (and fixed `b` if the framing uses it).

    Layout: ``prefix`` (model instruction) + ``shots`` solved example lines + the query line,
    joined by newlines. `prefix` is the model-specific instruction (config.ModelSpec.
    prompt_prefix); `shots` is an iterable of (a, b, answer) tuples (see fewshot_shots) — each
    rendered as ``<framing> <answer>``. With no shots and a single-line framing this is just
    ``prefix + query`` (unchanged). The prefix MUST be digit-free; operands are located in the
    QUERY line (after the last newline — see _digit_positions) so example digits are skipped.
    """
    tmpl = framing_template(operation, framing)
    if "{b}" in tmpl and b is None:
        raise ValueError(f"{operation}/{framing} needs a second operand b")
    lines = [tmpl.format(a=sa, b=("" if sb is None else sb)) + f" {sans}"
             for (sa, sb, sans) in shots]
    lines.append(tmpl.format(a=a, b=("" if b is None else b)))
    return prefix + "\n".join(lines)


def build_prompts(operation: str, framing: str, values, b: int | None = None,
                  *, prefix: str = "", shots=()) -> list[str]:
    """Prompts for the full operand-`a` sweep (b fixed), with optional model prefix and
    fixed few-shot `shots` (same examples across the sweep, so only the query operand varies)."""
    return [build_prompt(operation, framing, int(a), b, prefix=prefix, shots=shots)
            for a in values]


def _digit_positions(model, prompt: str) -> list[int]:
    """Digit-bearing token positions in the QUERY line — i.e. AFTER the last newline-bearing
    token. Few-shot example lines and any instruction prefix precede the query and are
    separated from it by '\\n', so their digits (example operands/answers) are skipped. For a
    single-line prompt with no newline this is the whole prompt (unchanged behavior)."""
    toks = model.to_str_tokens(prompt)
    start = 0
    for i, t in enumerate(toks):
        if "\n" in t:
            start = i + 1
    return [i for i in range(start, len(toks)) if any(c.isdigit() for c in toks[i])]


def read_token_index(model, prompt: str, read_token: str, operation: str,
                     framing: str) -> int:
    """Token index to read for this (prompt, read_token, operation, framing).

    a / b -> first / second digit-bearing token (operands precede any literal constant in
    every framing, and `a` precedes `b`). sum -> -1 (last token). Raises if `b` is
    requested on a framing without a second operand.
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
    if not template_has_b(operation, framing):
        raise ValueError(f"--read-token b is invalid for {operation}/{framing} "
                         f"(no second operand)")
    if len(pos) < 2:
        raise ValueError(f"no operand-b token found in {model.to_str_tokens(prompt)}")
    return pos[1]


def single_token_number_grid(model, operation: str, framing: str, lo: int, hi: int,
                             *, b: int | None = None) -> tuple[list[int], dict[int, int]]:
    """Operand-`a` sweep grid for the number-representation scripts, validated against the
    REAL prompt tokenization (the "more correct", model-agnostic replacement for the old
    space-prefixed-only `models.number_token_ids`, which returned an empty grid on Llama-3).

    Returns ``(values, id_map)`` where ``values`` is every integer ``n`` in ``[lo, hi]``
    whose operand-`a` is exactly ONE token at the operand-`a` position of the prompt
    ``build_prompt(operation, framing, n, b)`` (i.e. the first digit-bearing token equals
    ``str(n)`` — same convention as `read_token_index`), and ``id_map[n]`` is that token's
    in-context vocab id (the id to use for W_E embedding / W_U logit-lens of number ``n``).

    Tokenizer-agnostic: it never assumes how a leading space attaches, because it tokenizes
    the actual string. The caller applies ``repcli.contiguous_prefix`` for the even DFT grid.
    """
    values: list[int] = []
    id_map: dict[int, int] = {}
    for n in range(lo, hi + 1):
        prompt = build_prompt(operation, framing, n, b)
        str_toks = model.to_str_tokens(prompt)
        ids = model.to_tokens(prompt)[0]
        idx = next((i for i, t in enumerate(str_toks) if any(c.isdigit() for c in t)), None)
        if idx is None or str_toks[idx].strip() != str(n):
            continue  # operand absent or split across multiple tokens for this tokenizer
        values.append(n)
        id_map[n] = int(ids[idx].item())
    return values, id_map


def validate_read_token(read_token: str, operation: str, framing: str) -> None:
    """Cheap up-front check (no model needed) for obviously invalid combinations."""
    if read_token == "b" and not template_has_b(operation, framing):
        raise SystemExit(f"--read-token b is invalid for {operation}/{framing} (no second "
                         f"operand b); use an op/framing that has b.")


def get_task(name: str) -> Task:
    if name not in REGISTRY:
        raise KeyError(f"Unknown task {name!r}. Known: {list(REGISTRY)}")
    return REGISTRY[name]
