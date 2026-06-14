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

# --- FRAMING VARIANTS (proposed — same op, different surface form) --------------
# Tests framing-dependence [engels2024 §5]: the SAME operation can deploy a clean
# feature under one framing and none under another. Motivates the stub's framing-
# invariance benefit (C3).
ADDITION_WORDS = Task("addition_words", "framing-variant",
                      "{a} plus {b} equals", lambda a, b: a + b,
                      note="Word operator instead of '+'.")
ADDITION_WORDPROBLEM = Task("addition_wordproblem", "framing-variant",
                            "I have {a} apples and get {b} more. Now I have",
                            lambda a, b: a + b,
                            note="Natural-language framing; tests representation transfer.")


REGISTRY: dict[str, Task] = {t.name: t for t in [
    ADDITION, GREATER_THAN, SUBTRACTION, MULT_CONST,          # clean-core
    MULTIPLICATION, INT_DIVISION, MODULAR,                    # stress-set
    ADDITION_WORDS, ADDITION_WORDPROBLEM,                     # framing variants
]}


def by_tier(tier: str) -> list[Task]:
    return [t for t in REGISTRY.values() if t.tier == tier]


# --- Framings (surface forms) per task -----------------------------------------
# Each task's canonical symbolic form is its .template; extra framings are listed
# here. The accuracy probe and (later) feature tracking run over every framing —
# see ../wiki/notes/approach-decision-feature-tracking.md "Framing protocol".
EXTRA_FRAMINGS: dict[str, list[tuple[str, str]]] = {
    "addition": [
        ("word", "{a} plus {b} equals"),
        ("wordproblem", "I have {a} apples and then get {b} more, so I have"),
    ],
    "subtraction": [("word", "{a} minus {b} equals")],
    "multiplication": [("word", "{a} times {b} equals")],
    "mult_const": [("word", "{a} times 3 equals")],
    "int_division": [("word", "{a} divided by {b}, rounded down, is")],
    "modular": [("word", "the remainder when {a} plus {b} is divided by 7 is")],
    # greater_than is a comparison, handled separately from numeric accuracy.
}


def framings_for(task: Task) -> list[tuple[str, str]]:
    """[(framing_name, template), ...] starting with the canonical symbolic form."""
    return [("symbolic", task.template)] + EXTRA_FRAMINGS.get(task.name, [])


def get_task(name: str) -> Task:
    if name not in REGISTRY:
        raise KeyError(f"Unknown task {name!r}. Known: {list(REGISTRY)}")
    return REGISTRY[name]
