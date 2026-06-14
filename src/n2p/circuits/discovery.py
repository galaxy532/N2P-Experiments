"""Circuit-discovery entry points.

Decided method (see ../../../README.md §2 and approach-decision-circuit-identification.md):
  base method   = Edge Pruning [bhaskar2024]      (edge-level, L0 masks; recovers
                  Tracr ground truth exactly; scales to 13B)
  completeness  = noising + denoising [chen2025]  (catch OR-gate backup paths)
  ground truth  = Tracr [lindner2023]             (compiled program -> known circuit)

ACDC is NOT used. It is a *rejected* base method in our protocol (slow; misses
negative name-mover / previous-token heads; floods at low tau — see
../wiki/notes/verified-failure-modes.md item 1). It was a mistaken shortcut in the
first scaffold; removed.

Week-1 goal = a DISCOVERY SANITY CHECK on a known-answer setting: run Edge Pruning on
a Tracr-compiled task whose ground-truth circuit is known exactly, and confirm
recovery. Greater-Than is kept as a secondary *real-LM* target. The chen2025
completeness pass is layered in week 2.

This module is intentionally thin: it locates the Edge Pruning repo and drives it, so
the heavy, well-tested discovery code is not re-implemented here.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..config import REPO_ROOT

# Vendored/cloned repos live next to N2P-Experiments (the cowork root), since
# N2P-Experiments is its own git repo. Override via env on Paperspace.
WIKI_ROOT = Path(os.environ.get("N2P_WIKI_ROOT", REPO_ROOT.parent))
EDGE_PRUNING_DIR = Path(os.environ.get("EDGE_PRUNING_DIR", WIKI_ROOT / "Edge-Pruning"))


def edge_pruning_available() -> bool:
    return EDGE_PRUNING_DIR.exists() and (EDGE_PRUNING_DIR / "src").exists()


def _require_edge_pruning():
    if not edge_pruning_available():
        raise FileNotFoundError(
            f"Edge Pruning not found at {EDGE_PRUNING_DIR}.\n"
            "Clone it next to N2P-Experiments (or set EDGE_PRUNING_DIR):\n"
            "    git clone https://github.com/princeton-nlp/Edge-Pruning\n"
            "It is the chosen base discovery method [bhaskar2024].")


def run_edge_pruning_tracr(program: str = "reverse", **kw):
    """SANITY CHECK: Edge Pruning on a Tracr-compiled program with KNOWN ground truth.

    Tracr compiles a RASP program into exact transformer weights, so the true circuit
    is known by construction — the cleanest validation that our discovery pipeline
    recovers what it should before we trust it on novel arithmetic tasks. Edge Pruning
    is reported to *perfectly* recover Tracr ground-truth circuits [bhaskar2024].

    `program`: a Tracr task id (e.g. 'reverse', 'proportion'), matching the Tracr
    tasks used in the Edge Pruning / ACDC benchmarks.

    Returns the completed process; outputs land under EDGE_PRUNING_DIR. The exact CLI
    differs by Edge Pruning revision — fill the command from its README on first run
    (TODO marked below), then commit the working invocation here.
    """
    _require_edge_pruning()
    # TODO(week1, first Paperspace run): replace with the repo's actual entrypoint.
    # Edge Pruning ships task runners under src/ / experiments/; wire the Tracr task
    # here and assert recovered-edges == ground-truth-edges.
    raise NotImplementedError(
        "Fill in the Edge Pruning Tracr entrypoint from its README on first run, then "
        "commit the working command. Assert recovered edges == Tracr ground truth.")


def run_edge_pruning_task(task: str = "greater_than", model_key: str = "gptj", **kw):
    """Edge Pruning on a real-LM task (Greater-Than as the week-1 secondary check;
    the arithmetic tasks from weeks 2+). Add the chen2025 noising+denoising
    completeness pass on top before trusting the result (week 2)."""
    _require_edge_pruning()
    raise NotImplementedError(
        "Week-2 target: wire Edge Pruning for real-LM tasks + chen2025 completeness "
        "layer. See approach-decision-circuit-identification.md.")


def add_completeness_layer(*args, **kw):
    """chen2025 noising+denoising pass over a discovered circuit to recover OR-gate
    backup paths the (noising-based) base method misses. Week-2 implementation."""
    raise NotImplementedError("chen2025 completeness layer — scheduled week 2.")
