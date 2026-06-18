"""Central config: model registry, paths, device. Scripts reference models by KEY."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import torch

# --- Paths ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"
# HF cache lives ALONGSIDE the repo (sibling dir), never inside this git repo.
# Override with HF_HOME. Export it back so TransformerLens/huggingface (imported
# later) read the same location.
HF_HOME = Path(os.environ.get("HF_HOME", REPO_ROOT.parent / "hf_cache"))
os.environ.setdefault("HF_HOME", str(HF_HOME))

# --- Device --------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    tl_name: str          # TransformerLens HookedTransformer name
    n_layers: int
    d_model: int
    gated: bool = False
    # Layer band where the addition "helix" is built, per the arithmetic prior art.
    # GPT-J numbers from [kantamneni2025]; Llama band is a TODO to confirm empirically.
    build_layers: tuple[int, int] = (0, 0)


MODELS: dict[str, ModelSpec] = {
    # [kantamneni2025]: MLPs ~14-18 build helix(a+b) on GPT-J (28 layers).
    "gptj": ModelSpec("gptj", "EleutherAI/gpt-j-6B", "gpt-j-6b",
                      n_layers=28, d_model=4096, build_layers=(14, 18)),
    # Llama-3-8B: 32 layers, d_model 4096. [kantamneni2025] validated the helix on
    # Llama-3.1-8B but reports the build/read split on GPT-J. PRIOR for Llama: scale
    # GPT-J's mid-network build fraction (14-18 of 28 ≈ 0.50-0.64 depth) to 32 layers
    # -> ~16-21. CONFIRM empirically with run_helix_fit before trusting it.
    # CAVEAT: Llama's gated (SwiGLU) MLP makes the helix LESS cleanly causal than GPT-J
    # ([kantamneni2025] App. D) — lead on GPT-J; expect noisier causal validation here.
    "llama3-8b": ModelSpec("llama3-8b", "meta-llama/Meta-Llama-3-8B",
                           "meta-llama/Meta-Llama-3-8B",
                           n_layers=32, d_model=4096, gated=True, build_layers=(16, 21)),
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model key {key!r}. Known: {list(MODELS)}")
    return MODELS[key]


def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "nogit"


def run_id(seed: int) -> str:
    return f"{date.today().isoformat()}-{git_short_sha()}-s{seed}"


def run_dir(experiment: str, seed: int, *, label: str | None = None,
            meta: dict | None = None) -> Path:
    """Return (and create) the output directory for a run.

    Preferred (human-readable) layout: pass ``label`` = a descriptive sub-path such as
    ``"run_fourier/bare"`` (one folder per script, a sub-folder per context). Outputs are
    organized by that path and the latest run overwrites in place; full provenance — the
    date, git short sha, seed and exact command line — is written to ``run_meta.json``
    inside the folder, plus anything passed in ``meta``. Inside the folder, file names
    only need to carry what the path does *not* already say (e.g. the layer/site).

    Legacy: omit ``label`` to fall back to the immutable per-run id
    ``<date>-<sha>-s<seed>`` (kept for callers that have not migrated).
    """
    sub = label if label is not None else run_id(seed)
    d = RESULTS_DIR / experiment / sub
    d.mkdir(parents=True, exist_ok=True)
    prov = {
        "date": date.today().isoformat(),
        "git_sha": git_short_sha(),
        "seed": seed,
        "command": " ".join(sys.argv),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    if meta:
        prov.update(meta)
    (d / "run_meta.json").write_text(json.dumps(prov, indent=2))
    return d
