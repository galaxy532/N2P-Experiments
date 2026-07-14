#!/usr/bin/env bash
# One-time setup for Paperspace Gradient (A6000). Wires the Hugging Face cache to a
# sibling dir beside the repo (../hf_cache) and installs dependencies. Safe to re-run.
set -euo pipefail

# --- HF cache ALONGSIDE the repo (sibling dir), never inside this git repo.
#     Simple, easy to find/delete. Override with HF_HOME. ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "${REPO_ROOT}")"
export HF_HOME="${HF_HOME:-${PARENT_DIR}/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1          # faster downloads
mkdir -p "${HF_HOME}"

# Persist these env vars for future shells in this machine.
PROFILE="${HOME}/.bashrc"
grep -q 'N2P HF cache' "${PROFILE}" 2>/dev/null || cat >> "${PROFILE}" <<EOF

# --- N2P HF cache (added by setup_paperspace.sh) ---
export HF_HOME=${HF_HOME}
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF

echo "[setup] HF_HOME=${HF_HOME}"

# --- Dependencies ---
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install hf_transfer || true

# Torch is (re)installed LAST, pinned to a CUDA build that matches THIS machine's GPU driver.
# Every flag here is load-bearing (learned the hard way):
#   * LAST, because installing it before `pip install -r requirements.txt` lets transformer_lens
#     (needs torch>=2.6) drag torch forward to the base image's 2.13+cu130, which the A6000's
#     CUDA-12.4 driver cannot load -> torch.cuda.is_available()==False -> silent CPU fallback.
#   * Pin the +cu124 LOCAL version: pypi only has the cu130 wheel as "2.13.0", which does NOT
#     satisfy "==2.6.0+cu124", so the cu130 build can never win the version race.
#   * Install WITH deps (NO --no-deps): modern pytorch wheels do NOT bundle the CUDA runtime —
#     they depend on the nvidia-*-cu12 pip packages for libcudart.so.12 / libcublas.so.12
#     (pulled from the cu124 index). --extra-index-url pypi supplies the non-CUDA deps
#     (sympy pin, networkx, ...); the +cu124 pin keeps pypi from serving cu130.
# 2.6.0+cu124 is the one build that satisfies BOTH transformer_lens (>=2.6) AND the 12.4 driver
# (cu124 wheels top out at torch 2.6.0). Override TORCH_SPEC/TORCH_INDEX if your driver differs;
# `nvidia-smi` (top-right) shows the max CUDA it supports.
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0+cu124}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
echo "[setup] (re)installing ${TORCH_SPEC} from ${TORCH_INDEX} (+ pypi for non-CUDA deps)"
python3 -m pip uninstall -y torch || true
python3 -m pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX}" \
    --extra-index-url https://pypi.org/simple

# --- Fail LOUDLY now if CUDA is not actually usable, instead of 40 min into a GPU run. ---
python3 - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("[setup][FATAL] torch %s (cuda build %s) cannot see the GPU. Its CUDA build is too "
             "new for this driver (nvidia-smi top-right shows the max supported). Try an older "
             "CUDA index and re-run, e.g.:\n"
             "  TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash setup/setup_paperspace.sh"
             % (torch.__version__, torch.version.cuda))
print("[setup] torch %s CUDA OK -> %s" % (torch.__version__, torch.cuda.get_device_name(0)))
PY

# --- Results/logs dirs ---
mkdir -p results logs
[ -f logs/runlog.md ] || echo "# N2P run log (append one line per run)" > logs/runlog.md

echo "[setup] done. Next: export HF_TOKEN=... (for gated Llama-3), then: python3 setup/download_models.py"
