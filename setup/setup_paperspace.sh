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

# Torch FIRST, pinned to a CUDA build that matches THIS machine's GPU driver. Paperspace base
# images sometimes ship a torch too new for the A6000 driver (seen: 2.x+cu130 vs driver CUDA
# 12.4), which makes torch.cuda.is_available()==False -> silent CPU fallback (GPT-J runs never
# finish). We uninstall the base-image torch and install a matched build. Override TORCH_SPEC /
# TORCH_INDEX if your driver differs — `nvidia-smi` (top-right) shows the max CUDA it supports
# (cu121 works for driver CUDA >= 12.1; use cu124 to match 12.4 exactly).
TORCH_SPEC="${TORCH_SPEC:-torch==2.4.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
echo "[setup] installing ${TORCH_SPEC} from ${TORCH_INDEX}"
python3 -m pip uninstall -y torch || true
python3 -m pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX}" \
    --extra-index-url https://pypi.org/simple

python3 -m pip install -r requirements.txt
python3 -m pip install hf_transfer || true

# --- Fail LOUDLY now if CUDA is not actually usable, instead of 40 min into a GPU run. ---
python3 - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("[setup][FATAL] torch %s (cuda build %s) cannot see the GPU. It is probably "
             "built for a newer CUDA than this driver supports (see nvidia-smi). Reinstall a "
             "matching build and re-run, e.g.:\n"
             "  TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash setup/setup_paperspace.sh"
             % (torch.__version__, torch.version.cuda))
print("[setup] torch %s CUDA OK -> %s" % (torch.__version__, torch.cuda.get_device_name(0)))
PY

# --- Results/logs dirs ---
mkdir -p results logs
[ -f logs/runlog.md ] || echo "# N2P run log (append one line per run)" > logs/runlog.md

echo "[setup] done. Next: export HF_TOKEN=... (for gated Llama-3), then: python3 setup/download_models.py"
