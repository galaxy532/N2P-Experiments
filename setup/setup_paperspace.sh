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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install hf_transfer || true

# --- Results/logs dirs ---
mkdir -p results logs
[ -f logs/runlog.md ] || echo "# N2P run log (append one line per run)" > logs/runlog.md

echo "[setup] done. Next: export HF_TOKEN=... (for gated Llama-3), then: python setup/download_models.py"
