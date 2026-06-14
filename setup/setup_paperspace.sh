#!/usr/bin/env bash
# One-time setup for Paperspace Gradient (A6000). Wires the Hugging Face cache to the
# PERSISTENT /storage volume and installs dependencies. Safe to re-run.
set -euo pipefail

# --- Persistent cache (survives machine restarts; the default cache does NOT) ---
export STORAGE_ROOT="${STORAGE_ROOT:-/storage}"
export HF_HOME="${STORAGE_ROOT}/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1          # faster downloads
mkdir -p "${HF_HOME}"

# Persist these env vars for future shells in this machine.
PROFILE="${HOME}/.bashrc"
grep -q 'N2P HF cache' "${PROFILE}" 2>/dev/null || cat >> "${PROFILE}" <<EOF

# --- N2P HF cache (added by setup_paperspace.sh) ---
export STORAGE_ROOT=${STORAGE_ROOT}
export HF_HOME=${STORAGE_ROOT}/hf_cache
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
