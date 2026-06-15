"""Pre-cache the N2P models into the repo-root HF cache (<repo>/hf_cache).

Cache location is the repo root so it's easy to find and delete; override by
exporting HF_HOME. config.py defaults to the same path, so the loader reuses it.

    export HF_TOKEN=...          # required for gated Llama-3-8B; not needed for GPT-J
    python setup/download_models.py
    python setup/download_models.py --only gptj      # just one
"""
import argparse
import os
from pathlib import Path

# Cache lives ALONGSIDE the repo (<repo>/../hf_cache), NOT inside it — this is a git
# repo and the multi-GB models must not land in the working tree. Easy to find/delete.
# Override by exporting HF_HOME. Must be set BEFORE importing huggingface_hub, which
# freezes its cache path at import time.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO_ROOT.parent / "hf_cache"))

from huggingface_hub import snapshot_download

# Raw HF ids (registry keys → ids also live in src/n2p/config.py; kept in sync here
# deliberately so this script has no src dependency and can run standalone).
MODELS = {
    "gptj": "EleutherAI/gpt-j-6B",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(MODELS), default=None,
                    help="cache only this registry key (default: all)")
    args = ap.parse_args()

    hf_home = os.environ["HF_HOME"]
    print(f"[download] HF_HOME={hf_home}  (models land in {hf_home}/hub)")

    keys = [args.only] if args.only else list(MODELS)
    for k in keys:
        repo = MODELS[k]
        print(f"[download] {k} -> {repo}")
        token = os.environ.get("HF_TOKEN")
        if k == "llama3-8b" and not token:
            print("[download] SKIP llama3-8b: set HF_TOKEN and accept the license at "
                  "https://huggingface.co/meta-llama/Meta-Llama-3-8B")
            continue
        snapshot_download(repo_id=repo, token=token,
                          ignore_patterns=["*.pth", "original/*"])  # keep safetensors only
        print(f"[download] {k} cached.")

    print("[download] done.")


if __name__ == "__main__":
    main()
