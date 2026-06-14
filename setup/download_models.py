"""Pre-cache the N2P models into the persistent /storage HF cache.

Run once per Paperspace machine (after setup_paperspace.sh). Models persist in
/storage across restarts, so subsequent sessions skip the multi-GB download.

    export HF_TOKEN=...          # required for gated Llama-3-8B; not needed for GPT-J
    python setup/download_models.py
    python setup/download_models.py --only gptj      # just one
"""
import argparse
import os

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

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"[download] HF_HOME={hf_home}")
    if "/storage" not in hf_home:
        print("[download] WARNING: HF_HOME is not under /storage — cache may NOT persist "
              "across Paperspace restarts. Run setup/setup_paperspace.sh first.")

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
