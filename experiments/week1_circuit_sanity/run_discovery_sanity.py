"""Week 1 — discovery sanity check using the PROTOCOL'S base method (Edge Pruning).

Primary: run Edge Pruning [bhaskar2024] on a Tracr-compiled program whose ground-truth
circuit is known exactly, and confirm recovery. Secondary: Greater-Than on a real LM.

ACDC is deliberately NOT used (rejected method — see ../../src/n2p/circuits/discovery.py
and ../wiki/notes/verified-failure-modes.md item 1).

    git clone https://github.com/princeton-nlp/Edge-Pruning   # next to N2P-Experiments
    python experiments/week1_circuit_sanity/run_discovery_sanity.py --target tracr
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n2p.circuits import discovery   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["tracr", "greater_than"], default="tracr")
    ap.add_argument("--program", default="reverse", help="Tracr program id (for --target tracr)")
    ap.add_argument("--model", default="gptj")
    args = ap.parse_args()

    if not discovery.edge_pruning_available():
        print(f"[skip] Edge Pruning not found at {discovery.EDGE_PRUNING_DIR}.")
        print("       git clone https://github.com/princeton-nlp/Edge-Pruning next to "
              "N2P-Experiments (or set EDGE_PRUNING_DIR).")
        return

    if args.target == "tracr":
        print(f"[run] Edge Pruning on Tracr program '{args.program}' (known ground truth)")
        discovery.run_edge_pruning_tracr(program=args.program)
        print("[done] Compare recovered edges to the Tracr ground-truth circuit "
              "(should match exactly [bhaskar2024]).")
    else:
        print(f"[run] Edge Pruning on Greater-Than ({args.model})")
        discovery.run_edge_pruning_task(task="greater_than", model_key=args.model)
        print("[done] Compare to the canonical Greater-Than circuit (Hanna et al.).")


if __name__ == "__main__":
    main()
