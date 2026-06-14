# Week 1 — Circuit-discovery sanity check

**Goal:** prove our **chosen** discovery method recovers a **known** circuit before we
trust it on novel arithmetic tasks.

Per `../wiki/notes/approach-decision-circuit-identification.md` the base method is
**Edge Pruning** `[bhaskar2024]` (edge-level L0 masks; *perfectly recovers Tracr
ground-truth circuits*; scales to 13B), with a **noising+denoising completeness layer**
`[chen2025]` added in week 2. **ACDC is not used** — it is a rejected method (slow;
misses negative name-mover / previous-token heads; floods at low τ;
`../wiki/notes/verified-failure-modes.md` item 1). The first scaffold used ACDC as a
shortcut; that was wrong and has been removed.

## Run

```bash
# clone the base method next to N2P-Experiments (one-time):
git clone https://github.com/princeton-nlp/Edge-Pruning

# primary: known-ground-truth recovery on a Tracr-compiled program
python experiments/week1_circuit_sanity/run_discovery_sanity.py --target tracr --program reverse

# secondary: real-LM task
python experiments/week1_circuit_sanity/run_discovery_sanity.py --target greater_than --model gptj
```

## Why Tracr is the sanity target
Tracr `[lindner2023]` compiles a RASP program into exact transformer weights, so the
true circuit is known **by construction**. Edge Pruning recovering it exactly is the
cleanest validation of the pipeline. Greater-Than is a useful *real-LM* secondary
check (known approximate circuit, Hanna et al.) but has no exact ground truth.

## Status / TODO (first Paperspace run)
`src/n2p/circuits/discovery.py` locates the Edge Pruning repo and is wired to drive it,
but the exact CLI/entrypoint depends on the Edge Pruning revision — fill it from that
repo's README on the first run, assert `recovered_edges == tracr_ground_truth`, then
commit the working invocation. The `chen2025` completeness pass (`add_completeness_layer`)
is a week-2 task.
