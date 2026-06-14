# N2P run log (append one line per run)

Format: `## [YYYY-MM-DD] <script> | model=<m> task=<t> seed=<s> | <one-line result> | results/<path>`

## [2026-06-14] scaffold | repo initialized | Version A, week-1 number-rep + circuit-sanity scripts written (untested on GPU) | —
## [2026-06-14] revise | circuit discovery ACDC→Edge Pruning+Tracr; Llama build_layers prior (16,21); tasks=clean-core+stress-set; helix intercept bug fixed (synthetic R²=1.0) | logic unit-checked, GPU-untested | —
## [2026-06-14] add | week1_accuracy_probe (frozen few-shot accuracy per task×framing); per-task framings; greater_than canonical template; gated-MLP caveat | logic unit-checked, GPU-untested | —
